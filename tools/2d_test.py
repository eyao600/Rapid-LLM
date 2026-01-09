#!/usr/bin/env python3
"""
2D topology test harness for RAPID-LLM (training + inference).

Runs Mesh2D/Torus2D/KingMesh2D/HyperCube comparisons with and without GMap optimization,
prints tabular results, and emits bar plots.
"""

import copy
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sns.set()
sns.set_context("notebook", font_scale=1.5)

import config  # noqa: E402
from validation_scripts import huggingface_bench_validation as hfval  # noqa: E402
from astrasim_lib import ensure_chakra_available  # noqa: E402
from inference_timing import TimeCalculationLLMInference  # noqa: E402
from train_timing import TimeCalculationLLM  # noqa: E402

H100_GPU = True

if H100_GPU:
    TRAIN_HW_CONFIG = "validation_scripts/validation_configs/hardware-config/H100_SXM5_80GB_2d.yaml"
    INF_HW_CONFIG = "validation_scripts/validation_configs/hardware-config/H100_SXM5_80GB_2d.yaml"
else:
    TRAIN_HW_CONFIG = "validation_scripts/validation_configs/hardware-config/a100_80GB_2d_gmap_train.yaml"
    INF_HW_CONFIG = "validation_scripts/validation_configs/hardware-config/a100_80GB_2d_gmap_inf.yaml"

TRAIN_70B_MODEL_CONFIG = "validation_scripts/validation_configs/model-config/Llama3.1-70B_2d_train.yaml"
TRAIN_GPT175B_MODEL_CONFIG = "validation_scripts/validation_configs/model-config/GPT_175_B_2d_train.yaml"
INF_70B_MODEL_CONFIG = "validation_scripts/validation_configs/model-config/Llama3.1-70B_2d_inf.yaml"
INF_GPT175B_MODEL_CONFIG = "validation_scripts/validation_configs/model-config/GPT_175_B_2d_inf.yaml"

OUTPUT_ROOT = Path("output") / "2d_test"
PLOT_DIR = OUTPUT_ROOT
Y_LABEL = "Normalized batch runtime (s)"
CACHE_VERSION = 3
CACHE_PATH = OUTPUT_ROOT / "2d_test_cache.json"
VERBOSE = os.environ.get("RAPID_2D_TEST_VERBOSE", "1") != "0"
AUTO_PARALLELISM = True
PP_MAX = 8

NUM_WORKERS = int(os.environ.get("RAPID_2D_TEST_WORKERS", max(1, (os.cpu_count() or 1) - 1)))
hfval.ASTRA_CACHE_MODE = "NO_CACHE"
hfval.ASTRA_TMP_ROOT = Path("tmp") / "2d_test_runs"
hfval.CLEANUP_ASTRA_TMP = True

# TOPOLOGIES = ("Mesh2D", "Torus2D", "KingMesh2D", "HyperCube")
# COMP_TOPO = ("Torus2D", "KingMesh2D", "HyperCube")


TOPOLOGIES = ("Mesh2D", "Torus2D", "KingMesh2D")
COMP_TOPO = ("Torus2D", "KingMesh2D")


COMP_TOPO_SPECS = {
    "Torus2D": {
        "output": "2d_test_super_merged.png",
        "ylabel": "2D Torus over Mesh speedup",
        "title": "2D Torus vs Mesh inference/training runtime comparison",
        "dynamic_ylim": False,
    },
    "KingMesh2D": {
        "output": "2d_test_super_merged_kingmesh.png",
        "ylabel": "2D KingMesh over Mesh speedup",
        "title": "2D KingMesh vs Mesh inference/training runtime comparison",
        "dynamic_ylim": True,
    },
    "HyperCube": {
        "output": "2d_test_super_merged_hypercube.png",
        "ylabel": "HyperCube over Mesh speedup",
        "title": "HyperCube vs Mesh inference/training runtime comparison",
        "dynamic_ylim": True,
    },
}
# TOPOLOGIES = ("Mesh2D", "Torus2D")
# TRAIN_SHAPES = ((8, 8), (4, 16))
# TRAIN_MODELS = (
#     {
#         "label": "70B",
#         "config": TRAIN_70B_MODEL_CONFIG,
#         "parallelisms": (
#             {"tp": 32, "cp": 1, "pp": 2},
#             {"tp": 16, "cp": 1, "pp": 4},
#         ),
#         "axes": ("tp", "pp"),
#     },
#     {
#         "label": "GPT175B",
#         "config": TRAIN_GPT175B_MODEL_CONFIG,
#         "parallelisms": (
#             {"tp": 32, "cp": 1, "pp": 2},
#             {"tp": 16, "cp": 1, "pp": 4},
#         ),
#         "axes": ("tp", "pp"),
#     },
# )
TRAIN_SHAPES = ((4, 5), (5,6), (4, 8), (6, 6))
TRAIN_MODELS = (
    {
        "label": "70B",
        "config": TRAIN_70B_MODEL_CONFIG,
        "parallelisms": (
            {"tp": 10, "cp": 1, "pp": 2},  #4,5 = 20 = 10 * 2
            {"tp": 10, "cp": 1, "pp": 3},  #5,6 = 30 = 3 * 10
            {"tp": 8, "cp": 1, "pp": 4},  #6,6 = 48 = 8 * 6
            {"tp": 6, "cp": 1, "pp": 6},  #4,9 = 72 = 8 * 9
        ),
        "axes": ("tp", "pp"),
    },
    {
        "label": "GPT175B",
        "config": TRAIN_GPT175B_MODEL_CONFIG,
        "parallelisms": (
            {"tp": 10, "cp": 1, "pp": 2},  #4,5 = 20 = 10 * 2
            {"tp": 10, "cp": 1, "pp": 3},  #5,6 = 30 = 3 * 10
            {"tp": 8, "cp": 1, "pp": 4},  #6,6 = 48 = 8 * 6
            {"tp": 6, "cp": 1, "pp": 6},  #4,9 = 72 = 8 * 9
        ),
        "axes": ("tp", "pp"),
    },
)
INF_SHAPES = ((4, 5), (5, 6), (4, 8), (6, 6))
INF_MODELS = (
    {"label": "70B", "config": INF_70B_MODEL_CONFIG},
    {"label": "GPT175B", "config": INF_GPT175B_MODEL_CONFIG},
)

MODEL_DISPLAY_NAMES = {
    "70B": "Llama3-70B",
    "GPT175B": "GPT 175B",
}

_WORKER_BASE_HW: Optional[Dict[str, Any]] = None


def _init_worker(base_hw_config: Dict[str, Any]) -> None:
    hfval._worker_init(base_hw_config)
    global _WORKER_BASE_HW
    _WORKER_BASE_HW = base_hw_config


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _file_signature(path: str) -> str:
    payload = Path(path).read_bytes()
    return hashlib.sha1(payload).hexdigest()


def determine_model_mode(model_path: str) -> str:
    model_dict = read_yaml(model_path)
    model_param = model_dict.get("model_param") or {}
    mode = model_param.get("mode")
    if not mode:
        raise ValueError(f"model_param.mode must be defined in {model_path}")
    return str(mode)


def _write_temp_yaml(data: Dict[str, Any]) -> str:
    temp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    try:
        yaml.safe_dump(data, temp, default_flow_style=False, sort_keys=False)
        temp.flush()
        return temp.name
    finally:
        try:
            temp.close()
        except Exception:
            pass


def _update_hw_dict(
    base_hw: Dict[str, Any],
    *,
    topology: str,
    shape: Tuple[int, int],
    parallelism: Dict[str, int],
    parallelism_axes: Sequence[str],
    mb_override: Optional[int],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_hw)
    par = cfg.setdefault("parallelism", {})
    par["tp"] = int(parallelism["tp"])
    par["cp"] = int(parallelism["cp"])
    par["pp"] = int(parallelism["pp"])
    if mb_override is not None:
        par["mb"] = int(mb_override)
    train_block = par.setdefault("train", {})
    train_block["dp"] = 1
    train_block.setdefault("ep", 1)
    train_block.setdefault("tp_ep", True)
    inference_block = par.setdefault("inference", {})
    inference_block.setdefault("replica_count", 1)
    inference_block.setdefault("moe_dp", 1)

    net = cfg.setdefault("network", {})
    dims = list(net.get("dimensions") or [])
    if not dims:
        raise ValueError("Hardware config missing network.dimensions")
    dim0 = dims[0]
    dim0["size"] = [int(shape[0]), int(shape[1])]
    topo_block = dim0.setdefault("topology", {})
    topo_block["type"] = topology
    topo_block["optimize_2dmap"] = False
    dim0["parallelisms"] = list(parallelism_axes)
    net["dimensions"] = [dim0]
    return cfg


def _format_runtime(value: Optional[float]) -> str:
    if value is None:
        return "error"
    return f"{value:.6f}"


def _format_mem_fields(
    exceeded: Optional[bool],
    violation_gb: Optional[float],
) -> Tuple[str, str]:
    if exceeded is None:
        return "error", ""
    if exceeded:
        if violation_gb is None:
            return "yes", ""
        return "yes", f"{violation_gb:.2f}"
    return "no", "0.00"


def _slug(parts: Sequence[str]) -> str:
    cooked = "_".join(part.replace("/", "_") for part in parts if part)
    return cooked.replace(" ", "").replace(":", "_")


def _load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return {"version": CACHE_VERSION, "cases": {}}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {"version": CACHE_VERSION, "cases": {}}
    if data.get("version") != CACHE_VERSION:
        return {"version": CACHE_VERSION, "cases": {}}
    cases = data.get("cases")
    if not isinstance(cases, dict):
        data["cases"] = {}
    return data


def _save_cache(cache: Dict[str, Any]) -> None:
    try:
        with CACHE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)
    except Exception:
        pass


def _case_key(case: Dict[str, Any]) -> str:
    kind = case.get("kind")
    data: Dict[str, Any] = {
        "kind": kind,
        "model": case.get("model_label"),
        "model_sig": case.get("model_sig"),
        "hw_sig": case.get("hw_sig"),
        "topology": case.get("topology"),
        "shape": case.get("shape"),
    }
    if kind == "train":
        parallelism = case.get("parallelism") or {}
        data.update(
            {
                "tp": parallelism.get("tp"),
                "cp": parallelism.get("cp"),
                "pp": parallelism.get("pp"),
                "axes": list(case.get("parallelism_axes") or ()),
            }
        )
    elif kind == "inference":
        data["tp"] = int(case["shape"][0]) * int(case["shape"][1])
    return json.dumps(data, sort_keys=True)


def _is_cache_row(row: Dict[str, Any]) -> bool:
    return isinstance(row, dict) and "runtime_s" in row and "topology" in row


def _describe_case(case: Dict[str, Any]) -> str:
    kind = case.get("kind")
    if kind == "train":
        parallelism = case.get("parallelism") or {}
        return (
            f"train {case.get('model_label')} {case.get('topology')} "
            f"{case.get('shape')[0]}x{case.get('shape')[1]} "
            f"tp{parallelism.get('tp')} cp{parallelism.get('cp')} "
            f"pp{parallelism.get('pp')}"
        )
    if kind == "inference":
        shape = case.get("shape")
        tp = int(shape[0]) * int(shape[1])
        return (
            f"inference {case.get('model_label')} {case.get('topology')} "
            f"{shape[0]}x{shape[1]} tp{tp}"
        )
    return f"unknown {case}"


def _format_tokens(value: int) -> str:
    if value and value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)


def _format_shape_label(shape_label: str) -> str:
    if shape_label == "4x5":
        return "4x5[FRED]"
    return shape_label


def _shape_key(value: str) -> Tuple[int, int]:
    try:
        left, right = value.split("x", 1)
        right = right.split("[", 1)[0]
        return int(left), int(right)
    except Exception:
        return (0, 0)


def _candidate_parallelisms(total_devices: int) -> List[Dict[str, int]]:
    candidates = []
    for tp in range(1, total_devices + 1):
        if total_devices % tp != 0:
            continue
        pp = total_devices // tp
        if pp > PP_MAX:
            continue
        candidates.append({"tp": tp, "cp": 1, "pp": pp})
    return candidates


def _select_best_parallelisms(
    rows: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, int]]:
    best_valid: Dict[Tuple[str, str], Tuple[float, Dict[str, int]]] = {}
    best_any: Dict[Tuple[str, str], Tuple[float, Dict[str, int]]] = {}
    for row in rows:
        try:
            runtime = float(row["runtime_s"])
        except Exception:
            continue
        model = str(row.get("model", ""))
        shape = str(row.get("shape", ""))
        key = (model, shape)
        par = {
            "tp": int(row.get("tp", 0) or 0),
            "cp": int(row.get("cp", 0) or 0),
            "pp": int(row.get("pp", 0) or 0),
        }
        if runtime > 0:
            current = best_any.get(key)
            if current is None or runtime < current[0]:
                best_any[key] = (runtime, par)
        if str(row.get("mem_exceeded", "")).lower() == "no" and runtime > 0:
            current = best_valid.get(key)
            if current is None or runtime < current[0]:
                best_valid[key] = (runtime, par)
    selected: Dict[Tuple[str, str], Dict[str, int]] = {}
    for key, value in best_any.items():
        selected[key] = best_valid.get(key, value)[1]
    return selected


def _format_train_label(shape_label: str, tp: int, pp: int) -> str:
    return f"{_format_shape_label(shape_label)}\nTP={tp}\nPP={pp}"


def _format_inf_label(shape_label: str, tp: int) -> str:
    return f"{_format_shape_label(shape_label)}\nTP={tp}"


def _run_training_case(
    base_hw: Dict[str, Any],
    *,
    model_config_path: str,
    model_label: str,
    parallelism_axes: Sequence[str],
    topology: str,
    shape: Tuple[int, int],
    parallelism: Dict[str, int],
) -> Tuple[Optional[float], Optional[str], Optional[bool], Optional[float]]:
    mb = 2 * int(parallelism["pp"])
    run_dir = None
    prev_env: Dict[str, Optional[str]] = {}
    hw_dict = _update_hw_dict(
        base_hw,
        topology=topology,
        shape=shape,
        parallelism=parallelism,
        parallelism_axes=parallelism_axes,
        mb_override=mb,
    )
    temp_path = _write_temp_yaml(hw_dict)
    desc = _slug(
        [
            "train",
            model_label,
            f"{topology}",
            f"{shape[0]}x{shape[1]}",
            f"tp{parallelism['tp']}",
            f"cp{parallelism['cp']}",
            f"pp{parallelism['pp']}",
        ]
    )
    out_dir = OUTPUT_ROOT / "train" / desc
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_dir, prev_env = hfval._prepare_astra_tmp_dir()
        mode = determine_model_mode(model_config_path)
        hw_config = config.parse_config(temp_path, config_type="hardware")
        model_config = config.parse_config(model_config_path, config_type=mode)
        config.validate_configs(hw_config, model_config)
        ensure_chakra_available()
        calc = TimeCalculationLLM(hw_config, model_config, mode, output_dir=str(out_dir))
        runtime = float(calc.calc_time_llm())
        mem_exceeded = bool(getattr(calc, "memory_capacity_exceeded", False))
        mem_violation = float(getattr(calc, "memory_capacity_violation_gb", 0.0) or 0.0)
        return runtime, None, mem_exceeded, mem_violation
    except Exception as exc:
        return None, str(exc), None, None
    finally:
        if prev_env:
            hfval._restore_astra_env(prev_env)
        if run_dir and hfval.CLEANUP_ASTRA_TMP:
            shutil.rmtree(run_dir, ignore_errors=True)
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def _run_inference_case(
    base_hw: Dict[str, Any],
    *,
    topology: str,
    shape: Tuple[int, int],
    model_config_path: str,
) -> Tuple[Optional[float], Optional[str], Optional[bool], Optional[float]]:
    run_dir = None
    prev_env: Dict[str, Optional[str]] = {}
    tp = int(shape[0]) * int(shape[1])
    hw_dict = _update_hw_dict(
        base_hw,
        topology=topology,
        shape=shape,
        parallelism={"tp": tp, "cp": 1, "pp": 1},
        parallelism_axes=("tp",),
        mb_override=1,
    )
    temp_path = _write_temp_yaml(hw_dict)
    desc = _slug(["inference", f"{topology}", f"{shape[0]}x{shape[1]}", f"tp{tp}"])
    out_dir = OUTPUT_ROOT / "inference" / desc
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_dir, prev_env = hfval._prepare_astra_tmp_dir()
        mode = determine_model_mode(model_config_path)
        hw_config = config.parse_config(temp_path, config_type="hardware")
        model_config = config.parse_config(model_config_path, config_type=mode)
        config.validate_configs(hw_config, model_config)
        ensure_chakra_available()
        calc = TimeCalculationLLMInference(hw_config, model_config, mode, output_dir=str(out_dir))
        summary = calc.calc_total_inference_time()
        runtime = float(summary["total_inference_time"])
        mem_exceeded = bool(getattr(calc, "memory_capacity_exceeded", False))
        mem_violation = float(getattr(calc, "memory_capacity_violation_gb", 0.0) or 0.0)
        return runtime, None, mem_exceeded, mem_violation
    except Exception as exc:
        return None, str(exc), None, None
    finally:
        if prev_env:
            hfval._restore_astra_env(prev_env)
        if run_dir and hfval.CLEANUP_ASTRA_TMP:
            shutil.rmtree(run_dir, ignore_errors=True)
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def _case_worker(case: Dict[str, Any]) -> Dict[str, Any]:
    base_hw = _WORKER_BASE_HW
    if base_hw is None:
        return {
            "error": "Worker base hardware config not initialized",
            "_case_key": case.get("case_key"),
        }
    if VERBOSE:
        print(f"[2d_test] start {_describe_case(case)}", flush=True)
    kind = case.get("kind")
    if kind == "train":
        runtime, error, mem_exceeded, mem_violation = _run_training_case(
            base_hw,
            model_config_path=case["model_config"],
            model_label=case["model_label"],
            parallelism_axes=case["parallelism_axes"],
            topology=case["topology"],
            shape=case["shape"],
            parallelism=case["parallelism"],
        )
        mem_flag, mem_delta = _format_mem_fields(mem_exceeded, mem_violation)
        row = {
            "model": case["model_label"],
            "topology": case["topology"],
            "shape": f"{case['shape'][0]}x{case['shape'][1]}",
            "tp": case["parallelism"]["tp"],
            "cp": case["parallelism"]["cp"],
            "pp": case["parallelism"]["pp"],
            "mb": 4 * case["parallelism"]["pp"],
            "runtime_s": _format_runtime(runtime),
            "mem_exceeded": mem_flag,
            "mem_violation_gb": mem_delta,
        }
        if error:
            row["error"] = error
        row["_case_key"] = case.get("case_key")
        if VERBOSE:
            status = row["runtime_s"]
            print(f"[2d_test] done {_describe_case(case)} -> {status}", flush=True)
        return row
    if kind == "inference":
        runtime, error, mem_exceeded, mem_violation = _run_inference_case(
            base_hw,
            topology=case["topology"],
            shape=case["shape"],
            model_config_path=case["model_config"],
        )
        mem_flag, mem_delta = _format_mem_fields(mem_exceeded, mem_violation)
        tp = int(case["shape"][0]) * int(case["shape"][1])
        row = {
            "model": case["model_label"],
            "topology": case["topology"],
            "shape": f"{case['shape'][0]}x{case['shape'][1]}",
            "tp": tp,
            "cp": 1,
            "pp": 1,
            "mb": 1,
            "runtime_s": _format_runtime(runtime),
            "mem_exceeded": mem_flag,
            "mem_violation_gb": mem_delta,
        }
        if error:
            row["error"] = error
        row["_case_key"] = case.get("case_key")
        if VERBOSE:
            status = row["runtime_s"]
            print(f"[2d_test] done {_describe_case(case)} -> {status}", flush=True)
        return row
    return {"error": f"Unknown case kind: {kind}", "_case_key": case.get("case_key")}


def _run_cases_parallel(
    cases: Sequence[Dict[str, Any]],
    base_hw: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not cases:
        return []
    if VERBOSE:
        print(f"[2d_test] running {len(cases)} cases (workers={NUM_WORKERS})", flush=True)
    progress = None
    try:
        from tqdm import tqdm

        progress = tqdm(total=len(cases), desc="2d_test")
    except Exception:
        progress = None
    worker_count = min(NUM_WORKERS, len(cases), os.cpu_count() or 1)
    if worker_count <= 1:
        _init_worker(base_hw)
        results = []
        for case in cases:
            results.append(_case_worker(case))
            if progress is not None:
                progress.update(1)
        if progress is not None:
            progress.close()
        return results
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(base_hw,),
    ) as pool:
        results = []
        for result in pool.imap_unordered(_case_worker, cases):
            results.append(result)
            if progress is not None:
                progress.update(1)
        if progress is not None:
            progress.close()
        return results


def _print_table(rows: Sequence[Dict[str, Any]], header: Sequence[str]) -> None:
    rendered = [[str(row.get(col, "")) for col in header] for row in rows]
    widths = [len(col) for col in header]
    for row in rendered:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    sep = "  "
    header_line = sep.join(col.ljust(widths[idx]) for idx, col in enumerate(header))
    divider = sep.join("-" * widths[idx] for idx in range(len(header)))
    print(header_line)
    print(divider)
    for row in rendered:
        print(sep.join(row[idx].ljust(widths[idx]) for idx in range(len(header))))


def _plot_bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.45), 6))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_dual_topology_axis(
    ax: plt.Axes,
    labels: Sequence[str],
    mesh_values: Sequence[float],
    torus_values: Sequence[float],
    *,
    title: str,
    ylabel: Optional[str] = None,
) -> Tuple[List[Any], List[str]]:
    x = np.arange(len(labels))
    width = 0.38
    torus_handle = ax.bar(x - width / 2, torus_values, width, label="2D Torus")
    mesh_handle = ax.bar(x + width / 2, mesh_values, width, label="2D Mesh")
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    return [torus_handle, mesh_handle], ["2D Torus", "2D Mesh"]


def _default_mode_colors() -> Tuple[str, str]:
    palette = plt.rcParams.get("axes.prop_cycle", None)
    colors = palette.by_key().get("color", []) if palette else []
    inf_color = colors[0] if len(colors) > 0 else "#1f77b4"
    train_color = colors[1] if len(colors) > 1 else "#ff7f0e"
    return inf_color, train_color


def _compute_speedup_ratio(
    rows: Sequence[Dict[str, Any]],
    *,
    baseline_topology: str,
    target_topology: str,
) -> Dict[Tuple[str, str], float]:
    groups: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in rows:
        try:
            runtime = float(row["runtime_s"])
        except Exception:
            continue
        if runtime <= 0:
            continue
        model = str(row.get("model", ""))
        shape = str(row.get("shape", ""))
        topo = str(row.get("topology", ""))
        groups.setdefault(model, {}).setdefault(shape, {})[topo] = runtime
    ratios: Dict[Tuple[str, str], float] = {}
    for model, shape_map in groups.items():
        for shape_label, topo_map in shape_map.items():
            baseline_val = topo_map.get(baseline_topology)
            target_val = topo_map.get(target_topology)
            if baseline_val is None or target_val is None or target_val <= 0:
                continue
            ratios[(model, shape_label)] = baseline_val / target_val
    return ratios


def _speedup_ylim(
    train_speedup: Dict[Tuple[str, str], float],
    inf_speedup: Dict[Tuple[str, str], float],
    *,
    default: Tuple[float, float] = (0.95, 1.4),
) -> Tuple[float, float]:
    values = [val for val in train_speedup.values()] + [val for val in inf_speedup.values()]
    if not values:
        return default
    lo = min(values)
    hi = max(values)
    lo = min(lo, 1.0)
    hi = max(hi, 1.0)
    span = max(hi - lo, 0.0)
    pad = max(0.05, span * 0.1)
    lower = max(0.0, lo - pad)
    upper = hi + pad
    if upper <= lower:
        upper = lower + 0.1
    return (lower, upper)


def _plot_speedup_by_model(
    train_speedup: Dict[Tuple[str, str], float],
    inf_speedup: Dict[Tuple[str, str], float],
    *,
    model_order: Sequence[str],
    shapes: Sequence[str],
    output_path: Path,
) -> None:
    if not model_order or not shapes:
        return
    max_shapes = max(len(shapes), 1)
    axis_width = max(6.0, max_shapes * 0.7)
    fig, axes = plt.subplots(
        1,
        len(model_order),
        figsize=(axis_width * len(model_order), 6),
        sharey=True,
    )
    if len(model_order) == 1:
        axes = [axes]
    inf_color, train_color = _default_mode_colors()
    x = np.arange(len(shapes))
    width = 0.36

    for idx, (ax, model_label) in enumerate(zip(axes, model_order)):
        title = MODEL_DISPLAY_NAMES.get(model_label, model_label)
        inf_vals = []
        train_vals = []
        for shape in shapes:
            inf_val = inf_speedup.get((model_label, shape))
            train_val = train_speedup.get((model_label, shape))
            inf_vals.append(inf_val if inf_val is not None else np.nan)
            train_vals.append(train_val if train_val is not None else np.nan)
        ax.bar(x - width / 2, inf_vals, width, label="inference", color=inf_color)
        ax.bar(x + width / 2, train_vals, width, label="train", color=train_color)
        ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.6, alpha=0.85, zorder=3)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([_format_shape_label(s) for s in shapes], rotation=0, ha="center")
        ax.set_ylim(0.95, 1.4)
        if idx == 0:
            ax.set_ylabel("2D Torus over Mesh speedup")

    legend_axis = axes[1] if len(axes) > 1 else axes[0]
    legend_axis.legend(loc="center right")
    fig.suptitle("2D Torus vs Mesh inference/training runtime comparison", y=0.96)
    fig.supxlabel("2D topology shape")
    fig.subplots_adjust(top=0.86, bottom=0.11, left=0.09, right=0.98, wspace=0.08)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_super_merged_speedup(
    train_speedup: Dict[Tuple[str, str], float],
    inf_speedup: Dict[Tuple[str, str], float],
    *,
    model_order: Sequence[str],
    shapes: Sequence[str],
    output_path: Path,
    ylabel: str = "2D Torus over Mesh speedup",
    title: str = "2D Torus vs Mesh inference/training runtime comparison",
    y_limits: Optional[Tuple[float, float]] = None,
) -> None:
    if not model_order or not shapes:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(shapes) * 0.9), 6))
    inf_color, train_color = _default_mode_colors()
    model_colors = {}
    for model_label in model_order:
        if model_label == "GPT175B":
            model_colors[model_label] = train_color
        else:
            model_colors[model_label] = inf_color
    mode_order = ["inference", "train"]
    mode_hatches = {"inference": "", "train": "//"}
    combos = [(model_label, mode) for model_label in model_order for mode in mode_order]
    width = 0.72 / max(1, len(combos))
    offsets = (np.arange(len(combos)) - (len(combos) - 1) / 2) * width
    x = np.arange(len(shapes))

    prev_hatch_lw = plt.rcParams.get("hatch.linewidth", None)
    plt.rcParams["hatch.linewidth"] = 0.75
    try:
        for idx, (model_label, mode) in enumerate(combos):
            if mode == "train":
                values = [
                    train_speedup.get((model_label, shape))
                    if train_speedup.get((model_label, shape)) is not None
                    else np.nan
                    for shape in shapes
                ]
            else:
                values = [
                    inf_speedup.get((model_label, shape))
                    if inf_speedup.get((model_label, shape)) is not None
                    else np.nan
                    for shape in shapes
                ]
            ax.bar(
                x + offsets[idx],
                values,
                width,
                color=model_colors[model_label],
                hatch=mode_hatches[mode],
            )
    finally:
        if prev_hatch_lw is not None:
            plt.rcParams["hatch.linewidth"] = prev_hatch_lw

    ax.set_xticks(x)
    ax.set_xticklabels([_format_shape_label(s) for s in shapes], rotation=0, ha="center")
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    else:
        ax.set_ylim(0.95, 1.4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("2D topology and GPU count")
    ax.set_title(title)
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.6, alpha=0.85, zorder=3)

    mode_handles = [
        Patch(
            facecolor="#dddddd",
            hatch="",
            label="inference",
        ),
        Patch(
            facecolor="#dddddd",
            hatch="//",
            label="train",
        ),
    ]
    model_handles = [
        Patch(
            facecolor=model_colors.get(model_label, inf_color),
            label=MODEL_DISPLAY_NAMES.get(model_label, model_label),
        )
        for model_label in model_order
    ]
    ax.legend(handles=mode_handles + model_handles, loc="center right")
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.1, right=0.98)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    hfval._ensure_project_root_on_path()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    train_hw_base = read_yaml(TRAIN_HW_CONFIG)
    inf_hw_base = read_yaml(INF_HW_CONFIG)
    train_hw_sig = _file_signature(TRAIN_HW_CONFIG)
    inf_hw_sig = _file_signature(INF_HW_CONFIG)
    train_model_sigs = {model["config"]: _file_signature(model["config"]) for model in TRAIN_MODELS}
    inf_model_sigs = {model["config"]: _file_signature(model["config"]) for model in INF_MODELS}

    cache = _load_cache()
    cache_cases = cache.setdefault("cases", {})

    train_cases: List[Dict[str, Any]] = []
    if AUTO_PARALLELISM:
        if "Mesh2D" not in TOPOLOGIES:
            raise ValueError("AUTO_PARALLELISM requires Mesh2D to be in TOPOLOGIES")
        selection_cases: List[Dict[str, Any]] = []
        for model in TRAIN_MODELS:
            for shape in TRAIN_SHAPES:
                total_devices = int(shape[0]) * int(shape[1])
                for parallelism in _candidate_parallelisms(total_devices):
                    case = {
                        "kind": "train",
                        "model_label": model["label"],
                        "model_config": model["config"],
                        "model_sig": train_model_sigs.get(model["config"], ""),
                        "hw_sig": train_hw_sig,
                        "parallelism_axes": model.get("axes", ("tp", "cp", "pp")),
                        "topology": "Mesh2D",
                        "shape": shape,
                        "parallelism": parallelism,
                    }
                    case["case_key"] = _case_key(case)
                    selection_cases.append(case)
        selection_cached_rows = []
        selection_run_cases = []
        for case in selection_cases:
            cached = cache_cases.get(case["case_key"])
            if isinstance(cached, dict) and _is_cache_row(cached):
                selection_cached_rows.append(cached)
            else:
                selection_run_cases.append(case)
        if VERBOSE:
            print(
                f"[2d_test] auto-parallelism sweep: "
                f"{len(selection_cached_rows)}/{len(selection_cases)} cache hits",
                flush=True,
            )
        selection_rows = selection_cached_rows + (
            _run_cases_parallel(selection_run_cases, train_hw_base)
            if selection_run_cases
            else []
        )
        for row in selection_rows:
            case_key = row.get("_case_key")
            if case_key and _is_cache_row(row):
                cache_cases[case_key] = row
        best_parallelisms = _select_best_parallelisms(selection_rows)
        missing_best = []
        for model in TRAIN_MODELS:
            for shape in TRAIN_SHAPES:
                shape_label = f"{shape[0]}x{shape[1]}"
                parallelism = best_parallelisms.get((model["label"], shape_label))
                if not parallelism:
                    missing_best.append(f"{model['label']} {shape_label}")
                    continue
                for topology in TOPOLOGIES:
                    case = {
                        "kind": "train",
                        "model_label": model["label"],
                        "model_config": model["config"],
                        "model_sig": train_model_sigs.get(model["config"], ""),
                        "hw_sig": train_hw_sig,
                        "parallelism_axes": model.get("axes", ("tp", "cp", "pp")),
                        "topology": topology,
                        "shape": shape,
                        "parallelism": parallelism,
                    }
                    case["case_key"] = _case_key(case)
                    train_cases.append(case)
        if missing_best:
            print(
                "[2d_test] warning: no valid parallelism found for "
                + ", ".join(missing_best),
                flush=True,
            )
    else:
        for model in TRAIN_MODELS:
            parallelisms = list(model.get("parallelisms") or ())
            if len(parallelisms) != len(TRAIN_SHAPES):
                raise ValueError(
                    f"TRAIN_SHAPES ({len(TRAIN_SHAPES)}) must match parallelisms "
                    f"({len(parallelisms)}) for model {model.get('label')}"
                )
            for topology in TOPOLOGIES:
                for shape, parallelism in zip(TRAIN_SHAPES, parallelisms):
                    case = {
                        "kind": "train",
                        "model_label": model["label"],
                        "model_config": model["config"],
                        "model_sig": train_model_sigs.get(model["config"], ""),
                        "hw_sig": train_hw_sig,
                        "parallelism_axes": model.get("axes", ("tp", "cp", "pp")),
                        "topology": topology,
                        "shape": shape,
                        "parallelism": parallelism,
                    }
                    case["case_key"] = _case_key(case)
                    train_cases.append(case)

    inf_cases: List[Dict[str, Any]] = []
    for model in INF_MODELS:
        for shape in INF_SHAPES:
            for topology in TOPOLOGIES:
                case = {
                    "kind": "inference",
                    "model_label": model["label"],
                    "model_config": model["config"],
                    "model_sig": inf_model_sigs.get(model["config"], ""),
                    "hw_sig": inf_hw_sig,
                    "topology": topology,
                    "shape": shape,
                }
                case["case_key"] = _case_key(case)
                inf_cases.append(case)

    train_cached_rows = []
    train_run_cases = []
    for case in train_cases:
        cached = cache_cases.get(case["case_key"])
        if isinstance(cached, dict) and _is_cache_row(cached):
            train_cached_rows.append(cached)
        else:
            train_run_cases.append(case)

    inf_cached_rows = []
    inf_run_cases = []
    for case in inf_cases:
        cached = cache_cases.get(case["case_key"])
        if isinstance(cached, dict) and _is_cache_row(cached):
            inf_cached_rows.append(cached)
        else:
            inf_run_cases.append(case)

    print(
        f"[2d_test] cache: train {len(train_cached_rows)}/{len(train_cases)} "
        f"hits, inf {len(inf_cached_rows)}/{len(inf_cases)} hits",
        flush=True,
    )
    if VERBOSE and train_run_cases:
        print(f"[2d_test] train cases pending: {len(train_run_cases)}", flush=True)
    if VERBOSE and inf_run_cases:
        print(f"[2d_test] inference cases pending: {len(inf_run_cases)}", flush=True)

    train_rows = train_cached_rows + (
        _run_cases_parallel(train_run_cases, train_hw_base) if train_run_cases else []
    )
    inf_rows = inf_cached_rows + (
        _run_cases_parallel(inf_run_cases, inf_hw_base) if inf_run_cases else []
    )

    for row in train_rows + inf_rows:
        case_key = row.get("_case_key")
        if case_key and _is_cache_row(row):
            cache_cases[case_key] = row

    train_errors: List[str] = []
    for row in train_rows:
        error = row.get("error")
        if error:
            train_errors.append(
                f"{row.get('model')} {row.get('topology')} {row.get('shape')} tp{row.get('tp')} "
                f"cp{row.get('cp')} pp{row.get('pp')}: {error}"
            )

    inf_errors: List[str] = []
    for row in inf_rows:
        error = row.get("error")
        if error:
            inf_errors.append(
                f"{row.get('model')} {row.get('topology')} {row.get('shape')}: {error}"
            )

    train_rows.sort(
        key=lambda row: (
            row.get("model", ""),
            row.get("topology", ""),
            _shape_key(row.get("shape", "")),
            int(row.get("tp", 0) or 0),
            int(row.get("cp", 0) or 0),
            int(row.get("pp", 0) or 0),
        )
    )
    inf_rows.sort(
        key=lambda row: (
            row.get("model", ""),
            _shape_key(row.get("shape", "")),
            row.get("topology", ""),
        )
    )

    train_header = [
        "model",
        "topology",
        "shape",
        "tp",
        "cp",
        "pp",
        "mb",
        "runtime_s",
        "mem_exceeded",
        "mem_violation_gb",
        "error",
    ]
    inf_header = [
        "model",
        "topology",
        "shape",
        "tp",
        "cp",
        "pp",
        "mb",
        "runtime_s",
        "mem_exceeded",
        "mem_violation_gb",
        "error",
    ]

    print("\n=== Training Results ===")
    _print_table(train_rows, train_header)
    print("\n=== Inference Results ===")
    _print_table(inf_rows, inf_header)

    train_tsv = OUTPUT_ROOT / "2d_test_train.tsv"
    inf_tsv = OUTPUT_ROOT / "2d_test_inf.tsv"
    with train_tsv.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(train_header) + "\n")
        for row in train_rows:
            handle.write("\t".join(str(row.get(col, "")) for col in train_header) + "\n")
    with inf_tsv.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(inf_header) + "\n")
        for row in inf_rows:
            handle.write("\t".join(str(row.get(col, "")) for col in inf_header) + "\n")

    _save_cache(cache)

    train_titles: Dict[str, str] = {}
    for model in TRAIN_MODELS:
        params = read_yaml(model["config"]).get("model_param") or {}
        seq_len = int(params.get("seq_len", 0) or 0)
        display_name = MODEL_DISPLAY_NAMES.get(model["label"], model["label"])
        train_titles[model["label"]] = f"{display_name}\n[Seq={_format_tokens(seq_len)} tok]"

    inf_titles: Dict[str, str] = {}
    for model in INF_MODELS:
        params = read_yaml(model["config"]).get("model_param") or {}
        seq_len = int(params.get("seq_len", 0) or 0)
        decode_len = int(params.get("decode_len", 0) or 0)
        prefill_len = max(seq_len - decode_len, 0)
        display_name = MODEL_DISPLAY_NAMES.get(model["label"], model["label"])
        inf_titles[model["label"]] = (
            f"{display_name}\n"
            f"[Prefill={_format_tokens(prefill_len)} tok, "
            f"Decode={_format_tokens(decode_len)} tok]"
        )

    train_ok = [row for row in train_rows if row.get("runtime_s") != "error"]

    topo_groups: Dict[str, Dict[Tuple[str, int, int], Dict[str, float]]] = {}
    for row in train_ok:
        model = str(row.get("model", ""))
        combo = (
            str(row.get("shape", "")),
            int(row.get("tp", 0) or 0),
            int(row.get("pp", 0) or 0),
        )
        topo_groups.setdefault(model, {}).setdefault(combo, {})[row["topology"]] = float(
            row["runtime_s"]
        )

    train_experiments: Dict[str, List[Dict[str, Any]]] = {}
    for model_label, combo_map in topo_groups.items():
        experiments = []
        for combo, topo_map in combo_map.items():
            if "Mesh2D" not in topo_map or "Torus2D" not in topo_map:
                continue
            torus_val = topo_map["Torus2D"]
            if torus_val <= 0:
                continue
            shape_label, tp, pp = combo
            mesh_norm = topo_map["Mesh2D"] / torus_val
            experiments.append(
                {
                    "label": _format_train_label(shape_label, tp, pp),
                    "mesh": mesh_norm,
                    "torus": 1.0,
                }
            )
        experiments.sort(key=lambda item: item["label"])
        train_experiments[model_label] = experiments

    if len(train_experiments) > 1:
        label_sets = [
            set(item["label"] for item in items)
            for items in train_experiments.values()
            if items
        ]
        if label_sets:
            common_labels = set.intersection(*label_sets)
            if common_labels:
                for model_label, items in train_experiments.items():
                    train_experiments[model_label] = [
                        item for item in items if item["label"] in common_labels
                    ]
        for model_label, items in train_experiments.items():
            items.sort(key=lambda item: item["label"])

    train_order = [model["label"] for model in TRAIN_MODELS]
    if any(train_experiments.get(label) for label in train_order):
        max_items = max(
            len(train_experiments.get(label, [])) for label in train_order if train_order
        )
        axis_width = max(5.5, max_items * 0.7)
        fig, axes = plt.subplots(
            1,
            len(train_order),
            figsize=(axis_width * len(train_order), 6),
            sharey=True,
        )
        if len(train_order) == 1:
            axes = [axes]
        legend_handles: Optional[List[Any]] = None
        legend_labels: Optional[List[str]] = None
        for ax, model_label in zip(axes, train_order):
            entries = train_experiments.get(model_label, [])
            title = train_titles.get(model_label, model_label)
            if not entries:
                ax.set_title(title)
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            labels = [item["label"] for item in entries]
            mesh_vals = [item["mesh"] for item in entries]
            torus_vals = [item["torus"] for item in entries]
            handles, labels_text = _plot_dual_topology_axis(
                ax,
                labels,
                mesh_vals,
                torus_vals,
                title=title,
                ylabel=Y_LABEL if legend_handles is None else None,
            )
            if legend_handles is None:
                legend_handles = handles
                legend_labels = labels_text
        fig.suptitle("Training runtime vs 2D topology (64 GPUs)", y=0.99)
        if legend_handles and legend_labels:
            axes[-1].legend(legend_handles, legend_labels, loc="lower right")
        fig.subplots_adjust(top=0.81, bottom=0.11, left=0.1, right=0.98, wspace=0.08)
        fig.savefig(PLOT_DIR / "2d_test_train_topology.png", dpi=200)
        plt.close(fig)

    inf_ok = [row for row in inf_rows if row.get("runtime_s") != "error"]
    inf_groups: Dict[str, Dict[str, Dict[str, float]]] = {}
    for row in inf_ok:
        model = str(row.get("model", ""))
        shape_label = str(row.get("shape", ""))
        inf_groups.setdefault(model, {}).setdefault(shape_label, {})[
            row["topology"]
        ] = float(row["runtime_s"])

    inf_experiments: Dict[str, List[Dict[str, Any]]] = {}
    for model_label, shape_map in inf_groups.items():
        entries = []
        for shape_label, topo_map in shape_map.items():
            if "Mesh2D" not in topo_map or "Torus2D" not in topo_map:
                continue
            torus_val = topo_map["Torus2D"]
            if torus_val <= 0:
                continue
            dims = _shape_key(shape_label)
            tp = int(dims[0]) * int(dims[1])
            mesh_norm = topo_map["Mesh2D"] / torus_val
            entries.append(
                {
                    "label": _format_inf_label(shape_label, tp),
                    "mesh": mesh_norm,
                    "torus": 1.0,
                }
            )
        entries.sort(key=lambda item: item["label"])
        inf_experiments[model_label] = entries

    if len(inf_experiments) > 1:
        label_sets = [
            set(item["label"] for item in items)
            for items in inf_experiments.values()
            if items
        ]
        if label_sets:
            common_labels = set.intersection(*label_sets)
            if common_labels:
                for model_label, items in inf_experiments.items():
                    inf_experiments[model_label] = [
                        item for item in items if item["label"] in common_labels
                    ]
        for model_label, items in inf_experiments.items():
            items.sort(key=lambda item: item["label"])

    inf_order = [model["label"] for model in INF_MODELS]
    if any(inf_experiments.get(label) for label in inf_order):
        max_items = max(
            len(inf_experiments.get(label, [])) for label in inf_order if inf_order
        )
        axis_width = max(5.5, max_items * 0.7)
        fig, axes = plt.subplots(
            1,
            len(inf_order),
            figsize=(axis_width * len(inf_order), 6),
            sharey=True,
        )
        if len(inf_order) == 1:
            axes = [axes]
        legend_handles = None
        legend_labels = None
        for ax, model_label in zip(axes, inf_order):
            entries = inf_experiments.get(model_label, [])
            title = inf_titles.get(model_label, model_label)
            if not entries:
                ax.set_title(title)
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            labels = [item["label"] for item in entries]
            mesh_vals = [item["mesh"] for item in entries]
            torus_vals = [item["torus"] for item in entries]
            handles, labels_text = _plot_dual_topology_axis(
                ax,
                labels,
                mesh_vals,
                torus_vals,
                title=title,
                ylabel=Y_LABEL if legend_handles is None else None,
            )
            if legend_handles is None:
                legend_handles = handles
                legend_labels = labels_text
        fig.suptitle("Inference runtime vs 2D topology (TP only)", y=0.99)
        if legend_handles and legend_labels:
            axes[-1].legend(legend_handles, legend_labels, loc="lower right")
        fig.subplots_adjust(top=0.81, bottom=0.11, left=0.08, right=0.98, wspace=0.08)
        fig.savefig(PLOT_DIR / "2d_test_inf_topology.png", dpi=200)
        plt.close(fig)

    speedup_maps: Dict[str, Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]] = {}
    for topo in COMP_TOPO:
        speedup_maps[topo] = (
            _compute_speedup_ratio(
                train_ok,
                baseline_topology="Mesh2D",
                target_topology=topo,
            ),
            _compute_speedup_ratio(
                inf_ok,
                baseline_topology="Mesh2D",
                target_topology=topo,
            ),
        )
    train_speedup, inf_speedup = speedup_maps.get("Torus2D", ({}, {}))
    merged_shapes = sorted(
        {f"{shape[0]}x{shape[1]}" for shape in TRAIN_SHAPES}
        | {f"{shape[0]}x{shape[1]}" for shape in INF_SHAPES},
        key=_shape_key,
    )
    merged_order = [model["label"] for model in TRAIN_MODELS]
    for model in INF_MODELS:
        if model["label"] not in merged_order:
            merged_order.append(model["label"])
    if merged_shapes:
        _plot_speedup_by_model(
            train_speedup,
            inf_speedup,
            model_order=merged_order,
            shapes=merged_shapes,
            output_path=PLOT_DIR / "2d_test_merged_by_model.png",
        )
        for topo in COMP_TOPO:
            if topo not in COMP_TOPO_SPECS:
                continue
            train_map, inf_map = speedup_maps.get(topo, ({}, {}))
            spec = COMP_TOPO_SPECS[topo]
            y_limits = None
            if spec.get("dynamic_ylim", False):
                y_limits = _speedup_ylim(train_map, inf_map)
            _plot_super_merged_speedup(
                train_map,
                inf_map,
                model_order=merged_order,
                shapes=merged_shapes,
                output_path=PLOT_DIR / spec["output"],
                ylabel=spec["ylabel"],
                title=spec["title"],
                y_limits=y_limits,
            )

    if train_errors or inf_errors:
        print("\nErrors:")
        for msg in train_errors + inf_errors:
            print(f"- {msg}")

    mem_failures = [
        row
        for row in (train_rows + inf_rows)
        if str(row.get("mem_exceeded", "")).lower() == "yes"
    ]
    if mem_failures:
        print("\nMemory capacity exceeded:")
        for row in mem_failures:
            label = (
                f"{row.get('model')} {row.get('topology')} {row.get('shape')} "
                f"tp{row.get('tp')} cp{row.get('cp')} pp{row.get('pp')}"
            )
            violation = row.get("mem_violation_gb", "")
            suffix = f" (over by {violation} GiB)" if violation else ""
            print(f"- {label}{suffix}")
    else:
        print("\nMemory capacity exceeded: none")


if __name__ == "__main__":
    main()
