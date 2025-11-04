#!/usr/bin/env python3
"""
Parallelism sweep utility for DeepFlow LLM configurations.

Update the global configuration section below to point at the desired hardware
and model config files and to tailor the parallelism search space. The tool
enumerates every combination, filters those whose total GPU count falls inside
the configured bounds, evaluates runtime with DeepFlow, and plots a scatter
chart of accuracy (by default, 1 / runtime) versus GPU count with horizontal
jitter to avoid overlap.
"""

from __future__ import print_function

import copy
import itertools
import math
import os
import random
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from time_calculation_LLM import TimeCalculationLLM
from LLM_util import process_gemm_shapes


# -----------------------------------------------------------------------------
# Global configuration knobs (no CLI)
# -----------------------------------------------------------------------------

# Paths to the baseline configuration files
HARDWARE_CONFIG_PATH = "configs/hardware-config/a100_80GB.yaml"
MODEL_CONFIG_PATH = "configs/model-config/Llama3.1-405B.yaml"

# Parallelism values to sweep (dense grid). Edit to suit your search space.
# Keys should match the entries under the YAML "parallelism" section.
PARALLELISM_SWEEP = {
    "tp": [2**i for i in range(0, 9)],
    "cp": [2**i for i in range(0, 9)],
    "dp": [2**i for i in range(0, 9)],
    "lp": [2**i for i in range(0, 9)],
}

# Optional knobs that still live inside the parallelism section but do not
# affect GPU counts. Leave empty if you do not want to explore them.
OTHER_PARALLELISM_OPTIONS = {
    "tp_sp": [True],
}

# GPU count filter: only evaluate combinations whose TP*CP*DP*LP fall inside
# this inclusive range.
GPU_COUNT_MIN = 64
GPU_COUNT_MAX = 1024

# Select the y-axis metric for the scatter plot and report. Accepted values:
#   "runtime"     -> plot raw runtime in seconds (lower is better)
#   "performance" -> plot 1 / runtime (higher is better)
PLOT_METRIC = "runtime"

# Random seed for reproducible jitter in the scatter plot.
PLOT_JITTER_SEED = 1234
# Maximum absolute horizontal jitter (in GPU units).
PLOT_JITTER_WIDTH = 0.175

# Default output artefacts
PLOT_OUTPUT_PATH = "tools/parallelism_sweep.png"
PLOT_MFU_OUTPUT_PATH = "tools/parallelism_sweep_mfu.png"
REPORT_OUTPUT_PATH = "tools/parallelism_sweep.tsv"

# AstraSim cache handling within DeepFlow (mirrors run_perf default options).
ASTRA_CACHE_MODE = "NO_CACHE"  # Options: NO_CACHE, CACHE_READONLY, CACHE_READWRITE

# Maximum number of parallel worker processes (set <= available CPUs - 1). Set to 1 to disable multiprocessing.
MAX_WORKERS = 96


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def read_yaml(path):
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def determine_model_mode(model_config_path):
    model_dict = read_yaml(model_config_path)
    model_param = model_dict.get("model_param") or {}
    mode = model_param.get("mode")
    if not mode:
        raise ValueError("model_param.mode must be defined in {}".format(model_config_path))
    return mode


def cartesian_product(option_map):
    """Yield dictionaries for every combination inside option_map."""
    if not option_map:
        yield {}
        return
    keys = sorted(option_map.keys())
    value_lists = [option_map[key] for key in keys]
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def total_gpu_count(parallel_cfg):
    total = 1
    for axis in ("tp", "cp", "dp", "lp"):
        value = int(parallel_cfg.get(axis, 1) or 1)
        total *= max(1, value)
    return total


def make_temp_hw_config(base_hw_dict, parallel_settings):
    """Return parsed HW config for the given parallelism override."""
    updated = copy.deepcopy(base_hw_dict)
    parallel_block = updated.setdefault("parallelism", {})
    for key, value in parallel_settings.items():
        parallel_block[key] = value

    tmp_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    try:
        yaml.safe_dump(updated, tmp_file, default_flow_style=False, sort_keys=False)
        tmp_file.flush()
        tmp_file.close()
        return config.parse_config(tmp_file.name, config_type="hardware")
    finally:
        try:
            tmp_file.close()
        except Exception:
            pass
        try:
            os.unlink(tmp_file.name)
        except Exception:
            pass


def gpu_peak_flops(hw_config) -> float:
    """Return the theoretical peak FLOPs/s for a single GPU."""
    core = hw_config.tech_config.core
    bundles = core.num_bundles or 1
    per_cycle_flops = float(core.nominal_flop_rate_per_mcu)
    mcu_per_bundle = float(core.num_mcu_per_bundle)
    frequency = float(core.operating_frequency or core.nominal_frequency or 0.0)
    util = float(getattr(core, "util", 1.0) or 1.0)
    peak = per_cycle_flops * mcu_per_bundle * float(bundles)
    if frequency > 0:
        peak *= frequency
    return peak * util if peak > 0 else 0.0


def compute_total_flops(calculator: TimeCalculationLLM) -> float:
    """Estimate total FLOPs executed for one batch."""
    batch_size = calculator._effective_transformer_batch()
    vocab_size = calculator.vocab_size
    hidden_dim = calculator.hidden_dim
    seq_len = calculator.seq_len
    num_heads = calculator.num_heads
    kv_heads = calculator.kv_heads
    intermediate_size = calculator.intermediate_size
    num_SMs = calculator.hw_config.tech_config.core.num_bundles or 1

    transformer_timings, _ = calculator.compute_all_gemm_and_node_times(
        batch_size,
        vocab_size,
        hidden_dim,
        seq_len,
        num_heads,
        kv_heads,
        intermediate_size,
        num_SMs,
    )
    gemm_shapes = process_gemm_shapes(
        calculator,
        batch_size,
        seq_len,
        hidden_dim,
        num_heads,
        kv_heads,
        intermediate_size,
        vocab_size,
    )

    def _gemm_flops(shape) -> float:
        if shape is None:
            return 0.0
        try:
            dims = list(shape)
        except TypeError:
            return 0.0
        if len(dims) == 4:
            b, m, k, n = dims
            return 2.0 * float(b) * float(m) * float(k) * float(n)
        if len(dims) == 3:
            m, k, n = dims
            return 2.0 * float(m) * float(k) * float(n)
        return 0.0

    fallback_forward = {
        "qkv_proj": _gemm_flops(gemm_shapes.get("qkv_proj")),
        "attention_score": _gemm_flops(gemm_shapes.get("attention_score")),
        "attention_output": _gemm_flops(gemm_shapes.get("attention_output")),
        "output_proj": _gemm_flops(gemm_shapes.get("output_proj")),
        "ffn1": _gemm_flops(gemm_shapes.get("ffn1")),
        "ffn2": _gemm_flops(gemm_shapes.get("ffn2")),
        "linear_softmax": _gemm_flops(gemm_shapes.get("linear")),
    }
    fallback_forward["attention"] = (
        fallback_forward["attention_score"] + fallback_forward["attention_output"]
    )

    def _op_flops(name: str, fallback: float = 0.0, include_backward: bool = True) -> float:
        timing = transformer_timings.get(name)
        forward = 0.0
        if timing is not None and timing.forward is not None:
            forward = float(timing.forward.flops or 0.0)
        if forward <= 0 and fallback > 0:
            forward = fallback
        backward = 0.0
        if include_backward and timing is not None and timing.backward is not None:
            backward = float(timing.backward.flops or 0.0)
        if include_backward and backward <= 0 and forward > 0:
            backward = 2.0 * forward
        return forward + backward

    per_layer_ops = ("qkv_proj", "attention", "output_proj", "ffn1", "ffn2")
    per_layer_flops = sum(
        _op_flops(name, fallback_forward.get(name, 0.0)) for name in per_layer_ops
    )

    total_flops = per_layer_flops * calculator.num_layers

    total_flops += _op_flops("linear_softmax", fallback=fallback_forward.get("linear_softmax", 0.0), include_backward=True)
    total_flops += _op_flops("embedding", fallback=0.0, include_backward=True)

    return float(total_flops)


def evaluate_parallelism(hw_dict, model_config_obj, mode, parallel_settings):
    hw_config = make_temp_hw_config(hw_dict, parallel_settings)
    temp_dir = tempfile.mkdtemp(prefix="parallelism_sweep_")
    try:
        calculator = TimeCalculationLLM(hw_config, model_config_obj, mode, output_dir=temp_dir)
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            runtime = calculator.calc_time_llm()
            total_flops = compute_total_flops(calculator)
        performance = (1.0 / runtime) if runtime and runtime > 0.0 else float("nan")
        peak_flops = gpu_peak_flops(hw_config)
        num_gpus = total_gpu_count(parallel_settings)
        achieved_flops = (total_flops / runtime) if runtime > 0 else float("nan")
        denom = peak_flops * num_gpus if peak_flops and num_gpus else float("nan")
        mfu = (achieved_flops / denom) if denom and denom > 0 else float("nan")
        return {
            "runtime": runtime,
            "performance": performance,
            "total_flops": total_flops,
            "peak_flops": peak_flops,
            "mfu": mfu,
            "achieved_flops": achieved_flops,
            "memory_exceeded": getattr(calculator, "memory_capacity_exceeded", False),
            "memory_violation_gb": getattr(calculator, "memory_capacity_violation_gb", 0.0),
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def set_astrasim_cache_mode(mode_str):
    mapping = {
        "NO_CACHE": "NO_CACHE",
        "CACHE_READONLY": "CACHE_READONLY",
        "CACHE_READWRITE": "CACHE_READWRITE",
    }
    env_value = mapping.get(str(mode_str).strip().upper(), "NO_CACHE")
    os.environ["DEEPFLOW_ASTRA_CACHE_MODE"] = env_value


def write_report(results, path):
    header = [
        "num_gpus",
        "runtime_s",
        "performance_1_over_s",
        "total_flops",
        "achieved_flops_per_s",
        "peak_flops_per_gpu",
        "mfu",
        "memory_exceeded",
        "memory_violation_gb",
        "parallelism",
    ]
    with open(path, "w") as handle:
        handle.write("\t".join(header) + "\n")
        for entry in results:
            row = [
                str(entry["num_gpus"]),
                "{:.6f}".format(entry["runtime"]),
                (
                    "{:.6f}".format(entry["performance"])
                    if entry["performance"] == entry["performance"]
                    else "nan"
                ),
                "{:.6e}".format(entry["total_flops"]),
                "{:.6e}".format(entry["achieved_flops"]) if entry["achieved_flops"] == entry["achieved_flops"] else "nan",
                "{:.6e}".format(entry["peak_flops"]),
                "{:.6f}".format(entry["mfu"]) if entry["mfu"] == entry["mfu"] else "nan",
                str(entry["memory_exceeded"]),
                "{:.6f}".format(entry["memory_violation_gb"]),
                repr(entry["parallelism"]),
            ]
            handle.write("\t".join(row) + "\n")


def jitter_positions(gpu_counts, jitter_width):
    jittered = []
    for count in gpu_counts:
        offset = random.uniform(1 - jitter_width, 1 + jitter_width)
        jittered.append(max(count * offset, 1e-3))
    return jittered


def plot_results(results, output_path):
    if not results:
        print("No successful configurations to plot.", file=sys.stderr)
        return

    random.seed(PLOT_JITTER_SEED)
    xs = jitter_positions([item["num_gpus"] for item in results], PLOT_JITTER_WIDTH)
    metric_key = "runtime" if PLOT_METRIC.lower() == "runtime" else "performance"
    ys = [item[metric_key] for item in results]

    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, s=60, alpha=0.7, edgecolors="none")

    best = min(results, key=lambda item: item["runtime"])
    best_x = best["num_gpus"]
    best_y = best[metric_key]
    plt.scatter([best_x], [best_y], s=180, marker="*", c="red", label="Best runtime")

    plt.xlabel("Number of GPUs")
    if metric_key == "runtime":
        plt.ylabel("Runtime (s)")
    else:
        plt.ylabel("Performance (1 / s)")
    plt.xscale("log")
    if metric_key == "runtime":
        plt.yscale("log")
    plt.title("Parallelism sweep")
    plt.grid(alpha=0.3)
    xticks = sorted(set(item["num_gpus"] for item in results))
    plt.xticks(xticks, [str(int(x)) for x in xticks])
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print("Saved scatter plot to {}".format(output_path))


def plot_mfu(results, output_path):
    if not results:
        return
    random.seed(PLOT_JITTER_SEED)
    xs = jitter_positions([item["num_gpus"] for item in results], PLOT_JITTER_WIDTH)
    ys = [item["mfu"] for item in results]
    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, s=60, alpha=0.7, edgecolors="none")
    valid = [item for item in results if item["mfu"] == item["mfu"]]
    if valid:
        best = max(valid, key=lambda item: item["mfu"])
        plt.scatter([best["num_gpus"]], [best["mfu"]], s=180, marker="*", c="green", label="Max MFU")
    plt.xlabel("Number of GPUs")
    plt.ylabel("MFU")
    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.title("Parallelism sweep - MFU")
    plt.grid(alpha=0.3)
    xticks = sorted(set(item["num_gpus"] for item in results))
    plt.xticks(xticks, [str(int(x)) for x in xticks])
    if valid:
        plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print("Saved MFU scatter plot to {}".format(output_path))


_GLOBAL_MODEL_CONFIG = None
_GLOBAL_MODE = None
_GLOBAL_HW_DICT = None


def _worker_init(hw_dict, model_config_path, mode):
    global _GLOBAL_MODEL_CONFIG, _GLOBAL_MODE, _GLOBAL_HW_DICT
    _GLOBAL_HW_DICT = hw_dict
    _GLOBAL_MODE = mode
    _GLOBAL_MODEL_CONFIG = config.parse_config(model_config_path, config_type=mode)


def _worker_task(parallel_items: Tuple[Tuple[str, object], ...]):
    parallel_settings = {k: v for k, v in parallel_items}
    try:
        metrics = evaluate_parallelism(
            _GLOBAL_HW_DICT,
            _GLOBAL_MODEL_CONFIG,
            _GLOBAL_MODE,
            parallel_settings,
        )
        return {
            "status": "ok",
            "parallelism": parallel_settings,
            "metrics": metrics,
        }
    except Exception as exc:
        return {
            "status": "error",
            "parallelism": parallel_settings,
            "error": str(exc),
        }


def _build_tasks(
    gpu_choices: Iterable[Dict[str, int]],
    other_choices: Iterable[Dict[str, object]],
) -> List[Tuple[Tuple[str, object], ...]]:
    tasks: List[Tuple[Tuple[str, object], ...]] = []
    for gpu_choice in gpu_choices:
        for other_choice in other_choices:
            settings: Dict[str, object] = {}
            settings.update(gpu_choice)
            settings.update(other_choice)
            settings["mb"] = settings.get("lp", 1)
            tasks.append(tuple(sorted(settings.items())))
    return tasks


def main():
    set_astrasim_cache_mode(ASTRA_CACHE_MODE)

    base_hw_dict = read_yaml(HARDWARE_CONFIG_PATH)
    mode = determine_model_mode(MODEL_CONFIG_PATH)

    gpu_axes = list(PARALLELISM_SWEEP.keys())
    other_axes = list(OTHER_PARALLELISM_OPTIONS.keys())
    gpu_combos = list(cartesian_product(PARALLELISM_SWEEP))
    other_combos = list(cartesian_product(OTHER_PARALLELISM_OPTIONS))
    task_items = _build_tasks(gpu_combos, other_combos)
    print("Enumerating {} parallelism combinations: {}".format(len(task_items), ", ".join(gpu_axes)))

    results = []
    skipped_out_of_range = 0
    skipped_errors = 0
    error_messages: List[str] = []
    evaluated = 0

    filtered_tasks: List[Tuple[Tuple[str, object], ...]] = []
    for items in task_items:
        settings = dict(items)
        num_gpus = total_gpu_count(settings)
        if GPU_COUNT_MIN <= num_gpus <= GPU_COUNT_MAX:
            filtered_tasks.append(items)
        else:
            skipped_out_of_range += 1

    if not filtered_tasks:
        print("No configurations within GPU count bounds.")
        return

    available_cpus = max(1, os.cpu_count() or 1)
    if MAX_WORKERS is None or MAX_WORKERS <= 0:
        worker_limit = max(1, available_cpus - 1)
    else:
        worker_limit = min(MAX_WORKERS, max(1, available_cpus - 1))
    worker_count = max(1, worker_limit)
    print(f"Using {worker_count} worker process(es) (out of {available_cpus} CPUs).")

    if worker_count > 1 and len(filtered_tasks) > 1:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_worker_init,
            initargs=(base_hw_dict, MODEL_CONFIG_PATH, mode),
        ) as executor:
            futures = {executor.submit(_worker_task, items): items for items in filtered_tasks}
            with tqdm(total=len(filtered_tasks), desc="Evaluating", unit="config") as progress:
                for future in as_completed(futures):
                    progress.update(1)
                    result = future.result()
                    settings = result["parallelism"]
                    num_gpus = total_gpu_count(settings)
                    if result["status"] != "ok":
                        skipped_errors += 1
                        msg = result.get("error") or "unknown error"
                        error_messages.append(f"{settings}: {msg}")
                        continue
                    metrics = result["metrics"]
                    evaluated += 1
                    entry = {
                        "parallelism": settings,
                        "num_gpus": num_gpus,
                        "runtime": metrics["runtime"],
                        "performance": metrics["performance"],
                        "total_flops": metrics["total_flops"],
                        "achieved_flops": metrics["achieved_flops"],
                        "peak_flops": metrics["peak_flops"],
                        "mfu": metrics["mfu"],
                        "memory_exceeded": metrics["memory_exceeded"],
                        "memory_violation_gb": metrics["memory_violation_gb"],
                    }
                    results.append(entry)
    else:
        model_config_obj = config.parse_config(MODEL_CONFIG_PATH, config_type=mode)
        with tqdm(total=len(filtered_tasks), desc="Evaluating", unit="config") as progress:
            for items in filtered_tasks:
                settings = dict(items)
                num_gpus = total_gpu_count(settings)
                progress.update(1)
                try:
                    metrics = evaluate_parallelism(base_hw_dict, model_config_obj, mode, settings)
                except Exception as exc:
                    skipped_errors += 1
                    error_messages.append(f"{settings}: {exc}")
                    continue

                evaluated += 1
                entry = {
                    "parallelism": settings,
                    "num_gpus": num_gpus,
                    "runtime": metrics["runtime"],
                    "performance": metrics["performance"],
                    "total_flops": metrics["total_flops"],
                    "achieved_flops": metrics["achieved_flops"],
                    "peak_flops": metrics["peak_flops"],
                    "mfu": metrics["mfu"],
                    "memory_exceeded": metrics["memory_exceeded"],
                    "memory_violation_gb": metrics["memory_violation_gb"],
                }
                results.append(entry)

    total_skipped = skipped_out_of_range + skipped_errors
    if not results:
        print(
            "No valid configurations evaluated ({} skipped: {} out-of-range, {} errors).".format(
                total_skipped, skipped_out_of_range, skipped_errors
            )
        )
        if error_messages:
            print("Encountered errors for configurations:")
            for msg in error_messages:
                print(f"  {msg}")
        return

    best = min(results, key=lambda item: item["runtime"])
    print(
        f"Evaluated {evaluated} configuration(s); skipped {total_skipped} "
        f"(out_of_range={skipped_out_of_range}, errors={skipped_errors})."
    )
    if error_messages:
        print("Some configurations failed:")
        for msg in error_messages:
            print(f"  {msg}")
    print("\nBest configuration (lowest runtime):")
    print("  Parallelism: {}".format(best["parallelism"]))
    print("  GPUs: {}".format(best["num_gpus"]))
    print("  Runtime: {:.4f} s".format(best["runtime"]))
    print("  Performance (1/s): {:.4f}".format(best["performance"]))
    print("  Total FLOPs: {:.3e}".format(best["total_flops"]))
    print("  MFU: {:.3f}".format(best["mfu"]))
    if best["memory_exceeded"]:
        print("  Memory capacity exceeded by {:.3f} GB".format(best["memory_violation_gb"]))

    try:
        write_report(results, REPORT_OUTPUT_PATH)
        print("Wrote detailed report to {}".format(REPORT_OUTPUT_PATH))
    except Exception as exc:
        print("Warning: failed to write report: {}".format(exc), file=sys.stderr)

    plot_results(results, PLOT_OUTPUT_PATH)
    plot_mfu(results, PLOT_MFU_OUTPUT_PATH)


if __name__ == "__main__":
    main()
