# Copyright 2026 NanoCad lab, UCLA
# https://nanocad.ee.ucla.edu/
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

FIT_MODEL = False

try:
    from .validation_helpers import (
        ValidationSpec,
        run_validation_suite,
        parse_inference_time,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from validation_scripts.validation_helpers import (  # type: ignore
        ValidationSpec,
        run_validation_suite,
        parse_inference_time,
    )

# Data used to be in  https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/llama/performance.html in 2022.
# no longer available now, found via paper. "Performance Modeling and Workload Analysis of Distributed Large Language Model Training and Inference" by J. Kundu et al.

HW_CONFIGS = {
    "A100": "a100_80GB.yaml",
    "H100": "H100_SXM5_80GB.yaml",
}

MODEL_CONFIGS = {
    "Llama 2-7B": "Llama2-7B_inf.yaml",
    "Llama 2-13B": "Llama2-13B_inf.yaml",
    "Llama 2-70B": "Llama2-70B_inf.yaml",
}

MODEL_DISPLAY = {
    "Llama 2-7B": "Llama 2-7B",
    "Llama 2-13B": "Llama 2-13B",
    "Llama 2-70B": "Llama 2-70B",
}

NVIDIA_MODEL_CONFIGS = {
    "Llama 3.3-70B": "Llama3.1-70B_inf.yaml",
}

NVIDIA_MODEL_DISPLAY = {
    "Llama 3.3-70B": "Llama 3.3-70B",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PERF = os.path.join(PROJECT_ROOT, "run_perf.py")
VALIDATION_CONFIG_ROOT = os.path.join(PROJECT_ROOT, "validation_scripts", "validation_configs")
HARDWARE_CONFIG_PATH = os.path.join(VALIDATION_CONFIG_ROOT, "hardware-config")
MODEL_CONFIG_PATH = os.path.join(VALIDATION_CONFIG_ROOT, "model-config")
NVIDIA_MODEL_CONFIG_PATH = MODEL_CONFIG_PATH

FITTED_HW_SUFFIX = ".fitted.yaml"
IMEC_DATA_DIR = Path(__file__).parent / "imec_data"
NVIDIA_DATA_DIR = Path(__file__).parent / "nvidia_data"

NVIDIA_DATASETS = {
    "A100": {
        "csv": "8xA100_bf16_Llama3_3-70B.csv",
        "tp": 8,
        "model": "Llama 3.3-70B",
    },
    "H100": {
        "csv": "4xH100_fp16_Llama3_3-70B.csv",
        "tp": 4,
        "model": "Llama 3.3-70B",
    },
}


def _resolve_hw_config_path(device: str) -> str:
    base_name = HW_CONFIGS.get(device)
    if base_name is None:
        raise ValueError(f"No hardware config mapping for device {device}")
    base_path = os.path.join(HARDWARE_CONFIG_PATH, base_name)
    if FIT_MODEL:
        fitted_name = base_name.replace(".yaml", FITTED_HW_SUFFIX)
        fitted_path = os.path.join(HARDWARE_CONFIG_PATH, fitted_name)
        return fitted_path if os.path.exists(fitted_path) else base_path
    else:
        return base_path

def _load_data(csv_path: Path) -> pd.DataFrame:
    # Skip the leading "// filepath: ..." line by treating '/' as a comment char.
    df = pd.read_csv(csv_path, comment="/")
    # Normalize dtypes
    df["TP"] = df["TP"].astype(int)
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    return df.dropna(subset=["device", "model", "TP", "actual"])


@lru_cache(maxsize=None)
def _load_device_data(device: str) -> pd.DataFrame:
    csv_path = IMEC_DATA_DIR / f"{device}_inf.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Validation CSV not found for device {device}: {csv_path}")
    return _load_data(csv_path)


def _load_nvidia_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename_map = {
        "Input Tokens": "input_tokens",
        "Output Tokens": "output_tokens",
        "Concurrency": "concurrency",
        "TTFT (ms)": "ttft_ms",
        "ITL (ms)": "itl_ms",
        "Throughput (Tokens/s)": "throughput_tps",
    }
    missing = [key for key in rename_map if key not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in NVIDIA data {csv_path}: {missing}")
    df = df.rename(columns=rename_map)
    for col in ("input_tokens", "output_tokens", "concurrency", "ttft_ms", "itl_ms"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["input_tokens", "output_tokens", "concurrency", "ttft_ms", "itl_ms"])
    df["input_tokens"] = df["input_tokens"].astype(int)
    df["output_tokens"] = df["output_tokens"].astype(int)
    df["concurrency"] = df["concurrency"].astype(int)
    df["actual"] = (df["ttft_ms"] + df["itl_ms"] * (df["output_tokens"] - 1).clip(lower=0)) / 1000.0
    return df


@lru_cache(maxsize=None)
def _load_nvidia_device_data(device: str) -> pd.DataFrame:
    dataset = NVIDIA_DATASETS.get(device)
    if not dataset:
        raise ValueError(f"No NVIDIA dataset defined for device {device}")
    csv_path = NVIDIA_DATA_DIR / dataset["csv"]
    if not csv_path.exists():
        raise FileNotFoundError(f"NVIDIA CSV not found for device {device}: {csv_path}")
    df = _load_nvidia_data(csv_path)
    df["device"] = device
    df["model"] = dataset["model"]
    df["TP"] = int(dataset["tp"])
    return df


def _iter_tests(df: pd.DataFrame):
    for _, row in df.iterrows():
        yield (row["device"], row["model"], row["TP"])


def _iter_nvidia_tests(df: pd.DataFrame):
    for _, row in df.iterrows():
        yield (
            row["device"],
            row["model"],
            row["TP"],
            row["input_tokens"],
            row["output_tokens"],
            row["concurrency"],
        )


def _build_spec(device: str, model: str, tp: int, idx: int, network_ignored: bool) -> Tuple[ValidationSpec, str, str]:
    label = f"{device} {model} TP={tp}"

    model_overrides = {
        "model_param": {
            "global_batch_size": 1,
            "seq_len": 400,
            "decode_len": 200
        }
    }

    hw_overrides: Dict[str, Dict[str, object]] = {
        "parallelism": {
            "tp": int(tp),
            "tp_sp": True,
            "cp": 1,
            "pp": 1,
            "mb": 1,
            "train": {"dp": 1, "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        }
    }
    if network_ignored:
        override_bw = {
            "network": {
                "dimensions": [
                    {
                        "id": "dim0",
                        "topology": {"bandwidth": "100000 GB", "latency": "1e-9" },
                    }
                ]
            },
        }
        hw_overrides.update(override_bw)

    spec = ValidationSpec(
        label=label,
        model_overrides=model_overrides,
        hardware_overrides=hw_overrides,
        model_config_path=os.path.join(MODEL_CONFIG_PATH, MODEL_CONFIGS.get(model)),
        hardware_config_path=_resolve_hw_config_path(device),
        metadata={"device": device, "model": model, "tp": int(tp)},
        order=idx,
    )
    hw_config_path = spec.hardware_config_path  # type: ignore[arg-type]
    model_config_path = spec.model_config_path  # type: ignore[arg-type]
    return spec, hw_config_path, model_config_path


def _build_nvidia_spec(
    device: str,
    model: str,
    tp: int,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    idx: int,
    network_ignored: bool,
) -> Tuple[ValidationSpec, str, str]:
    label = (
        f"NVIDIA {device} {model} TP={tp} "
        f"in={input_tokens} out={output_tokens} bs={concurrency}"
    )

    model_overrides = {
        "model_param": {
            "global_batch_size": int(concurrency),
            "seq_len": int(input_tokens) + int(output_tokens),
            "decode_len": int(output_tokens),
        }
    }

    hw_overrides: Dict[str, Dict[str, object]] = {
        "parallelism": {
            "tp": int(tp),
            "tp_sp": True,
            "cp": 1,
            "pp": 1,
            "mb": 1,
            "train": {"dp": 1, "ep": 1, "tp_ep": True},
            "inference": {"replica_count": 1, "moe_dp": 1},
        }
    }
    if network_ignored:
        override_bw = {
            "network": {
                "dimensions": [
                    {
                        "id": "dim0",
                        "topology": {"bandwidth": "100000 GB", "latency": "1e-9"},
                    }
                ]
            },
        }
        hw_overrides.update(override_bw)

    model_cfg = NVIDIA_MODEL_CONFIGS.get(model)
    if model_cfg is None:
        raise ValueError(f"No model config mapping for NVIDIA model {model}")

    spec = ValidationSpec(
        label=label,
        model_overrides=model_overrides,
        hardware_overrides=hw_overrides,
        model_config_path=os.path.join(NVIDIA_MODEL_CONFIG_PATH, model_cfg),
        hardware_config_path=_resolve_hw_config_path(device),
        metadata={
            "device": device,
            "model": model,
            "tp": int(tp),
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "concurrency": int(concurrency),
            "suite": "NVIDIA",
        },
        order=idx,
    )
    hw_config_path = spec.hardware_config_path  # type: ignore[arg-type]
    model_config_path = spec.model_config_path  # type: ignore[arg-type]
    return spec, hw_config_path, model_config_path


def _lookup_actual(device: str, model: str, tp: int) -> float:
    data = _load_device_data(device)
    matches = data.loc[
        (data["device"] == device) & (data["model"] == model) & (data["TP"] == int(tp)),
        "actual",
    ]
    if matches.empty:
        raise ValueError(f"No reference inference time for {device} {model} TP={tp}")
    return float(matches.values[0])


def _lookup_nvidia_actual(
    device: str,
    model: str,
    tp: int,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
) -> float:
    data = _load_nvidia_device_data(device)
    matches = data.loc[
        (data["device"] == device)
        & (data["model"] == model)
        & (data["TP"] == int(tp))
        & (data["input_tokens"] == int(input_tokens))
        & (data["output_tokens"] == int(output_tokens))
        & (data["concurrency"] == int(concurrency)),
        "actual",
    ]
    if matches.empty:
        raise ValueError(
            "No NVIDIA reference inference time for "
            f"{device} {model} TP={tp} in={input_tokens} out={output_tokens} bs={concurrency}"
        )
    return float(matches.values[0])


def run_single(
    device: str,
    model: str,
    tp: int,
    *,
    network_ignored: bool = True,
    actual_inference_time_s: Optional[float] = None,
    emit_logs: bool = False,
) -> Dict[str, object]:
    spec, hw_config_path, model_config_path = _build_spec(device, model, tp, idx=0, network_ignored=network_ignored)
    validation_results = run_validation_suite(
        [spec],
        base_model_config_path=model_config_path,
        base_hardware_config_path=hw_config_path,
        result_parser=parse_inference_time,
        run_perf_path=RUN_PERF,
    )

    result = validation_results[0]
    inference_time_s = float(result.metrics.get("inference_time_s", float("nan"))) if result.success else float("nan")
    err_detail = None if result.success else (result.error or "RAPID-LLM run failed")

    if actual_inference_time_s is None:
        try:
            actual_inference_time_s = _lookup_actual(device, model, tp)
        except Exception:
            actual_inference_time_s = float("nan")

    if math.isnan(inference_time_s) or actual_inference_time_s == 0 or math.isnan(actual_inference_time_s):
        pct_error = float("nan")
    else:
        pct_error = abs(inference_time_s - actual_inference_time_s) / actual_inference_time_s * 100.0

    if emit_logs:
        block_lines = [
            f"\n=== Result (device={device}, model={model}, TP={tp}) ===",
        ]
        if not math.isnan(inference_time_s):
            block_lines.append(f"  RAPID-LLM Inference Time: {inference_time_s:.2f}s")
            block_lines.append(f"  Actual Inference Time:   {actual_inference_time_s:.2f}s")
            block_lines.append(f"  Percent Error:          {pct_error:.2f}%")
        else:
            block_lines.append(f"  RAPID-LLM run failed. {err_detail or ''}".rstrip())
            if result.raw_output:
                block_lines.append(result.raw_output.strip())
        print("\n".join(block_lines))

    return {
        "success": result.success,
        "inference_time_s": inference_time_s,
        "actual_inference_time_s": actual_inference_time_s,
        "pct_error": pct_error,
        "error": err_detail,
        "raw_output": result.raw_output,
    }


def build_specs_for_device(
    device: str,
    *,
    network_ignored: bool = True,
    models: Optional[Iterable[str]] = None,
) -> Tuple[List[ValidationSpec], Dict[Tuple[str, int], float], str, str]:
    data = _load_device_data(device)
    model_filter = set(models) if models is not None else None
    specs: List[ValidationSpec] = []
    actual_lookup: Dict[Tuple[str, int], float] = {}
    base_model_path: Optional[str] = None
    hw_config_path: Optional[str] = None
    idx = 0
    for device_val, model, tp in _iter_tests(data):
        if model_filter and model not in model_filter:
            continue
        spec, hw_path, model_path = _build_spec(device_val, model, tp, idx, network_ignored)
        specs.append(spec)
        actual_lookup[(model, int(tp))] = _lookup_actual(device_val, model, int(tp))
        base_model_path = base_model_path or model_path
        hw_config_path = hw_config_path or hw_path
        idx += 1

    if not specs:
        raise ValueError(f"No validation specs generated for device={device} (models={models}).")
    return specs, actual_lookup, base_model_path, hw_config_path


def build_nvidia_specs_for_device(
    device: str,
    *,
    network_ignored: bool = True,
    models: Optional[Iterable[str]] = None,
) -> Tuple[List[ValidationSpec], Dict[Tuple[str, int, int, int, int], float], str, str]:
    data = _load_nvidia_device_data(device)
    model_filter = set(models) if models is not None else None
    specs: List[ValidationSpec] = []
    actual_lookup: Dict[Tuple[str, int, int, int, int], float] = {}
    base_model_path: Optional[str] = None
    hw_config_path: Optional[str] = None
    idx = 0
    for device_val, model, tp, input_tokens, output_tokens, concurrency in _iter_nvidia_tests(data):
        if model_filter and model not in model_filter:
            continue
        spec, hw_path, model_path = _build_nvidia_spec(
            device_val,
            model,
            int(tp),
            int(input_tokens),
            int(output_tokens),
            int(concurrency),
            idx,
            network_ignored,
        )
        specs.append(spec)
        actual_lookup[(model, int(tp), int(input_tokens), int(output_tokens), int(concurrency))] = (
            _lookup_nvidia_actual(device_val, model, int(tp), int(input_tokens), int(output_tokens), int(concurrency))
        )
        base_model_path = base_model_path or model_path
        hw_config_path = hw_config_path or hw_path
        idx += 1

    if not specs:
        raise ValueError(f"No NVIDIA validation specs generated for device={device} (models={models}).")
    return specs, actual_lookup, base_model_path, hw_config_path


def compute_pct_errors(results, actual_lookup: Dict[Tuple[str, int], float]):
    rows: List[Dict[str, object]] = []
    for res in results:
        metadata = res.spec.metadata or {}
        model = metadata.get("model")
        tp = int(metadata.get("tp")) if "tp" in metadata else None
        inf_time = float(res.metrics.get("inference_time_s", float("nan"))) if res.success else float("nan")
        actual = actual_lookup.get((model, tp)) if model is not None and tp is not None else float("nan")
        if math.isnan(inf_time) or actual is None or actual == 0 or math.isnan(actual):
            pct_error = float("nan")
        else:
            pct_error = abs(inf_time - actual) / actual * 100.0
        rows.append(
            {
                "device": metadata.get("device"),
                "model": model,
                "tp": tp,
                "inference_time_s": inf_time,
                "actual_inference_time_s": actual,
                "pct_error": pct_error,
                "display_model": MODEL_DISPLAY.get(str(model), str(model)),
                "success": res.success,
                "error": res.error,
            }
        )
    return rows


def compute_nvidia_pct_errors(
    results,
    actual_lookup: Dict[Tuple[str, int, int, int, int], float],
):
    rows: List[Dict[str, object]] = []
    for res in results:
        metadata = res.spec.metadata or {}
        model = metadata.get("model")
        tp = int(metadata.get("tp")) if "tp" in metadata else None
        input_tokens = int(metadata.get("input_tokens")) if "input_tokens" in metadata else None
        output_tokens = int(metadata.get("output_tokens")) if "output_tokens" in metadata else None
        concurrency = int(metadata.get("concurrency")) if "concurrency" in metadata else None
        inf_time = float(res.metrics.get("inference_time_s", float("nan"))) if res.success else float("nan")
        actual = (
            actual_lookup.get((model, tp, input_tokens, output_tokens, concurrency))
            if model is not None and tp is not None and input_tokens is not None
            and output_tokens is not None and concurrency is not None
            else float("nan")
        )
        if math.isnan(inf_time) or actual == 0 or math.isnan(actual):
            pct_error = float("nan")
        else:
            pct_error = abs(inf_time - actual) / actual * 100.0
        rows.append(
            {
                "device": metadata.get("device"),
                "model": model,
                "tp": tp,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "concurrency": concurrency,
                "inference_time_s": inf_time,
                "actual_inference_time_s": actual,
                "pct_error": pct_error,
                "display_model": NVIDIA_MODEL_DISPLAY.get(str(model), str(model)),
                "success": res.success,
                "error": res.error,
                "suite": metadata.get("suite", "NVIDIA"),
            }
        )
    return rows


def plot_device(df: pd.DataFrame, device: str, outdir: Path) -> Path | None:
    sub = df[df["device"] == device].copy()
    if sub.empty:
        return None
    sub.sort_values(["model", "TP"], inplace=True)

    # Hierarchical grouping: model cluster (primary) -> TP subgroup (secondary with two bars: seconds & actual)
    # Enforce explicit primary model order
    desired_order = ["Llama 2-7B", "Llama 2-13B", "Llama 2-70B"]
    unique_models = list(pd.unique(sub["model"]))
    models = [m for m in desired_order if m in unique_models] + [
        m for m in unique_models if m not in desired_order
    ]
    display_models = [MODEL_DISPLAY.get(m, m) for m in models]

    bar_width = 0.28  # width per individual bar
    tp_gap = 0.08  # gap between TP subgroups inside a model cluster
    model_gap = 0.7  # gap between model clusters

    x_seconds = []
    x_actual = []
    seconds_vals = []
    actual_vals = []
    tp_midpoints = []
    tp_labels = []
    model_centers = []
    model_bounds = []

    current_x = 0.0
    for m in models:
        m_rows = sub[sub["model"] == m]
        tps = sorted(m_rows["TP"].unique())
        cluster_start = current_x
        for tp in tps:
            row = m_rows[m_rows["TP"] == tp].iloc[0]
            x_sec = current_x
            x_act = current_x + bar_width
            x_seconds.append(x_sec)
            x_actual.append(x_act)
            seconds_vals.append(row["seconds"])
            actual_vals.append(row["actual"])
            # Midpoint of the TP subgroup (between the two bars)
            tp_mid = x_sec + bar_width * 0.5
            tp_midpoints.append(tp_mid)
            tp_labels.append(f"TP{tp}")
            current_x += 2 * bar_width + tp_gap
        cluster_end = current_x - tp_gap  # last subgroup end (exclude trailing tp_gap)
        model_centers.append((cluster_start + cluster_end) / 2)
        model_bounds.append((cluster_start, cluster_end))
        current_x += model_gap  # gap after cluster

    # Dynamic figure width
    fig_w = max(8.0, 0.12 * len(tp_midpoints) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    bars_seconds = ax.bar(
        x_seconds, seconds_vals, bar_width, label="RAPID-LLM", color="#1f77b4"
    )
    bars_actual = ax.bar(
        x_actual, actual_vals, bar_width, label="Actual", color="#0bbd37"
    )

    # Primary ticks: model names at cluster centers
    ax.set_xticks(model_centers)
    ax.set_xticklabels(display_models, fontsize=11)
    ax.set_ylabel("Inference Latency (s)")
    ax.set_title(
        f"Validation of Inference Latency on Systems of {device} (network ignored)"
    )

    # Secondary TP labels ABOVE bars (centered over subgroup)
    ymin, ymax = ax.get_ylim()
    pad = 0.02 * (ymax - ymin)
    subgroup_max = [max(s, a) for s, a in zip(seconds_vals, actual_vals)]
    new_ymax = max(ymax, max(subgroup_max) + 3 * pad)
    ax.set_ylim(ymin, new_ymax)
    for mid, lbl, v in zip(tp_midpoints, tp_labels, subgroup_max):
        ax.text(mid, v + pad, lbl, ha="center", va="bottom", fontsize=8)

    # Draw light separators between model clusters
    for start, end in model_bounds:
        ax.axvspan(
            start - bar_width * 0.5,
            end - bar_width * 0.5,
            facecolor="#000000",
            alpha=0.02,
        )

    # Custom legend
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.margins(x=0.02, y=0.1)

    outpath = outdir / f"inf_{device}.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def plot_nvidia_device(df: pd.DataFrame, device: str, outdir: Path) -> Path | None:
    sub = df[df["device"] == device].copy()
    if sub.empty:
        return None
    sub.sort_values(["input_tokens", "output_tokens", "concurrency"], inplace=True)

    labels = [
        f"in{int(row['input_tokens'])}-out{int(row['output_tokens'])}-bs{int(row['concurrency'])}"
        for _, row in sub.iterrows()
    ]
    seconds_vals = sub["seconds"].astype(float).tolist()
    actual_vals = sub["actual"].astype(float).tolist()

    x = list(range(len(labels)))
    bar_width = 0.35
    fig_w = max(8.0, 0.4 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar([i - bar_width / 2 for i in x], seconds_vals, bar_width, label="RAPID-LLM")
    ax.bar([i + bar_width / 2 for i in x], actual_vals, bar_width, label="Actual")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Inference Latency (s)")
    ax.set_title(f"NVIDIA inference validation on {device}")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    outpath = outdir / f"nvidia_inf_{device}.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def _plot_nvidia_error_bars(rows: List[Dict[str, object]], path: Path, title: str) -> Path:
    labels: List[str] = []
    errors: List[float] = []
    for row in rows:
        model = row.get("display_model") or row.get("model")
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        concurrency = row.get("concurrency")
        labels.append(f"{model} in{input_tokens}-out{output_tokens}-bs{concurrency}")
        errors.append(float(row.get("pct_error", float("nan"))))

    fig_w = max(6.0, 0.5 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    bars = ax.bar(range(len(errors)), errors, color="#1f77b4")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Percent Error")
    ax.set_title(title)
    for rect, err in zip(bars, errors):
        if math.isnan(err):
            continue
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, f"{err:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run(
    enable_plot: bool = True,
    network_ignored: bool = True,
    device: str = "A100",
    models: Optional[Sequence[str]] = None,
    emit_logs: bool = True,
):
    pct_errors = []
    data = _load_device_data(device)

    specs, actual_lookup, base_model_path, hw_config_path = build_specs_for_device(
        device, network_ignored=network_ignored, models=models
    )

    validation_results = run_validation_suite(
        specs,
        base_model_config_path=base_model_path,
        base_hardware_config_path=hw_config_path,
        result_parser=parse_inference_time,
        run_perf_path=RUN_PERF,
    )

    rows = compute_pct_errors(validation_results, actual_lookup)

    for row in rows:
        pct_error = float(row["pct_error"])
        pct_errors.append(pct_error)
        data.loc[
            (data["device"] == row["device"]) & (data["model"] == row["model"]) & (data["TP"] == row["tp"]),
            "seconds",
        ] = row["inference_time_s"]

        data.loc[
            (data["device"] == row["device"]) & (data["model"] == row["model"]) & (data["TP"] == row["tp"]),
            "pct_error",
        ] = pct_error

        if emit_logs:
            block_lines = [
                f"\n=== Result (device={row['device']}, model={row['model']}, TP={row['tp']}) ===",
            ]
            if not math.isnan(row["inference_time_s"]):
                block_lines.append(f"  RAPID-LLM Inference Time: {float(row['inference_time_s']):.2f}s")
                block_lines.append(f"  Actual Inference Time:   {float(row['actual_inference_time_s']):.2f}s")
                block_lines.append(f"  Percent Error:          {pct_error:.2f}%")
            else:
                block_lines.append(f"  RAPID-LLM run failed. {(row.get('error') or '')}".rstrip())
            print("\n".join(block_lines))

    if enable_plot:
        out_dir = os.path.join(PROJECT_ROOT, "output", "validation")
        os.makedirs(out_dir, exist_ok=True)
        out_dir = Path(out_dir)

        outputs = []
        for device in ["A100", "H100"]:
            out = plot_device(data, device, out_dir)
            if out is not None:
                outputs.append(out)
        # Secondary bar plot matching the training style (combined devices).
        bar_path = _plot_error_bars(
            rows,
            path=out_dir / "inf_errors_bar.png",
            title="Inference validation (combined)",
        )
        outputs.append(bar_path)
        if not outputs:
            print("No plots generated (no matching device rows).")
        else:
            for p in outputs:
                print(f"Saved: {p}")
    valid_errors = [e for e in pct_errors if not math.isnan(e)]
    avg_abs_error = sum(valid_errors) / len(valid_errors) if valid_errors else float("nan")
    if emit_logs:
        print("Average absolute percent error across all tests: {:.2f}%".format(avg_abs_error))
    return {"avg_abs_error": avg_abs_error, "rows": rows}


def run_nvidia(
    enable_plot: bool = True,
    network_ignored: bool = True,
    device: str = "A100",
    models: Optional[Sequence[str]] = None,
    emit_logs: bool = True,
):
    pct_errors = []
    data = _load_nvidia_device_data(device)

    specs, actual_lookup, base_model_path, hw_config_path = build_nvidia_specs_for_device(
        device, network_ignored=network_ignored, models=models
    )

    validation_results = run_validation_suite(
        specs,
        base_model_config_path=base_model_path,
        base_hardware_config_path=hw_config_path,
        result_parser=parse_inference_time,
        run_perf_path=RUN_PERF,
    )

    rows = compute_nvidia_pct_errors(validation_results, actual_lookup)

    for row in rows:
        pct_error = float(row["pct_error"])
        pct_errors.append(pct_error)
        data.loc[
            (data["device"] == row["device"])
            & (data["model"] == row["model"])
            & (data["TP"] == row["tp"])
            & (data["input_tokens"] == row["input_tokens"])
            & (data["output_tokens"] == row["output_tokens"])
            & (data["concurrency"] == row["concurrency"]),
            "seconds",
        ] = row["inference_time_s"]

        data.loc[
            (data["device"] == row["device"])
            & (data["model"] == row["model"])
            & (data["TP"] == row["tp"])
            & (data["input_tokens"] == row["input_tokens"])
            & (data["output_tokens"] == row["output_tokens"])
            & (data["concurrency"] == row["concurrency"]),
            "pct_error",
        ] = pct_error

        if emit_logs:
            block_lines = [
                (
                    f"\n=== NVIDIA Result (device={row['device']}, model={row['model']}, "
                    f"TP={row['tp']}, in={row['input_tokens']}, out={row['output_tokens']}, "
                    f"bs={row['concurrency']}) ==="
                ),
            ]
            if not math.isnan(row["inference_time_s"]):
                block_lines.append(f"  RAPID-LLM Inference Time: {float(row['inference_time_s']):.2f}s")
                block_lines.append(f"  Actual Inference Time:   {float(row['actual_inference_time_s']):.2f}s")
                block_lines.append(f"  Percent Error:          {pct_error:.2f}%")
            else:
                block_lines.append(f"  RAPID-LLM run failed. {(row.get('error') or '')}".rstrip())
            print("\n".join(block_lines))

    if enable_plot:
        out_dir = os.path.join(PROJECT_ROOT, "output", "validation")
        os.makedirs(out_dir, exist_ok=True)
        out_dir = Path(out_dir)
        outputs = []
        out = plot_nvidia_device(data, device, out_dir)
        if out is not None:
            outputs.append(out)
        error_bar_path = _plot_nvidia_error_bars(
            rows,
            path=out_dir / f"nvidia_inf_errors_bar_{device}.png",
            title=f"NVIDIA inference validation ({device})",
        )
        outputs.append(error_bar_path)
        if not outputs:
            print("No NVIDIA plots generated (no matching device rows).")
        else:
            for p in outputs:
                print(f"Saved: {p}")

    valid_errors = [e for e in pct_errors if not math.isnan(e)]
    avg_abs_error = sum(valid_errors) / len(valid_errors) if valid_errors else float("nan")
    if emit_logs:
        print("NVIDIA average absolute percent error across all tests: {:.2f}%".format(avg_abs_error))
    return {"avg_abs_error": avg_abs_error, "rows": rows}


def _plot_error_bars(rows: List[Dict[str, object]], path: Path, title: str) -> Path:
    # Build labels like "<model> xTP<tp>", drop device prefixes.
    labels: List[str] = []
    errors: List[float] = []
    for row in rows:
        model = row.get("display_model") or row.get("model")
        tp = row.get("tp")
        labels.append(f"{model} xTP{tp}")
        errors.append(float(row.get("pct_error", float("nan"))))

    fig_w = max(6.0, 0.6 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    bars = ax.bar(range(len(errors)), errors, color="#1f77b4")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Percent Error")
    ax.set_title(title)
    for rect, err in zip(bars, errors):
        if math.isnan(err):
            continue
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height, f"{err:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_combined_a100_error_bars(
    imec_rows: List[Dict[str, object]],
    nvidia_rows: List[Dict[str, object]],
    outdir: Path,
) -> Optional[Path]:
    imec_rows = [row for row in imec_rows if row.get("device") == "A100"]
    nvidia_rows = [row for row in nvidia_rows if row.get("device") == "A100"]
    if not imec_rows or not nvidia_rows:
        return None

    imec_order = {"Llama 2-7B": 0, "Llama 2-13B": 1, "Llama 2-70B": 2}
    imec_models = sorted(
        {row.get("model") for row in imec_rows if row.get("model")},
        key=lambda m: (imec_order.get(m, 99), str(m)),
    )
    nvidia_models = sorted({row.get("model") for row in nvidia_rows if row.get("model")})
    all_models = [m for m in imec_models if m] + [m for m in nvidia_models if m and m not in imec_models]

    palette = sns.color_palette("tab10", n_colors=max(4, len(all_models) + 1))
    color_map = {model: palette[idx] for idx, model in enumerate(all_models)}
    bottom_color = palette[min(len(imec_models), len(palette) - 1)]

    font_params = {
        "font.size": plt.rcParams.get("font.size", 10.0) * 1.3,
        "axes.titlesize": plt.rcParams.get("axes.titlesize", 12.0) * 1.3,
        "axes.labelsize": plt.rcParams.get("axes.labelsize", 11.0) * 1.3,
        "xtick.labelsize": plt.rcParams.get("xtick.labelsize", 10.0) * 1.3,
        "ytick.labelsize": plt.rcParams.get("ytick.labelsize", 10.0) * 1.3,
        "legend.fontsize": plt.rcParams.get("legend.fontsize", 10.0) * 1.3,
    }

    # Top: IMEC (prefill/decode 200/200).
    imec_tps = sorted({int(row.get("tp")) for row in imec_rows if row.get("tp") is not None})
    imec_vals: Dict[Tuple[str, int], float] = {}
    for row in imec_rows:
        model = row.get("model")
        tp = int(row.get("tp")) if row.get("tp") is not None else None
        if model is None or tp is None:
            continue
        imec_vals[(model, tp)] = float(row.get("pct_error", float("nan")))

    bar_width = 0.2
    group_gap = 0.3
    top_positions = []
    top_labels = []
    top_x = []
    top_heights = []
    top_colors = []
    current_x = 0.0
    for tp in imec_tps:
        group_start = current_x
        for idx, model in enumerate(imec_models):
            value = imec_vals.get((model, tp))
            if value is None or math.isnan(value):
                continue
            top_x.append(current_x + idx * bar_width)
            top_heights.append(value)
            top_colors.append(color_map.get(model, palette[0]))
        group_center = group_start + (len(imec_models) - 1) * bar_width / 2
        top_positions.append(group_center)
        top_labels.append(f"(200/200) TP{tp}")
        current_x += len(imec_models) * bar_width + group_gap

    # Bottom: NVIDIA (include batch size; grouped by input/output + TP).
    nvidia_entries = []
    bs_levels = sorted(
        {int(row.get("concurrency")) for row in nvidia_rows if row.get("concurrency") is not None}
    )
    hatch_map = {1: "", 5: "xx", 25: "oo"}
    for row in nvidia_rows:
        model = row.get("model")
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        tp = row.get("tp")
        concurrency = row.get("concurrency")
        if None in (model, input_tokens, output_tokens, tp, concurrency):
            continue
        value = float(row.get("pct_error", float("nan")))
        if math.isnan(value):
            continue
        nvidia_entries.append(
            {
                "model": str(model),
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "tp": int(tp),
                "concurrency": int(concurrency),
                "pct_error": value,
            }
        )
    nvidia_entries.sort(
        key=lambda item: (item["input_tokens"], item["output_tokens"], item["tp"], item["concurrency"])
    )

    group_keys = []
    nvidia_vals: Dict[Tuple[int, int, int, int], float] = {}
    for entry in nvidia_entries:
        key = (entry["input_tokens"], entry["output_tokens"], entry["tp"])
        if key not in group_keys:
            group_keys.append(key)
        nvidia_vals[(entry["input_tokens"], entry["output_tokens"], entry["tp"], entry["concurrency"])] = (
            entry["pct_error"]
        )

    bottom_positions = []
    bottom_labels = []
    bottom_x = []
    bottom_heights = []
    bottom_hatches = []
    current_x = 0.0
    bs_count = max(1, len(bs_levels))
    for input_tokens, output_tokens, tp in group_keys:
        group_center = current_x
        for idx, bs in enumerate(bs_levels):
            value = nvidia_vals.get((input_tokens, output_tokens, tp, bs))
            if value is None:
                continue
            offset = (idx - (bs_count - 1) / 2) * bar_width
            bottom_x.append(group_center + offset)
            bottom_heights.append(value)
            bottom_hatches.append(hatch_map.get(bs, "///"))
        bottom_positions.append(group_center)
        bottom_labels.append(f"({input_tokens}/{output_tokens}) TP{tp}")
        current_x += bs_count * bar_width + group_gap

    with plt.rc_context(font_params):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=False,
            gridspec_kw={"hspace": 0.3},
        )
        axes[0].bar(top_x, top_heights, bar_width, color=top_colors)
        axes[0].set_title("NVIDIA Llama2 (bs=1)")
        axes[0].set_ylabel("")
        axes[0].set_xticks(top_positions)
        axes[0].set_xticklabels(top_labels, fontstyle="italic")
        axes[0].grid(axis="y", linestyle="--", alpha=0.8)
        if top_heights:
            top_max = max(top_heights)
            axes[0].set_ylim(0, top_max * 1.3)

        model_handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_map[model])
            for model in imec_models
        ]
        model_labels = [MODEL_DISPLAY.get(model, model) for model in imec_models]
        axes[0].legend(model_handles, model_labels, loc="upper left")

        bars = axes[1].bar(
            bottom_x,
            bottom_heights,
            bar_width,
            color=bottom_color,
            edgecolor="black",
            linewidth=0.6,
        )
        for rect, hatch in zip(bars, bottom_hatches):
            rect.set_hatch(hatch)
        axes[1].set_title("NVIDIA NIM Llama3-70B")
        axes[1].set_ylabel("")
        axes[1].set_xticks(bottom_positions)
        axes[1].set_xticklabels(bottom_labels, fontstyle="italic")
        axes[1].grid(axis="y", linestyle="--", alpha=0.8)
        if bottom_heights:
            bottom_max = max(bottom_heights)
            axes[1].set_ylim(0, bottom_max * 1.375)

        bs_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor=bottom_color, edgecolor="black", hatch=hatch_map.get(bs, "///"))
            for bs in bs_levels
        ]
        bs_labels = [f"bs={bs}" for bs in bs_levels]
        axes[1].legend(bs_handles, bs_labels, loc="upper right")

        pad = bar_width * 0.7
        if top_x:
            axes[0].set_xlim(min(top_x) - pad, max(top_x) + pad)
            axes[0].margins(x=0)
        if bottom_x:
            axes[1].set_xlim(min(bottom_x) - pad, max(bottom_x) + pad)
            axes[1].margins(x=0)

        fig.suptitle("Inference Validation (A100 80 GB)", y=0.975)
        fig.supylabel("Total runtime estimation error (%)")
        fig.supxlabel("(Input tokens/Output tokens) and TP degree")
        fig.subplots_adjust(top=0.88, bottom=0.12, left=0.09, right=0.98)

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "inf_errors_bar_combined_a100.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    imec_rows = None
    nvidia_rows = None
    # for device_name in ("A100", "H100"):
    for device_name in ("A100",):
        try:
            result = run(enable_plot=True, network_ignored=False, device=device_name)
            if device_name == "A100":
                imec_rows = result.get("rows")
        except FileNotFoundError:
            pass
    # for device_name in ("A100", "H100"):
    for device_name in ("A100",):
        try:
            result = run_nvidia(enable_plot=True, network_ignored=False, device=device_name)
            if device_name == "A100":
                nvidia_rows = result.get("rows")
        except FileNotFoundError:
            pass
    if imec_rows and nvidia_rows:
        out_dir = Path(PROJECT_ROOT) / "output" / "validation"
        combined = _plot_combined_a100_error_bars(imec_rows, nvidia_rows, out_dir)
        if combined is not None:
            print(f"Saved: {combined}")
