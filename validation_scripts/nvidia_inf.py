import argparse
import math
import os
from pathlib import Path
import pandas as pd
from typing import List
import matplotlib.pyplot as plt

from .validation_helpers import (
    ValidationSpec,
    run_validation_suite,
    parse_inference_time,
)

HW_CONFIGS = {
    "A100": "a100_80GB.yaml",
    "H100": "H100_SXM5_80GB.yaml",
}

MODEL_CONFIGS = {
    "Llama 2-7B": "Llama2-7B_inf.yaml",
    "Llama 2-13B": "Llama2-13B_inf.yaml",
    "Llama 2-70B": "Llama2-70B_inf.yaml",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_PERF = os.path.join(PROJECT_ROOT, "run_perf.py")
HARDWARE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/hardware-config")
MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs/model-config")


def _load_data(csv_path: Path) -> pd.DataFrame:
    # Skip the leading "// filepath: ..." line by treating '/' as a comment char.
    df = pd.read_csv(csv_path, comment="/")
    # Normalize dtypes
    df["TP"] = df["TP"].astype(int)
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df["seconds"] = pd.to_numeric(df["seconds"], errors="coerce")
    return df.dropna(subset=["device", "model", "TP", "actual", "seconds"])


def _iter_tests(df: pd.DataFrame):
    for _, row in df.iterrows():
        yield (row["device"], row["model"], row["TP"])


def _run_test(
    device: str, model: str, tp: int, idx: int, network_ignored: bool = True
) -> float:
    print(f"Running test {idx + 1}: Device={device}, Model={model}, TP={tp}")
    specs: List[ValidationSpec] = []
    label = f"{device} {model} TP={tp}"

    model_overrides = {
        "model_param": {
            "global_batch_size": 1,
            "seq_len": 400,
            "decode_len": 200
        }
    }

    hw_overrides = {
        "parallelism": {
            "dp": 1,
            "tp": int(tp),
            "tp_sp": False,
            "cp": 1,
            "lp": 1,
            "mb": 1,
        }
    }
    if network_ignored:
        override_bw = {
            "network": {
                "dimensions": [
                    {
                        "id": "dim0",
                        "topology": {"bandwidth": "100000 GB", "latency": "0"},
                    }
                ]
            },
            "execution_backend": {
                "model": "analytical",
            },
        }
        hw_overrides.update(override_bw)

    specs.append(
        ValidationSpec(
            label=label,
            model_overrides=model_overrides,
            hardware_overrides=hw_overrides,
            order=idx,
        )
    )

    hw_config_path = os.path.join(HARDWARE_CONFIG_PATH, HW_CONFIGS.get(device))
    model_config_path = os.path.join(MODEL_CONFIG_PATH, MODEL_CONFIGS.get(model))

    validation_results = run_validation_suite(
        specs,
        base_model_config_path=model_config_path,
        base_hardware_config_path=hw_config_path,
        result_parser=parse_inference_time,
        run_perf_path=RUN_PERF,
    )

    result = validation_results[0]
    if result.success:
        inf_time_s = float(result.metrics.get("inference_time_s", float("nan")))
    else:
        inf_time_s = float("nan")
        err_detail = result.error or "DeepFlow run failed"
        print(f" ERROR: {err_detail}")
        if result.raw_output:
            print(result.raw_output.strip())

    return inf_time_s


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
        x_seconds, seconds_vals, bar_width, label="DeepFlow", color="#1f77b4"
    )
    bars_actual = ax.bar(
        x_actual, actual_vals, bar_width, label="Actual [1]", color="#0bbd37"
    )

    # Primary ticks: model names at cluster centers
    ax.set_xticks(model_centers)
    ax.set_xticklabels(models, fontsize=11)
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

    # Add citation below the graph
    citation = "[1] J. Kundu et al., “Performance Modeling and Workload Analysis of Distributed Large Language Model Training and Inference,” (2024)"
    plt.subplots_adjust(bottom=0.1)
    fig.text(0.5, 0.015, citation, ha="center", va="bottom", fontsize=8)

    outpath = outdir / f"inf_{device}.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


def run(enable_plot: bool = True, network_ignored: bool = True):
    pct_errors = []
    data = _load_data(Path(__file__).parent / "nvidia_inf.csv")

    for idx, (device, model, tp) in enumerate(_iter_tests(data)):
        seconds = _run_test(device, model, tp, idx, network_ignored)

        actual = data.loc[
            (data["device"] == device) & (data["model"] == model) & (data["TP"] == tp),
            "actual",
        ].values[0]

        pct_error = (
            abs(seconds - actual) / actual * 100.0 if actual != 0 else float("nan")
        )
        pct_errors.append(pct_error)

        data.loc[
            (data["device"] == device) & (data["model"] == model) & (data["TP"] == tp),
            "seconds",
        ] = seconds

        data.loc[
            (data["device"] == device) & (data["model"] == model) & (data["TP"] == tp),
            "pct_error",
        ] = pct_error

        block_lines = [
            f"\n=== Result (device={device}, model={model}, TP={tp}) ===",
        ]
        if not math.isnan(seconds):
            block_lines.append(f"  DeepFlow Inference Time: {seconds:.2f}s")
            block_lines.append(f"  Actual Inference Time:   {actual:.2f}s")
            block_lines.append(f"  Percent Error:          {pct_error:.2f}%")
        else:
            block_lines.append("  DeepFlow run failed.")
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
        if not outputs:
            print("No plots generated (no matching device rows).")
        else:
            for p in outputs:
                print(f"Saved: {p}")
    avg_abs_error = sum(e for e in pct_errors if not math.isnan(e)) / len(
        [e for e in pct_errors if not math.isnan(e)]
    )
    # print("Average absolute percent error across all tests: {:.2f}%".format(avg_abs_error))
    return {"avg_abs_error": avg_abs_error}


if __name__ == "__main__":
    run(enable_plot=True, network_ignored=True)
