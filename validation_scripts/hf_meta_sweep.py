#!/usr/bin/env python3
"""
Wrapper to run multiple HF validation sweeps with lightweight meta caching.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation_scripts import huggingface_bench_validation as hf  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "output" / "validation" / "hf_sweep"
SAMPLE_DIR = OUTPUT_ROOT / "samples"
DEFAULT_POINTS = 125
DEFAULT_SEED = 1337
RUN_KEY_VERSION = 1
NUM_WORKERS = 85


def _file_signature(path: Path) -> Dict[str, Any]:
    try:
        stat = path.stat()
        return {"path": str(path.resolve()), "mtime": stat.st_mtime, "size": stat.st_size}
    except FileNotFoundError:
        return {"path": str(path), "mtime": None, "size": None}


def _hash_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _ensure_sample_csv(points: int, seed: int, source_csv: Path) -> Path:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample_csv = SAMPLE_DIR / f"hf_sample_points{points}_seed{seed}.csv"
    meta_path = sample_csv.with_suffix(".meta.json")
    source_sig = _file_signature(source_csv)
    expected_meta = {
        "version": 1,
        "points": points,
        "seed": seed,
        "source_csv": source_sig,
    }

    if sample_csv.exists() and meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if existing == expected_meta:
                return sample_csv
        except json.JSONDecodeError:
            pass

    df = pd.read_csv(source_csv)
    df = df[df["status"] == "Success"].copy()
    if df.empty:
        raise RuntimeError("No Success rows available after filtering OOM/Other statuses.")

    rng = random.Random(seed)
    indices = list(df.index)
    rng.shuffle(indices)
    keep = indices[: min(points, len(indices))]
    sample = df.loc[keep].copy()
    sample.to_csv(sample_csv, index=False)

    meta_path.write_text(json.dumps(expected_meta, indent=2, sort_keys=True), encoding="utf-8")
    return sample_csv


def _compute_metric_stats(values: pd.Series) -> Optional[Dict[str, float]]:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return None
    arr = series.to_numpy(dtype=float)
    abs_arr = np.abs(arr)
    return {
        "count": int(len(arr)),
        "mean_error": float(np.mean(arr)),
        "median_error": float(np.median(arr)),
        "std_error": float(np.std(arr)),
        "mean_abs_error": float(np.mean(abs_arr)),
        "p90_abs_error": float(np.percentile(abs_arr, 90)),
        "p95_abs_error": float(np.percentile(abs_arr, 95)),
    }


def _load_overall_stats(results_csv: Path) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(results_csv)
    stats: Dict[str, Dict[str, float]] = {}
    mem_res = _compute_metric_stats(df.get("mem_res_error_pct", pd.Series(dtype=float)))
    if mem_res:
        stats["mem_res"] = mem_res
    time_stats = _compute_metric_stats(df.get("time_error_pct", pd.Series(dtype=float)))
    if time_stats:
        stats["time"] = time_stats
    tok_stats = _compute_metric_stats(df.get("tok_s_error_pct", pd.Series(dtype=float)))
    if tok_stats:
        stats["tok_s"] = tok_stats
    return stats


def _apply_network_bw(cfg: Dict[str, Any], bandwidth_gb: float) -> None:
    net = cfg.get("network", {})
    for dim in net.get("dimensions", []):
        topo = dim.get("topology", {})
        topo["bandwidth"] = f"{int(bandwidth_gb)} GB"


def _apply_core_util(cfg: Dict[str, Any], util: float) -> None:
    cfg.setdefault("tech_param", {}).setdefault("core", {})["util"] = float(util)


def _apply_dram_util(cfg: Dict[str, Any], util: float) -> None:
    cfg.setdefault("tech_param", {}).setdefault("DRAM", {})["util"] = float(util)


def _write_hw_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def _run_key(
    sample_csv: Path,
    hw_config_path: Path,
    run_params: Dict[str, Any],
) -> str:
    payload = {
        "version": RUN_KEY_VERSION,
        "sample_csv": _file_signature(sample_csv),
        "hw_config": _file_signature(hw_config_path),
        "hf_script": _file_signature(PROJECT_ROOT / "validation_scripts" / "huggingface_bench_validation.py"),
        "run_params": run_params,
    }
    return _hash_payload(payload)


def _maybe_skip_run(output_dir: Path, run_key: str) -> bool:
    meta_path = output_dir / "meta.json"
    results_path = output_dir / "validation_results.csv"
    if not (meta_path.exists() and results_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return meta.get("run_key") == run_key and meta.get("status") == "complete"


def _write_meta(output_dir: Path, run_key: str, run_params: Dict[str, Any]) -> None:
    meta_path = output_dir / "meta.json"
    meta = {
        "run_key": run_key,
        "run_params": run_params,
        "status": "complete",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _run_variant(
    label: str,
    base_hw_cfg: Dict[str, Any],
    sample_csv: Path,
    *,
    output_dir: Path,
    hw_config_path: Path,
    core_util: Optional[float] = None,
    dram_util: Optional[float] = None,
    network_bw_gb: Optional[float] = None,
    points: int,
    seed: int,
    shuffle_seed: int,
    mode: str,
    num_workers: int,
    force: bool,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_hw_cfg)
    if core_util is not None:
        _apply_core_util(cfg, core_util)
    if dram_util is not None:
        _apply_dram_util(cfg, dram_util)
    if network_bw_gb is not None:
        _apply_network_bw(cfg, network_bw_gb)

    _write_hw_config(hw_config_path, cfg)

    run_params = {
        "label": label,
        "points": points,
        "seed": seed,
        "shuffle_seed": shuffle_seed,
        "mode": mode,
        "num_workers": num_workers,
        "core_util": core_util,
        "dram_util": dram_util,
        "network_bw_gb": network_bw_gb,
    }
    run_key = _run_key(sample_csv, hw_config_path, run_params)
    if not force and _maybe_skip_run(output_dir, run_key):
        print(f"[SKIP] {label}: cached result found")
        summary = _summarize_run(output_dir, label, run_params)
        summary["cached"] = True
        return summary

    output_dir.mkdir(parents=True, exist_ok=True)
    category_output_dir = output_dir / "categories"

    hf.SHUFFLE_SEED = shuffle_seed
    hf.run(
        csv_path=sample_csv,
        hw_config_path=hw_config_path,
        output_dir=output_dir,
        mode=mode,
        num_workers=num_workers,
        assume_bf16=hf.ASSUME_BF16,
        filter_pp_fix=hf.FILTER_PP_FIX,
        max_rows=None,
        enable_plots=hf.ENABLE_PLOTS,
        enable_global_plots=hf.ENABLE_GLOBAL_PLOTS,
        enable_category_plots=hf.ENABLE_CATEGORY_PLOTS,
        category_run=hf.CATEGORY_RUN,
        category_min_rows=hf.CATEGORY_MIN_ROWS,
        category_output_dir=category_output_dir,
        emit_logs=hf.EMIT_LOGS,
        chunk_size=hf.CHUNK_SIZE,
        enable_cache=True,
        cache_path=output_dir / "validation_cache.jsonl",
        rebuild_from_cache_only=False,
    )

    _write_meta(output_dir, run_key, run_params)
    summary = _summarize_run(output_dir, label, run_params)
    summary["cached"] = False
    return summary


def _load_base_hw_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _summarize_run(output_dir: Path, label: str, run_params: Dict[str, Any]) -> Dict[str, Any]:
    results_csv = output_dir / "validation_results.csv"
    stats = _load_overall_stats(results_csv)
    row_count = 0
    if results_csv.exists():
        row_count = int(pd.read_csv(results_csv).shape[0])
    summary = {
        "label": label,
        "output_dir": str(output_dir),
        "results_csv": str(results_csv),
        "rows": row_count,
        "mode": run_params.get("mode"),
        "num_workers": run_params.get("num_workers"),
        "points": run_params.get("points"),
        "seed": run_params.get("seed"),
        "core_util": run_params.get("core_util"),
        "dram_util": run_params.get("dram_util"),
        "network_bw_gb": run_params.get("network_bw_gb"),
    }
    for metric, metric_stats in stats.items():
        for key, val in metric_stats.items():
            summary[f"{metric}_{key}"] = val
    return summary


def _print_mem_res_table(rows: Sequence[Dict[str, Any]]) -> None:
    headers = [
        "label",
        "rows",
        "mem_res_count",
        "mem_res_mean_error",
        "mem_res_mean_abs_error",
        "mem_res_p90_abs_error",
        "mem_res_p95_abs_error",
    ]
    print("\n=== mem_res comparison ===")
    print(" | ".join(headers))
    print("-" * 95)
    for row in rows:
        values = [
            str(row.get("label", "")),
            str(row.get("rows", "")),
            str(row.get("mem_res_count", "")),
            f"{row.get('mem_res_mean_error', float('nan')):.2f}",
            f"{row.get('mem_res_mean_abs_error', float('nan')):.2f}",
            f"{row.get('mem_res_p90_abs_error', float('nan')):.2f}",
            f"{row.get('mem_res_p95_abs_error', float('nan')):.2f}",
        ]
        print(" | ".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="HF validation meta sweep wrapper.")
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS, help="Number of Success rows to sample.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed for the Success subset.")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used to shuffle the sampled rows inside the validation run.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore meta cache and rerun all sweeps.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="Worker count for validation.")
    parser.add_argument(
        "--mode",
        default=hf.MODE,
        choices=("mem_only", "perf_only", "both"),
        help="Validation mode to run.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variants to run (baseline,net_bw_inf,mem_util,core_util,core_mem_util).",
    )
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sample_csv = _ensure_sample_csv(args.points, args.seed, hf.CSV_PATH)

    base_hw_cfg = _load_base_hw_config(hf.HW_CONFIG_PATH)

    selected = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not selected or selected == ["all"]:
        selected = ["baseline", "net_bw_inf", "mem_util", "core_util", "core_mem_util"]

    def run_mem_util(util: float, label: str) -> Dict[str, Any]:
        return _run_variant(
            label=label,
            base_hw_cfg=base_hw_cfg,
            sample_csv=sample_csv,
            output_dir=OUTPUT_ROOT / label,
            hw_config_path=OUTPUT_ROOT / label / "hardware.yaml",
            core_util=None,
            dram_util=util,
            network_bw_gb=None,
            points=args.points,
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            mode=args.mode,
            num_workers=args.num_workers,
            force=args.force,
        )

    def run_core_util(util: float, label: str) -> Dict[str, Any]:
        return _run_variant(
            label=label,
            base_hw_cfg=base_hw_cfg,
            sample_csv=sample_csv,
            output_dir=OUTPUT_ROOT / label,
            hw_config_path=OUTPUT_ROOT / label / "hardware.yaml",
            core_util=util,
            dram_util=None,
            network_bw_gb=None,
            points=args.points,
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            mode=args.mode,
            num_workers=args.num_workers,
            force=args.force,
        )

    def run_core_mem_util(core_util: float, dram_util: float, label: str) -> Dict[str, Any]:
        return _run_variant(
            label=label,
            base_hw_cfg=base_hw_cfg,
            sample_csv=sample_csv,
            output_dir=OUTPUT_ROOT / label,
            hw_config_path=OUTPUT_ROOT / label / "hardware.yaml",
            core_util=core_util,
            dram_util=dram_util,
            network_bw_gb=None,
            points=args.points,
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            mode=args.mode,
            num_workers=args.num_workers,
            force=args.force,
        )

    summaries = []

    if "baseline" in selected:
        summaries.append(_run_variant(
            label="baseline",
            base_hw_cfg=base_hw_cfg,
            sample_csv=sample_csv,
            output_dir=OUTPUT_ROOT / "baseline",
            hw_config_path=OUTPUT_ROOT / "baseline" / "hardware.yaml",
            core_util=None,
            dram_util=None,
            network_bw_gb=None,
            points=args.points,
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            mode=args.mode,
            num_workers=args.num_workers,
            force=args.force,
        ))

    if "net_bw_inf" in selected:
        summaries.append(_run_variant(
            label="net_bw_inf",
            base_hw_cfg=base_hw_cfg,
            sample_csv=sample_csv,
            output_dir=OUTPUT_ROOT / "net_bw_inf",
            hw_config_path=OUTPUT_ROOT / "net_bw_inf" / "hardware.yaml",
            core_util=None,
            dram_util=None,
            network_bw_gb=9_999_999,
            points=args.points,
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            mode=args.mode,
            num_workers=args.num_workers,
            force=args.force,
        ))

    if "mem_util" in selected:
        summaries.append(run_mem_util(0.75, "mem_util_0p75"))

    if "core_util" in selected:
        summaries.append(run_core_util(0.75, "core_util_0p75"))

    if "core_mem_util" in selected:
        summaries.append(run_core_mem_util(0.75, 0.75, "core_mem_util_0p75"))

    if summaries:
        comp_csv = OUTPUT_ROOT / "comp.csv"
        pd.DataFrame(summaries).to_csv(comp_csv, index=False)
        print(f"\nSaved comparison CSV: {comp_csv}")
        _print_mem_res_table(summaries)


if __name__ == "__main__":
    main()
