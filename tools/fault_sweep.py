#!/usr/bin/env python3
"""
Fault sensitivity sweep for RAPID-LLM parallelism configurations.

Generates a representative set of parallelism tuples that map to a fixed GPU
budget, evaluates baseline runtimes, and then injects random soft faults across
network dimensions to gauge runtime variability.
"""

import argparse
import atexit
import datetime
import copy
import json
import math
import os
import random
import signal
import shlex
import sys
import threading
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

try:  # Optional styling dependency
    import seaborn as sns  # type: ignore
except ImportError:  # pragma: no cover
    sns = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

from parallelism_sweep import (  # noqa: E402
    MODEL_CONFIG_PATH,
    ASTRA_CACHE_MODE,
    determine_model_mode,
    evaluate_parallelism,
    make_temp_hw_config,
    read_yaml,
    set_astrasim_cache_mode,
    tp_cp_product_is_power_of_two_square,
)
import memory_estimation  # noqa: E402


class FlowSeq(list):
    """List subclass that forces YAML flow-style serialization."""


def _represent_flow_seq(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


yaml.SafeDumper.add_representer(FlowSeq, _represent_flow_seq)


def _quantize_weight(value: float, decimals: int = 2) -> float:
    return round(float(value), decimals)

# -----------------------------------------------------------------------------
# Fault sweep configuration
# -----------------------------------------------------------------------------

HARDWARE_CONFIG_PATH = "configs/hardware-config/a100_80GB.yaml"
MODEL_CONFIG_PATH = "configs/model-config/Llama3.1-70B.yaml"

TARGET_NUM_GPUS = 64
SAMPLE_COUNT = 50
FAULT_ITER = 125
FAULT_WORKERS = 75
FAULT_MAG = [0.5, 0.1]  # May also be a (mean, std) tuple
NUM_FAULTS = [1]

MIN_ALLOWED_TP = 4
MAX_ALLOWED_TP = 16
MIN_ALLOWED_CP = 1
MAX_ALLOWED_CP = 1
MIN_ALLOWED_DP = 1
MAX_ALLOWED_DP = 128
MIN_ALLOWED_LP = 1
MAX_ALLOWED_LP = 16
ALLOWED_FAULT_DIMS: Optional[List[int]] = None

PLOT_OUTPUT_PATH = "tools/fault_sweep.png"
REPORT_OUTPUT_PATH = "tools/fault_sweep.tsv"
FAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "astra_cache_faults")
RESULTS_JSON_PATH = Path("tools/results.json")

RANDOM_SEED = 1337
RESULTS_WRITE_LOCK = threading.Lock()
_ACTIVE_EXECUTOR: Optional[ProcessPoolExecutor] = None
_OLD_SIGINT_HANDLER = None
_OLD_SIGTERM_HANDLER = None

# When True, discard configurations whose tp*cp product is not a square power of two.
ENFORCE_SQUARE_TP_CP = False

# Debug dumping controls
DEBUG_MODE = False
DEBUG_OUTPUT_BASE = Path("dbg_hw_conf")
DEBUG_RUN_DIR: Optional[Path] = None
DEBUG_SAVED_PATHS: List[Path] = []
_DEBUG_FILE_INDEX = 0

# Failure dump controls
FAILURE_OUTPUT_BASE = DEBUG_OUTPUT_BASE / "failures"
FAILURE_DUMP_DIR: Optional[Path] = None
FAILURE_SAVED_PATHS: List[Path] = []
_FAILURE_FILE_INDEX = 0


def _parse_int_list(value: Optional[str]) -> List[int]:
    if not value:
        return []
    parts = [item.strip() for item in value.split(',')]
    result: List[int] = []
    for part in parts:
        if not part:
            continue
        result.append(int(part))
    return result


def _parse_float_tuple(value: Optional[str]) -> Tuple[float, float]:
    if not value:
        return (0.0, 0.0)
    parts = [item.strip() for item in value.split(',') if item.strip()]
    if len(parts) == 1:
        val = float(parts[0])
        return (val, 0.0)
    if len(parts) >= 2:
        return (float(parts[0]), float(parts[1]))
    return (0.0, 0.0)

# Network topologies in AstraSim that understand per-link fault annotations.
# The neighbor builders below encode each topology's link structure so that
# fault injection only targets physically valid edges.
TOPOLOGIES_WITH_FAULTY_LINK_SUPPORT = {
    "ring",
    "torus2d",
    "mesh",
    "mesh2d",
    "kingmesh2d",
}

_TOPOLOGIES_EXPAND_TO_2D = {"mesh2d", "torus2d"}


def _normalize_topology_name(raw: object) -> str:
    if raw is None:
        return ""
    return str(raw).strip().lower().replace("-", "").replace("_", "")


def _infer_non_recursive_from(dimensions: Sequence[object]) -> int:
    if not dimensions:
        return 0
    first = dimensions[0]
    topo = ""
    if isinstance(first, dict):
        topo_dict = first.get("topology")
        if isinstance(topo_dict, dict):
            topo = _normalize_topology_name(topo_dict.get("type", ""))
    return 2 if topo in _TOPOLOGIES_EXPAND_TO_2D else 0


def _expanded_start_indices(dimensions: Sequence[object]) -> List[int]:
    starts: List[int] = []
    cursor = 0
    for dim in dimensions:
        starts.append(cursor)
        topo = ""
        if isinstance(dim, dict):
            topo_dict = dim.get("topology")
            if isinstance(topo_dict, dict):
                topo = _normalize_topology_name(topo_dict.get("type", ""))
        cursor += 2 if topo in _TOPOLOGIES_EXPAND_TO_2D else 1
    return starts


def _parallelism_product(names: Iterable[object], parallelism: Dict[str, object]) -> int:
    product = 1
    for entry in names:
        axis = str(entry).strip().lower()
        if not axis:
            continue
        value = parallelism.get(axis, 1)
        try:
            factor = int(value)
        except (TypeError, ValueError):
            factor = 1
        product *= max(1, factor)
    return max(1, product)


def _resolve_dimension_size(dim: dict, parallelism: Dict[str, object]) -> int:
    size_raw = dim.get("size", "auto")
    if isinstance(size_raw, str):
        normalized = size_raw.strip().lower()
        if normalized == "auto":
            return _parallelism_product(dim.get("parallelisms", []) or [], parallelism)
        if normalized.startswith("(") and normalized.endswith(")"):
            inner = normalized[1:-1].strip()
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            if len(parts) == 2:
                try:
                    return max(1, int(parts[0]) * int(parts[1]))
                except (TypeError, ValueError):
                    return 1
        try:
            return max(1, int(normalized))
        except (TypeError, ValueError):
            return 1
    if isinstance(size_raw, (list, tuple)):
        if len(size_raw) == 2:
            try:
                return max(1, int(size_raw[0]) * int(size_raw[1]))
            except (TypeError, ValueError):
                return 1
        if len(size_raw) == 1:
            try:
                return max(1, int(size_raw[0]))
            except (TypeError, ValueError):
                return 1
    try:
        return max(1, int(size_raw))
    except (TypeError, ValueError):
        return 1


def _dimension_sizes(dimensions: Sequence[object], parallelism: Dict[str, object]) -> List[int]:
    sizes: List[int] = []
    for dim in dimensions:
        if not isinstance(dim, dict):
            sizes.append(1)
            continue
        sizes.append(_resolve_dimension_size(dim, parallelism))
    return sizes


def _pair_set(pairs: Iterable[Tuple[int, int]]) -> set[Tuple[int, int]]:
    normalized = set()
    for a, b in pairs:
        lo, hi = (a, b) if a <= b else (b, a)
        normalized.add((int(lo), int(hi)))
    return normalized


def _collect_possible_fault_links(hw_dict: Dict[str, object]) -> set[Tuple[int, int]]:
    network = hw_dict.get("network")
    if not isinstance(network, dict):
        return set()
    dimensions = network.get("dimensions")
    if not isinstance(dimensions, list):
        return set()
    parallelism = hw_dict.get("parallelism")
    if not isinstance(parallelism, dict):
        parallelism = {}
    dim_sizes = _dimension_sizes(dimensions, parallelism)
    expanded_starts = _expanded_start_indices(dimensions)
    non_recursive_from = _infer_non_recursive_from(dimensions)
    possible: set[Tuple[int, int]] = set()
    for dim_index, dim in enumerate(dimensions):
        if not isinstance(dim, dict):
            continue
        topology = dim.get("topology")
        topo_type = ""
        if isinstance(topology, dict):
            topo_type = _normalize_topology_name(topology.get("type", ""))
        if topo_type not in TOPOLOGIES_WITH_FAULTY_LINK_SUPPORT:
            continue
        dim_size = dim_sizes[dim_index] if dim_index < len(dim_sizes) else 1
        if dim_size < 2:
            continue
        neighbor_map = _neighbors_for_topology(topo_type, dim_size)
        stride = 1
        for idx in range(dim_index):
            if idx < len(dim_sizes):
                stride *= max(1, int(dim_sizes[idx]))
        stride = max(1, stride)
        expanded_start = expanded_starts[dim_index] if dim_index < len(expanded_starts) else dim_index
        non_recursive_dim = expanded_start >= non_recursive_from
        if non_recursive_dim or stride <= 1:
            offsets = [0]
        else:
            offsets = list(range(stride))
        for offset in offsets:
            for src, neighbors in enumerate(neighbor_map):
                for dst in neighbors:
                    src_id = offset + int(src) * stride
                    dst_id = offset + int(dst) * stride
                    possible.add((min(src_id, dst_id), max(src_id, dst_id)))
    return possible


def _run_fault_injection_self_test() -> None:
    mesh2d_ring_hw = {
        "parallelism": {"tp": 2, "cp": 2, "dp": 2, "lp": 1},
        "network": {
            "dimensions": [
                {
                    "id": "dim0",
                    "topology": {"type": "Mesh2D"},
                    "size": "auto",
                    "parallelisms": ["tp", "cp"],
                },
                {
                    "id": "dim1",
                    "topology": {"type": "Ring"},
                    "size": "auto",
                    "parallelisms": ["dp"],
                },
            ]
        },
    }
    mesh2d_ring_expected = _pair_set(
        [
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4),
        ]
    )

    ring_ring_hw = {
        "parallelism": {"tp": 4, "cp": 1, "dp": 4, "lp": 1},
        "network": {
            "dimensions": [
                {
                    "id": "dim0",
                    "topology": {"type": "Ring"},
                    "size": "auto",
                    "parallelisms": ["tp"],
                },
                {
                    "id": "dim1",
                    "topology": {"type": "Ring"},
                    "size": "auto",
                    "parallelisms": ["dp"],
                },
            ]
        },
    }
    ring_ring_expected = _pair_set(
        [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (8, 9), (9, 10), (10, 11), (11, 8),
            (12, 13), (13, 14), (14, 15), (15, 12),
            (0, 4), (4, 8), (8, 12), (12, 0),
        ]
    )

    cases = [
        ("mesh2d_ring_2x2_plus_ring2", mesh2d_ring_hw, mesh2d_ring_expected),
        ("ring_ring_4x4", ring_ring_hw, ring_ring_expected),
    ]

    for label, hw_dict, expected in cases:
        observed = _collect_possible_fault_links(hw_dict)
        if not observed:
            raise RuntimeError(f"[self-test] {label}: no candidate fault links generated.")
        extras = observed - expected
        if extras:
            extras_sorted = ", ".join(f"{a}-{b}" for a, b in sorted(extras))
            raise RuntimeError(
                f"[self-test] {label}: observed links outside expected set: {extras_sorted}"
            )

    banner = "=" * 72
    print(f"\n{banner}\nCHECKMARK: FAULT INJECTION SELF-TEST PASSED\n{banner}\n")

def _build_ring_neighbors(node_count: int) -> List[List[int]]:
    if node_count <= 0:
        return []
    if node_count == 1:
        return [[]]
    neighbors: List[List[int]] = []
    for idx in range(node_count):
        left = (idx - 1) % node_count
        right = (idx + 1) % node_count
        entries: List[int] = []
        for candidate in (left, right):
            if candidate != idx and candidate not in entries:
                entries.append(candidate)
        neighbors.append(entries)
    return neighbors


def _build_mesh_neighbors(node_count: int) -> List[List[int]]:
    if node_count <= 0:
        return []
    neighbors: List[List[int]] = []
    for idx in range(node_count):
        entries: List[int] = []
        if idx - 1 >= 0:
            entries.append(idx - 1)
        if idx + 1 < node_count:
            entries.append(idx + 1)
        neighbors.append(entries)
    return neighbors


def _build_mesh2d_neighbors(node_count: int) -> List[List[int]]:
    if node_count <= 0:
        return []
    dim = int(math.isqrt(node_count))
    if dim * dim != node_count:
        raise ValueError(f"Mesh2D topology requires perfect square node count, got {node_count}.")
    neighbors: List[List[int]] = [[] for _ in range(node_count)]
    for idx in range(node_count):
        row, col = divmod(idx, dim)
        entries: List[int] = []
        if col - 1 >= 0:
            entries.append(row * dim + (col - 1))
        if col + 1 < dim:
            entries.append(row * dim + (col + 1))
        if row - 1 >= 0:
            entries.append((row - 1) * dim + col)
        if row + 1 < dim:
            entries.append((row + 1) * dim + col)
        neighbors[idx] = entries
    return neighbors


def _build_torus2d_neighbors(node_count: int) -> List[List[int]]:
    if node_count <= 0:
        return []
    dim = int(math.isqrt(node_count))
    if dim * dim != node_count:
        raise ValueError(f"Torus2D topology requires perfect square node count, got {node_count}.")
    neighbors: List[List[int]] = [[] for _ in range(node_count)]
    if dim == 1:
        return neighbors
    for idx in range(node_count):
        row, col = divmod(idx, dim)
        left = row * dim + ((col - 1) % dim)
        right = row * dim + ((col + 1) % dim)
        up = ((row - 1) % dim) * dim + col
        down = ((row + 1) % dim) * dim + col
        entries = []
        for candidate in (left, right, up, down):
            if candidate != idx and candidate not in entries:
                entries.append(candidate)
        neighbors[idx] = entries
    return neighbors


def _build_kingmesh2d_neighbors(node_count: int) -> List[List[int]]:
    if node_count <= 0:
        return []
    dim = int(math.isqrt(node_count))
    if dim * dim != node_count:
        raise ValueError(f"KingMesh2D topology requires perfect square node count, got {node_count}.")
    neighbors: List[List[int]] = [[] for _ in range(node_count)]
    for idx in range(node_count):
        row, col = divmod(idx, dim)
        entries: List[int] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nr = row + dy
                nc = col + dx
                if 0 <= nr < dim and 0 <= nc < dim:
                    candidate = nr * dim + nc
                    if candidate != idx:
                        entries.append(candidate)
        neighbors[idx] = entries
    return neighbors


_NEIGHBOR_BUILDERS = {
    "ring": _build_ring_neighbors,
    "mesh": _build_mesh_neighbors,
    "mesh2d": _build_mesh2d_neighbors,
    "torus2d": _build_torus2d_neighbors,
    "kingmesh2d": _build_kingmesh2d_neighbors,
}


_NEIGHBOR_CACHE: Dict[Tuple[str, int], Tuple[Tuple[int, ...], ...]] = {}


def _neighbors_for_topology(topo_type: str, node_count: int) -> Tuple[Tuple[int, ...], ...]:
    key = (topo_type, int(node_count))
    cached = _NEIGHBOR_CACHE.get(key)
    if cached is not None:
        return cached
    builder = _NEIGHBOR_BUILDERS.get(topo_type)
    if builder is None:
        raise ValueError(f"No neighbor builder registered for topology '{topo_type}'.")
    neighbor_lists = builder(int(node_count))
    normalized = tuple(tuple(neigh) for neigh in neighbor_lists)
    _NEIGHBOR_CACHE[key] = normalized
    return normalized


def _candidate_fault_dimensions(settings: Dict[str, int]) -> List[Tuple[int, int]]:
    candidates: List[Tuple[int, int]] = []
    dim0_nodes = int(settings.get("tp", 1)) * int(settings.get("cp", 1))
    if dim0_nodes >= 2 and (ALLOWED_FAULT_DIMS is None or 0 in ALLOWED_FAULT_DIMS):
        candidates.append((0, dim0_nodes))
    dim1_nodes = int(settings.get("lp", 1)) * int(settings.get("dp", 1))
    if dim1_nodes >= 2 and (ALLOWED_FAULT_DIMS is None or 1 in ALLOWED_FAULT_DIMS):
        candidates.append((1, dim1_nodes))
    return candidates


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAPID-LLM fault sensitivity sweep")
    parser.add_argument("--target-num-gpus", type=int, default=TARGET_NUM_GPUS)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    parser.add_argument("--fault-iter", type=int, default=FAULT_ITER)
    parser.add_argument("--fault-workers", type=int, default=FAULT_WORKERS)
    parser.add_argument("--fault-mag", type=str, default=f"{FAULT_MAG[0]},{FAULT_MAG[1] if isinstance(FAULT_MAG, (tuple, list)) else 0.0}", help="Mean[,std] for fault magnitude")
    parser.add_argument("--num-faults", type=str, default=",".join(str(n) for n in NUM_FAULTS), help="Comma-separated list of number of faults per evaluation")
    parser.add_argument("--min-tp", type=int, default=MIN_ALLOWED_TP)
    parser.add_argument("--max-tp", type=int, default=MAX_ALLOWED_TP)
    parser.add_argument("--min-cp", type=int, default=MIN_ALLOWED_CP)
    parser.add_argument("--max-cp", type=int, default=MAX_ALLOWED_CP)
    parser.add_argument("--min-dp", type=int, default=MIN_ALLOWED_DP)
    parser.add_argument("--max-dp", type=int, default=MAX_ALLOWED_DP)
    parser.add_argument("--min-lp", type=int, default=MIN_ALLOWED_LP)
    parser.add_argument("--max-lp", type=int, default=MAX_ALLOWED_LP)
    parser.add_argument("--allowed-fault-dims", type=str, default=None, help="Comma-separated list of network dimension indices eligible for faults")
    parser.add_argument("--plot-output", type=str, default=PLOT_OUTPUT_PATH)
    parser.add_argument("--report-output", type=str, default=REPORT_OUTPUT_PATH)
    parser.add_argument("--results-json", type=str, default=str(RESULTS_JSON_PATH))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--enforce-square-tp-cp",
        action="store_true",
        help="Require tp*cp to be a square power of two when sampling parallelism tuples.",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable debug dumps of hardware configs and exit after generation.",
    )
    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# Parallelism sampling helpers
# -----------------------------------------------------------------------------

def _divisors(n: int) -> List[int]:
    divs = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
    return sorted(divs)


def _enumerate_parallelism_vectors(num_gpus: int) -> List[Tuple[int, int, int, int]]:
    vectors: List[Tuple[int, int, int, int]] = []
    for tp in _divisors(num_gpus):
        rem_tp = num_gpus // tp
        for cp in _divisors(rem_tp):
            rem_cp = rem_tp // cp
            for dp in _divisors(rem_cp):
                lp = rem_cp // dp
                vectors.append((tp, cp, dp, lp))
    return vectors


def _log_distance(a: Sequence[int], b: Sequence[int]) -> float:
    total = 0.0
    for x, y in zip(a, b):
        total += abs(math.log2(max(x, 1)) - math.log2(max(y, 1)))
    return total


def _select_representative_vectors(candidates: List[Tuple[int, int, int, int]], count: int) -> List[Tuple[int, int, int, int]]:
    if not candidates:
        return []
    selected: List[Tuple[int, int, int, int]] = []

    # Seed with extreme configurations
    extremes = [
        max(candidates, key=lambda v: v[0]),
        max(candidates, key=lambda v: v[1]),
        max(candidates, key=lambda v: v[2]),
        max(candidates, key=lambda v: v[3]),
    ]
    for cand in extremes:
        if cand not in selected:
            selected.append(cand)
        if len(selected) >= count:
            return selected[:count]

    # Farthest-point sampling to promote coverage
    remaining = [v for v in candidates if v not in selected]
    while remaining and len(selected) < count:
        best_vec = None
        best_score = -1.0
        for vec in remaining:
            min_dist = min(_log_distance(vec, sel) for sel in selected)
            if min_dist > best_score:
                best_score = min_dist
                best_vec = vec
        if best_vec is None:
            break
        selected.append(best_vec)
        remaining.remove(best_vec)

    # If still short, append random leftovers
    while remaining and len(selected) < count:
        candidate = remaining.pop(random.randrange(len(remaining)))
        selected.append(candidate)

    return selected[:count]


def _filtered_parallelism_vectors(num_gpus: int) -> List[Tuple[int, int, int, int]]:
    tuples = _enumerate_parallelism_vectors(num_gpus)
    filtered: List[Tuple[int, int, int, int]] = []
    for tp, cp, dp, lp in tuples:
        if not (MIN_ALLOWED_TP <= tp <= MAX_ALLOWED_TP):
            continue
        if not (MIN_ALLOWED_CP <= cp <= MAX_ALLOWED_CP):
            continue
        if not (MIN_ALLOWED_DP <= dp <= MAX_ALLOWED_DP):
            continue
        if not (MIN_ALLOWED_LP <= lp <= MAX_ALLOWED_LP):
            continue
        if ENFORCE_SQUARE_TP_CP and not tp_cp_product_is_power_of_two_square(tp, cp):
            continue
        filtered.append((tp, cp, dp, lp))
    return filtered


def _make_parallelism_settings(tp: int, cp: int, dp: int, lp: int) -> Dict[str, int]:
    return {
        "tp": tp,
        "cp": cp,
        "dp": dp,
        "lp": lp,
        "mb": 1 if lp == 1 else lp,
        "tp_sp": True,
    }


def generate_parallelism_samples(
    num_gpus: int,
    sample_count: int,
    base_hw_dict: Dict[str, object],
    model_config_obj: object,
    mode: str,
    workers: Optional[int] = None,
) -> List[Dict[str, int]]:
    vectors = _filtered_parallelism_vectors(num_gpus)
    if not vectors:
        return []
    # Keep ordering stable but cover space by using representative sampling.
    prioritized = deque(_select_representative_vectors(vectors, len(vectors)))
    selected: List[Dict[str, int]] = []

    worker_count = max(1, int(workers) if workers is not None else 1)
    worker_count = min(worker_count, max(1, os.cpu_count() or 1))

    print(f"Choosing out of {len(vectors)} candidate parallelism configurations...")

    if worker_count <= 1:
        while prioritized and len(selected) < sample_count:
            tp, cp, dp, lp = prioritized.popleft()
            settings = _make_parallelism_settings(tp, cp, dp, lp)
            try:
                # Use fast memory check instead of full evaluation
                mem_check = check_memory_capacity(base_hw_dict, model_config_obj, mode, settings)
            except Exception as exc:  # pragma: no cover - defensive
                print(
                    f"Warning: failed to check memory for parallelism {settings} during sampling: {exc}",
                    file=sys.stderr,
                )
                continue
            if bool(mem_check.get("memory_exceeded", False)):
                continue
            selected.append(settings)
    else:
        backlog_target = worker_count * 3
        in_flight: Dict[object, Dict[str, int]] = {}
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_parallelism_worker_init,
            initargs=(base_hw_dict, MODEL_CONFIG_PATH, mode),
        ) as executor:
            while (prioritized or in_flight) and len(selected) < sample_count:
                while (
                    prioritized
                    and len(in_flight) < backlog_target
                    and len(selected) + len(in_flight) < sample_count + backlog_target
                ):
                    tp, cp, dp, lp = prioritized.popleft()
                    settings = _make_parallelism_settings(tp, cp, dp, lp)
                    fut = executor.submit(_sampling_worker, settings)
                    in_flight[fut] = settings
                if not in_flight:
                    break
                completed = next(as_completed(list(in_flight.keys())))
                settings = in_flight.pop(completed, None)
                try:
                    result = completed.result()
                except Exception as exc:  # pragma: no cover - defensive
                    print(
                        f"Warning: failed to evaluate parallelism {settings} during sampling: {exc}",
                        file=sys.stderr,
                    )
                    continue
                if result.get("error"):
                    print(
                        f"Warning: failed to evaluate parallelism {settings} during sampling: {result.get('error')}",
                        file=sys.stderr,
                    )
                    continue
                if bool(result.get("memory_exceeded", False)):
                    continue
                if settings is not None:
                    selected.append(settings)
                    print(f"Selected parallelism configuration: {settings}")

    if len(selected) < sample_count and not prioritized:
        print(
            f"Warning: only {len(selected)} parallelism configuration(s) fit in memory "
            f"out of requested {sample_count}.",
            file=sys.stderr,
        )
    return selected


# -----------------------------------------------------------------------------
# Fault injection helpers
# -----------------------------------------------------------------------------

def sample_fault_magnitude() -> float:
    if isinstance(FAULT_MAG, (tuple, list)) and len(FAULT_MAG) == 2:
        mean, std = FAULT_MAG
        value = random.gauss(mean, std)
    else:
        value = float(FAULT_MAG)
    value = max(0.0, min(0.95, value))
    value = _quantize_weight(value)
    return max(0.1, value) # min of 10% bw, max of 95% bw


def make_fault_mutator(fault_specs: Sequence[Tuple[int, float, int]]):
    fault_specs = [
        (int(dim_idx), float(weight), int(node_count))
        for dim_idx, weight, node_count in fault_specs
    ]

    def _mutator(hw_dict: Dict[str, object]) -> None:
        network = hw_dict.get("network")
        if not isinstance(network, dict):
            return
        dimensions = network.get("dimensions")
        if not isinstance(dimensions, list):
            return
        parallelism = hw_dict.get("parallelism")
        if not isinstance(parallelism, dict):
            parallelism = {}
        dim_sizes = _dimension_sizes(dimensions, parallelism)
        expanded_starts = _expanded_start_indices(dimensions)
        non_recursive_from = _infer_non_recursive_from(dimensions)
        fault_entries: List[FlowSeq] = []
        for dimension_index, weight, node_count in fault_specs:
            if not (0 <= dimension_index < len(dimensions)):
                continue
            dim = dimensions[dimension_index]
            if not isinstance(dim, dict):
                continue
            topology = dim.get("topology")
            topo_type = ""
            if isinstance(topology, dict):
                topo_raw = topology.get("type", "")
                if isinstance(topo_raw, str):
                    topo_type = _normalize_topology_name(topo_raw)
            supports_faulty_links = topo_type in TOPOLOGIES_WITH_FAULTY_LINK_SUPPORT
            if not supports_faulty_links:
                label = dim.get("label", f"dim{dimension_index}")
                raise ValueError(
                    f"Topology '{topo_type or 'unknown'}' on network dimension '{label}' "
                    "does not support faulty_links injection."
                )
            dim_size = dim_sizes[dimension_index] if dimension_index < len(dim_sizes) else node_count
            if dim_size < 2:
                continue
            try:
                neighbor_map = _neighbors_for_topology(topo_type, dim_size)
            except ValueError as exc:
                raise ValueError(
                    f"Unable to determine neighbors for topology '{topo_type or 'unknown'}' "
                    f"with node_count={dim_size}: {exc}"
                ) from exc
            valid_sources = [idx for idx, neigh in enumerate(neighbor_map) if neigh]
            if not valid_sources:
                continue
            src = random.choice(valid_sources)
            dst_candidates = neighbor_map[src]
            dst = random.choice(dst_candidates)
            stride = 1
            for idx in range(dimension_index):
                if idx < len(dim_sizes):
                    stride *= max(1, int(dim_sizes[idx]))
            stride = max(1, stride)
            expanded_start = expanded_starts[dimension_index] if dimension_index < len(expanded_starts) else dimension_index
            non_recursive_dim = expanded_start >= non_recursive_from
            lower_offset = 0
            if not non_recursive_dim and stride > 1:
                lower_offset = random.randrange(stride)
            src = lower_offset + int(src) * stride
            dst = lower_offset + int(dst) * stride
            weight_clamped = float(max(0.0, min(1.0, weight)))
            weight_clamped = _quantize_weight(weight_clamped)
            dim.pop("faulty_links", None)
            fault_entries.append(FlowSeq([int(src), int(dst), weight_clamped]))

        if fault_entries:
            network["faulty_links"] = FlowSeq(fault_entries)
        else:
            network.pop("faulty_links", None)

    return _mutator


# -----------------------------------------------------------------------------
# Fast memory checking using new memory estimation API
# -----------------------------------------------------------------------------

def check_memory_capacity(base_hw_dict, model_config_obj, mode, parallel_settings, hw_mutator=None):
    """
    Fast memory capacity check using the new memory estimation API.

    Returns a dictionary with:
        - memory_exceeded: bool indicating if capacity is violated
        - memory_violation_gb: float indicating how much over capacity (if exceeded)
        - summary: the full memory estimation summary dict

    This is much faster than running full timing calculations.
    """
    hw_config, _ = make_temp_hw_config(base_hw_dict, parallel_settings, hw_mutator=hw_mutator)

    try:
        # Use the dispatcher function which automatically determines inference vs training
        summary = memory_estimation.estimate_memory(
            hw_config,
            model_config_obj,
            mode=mode,
            output_dir=None,  # Don't write outputs for fast checks
        )

        # Check if this is training (has capacity_exceeded key) or inference (has max_peak_gb key)
        if 'capacity_exceeded' in summary:
            # Training workload
            capacity_exceeded = bool(summary.get('capacity_exceeded', False))
            violation = float(summary.get('capacity_violation_gb', 0.0) or 0.0)
            return {
                "memory_exceeded": capacity_exceeded,
                "memory_violation_gb": violation,
                "summary": summary,
            }
        else:
            # Inference workload - check if max_peak exceeds capacity
            max_peak = summary.get('max_peak_gb', 0.0)
            capacity = summary.get('capacity_gb')
            if capacity is not None and max_peak > capacity:
                violation = max_peak - capacity
                return {
                    "memory_exceeded": True,
                    "memory_violation_gb": violation,
                    "summary": summary,
                }
            return {
                "memory_exceeded": False,
                "memory_violation_gb": 0.0,
                "summary": summary,
            }
    except Exception as exc:
        # If memory check fails, return conservatively (assume no violation)
        # The full evaluate_parallelism will catch any real errors
        print(f"Warning: fast memory check failed: {exc}", file=sys.stderr)
        return {
            "memory_exceeded": False,
            "memory_violation_gb": 0.0,
            "summary": {},
        }


# -----------------------------------------------------------------------------
# Multiprocessing helpers
# -----------------------------------------------------------------------------

_WORKER_HW_DICT = None
_WORKER_MODEL_CONFIG = None
_WORKER_MODE = None


def _parallelism_worker_init(base_hw_dict, model_config_path, mode):
    global _WORKER_HW_DICT, _WORKER_MODEL_CONFIG, _WORKER_MODE
    set_astrasim_cache_mode(ASTRA_CACHE_MODE)
    _WORKER_HW_DICT = base_hw_dict
    _WORKER_MODEL_CONFIG = config.parse_config(model_config_path, config_type=mode)
    _WORKER_MODE = mode


def _execute_parallelism_task(base_hw_dict, model_config_obj, mode, task: Dict[str, object]) -> Dict[str, object]:
    kind = task.get("kind")
    settings = dict(task.get("settings", {}))
    key = task.get("key")
    settings_key = tuple(task.get("settings_key", ()))
    task_id = task.get("task_id")
    fault_iter = task.get("fault_iter")
    faults: List[Tuple[int, float, int]] = []
    num_faults = 0
    if kind == "fault":
        raw_faults = task.get("faults", [])
        try:
            faults = [tuple(spec) for spec in raw_faults]
        except Exception:
            faults = []
        num_faults = int(task.get("num_faults", len(faults)))
    try:
        mutator = None
        if kind == "fault":
            mutator = make_fault_mutator(faults)
        metrics = evaluate_parallelism(
            base_hw_dict,
            model_config_obj,
            mode,
            settings,
            hw_mutator=mutator,
        )
        mem_violation = metrics.get("memory_violation_gb", 0.0) or 0.0
        result: Dict[str, object] = {
            "status": "ok",
            "kind": kind,
            "key": key,
            "settings": settings,
            "settings_key": settings_key,
            "runtime": float(metrics["runtime"]),
            "hw_yaml": metrics.get("hw_yaml"),
            "task_id": task_id,
            "memory_exceeded": bool(metrics.get("memory_exceeded", False)),
            "memory_violation_gb": float(mem_violation),
        }
        if fault_iter is not None:
            result["fault_iter"] = int(fault_iter)
        if kind == "fault":
            result.update(
                {
                    "faults": faults,
                    "num_faults": num_faults,
                }
            )
        else:
            result.update({"faults": [], "num_faults": 0})
        return result
    except Exception as exc:
        error_result = {
            "status": "error",
            "kind": kind,
            "key": key,
            "settings": settings,
            "settings_key": settings_key,
            "task_id": task_id,
            "error": str(exc),
        }
        if fault_iter is not None:
            error_result["fault_iter"] = int(fault_iter)
        if kind == "fault":
            error_result.update({"faults": faults, "num_faults": num_faults})
        else:
            error_result.update({"faults": [], "num_faults": 0})
        return error_result


def _parallelism_worker_task(task: Dict[str, object]) -> Dict[str, object]:
    if _WORKER_HW_DICT is None or _WORKER_MODEL_CONFIG is None or _WORKER_MODE is None:
        raise RuntimeError("Worker initialisation missing before task execution.")
    return _execute_parallelism_task(_WORKER_HW_DICT, _WORKER_MODEL_CONFIG, _WORKER_MODE, task)


def _sampling_worker(settings: Dict[str, int]) -> Dict[str, object]:
    if _WORKER_HW_DICT is None or _WORKER_MODEL_CONFIG is None or _WORKER_MODE is None:
        raise RuntimeError("Worker initialisation missing before task execution.")
    try:
        # Use fast memory check instead of full evaluation during sampling
        mem_check = check_memory_capacity(_WORKER_HW_DICT, _WORKER_MODEL_CONFIG, _WORKER_MODE, settings)
        return {
            "settings": settings,
            "memory_exceeded": bool(mem_check.get("memory_exceeded", False)),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"settings": settings, "memory_exceeded": True, "error": str(exc)}


def _parallelism_key(settings: Dict[str, object]) -> Tuple[Tuple[str, object], ...]:
    return tuple(sorted(settings.items()))


def _serialize_settings_key(key: Tuple[Tuple[str, object], ...]) -> List[List[object]]:
    return [[k, v] for k, v in key]


def _deserialize_settings_key(raw: Iterable[Iterable[object]]) -> Tuple[Tuple[str, object], ...]:
    return tuple((str(k), v) for k, v in raw)


def _normalise_faults_raw(faults: Iterable[Iterable[object]]) -> List[List[object]]:
    normalised: List[List[object]] = []
    for dim, weight, nodes in faults:
        normalised.append([int(dim), _quantize_weight(float(weight)), int(nodes)])
    return normalised


def _task_identifier(
    kind: str,
    settings_key: Tuple[Tuple[str, object], ...],
    *,
    faults: Sequence[Tuple[int, float, int]] | None = None,
    num_faults: int | None = None,
    fault_iter: Optional[int] = None,
) -> str:
    payload: Dict[str, object] = {
        "kind": kind,
        "settings_key": _serialize_settings_key(settings_key),
    }
    if fault_iter is not None:
        payload["fault_iter"] = int(fault_iter)
    if kind == "fault":
        effective_faults = faults or ()
        payload["num_faults"] = int(num_faults if num_faults is not None else len(effective_faults))
        payload["faults"] = _normalise_faults_raw(effective_faults)
    return json.dumps(payload, sort_keys=True)


def _load_existing_results(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
        print(f"Warning: results file {path} did not contain a JSON object. Ignoring existing data.", file=sys.stderr)
    except Exception as exc:
        print(f"Warning: failed to load results file {path}: {exc}", file=sys.stderr)
    return {}


def _record_partial_result(results_store: Dict[str, Dict[str, object]], entry: Dict[str, object]) -> None:
    task_id = entry.get("task_id")
    if not task_id:
        return
    results_store[task_id] = {
        "kind": entry.get("kind"),
        "settings_key": _serialize_settings_key(tuple(entry.get("settings_key", ()))),
        "runtime": entry.get("runtime"),
        "memory_exceeded": bool(entry.get("memory_exceeded", False)),
        "memory_violation_gb": float(entry.get("memory_violation_gb", 0.0) or 0.0),
    }
    if entry.get("kind") == "fault":
        if "fault_iter" in entry:
            results_store[task_id]["fault_iter"] = int(entry.get("fault_iter"))
        results_store[task_id]["num_faults"] = int(entry.get("num_faults", 0))
        results_store[task_id]["faults"] = _normalise_faults_raw(entry.get("faults", []))
    tmp_path = RESULTS_JSON_PATH.with_suffix(".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_WRITE_LOCK:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(results_store, handle, indent=2)
        tmp_path.replace(RESULTS_JSON_PATH)


def _dump_debug_hw_config(result: Dict[str, object]) -> Optional[Path]:
    if not DEBUG_MODE:
        return None
    yaml_text = result.get("hw_yaml")
    if not isinstance(yaml_text, str) or not yaml_text.strip():
        return None
    if DEBUG_RUN_DIR is None:
        return None
    global _DEBUG_FILE_INDEX
    _DEBUG_FILE_INDEX += 1
    settings = result.get("settings") or {}
    kind = str(result.get("kind") or "task")
    tp = settings.get("tp")
    cp = settings.get("cp")
    dp = settings.get("dp")
    lp = settings.get("lp")
    label = f"{kind}_{_DEBUG_FILE_INDEX:03d}"
    if all(isinstance(val, (int, float)) for val in (tp, cp, dp, lp)):
        label += f"_tp{int(tp)}_cp{int(cp)}_dp{int(dp)}_lp{int(lp)}"
    num_faults = result.get("num_faults")
    if kind == "fault" and isinstance(num_faults, int):
        label += f"_{num_faults}faults"
    file_path = DEBUG_RUN_DIR / f"{label}.yaml"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(yaml_text)
    DEBUG_SAVED_PATHS.append(file_path)
    return file_path


def _generate_hw_yaml_from_task(base_hw_dict: Dict[str, object], task: Dict[str, object]) -> Optional[str]:
    if base_hw_dict is None:
        return None
    try:
        hw_copy: Dict[str, object] = copy.deepcopy(base_hw_dict)
    except Exception as exc:
        print(f"Warning: failed to clone base hardware config: {exc}", file=sys.stderr)
        return None

    settings = dict(task.get("settings", {}) or {})
    parallel_block = hw_copy.setdefault("parallelism", {})
    if isinstance(parallel_block, dict):
        for key, value in settings.items():
            parallel_block[key] = value

    mutator = None
    faults: Sequence[Tuple[int, float, int]] = ()
    if task.get("kind") == "fault":
        raw_faults = task.get("faults") or []
        try:
            faults = [tuple(spec) for spec in raw_faults]  # type: ignore[assignment]
        except Exception:
            faults = ()
        mutator = make_fault_mutator(faults)

    try:
        if mutator is not None:
            mutator(hw_copy)
    except Exception as exc:
        print(f"Warning: failed to apply fault mutator for debug dump: {exc}", file=sys.stderr)

    try:
        return yaml.safe_dump(hw_copy, default_flow_style=False, sort_keys=False)
    except Exception as exc:
        print(f"Warning: failed to serialize debug hardware config: {exc}", file=sys.stderr)
        return None


def _ensure_failure_dump_dir() -> Path:
    global FAILURE_DUMP_DIR
    if FAILURE_DUMP_DIR is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        FAILURE_DUMP_DIR = FAILURE_OUTPUT_BASE / timestamp
        FAILURE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Saving failing hardware configs to {FAILURE_DUMP_DIR}")
    return FAILURE_DUMP_DIR


def _failure_label_from_settings(kind: str, settings: Dict[str, object], num_faults: Optional[int]) -> str:
    label = kind or "task"
    tp = settings.get("tp")
    cp = settings.get("cp")
    dp = settings.get("dp")
    lp = settings.get("lp")
    if all(isinstance(val, (int, float)) for val in (tp, cp, dp, lp)):
        label += f"_tp{int(tp)}_cp{int(cp)}_dp{int(dp)}_lp{int(lp)}"
    if kind == "fault" and isinstance(num_faults, int):
        label += f"_{num_faults}faults"
    return label


def _dump_failure_hw_config(
    base_hw_dict: Dict[str, object],
    result: Dict[str, object],
    error_message: Optional[str] = None,
) -> Optional[Path]:
    task_like = {
        "kind": result.get("kind"),
        "settings": result.get("settings"),
        "faults": result.get("faults"),
        "num_faults": result.get("num_faults"),
    }
    yaml_text = _generate_hw_yaml_from_task(base_hw_dict, task_like)
    failure_dir = _ensure_failure_dump_dir()
    global _FAILURE_FILE_INDEX
    _FAILURE_FILE_INDEX += 1
    settings = dict(result.get("settings", {}) or {})
    kind = str(result.get("kind") or "task")
    num_faults = result.get("num_faults")
    label = _failure_label_from_settings(kind, settings, num_faults if isinstance(num_faults, int) else None)
    base_filename = f"failure_{_FAILURE_FILE_INDEX:03d}_{label}"
    yaml_path = failure_dir / f"{base_filename}.yaml"
    command_line = None
    if isinstance(yaml_text, str) and yaml_text.strip():
        with yaml_path.open("w", encoding="utf-8") as handle:
            handle.write(yaml_text)
        FAILURE_SAVED_PATHS.append(yaml_path)
        print(f"Stored failing hardware config at {yaml_path}")
        cli_hw_path = os.path.relpath(yaml_path, os.getcwd())
        command_line = (
            f"uv run run_perf.py --hardware_config {shlex.quote(cli_hw_path)} "
            f"--model_config {shlex.quote(MODEL_CONFIG_PATH)}"
        )
    else:
        yaml_path = None
    if error_message or command_line:
        log_path = failure_dir / f"{base_filename}_error.txt"
        with log_path.open("w", encoding="utf-8") as handle:
            if error_message:
                handle.write(str(error_message).strip() + "\n")
            if command_line:
                handle.write("\n# Reproduce failing configuration\n")
                handle.write(command_line + "\n")
        FAILURE_SAVED_PATHS.append(log_path)
    return yaml_path


def _terminate_active_executor() -> None:
    global _ACTIVE_EXECUTOR
    executor = _ACTIVE_EXECUTOR
    if executor is None:
        return
    _ACTIVE_EXECUTOR = None
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    processes = getattr(executor, "_processes", None)
    if isinstance(processes, dict):
        for proc in processes.values():
            try:
                if proc is not None and proc.is_alive():
                    proc.kill()
            except Exception:
                continue


def _signal_handler(signum, frame):  # type: ignore[override]
    _terminate_active_executor()
    if signum == signal.SIGINT:
        raise KeyboardInterrupt()
    sys.exit(128 + signum)


def _install_signal_handlers() -> None:
    global _OLD_SIGINT_HANDLER, _OLD_SIGTERM_HANDLER
    _OLD_SIGINT_HANDLER = signal.getsignal(signal.SIGINT)
    _OLD_SIGTERM_HANDLER = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_terminate_active_executor)


# -----------------------------------------------------------------------------
# Reporting and plotting
# -----------------------------------------------------------------------------

def summarise_fault_runs(runtimes: Iterable[float]) -> Tuple[float, float, float]:
    values = [float(v) for v in runtimes if math.isfinite(v)]
    if not values:
        return float("nan"), float("nan"), float("nan")
    return min(values), max(values), float(sum(values) / len(values))


def write_fault_report(records: List[Dict[str, object]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = [
        "tp",
        "cp",
        "dp",
        "lp",
        "baseline_runtime",
        "fault_min",
        "fault_max",
        "fault_mean",
    ]
    with open(path, "w") as handle:
        handle.write("\t".join(header) + "\n")
        for record in records:
            row = [
                str(record["parallelism"]["tp"]),
                str(record["parallelism"]["cp"]),
                str(record["parallelism"]["dp"]),
                str(record["parallelism"]["lp"]),
                f"{record['baseline_runtime']:.6f}",
                f"{record['fault_min']:.6f}" if math.isfinite(record["fault_min"]) else "nan",
                f"{record['fault_max']:.6f}" if math.isfinite(record["fault_max"]) else "nan",
                f"{record['fault_mean']:.6f}" if math.isfinite(record["fault_mean"]) else "nan",
            ]
            handle.write("\t".join(row) + "\n")


def plot_fault_sensitivity(records: List[Dict[str, object]], output_path: str, num_gpus: int) -> None:
    if not records:
        print("No records to plot.", file=sys.stderr)
        return

    if sns is not None:  # pragma: no branch
        sns.set_theme(style="whitegrid")

    xs = np.arange(len(records))
    baseline = np.array([rec["baseline_runtime"] for rec in records], dtype=float)
    fault_mean = np.array([rec["fault_mean"] for rec in records], dtype=float)
    fault_min = np.array([rec["fault_min"] for rec in records], dtype=float)
    fault_max = np.array([rec["fault_max"] for rec in records], dtype=float)

    config_labels = [rec["label"] for rec in records]

    plt.figure(figsize=(12, 6))
    plt.scatter(xs, baseline, color="#1f78b4", marker="o", s=80, label="Baseline")

    fault_lower = []
    fault_upper = []
    for mean_val, min_val, max_val in zip(fault_mean, fault_min, fault_max):
        if math.isfinite(mean_val) and math.isfinite(min_val):
            fault_lower.append(max(mean_val - min_val, 0.0))
        else:
            fault_lower.append(0.0)
        if math.isfinite(mean_val) and math.isfinite(max_val):
            fault_upper.append(max(max_val - mean_val, 0.0))
        else:
            fault_upper.append(0.0)

    plt.errorbar(
        xs,
        fault_mean,
        yerr=[fault_lower, fault_upper],
        fmt="s",
        color="#d7301f",
        ecolor="#fb6a4a",
        elinewidth=2,
        capsize=6,
        capthick=1.6,
        markersize=7,
        label="Fault mean ± range",
    )

    # Candlestick-style shading for min/max bounds
    for x, fmin, fmax in zip(xs, fault_min, fault_max):
        if math.isfinite(fmin) and math.isfinite(fmax):
            plt.fill_between([x - 0.22, x + 0.22], [fmin, fmin], [fmax, fmax], color="#fdd0a2", alpha=0.4)

    plt.xticks(xs, config_labels, rotation=45, ha="right", fontsize=16)
    plt.ylabel("Runtime (s)", fontsize=19)
    plt.title(f"Fault Sensitivity Across Parallelism Configurations (Num GPUs = {num_gpus})", fontsize=18)
    plt.grid(alpha=0.3, axis="y")
    plt.legend(fontsize=16)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved fault sensitivity plot to {output_path}")


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def main(cli_args: Optional[Sequence[str]] = None) -> int:
    global TARGET_NUM_GPUS, SAMPLE_COUNT, FAULT_ITER, FAULT_WORKERS, FAULT_MAG
    global NUM_FAULTS, MIN_ALLOWED_TP, MAX_ALLOWED_TP, MIN_ALLOWED_CP, MAX_ALLOWED_CP
    global MIN_ALLOWED_DP, MAX_ALLOWED_DP, MIN_ALLOWED_LP, MAX_ALLOWED_LP
    global ALLOWED_FAULT_DIMS, PLOT_OUTPUT_PATH, REPORT_OUTPUT_PATH, RESULTS_JSON_PATH
    global RANDOM_SEED, ENFORCE_SQUARE_TP_CP, DEBUG_MODE, DEBUG_RUN_DIR, DEBUG_SAVED_PATHS
    global _DEBUG_FILE_INDEX, FAILURE_DUMP_DIR, FAILURE_SAVED_PATHS, _FAILURE_FILE_INDEX

    args = parse_args(cli_args)

    TARGET_NUM_GPUS = args.target_num_gpus
    SAMPLE_COUNT = args.sample_count
    FAULT_ITER = args.fault_iter
    FAULT_WORKERS = args.fault_workers

    mag_mean, mag_std = _parse_float_tuple(args.fault_mag)
    FAULT_MAG = (mag_mean, mag_std)

    parsed_num_faults = _parse_int_list(args.num_faults)
    NUM_FAULTS = parsed_num_faults or [1]

    MIN_ALLOWED_TP = max(1, args.min_tp)
    MAX_ALLOWED_TP = max(MIN_ALLOWED_TP, args.max_tp)
    MIN_ALLOWED_CP = max(1, args.min_cp)
    MAX_ALLOWED_CP = max(MIN_ALLOWED_CP, args.max_cp)
    MIN_ALLOWED_DP = max(1, args.min_dp)
    MAX_ALLOWED_DP = max(MIN_ALLOWED_DP, args.max_dp)
    MIN_ALLOWED_LP = max(1, args.min_lp)
    MAX_ALLOWED_LP = max(MIN_ALLOWED_LP, args.max_lp)

    allowed_dims_list = _parse_int_list(args.allowed_fault_dims) if args.allowed_fault_dims else []
    ALLOWED_FAULT_DIMS = allowed_dims_list if allowed_dims_list else None

    PLOT_OUTPUT_PATH = args.plot_output
    REPORT_OUTPUT_PATH = args.report_output
    RESULTS_JSON_PATH = Path(args.results_json)

    RANDOM_SEED = args.seed
    random.seed(RANDOM_SEED)
    ENFORCE_SQUARE_TP_CP = ENFORCE_SQUARE_TP_CP or bool(args.enforce_square_tp_cp)
    DEBUG_MODE = bool(args.debug_mode)
    _DEBUG_FILE_INDEX = 0
    _FAILURE_FILE_INDEX = 0
    FAILURE_SAVED_PATHS.clear()
    FAILURE_DUMP_DIR = None
    if DEBUG_MODE:
        FAULT_WORKERS = 1
        DEBUG_SAVED_PATHS.clear()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        DEBUG_RUN_DIR = DEBUG_OUTPUT_BASE / timestamp
        DEBUG_RUN_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Debug mode enabled. Hardware configs will be written to {DEBUG_RUN_DIR}")
    else:
        DEBUG_RUN_DIR = None
        DEBUG_SAVED_PATHS.clear()

    set_astrasim_cache_mode(ASTRA_CACHE_MODE)
    _install_signal_handlers()
    _run_fault_injection_self_test()

    base_hw_dict = read_yaml(HARDWARE_CONFIG_PATH)
    mode = determine_model_mode(MODEL_CONFIG_PATH)
    model_config_obj = config.parse_config(MODEL_CONFIG_PATH, config_type=mode)

    # existing_results = _load_existing_results(RESULTS_JSON_PATH)
    # if existing_results:
    #     print(f"Loaded {len(existing_results)} cached result entries from {RESULTS_JSON_PATH}")
    existing_results = {}
    results_store: Dict[str, Dict[str, object]] = dict(existing_results)

    parallelism_samples = generate_parallelism_samples(
        TARGET_NUM_GPUS, SAMPLE_COUNT, base_hw_dict, model_config_obj, mode, workers=FAULT_WORKERS
    )
    if not parallelism_samples:
        print("Unable to generate parallelism samples.", file=sys.stderr)
        return

    records: List[Dict[str, object]] = []
    records_by_key: Dict[Tuple[Tuple[str, object], ...], Dict[str, object]] = {}
    baseline_task_queue: List[Dict[str, object]] = []
    debug_task_queue: Optional[List[Dict[str, object]]] = [] if DEBUG_MODE else None

    for settings in parallelism_samples:
        settings_copy = dict(settings)
        key = _parallelism_key(settings_copy)
        settings_key = key

        candidates_for_config = _candidate_fault_dimensions(settings_copy)
        if not candidates_for_config:
            continue

        records_by_key[key] = {
            "parallelism": settings_copy,
            "baseline_runtime": None,
            "fault_runtimes": [],
            "fault_details": [],
            "memory_exceeded": False,
            "memory_violation_gb": 0.0,
        }

        baseline_task_id = _task_identifier("baseline", settings_key)
        baseline_cached = existing_results.get(baseline_task_id)
        if baseline_cached is not None:
            runtime = baseline_cached.get("runtime")
            if runtime is not None:
                records_by_key[key]["baseline_runtime"] = float(runtime)
            mem_flag = bool(baseline_cached.get("memory_exceeded", False))
            records_by_key[key]["memory_exceeded"] = mem_flag
            records_by_key[key]["memory_violation_gb"] = float(
                baseline_cached.get("memory_violation_gb", 0.0) or 0.0
            )
        else:
            baseline_task_queue.append(
                {
                    "kind": "baseline",
                    "settings": settings_copy,
                    "key": key,
                    "settings_key": settings_key,
                    "task_id": baseline_task_id,
                }
            )
        if debug_task_queue is not None:
            for num_faults in NUM_FAULTS:
                repeats = max(1, int(num_faults))
                for fault_iter in range(FAULT_ITER):
                    fault_specs: List[Tuple[int, float, int]] = []
                    for _ in range(repeats):
                        fault_value = sample_fault_magnitude()
                        dim_choice, node_count = random.choice(candidates_for_config)
                        fault_specs.append((dim_choice, fault_value, node_count))

                    fault_task_id = _task_identifier(
                        "fault",
                        settings_key,
                        faults=fault_specs,
                        num_faults=repeats,
                        fault_iter=fault_iter,
                    )
                    cached_fault = existing_results.get(fault_task_id)
                    if cached_fault is not None:
                        continue
                    debug_task_queue.append(
                        {
                            "kind": "fault",
                            "settings": settings_copy,
                            "key": key,
                            "settings_key": settings_key,
                            "task_id": fault_task_id,
                            "faults": fault_specs,
                            "num_faults": repeats,
                            "fault_iter": fault_iter,
                        }
                    )

    if DEBUG_MODE:
        debug_tasks: List[Dict[str, object]] = []
        if baseline_task_queue:
            debug_tasks.extend(baseline_task_queue)
        if debug_task_queue:
            debug_tasks.extend(debug_task_queue)
        if not debug_tasks:
            print("Debug mode enabled but no evaluation tasks were generated.")
        else:
            for task in debug_tasks:
                yaml_text = _generate_hw_yaml_from_task(base_hw_dict, task)
                debug_entry = {
                    "kind": task.get("kind"),
                    "settings": task.get("settings"),
                    "num_faults": task.get("num_faults"),
                    "faults": task.get("faults"),
                    "hw_yaml": yaml_text,
                }
                _dump_debug_hw_config(debug_entry)
        if DEBUG_RUN_DIR:
            if DEBUG_SAVED_PATHS:
                print(f"Debug mode: saved {len(DEBUG_SAVED_PATHS)} hardware config(s) to {DEBUG_RUN_DIR}")
            else:
                print(f"Debug mode enabled but no hardware configs were generated. Directory: {DEBUG_RUN_DIR}")
        else:
            print("Debug mode enabled but no output directory was initialised.")
        return 0

    def handle_task_result(result: Dict[str, object]) -> None:
        if not isinstance(result, dict):
            return
        key = result.get("key")
        rec = records_by_key.get(key)
        if rec is None:
            return
        status = result.get("status")
        if status != "ok":
            error_msg = result.get("error")
            print(
                f"Warning: task failed for settings {result.get('settings')}: {error_msg}",
                file=sys.stderr,
            )
            _dump_failure_hw_config(base_hw_dict, result, error_msg)
            return
        kind = result.get("kind")
        runtime = float(result.get("runtime", float("nan")))
        memory_exceeded = bool(result.get("memory_exceeded", False))
        memory_violation = float(result.get("memory_violation_gb", 0.0) or 0.0)
        if memory_exceeded:
            already_marked = rec.get("memory_exceeded", False)
            rec["memory_exceeded"] = True
            rec["memory_violation_gb"] = memory_violation
            if not already_marked:
                print(
                    f"Skipping configuration {result.get('settings')} due to memory capacity violation "
                    f"({memory_violation:.3f} GB).",
                    file=sys.stderr,
                )
            _record_partial_result(results_store, result)
            return
        if kind == "baseline":
            rec["baseline_runtime"] = runtime
        else:
            rec["fault_runtimes"].append(runtime)
            rec["fault_details"].append(
                {
                    "num_faults": result.get("num_faults"),
                    "faults": _normalise_faults_raw(result.get("faults", [])),
                }
            )
        _record_partial_result(results_store, result)
        _dump_debug_hw_config(result)

    def run_task_queue(
        task_queue: List[Dict[str, object]],
        desc: str,
        executor: Optional[ProcessPoolExecutor] = None,
    ) -> List[Dict[str, object]]:
        if not task_queue:
            return []
        results: List[Dict[str, object]] = []
        if executor is not None:
            global _ACTIVE_EXECUTOR
            _ACTIVE_EXECUTOR = executor
            futures = [executor.submit(_parallelism_worker_task, task) for task in task_queue]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="task"):
                try:
                    result = fut.result()
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"Warning: parallel task raised an exception: {exc}", file=sys.stderr)
                    continue
                results.append(result)
        else:
            if FAULT_WORKERS and FAULT_WORKERS > 1:
                available_cpus = max(1, os.cpu_count() or 1)
                worker_count = min(max(1, FAULT_WORKERS), len(task_queue), available_cpus)
                print(f"Launching ProcessPoolExecutor with {worker_count} worker(s) for {desc}.")
                executor = ProcessPoolExecutor(
                    max_workers=worker_count,
                    initializer=_parallelism_worker_init,
                    initargs=(base_hw_dict, MODEL_CONFIG_PATH, mode),
                )
                try:
                    _ACTIVE_EXECUTOR = executor
                    futures = [executor.submit(_parallelism_worker_task, task) for task in task_queue]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="task"):
                        try:
                            result = fut.result()
                        except Exception as exc:  # pragma: no cover - defensive
                            print(f"Warning: parallel task raised an exception: {exc}", file=sys.stderr)
                            continue
                        results.append(result)
                finally:
                    _terminate_active_executor()
            else:
                print(f"Running {desc} sequentially (FAULT_WORKERS <= 1).")
                for task in tqdm(task_queue, desc=desc, unit="task"):
                    try:
                        result = _execute_parallelism_task(base_hw_dict, model_config_obj, mode, task)
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"Warning: sequential task raised an exception: {exc}", file=sys.stderr)
                        continue
                    results.append(result)
        return results

    if baseline_task_queue:
        print(f"Prepared {len(baseline_task_queue)} baseline task(s).")
        baseline_results = run_task_queue(baseline_task_queue, "Baseline evaluations")
        for result in baseline_results:
            handle_task_result(result)
    else:
        print("No new baseline tasks; using cached results where available.")

    # Fault evaluation with fairness: all configs share the same fault iteration index.
    if not NUM_FAULTS or FAULT_ITER <= 0:
        print("No fault iterations configured; skipping fault evaluations.")
    else:
        print("Beginning fault evaluations with fairness policy (max retries per iteration: 10).")

    eligible_configs = []
    for settings in parallelism_samples:
        key = _parallelism_key(settings)
        state = records_by_key.get(key) or {}
        if state.get("memory_exceeded") or state.get("baseline_runtime") is None:
            continue
        candidates_for_config = _candidate_fault_dimensions(settings)
        if not candidates_for_config:
            continue
        eligible_configs.append((settings, key, candidates_for_config))

    fault_executor: Optional[ProcessPoolExecutor] = None
    fault_worker_count = 0
    progress_bar = None
    try:
        if eligible_configs and FAULT_WORKERS and FAULT_WORKERS > 1:
            available_cpus = max(1, os.cpu_count() or 1)
            fault_worker_count = min(max(1, FAULT_WORKERS), FAULT_ITER, available_cpus)
            if fault_worker_count > 1:
                print(f"Initializing shared ProcessPoolExecutor with {fault_worker_count} worker(s) for fault iterations.")
                fault_executor = ProcessPoolExecutor(
                    max_workers=fault_worker_count,
                    initializer=_parallelism_worker_init,
                    initargs=(base_hw_dict, MODEL_CONFIG_PATH, mode),
                )
                _ACTIVE_EXECUTOR = fault_executor

        def _build_fault_iteration(iter_idx: int, repeats: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
            fault_task_queue: List[Dict[str, object]] = []
            pending_results: List[Dict[str, object]] = []
            for settings, key, candidates_for_config in eligible_configs:
                fault_specs: List[Tuple[int, float, int]] = []
                for _ in range(repeats):
                    fault_value = sample_fault_magnitude()
                    dim_choice, node_count = random.choice(candidates_for_config)
                    fault_specs.append((dim_choice, fault_value, node_count))

                fault_task_id = _task_identifier(
                    "fault",
                    key,
                    faults=fault_specs,
                    num_faults=repeats,
                    fault_iter=iter_idx,
                )
                cached_fault = existing_results.get(fault_task_id)
                if cached_fault is not None:
                    if bool(cached_fault.get("memory_exceeded", False)):
                        pending_results.append(
                            {
                                "status": "error",
                                "kind": "fault",
                                "key": key,
                                "settings": settings,
                                "settings_key": key,
                                "task_id": fault_task_id,
                                "memory_exceeded": True,
                                "memory_violation_gb": cached_fault.get("memory_violation_gb", 0.0),
                            }
                        )
                        continue
                    runtime = cached_fault.get("runtime")
                    if runtime is not None:
                        pending_results.append(
                            {
                                "status": "ok",
                                "kind": "fault",
                                "key": key,
                                "settings": settings,
                                "settings_key": key,
                                "task_id": fault_task_id,
                                "runtime": float(runtime),
                                "faults": cached_fault.get("faults", fault_specs),
                                "num_faults": cached_fault.get("num_faults", repeats),
                                "fault_iter": iter_idx,
                                "memory_exceeded": False,
                                "memory_violation_gb": float(cached_fault.get("memory_violation_gb", 0.0) or 0.0),
                            }
                        )
                    continue

                fault_task_queue.append(
                    {
                        "kind": "fault",
                        "settings": settings,
                        "key": key,
                        "settings_key": key,
                        "task_id": fault_task_id,
                        "faults": fault_specs,
                        "num_faults": repeats,
                        "fault_iter": iter_idx,
                    }
                )
            return fault_task_queue, pending_results

        def _start_fault_iteration(iter_idx: int, repeats: int, attempt: int) -> None:
            task_queue, cached_results = _build_fault_iteration(iter_idx, repeats)
            if not task_queue and not cached_results:
                if progress_bar is not None:
                    progress_bar.update(1)
                return
            iteration_key = (iter_idx, repeats)
            iteration_state[iteration_key] = {
                "attempt": attempt,
                "remaining": len(task_queue),
                "results": list(cached_results),
            }
            if not task_queue:
                _finalise_iteration(iteration_key)
                return
            for task in task_queue:
                if fault_executor is not None:
                    fut = fault_executor.submit(_parallelism_worker_task, task)
                    in_flight[fut] = iteration_key
                else:
                    try:
                        result = _execute_parallelism_task(base_hw_dict, model_config_obj, mode, task)
                    except Exception as exc:  # pragma: no cover - defensive
                        result = {
                            "status": "error",
                            "kind": "fault",
                            "key": task.get("key"),
                            "settings": task.get("settings"),
                            "settings_key": task.get("settings_key"),
                            "task_id": task.get("task_id"),
                            "error": str(exc),
                        }
                    iteration_state[iteration_key]["results"].append(result)
                    iteration_state[iteration_key]["remaining"] -= 1
            if fault_executor is None:
                _finalise_iteration(iteration_key)

        def _finalise_iteration(iteration_key: Tuple[int, int]) -> None:
            state = iteration_state.get(iteration_key)
            if state is None or state["remaining"] > 0:
                return
            iter_idx, repeats = iteration_key
            all_results = state.get("results", [])
            failed = any((res.get("status") != "ok") or bool(res.get("memory_exceeded", False)) for res in all_results)
            if failed:
                last_error = None
                for res in reversed(all_results):
                    if (res.get("status") != "ok") or bool(res.get("memory_exceeded", False)):
                        if res.get("error"):
                            last_error = str(res.get("error"))
                            _dump_failure_hw_config(base_hw_dict, res, res.get("error"))
                        elif res.get("memory_exceeded"):
                            last_error = (
                                f"memory exceeded ({float(res.get('memory_violation_gb', 0.0) or 0.0):.3f} GB over)"
                            )
                        else:
                            last_error = "unknown error"
                        break
                attempt = state.get("attempt", 1)
                if attempt >= 10:
                    msg = (
                        f"Omitting fault iteration {iter_idx} for num_faults={repeats} across all configs "
                        f"after {attempt} failed attempt(s)."
                    )
                    if last_error:
                        msg += f" Last error: {last_error}"
                    print(msg, file=sys.stderr)
                    if progress_bar is not None:
                        progress_bar.update(1)
                else:
                    pending_iterations.append((iter_idx, repeats, attempt + 1))
                iteration_state.pop(iteration_key, None)
                return
            for res in all_results:
                handle_task_result(res)
            iteration_state.pop(iteration_key, None)
            if progress_bar is not None:
                progress_bar.update(1)

        if eligible_configs and NUM_FAULTS and FAULT_ITER > 0:
            pending_iterations: deque[Tuple[int, int, int]] = deque()
            for num_faults in NUM_FAULTS:
                repeats = max(1, int(num_faults))
                for iter_idx in range(FAULT_ITER):
                    pending_iterations.append((iter_idx, repeats, 1))

            in_flight: Dict[object, Tuple[int, int]] = {}
            iteration_state: Dict[Tuple[int, int], Dict[str, object]] = {}
            backlog_target = max(1, (fault_worker_count or 1) * 4)
            total_iterations = len(pending_iterations)
            progress_bar = tqdm(total=total_iterations, desc="Fault iterations", unit="iter")

            while pending_iterations or in_flight:
                while pending_iterations and (fault_executor is None or len(in_flight) < backlog_target):
                    iter_idx, repeats, attempt = pending_iterations.popleft()
                    _start_fault_iteration(iter_idx, repeats, attempt)

                if fault_executor is None:
                    if not pending_iterations:
                        break
                    continue

                if not in_flight:
                    continue
                try:
                    next_future = next(as_completed(list(in_flight.keys()), timeout=None))
                except StopIteration:
                    continue
                iteration_key = in_flight.pop(next_future, None)
                if iteration_key is None:
                    continue
                try:
                    result = next_future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    result = {"status": "error", "error": str(exc)}
                state = iteration_state.get(iteration_key)
                if state is not None:
                    state["results"].append(result)
                    state["remaining"] = max(0, int(state["remaining"]) - 1)
                    _finalise_iteration(iteration_key)
    finally:
        try:
            if progress_bar is not None:
                progress_bar.close()
        except Exception:
            pass
        if fault_executor is not None:
            _ACTIVE_EXECUTOR = fault_executor
            _terminate_active_executor()

    for settings in parallelism_samples:
        key = _parallelism_key(settings)
        state = records_by_key.get(key)
        if not state:
            continue
        if state.get("memory_exceeded"):
            continue
        baseline_runtime = state.get("baseline_runtime")
        if baseline_runtime is None:
            print(f"Warning: missing baseline runtime for settings {settings}", file=sys.stderr)
            continue
        fault_min, fault_max, fault_mean = summarise_fault_runs(state.get("fault_runtimes", []))
        record = {
            "parallelism": state["parallelism"],
            "baseline_runtime": float(baseline_runtime),
            "fault_min": fault_min,
            "fault_max": fault_max,
            "fault_mean": fault_mean,
            "label": f"tp{settings['tp']}-cp{settings['cp']}-dp{settings['dp']}-pp{settings['lp']}",
        }
        records.append(record)

    if not records:
        print("No successful sweep results collected.", file=sys.stderr)
        return 0

    write_fault_report(records, REPORT_OUTPUT_PATH)
    print(f"Wrote fault sweep report to {REPORT_OUTPUT_PATH}")
    plot_fault_sensitivity(records, PLOT_OUTPUT_PATH, TARGET_NUM_GPUS)
    if FAILURE_SAVED_PATHS and FAILURE_DUMP_DIR:
        print(
            f"Captured {len(FAILURE_SAVED_PATHS)} failing hardware config artefact(s) in {FAILURE_DUMP_DIR}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
