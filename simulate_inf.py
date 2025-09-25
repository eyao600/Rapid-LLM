"""LLM inference prefill simulation entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass

from timecalculation_inf import LLMInferencePrefillArtifacts


@dataclass
class LLMInferencePrefillSummary:
    """Summary of a prefill-only inference pass."""

    total_time: float


def simulate_llm_inference_prefill(
    artifacts: LLMInferencePrefillArtifacts,
    output_dir: str,
) -> LLMInferencePrefillSummary:
    """Simulate the prefill graph and persist a simple report."""

    timed_root = artifacts.pipeline_graph.convert_comm_sizes_to_times(
        artifacts.pipeline_root,
        artifacts.time_calc.network_model,
        artifacts.interconnect_params,
    )

    total_time = artifacts.pipeline_graph.simulate(timed_root)

    output_path = os.path.join(output_dir, "LLM_inference_prefill_results.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as handle:
        handle.write("\n\n==============================================\n")
        handle.write("Inference Prefill Results\n")
        handle.write("==============================================\n")
        handle.write(f"Prefill Time: {total_time:.8f}\n")

    print(f"LLM inference prefill total time: {total_time}")

    return LLMInferencePrefillSummary(total_time=total_time)


__all__ = ["LLMInferencePrefillSummary", "simulate_llm_inference_prefill"]
