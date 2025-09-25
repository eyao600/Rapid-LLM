"""LLM inference prefill time-calculation entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from simulate_LLM import Graph
from time_calculation_LLM import TimeCalculationLLM


@dataclass
class LLMInferencePrefillArtifacts:
    """Artifacts required to simulate an inference prefill run."""

    time_calc: TimeCalculationLLM
    pipeline_graph: Graph
    pipeline_root: Any
    interconnect_params: Dict[str, Tuple[float, float]]
    node_breakdown: Dict[str, float]
    gemm_results: Dict[str, Dict[str, Any]]


def _build_pipeline_graph(
    time_calc: TimeCalculationLLM,
    node_breakdown: Dict[str, float],
    comm_metadata: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> Tuple[Graph, Any]:
    """Construct a forward-only pipeline graph for inference prefill."""

    comp_times = {
        "embedding_f": node_breakdown["embedding_f"],
        "embedding_b": 0.0,
        "linear_softmax_f": node_breakdown["linear_softmax_f"],
        "linear_softmax_b": 0.0,
        "transformer_f": node_breakdown["transformer_time_f"],
        "transformer_b": 0.0,
        "cross_layer_f": 0.0,
        "cross_layer_b": 0.0,
    }

    misc_metadata = {
        "num_batch": time_calc.mb,
        "num_layer": time_calc.num_layers,
        "all_reduce": getattr(time_calc, "all_reduce", "the end"),
    }

    graph = Graph(
        mode="pipeline",
        dp=time_calc.dp,
        lp=time_calc.lp,
        kp1=time_calc.kp1,
        kp2=time_calc.kp2,
        tp_mode=time_calc.t,
        comp_times=comp_times,
        comm_metadata=comm_metadata,
        misc_metadata=misc_metadata,
    )

    root = graph.construct_fwd_bwd_graph(include_backward=False)

    # Persist a checkpoint of the forward-only graph if requested for debugging.
    if os.environ.get("DEEPFLOW_VISUALIZE_GRAPHS"):
        graph.save_graph(root, output_dir.rstrip(os.sep) + os.sep, "pipeline_graph_inference_prefill")

    return graph, root


def _build_comm_metadata(time_calc: TimeCalculationLLM) -> Dict[str, Dict[str, Any]]:
    batch_size = time_calc._effective_transformer_batch()
    hidden_dim = time_calc.hidden_dim
    seq_len = time_calc.seq_len
    vocab_size = time_calc.vocab_size
    ffn_dim = time_calc.ffn_dim if hasattr(time_calc, "ffn_dim") else time_calc.hidden_dim * time_calc.ffn_mult

    reduction_sizes = time_calc.get_data_parallel_reduction_sizes(hidden_dim, ffn_dim)
    local_comp = time_calc.get_data_parallel_local_computation(hidden_dim, ffn_dim)
    embedding_size = int(
        time_calc.precision * vocab_size * hidden_dim
    ) + int(time_calc.precision * seq_len * hidden_dim)
    softmax_size = int(time_calc.precision * hidden_dim * vocab_size)
    cross_layer_bytes = time_calc.get_inter_layer_comm_latency_llm(batch_size, hidden_dim, seq_len)[1]

    return time_calc._build_comm_metadata(
        reduction_sizes=reduction_sizes,
        local_comp=local_comp,
        embedding_size=embedding_size,
        softmax_size=softmax_size,
        cross_layer_bytes=cross_layer_bytes,
    )


def calculate_llm_inference_prefill(exp_hw_config, exp_model_config, mode, output_dir) -> LLMInferencePrefillArtifacts:
    """Prepare the artifacts required to run an inference prefill simulation."""

    del mode  # Mode is implied by the model configuration for LLM workloads.

    time_calc = TimeCalculationLLM(exp_hw_config, exp_model_config, "LLM", output_dir=output_dir)

    # Mirror the training flow up to graph creation so tensor shapes and timing reuse existing logic.
    time_calc.readjust_type()

    batch_size = time_calc._effective_transformer_batch()
    vocab_size = time_calc.vocab_size
    hidden_dim = time_calc.hidden_dim
    seq_len = time_calc.seq_len
    num_heads = time_calc.num_heads
    ffn_dim = time_calc.hidden_dim * time_calc.ffn_mult if time_calc.ffn_mult else time_calc.ffn_dim

    gemm_results, node_breakdown = time_calc.compute_all_gemm_and_node_times(
        batch_size,
        vocab_size,
        hidden_dim,
        seq_len,
        num_heads,
        ffn_dim,
    )

    comm_metadata = _build_comm_metadata(time_calc)

    pipeline_graph, root = _build_pipeline_graph(time_calc, node_breakdown, comm_metadata, time_calc.output_dir)

    interconnect_params = time_calc._build_interconnect_params()

    return LLMInferencePrefillArtifacts(
        time_calc=time_calc,
        pipeline_graph=pipeline_graph,
        pipeline_root=root,
        interconnect_params=interconnect_params,
        node_breakdown=node_breakdown,
        gemm_results=gemm_results,
    )


__all__ = ["LLMInferencePrefillArtifacts", "calculate_llm_inference_prefill"]
