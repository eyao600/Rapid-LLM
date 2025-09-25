# KV Cache Integration Plan

## Goals
- Extend DeepFlow's inference pipeline to model KV-cache traffic explicitly (cache is always present for LLM inference).
- Represent cache interactions as dedicated compute nodes so all execution modes (analytical, hybrid, full AstraSim hierarchical, full AstraSim flattened) observe consistent timing behavior.
- Maintain current correctness for training runs while enriching inference with cache-aware telemetry.

## Implementation Phases

### 1. Surface Cache Metadata
- Ensure inference-oriented hardware configs provide cache metadata (e.g., `inference.kvcache_precision`, `inference.kvcache_type`, `inference.kvcache_fetch_overlap`) even though cache usage itself is unconditional.
- Confirm `Model_LLM` (`model.py`) exposes cache parameters without any enable/disable guards.

### 2. Compute KV Cache Traffic Metadata
- Implement a helper in `LLM_util.py` that returns per-layer KV bytes (keys + values) based on `batch_size`, `num_heads`, `head_dim`, and precision.
- Reuse the helper across prefill and decode paths so cache writes/reads stay in sync with GEMM-derived sequence lengths.

- After GEMM timings are computed in `TimeCalculationLLMInference.calc_time`, calculate cache population cost per layer using `self.roofline(0, kv_bytes, ...)` and total it across pipeline ranks.
- Inject `kv_cache_store` timings into the `node_breakdown`/`gemm_results` map passed to `_prepare_execution_graphs`.
- Update `Graph.construct_fwd_bwd_graph` to create forward-only `kv_cache_store` nodes when the comp-time key is present, tagging them as compute nodes with distinctive labels so visualization tooling can color them while keeping them off the critical path.

- Within `DecodeGraph._execute_decode_step`, compute cache read/write bytes for the current sequence length and add `kv_cache_fetch` (read existing cache) and `kv_cache_store` (append new token) entries to `gemm_results`.
- Extend `_decode_node_breakdown` in `TimeCalculationLLMInference` to include these entries so the pipeline graph knows about cache operations, with an option to overlap fetch latency with preceding compute when hardware supports it.
- Ensure decode sampling/interpolation still works when cache ops are present (i.e., interpolated totals include cache timing).

### 5. Graph & Dispatcher Adjustments
- Treat new cache nodes as standard compute nodes: assign consistent naming (e.g., `kv_cache_store`, `kv_cache_fetch`) and a distinct metadata flag so `convert_comm_sizes_to_times` ignores them but `simulate` accumulates their runtime and visualization tools can color them differently.
- Update `PipelineGraphFlattener` to clone `kv_cache_*` nodes without special handling (they should behave like other pointwise ops).
- Verify analytical mode (no transformer graph) still runs because cache nodes appear only in the pipeline graph; ensure code paths guard against missing transformer entries.

### 6. Execution-Mode Validation
- **Analytical:** confirm cache nodes are simulated by `Graph.simulate`, and total time matches expectations.
- **Hybrid:** ensure transformer graph includes cache nodes or, if they belong only to the pipeline graph, that hybrid execution still overlays AstraSim results correctly.
- **Full AstraSim Hierarchical:** confirm new nodes persist into the serialized pipeline graph and remain classified as compute (no communication metadata).
- **Full AstraSim Flattened:** update flattening logic if necessary so cache nodes clone correctly and AstraSim rank counts remain unchanged.

### 7. Instrumentation & Tests
- Add logging summarizing total cache read/write bytes per run for quick sanity checks.
- Create a lightweight regression (e.g., in `run_perf.py`) that runs inference with and without cache enabled and compares timing deltas for a short decode sequence.

## Future Work
- Support multi-tier cache placement (HBM + SRAM), selecting tier based on capacity/latency trade-offs.
- Model cache eviction policies for very long decode sequences.
- Expose cache statistics in output artifacts (graphs, CSVs) to aid downstream analysis.
