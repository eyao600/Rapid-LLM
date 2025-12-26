import math
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional

import llm_util


class MemKind(Enum):
    TRANSFORMER = "transformer"
    EMBEDDING = "embedding"
    SOFTMAX = "softmax"
    OPTIMIZER = "optimizer"
    LAYERNORM1 = "layernorm1"
    QKV_PROJ = "qkv_proj"
    ATTENTION = "attention"
    OUTPUT_PROJ = "output_proj"
    LAYERNORM2 = "layernorm2"
    MLP = "MLP"
    ROUTER = "router"


_OP_NAME_TO_MEM_KIND: Dict[str, MemKind] = {
    kind.value: kind for kind in MemKind
}


def mem_kind_from_op_name(op_name: str) -> MemKind:
    if op_name in _OP_NAME_TO_MEM_KIND:
        return _OP_NAME_TO_MEM_KIND[op_name]
    raise ValueError(f"Unsupported mem_kind op name: {op_name!r}")


TRANSFORMER_OP_KINDS: FrozenSet[MemKind] = frozenset(
    {
        MemKind.LAYERNORM1,
        MemKind.QKV_PROJ,
        MemKind.ATTENTION,
        MemKind.OUTPUT_PROJ,
        MemKind.LAYERNORM2,
        MemKind.MLP,
        MemKind.ROUTER,
        MemKind.TRANSFORMER,
    }
)


NON_TRANSFORMER_KINDS: FrozenSet[MemKind] = frozenset(
    {
        MemKind.EMBEDDING,
        MemKind.SOFTMAX,
        MemKind.OPTIMIZER,
    }
)


class MemoryEstimator:
    """Build memory sizing inputs for graph-based peak memory simulation."""

    def __init__(self, time_calc: Any) -> None:
        self.time_calc = time_calc

    def build_memory_data(
        self,
        *,
        mode: str,
        batch_size: int,
        seq_len: int,
        gemm_shapes: Optional[Dict[str, Any]] = None,
        kv_cache_tokens: Optional[int] = None,
        zero3_ephemeral_peak_bytes: Optional[float] = None,
    ) -> Dict[str, Any]:
        tc = self.time_calc
        precision = tc.precision

        dp = max(1, int(getattr(tc, "dp", 1)))
        tp = max(1, int(getattr(tc, "tp", 1)))
        lp = max(1, int(getattr(tc, "lp", 1)))
        cp = max(1, int(getattr(tc, "cp", 1)))
        zero_stage = int(getattr(tc, "zero_stage", 0) or 0)
        flash_attention = bool(getattr(tc, "flash_attention", False))
        full_recomputation = bool(getattr(tc, "full_recomputation", False))
        use_moe = bool(getattr(tc, "use_moe", False))
        if mode == "inference":
            dp = 1
            if cp != 1:
                raise ValueError("Inference memory estimation requires cp=1 (context parallelism is WIP).")

        if use_moe and (dp != 1 or tp != 1 or cp != 1 or lp != 1):
            raise RuntimeError(
                "MoE memory estimation is only supported for single-GPU configurations (dp=tp=cp=lp=1)."
            )

        seq_len_eff = float(seq_len) / float(cp)

        (
            transformer_mem_layer,
            transformer_act_layer,
            transformer_act_layer_inf,
            transformer_static_layer,
            gradient_mem_layer,
            optimizer_mem_layer,
            weight_memory_layer,
        ) = llm_util.get_transformer_mem_layer(
            dp=dp,
            tp=tp,
            lp=lp,
            mb=max(1, int(getattr(tc, "mb", 1))),
            batch_size=batch_size,
            hidden_dim=tc.hidden_dim,
            seq_len=seq_len_eff,
            intermediate_size=tc.intermediate_size,
            n_heads=tc.num_heads,
            precision=precision,
            model_type=tc.model_type,
            zero_stage=zero_stage,
            flash_attention=flash_attention,
            full_recomputation=full_recomputation,
        )

        (
            _total_params_per_rank,
            max_layer_params,
            params_per_layer_per_rank,
            embedding_params_per_rank,
            output_params_per_rank,
        ) = tc._param_stats_per_rank(tc.hidden_dim, tc.intermediate_size, tc.vocab_size)

        if zero3_ephemeral_peak_bytes is None:
            if zero_stage >= 3 and dp > 1:
                zero3_ephemeral_peak_bytes = max_layer_params * precision.parameters
            else:
                zero3_ephemeral_peak_bytes = 0.0

        extra_static_bytes_per_device: Dict[int, float] = {}

        bytes_per_param_weight = 0.0
        bytes_per_param_grad = 0.0
        bytes_per_param_opt = 0.0
        if params_per_layer_per_rank:
            denom = float(params_per_layer_per_rank)
            bytes_per_param_weight = float(weight_memory_layer) / denom
            if mode == "training":
                bytes_per_param_grad = float(gradient_mem_layer) / denom
                bytes_per_param_opt = float(optimizer_mem_layer) / denom

        per_param_static_bytes = (
            bytes_per_param_weight + bytes_per_param_grad + bytes_per_param_opt
        )
        embedding_static_bytes = float(embedding_params_per_rank) * per_param_static_bytes
        lm_head_static_bytes = float(output_params_per_rank) * per_param_static_bytes
        const_mem_offset_bytes = 0.0
        hw_config = getattr(tc, "hw_config", None)
        sw_config = getattr(hw_config, "sw_config", None) if hw_config is not None else None
        if sw_config is not None:
            const_mem_offset_bytes = float(getattr(sw_config, "const_mem_offset", 0.0) or 0.0)

        def _build_rank_layout():
            hw_config = getattr(tc, "hw_config", None)
            layout = getattr(hw_config, "network_layout", None)
            dimensions = getattr(layout, "dimensions", None) if layout is not None else None
            if not dimensions:
                return None

            axis_sizes = {"tp": tp, "cp": cp, "lp": lp, "dp": dp}
            axis_order = []
            for dim in dimensions:
                dim_axes = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ()) or ()]
                for name in dim_axes:
                    if name not in axis_sizes:
                        raise ValueError(
                            f"Unsupported parallelism axis '{name}' in network layout. "
                            "Supported axes for memory estimation are: tp, cp, lp, dp."
                        )
                    if name not in axis_order:
                        axis_order.append(name)

                declared = int(getattr(dim, "size", 1))
                expected = 1
                for axis_name in dim_axes:
                    expected *= axis_sizes.get(axis_name, 1)
                if expected != declared:
                    raise ValueError(
                        f"Network dimension '{getattr(dim, 'label', getattr(dim, 'id', '<unnamed>'))}' "
                        f"size mismatch: declared {declared}, but parallelism factors imply {expected}."
                    )

            if not axis_order:
                return None
            if tp > 1 and "tp" not in axis_order:
                raise ValueError("Network layout must include 'tp' when tensor parallelism > 1.")
            if cp > 1 and "cp" not in axis_order:
                raise ValueError("Network layout must include 'cp' when context parallelism > 1.")
            if lp > 1 and "lp" not in axis_order:
                raise ValueError("Network layout must include 'lp' when pipeline parallelism > 1.")

            axis_strides = {}
            span = 1
            for axis in axis_order:
                axis_strides[axis] = span
                span *= axis_sizes[axis]
            return axis_order, axis_sizes, axis_strides

        def _hw_id_for_rank(stage_id: int, tp_rank: int, layout):
            if layout is None:
                return stage_id * par_degree + tp_rank
            axis_order, axis_sizes, axis_strides = layout
            coords = {}
            if "tp" in axis_order:
                coords["tp"] = tp_rank % axis_sizes["tp"]
            if "cp" in axis_order:
                coords["cp"] = (tp_rank // axis_sizes["tp"]) % axis_sizes["cp"]
            if "lp" in axis_order:
                if stage_id < 0 or stage_id >= axis_sizes.get("lp", 1):
                    raise ValueError(f"stage_id {stage_id} is out of range for lp={axis_sizes.get('lp', 1)}")
                coords["lp"] = stage_id % axis_sizes["lp"]

            linear_rank = 0
            for axis in axis_order:
                coord = coords.get(axis, 0)
                size = axis_sizes.get(axis, 1)
                if coord < 0 or coord >= size:
                    raise ValueError(f"Coordinate {coord} for axis '{axis}' is out of range <{size}")
                stride = axis_strides.get(axis)
                if stride is None:
                    raise KeyError(f"Rank layout stride missing for axis '{axis}'")
                linear_rank += coord * stride
            return linear_rank

        par_degree = max(1, tp * cp)
        layout = None
        if embedding_static_bytes or lm_head_static_bytes or const_mem_offset_bytes:
            layout = _build_rank_layout()
        if embedding_static_bytes or lm_head_static_bytes:
            last_stage = max(0, lp - 1)
            for tp_rank in range(par_degree):
                if embedding_static_bytes:
                    hw_id = _hw_id_for_rank(0, tp_rank, layout)
                    extra_static_bytes_per_device[hw_id] = extra_static_bytes_per_device.get(hw_id, 0.0) + embedding_static_bytes
                if lm_head_static_bytes:
                    hw_id = _hw_id_for_rank(last_stage, tp_rank, layout)
                    extra_static_bytes_per_device[hw_id] = extra_static_bytes_per_device.get(hw_id, 0.0) + lm_head_static_bytes
        if const_mem_offset_bytes:
            for stage_id in range(lp):
                for tp_rank in range(par_degree):
                    hw_id = _hw_id_for_rank(stage_id, tp_rank, layout)
                    extra_static_bytes_per_device[hw_id] = (
                        extra_static_bytes_per_device.get(hw_id, 0.0) + const_mem_offset_bytes
                    )

        kv_cache_bytes_per_layer = 0.0
        if gemm_shapes is None:
            gemm_shapes = llm_util.process_gemm_shapes(
                tc,
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=tc.hidden_dim,
                num_heads=tc.num_heads,
                kv_heads=tc.kv_heads,
                intermediate_size=tc.intermediate_size,
                vocab_size=tc.vocab_size,
            )

        gemm_type_map = {
            "qkv_proj": "qkv",
            "attention_score": "attention_score",
            "attention_output": "attention_output",
            "output_proj": "out_proj",
            "ffn1": "ffn1",
            "ffn2": "ffn2",
            "linear": "linear_softmax",
            "router": "ffn1",
        }

        def _gemm_out_bytes(key: str, gemm_type: str) -> float:
            if key not in gemm_shapes:
                raise RuntimeError(f"Missing GEMM shape for '{key}'")
            elements = tc._gemm_output_elements(gemm_shapes[key], gemm_type)
            return float(elements) * precision.activations

        qkv_bytes = _gemm_out_bytes("qkv_proj", gemm_type_map["qkv_proj"])
        attn_out_bytes = _gemm_out_bytes("attention_output", gemm_type_map["attention_output"])
        out_proj_bytes = _gemm_out_bytes("output_proj", gemm_type_map["output_proj"])
        ffn1_bytes = _gemm_out_bytes("ffn1", gemm_type_map["ffn1"])
        ffn2_bytes = _gemm_out_bytes("ffn2", gemm_type_map["ffn2"])
        attn_score_bytes = 0.0
        if not flash_attention:
            attn_score_bytes = _gemm_out_bytes("attention_score", gemm_type_map["attention_score"])

        softmax_out_bytes = 0.0
        if "linear" in gemm_shapes:
            softmax_out_bytes = _gemm_out_bytes("linear", gemm_type_map["linear"])
        else:
            tokens = float(batch_size) * float(seq_len_eff)
            vocab_shard = math.ceil(float(tc.vocab_size) / float(max(1, tp * cp)))
            softmax_out_bytes = tokens * float(vocab_shard) * float(precision.activations)

        router_bytes = 0.0
        if use_moe:
            router_bytes = _gemm_out_bytes("router", gemm_type_map["router"])

        # Activation output sizes by operation.
        # LayerNorm outputs have shape (batch, seq, hidden_dim), same as out_proj.
        # When FlashAttention is OFF, we store the softmax output (same shape as
        # attention scores) for backward; this is added to ATTENTION below.
        base_outputs = {
            MemKind.LAYERNORM1: out_proj_bytes,  # (batch, seq, hidden_dim)
            MemKind.QKV_PROJ: qkv_bytes,
            MemKind.ATTENTION: attn_out_bytes,
            MemKind.OUTPUT_PROJ: out_proj_bytes,  # (batch, seq, hidden_dim)
            MemKind.LAYERNORM2: out_proj_bytes,  # (batch, seq, hidden_dim)
            MemKind.MLP: ffn2_bytes,
            MemKind.ROUTER: router_bytes,
            MemKind.EMBEDDING: out_proj_bytes,
            MemKind.SOFTMAX: softmax_out_bytes,
            MemKind.OPTIMIZER: 0.0,
        }
        parallelism_mode = None
        try:
            parallelism_mode = tc.get_parallelism_mode()
        except AttributeError:
            parallelism_mode = None
        if parallelism_mode is not None:
            mode_value = str(getattr(parallelism_mode, "value", parallelism_mode)).lower()
            if mode_value.endswith("tensor_sequence"):
                seq_degree = max(1, int(tc._sequence_parallel_degree()))
                hidden_full_bytes = (
                    float(batch_size)
                    * float(seq_len)
                    * float(tc.hidden_dim)
                    * float(precision.activations)
                )
                hidden_shard_bytes = (
                    float(batch_size)
                    * float(math.ceil(float(seq_len) / float(seq_degree)))
                    * float(tc.hidden_dim)
                    * float(precision.activations)
                )
                base_outputs[MemKind.OUTPUT_PROJ] = hidden_shard_bytes
                base_outputs[MemKind.MLP] = hidden_shard_bytes
                base_outputs[MemKind.LAYERNORM1] = hidden_full_bytes
                base_outputs[MemKind.LAYERNORM2] = hidden_full_bytes

        transformer_fallback_bytes = transformer_act_layer
        if mode != "training":
            transformer_fallback_bytes = transformer_act_layer_inf
        base_outputs[MemKind.TRANSFORMER] = transformer_fallback_bytes

        persistent_bytes_by_kind = {kind: 0.0 for kind in base_outputs}
        transient_bytes_by_kind = {kind: 0.0 for kind in base_outputs}

        store_outputs = mode == "training" and not full_recomputation
        if store_outputs:
            for kind, bytes_val in base_outputs.items():
                persistent_bytes_by_kind[kind] = float(bytes_val or 0.0)
            if attn_score_bytes:
                persistent_bytes_by_kind[MemKind.ATTENTION] += float(attn_score_bytes)
            persistent_bytes_by_kind[MemKind.MLP] += float(ffn1_bytes)
        else:
            for kind, bytes_val in base_outputs.items():
                transient_bytes_by_kind[kind] = float(bytes_val or 0.0)
            if attn_score_bytes:
                transient_bytes_by_kind[MemKind.ATTENTION] += float(attn_score_bytes)
            transient_bytes_by_kind[MemKind.MLP] += float(ffn1_bytes)

        if mode != "training" and kv_cache_tokens:
            # KV cache stores K and V tensors with shape (kv_heads, head_dim, seq).
            # For GQA/MQA, kv_heads < num_heads, so we must use kv_heads directly
            # rather than deriving from the attention score GEMM (which uses Q heads).
            kv_heads = int(getattr(tc, "kv_heads", tc.num_heads))
            head_dim = tc.hidden_dim // tc.num_heads
            kv_heads_per_tp = math.ceil(kv_heads / tp)
            kv_tokens = int(kv_cache_tokens)
            kv_cache_bytes_per_layer = (
                float(batch_size)
                * float(kv_heads_per_tp)
                * float(head_dim)
                * float(kv_tokens)
                * float(precision.kv_cache)
                * 2.0  # K and V
            )

        param_gather_bytes = 0.0
        if mode == "training" and zero3_ephemeral_peak_bytes and dp > 1 and zero_stage >= 3:
            param_gather_bytes = float(zero3_ephemeral_peak_bytes)

        return {
            "activation_mem_per_layer": transformer_act_layer,
            "activation_mem_per_layer_inference": transformer_act_layer_inf,
            "weight_mem_per_layer": weight_memory_layer,
            "gradient_mem_per_layer": gradient_mem_layer,
            "optimizer_mem_per_layer": optimizer_mem_layer,
            "static_mem_per_layer": transformer_static_layer,
            "total_mem_per_layer": transformer_mem_layer,
            "persistent_bytes_by_kind": persistent_bytes_by_kind,
            "transient_bytes_by_kind": transient_bytes_by_kind,
            "extra_static_bytes_per_device": extra_static_bytes_per_device,
            "kv_cache_bytes_per_layer": kv_cache_bytes_per_layer,
            "zero3_ephemeral_peak_bytes": zero3_ephemeral_peak_bytes,
            "param_gather_bytes": param_gather_bytes,
        }

    def simulate_peak(
        self,
        graph_root: Any,
        memory_data: Dict[str, Any],
        *,
        mode: str,
        filename: Optional[str] = None,
    ) -> Any:
        """Run memory simulation on the provided graph root."""
        def _is_non_flattened(root: Any) -> bool:
            stack = list(root if isinstance(root, (list, tuple)) else [root])
            visited = set()
            while stack:
                current = stack.pop()
                if current is None:
                    continue
                current_id = id(current)
                if current_id in visited:
                    continue
                visited.add(current_id)

                name = getattr(current, "name", "")
                if isinstance(name, str) and name.startswith("transformer_layer"):
                    mem_kind = getattr(current, "mem_kind", None)
                    if mem_kind == MemKind.TRANSFORMER:
                        return True

                children = getattr(current, "children", None)
                if isinstance(children, (list, tuple)):
                    stack.extend(children)
                elif children is not None:
                    stack.append(children)
            return False

        if _is_non_flattened(graph_root):
            raise RuntimeError(
                "Memory simulation requires a flattened graph. "
                "Use LLMExecutionDispatcher.build_flattened_root_for_memory()."
            )
        return self.time_calc._simulate_with_memory(
            graph_root,
            memory_data,
            mode=mode,
            filename=filename,
        )


def estimate_inference_memory(
    exp_hw_config,
    exp_model_config,
    *,
    mode: str = "LLM",
    output_dir: Optional[str] = None,
    **overrides,
) -> Dict[str, Any]:
    return llm_util.estimate_inference_memory(
        exp_hw_config,
        exp_model_config,
        mode=mode,
        output_dir=output_dir,
        **overrides,
    )


def estimate_training_memory(
    exp_hw_config,
    exp_model_config,
    *,
    mode: str = "LLM",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    from train_timing import TimeCalculationLLM

    tc = TimeCalculationLLM(exp_hw_config, exp_model_config, mode, output_dir=output_dir)
    return tc.estimate_memory_only()


def estimate_memory(
    exp_hw_config,
    exp_model_config,
    *,
    mode: str = "LLM",
    output_dir: Optional[str] = None,
    **overrides,
) -> Dict[str, Any]:
    run_type = str(getattr(exp_model_config.model_config, "run_type", "training")).lower()
    if run_type == "inference":
        return estimate_inference_memory(
            exp_hw_config,
            exp_model_config,
            mode=mode,
            output_dir=output_dir,
            **overrides,
        )
    return estimate_training_memory(
        exp_hw_config,
        exp_model_config,
        mode=mode,
        output_dir=output_dir,
    )
