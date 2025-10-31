import math
import os
import json
from enum import Enum
from typing import Any, Dict, Tuple, Optional, List
import simulate_LLM
from LLM_execution import ExecutionMode, LLMExecutionDispatcher
from simulate_LLM import Graph
import LLM_util
from time_calculation import TimeCalculation
from itertools import zip_longest  # for element-wise aggregation of memory access lists

def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}

    
class ParallelismMode(Enum):
    TENSOR = "tensor"
    TENSOR_SEQUENCE = "tensor_sequence"
    CONTEXT = "context"
    TENSOR_CONTEXT_HYBRID = "tensor_context_hybrid"
    SINGLE = "single"

# TODO: verify all 4 of these
SWIGLU_SILU_FORWARD_FLOPS_PER_ELEMENT = 10
SWIGLU_SILU_BACKWARD_FLOPS_PER_ELEMENT = 20
GELU_FORWARD_FLOPS_PER_ELEMENT = 10
GELU_BACKWARD_FLOPS_PER_ELEMENT = 20
LAYER_NORM_FORWARD_FLOPS_PER_ELEMENT = 7
LAYER_NORM_BACKWARD_FLOPS_PER_ELEMENT = 14
LAYER_NORM_FORWARD_MEM_ACCESSES = 2
LAYER_NORM_BACKWARD_MEM_ACCESSES = 4
SOFTMAX_FORWARD_FLOPS_PER_ELEMENT = 7 # exponentiation 4FLOPS, subtract max 1FLOPS, dropout 2FLOPS
SOFTMAX_FORWARD_MEM_ACCESSES = 4
SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT = 4 
SOFTMAX_BACKWARD_MEM_ACCESSES = 3

# Map each parallelism mode to operation-level collective specs used across the
# metadata pipeline. Each spec records the collective kind, the participant
# scope (tp/cp/seq/etc.), the interconnect label, and an identifying suffix.
COMM_RULE_DEFAULT_KEY = "__default__"
COMMUNICATION_RULES: Dict[
    ParallelismMode, Dict[str, Dict[str, Optional[Dict[str, str]]]]
] = {
    ParallelismMode.TENSOR: {
        COMM_RULE_DEFAULT_KEY: {
            'forward': {'kind': 'all_reduce', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_reduce'},
            'backward': {'kind': 'all_reduce', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_reduce'},
        },
    },
    ParallelismMode.TENSOR_SEQUENCE: {
        COMM_RULE_DEFAULT_KEY: {'forward': None, 'backward': None},
        'layernorm1': {
            'forward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
            'backward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
        },
        'layernorm2': {
            'forward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
            'backward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
        },
        'MHA': {
            'forward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
            'backward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
        },
        'MLP': {
            'forward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
            'backward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
        },
    },
    ParallelismMode.CONTEXT: {
        COMM_RULE_DEFAULT_KEY: {
            'forward': {'kind': 'all_gather', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'all_gather'},
            'backward': {'kind': 'reduce_scatter', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'reduce_scatter'},
        },
        'attention': {'forward': None, 'backward': {'kind': 'reduce_scatter', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'reduce_scatter'}},
        'output_proj': {'forward': None, 'backward': {'kind': 'all_gather', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'all_gather'}},
        'qkv_proj': {'forward': {'kind': 'all_gather', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'all_gather'}, 'backward': None},
    },
    ParallelismMode.TENSOR_CONTEXT_HYBRID: {
        COMM_RULE_DEFAULT_KEY: {'forward': None, 'backward': None},
        'layernorm1': {
            'forward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
            'backward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
        },
        'layernorm2': {
            'forward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
            'backward': {'kind': 'all_gather', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'all_gather'},
        },
        'MLP': {
            'forward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
            'backward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
        },
        'attention': {'forward': None, 'backward': {'kind': 'reduce_scatter', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'reduce_scatter'}},
        'output_proj': {
            'forward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'cp', 'suffix': 'reduce_scatter'},
            'backward': {'kind': 'all_gather', 'participants': 'cp', 'interconnect': 'cp', 'suffix': 'all_gather'},
        },
        'qkv_proj': {
            'forward': {'kind': 'all_gather', 'participants': 'cp', 'interconnect': 'tp', 'suffix': 'all_gather'},
            'backward': {'kind': 'reduce_scatter', 'participants': 'tp', 'interconnect': 'tp', 'suffix': 'reduce_scatter'},
        },
    },
    ParallelismMode.SINGLE: {
        COMM_RULE_DEFAULT_KEY: {'forward': None, 'backward': None},
    },
}
class GemmType(Enum):
    ATTENTION_SCORE = "attention_score"
    ATTENTION_OUTPUT = "attention_output"
    QKV = "qkv"
    OUT_PROJ = "out_proj"
    FFN1 = "ffn1"
    FFN2 = "ffn2"
    LINEAR_SOFTMAX = "linear_softmax"


class TimeCalculationLLM(TimeCalculation):
    def __init__(self, hw_config, model_config, mode, output_dir: Optional[str] = None):
# Mode parameter
        execution_mode = self._derive_execution_mode(hw_config)
        astra_policy = self._map_execution_mode_to_policy(execution_mode)

        super().__init__(
            hw_config,
            model_config,
            mode,
            astra_policy_override=astra_policy,
        )
        if self.num_workers > 1 and self.use_moe:
            raise ValueError("MoE is only supported for single-worker runs (system_hierarchy.num_workers must be 1).")
          
        self.output_dir = os.path.abspath(output_dir) if output_dir else os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
        self._generate_graphs = _env_flag("DEEPFLOW_VISUALIZE_GRAPHS")
        self.persist_astrasim_artifacts = _env_flag("DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS")
        self._debug_memory = _env_flag("DEEPFLOW_DEBUG_MEMORY")
        self._memory_breakdown_debug = None
        self.execution_mode = execution_mode
        inference_cfg = getattr(hw_config, "inference_config", None)

        self.all_reduce = "every layer"
        self.model_type = self.model.model_type
        self.tied_embeddings = getattr(self.model, "tied_embeddings", True)

        self.memory_capacity_exceeded = False
        self.memory_capacity_violation_gb = 0.0
        self.zero3_ephemeral_peak_bytes = 0.0
        self.pipeline_graph: Optional[Graph] = None
        self.pipeline_root: Optional[Any] = None
        self.pipeline_interconnect: Optional[Dict[str, Tuple[float, float]]] = None
        self.transformer_graph: Optional[Graph] = None
        self.transformer_forward_root: Optional[Any] = None
        self.transformer_backward_root: Optional[Any] = None
        self.transformer_analytical_time_forward: Optional[float] = None
        self.transformer_analytical_time_backward: Optional[float] = None
        self.transformer_astrasim_time_forward: Optional[float] = None
        self.transformer_astrasim_time_backward: Optional[float] = None
        self.transformer_astrasim_per_rank_forward: Optional[List[float]] = None
        self.transformer_astrasim_per_rank_backward: Optional[List[float]] = None
        self.pipeline_astrasim_time: Optional[float] = None
        self.pipeline_astrasim_per_rank: Optional[List[float]] = None

    def _sequence_parallel_degree(self) -> int:
        """Return tensor-parallel degree used for sequence-parallel collectives.
            return 1 if no sequence parallelism is used.
            return tp if tp-sp is True and no context parallelism is used
            return cp if context parallelism is used
        """
        if self.tp_sp and self.cp == 1: #tensor parallelism only
            return self.tp
        elif self.cp > 1: #context parallelism or cp-tp hybrid parallelism
            return self.cp
        else: 
            return 1

    def get_parallelism_mode(self):
        if self.tp_sp and self.tp > 1 and self.cp == 1:
            return ParallelismMode.TENSOR_SEQUENCE
        elif self.tp > 1 and self.cp == 1:
            return ParallelismMode.TENSOR
        elif self.cp > 1 and self.tp == 1:
            return ParallelismMode.CONTEXT
        elif self.cp > 1 and self.tp > 1:
            return ParallelismMode.TENSOR_CONTEXT_HYBRID
        else:
            return ParallelismMode.SINGLE

    def _param_stats_per_rank(
        self,
        hidden_dim: int,
        intermediate_size: int,
        vocab_size: int,
    ) -> Tuple[float, float, float, float, float]:
        """Return detailed per-rank parameter counts used for ZeRO modeling."""

        tp = max(1, self.tp)
        lp = max(1, self.lp)

        ffn_proj_factor = 3 if str(self.model_type).lower() == "llama" else 2
        transformer_param_layer = 4 * hidden_dim * hidden_dim + intermediate_size * ffn_proj_factor * hidden_dim
        params_per_layer_per_rank = transformer_param_layer / tp

        total_transformer_params = params_per_layer_per_rank * self.num_layers
        if lp == 1:
            transformer_params_local = total_transformer_params
        else:
            layers_per_stage = math.ceil(self.num_layers / lp)
            transformer_params_local = params_per_layer_per_rank * layers_per_stage
            transformer_params_local = min(transformer_params_local, total_transformer_params)

        embedding_params = vocab_size * hidden_dim / tp
        output_params = 0.0 if self.tied_embeddings else embedding_params

        if lp == 1:
            total_params_per_rank = transformer_params_local + embedding_params + output_params
        else:
            # Pipeline splits embeddings/output across stages. Approximate their contribution by spreading across stages.
            total_params_per_rank = transformer_params_local + (embedding_params / lp) + (output_params / lp)

        max_layer_params = max(params_per_layer_per_rank, embedding_params, output_params)
        return (
            total_params_per_rank,
            max_layer_params,
            params_per_layer_per_rank,
            embedding_params,
            output_params,
        )

    def get_kv_size_bytes(self) -> int:
        """Return the total size in bytes of the KV cache."""
        total_elements = 2 * self.seq_len * self.microB * self.hidden_dim / self.num_heads * self.kv_heads
        return total_elements * self.precision.kv_cache

    @staticmethod
    def _derive_execution_mode(hw_config) -> ExecutionMode:
        backend = getattr(hw_config, "execution_backend", None)
        if not backend or getattr(backend, "model", "analytical").lower() != "astra":
            return ExecutionMode.ANALYTICAL

        mode_str = "hybrid"
        astra_cfg = getattr(backend, "astra", None)
        if astra_cfg and getattr(astra_cfg, "mode", None):
            mode_str = str(astra_cfg.mode).lower()

        for candidate in ExecutionMode:
            if candidate.value == mode_str:
                return candidate
        print(f"[WARN] Unknown execution mode '{mode_str}', defaulting to 'hybrid'.")
        return ExecutionMode.HYBRID

    @staticmethod
    def _map_execution_mode_to_policy(mode: ExecutionMode) -> str:
        if mode == ExecutionMode.ANALYTICAL:
            return 'analytical'
        if mode in (
            ExecutionMode.FULL_ASTRASIM_HIERARCHICAL,
            ExecutionMode.FULL_ASTRASIM_FLATTENED,
        ):
            return 'full'
        # Hybrid still uses the analytical pipeline, so keep the hybrid policy.
        return 'hybrid'

    @staticmethod
    def _expand_gemm_descriptor(gemm: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        if len(gemm) == 3:
            return 1, gemm[0], gemm[1], gemm[2]
        if len(gemm) == 4:
            return gemm[0], gemm[1], gemm[2], gemm[3]
        raise ValueError(f"Unsupported GEMM descriptor length: {len(gemm)}")

    @classmethod
    def _effective_dims(cls, gemm: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        batch, m, k, n = cls._expand_gemm_descriptor(gemm) 
        return batch, batch * m, k, n
    def _ffn1_output_dim(self, intermediate_size: int) -> int:
        return 2 * intermediate_size if self.model_type == "llama" else intermediate_size
    # def sequence
    def get_tensor_reduction_time(self, total_bytes: int, kind: str, name: str, participants: Optional[int] = None) -> float:
        """Return collective time for tensor-parallel reductions.
           receives total_bytes, which is the total size of the data to be reduced across all participants.
        """
        if not total_bytes:
            return 0.0

        if not participants:
            participants = int(self.tp)

        reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.IBTP,
            ll=self.LLTP,
            local_bytes=0,
            debug_label=name or "comm",
        )
        return reduction_time

    def flash_attention_kernel_forward(self, batch_size, hidden_dim, seq_len, num_heads, kv_heads, num_SMs) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Return time for flash attention kernel."""
        
        shard_seq = math.ceil(seq_len / max(1, self.cp)) 
        d = hidden_dim // num_heads # gemm shape for one head is (seq_len, d) x (d, seq_len)
        Bc = self.attention_tile_size # kv tile size
        Br = min(Bc, d) # q tile size
        Tr = math.ceil(shard_seq / Br) #number of q tiles
        Tc =  math.ceil(shard_seq / Bc) #number of kv tiles
        

        attention_forward_reduction_time = 0
        attention_size_f = 0
        # assuming key and value are preloaded into shared memory before attention computation
        load_kv_bytes = Bc * d * 2 * num_SMs * self.precision.activations #load key and value for one tile from HBM to SRAM
        initial_load_time = self.roofline(0, load_kv_bytes, "flash_attention_initial_load", mem_level=self.num_levels - 1) #assume key and value of one attention head is loaded from HBM to SRAM
        initial_load_time_per_head = initial_load_time * math.ceil(Tc/ num_SMs)
        
        # attention score gemm
        load_q_bytes = Br * d * self.precision.activations #load query for one tile assuming k is already in shared memory
        attn_score_time_per_tile = self.getGEMMTime(Br, d, Bc, "attention_score_f",read_bytes_l2=load_q_bytes, flashattn_enable=True)[0] 
        attn_score_time = attn_score_time_per_tile * Tc * Tr #attention score gemm time for one head

        # Softmax time
        elements = Br * Bc
        flops = SOFTMAX_FORWARD_FLOPS_PER_ELEMENT * elements  
        attn_scale_softmax_time = self.roofline(flops, 1, "attention_scale_softmax", mem_level=self.num_levels - 1) * Tc * Tr #use roofline model for softmax time with no memory access, memory access set to 1 because roofline does not accept 0 memory access
        
        # attention output gemm
        output_bytes = Br * d * self.precision.activations #load value for one tile S is already in shared memory
        attn_output_time_per_tile = self.getGEMMTime(Br, Bc, d, "attention_output", read_bytes_l2=output_bytes, write_bytes_l2=output_bytes, flashattn_enable=True)[0] 
        attn_output_time = attn_output_time_per_tile * Tc * Tr #attention output gemm time for one head
        
        load_time = initial_load_time_per_head * batch_size * kv_heads / max(1, self.tp) #load time for key and value are needed once per kv head
        attn_score_time *= batch_size * num_heads / max(1, self.tp) + self.O
        attn_scale_softmax_time *= batch_size * num_heads / max(1, self.tp)
        attn_output_time *= batch_size * num_heads / max(1, self.tp) + self.O
        
        attention_forward_gemm_time = load_time + attn_score_time + attn_scale_softmax_time + attn_output_time
        attention_forward_time = attention_forward_gemm_time + attention_forward_reduction_time

        # HBM traffic consists of only reading Q, K, V once and writing output once
        attention_mem = 2 * seq_len * (hidden_dim + d * kv_heads) * self.precision.activations
        
        return attention_forward_time, attention_forward_gemm_time, attention_forward_reduction_time, attention_size_f, attention_mem
    
    
    def flash_attention_kernel_backward(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        #TODO: should be repleced with a modified version of getgemmtime that considers flash attention tiling
        """Return time for flash attention kernel."""
        #FIXME not finished
        #TODO gqa support
        batch_size = self.microB
        seq_len = self.seq_len
        hidden_dim = self.hidden_dim
        num_heads = self.num_heads
        shard_seq = math.ceil(seq_len / max(1, self.cp)) 
        d = hidden_dim // num_heads #gemm shape for one head is (seq_len, d) x (d, seq_len)
        Bc = self.attention_tile_size
        Br = min(Bc, d)
        Tr = math.ceil(shard_seq / Br) #number of row tiles
        Tc =  math.ceil(shard_seq / Bc) #number of column tiles
        print(f"Flash Attention: Br={Br}, Tr={Tr}, d={d}, Bc={Bc}, Tc={Tc}")
        #TODO load 3 tiled gemm inputs and 1 output from HBM to SRAM
        
        #TODO disable HBM access
        attention_forward_reduction_time = 0
        attention_size_f = 0
        
        attn_score_time = self.getGEMMTime(Br, d, Bc, "attention_score_f")[0] * Tc * Tr #attention score gemm time for one head
        number_flops =  Br * Bc * d * Tc * Tr * batch_size * num_heads
        print(f"Flash Attention: attention score gemm flops={number_flops/1e9} GFLOPS")
         
        # Softmax time
        elements = Br * Bc
        flops = SOFTMAX_FORWARD_FLOPS_PER_ELEMENT * elements  # exponentiation + subtract max + dropout
        attn_scale_softmax_time = self.roofline(flops, 1, "attention_scale_softmax", mem_level=self.num_levels - 1) * Tc * Tr #use roofline model for softmax time with no memory access, memory access set to 1 because roofline does not accept 0 memory access
        attn_output_time = self.getGEMMTime(Br, Bc, d, "attention_output")[0] * Tc * Tr #attention output gemm time for one head
        attn_score_time *= batch_size * num_heads / max(1, self.tp)
        attn_scale_softmax_time *= batch_size * num_heads / max(1, self.tp)
        attn_output_time *= batch_size * num_heads / max(1, self.tp)
        
        attention_forward_gemm_time = attn_score_time + attn_scale_softmax_time + attn_output_time
        
        attention_forward_time = attention_forward_gemm_time + attention_forward_reduction_time
        attention_size_f = 0
        
    
        return attention_forward_time, attention_forward_gemm_time, attention_forward_reduction_time, attention_size_f
        
        
    @staticmethod
    def _normalize_gemm_type(gemm_type: Optional[GemmType]) -> Optional[GemmType]:
        if gemm_type is None or isinstance(gemm_type, GemmType):
            return gemm_type
        raise TypeError(f"Unsupported gemm type specifier: {gemm_type!r}")
    
    # assuming no context parallelism for now
    def parallelism_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Any:
        parallelism_mode = self.get_parallelism_mode()
        if parallelism_mode == ParallelismMode.TENSOR or parallelism_mode == ParallelismMode.TENSOR_SEQUENCE:
            return self._tensor_parallelism_gemm_forward(gemm, name, gemm_type) # also return flops and mem accesses
        elif parallelism_mode == ParallelismMode.CONTEXT:
            return self._context_parallelism_gemm_forward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            return self._tensor_context_hybrid_gemm_forward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.SINGLE:
            return self.single_gpu_gemm_forward(gemm, name, gemm_type) # also return flops and mem accesses
        else:
            raise ValueError(f"Unsupported parallelism mode: {parallelism_mode}")
        
    def parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Any:
        parallelism_mode = self.get_parallelism_mode()
        if parallelism_mode == ParallelismMode.TENSOR or parallelism_mode == ParallelismMode.TENSOR_SEQUENCE:
            return self._tensor_parallelism_gemm_backward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.CONTEXT:
            return self._context_parallelism_gemm_backward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            return self._tensor_context_hybrid_gemm_backward(gemm, name, gemm_type)

        elif parallelism_mode == ParallelismMode.SINGLE:
            return self.single_gpu_gemm_backward(gemm, name, gemm_type)
        else:
            raise ValueError(f"Unsupported parallelism mode: {parallelism_mode}")
        
    def single_gpu_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        total_flops = 2 * batch * m * k * n
        mem_accesses = []
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):  # attention gemm
            gemm_time = self.getGEMMTime(m, k, n, name, disable_overhead=True)[0] * batch + self.O
        elif (gemm_type == GemmType.FFN1 or gemm_type == GemmType.FFN2) and self.use_moe:
            gemm_time,_,_, mem_accesses = self.getGEMMTime(m, k, n, name)
            gemm_time *= self.moe_num_experts
        else :
            gemm_time,_,_, mem_accesses = self.getGEMMTime(m, k, n, name)
        return gemm_time, 0, 0, total_flops, mem_accesses
    
    def single_gpu_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):  # attention gemm
            grad_time_act = self.getGEMMTime(m, k, n, name, disable_overhead=True)[0] * batch + self.O
            grad_time_wt = self.getGEMMTime(k, m, n, name, disable_overhead=True)[0] * batch + self.O
        elif (gemm_type == GemmType.FFN1 or gemm_type == GemmType.FFN2) and self.use_moe:
            grad_time_act = self.getGEMMTime(m, n, k, name)[0] * self.moe_num_experts
            grad_time_wt = self.getGEMMTime(k, m, n, name)[0] * self.moe_num_experts
        else :
            grad_time_act = self.getGEMMTime(m, n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, m, n, name)[0]
        gemm_time = grad_time_act + grad_time_wt
        return gemm_time, 0, 0
        
    def _tensor_context_hybrid_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        """
        Megatron-LM style TPxCP hybrid forward GEMM behavior.

        Sharding:
        • CP shards the token (sequence) dimension M: shard_m = ceil(M / cp).
        • TP shards either K (row-wise) or N (column-wise), depending on gemm_type:
            - Column-wise (split N): QKV, FFN1, LINEAR_SOFTMAX 
            - Row-wise   (split K): OUT_PROJ, FFN2

        Per-op compute & communication:
        • ATTENTION_SCORE / ATTENTION_OUTPUT:
            -attention gemm shape is (shard_m, k) x (k, n) for each head, should scale with batch * num_heads / tp, number of heads is already multiplied in batch dimension
            - No extra collective here (K/V movement is modeled in QKV).

        • QKV (column-wise over N):
            -  CP all_gather(K,V) so each CP rank holds full-context K/V for attention.

        • OUT_PROJ, FFN2 (row-wise over K):
            - same as tensor parallelism forward gemm with row-wise sharding.:

        • FFN1 (column-wise over N):
            - No TP/CP collective in forward same as in tensor parallelism.

        """
        batch, m, k, n = self._expand_gemm_descriptor(gemm)

        participants = 0
        total_bytes = 0
        reduction_time = 0

        shard_m = math.ceil(m / max(1, self.cp))
        shard_k = math.ceil(k / max(1, self.tp))
        shard_n = math.ceil(n / max(1, self.tp))
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-context hybrid forward GEMM")
        
        if gemm_type == GemmType.ATTENTION_SCORE:  # attention gemm
            gemm_time = self.getGEMMTime(shard_m, k, n, name, disable_overhead=True)[0] * batch / max(1, self.tp) + self.O
        elif gemm_type == GemmType.ATTENTION_OUTPUT:  # attention gemm
            gemm_time = self.getGEMMTime(shard_m, k, n, name, disable_overhead=True)[0] * batch / max(1, self.tp) + self.O
        elif gemm_type == GemmType.QKV:  # column wise
            gemm_time = self.getGEMMTime(shard_m, k, shard_n, name)[0]
            total_bytes = self.get_kv_size_bytes()  / self.tp # each tp group holds a tp shard of kv for each cp group
            kind = "all_gather"
            participants = self.cp # all gather K V for each cp group
        elif gemm_type == GemmType.OUT_PROJ:
            gemm_time = self.getGEMMTime(shard_m, shard_k, n, name)[0]
            total_bytes = math.ceil(self.precision.activations * shard_m * n)
            kind = "reduce_scatter"
            participants = self.tp # reduce scatter output activation for each tp group
        elif gemm_type == GemmType.FFN2:  # row wise
            gemm_time = self.getGEMMTime(shard_m, shard_k, n, name)[0]
            total_bytes = math.ceil(self.precision.activations * shard_m * n)
            kind = "reduce_scatter"
            participants = self.tp  # reduce scatter output activation for each tp group
        elif gemm_type == GemmType.FFN1:  # column wise
            gemm_time = self.getGEMMTime(shard_m, k, shard_n, name)[0]
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        
        if total_bytes > 0:
            reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.IBTP,
            ll=self.LLTP,
            local_bytes=0,
            debug_label=name or "comm",
        )
        return gemm_time, reduction_time, total_bytes
    def _tensor_context_hybrid_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        """
        Megatron-LM style TPxCP hybrid backward GEMM behavior.

        Sharding:
        • CP shards tokens (M): shard_m = ceil(M / cp).
        • TP shards either N (column-wise) or K (row-wise) per gemm_type:
            - Column-wise (split N): QKV, FFN1, LINEAR_SOFTMAX (vocab proj)
            - Row-wise   (split K): OUT_PROJ, FFN2
            - Attention inner GEMMs: per-head work scaled by 1/tp.

        Backward rules (compute + comm):
        • Column-wise (QKV / FFN1 / LINEAR_SOFTMAX):
            - dX = dY_i @ W_i^T  → shape [shard_m, K]  → TP reduction on dX (sum):
                use reduce-scatter on TP (keeps token sharding), 
            - dW_i = X^T @ dY_i  → local, no TP comm
            - (QKV only) in CP: dK, dV must be reduce-scattered on CP back to token owners.
                Bytes ≈ get_kv_size_bytes() / tp (TP shard per CP group).

        • ATTENTION_SCORE / ATTENTION_OUTPUT:
            - Backward GEMMs are local on token shard with 1/tp scaling across heads.
            - CP/TP collectives handled in QKV branch (for K/V only), thus no comm here.
        """
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        # size_bytes = 0
        participants = 0
        total_bytes = 0
        reduction_time = 0

        shard_m = math.ceil(m / max(1, self.cp))
        shard_k = math.ceil(k / max(1, self.tp))
        shard_n = math.ceil(n / max(1, self.tp))
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-context hybrid backward GEMM")
        
        if gemm_type == GemmType.ATTENTION_SCORE:  # attention gemm
            grad_time_act = self.getGEMMTime(shard_m, n, k, name, disable_overhead=True)[0] * batch / max(1, self.tp) + self.O
            grad_time_wt = self.getGEMMTime(k, shard_m, n, name, disable_overhead=True)[0] * batch / max(1, self.tp) + self.O
            total_bytes = self.precision.grad_communication * k * n * batch * 2 / self.tp # weight gradient of K V need to be reduce scattered  *2 account for both attn key and value
            kind = "reduce_scatter"
            participants = self.cp
        elif gemm_type == GemmType.ATTENTION_OUTPUT:  # attention gemm
            grad_time_act = self.getGEMMTime(shard_m, n, k, name)[0] * batch / max(1, self.tp)
            grad_time_wt = self.getGEMMTime(k, shard_m, n, name)[0] * batch / max(1, self.tp)
        elif gemm_type == GemmType.QKV:  # column wise
            grad_time_act = self.getGEMMTime(shard_m, shard_n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, shard_m, shard_n, name)[0]
            total_bytes = math.ceil(self.precision.grad_communication * shard_m * k)
            kind = "reduce_scatter"
            participants = self.tp
        elif gemm_type == GemmType.OUT_PROJ:
            grad_time_act = self.getGEMMTime(shard_m, n, shard_k, name)[0]
            grad_time_wt = self.getGEMMTime(shard_k, shard_m, n, name)[0]
            total_bytes = self.get_kv_size_bytes() / self.tp
            kind = "all_gather"
            participants = self.cp
        elif gemm_type == GemmType.FFN2:  # row wise
            grad_time_act = self.getGEMMTime(shard_m, n, shard_k, name)[0]
            grad_time_wt = self.getGEMMTime(shard_k, shard_m, n, name)[0]
        elif gemm_type == GemmType.FFN1:  # column wise
            
            grad_time_act = self.getGEMMTime(shard_m, shard_n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, shard_m, shard_n, name)[0]
            total_bytes = math.ceil(self.precision.grad_communication * shard_m * k)
            kind = "reduce_scatter"
            participants = self.tp
        elif gemm_type == GemmType.LINEAR_SOFTMAX:
            grad_time_act = self.getGEMMTime(shard_m, n, shard_k, name)[0]
            grad_time_wt = self.getGEMMTime(shard_k, shard_m, n, name)[0]
            total_bytes = math.ceil(self.precision.grad_communication * shard_m * shard_k) * self.cp * self.tp # in tp-cp hybrid parallelism, the linear softmax weight is sharded by both tp and cp
            kind = "all_gather"
            participants = self.cp * self.tp
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        gemm_time = grad_time_act + grad_time_wt
        if total_bytes > 0:
            reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.IBTP,
            ll=self.LLTP,
            local_bytes=0,
            debug_label=name or "comm",
        )
        return gemm_time, reduction_time, total_bytes
    def _tensor_parallelism_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        """
        Tensor-parallel forward GEMM behavior.

        • Multi-Head Attention (MHA):
          - With tensor parallelism, the attention computation is sharded along the head dimension,
            so each TP rank handles a subset of heads. No communication is needed for
            ATTENTION_SCORE / ATTENTION_OUTPUT GEMMs in the forward pass.
            attention gemm time is scaled by batch / tp

        • After attention, before feeding MLP:
          - If TP-only, we all-reduce to gather the full attention output.
          - If TP+SP (sequence parallel enabled), each TP rank keeps only a shard of the sequence
            after layernorm, so we use reduce-scatter instead of all-reduce.
            
        • Sharding rules by gemm_type:
          - QKV, FFN1: **column-wise sharding** (split along output dimension N).
            Each TP rank produces its local output columns independently - no communication needed.
          - OUT_PROJ, FFN2: **row-wise sharding** (split along input dimension K).
            Each TP rank computes partial sums that must be combined across ranks via
            all-reduce or reduce-scatter.
          - LINEAR_SOFTMAX (final logits projection): **column-wise sharding**.
            The output projection weight [hidden_dim, vocab_size] is split by vocab dimension.
            Each TP rank computes logits for its vocab slice, and results are all-gathered
        """
        tp_mode = self.get_parallelism_mode()
        if gemm_type == GemmType.LINEAR_SOFTMAX:
            comm_kind_fwd = "all_gather"
        else:
            comm_kind_fwd = "all_reduce" if tp_mode == ParallelismMode.TENSOR else "reduce_scatter"
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        size_bytes = 0
        total_bytes = 0
        reduction_time = 0
        total_flops = 2 * batch * m * k * n
        mem_accesses = []
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-parallel forward GEMM")
        
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):  # attention gemm
            gemm_time,_,_, mem_accesses = self.getGEMMTime(m, k, n, name, disable_overhead=True)
            gemm_time = gemm_time * batch / max(1, self.tp) + self.O
        elif gemm_type in (GemmType.FFN1, GemmType.FFN2) and self.use_moe:
            # MoE FFN layers replicate experts across all TP ranks, so no TP sharding here.
            gemm_time,_,_, mem_accesses = self.getGEMMTime(m, k, n, name)
            gemm_time *= self.moe_num_experts
        elif gemm_type in (GemmType.QKV, GemmType.FFN1):  # column wise
            shard_n = math.ceil(n / max(1, self.tp))
            gemm_time,_,_, mem_accesses = self.getGEMMTime(m, k, shard_n, name)
        elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):  # row wise
            shard_k = math.ceil(k / max(1, self.tp))
            gemm_time,_,_, mem_accesses = self.getGEMMTime(m, shard_k, n, name)
            size_bytes = math.ceil(self.precision.activations * m * n)
            participants = self.tp
        elif gemm_type == GemmType.LINEAR_SOFTMAX: #assuming linear softmax is always column wise sharded
            shard_n = math.ceil(n / max(1, self.tp * self.cp ))
            gemm_time,_,_, mem_accesses = self.getGEMMTime(m, k, shard_n, name)
            size_bytes = math.ceil(self.precision.activations * m * n)
            participants = self.tp * self.cp
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
            
        if size_bytes > 0:
            total_bytes = size_bytes #we already has the totol bytes for all reduce not bytes per rank
            reduction_time = self.get_tensor_reduction_time(total_bytes, kind=comm_kind_fwd, participants=participants, name=name)


        return gemm_time, reduction_time, total_bytes, total_flops, mem_accesses
    
    def _tensor_parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None, comm_after: bool = False) -> Tuple[float, float]:
        """
        Tensor-parallel backward GEMM behavior.

        We model the time for two backward GEMMs per op:
          • grad wrt activation (dX)
          • grad wrt weight (dW)

        • ATTENTION_SCORE / ATTENTION_OUTPUT
          - Per-rank work scales with batch/tp (each rank handles a subset of heads).

        • Column-wise sharded ops :
          - QKV, FFN1, and LINEAR_SOFTMAX:
            - Local backward GEMMs:
                dX:  [m, k]  via (dY_i @ W_i^T)
                dW:  [k, n/tp] via (X^T @ dY_i)
            - dX is a partial across ranks** → requires tensor reduction
              (all-reduce if TP-only; reduce-scatter if TP+SP).

        • Row-wise sharded ops :
          - OUT_PROJ, FFN2:
            - Local backward GEMMs:
                dX_i: [m, k/tp] via (dY @ W_i^T) - disjoint along K, no cross-rank sum
                dW_i: [k/tp, n] via (X_i^T @ dY)
            -  no tensor reduction on dX for row-wise in backward.

        """
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        seq_degree = self._sequence_parallel_degree()
        act_bytes = 0
        total_bytes = 0
        if gemm_type == GemmType.LINEAR_SOFTMAX:
            comm_kind_bwd = "all_reduce"
        else:
            comm_kind_bwd = "all_reduce" if seq_degree == 1 else "reduce_scatter"
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-parallel backward GEMM")

        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):
            grad_time_act = self.getGEMMTime(m, n, k, name, disable_overhead=True)[0] * batch / max(1, self.tp) + self.O
            grad_time_wt = self.getGEMMTime(k, m, n, name, disable_overhead=True)[0] * batch / max(1, self.tp) + self.O
        elif gemm_type in (GemmType.QKV, GemmType.FFN1):  # column wise
            shard_n = math.ceil(n / max(1, self.tp))
            grad_time_act = self.getGEMMTime(m, shard_n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, m, shard_n, name)[0]
            act_bytes = math.ceil(self.precision.grad_communication * m * k)
            participants = self.tp
        elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):  # row wise
            shard_k = math.ceil(k / max(1, self.tp))
            grad_time_act = self.getGEMMTime(m, n, shard_k, name)[0]
            grad_time_wt = self.getGEMMTime(shard_k, m, n, name)[0]
        elif gemm_type == GemmType.LINEAR_SOFTMAX:
            shard_n = math.ceil(n / max(1, self.tp * self.cp))
            grad_time_act = self.getGEMMTime(m, shard_n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, m, shard_n, name)[0]
            act_bytes = math.ceil(self.precision.grad_communication * m * k)
            participants = self.tp * self.cp
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        gemm_time = grad_time_act + grad_time_wt
        reduction_time = 0
        if act_bytes > 0:
            total_bytes = act_bytes #total bytes for all reduce
            reduction_time = self.get_tensor_reduction_time(total_bytes, kind=comm_kind_bwd, participants=participants, name=name)


        return gemm_time, reduction_time, total_bytes
    def _context_parallelism_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        """
        Megatron-LM style context-parallel (CP) forward GEMM behavior.

        • CP shards the token (sequence) dimension M across cp ranks:
            shard_m = ceil(M / cp). Each rank processes a disjoint subset of tokens.
        • Each GEMM is performed locally on the rank’s token slice.
        • Communication is required only when gathering K and V for attention,
          since every rank must hold the full set of keys and values to compute
          attention scores.

        """

        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        total_bytes = 0
        reduction_time = 0
        shard_m = math.ceil(m / max(1, self.cp))
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for context-parallel forward GEMM")
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):  # attention gemm
            gemm_time = self.getGEMMTime(shard_m, k, n, name, disable_overhead=True)[0] * batch + self.O
        elif gemm_type == GemmType.QKV:  # qkv gemm
            gemm_time = self.getGEMMTime(shard_m, k, n, name)[0]
            total_bytes = self.get_kv_size_bytes()
        elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN1, GemmType.FFN2):
            gemm_time = self.getGEMMTime(shard_m, k, n, name)[0]
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        if gemm_type == GemmType.QKV:
            kind = "all_gather" 
            reduction_time = self.network_model.collective(
                kind=kind,
                size_bytes=total_bytes,
                participants=self.cp,
                ib=self.IBTP,
                ll=self.LLTP,
                local_bytes=0,
                debug_label=name or "comm",
            )

        return gemm_time, reduction_time, total_bytes

    def _context_parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None, comm_after: bool = False) -> Tuple[float, float]:

        """
        Megatron-LM style context-parallel (CP) backward GEMM behavior.

        Communication rules in CP (sequence-parallel) backward:
        • QKV projection:
            - Forward did an all-gather(K, V) across cp ranks.
            - Backward must return gradients to token owners:
                    dK, dV → reduce-scatter over cp ranks.
        • Output projection:
            - K V need to be gathered again to compute activation gradients.
        """
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        total_bytes = 0
        reduction_time = 0
        shard_m = math.ceil(m / max(1, self.cp))
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for context-parallel backward GEMM")
        if gemm_type == GemmType.ATTENTION_SCORE:
            grad_time_act = self.getGEMMTime(shard_m, n, k, name, disable_overhead=True)[0] * batch + self.O
            grad_time_wt = self.getGEMMTime(k, shard_m, n, name, disable_overhead=True)[0] * batch + self.O
            total_bytes = self.precision.grad_communication * k * n * batch * 2 # account for both Q and K
            kind = "reduce_scatter"
        elif gemm_type == GemmType.ATTENTION_OUTPUT:  # attention gemm
            grad_time_act = self.getGEMMTime(shard_m, n, k, name, disable_overhead=True)[0] * batch + self.O
            grad_time_wt = self.getGEMMTime(k, shard_m, n, name, disable_overhead=True)[0] * batch + self.O
        elif gemm_type in (GemmType.QKV, GemmType.FFN1, GemmType.FFN2):
            grad_time_act = self.getGEMMTime(shard_m, n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, shard_m, n, name)[0]
        elif gemm_type == GemmType.OUT_PROJ:
            grad_time_act = self.getGEMMTime(shard_m, n, k, name)[0]
            grad_time_wt = self.getGEMMTime(k, shard_m, n, name)[0]
            total_bytes = self.get_kv_size_bytes()
            kind = "all_gather"
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        gemm_time = grad_time_act + grad_time_wt
        if total_bytes > 0:
            reduction_time = self.network_model.collective(
                kind=kind,
                size_bytes=total_bytes,
                participants=self.cp,
                ib=self.IBTP,
                ll=self.LLTP,
                local_bytes=0,
                debug_label=name or "comm",
            )
        return gemm_time, reduction_time, total_bytes
    

                
    def get_embedding_f(self, vocab_size, seq_len, hidden_dim):
        """
        Calculates the total time required for embedding operations, including computation and data transfer.
        """
        batch = self._effective_transformer_batch()
        embedding_mem = vocab_size * hidden_dim * self.precision.activations + seq_len * batch * hidden_dim * self.precision.activations
        embedding_time = self.roofline(
            0,
            embedding_mem,
            name="embedding_f",
            mem_level=self.num_levels - 1,
        ) + self.O
        if self.h2d_bandwidth and self.h2d_bandwidth > 0:
            embedding_transfer_time = embedding_mem / self.h2d_bandwidth
        else:
            embedding_transfer_time = 0.0
        if self.debug:
            print(
                "Embedding_mem: {:,}, transfer_time: {:.6f}".format(
                    int(embedding_mem / 1e9), embedding_transfer_time
                )
            )
        return embedding_time + embedding_transfer_time, embedding_mem

    def get_linear_softmax_f(self, gemm):
        """Estimate time for final projection + softmax forward.
            assuming linear softmax gemm always use tensor parallelism sharded by vocab dimension
        """
        _, effective_m, k, n = self._effective_dims(gemm)

        gemm_time, reduction_time, size_bytes,_,_ = self._tensor_parallelism_gemm_forward(gemm, "linear_softmax_f", gemm_type=GemmType.LINEAR_SOFTMAX)

            
        elements = effective_m * n / (self.tp * self.cp) # each tp-cp group holds a shard of the vocab dimension
        point_flop = elements * SOFTMAX_FORWARD_FLOPS_PER_ELEMENT
        # TODO:
        # unsure if should be precision.activations or precision.stats
        point_mem = self.precision.activations * elements * SOFTMAX_FORWARD_MEM_ACCESSES 
        point_time = self.roofline(
            point_flop,
            point_mem,
            name="pointwise-linear-softmax-f",
            mem_level=self.num_levels - 1,
        ) + 4 * self.O

        if self.debug:
            print(
                "Linear Softmax (f) point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("Linear Softmax (f) point_time: {:,}\n".format(point_time))

        return gemm_time + reduction_time + point_time, point_mem
    
    def get_linear_softmax_b(self, gemm):


        _, effective_m, k, n = self._effective_dims(gemm)

        gemm_time, reduction_time, size_bytes = self._tensor_parallelism_gemm_backward(gemm, "linear_softmax_b", gemm_type=GemmType.LINEAR_SOFTMAX)
        elements = effective_m * n / (self.tp * self.cp) # each tp-cp group holds a shard of the vocab dimension
        point_flop = elements * SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT
        # TODO:
        # same here, unsure if should be precision.activations or precision.stats
        point_mem = self.precision.activations * elements * SOFTMAX_BACKWARD_MEM_ACCESSES

        point_time = self.roofline(
            point_flop,
            point_mem,
            name="pointwise-linear-softmax-b",
            mem_level=self.num_levels - 1,
        ) + 4 * self.O

        if self.debug:
            print(
                "Linear Softmax (b) point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("Linear Softmax (b) point_time: {:,}\n".format(point_time))

        return gemm_time + reduction_time + point_time
    def get_scale_softmax_f(self, gemm):
        """
        Estimate time for scale + softmax forward.
        total elements is divided by tp and cp since in tp-cp hybrid parallelism,
        """
        batch, m, _, n = self._expand_gemm_descriptor(gemm)
        elements = math.ceil(batch * m * n / (self.tp * self.cp))
        flops = elements * (SOFTMAX_FORWARD_FLOPS_PER_ELEMENT + 1)  # +1 for scaling
        mem = self.precision.activations * elements * (SOFTMAX_FORWARD_MEM_ACCESSES )  

        time = self.roofline(
            flops,
            mem,
            name="pointwise-scale-softmax-f",
            mem_level=self.num_levels - 1,
        ) + self.O

        return time
    
    def get_scale_softmax_b(self, gemm):
        batch, m, _, n = self._expand_gemm_descriptor(gemm)
        elements = math.ceil(batch * m * n / (self.tp * self.cp))
        flops = elements * (SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT + 1)  # +1 for scaling
        mem = self.precision.activations * elements * (SOFTMAX_BACKWARD_MEM_ACCESSES)  

        time = self.roofline(
            flops,
            mem,
            name="pointwise-scale_softmax-b",
            mem_level=self.num_levels - 1,
        ) +  self.O


        if self.debug:
            print(
                "Scale Softmax (b) flop: {:,}, mem: {:,}".format(
                    int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Scale Softmax(b) time: {:,}".format(time))


        return time 
        
    def get_residual_f(self, tensor_shape):
        # Residual operates on full tensor, not just GEMM output dimension
        # TODO: double check!
        batch, m, _, n = self._expand_gemm_descriptor(tensor_shape)
        elements = batch * m * n

        flops = 2 * elements  # add + bias
        mem = self.precision.activations * elements * 3  # read main, read residual, write out
        time = self.roofline(flops, mem, name="pointwise-residual-f", mem_level=self.num_levels - 1) + self.O

        if self.debug:
            print(
                "Residual (f) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Residual (f) time: {:,}\n".format(time))

        return time

    def get_residual_b(self, tensor_shape):
        # Residual operates on full tensor, not just GEMM output dimension
        # TODO: double check!
        batch, m, _, n = self._expand_gemm_descriptor(tensor_shape)
        elements = batch * m * n

        flops = elements  # dL/dx = dL/dy passthrough
        mem = self.precision.gradients * elements * 3  # read grad, read forward residual, write grad
        time = self.roofline(flops, mem, name="pointwise-residual-b", mem_level=self.num_levels - 1) + self.O

        if self.debug:
            print(
                "Residual (b) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Residual (b) time: {:,}\n".format(time))

        return time

    def get_gelu_f(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        compute_flops = elements * hidden * GELU_FORWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.activations * elements * hidden * 2  # read, write

        time = self.roofline(compute_flops, mem_bytes, name="pointwise-gelu-f", mem_level=self.num_levels - 1) + 2 * self.O

        if self.debug:
            print(
                "GELU (f) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(compute_flops / 1e9), int(mem_bytes / 1e9)
                )
            )
            print("GELU (f) time: {:,}\n".format(time))

        return time
    def get_gelu_b(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        compute_flops = elements * hidden * GELU_BACKWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.gradients * elements * hidden * 3  # read grad, read forward, write grad

        time = self.roofline(compute_flops, mem_bytes, name="pointwise-gelu-b", mem_level=self.num_levels - 1) + 3 * self.O

        if self.debug:
            print(
                "GELU (b) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(compute_flops / 1e9), int(mem_bytes / 1e9)
                )
            )
            print("GELU (b) time: {:,}\n".format(time))
        return time

    def get_swiglu_f(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        gate_hidden = max(hidden // 2, 1)
        compute_flops = elements * (
            gate_hidden * SWIGLU_SILU_FORWARD_FLOPS_PER_ELEMENT + gate_hidden
        )
        reads = 2 * gate_hidden  # gate and up activations
        writes = gate_hidden  # SwiGLU output
        mem_bytes = self.precision.activations * elements * (reads + writes)
        time = self.roofline(compute_flops, mem_bytes, name="pointwise-swiglu-f", mem_level=self.num_levels - 1) + 2 * self.O
        if self.debug:
            print(
                "SwiGLU (f) gate elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int((elements * gate_hidden) / 1e6),
                    int(compute_flops / 1e9),
                    int(mem_bytes / 1e9),
                )
            )
            print("SwiGLU (f) time: {:,}\n".format(time))
        return time

    def get_swiglu_b(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        gate_hidden = max(hidden // 2, 1)
        compute_flops = elements * (
            gate_hidden * SWIGLU_SILU_BACKWARD_FLOPS_PER_ELEMENT + 2 * gate_hidden
        )
        reads = 2 * gate_hidden  # gate and up activations
        reads += gate_hidden  # upstream gradient
        writes = 2 * gate_hidden  # gradients for gate and up projections
        mem_bytes = self.precision.gradients * elements * (reads + writes)
        time = self.roofline(compute_flops, mem_bytes, name="pointwise-swiglu-b", mem_level=self.num_levels - 1) + 3 * self.O
        if self.debug:
            print(
                "SwiGLU (b) gate elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int((elements * gate_hidden) / 1e6),
                    int(compute_flops / 1e9),
                    int(mem_bytes / 1e9),
                )
            )
            print("SwiGLU (b) time: {:,}\n".format(time))
        return time

    def get_layernorm_f(self, batch, seq_len, d_model, comm_after=False):
        tp_mode = self.get_parallelism_mode()
        seq_degree = self._sequence_parallel_degree()
        if tp_mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            elements = batch * math.ceil(seq_len / seq_degree) * d_model / self.tp
        elif tp_mode == ParallelismMode.TENSOR_SEQUENCE:
            elements = batch * math.ceil(seq_len / seq_degree) * d_model
        else:
            elements = batch * seq_len * d_model
        compute_flops = elements * LAYER_NORM_FORWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.stats * elements * LAYER_NORM_FORWARD_MEM_ACCESSES
        compute_time = self.roofline(
            compute_flops,
            mem_bytes,
            name="pointwise-layernorm-f",
            mem_level=self.num_levels - 1,
        ) + 3 * self.O
        if tp_mode in (ParallelismMode.TENSOR_SEQUENCE, ParallelismMode.TENSOR_CONTEXT_HYBRID):  # all-gather after layernorm
            per_rank_bytes = self.precision.stats * elements
            total_bytes = int(math.ceil(per_rank_bytes * self.tp))
            reduction_time = self.network_model.collective(
                kind="all_gather",
                size_bytes=total_bytes,
                participants=self.tp,
                ib=self.IBTP,
                ll=self.LLTP,
                local_bytes=0,
                debug_label="layernorm_f_all_gather",
            )
        else:
            reduction_time = 0.0
            total_bytes = 0

        return compute_time, reduction_time, total_bytes
    
    

    def get_layernorm_b(self, batch, seq_len, d_model, comm_after=False):
        tp_mode = self.get_parallelism_mode()
        seq_degree = self._sequence_parallel_degree()
        if tp_mode == ParallelismMode.TENSOR_CONTEXT_HYBRID:
            elements = batch * math.ceil(seq_len / seq_degree) * d_model / self.tp
        else:
            elements = batch * math.ceil(seq_len / seq_degree) * d_model
        compute_flops = elements * LAYER_NORM_BACKWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.stats * elements * LAYER_NORM_BACKWARD_MEM_ACCESSES

        compute_time = self.roofline(
            compute_flops,
            mem_bytes,
            name="pointwise-layernorm-b",
            mem_level=self.num_levels - 1,
        ) + 4 * self.O
        if tp_mode in (ParallelismMode.TENSOR_SEQUENCE, ParallelismMode.TENSOR_CONTEXT_HYBRID):  # communication after layernorm
            per_rank_bytes = self.precision.grad_communication * elements
            total_bytes = int(math.ceil(per_rank_bytes * self.tp))
            reduction_time = self.network_model.collective(
                kind="all_gather",
                size_bytes=total_bytes,
                participants=self.tp,
                ib=self.IBTP,
                ll=self.LLTP,
                local_bytes=0,
                debug_label="layernorm_b_all_gather",
            )

        else:
            reduction_time = 0.0
            total_bytes = 0


        return compute_time, reduction_time, total_bytes

    
    def get_embedding_b(self, vocab_size, seq_len, hidden_dim):
        batch = self._effective_transformer_batch()
        embedding_mem = vocab_size * hidden_dim * self.precision.gradients + seq_len * batch * hidden_dim * self.precision.gradients
        embedding_mem_time = self.roofline(
            0,
            embedding_mem,
            name="embedding_b",
            mem_level=self.num_levels - 1,
        ) + self.O

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_mem_time
    

    def get_inter_layer_comm_latency_llm(self, batch_size, hidden_dim, seq_len): #calculate the cross-layer communication latency
        w = 0
        w_size = 0
        if self.lp > 1:
            w_size = self.precision.activations * batch_size * hidden_dim * seq_len
            transfer_time = w_size / self.IBL + self.LLL
            mem_time = self.roofline(0, 2 * w_size, name="inter_layer", mem_level=self.num_levels - 1)
            # 2: read from memory of previous layer and write to the memory of the next layer
            w = mem_time + transfer_time
        return w, w_size

    def get_data_parallel_reduction_sizes(self, d, intermediate_size):
        """Calculate communication sizes for data parallel reductions (no timing)."""
        if not getattr(self, "dp", 1) or self.dp <= 1:
            # No communication needed for dp=1
            return {
                'qkv_size': 0,
                'output_size': 0,
                'ffn_size': 0,
                'total_size': 0
            }

        # Calculate sizes only
        qkv_size = math.ceil(self.precision.grad_communication * d * 3 * d)
        output_size = math.ceil(self.precision.grad_communication * d * d)
        ffn1_dim = self._ffn1_output_dim(intermediate_size)
        ffn1_size = math.ceil(self.precision.grad_communication * ffn1_dim * d)
        ffn2_size = math.ceil(self.precision.grad_communication * intermediate_size * d)
        total_size = qkv_size + output_size + ffn1_size + ffn2_size

        return {
            'qkv_size': qkv_size,
            'output_size': output_size,
            'ffn_size': ffn1_size + ffn2_size,
            'total_size': total_size
        }

    def get_data_parallel_local_computation(self, d, intermediate_size):
        """Calculate local computation times for apply_grad operations."""
        qkv_local = self.apply_grad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")
        output_local = self.apply_grad(Dim0=d, Dim1=d, name="output_proj reduction")
        ffn1_dim = self._ffn1_output_dim(intermediate_size)
        ffn_local = (
            self.apply_grad(Dim0=ffn1_dim, Dim1=d, name="ffn1 reduction")
            + self.apply_grad(Dim0=intermediate_size, Dim1=d, name="ffn2 reduction")
        )

        return {
            'qkv_local': qkv_local,
            'output_local': output_local,
            'ffn_local': ffn_local,
            'total_local': qkv_local + output_local + ffn_local
        }

    def get_data_parallel_reduction_llm(self, d, intermediate_size): 
        # If no data parallelism, still apply gradients locally but no cross-device reduction
        if self.dp <= 1:
            apply_grad_time = 0.0
            apply_grad_time += self.apply_grad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")
            apply_grad_time += self.apply_grad(Dim0=d, Dim1=d, name="output_proj reduction")
            ffn1_dim = self._ffn1_output_dim(intermediate_size)
            apply_grad_time += self.apply_grad(Dim0=ffn1_dim, Dim1=d, name="ffn1 reduction")
            apply_grad_time += self.apply_grad(Dim0=intermediate_size, Dim1=d, name="ffn2 reduction")
            if self.debug:
                print(f"(dp=1) apply_grad_time: {apply_grad_time}")
            return apply_grad_time

        reduction_sizes = self.get_data_parallel_reduction_sizes(d, intermediate_size)
        grad_bytes = int(reduction_sizes['total_size'])
        (
            total_params_per_rank,
            _,
            _,
            _,
            _,
        ) = self._param_stats_per_rank(d, intermediate_size, self.vocab_size)
        param_bytes = int(math.ceil(total_params_per_rank * self.precision.parameters))
        grad_shard = self.dp if self.zero_stage >= 2 else 1

        if grad_shard > 1: # ZeRO stages >= 2: gradients are sharded via reduce-scatter and parameters are gathered.
            apply_grad_time = 0.0
            apply_grad_time += self.apply_grad(Dim0=d, Dim1=3 * d, name="qkv_proj reduction")
            apply_grad_time += self.apply_grad(Dim0=d, Dim1=d, name="output_proj reduction")
            ffn1_dim = self._ffn1_output_dim(intermediate_size)
            apply_grad_time += self.apply_grad(Dim0=ffn1_dim, Dim1=d, name="ffn1 reduction")
            apply_grad_time += self.apply_grad(Dim0=intermediate_size, Dim1=d, name="ffn2 reduction")
            # divide by dp to get the time per device 
            apply_grad_time /= grad_shard

            reduction_time = self.network_model.collective(
                kind="reduce_scatter",
                size_bytes=grad_bytes,
                participants=int(self.dp),
                ib=self.IBD,
                ll=self.LLD,
                local_bytes=0.0,
                debug_label="dp_zero_gradients",
            )

            gather_time = 0.0
            if self.zero_stage == 2:
                gather_bytes = int(math.ceil(param_bytes))
                gather_time += self.network_model.collective(
                    kind="all_gather",
                    size_bytes=gather_bytes,
                    participants=int(self.dp),
                    ib=self.IBD,
                    ll=self.LLD,
                    local_bytes=0.0,
                    debug_label="dp_zero_param_gather_zero2",
                )
            if self.debug:
                print(f"apply_grad_time (ZeRO-{self.zero_stage}): {apply_grad_time}")
            # TODO(ZeRO): revisit optimizer FLOPs once ZeRO-3 materialization is finalized.
            return reduction_time + gather_time + apply_grad_time
        else: # legacy path for zero stages <= 2.
            reduction_time = 0.0
            apply_grad_time = 0.0

            total_bytes = math.ceil(self.precision.grad_communication * d * 3 * d)
            reduction_time += self.network_model.collective(
                kind="all_reduce",
                size_bytes=total_bytes,
                participants=int(self.dp),
                ib=self.IBD,
                ll=self.LLD,
                local_bytes=0.0,
                debug_label="qkv_proj reduction",
            )
            apply_grad_time += self.apply_grad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")

            total_bytes = math.ceil(self.precision.grad_communication * d * d)
            reduction_time += self.network_model.collective(
                kind="all_reduce",
                size_bytes=total_bytes,
                participants=int(self.dp),
                ib=self.IBD,
                ll=self.LLD,
                local_bytes=0.0,
                debug_label="output_proj reduction",
            )
            apply_grad_time += self.apply_grad(Dim0=d, Dim1=d, name="output_proj reduction")

            ffn1_dim = self._ffn1_output_dim(intermediate_size)
            total_bytes = math.ceil(self.precision.grad_communication * ffn1_dim * d)
            reduction_time += self.network_model.collective(
                kind="all_reduce",
                size_bytes=total_bytes,
                participants=int(self.dp),
                ib=self.IBD,
                ll=self.LLD,
                local_bytes=0.0,
                debug_label="ffn1 reduction",
            )
            apply_grad_time += self.apply_grad(Dim0=ffn1_dim, Dim1=d, name="ffn1 reduction")

            total_bytes = math.ceil(self.precision.grad_communication * intermediate_size * d)
            reduction_time += self.network_model.collective(
                kind="all_reduce",
                size_bytes=total_bytes,
                participants=int(self.dp),
                ib=self.IBD,
                ll=self.LLD,
                local_bytes=0.0,
                debug_label="ffn2 reduction",
            )
            apply_grad_time += self.apply_grad(Dim0=intermediate_size, Dim1=d, name="ffn2 reduction")

            if self.debug:
                print(f"apply_grad_time: {apply_grad_time}")

            return reduction_time + apply_grad_time
    
    def _combine_mem(self, *args):
            combined = {}
            for d in args:
                mem_levels = self._mem_levels(d)
                for k, v in mem_levels.items():
                    combined[k] = combined.get(k, 0) + v
            return combined
        
    def _mem_levels(self, arr):
        if isinstance(arr, (int, float)):
            return {f"L{i}": (arr if i == self.num_levels - 1 else 0) for i in range(self.num_levels)}
        return {f"L{i}": v for i, v in enumerate(arr or [])}

    # TODO TODO:
    # we need a significant refactor here. The comm sizes are ingested in a weird way and never used. Instead we use old precomputed sizes.
    # FIX at some point!
    def compute_all_gemm_and_node_times(self, batch_size, vocab_size, hidden_dim, seq_len, num_heads, kv_heads, intermediate_size, num_SMs):
        """Compute latency for all GEMM operations and node breakdown times in one function."""

        # Process GEMM shapes
        gemm_shapes = LLM_util.process_gemm_shapes(
            self, batch_size, seq_len, hidden_dim, num_heads, kv_heads, intermediate_size, vocab_size
        )
        if self.debug:
            print(
                "generating gemm shapes for transformer batch size:",
                batch_size,
                "seq_len:",
                seq_len,
                "hidden_dim:",
                hidden_dim,
                "num_heads:",
                num_heads,
                "kv_heads:",
                kv_heads,
                "intermediate_size:",
                intermediate_size,
            )
        gemm_qkv_proj = gemm_shapes["qkv_proj"]
        gemm_attention_score = gemm_shapes["attention_score"]
        gemm_attention_output = gemm_shapes["attention_output"]
        gemm_output_proj = gemm_shapes["output_proj"]
        gemm_ffn1 = gemm_shapes["ffn1"]
        gemm_ffn2 = gemm_shapes["ffn2"]
        gemm_linear = gemm_shapes["linear"]

        # Compute GEMM times and organize in structured dict
        transformer_results = {}

        # assuming no context-parallelism for now and only running inference

        # Helpers to handle variable-length returns (supports TP/Single returns with flops/mem and CP/Hybrid without)
        def _extract_forward(ret):
            # ret may be (time, red, size, flops, mem) or (time, red, size)
            if len(ret) >= 5:
                return ret[0], ret[1], ret[2], ret[3], ret[4]
            elif len(ret) == 3:
                return ret[0], ret[1], ret[2], 0, []
            else:
                return ret[0], 0, 0, 0, []

        # QKV Projection GEMM
        qkv_proj_gemm_f,  qkv_proj_reduction_f, qkv_proj_size_f, qkv_proj_flops_f, qkv_proj_mem_f = _extract_forward(
            self.parallelism_gemm_forward(gemm_qkv_proj, "qkv_projection_f", gemm_type=GemmType.QKV)
        )
        qkv_proj_gemm_b,  qkv_proj_reduction_b, qkv_proj_size_b = self.parallelism_gemm_backward(gemm_qkv_proj, "qkv_projection_b", gemm_type=GemmType.QKV)
        qkv_proj_f = qkv_proj_gemm_f + qkv_proj_reduction_f
        qkv_proj_b = qkv_proj_gemm_b + qkv_proj_reduction_b
        transformer_results['qkv_proj'] = {
            'forward': qkv_proj_f, 'backward': qkv_proj_b,
            'forward_gemm': qkv_proj_gemm_f, 'forward_reduction': qkv_proj_reduction_f,
            'backward_gemm': qkv_proj_gemm_b, 'backward_reduction': qkv_proj_reduction_b,
            'comm_size_forward': qkv_proj_size_f, 'comm_size_backward': qkv_proj_size_b,
            'flops': qkv_proj_flops_f,
            'memory_accesses': self._mem_levels(qkv_proj_mem_f),
        }

        if self.flash_attention is False:
            # Attention Score GEMM
            attn_score_gemm_f,  attn_score_reduction_f, attn_score_size_f, attn_score_flops_f, attn_score_mem_f = _extract_forward(
                self.parallelism_gemm_forward(gemm_attention_score, "attention_score_f", gemm_type=GemmType.ATTENTION_SCORE)
            )
            attn_score_gemm_b,  attn_score_reduction_b, attn_score_size_b = self.parallelism_gemm_backward(gemm_attention_score, "attention_score_b", gemm_type=GemmType.ATTENTION_SCORE)
            attention_score_f = attn_score_gemm_f + attn_score_reduction_f
            attention_score_b = attn_score_gemm_b + attn_score_reduction_b
            transformer_results['attention_score'] = {
                'forward': attention_score_f, 'backward': attention_score_b,
                'forward_gemm': attn_score_gemm_f, 'forward_reduction': attn_score_reduction_f,
                'backward_gemm': attn_score_gemm_b, 'backward_reduction': attn_score_reduction_b,
                'comm_size_forward': attn_score_size_f, 'comm_size_backward': attn_score_size_b,
                'flops': attn_score_flops_f,
                'memory_accesses': self._mem_levels(attn_score_mem_f),
            }
            attention_scale_softmax_f = self.get_scale_softmax_f(gemm=gemm_attention_score)
            attention_scale_softmax_b = self.get_scale_softmax_b(gemm=gemm_attention_score)
            transformer_results['attention_scale_softmax'] = {'forward': attention_scale_softmax_f, 'backward': attention_scale_softmax_b}

            # Attention Output GEMM
            attn_out_gemm_f,  attn_out_reduction_f, attn_out_size_f, attn_out_flops_f, attn_out_mem_f = _extract_forward(
                self.parallelism_gemm_forward(gemm_attention_output, "attention_output_f", gemm_type=GemmType.ATTENTION_OUTPUT)
            )
            attn_out_gemm_b,  attn_out_reduction_b, attn_out_size_b = self.parallelism_gemm_backward(gemm_attention_output, "attention_output_b", gemm_type=GemmType.ATTENTION_OUTPUT)
            attention_output_f = attn_out_gemm_f + attn_out_reduction_f
            attention_output_b = attn_out_gemm_b + attn_out_reduction_b
            transformer_results['attention_output'] = {
                'forward': attention_output_f, 'backward': attention_output_b,
                'forward_gemm': attn_out_gemm_f, 'forward_reduction': attn_out_reduction_f,
                'backward_gemm': attn_out_gemm_b, 'backward_reduction': attn_out_reduction_b,
                'comm_size_forward': attn_out_size_f, 'comm_size_backward': attn_out_size_b,
                'flops': attn_out_flops_f,
                'memory_accesses': self._mem_levels(attn_out_mem_f),
            }
            # Aggregate attention (score + softmax + output)
            attention_f = attention_score_f + attention_scale_softmax_f + attention_output_f
            attention_b = attention_score_b + attention_scale_softmax_b + attention_output_b
            attention_gemm_f = attn_score_gemm_f + attn_out_gemm_f + attention_scale_softmax_f
            attention_reduction_f = attn_score_reduction_f + attn_out_reduction_f
            attention_gemm_b = attn_score_gemm_b + attn_out_gemm_b + attention_scale_softmax_b
            attention_reduction_b = attn_score_reduction_b + attn_out_reduction_b
            attention_size_f = attn_score_size_f + attn_out_size_f
            attention_size_b = attn_score_size_b + attn_out_size_b
            # Aggregate flops and per-level memory accesses (score + output)
            attention_flops = (attn_score_flops_f or 0) + (attn_out_flops_f or 0)
            attention_mem = [
                (a or 0) + (b or 0)
                for a, b in zip_longest(attn_score_mem_f or [], attn_out_mem_f or [], fillvalue=0)
            ]
            transformer_results["attention"] = {
                'forward': attention_f,
                'backward': attention_b,
                "forward_gemm": attention_gemm_f,
                "forward_reduction": attention_reduction_f,
                "backward_gemm": attention_gemm_b,
                "backward_reduction": attention_reduction_b,
                "comm_size_forward": attention_size_f,
                "comm_size_backward": attention_size_b,
                "flops": attention_flops,
                "memory_accesses": self._mem_levels(attention_mem),
            }
        elif self.flash_attention is True:
            attention_f, attention_gemm_f, attention_reduction_f, attention_size_f, attention_mem = self.flash_attention_kernel_forward(batch_size,hidden_dim, seq_len, num_heads, kv_heads, num_SMs)
            # TODO WIP.
            attention_b, attention_gemm_b, attention_reduction_b, attention_size_b = 2 * attention_f, 2 * attention_gemm_f, 2 * attention_reduction_f, 2 * attention_size_f

            # flops for flash attention is 2* (QK^T + SV)
            attention_flops = 2 * (batch_size * seq_len * (hidden_dim ** 2) + batch_size * (seq_len ** 2) * hidden_dim)
            
            transformer_results['attention'] = {
                'forward': attention_f, 'backward': attention_b,
                'forward_gemm': attention_gemm_f, 'forward_reduction': attention_reduction_f,
                'backward_gemm': attention_gemm_b, 'backward_reduction': attention_reduction_b,
                'comm_size_forward': attention_size_f, 'comm_size_backward': attention_size_b,
                'flops': attention_flops,
                'memory_accesses': self._mem_levels(attention_mem), # HBM only
            }
        else:
            raise ValueError("flash_attention should be either True or False")

        # Output Projection GEMM
        out_proj_gemm_f, out_proj_reduction_f, out_proj_size_f, out_proj_flops_f, out_proj_mem_f = _extract_forward(
            self.parallelism_gemm_forward(gemm_output_proj, "output_projection_f", gemm_type=GemmType.OUT_PROJ)
        )
        out_proj_gemm_b,  out_proj_reduction_b, out_proj_size_b = self.parallelism_gemm_backward(gemm_output_proj, "output_projection_b", gemm_type=GemmType.OUT_PROJ)
        output_proj_f = out_proj_gemm_f + out_proj_reduction_f
        output_proj_b = out_proj_gemm_b + out_proj_reduction_b
        transformer_results['output_proj'] = {
            'forward': output_proj_f, 'backward': output_proj_b,
            'forward_gemm': out_proj_gemm_f, 'forward_reduction': out_proj_reduction_f,
            'backward_gemm': out_proj_gemm_b, 'backward_reduction': out_proj_reduction_b,
            'comm_size_forward': out_proj_size_f, 'comm_size_backward': out_proj_size_b,
            'flops': out_proj_flops_f,
            'memory_accesses': self._mem_levels(out_proj_mem_f),
        }

        # FFN1 GEMM
        ffn1_gemm_f,  ffn1_reduction_f, ffn1_size_f, ffn1_flops_f, ffn1_mem_f = _extract_forward(
            self.parallelism_gemm_forward(gemm_ffn1, "ffn_f", gemm_type=GemmType.FFN1)
        )
        ffn1_gemm_b,  ffn1_reduction_b, ffn1_size_b = self.parallelism_gemm_backward(gemm_ffn1, "ffn_b", gemm_type=GemmType.FFN1)
        ffn1_f = ffn1_gemm_f + ffn1_reduction_f
        ffn1_b = ffn1_gemm_b + ffn1_reduction_b
        transformer_results['ffn1'] = {
            'forward': ffn1_f, 'backward': ffn1_b,
            'forward_gemm': ffn1_gemm_f, 'forward_reduction': ffn1_reduction_f,
            'backward_gemm': ffn1_gemm_b, 'backward_reduction': ffn1_reduction_b,
            'comm_size_forward': ffn1_size_f, 'comm_size_backward': ffn1_size_b,
            'flops': ffn1_flops_f,
            'memory_accesses': self._mem_levels(ffn1_mem_f),
        }

        # FFN2 GEMM
        ffn2_gemm_f, ffn2_reduction_f, ffn2_size_f, ffn2_flops_f, ffn2_mem_f = _extract_forward(
            self.parallelism_gemm_forward(gemm_ffn2, "ffn2_f", gemm_type=GemmType.FFN2)
        )
        ffn2_gemm_b,  ffn2_reduction_b, ffn2_size_b = self.parallelism_gemm_backward(gemm_ffn2, "ffn2_b", gemm_type=GemmType.FFN2)
        ffn2_f = ffn2_gemm_f + ffn2_reduction_f
        ffn2_b = ffn2_gemm_b + ffn2_reduction_b
        transformer_results['ffn2'] = {
            'forward': ffn2_f, 'backward': ffn2_b,
            'forward_gemm': ffn2_gemm_f, 'forward_reduction': ffn2_reduction_f,
            'backward_gemm': ffn2_gemm_b, 'backward_reduction': ffn2_reduction_b,
            'comm_size_forward': ffn2_size_f , 'comm_size_backward': ffn2_size_b,
            'flops': ffn2_flops_f,
            'memory_accesses': self._mem_levels(ffn2_mem_f),
        }
        
            

        # Calculate non-GEMM operations
        embedding_f, embedding_mem = self.get_embedding_f(vocab_size=vocab_size, seq_len=seq_len, hidden_dim=hidden_dim)
        embedding_b = self.get_embedding_b(vocab_size=vocab_size, seq_len=seq_len, hidden_dim=hidden_dim)
        transformer_results['embedding'] = {'forward': embedding_f, 'backward': embedding_b, 'memory_accesses': self._mem_levels(embedding_mem)}

        residual1_f = self.get_residual_f(tensor_shape=gemm_output_proj)
        residual1_b = self.get_residual_b(tensor_shape=gemm_output_proj)
        # transformer_results['residual1'] = {'forward': residual1_f, 'backward': residual1_b}

        layernorm1_f, layernorm_reduction_f, LN1_comm_bytes_f= self.get_layernorm_f(batch=batch_size, seq_len=seq_len, d_model=hidden_dim)
        layernorm1_b, layernorm_reduction_b, LN1_comm_bytes_b= self.get_layernorm_b(batch=batch_size, seq_len=seq_len, d_model=hidden_dim)
        transformer_results['layernorm1'] = {
            'forward': layernorm1_f + residual1_f +layernorm_reduction_f,
            'forward_compute': layernorm1_f + residual1_f, 
            'forward_reduction':layernorm_reduction_f, 
            'backward': layernorm1_b + residual1_b + layernorm_reduction_b,
            "backward_compute":layernorm1_b + residual1_b , 
            'backward_reduction': layernorm_reduction_b, 
            'comm_size_forward': LN1_comm_bytes_f, 'comm_size_backward': LN1_comm_bytes_b
        }

        residual2_f = self.get_residual_f(tensor_shape=gemm_ffn2)
        residual2_b = self.get_residual_b(tensor_shape=gemm_ffn2)
        # transformer_results['residual2'] = {'forward': residual2_f, 'backward': residual2_b}

        layernorm2_f, layernorm_reduction_f, LN2_comm_bytes_f = self.get_layernorm_f(batch=batch_size, seq_len=seq_len, d_model=hidden_dim)
        layernorm2_b, layernorm_reduction_b, LN2_comm_bytes_b = self.get_layernorm_b(batch=batch_size, seq_len=seq_len, d_model=hidden_dim)
        transformer_results['layernorm2'] = {'forward': layernorm2_f + residual2_f + layernorm_reduction_f, "forward_compute": layernorm2_f + residual2_f, 'forward_reduction':layernorm_reduction_f,'backward': layernorm2_b + residual2_b + layernorm_reduction_b, "backward_compute":layernorm2_b + residual2_b, 'backward_reduction': layernorm_reduction_b, 'comm_size_forward': LN2_comm_bytes_f, 'comm_size_backward': LN2_comm_bytes_b}
        
        if self.model_type == "llama":
            act_f = self.get_swiglu_f(tensor_shape=gemm_ffn1)
            act_b = self.get_swiglu_b(tensor_shape=gemm_ffn1)
        else:
            act_f = self.get_gelu_f(tensor_shape=gemm_ffn1)
            act_b = self.get_gelu_b(tensor_shape=gemm_ffn1)
        transformer_results['gelu'] = {'forward': act_f, 'backward': act_b}

        linear_softmax_f, linear_softmax_mem = self.get_linear_softmax_f(gemm=gemm_linear)
        linear_softmax_b = self.get_linear_softmax_b(gemm=gemm_linear)
        transformer_results['linear_softmax'] = {
            'forward': linear_softmax_f, 
            'backward': linear_softmax_b, 
            'memory_accesses': self._mem_levels(linear_softmax_mem)}

        # Calculate MHA and FFN times directly from results dict
        mha_time_f = ( 
            transformer_results['qkv_proj']['forward'] + transformer_results['attention']['forward'] + transformer_results['output_proj']['forward'] 
        )
        
        
        mha_time_b = ( 
            transformer_results['qkv_proj']['backward'] + transformer_results['attention']['backward'] + transformer_results['output_proj']['backward'] 
        )
        transformer_results['MHA'] = {
            'forward': mha_time_f,
            'backward': mha_time_b,
            "forward_reduction": qkv_proj_reduction_f + attention_reduction_f + out_proj_reduction_f,
            "backward_reduction": qkv_proj_reduction_b + attention_reduction_b + out_proj_reduction_b,
            "comm_size_forward": qkv_proj_size_f + attention_size_f + out_proj_size_f,
            "comm_size_backward": qkv_proj_size_b + attention_size_b + out_proj_size_b,
            "flops": qkv_proj_flops_f + transformer_results['attention']['flops'] + out_proj_flops_f,
            "memory_accesses": self._combine_mem(
                qkv_proj_mem_f,
                attention_mem,
                out_proj_mem_f,
            ),

        }

        ffn_time_f = transformer_results['ffn1']['forward'] + transformer_results['ffn2']['forward'] + transformer_results['gelu']['forward']
        ffn_time_b = (
            transformer_results['ffn1']['backward'] + transformer_results['ffn2']['backward'] + transformer_results['gelu']['backward']
        )
        transformer_results['MLP'] = {
            'forward': ffn_time_f,
            'backward': ffn_time_b,
            "forward_reduction": ffn2_reduction_f + ffn1_reduction_f,
            "backward_reduction": ffn1_reduction_b + ffn2_reduction_b,
            "comm_size_forward": ffn2_size_f + ffn1_size_f,
            "comm_size_backward": ffn1_size_b + ffn2_size_b,
            "flops": ffn1_flops_f + ffn2_flops_f,
            "memory_accesses": self._combine_mem(
                ffn1_mem_f,
                ffn2_mem_f,
            ),
        }
        # Calculate transformer times directly
        
        transformer_time_f = (
            transformer_results['MHA']['forward'] + transformer_results['MLP']['forward']  +
            transformer_results['layernorm1']['forward'] + transformer_results['layernorm2']['forward']
        )
        transformer_time_b = (
            transformer_results['MHA']['backward'] + transformer_results['MLP']['backward'] +
            transformer_results['layernorm1']['backward'] + transformer_results['layernorm2']['backward']
        )
        

        node_breakdown = {
            'transformer_time_f': transformer_time_f,
            'transformer_time_b': transformer_time_b,
            'embedding_f': transformer_results['embedding']['forward'],
            'embedding_b': transformer_results['embedding']['backward'],
            'linear_softmax_f': transformer_results['linear_softmax']['forward'],
            'linear_softmax_b': transformer_results['linear_softmax']['backward']
        }

        if self._generate_graphs:
            results_path = os.path.join(self.output_dir, "transformer_results.txt")
            with open(results_path, "w", encoding="utf-8") as results_file:
                json.dump(
                    {
                        "transformer_results": transformer_results,
                        "node_breakdown": node_breakdown,
                    },
                    results_file,
                    indent=2,
                    sort_keys=True,
                )

        return transformer_results, node_breakdown


    def _effective_transformer_batch(self) -> int:
        if self.lp > 1:
            return self.microB
        if self.dp > 1:
            return self.miniB
        return self.batch_size

    def _build_comm_metadata(
        self,
        reduction_sizes: Dict[str, int],
        local_comp: Dict[str, float], 
        embedding_size: int,
        softmax_size: int,
        cross_layer_bytes: int,
        zero2_embedding_gather_bytes: float = 0.0,
        zero2_transformer_gather_bytes: float = 0.0,
        zero2_softmax_gather_bytes: float = 0.0,
        zero3_embedding_gather_bytes: float = 0.0,
        zero3_transformer_gather_bytes: float = 0.0,
        zero3_softmax_gather_bytes: float = 0.0,
    ) -> Dict[str, Dict[str, Any]]:
        grad_collective = 'reduce_scatter' if (self.zero_stage >= 2 and self.dp > 1) else 'all_reduce'
        metadata = {
            'transformer': {
                'size': reduction_sizes['total_size'],
                'type': grad_collective,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': local_comp['total_local']
            },
            'embedding': {
                'size': embedding_size,
                'type': grad_collective,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0
            },
            'softmax': {
                'size': softmax_size,
                'type': grad_collective,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0
            },
            'cross_layer': {
                'size': cross_layer_bytes,
                'type': 'pipeline',
                'participants': 2,
                'interconnect_type': 'lp',
                'local_comp_time': 0
            }
        }
        if zero2_embedding_gather_bytes:
            metadata['zero2_embedding_gather'] = {
                'size': int(math.ceil(zero2_embedding_gather_bytes)),
                'type': 'all_gather',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
            }
        if zero2_transformer_gather_bytes:
            metadata['zero2_transformer_gather'] = {
                'size': int(math.ceil(zero2_transformer_gather_bytes)),
                'type': 'all_gather',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
            }
        if zero2_softmax_gather_bytes:
            metadata['zero2_softmax_gather'] = {
                'size': int(math.ceil(zero2_softmax_gather_bytes)),
                'type': 'all_gather',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
            }
        if zero3_embedding_gather_bytes:
            metadata['zero3_embedding_gather'] = {
                'size': int(math.ceil(zero3_embedding_gather_bytes)),
                'type': 'all_gather',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
            }
        if zero3_transformer_gather_bytes:
            metadata['zero3_transformer_gather'] = {
                'size': int(math.ceil(zero3_transformer_gather_bytes)),
                'type': 'all_gather',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
                'tp_shard': True,
            }
        if zero3_softmax_gather_bytes:
            metadata['zero3_softmax_gather'] = {
                'size': int(math.ceil(zero3_softmax_gather_bytes)),
                'type': 'all_gather',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
            }
        return metadata

    def _populate_transformer_comm_metadata(
        self,
        entry: Dict[str, Any],
        metadata: Dict[str, Dict[str, Any]],
        # gemm_spec: Tuple[int, ...],
        comm_bytes_fwd: int,
        comm_bytes_bwd: int,
    ) -> None:
        """Attach tensor-parallel collectives for a GEMM to metadata and entry."""


        if not self.tp or self.tp <= 1 and self.cp <= 1:
            return



        def add_comm(direction: str, suffix: str, kind: str, size_bytes: float, participants: int, interconnect: str) -> None:
            if participants <= 1:
                return
            bytes_int = int(math.ceil(size_bytes))
            if bytes_int <= 0:
                return
            key = f"{entry['name']}_{direction}_{suffix}"
            # ensure uniqueness when multiple collectives share the same suffix
            unique_key = key
            counter = 1
            while unique_key in metadata:
                counter += 1
                unique_key = f"{key}_{counter}"

            metadata[unique_key] = {
                'size': bytes_int,
                'type': kind,
                'participants': int(participants),
                'interconnect_type': interconnect,
            }
            entry[direction]['comm_keys'].append(unique_key)

        parallelism_mode = self.get_parallelism_mode()
        rules_by_mode = COMMUNICATION_RULES.get(parallelism_mode)
        if not rules_by_mode:
            return

        spec = rules_by_mode.get(entry['name'])
        if spec is None:
            spec = rules_by_mode.get(COMM_RULE_DEFAULT_KEY)
        if not spec:
            return


        participants_lookup = {
            'tp': getattr(self, 'tp', 0),
            'cp': getattr(self, 'cp', 0),
            'dp': getattr(self, 'dp', 0),
            'lp': getattr(self, 'lp', 0),
        }

        for direction in ('forward', 'backward'):
            rule = spec.get(direction)
            if not rule:
                continue
            size_bytes = comm_bytes_fwd if direction == 'forward' else comm_bytes_bwd
            scope = rule.get('participants')
            participants = participants_lookup.get(scope or '', 0)
            if participants <= 1:
                continue
            kind = rule['kind']
            suffix = rule.get('suffix', kind)
            interconnect = rule.get('interconnect', scope or 'tp')
            add_comm(direction, suffix, kind, size_bytes, participants, interconnect)

    def _build_interconnect_params(self) -> Dict[str, Tuple[float, float]]:
        return {
            'dp': (self.IBD, self.LLD),
            'lp': (self.IBL, self.LLL),
            'tp': (self.IBTP, self.LLTP),
            'cp': (self.IBTP, self.LLTP),
        }

    def _prepare_execution_graphs(
        self,
        *,
        node_breakdown: Dict[str, float],
        transformer_results: Dict[str, Dict[str, Any]],
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        intermediate_size: int,
        vocab_size: int,
        include_pipeline_backward: bool,
        include_transformer_backward: bool,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,  # optional override (decode only)
        zero2_embedding_gather_bytes: float = 0.0,
        zero2_transformer_gather_bytes: float = 0.0,
        zero2_softmax_gather_bytes: float = 0.0,
        zero3_embedding_gather_bytes: float = 0.0,
        zero3_transformer_gather_bytes: float = 0.0,
        zero3_softmax_gather_bytes: float = 0.0,
    ) -> Tuple[Graph, Any, Optional[Graph], Optional[Any], Optional[Any], Dict[str, Tuple[float, float]]]:
        """Build pipeline/transformer graphs shared across training and inference."""

        if not include_pipeline_backward and not include_transformer_backward:
            # Forward-only inference: skip training all-reduce bookkeeping.
            reduction_sizes = {
                'qkv_size': 0,
                'output_size': 0,
                'ffn_size': 0,
                'total_size': 0,
            }
            local_comp = {
                'qkv_local': 0.0,
                'output_local': 0.0,
                'ffn_local': 0.0,
                'total_local': 0.0,
            }
        else:
            reduction_sizes = self.get_data_parallel_reduction_sizes(hidden_dim, intermediate_size)
            local_comp = self.get_data_parallel_local_computation(hidden_dim, intermediate_size)

        # these are used for dp all-reduce/reduce-scatter.
        # TODO: check the embedding_size below. I think only the first term is needed.
        embedding_size = math.ceil(self.precision.grad_communication * vocab_size * hidden_dim) + math.ceil(self.precision.grad_communication * seq_len * hidden_dim * batch_size)
        softmax_size = math.ceil(self.precision.grad_communication * hidden_dim * vocab_size)
        cross_layer_bytes = self.get_inter_layer_comm_latency_llm(batch_size, hidden_dim, seq_len)[1]

        comm_metadata = self._build_comm_metadata(
            reduction_sizes=reduction_sizes,
            local_comp=local_comp, 
            embedding_size=embedding_size,
            softmax_size=softmax_size,
            cross_layer_bytes=cross_layer_bytes,
            zero2_embedding_gather_bytes=zero2_embedding_gather_bytes,
            zero2_transformer_gather_bytes=zero2_transformer_gather_bytes,
            zero2_softmax_gather_bytes=zero2_softmax_gather_bytes,
            zero3_embedding_gather_bytes=zero3_embedding_gather_bytes,
            zero3_transformer_gather_bytes=zero3_transformer_gather_bytes,
            zero3_softmax_gather_bytes=zero3_softmax_gather_bytes,
        )

        transformer_operation_entries: List[Dict[str, Any]] = []
        transformer_comm_metadata: Dict[str, Dict[str, Any]] = {}
        parallelism_mode = self.get_parallelism_mode()
        if parallelism_mode in (ParallelismMode.CONTEXT, ParallelismMode.TENSOR_CONTEXT_HYBRID):
            for key in ("layernorm1", "qkv_proj", "attention", "output_proj", "layernorm2", "MLP"):
                try:
                    spec = transformer_results[key]
                except KeyError:
                    print(f"Key {key} not found in transformer_results")
                    print(transformer_results)
                    exit()
                fwd_time = spec['forward_gemm'] if 'forward_gemm' in spec else spec['forward_compute'] if 'forward_compute' in spec else spec['forward']
                bwd_time = spec['backward_gemm'] if 'backward_gemm' in spec else spec['backward_compute'] if 'backward_compute' in spec else spec['backward']
                fwd_red = spec.get('forward_reduction', 0.0)
                bwd_red = spec.get('backward_reduction', 0.0)
                comm_bytes_fwd = spec.get('comm_size_forward', 0)
                comm_bytes_bwd = spec.get('comm_size_backward', 0)

                entry = {
                    "name": key,
                    "forward": {
                        "duration": fwd_time,
                        "reduction": fwd_red,
                        "comm_keys": [],
                    },
                    "backward": {
                        "duration": bwd_time,
                        "reduction": bwd_red,
                        "comm_keys": [],
                    },
                }

                self._populate_transformer_comm_metadata(
                    entry=entry,
                    metadata=transformer_comm_metadata,
                    comm_bytes_fwd=comm_bytes_fwd,
                    comm_bytes_bwd=comm_bytes_bwd,
                )

                transformer_operation_entries.append(entry)
            
        elif parallelism_mode in (ParallelismMode.TENSOR, ParallelismMode.TENSOR_SEQUENCE, ParallelismMode.SINGLE):
            for key in ("layernorm1", "MHA", "layernorm2", "MLP"):
                spec = transformer_results[key]
                fwd_time = spec['forward']
                bwd_time = spec['backward']
                fwd_red = spec.get('forward_reduction', 0.0)
                bwd_red = spec.get('backward_reduction', 0.0)
                comm_bytes_fwd = spec.get('comm_size_forward', 0)
                comm_bytes_bwd = spec.get('comm_size_backward', 0)

                entry = {
                    "name": key,
                    "forward": {
                        "duration": fwd_time,
                        "reduction": fwd_red,
                        "comm_keys": [],
                    },
                    "backward": {
                        "duration": bwd_time,
                        "reduction": bwd_red,
                        "comm_keys": [],
                    },
                }

                self._populate_transformer_comm_metadata(
                    entry=entry,
                    metadata=transformer_comm_metadata,
                    comm_bytes_fwd=comm_bytes_fwd,
                    comm_bytes_bwd=comm_bytes_bwd,
                )

                transformer_operation_entries.append(entry)

        else:
            raise ValueError(f"Unsupported parallelism mode: {parallelism_mode}")
        transformer_graph: Optional[Graph] = None
        transformer_forward_root: Optional[Any] = None
        transformer_backward_root: Optional[Any] = None

        transformer_comp_times = {
            "transformer": {
                "gemms": transformer_operation_entries,
            }
        }

        transformer_graph = simulate_LLM.Graph(
            mode="transformer",
            dp=self.dp,
            lp=self.lp,
            tp=self.tp,
            cp=self.cp,
            comp_times=transformer_comp_times,
            comm_metadata=transformer_comm_metadata,
            misc_metadata={"dp_zero_stage": self.zero_stage},
        )
        transformer_forward_root = transformer_graph.construct_transformer_graph(direction="forward")
        if include_transformer_backward:
            transformer_backward_root = transformer_graph.construct_transformer_graph(direction="backward")

        comp_times = {
            "embedding_f": node_breakdown.get('embedding_f', 0.0),
            "embedding_b": node_breakdown.get('embedding_b', 0.0) if include_pipeline_backward else 0.0,
            "linear_softmax_f": node_breakdown.get('linear_softmax_f', 0.0),
            "linear_softmax_b": node_breakdown.get('linear_softmax_b', 0.0) if include_pipeline_backward else 0.0,
            "transformer_f": node_breakdown.get('transformer_time_f', 0.0),
            "transformer_b": node_breakdown.get('transformer_time_b', 0.0) if include_pipeline_backward else 0.0,
            "cross_layer_f": 0.0,
            "cross_layer_b": 0.0,
        }
        misc_metadata = {
            "num_batch": self.mb,
            "num_layer": self.num_layers,
            "all_reduce": "every_layer",
            "dp_zero_stage": self.zero_stage,
        }

        pipeline_graph_obj = simulate_LLM.Graph(
            mode="pipeline",
            dp=self.dp,
            lp=self.lp,
            tp=self.tp,
            cp=self.cp,
            comp_times=comp_times,
            comm_metadata=comm_metadata,
            misc_metadata=misc_metadata,
        )
        graph_root = pipeline_graph_obj.construct_fwd_bwd_graph(include_backward=include_pipeline_backward)
        interconnect_params = self._build_interconnect_params()

        return (
            pipeline_graph_obj,
            graph_root,
            transformer_graph,
            transformer_forward_root,
            transformer_backward_root,
            interconnect_params,
        )

    def calc_time_llm(self):
        """Calculate time for LLM model."""
        # Extract model parameters
        batch_size = self._effective_transformer_batch()
        vocab_size = self.vocab_size
        hidden_dim = self.hidden_dim
        seq_len = self.seq_len
        num_heads = self.num_heads
        intermediate_size = self.intermediate_size
        kv_heads = self.kv_heads
        
        # ZeRO data-parallel stages:
        #   stage 0 – identical to DDP (replicated params/grads/optimizer)
        #   stage 1 – shard optimizer state only (communication unchanged)
        #   stage 2 – shard optimizer and gradients (grad RS + one param AG)
        #   stage 3 – shard optimizer, gradients, and parameters (grad RS + two param AG, per-layer materialization)
        self.readjust_type()

        (
            total_params_per_rank,
            max_layer_params,
            params_per_layer_per_rank,
            embedding_params_per_rank,
            output_params_per_rank,
        ) = self._param_stats_per_rank(hidden_dim, intermediate_size, vocab_size)

        param_bytes = total_params_per_rank * self.precision.parameters
        transformer_param_layer_bytes = params_per_layer_per_rank * self.precision.parameters
        embedding_param_bytes = embedding_params_per_rank * self.precision.parameters
        softmax_param_bytes = output_params_per_rank * self.precision.parameters

        if self.zero_stage == 2 and self.dp > 1:
            zero2_embedding_gather_bytes = embedding_param_bytes
            zero2_transformer_gather_bytes = transformer_param_layer_bytes
            zero2_softmax_gather_bytes = softmax_param_bytes
        else:
            zero2_embedding_gather_bytes = 0.0
            zero2_transformer_gather_bytes = 0.0
            zero2_softmax_gather_bytes = 0.0

        if self.zero_stage >= 3 and self.dp > 1:
            zero3_embedding_gather_bytes = embedding_param_bytes
            zero3_transformer_gather_bytes = transformer_param_layer_bytes
            zero3_softmax_gather_bytes = softmax_param_bytes
        else:
            zero3_embedding_gather_bytes = 0.0
            zero3_transformer_gather_bytes = 0.0
            zero3_softmax_gather_bytes = 0.0

        # zero stage 3 creates a need for materialization of parameters per layer. This is the peak memory requirement.
        # these bytes are 'ephemeral' as they get discarded after, but still need to accounted for in the memory sim.
        self.zero3_ephemeral_peak_bytes = (
            max_layer_params * self.precision.parameters if self.zero_stage >= 3 else 0.0
        )

        num_SMs = self.hw_config.tech_config.core.num_bundles
        transformer_results, node_breakdown = self.compute_all_gemm_and_node_times(
            batch_size, vocab_size, hidden_dim, seq_len, num_heads, kv_heads, intermediate_size, num_SMs
        )

        # transformer mem layer considers zero stage internally
        transformer_mem_layer, transformer_act_layer, transformer_act_layer_inf, transformer_static_layer, gradient_mem_layer, optimizer_mem_layer, weight_memory_layer = (
            LLM_util.get_transformer_mem_layer(
                dp=self.dp,
                tp=self.tp,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                seq_len=seq_len / self.cp,
                intermediate_size=intermediate_size,
                n_heads=num_heads,
                precision=self.precision,
                model_type=self.model_type,
                zero_stage=self.zero_stage,
            )
        )

        memory_data = {
            'activation_mem_per_layer': transformer_act_layer,
            'activation_mem_per_layer_inference': transformer_act_layer_inf,
            'weight_mem_per_layer': weight_memory_layer,
            'gradient_mem_per_layer': gradient_mem_layer,
            'optimizer_mem_per_layer': optimizer_mem_layer,
            'static_mem_per_layer': transformer_static_layer,
            'total_mem_per_layer': transformer_mem_layer,
            'zero3_ephemeral_peak_bytes': self.zero3_ephemeral_peak_bytes,
        }
        # NOTE: simulate_memory currently ignores the ZeRO-specific entry above. It needs to be updated to handle them.

        if self._debug_memory:
            layers_per_device = max(1, math.ceil(self.num_layers / max(1, self.lp)))
            to_gib = lambda bytes_val: float(bytes_val) / float(1024 ** 3)
            self._memory_breakdown_debug = {
                "layers_per_device": layers_per_device,
                "zero_stage": self.zero_stage,
                "activation_gib": to_gib(transformer_act_layer * layers_per_device),
                "weight_gib": to_gib(weight_memory_layer * layers_per_device),
                "gradient_gib": to_gib(gradient_mem_layer * layers_per_device),
                "optimizer_gib": to_gib(optimizer_mem_layer * layers_per_device),
                "static_gib": to_gib(transformer_static_layer * layers_per_device),
                "total_layer_gib": to_gib(transformer_mem_layer * layers_per_device),
            }
            if self.zero_stage >= 3 and self.zero3_ephemeral_peak_bytes:
                self._memory_breakdown_debug["zero3_ephemeral_gib"] = to_gib(
                    self.zero3_ephemeral_peak_bytes
                )

    

        if self.debug:
            print("self.tp:", self.tp)
            print("Calculating LLM time...")
            print(
                "simulating parallelism with dp = {}, lp = {}, total data batch = {}, "
                "for each dp node, data batch = {}, for each pipeline stage, data batch = {}".format(
                    self.dp, self.lp, self.batch_size, self.miniB, self.microB
                )
            )
            print("data-parallel zero stage:", self.zero_stage)
            print("total number of workers: {}".format(self.num_workers))
            print("number of workers for each data parallelism batch: {}".format(self.num_workers_dp))
            print("number of workers for each pipeline stage: {}".format(self.num_workers_lp))
        

        (
            pipeline_graph_obj,
            graph_root,
            transformer_graph,
            transformer_forward_root,
            transformer_backward_root,
            interconnect_params,
        ) = self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            transformer_results=transformer_results,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            # num_heads=num_heads,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            include_pipeline_backward=True,
            include_transformer_backward=True,
            zero2_embedding_gather_bytes=zero2_embedding_gather_bytes,
            zero2_transformer_gather_bytes=zero2_transformer_gather_bytes,
            zero2_softmax_gather_bytes=zero2_softmax_gather_bytes,
            zero3_embedding_gather_bytes=zero3_embedding_gather_bytes,
            zero3_transformer_gather_bytes=zero3_transformer_gather_bytes,
            zero3_softmax_gather_bytes=zero3_softmax_gather_bytes,
        )

        self.transformer_graph = transformer_graph
        self.transformer_forward_root = transformer_forward_root
        self.transformer_backward_root = transformer_backward_root
        self.transformer_analytical_time_forward = node_breakdown['transformer_time_f']
        self.transformer_analytical_time_backward = node_breakdown['transformer_time_b']

        self.pipeline_graph = pipeline_graph_obj
        self.pipeline_root = graph_root
        self.pipeline_interconnect = interconnect_params

        mode = self.execution_mode
        
        dispatcher = LLMExecutionDispatcher(
            time_calc=self,
            pipeline_graph=self.pipeline_graph,
            pipeline_root=self.pipeline_root,
            interconnect_params=self.pipeline_interconnect,
            transformer_graph=self.transformer_graph,
            transformer_forward_root=self.transformer_forward_root,
            transformer_backward_root=self.transformer_backward_root,
        )
        mode = self.execution_mode
        try:
            result = dispatcher.run(mode)
        except NotImplementedError as exc:
            raise NotImplementedError(f"{exc}. Selected execution mode '{mode.value}'.") from exc

        time_fw_bw = result.total_time

        pipeline_root = result.graph_root
        self.pipeline_graph = dispatcher.pipeline_graph
        self.pipeline_root = pipeline_root
        self.pipeline_interconnect = dispatcher.interconnect_params
        


        # debug helper. If set, print analytical transformer time and actual transformer time
        if self._generate_graphs:
            print(f"Analytical transformer forward time: {self.transformer_analytical_time_forward:.4f}s")
            print(f"Analytical transformer backward time: {self.transformer_analytical_time_backward:.4f}s")
            if self.transformer_astrasim_time_forward is not None and self.transformer_astrasim_time_backward is not None:
                print(f"Actual transformer forward time: {self.transformer_astrasim_time_forward:.4f}s")
                print(f"Actual transformer backward time: {self.transformer_astrasim_time_backward:.4f}s")

        self.tot_time = time_fw_bw

        return time_fw_bw
        
    def _simulate_with_memory( 
        self,
        graph_root: Any,
        memory_data: Dict[str, Any],
        mode: str = "training" #training or inference
    ) -> Tuple[float, float]:
        """Run memory-aware simulation and report duration plus peak usage."""

        time_with_memory, peak_mem = self.pipeline_graph.simulate_memory(
            graph_root,
            memory_data,
            mode,
            self.output_dir,
            
        )
        self.memory_peak_gb = peak_mem

        hardware_mem_bytes = getattr(self.DRAM, "size", None)
        if hardware_mem_bytes is None:
            hardware_mem_bytes = getattr(
                getattr(self.hw_config, "tech_config", object()),
                "DRAM",
                object(),
            )
            hardware_mem_bytes = getattr(hardware_mem_bytes, "size", None)

        if hardware_mem_bytes is not None:
            hardware_mem_gib = float(hardware_mem_bytes) / float(1024 ** 3)
            self.memory_capacity_per_device_gb = hardware_mem_gib
            mem_delta = hardware_mem_gib - peak_mem
            self.memory_headroom_gb = mem_delta
            memory_dir = os.path.join(self.output_dir, "memory-summary")
            os.makedirs(memory_dir, exist_ok=True)
            info_lines = [
                f"Simulation mode: {mode}",
                f"Hardware memory capacity (per gpu): {hardware_mem_gib:.2f} GiB",
                f"Simulated peak memory usage(per gpu): {peak_mem:.2} GiB",
            ]
            if self.zero_stage >= 3 and self.zero3_ephemeral_peak_bytes:
                info_lines.append(
                    "ZeRO-3 ephemeral param gather (per gpu): {:.2f} GiB".format(
                        self.zero3_ephemeral_peak_bytes / float(1024 ** 3)
                    )
                )
            if mem_delta < 0:
                info_lines.append(f"[WARN] Peak memory exceeds capacity by {abs(mem_delta):.6f} GiB")
                self.memory_capacity_exceeded = True
                self.memory_capacity_violation_gb = max(self.memory_capacity_violation_gb, abs(mem_delta))
            else:
                info_lines.append(f"Remaining memory headroom: {mem_delta:.6f} GiB")
            info_path = os.path.join(memory_dir, "memory_capacity_comparison.txt")
            with open(info_path, "w", encoding="utf-8") as info_file:
                info_file.write("\n".join(info_lines) + "\n")
            if self._debug_memory:
                breakdown = self._memory_breakdown_debug or {}
                print("[DEEPFLOW] Memory summary (per device):")
                print("  Capacity: {:.2f} GiB".format(hardware_mem_gib))
                print("  Simulated peak usage: {:.2f} GiB".format(peak_mem))
                print("  Headroom: {:.2f} GiB".format(mem_delta))
                if breakdown:
                    print("  Layers per device: {}".format(breakdown.get("layers_per_device")))
                    print("  Breakdown (GiB, approx per device):")
                    # print("    Activations: {:.2f}".format(breakdown.get("activation_gib", 0.0)))
                    print("    Weights: {:.2f}".format(breakdown.get("weight_gib", 0.0)))
                    print("    Gradients: {:.2f}".format(breakdown.get("gradient_gib", 0.0)))
                    print("    Optimizer: {:.2f}".format(breakdown.get("optimizer_gib", 0.0)))
                    print("    Static buffers: {:.2f}".format(breakdown.get("static_gib", 0.0)))
                    # print("    Total (layer-based): {:.2f}".format(breakdown.get("total_layer_gib", 0.0)))
        else:
            self.memory_capacity_per_device_gb = None
            self.memory_headroom_gb = None
            if self._debug_memory:
                print("[DEEPFLOW] Memory summary (per device):")
                print("  Capacity: unknown")
                print("  Simulated peak usage: {:.6f} GiB".format(peak_mem))

        return time_with_memory, peak_mem

    def get_time(self):
        return self.tot_time

    def memory_capacity_warning(self) -> Optional[str]:
        violation = max(0.0, getattr(self, "memory_capacity_violation_gb", 0.0))
        if violation > 0.0:
            return (
                f"[WARN] Peak memory exceeds capacity by {violation:.2f} GiB. "
                "Please change parallelism settings for realistic results."
            )
        return None
