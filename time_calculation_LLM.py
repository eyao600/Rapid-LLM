import math
import os
import pickle
import sys
import config
import shutil
import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Tuple, Optional, List, Set
# import numpy as np
import simulate_LLM
from parallelism import Parallelism
from topology import Topology
from simulate_LLM import Graph
import LLM_util
from hw_component import Core, MemoryHierarchy, Network
from model import Model_LSTM, Model_GEMM, Model_LLM
from tile import TiledGEMM, formatBytes
from astrasim_lib.executor import run_astra_simulation_only_onepath
from functools import lru_cache

from simulate_LLM import visualize_graph
from time_calculation import TimeCalculation
from util import disk_cache_method
# algByte = False  # algorithmic ops false
# proj = False  # consider projection layer, turn off for end-2-end validation, as baeline model does not have projection layer
validating_v100 = True

debug = True

showing_ms = False # Show time in ms if True    show time in us if False
if showing_ms:
    m=1e3
    second = "ms"
else:
    m=1e6
    second = "us"


def _env_flag(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}


class ExecutionMode(Enum):
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"
    FULL_ASTRASIM_HIERARCHICAL = "full_astrasim_hierarchical"
    FULL_ASTRASIM_FLATTENED = "full_astrasim_flattened"


@dataclass
class ExecutionResult:
    total_time: float
    graph_root: Any
    mode: ExecutionMode


@dataclass
class TransformerTimings:
    forward: float
    backward: float


class PipelineGraphFlattener:
    """Expand pipeline transformer nodes into explicit tensor-parallel subgraphs."""

    def __init__(
        self,
        pipeline_graph: Graph,
        transformer_graph: Graph,
    ) -> None:
        if transformer_graph is None:
            raise ValueError("Transformer graph is required for flattening")

        transformer_cfg = getattr(transformer_graph, "transformer_cfg", None) or {}
        gemm_entries = transformer_cfg.get("gemms")
        if not gemm_entries:
            raise ValueError("Transformer GEMM template is missing")

        self.pipeline_graph = pipeline_graph
        self.transformer_graph = transformer_graph
        self._gemm_entries = list(gemm_entries)
        tp_degree = transformer_graph.tp
        self._tp_degree = max(1, int(tp_degree))

        self._clone_cache: Dict[int, Any] = {}
        self._op_id_counter: int = 0

    @property
    def tp_degree(self) -> int:
        return self._tp_degree

    def build(self, root: Any) -> Any:
        """Return a flattened clone of the provided pipeline root."""

        if root is None:
            raise ValueError("Pipeline root is required for flattening")
        return self._clone(root)

    def _clone(self, obj: Any) -> Any:
        if obj is None:
            return None

        obj_id = id(obj)
        if obj_id in self._clone_cache:
            return self._clone_cache[obj_id]

        if isinstance(obj, simulate_LLM.Node):
            if obj.name in {"transformer", "transformer_b"}:
                expanded = self._expand_transformer_node(obj)
                self._clone_cache[obj_id] = expanded
                return expanded

            if "linear_softmax" in obj.name:
                # for linear softmax we need to carefully look at hw id to choose.
                cloned = simulate_LLM.Node(
                    obj.name,
                    self._next_op_id(),
                    self._hw_id_for_rank(obj.hw_id, 0),
                    obj.duration,
                    fwd=obj.fwd,
                    is_kv_cache=getattr(obj, "is_kv_cache", False),
                )
            else:
                cloned = simulate_LLM.Node(
                    obj.name,
                    self._next_op_id(),
                    obj.hw_id,
                    obj.duration,
                    fwd=obj.fwd,
                    is_kv_cache=getattr(obj, "is_kv_cache", False),
                )

            self._clone_cache[obj_id] = cloned
            self._copy_metadata(obj, cloned)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned, child_clone)
            return cloned

        if isinstance(obj, simulate_LLM.Edge):
            cloned_edge = simulate_LLM.Edge(
                obj.name,
                self._next_op_id(),
                obj.duration,
                is_all_reduce=getattr(obj, "is_all_reduce", False),
                comm_size_bytes=getattr(obj, "comm_size_bytes", 0),
                comm_type=getattr(obj, "comm_type", None),
                participants=getattr(obj, "participants", 1),
                comm_interconnect_type=getattr(obj, "comm_interconnect_type", None),
            )
            self._clone_cache[obj_id] = cloned_edge
            self._copy_metadata(obj, cloned_edge)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned_edge, child_clone)
            return cloned_edge

        if isinstance(obj, simulate_LLM.Data_batch):
            cloned_batch = simulate_LLM.Data_batch(obj.name, obj.batch_id, obj.duration)
            self._clone_cache[obj_id] = cloned_batch
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned_batch, child_clone)
            return cloned_batch

        if isinstance(obj, simulate_LLM.Gradient):
            cloned_grad = simulate_LLM.Gradient(obj.name, self._next_op_id(), obj.hw_id, obj.duration)
            self._clone_cache[obj_id] = cloned_grad
            self._copy_metadata(obj, cloned_grad)
            for child in getattr(obj, "children", []):
                child_clone = self._clone(child)
                if child_clone is not None:
                    self._attach(cloned_grad, child_clone)
            return cloned_grad

        raise TypeError(f"Unsupported graph element type: {type(obj)!r}")

    def _expand_transformer_node(self, node: simulate_LLM.Node) -> Tuple[Any, ...]:
        node_id = id(node)
        if node_id in self._clone_cache:
            cached_entry = self._clone_cache[node_id]
            if isinstance(cached_entry, (list, tuple)):
                return tuple(cached_entry)

        stage_id = getattr(node, "stage_id", node.hw_id)
        micro_batch = getattr(node, "micro_batch_index", None)
        layer_index = getattr(node, "layer_index", None)
        direction = getattr(node, "direction", "forward" if node.fwd else "backward")

        rank_heads: List[Any] = []
        rank_tails: List[Any] = []
        

        for tp_rank in range(self._tp_degree):
            previous: Optional[Any] = None
            head: Optional[Any] = None
            hw_id = self._hw_id_for_rank(stage_id, tp_rank)

            gemm_iterable = self._gemm_entries
            if direction == "backward":
                gemm_iterable = list(reversed(self._gemm_entries))

            for gemm_idx, entry in enumerate(gemm_iterable):
                entry_name = entry.get("name", f"g{gemm_idx}")
                cfg = entry.get(direction, {})
                duration = cfg.get("duration")
                if duration is None:
                    raise ValueError(
                        f"Missing duration for transformer entry '{entry_name}' in direction '{direction}'"
                    )

                gemm_node = simulate_LLM.Node(
                    name=self._format_gemm_name(entry_name, direction, micro_batch, layer_index, tp_rank),
                    op_id=self._next_op_id(),
                    hw_id=hw_id,
                    duration=duration,
                    fwd=(direction == "forward"),
                )
                gemm_node.stage_id = stage_id
                gemm_node.tp_rank = tp_rank
                gemm_node.micro_batch_index = micro_batch
                gemm_node.layer_index = layer_index
                gemm_node.direction = direction

                if previous is not None:
                    previous.add_child(gemm_node)
                previous = gemm_node
                if head is None:
                    head = gemm_node

                for comm_key in cfg.get("comm_keys", []):
                    comm_edge = self._create_transformer_comm_edge(
                        comm_key,
                        hw_id,
                        stage_id,
                        micro_batch,
                        layer_index,
                        direction,
                        tp_rank,
                    )
                    previous.add_child(comm_edge)
                    previous = comm_edge

            if head is None:
                raise ValueError("Transformer expansion produced no GEMM nodes")

            rank_heads.append(head)
            rank_tails.append(previous or head)

        dp_children: List[Any] = []
        other_children: List[Any] = []

        for child in getattr(node, "children", []):
            comm_type = getattr(child, "comm_interconnect_type", None)
            if comm_type == "dp":
                dp_children.append(child)
            else:
                other_children.append(child)

        # Keep the main trunk pointing to the per-rank compute tails.
        downstream_parents: List[Any] = list(rank_tails)

        # Attach DP collectives as side branches from the compute tails, without
        # reparenting the trunk. This preserves the true cross-layer pipeline
        # edge between compute nodes for ET conversion.
        for child in dp_children:
            child_clone = self._clone(child)
            if child_clone is None:
                continue
            self._attach(rank_tails[0], child_clone) # only attach to the first tail for DP collectives

        # Non-DP edges (e.g., cross_layer) stay on the trunk so (parent, target)
        # compute → compute pipeline edges remain visible.
        for child in other_children:
            # Special-case marked cross_layer edges (set in original graph):
            # create one per TP rank and wire tail[r] -> cross_layer_r -> next_head[r].
            is_pipeline_edge = False
            if isinstance(child, simulate_LLM.Edge):
                comm_type = getattr(child, "comm_type", None)
                if comm_type == "pipeline":
                    is_pipeline_edge = True
            if is_pipeline_edge:
                # Determine per-rank byte size (ceil split)
                try:
                    total_bytes = int(getattr(child, "comm_size_bytes", 0))
                except Exception:
                    total_bytes = 0
                per_rank_bytes = int(math.ceil(float(total_bytes) / float(max(1, self._tp_degree))))

                # Clone the original targets of this pipeline edge
                target_clones: List[Any] = []
                for tgt in getattr(child, "children", []):
                    tgt_clone = self._clone(tgt)
                    if tgt_clone is None:
                        continue
                    target_clones.append(tgt_clone)
                if not target_clones:
                    # No downstream target; skip safely
                    continue

                # For each TP rank, create its own pipeline edge and connect
                for r, tail in enumerate(rank_tails):
                    # Create rank-specific pipeline edge
                    edge_obj = simulate_LLM.Edge(
                        name=f"{getattr(child, 'name', '')}_rank{r}",
                        op_id=self._next_op_id(),
                        duration=0,
                        is_all_reduce=False,
                        comm_size_bytes=per_rank_bytes,
                        comm_type="pipeline",
                        participants=2,
                        comm_interconnect_type="lp",
                    )
                    edge_obj.is_cross_layer = True
                    tail.add_child(edge_obj)
                    # Also anchor to the compute node (two parents) for mapping clarity
                    last_compute = rank_heads[r]
                    # Find the nearest compute ancestor for this rank: walk back from tail if needed
                    compute_anchor = None
                    cur = tail
                    visited_ids = set()
                    while cur is not None and id(cur) not in visited_ids:
                        visited_ids.add(id(cur))
                        if isinstance(cur, simulate_LLM.Node):
                            compute_anchor = cur
                            break
                        parents = getattr(cur, "parents", [])
                        cur = parents[-1] if parents else None
                    if compute_anchor is not None and compute_anchor is not edge_obj:
                        compute_anchor.add_child(edge_obj)

                    # Connect to each cloned target, aligning ranks where possible
                    for tgt_clone in target_clones:
                        if isinstance(tgt_clone, (list, tuple)):
                            # Map by identity index when available
                            idx = r % len(tgt_clone)
                            edge_obj.add_child(tgt_clone[idx])
                        else:
                            edge_obj.add_child(tgt_clone)
                continue

            # Default path for non-pipeline children
            child_clone = self._clone(child)
            if child_clone is None:
                continue
            self._attach(downstream_parents, child_clone)

        heads_tuple = tuple(rank_heads)
        self._clone_cache[node_id] = heads_tuple
        return heads_tuple

    def _create_transformer_comm_edge(
        self,
        comm_key: str,
        hw_id: int,
        stage_id: int,
        micro_batch: Optional[int],
        layer_index: Optional[int],
        direction: str,
        tp_rank: int,
    ) -> simulate_LLM.Edge:
        comm_info = self.transformer_graph.comm_metadata.get(comm_key, {})
        is_all_reduce = comm_info.get("type") == "all_reduce"

        comm_edge = self.transformer_graph.create_comm_edge(
            name=comm_key,
            op_id=self._next_op_id(),
            comm_key=comm_key,
            is_all_reduce=is_all_reduce,
            local_hw_id=hw_id,
        )
        comm_edge.stage_id = stage_id
        comm_edge.micro_batch_index = micro_batch
        comm_edge.layer_index = layer_index
        comm_edge.direction = direction
        comm_edge.tp_rank = tp_rank
        return comm_edge

    def _copy_metadata(self, source: Any, target: Any) -> None:
        for attr in (
            "micro_batch_index",
            "layer_index",
            "direction",
            "stage_id",
            "tp_rank",
            "is_kv_cache",
        ):
            if hasattr(source, attr):
                setattr(target, attr, getattr(source, attr))

    def _attach(self, parent: Any, child: Any) -> None:
        if parent is None or child is None:
            return

        if isinstance(parent, (list, tuple)):
            for item in parent:
                self._attach(item, child)
            return

        if isinstance(child, (list, tuple)):
            for item in child:
                self._attach(parent, item)
            return

        parent.add_child(child)

    def _format_gemm_name(
        self,
        base_name: str,
        direction: str,
        micro_batch: Optional[int],
        layer_index: Optional[int],
        tp_rank: int,
    ) -> str:
        return f"{base_name}_{direction}_mb{micro_batch}_l{layer_index}_rank{tp_rank}"

    def _next_op_id(self) -> int:
        self._op_id_counter += 1
        return self._op_id_counter

    def _hw_id_for_rank(self, stage_id: int, tp_rank: int) -> int:
        stage_int = int(stage_id) if stage_id is not None else 0
        return stage_int * self._tp_degree + tp_rank

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
        self.output_dir = os.path.abspath(output_dir) if output_dir else os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
        self._generate_graphs = _env_flag("DEEPFLOW_VISUALIZE_GRAPHS")
        self.persist_astrasim_artifacts = _env_flag("DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS")
        self.execution_mode = execution_mode
        inference_cfg = getattr(hw_config, "inference_config", None)
        if inference_cfg and getattr(inference_cfg, "kvcache_precision", None) is not None:
            self.kv_cache_precision = inference_cfg.kvcache_precision
        else:
            self.kv_cache_precision = self.precision
        self.kv_cache_type = getattr(inference_cfg, "kvcache_type", "hbm_only") if inference_cfg else "hbm_only"
        self.kv_cache_fetch_overlap = bool(getattr(inference_cfg, "kvcache_fetch_overlap", False)) if inference_cfg else False
        self.all_reduce = self.model.all_reduce # when the reduce happens in data parallelism options: "the end"  "every layer"
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
        """Return tensor-parallel degree used for sequence-parallel collectives."""

        return self.tp if getattr(self, "tp_sp", False) else 1

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
    # def sequence
    def get_tensor_reduction_time(self, size_bytes: int, kind: str, name: str ) -> float:
        
        reduction_time = self.network_model.collective(
                kind=kind, 
                size_bytes=size_bytes,#for each rank or total?
                participants=int(self.tp),
                ib=self.IBTP,
                ll=self.LLTP,
                local_bytes= 0, 
                debug_label=name or "comm",
        )
        return reduction_time
        
        
    def _tensor_parallelism_gemm_forward(self, gemm: Tuple[int, ...], name: str, tensor_type = None, comm_after = False) -> Tuple[float, float]:
        # tensor_type:"row", "column" determines the way gemm is distributed
        # comm_after: whether there is communication after gemm, for example, in attention output projection and ffn2
        
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        # reduction_time = 0.0
        
        if batch > 1: # for attention gemm only
            gemm_time = self.getGEMMTime(m, k, n, name)[0] * batch/self.tp #each gemm is independent, so multiply by batch size and divide by tp
            print("calculating attention gemm time of shape:", (m,k,n))
            print(f"gemm_time for attention is multiplied by batch/tp: {batch}/{self.tp}")
        else: # for qkv, output_projection, ffn1, ffn2
            if tensor_type == "column": 
                gemm_time = self.getGEMMTime(m, k, math.ceil(n / self.tp), name)[0]
                print("calculating column wise gemm time of shape:", (m,k,math.ceil(n / self.tp)))
            elif tensor_type == "row":
                gemm_time = self.getGEMMTime(m, math.ceil(k/self.tp), n, name)[0]
                print("calculating row wise gemm time of shape:", (m,math.ceil(k/self.tp),n))
            else:
                raise ValueError(f"Invalid tensor type: {tensor_type}")
        if comm_after: #only after output_projection and ffn2
            if tensor_type == "column":
                size_bytes = math.ceil(self.precision * m * n / self.tp )
                print(f"communication after gemm of size {m,n} divided by tp {self.tp}: {size_bytes}")
            elif tensor_type == "row":
                size_bytes = math.ceil(self.precision * m * n )
                print(f"communication after gemm of size {m,n} without tp division: {size_bytes}")
            



        return gemm_time, size_bytes if comm_after else 0
    def _tensor_parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, tensor_type = None, comm_after = False) -> Tuple[float, float]:
        # tensor_type:"row", "column" determines the way gemm is distributed
        # comm_after: whether there is communication after gemm, for example, in attention output projection and ffn2
        """
        in backward pass, for each gemm, we need to calculate the time for gradient w.r.t activation and weight
        for attention gemm, we can simply multiply the time of one gemm by batch size and divide by tp
        the all reduce happens after the qkv projection and ffn1 gemm
        reduction size is the size of activation
        """
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        reduction_time = 0.0
        
        if batch > 1: # for attention gemm only
            grad_time_act = self.getGEMMTime(m, n, k, name)[0] * batch/self.tp #each gemm is independent, so multiply by batch size and divide by tp
            grad_time_wt = self.getGEMMTime(k, m, n, name)[0] * batch/self.tp
            print("calculating attention gemm time of shape: ", (m,n,k) , (k,m,n))
            print(f"grad_time for attention is multiplied by batch/tp: {batch}/{self.tp}")
            # print("calculating attention gemm time:", (k,m,n))
        else: # for qkv, output_projection, ffn1, ffn2
            if tensor_type == "column": 
                act_bytes = math.ceil(self.precision * m * k )
                grad_time_act = self.getGEMMTime(m, math.ceil(n / self.tp), k, name)[0]
                grad_time_wt = self.getGEMMTime(k, m, math.ceil(n / self.tp), name)[0]
                print("calculating column wise gemm time of shape for gradient:", (m,math.ceil(n / self.tp),k), (k,m,math.ceil(n / self.tp)))
            elif tensor_type == "row":
                act_bytes = math.ceil(self.precision * m * math.ceil(k/self.tp))
                grad_time_act = self.getGEMMTime(m, n, math.ceil(k/self.tp), name)[0]
                grad_time_wt = self.getGEMMTime(math.ceil(k/self.tp), m, n, name)[0]
                print("calculating row wise gemm time of shape for gradient:", (m, n, math.ceil(k/self.tp)), (math.ceil(k/self.tp), m, n))
            else:
                raise ValueError(f"Invalid tensor type: {tensor_type}")
        # if comm_after: #only after qkv_projection and ffn1 in backward pass
            
        #     # size_bytes = math.ceil(self.precision * m * n / self.tp ) 
        #     print("all reduce size in backward:", act_bytes)
        #     reduction_time = self.network_model.collective(
        #         kind="all_reduce", 
        #         size_bytes=act_bytes,#for each rank or total?
        #         participants=int(self.tp),
        #         ib=self.IBTP,
        #         ll=self.LLTP,
        #         local_bytes= 0, 
        #         debug_label=name or "comm",
        # )
        gemm_time = grad_time_act + grad_time_wt

        return gemm_time,  act_bytes if comm_after else 0
                
                
    def _distributed_gemm_forward(self, gemm: Tuple[int, ...], name: str, tensor_type=None, comm_after = False) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        if self.t is None or (self.kp1 == 1 and self.kp2 == 1):
            gemm_time, _, _, mem_access = self.getGEMMTime(m, k, n, name)
            # dram_access = mem_access[3]
            # in GB:
            # dram_access = dram_access / 1e9
            # print(f"{name} gemm_time: {gemm_time}, mem_access: {dram_access} GB")
            reduction_time = 0.0
            gemm_time *= batch # handled internally for CR and RC
        elif self.t == "CR":
            gemm_time, reduction_time = self.getDistGEMM_f_kp1(m, k, n, self.kp1, name, batch = batch)
        elif self.t == "RC":
            gemm_time, reduction_time = self.getDistGEMM_f_kp2(m, k, n, self.kp1, self.kp2, name, batch = batch)
        else:
            raise ValueError(f"Invalid tensor parallel strategy: {self.t}")

        return gemm_time, reduction_time


    def _distributed_gemm_backward(self, gemm: Tuple[int, ...], name: str) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        if self.t is None or (self.kp1 == 1 and self.kp2 == 1):
            grad_act_time, _, _, _ = self.getGEMMTime(m, n, k, f"{name}_act")
            grad_wt_time, _, _, _ = self.getGEMMTime(k, m, n, f"{name}_wt")
            gemm_time = grad_act_time + grad_wt_time
            reduction_time = 0.0
            gemm_time *= batch # handled internally for CR and RC
        elif self.t == "CR":
            gemm_time, reduction_time = self.getDistGEMM_b_kp1(m, k, n, self.kp1, name, batch = batch)
        elif self.t == "RC":
            gemm_time, reduction_time = self.getDistGEMM_b_kp2(m, k, n, self.kp1, self.kp2, name, batch = batch)
        else:
            raise ValueError(f"Invalid tensor parallel strategy: {self.t}")
        return gemm_time, reduction_time


    def get_embedding_f(self):
        """
        Calculates the total time required for embedding operations, including computation and data transfer.
        """
        batch = self._effective_transformer_batch()
        embedding_mem = 2 * self.seq_len * batch * self.hidden_dim * self.precision
        embedding_time = self.roofline(0, embedding_mem, name="embedding_f") + self.O
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
        return embedding_time + embedding_transfer_time

    def get_linear_softmax_f(self, gemm):
        """Estimate time for final projection + softmax forward."""
        _, effective_m, k, n = self._effective_dims(gemm)
        # if self.tp >1:
        # gemm_time = 0 
        # reduction_time = 0
        gemm_time,  size_bytes = self._tensor_parallelism_gemm_forward(gemm, "linear_softmax_f", tensor_type="row", comm_after=True)
        reduction_time = self.get_tensor_reduction_time(size_bytes, kind="all_reduce", name="linear_softmax_f")
        point_flop = effective_m * (3 * n - 1)
        point_mem = self.precision * effective_m * (7 * n)
        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise-linear-softmax-f")
            + 4 * self.O
        )

        if self.debug:
            print(
                "Linear Softmax (f) point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("Linear Softmax (f) point_time: {:,}\n".format(point_time))

        return gemm_time + reduction_time + point_time
    def get_scale_softmax_f(self, gemm):
        """
        Process blocks of keys/values at a time (tiling).
        For each block, apply the online softmax trick to update running max & denominator
        each block after QK^T results in a s*s 2D matrix
        model each block with a gemm of shape (s,1,s)
        multiple by n_heads and batch size
        the total time should be divided by tensor parallelism degree
        """
        print("Calculating scale softmax forward")
        print("GEMM:", gemm)

        time = gemm[0] * self.getGEMMTime(gemm[1], 1, gemm[3], "scale_softmax_f")[0] / self.tp
        
        return time 
    
    def get_scale_softmax_b(self, gemm):
        _, effective_m, _, n = self._effective_dims(gemm)
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        reduction_time = 0.0
        

        scale_flop = effective_m * n
        scale_mem = self.precision * effective_m * n * 3

        # Backward softmax uses forward probabilities and gradient accumulation (≈6 ops/elt)
        softmax_flop = effective_m * n * 6
        softmax_mem = self.precision * effective_m * n * 10

        scale_time = (
            self.roofline(scale_flop, scale_mem, name="pointwise-scale-b")
            + self.O
        )
        softmax_time = (
            self.roofline(softmax_flop, softmax_mem, name="pointwise-softmax-b")
            + 4 * self.O
        )

        if self.debug:
            print(
                "Scale (b) flop: {:,}, mem: {:,}".format(
                    int(scale_flop / 1e9), int(scale_mem / 1e9)
                )
            )
            print("Scale (b) time: {:,}".format(scale_time))
            print(
                "Softmax (b) flop: {:,}, mem: {:,}".format(
                    int(softmax_flop / 1e9), int(softmax_mem / 1e9)
                )
            )
            print("Softmax (b) time: {:,}\n".format(softmax_time))

        return scale_time + softmax_time
    def get_residual_f(self, tensor_shape):
        _, elements, _, _ = self._effective_dims(tensor_shape)
        flops = 2 * elements  # add + bias
        mem = self.precision * elements * 3  # read main, read residual, write out

        time = self.roofline(flops, mem, name="pointwise-residual-f") + self.O

        if self.debug:
            print(
                "Residual (f) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Residual (f) time: {:,}\n".format(time))

        return time

    def get_residual_b(self, tensor_shape):
        _, elements, _, _ = self._effective_dims(tensor_shape)
        flops = elements  # dL/dx = dL/dy passthrough
        mem = self.precision * elements * 3  # read grad, read forward residual, write grad

        time = self.roofline(flops, mem, name="pointwise-residual-b") + self.O

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
        compute_flops = elements * hidden * 40  # approx 40 flops per element for GELU
        mem_bytes = self.precision * elements * hidden * 2  # read, write

        time = self.roofline(compute_flops, mem_bytes, name="pointwise-gelu-f") + 2 * self.O

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
        compute_flops = elements * hidden * 80  # approx 80 flops per element for GELU backward
        mem_bytes = self.precision * elements * hidden * 3  # read grad, read forward, write grad

        time = self.roofline(compute_flops, mem_bytes, name="pointwise-gelu-b") + 3 * self.O

        if self.debug:
            print(
                "GELU (b) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(compute_flops / 1e9), int(mem_bytes / 1e9)
                )
            )
            print("GELU (b) time: {:,}\n".format(time))

        return time
    def get_layernorm_f(self, batch, seq_len, d_model, comm_after=False):
        seq_degree = self._sequence_parallel_degree()
        if seq_degree == 1:
            elements = batch * seq_len * d_model
        else:  # sequence parallelism partitions tokens
            elements = batch * math.ceil(seq_len / seq_degree) * d_model
        compute_flops = elements * 7
        mem_bytes = self.precision * elements * 2

        if seq_degree > 1:  # all-gather after layernorm when sequence parallelism is used
            size_bytes = self.precision * elements
            reduction_time = self.network_model.collective(
                kind="all_gather", 
                size_bytes=size_bytes,#for each rank or total?
                participants=int(seq_degree),
                ib=self.IBTP,
                ll=self.LLTP, #TODO fix the bandwidth and latency for sequence parallelism
                local_bytes= 0, 
                debug_label="layernorm_f_all_gather",
        )
        else:
            reduction_time = 0.0
            size_bytes = 0

        compute_time = self.roofline(compute_flops, mem_bytes, name="pointwise-layernorm-f") + 3 * self.O


        return compute_time, reduction_time, size_bytes
    
    

    def get_layernorm_b(self, batch, seq_len, d_model, comm_after=False):
        seq_degree = self._sequence_parallel_degree()
        if seq_degree == 1:
            elements = batch * seq_len * d_model
        else:  # sequence parallelism partitions tokens
            elements = batch * math.ceil(seq_len / seq_degree) * d_model
        compute_flops = elements * 14
        mem_bytes = self.precision * elements * 4

        compute_time = self.roofline(compute_flops, mem_bytes, name="pointwise-layernorm-b") + 4 * self.O
        if seq_degree > 1:  # all-gather after layernorm when sequence parallelism is used
            size_bytes = self.precision * elements
            reduction_time = self.network_model.collective(
                kind="all_gather", 
                size_bytes=size_bytes,#for each rank or total?
                participants=int(seq_degree),
                ib=self.IBTP,
                ll=self.LLTP,
                local_bytes= 0, 
                debug_label="layernorm_b_all_gather",
        )
            
        else:
            reduction_time = 0.0
            size_bytes = 0


        return compute_time, reduction_time, size_bytes
    def get_linear_softmax_b(self, gemm):
        _, effective_m, k, n = self._effective_dims(gemm)
        gemm_time, reduction_time = self._distributed_gemm_backward(gemm, "linear_softmax_b")

        point_flop = effective_m * n * 6
        point_mem = self.precision * effective_m * n * 10

        point_time = (
            self.roofline(point_flop, point_mem, name="pointwise-linear-softmax-b")
            + 4 * self.O
        )

        if self.debug:
            print(
                "Linear Softmax (b) point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("Linear Softmax (b) point_time: {:,}\n".format(point_time))

        return gemm_time + reduction_time + point_time
    
    def get_embedding_b(self):
        batch = self._effective_transformer_batch()
        embedding_mem = 2 * self.seq_len * batch * self.hidden_dim * self.precision
        embedding_mem_time = self.roofline(0, embedding_mem, name="embedding_b") + self.O

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_mem_time
    
    def check_memory(self, hw_config, model_config): #check whether memory usage exceeds capacity
        """Check memory usage."""
        total_mem_capacity = self.memory_capacity # in bytes
        print(f"Total Memory Capacity: {total_mem_capacity / 1e9} GB")
        total_mem = LLM_util.getTotMemReq(hw_config, model_config)[0]
        print(f"Total Memory Usage estimation: {total_mem / 1e9} GB")
        if total_mem > total_mem_capacity:
            print("Warning: Total memory usage exceeds memory capacity!")
            sys.exit("Program terminated due to memory capacity exceeded.")

        return

    # This function should be deleted.
    # def get_dp_overhead(self, d, ffn_dim, n_layers): #calculate the reduction time and apply_grad_time for data parallelism 

    #     reduction_time = 0
    #     apply_grad_time = 0

    #     reduction_time += self.getR(Dim0=d, Dim1=3*d, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="qkv_proj reduction") #get the reduction time between all nodes in one data parallel group
    #     apply_grad_time += self.applyGrad(Dim0=d, Dim1=3*d, name="qkv_proj grad")

    #     reduction_time += self.getR(Dim0=d, Dim1=d, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="output_proj reduction")
    #     apply_grad_time += self.applyGrad(Dim0=d, Dim1=d, name="output_proj grad")

    #     reduction_time += 2*self.getR(Dim0=ffn_dim, Dim1=d, p=self.dp, ib=self.IBD, ll=self.LLD, partial=False, allReduce=True, name="ffn reduction")
    #     apply_grad_time += 2*self.applyGrad(Dim0=ffn_dim, Dim1=d, name="ffn grad")
    #     print(f"apply_grad_time: {apply_grad_time}")
        
    #     if self.all_reduce == "the end":
    #         reduction_time *= n_layers
    #         apply_grad_time *= n_layers
    #     elif self.all_reduce == "every layer": #return the reduction time for each layer
    #         pass
    #     else:
    #         sys.exit("Invalid all_reduce option")
    #     return reduction_time , apply_grad_time
    
    
    def get_inter_layer_comm_latency_llm(self, batch_size, hidden_dim, seq_len): #calculate the cross-layer communication latency
        w = 0
        w_size = 0
        if self.lp > 1:
            w_size = self.precision * batch_size * hidden_dim * seq_len
            transfer_time = w_size / self.IBL + self.LLL
            mem_time = self.roofline(0, 2 * w_size, name="inter_layer")
            # 2: read from memory of previous layer and write to the memory of the next layer
            w = mem_time + transfer_time
        return w, w_size
    def get_data_parallel_reduction_sizes(self, d, ffn_dim):
        """Calculate communication sizes for data parallel reductions (no timing)."""
        if not getattr(self, "dp", 1) or self.dp <= 1:
            # No communication needed for dp=1
            return {
                'qkv_size': 0,
                'output_size': 0,
                'ffn_size': 0,
                'total_size': 0
            }

        # Calculate sizes only (no timing)
        qkv_size = math.ceil(self.precision * d * 3 * d)
        output_size = math.ceil(self.precision * d * d)
        ffn_size = math.ceil(self.precision * ffn_dim * d)
        total_size = qkv_size + output_size + 2 * ffn_size  # FFN appears twice

        return {
            'qkv_size': qkv_size,
            'output_size': output_size,
            'ffn_size': ffn_size,
            'total_size': total_size
        }

    def get_data_parallel_local_computation(self, d, ffn_dim):
        """Calculate local computation times for apply_grad operations."""
        qkv_local = self.apply_grad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")
        output_local = self.apply_grad(Dim0=d, Dim1=d, name="output_proj reduction")
        ffn_local = 2 * self.apply_grad(Dim0=ffn_dim, Dim1=d, name="ffn reduction")

        return {
            'qkv_local': qkv_local,
            'output_local': output_local,
            'ffn_local': ffn_local,
            'total_local': qkv_local + output_local + ffn_local
        }

    def get_data_parallel_reduction_llm(self, d, ffn_dim):
        # If no data parallelism, still apply gradients locally but no cross-device reduction
        if not getattr(self, "dp", 1) or self.dp <= 1:
            apply_grad_time = 0.0
            apply_grad_time += self.apply_grad(Dim0=d, Dim1=3*d, name="qkv_proj reduction")
            apply_grad_time += self.apply_grad(Dim0=d, Dim1=d, name="output_proj reduction")
            apply_grad_time += 2 * self.apply_grad(Dim0=ffn_dim, Dim1=d, name="ffn reduction")
            if self.debug:
                print(f"(dp=1) apply_grad_time: {apply_grad_time}")
            return apply_grad_time

        # k = 2 * self.D
        # n = 4 * self.D
        # dim1 = self.kp_hidden_dim1
        # dim2 = self.kp_hidden_dim2
        w_data = 4*d*d + 2*ffn_dim*d # total parameters need to be reduced
        reduction_time = 0.0
        apply_grad_time = 0.0

        total_bytes = math.ceil(self.precision * d * 3 * d)
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

        total_bytes = math.ceil(self.precision * d * d)
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

        total_bytes = math.ceil(self.precision * ffn_dim * d)
        reduction_time += 2 * self.network_model.collective(
            kind="all_reduce",
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.IBD,
            ll=self.LLD,
            local_bytes=0.0,
            debug_label="ffn reduction",
        )
        apply_grad_time += 2 * self.apply_grad(Dim0=ffn_dim, Dim1=d, name="ffn reduction")

        if self.debug:
            print(f"apply_grad_time: {apply_grad_time}")

        return reduction_time + apply_grad_time
    
    
    
    def compute_all_gemm_and_node_times(self, batch_size, vocab_size, hidden_dim, seq_len, num_heads, kv_heads, ffn_dim):
        """Compute latency for all GEMM operations and node breakdown times in one function."""

        # Process GEMM shapes
        gemm_shapes = LLM_util.process_gemm_shapes(
            batch_size, seq_len, hidden_dim, num_heads, kv_heads, ffn_dim, vocab_size, option="multiply_batch_into_m"
        )
        print("generating gemm shapes for transformer batch size:", batch_size, "seq_len:", seq_len, "hidden_dim:", hidden_dim, "num_heads:", num_heads,  "kv_heads", kv_heads, "ffn_dim:",ffn_dim)
        gemm_qkv_proj = gemm_shapes["qkv_proj"]
        gemm_attention_score = gemm_shapes["attention_score"]
        gemm_attention_output = gemm_shapes["attention_output"]
        gemm_output_proj = gemm_shapes["output_proj"]
        gemm_ffn1 = gemm_shapes["ffn1"]
        gemm_ffn2 = gemm_shapes["ffn2"]
        gemm_linear = gemm_shapes["linear"]

        # Compute GEMM times and organize in structured dict
        transformer_results = {}
        seq_degree = self._sequence_parallel_degree()
        comm_kind_fwd = "all_reduce" if seq_degree == 1 else "reduce_scatter"
        comm_kind_bwd = "all_reduce" if seq_degree == 1 else "reduce_scatter"
        # QKV Projection GEMM
        qkv_proj_gemm_f,  qkv_proj_size = self._tensor_parallelism_gemm_forward(gemm_qkv_proj, "qkv_projection_f", tensor_type="column")
        # qkv_proj_reduction_f = self.get_tensor_reduction_time(qkv_proj_size, kind="all_reduce", name="qkv_projection_f") if qkv_proj_size else 0
        qkv_proj_gemm_b,  qkv_proj_size_b = self._tensor_parallelism_gemm_backward(gemm_qkv_proj, "qkv_projection_b", tensor_type="column", comm_after=True)
        qkv_proj_reduction_b = self.get_tensor_reduction_time(qkv_proj_size_b, kind=comm_kind_bwd, name="qkv_projection_b") if qkv_proj_size_b else 0
        qkv_proj_f = qkv_proj_gemm_f 
        qkv_proj_b = qkv_proj_gemm_b + qkv_proj_reduction_b
        transformer_results['qkv_proj'] = {
            'forward': qkv_proj_f, 'backward': qkv_proj_b,
            'forward_gemm': qkv_proj_gemm_f, 'forward_reduction': 0,
            'backward_gemm': qkv_proj_gemm_b, 'backward_reduction': qkv_proj_reduction_b,
            'comm_size': qkv_proj_size_b
        }

        # Attention Score GEMM
        attn_score_gemm_f,  _ = self._tensor_parallelism_gemm_forward(gemm_attention_score, "attention_score_f")
        # attn_score_reduction_f = self.get_tensor_reduction_time(attn_score_size, kind="all_reduce", name="attention_score_f") if attn_score_size else 0
        attn_score_gemm_b,  _ = self._tensor_parallelism_gemm_backward(gemm_attention_score, "attention_score_b")
        # attn_score_reduction_b = self.get_tensor_reduction_time(attn_score_size, kind="all_reduce", name="attention_score_b") if attn_score_size else 0
        attention_score_f = attn_score_gemm_f 
        attention_score_b = attn_score_gemm_b 
        transformer_results['attention_score'] = {
            'forward': attention_score_f, 'backward': attention_score_b,
            'forward_gemm': attn_score_gemm_f, 
            'backward_gemm': attn_score_gemm_b, 
            
        }

        # Attention Output GEMM
        attn_out_gemm_f,  _ = self._tensor_parallelism_gemm_forward(gemm_attention_output, "attention_output_f")
        # attn_out_reduction_f = self.get_tensor_reduction_time(attn_out_size, kind="all_reduce", name="attention_output_f") if attn_out_size else 0
        attn_out_gemm_b,  _ = self._tensor_parallelism_gemm_backward(gemm_attention_output, "attention_output_b")
        # attn_out_reduction_b = self.get_tensor_reduction_time(attn_out_size, kind="all_reduce", name="attention_output_b") if attn_out_size else 0
        attention_output_f = attn_out_gemm_f 
        attention_output_b = attn_out_gemm_b 
        transformer_results['attention_output'] = {
            'forward': attention_output_f, 'backward': attention_output_b,
            'forward_gemm': attn_out_gemm_f, 
            'backward_gemm': attn_out_gemm_b, 
        }

        # Output Projection GEMM
        out_proj_gemm_f, out_proj_size = self._tensor_parallelism_gemm_forward(gemm_output_proj, "output_projection_f", tensor_type="row", comm_after=True)
        out_proj_reduction_f = self.get_tensor_reduction_time(out_proj_size, kind=comm_kind_fwd, name="output_projection_f") if out_proj_size else 0
        out_proj_gemm_b,  _ = self._tensor_parallelism_gemm_backward(gemm_output_proj, "output_projection_b", tensor_type="row", comm_after=False)
        output_proj_f = out_proj_gemm_f + out_proj_reduction_f
        output_proj_b = out_proj_gemm_b 
        transformer_results['output_proj'] = {
            'forward': output_proj_f, 'backward': output_proj_b,
            'forward_gemm': out_proj_gemm_f, 'forward_reduction': out_proj_reduction_f,
            'backward_gemm': out_proj_gemm_b, 'backward_reduction': 0,
            'comm_size': out_proj_size
        }

        # FFN1 GEMM
        ffn1_gemm_f,  ffn1_size = self._tensor_parallelism_gemm_forward(gemm_ffn1, "ffn_f", tensor_type="column")
        ffn1_gemm_b,  ffn1_size_b = self._tensor_parallelism_gemm_backward(gemm_ffn1, "ffn_b", tensor_type="column", comm_after=True)
        # ffn1_reduction_f = self.get_tensor_reduction_time(ffn1_size, kind="all_reduce", name="ffn_f") if ffn1_size else 0
        ffn1_reduction_b = self.get_tensor_reduction_time(ffn1_size_b, kind=comm_kind_bwd, name="ffn_b") if ffn1_size_b else 0
        ffn1_f = ffn1_gemm_f 
        ffn1_b = ffn1_gemm_b + ffn1_reduction_b
        transformer_results['ffn1'] = {
            'forward': ffn1_f, 'backward': ffn1_b,
            'forward_gemm': ffn1_gemm_f, 'forward_reduction': 0,
            'backward_gemm': ffn1_gemm_b, 'backward_reduction': ffn1_reduction_b,
            'comm_size': ffn1_size_b
        }

        # FFN2 GEMM
        ffn2_gemm_f,  ffn2_size = self._tensor_parallelism_gemm_forward(gemm_ffn2, "ffn2_f", tensor_type="row", comm_after=True)
        ffn2_reduction_f = self.get_tensor_reduction_time(ffn2_size, kind=comm_kind_fwd, name="ffn2_f") if ffn2_size else 0
        ffn2_gemm_b,  _ = self._tensor_parallelism_gemm_backward(gemm_ffn2, "ffn2_b", tensor_type="row", comm_after=False)
        ffn2_f = ffn2_gemm_f + ffn2_reduction_f
        ffn2_b = ffn2_gemm_b 
        transformer_results['ffn2'] = {
            'forward': ffn2_f, 'backward': ffn2_b,
            'forward_gemm': ffn2_gemm_f, 'forward_reduction': ffn2_reduction_f,
            'backward_gemm': ffn2_gemm_b, 'backward_reduction': 0,
            'comm_size': ffn2_size
        }
        
            

        # Calculate non-GEMM operations
        embedding_f = self.get_embedding_f()
        embedding_b = self.get_embedding_b()
        transformer_results['embedding'] = {'forward': embedding_f, 'backward': embedding_b}

        attention_scale_softmax_f = self.get_scale_softmax_f(gemm=gemm_attention_score)
        attention_scale_softmax_b = self.get_scale_softmax_b(gemm=gemm_attention_score)
        transformer_results['attention_scale_softmax'] = {'forward': attention_scale_softmax_f, 'backward': attention_scale_softmax_b}

        residual1_f = self.get_residual_f(tensor_shape=gemm_output_proj)
        residual1_b = self.get_residual_b(tensor_shape=gemm_output_proj)
        # transformer_results['residual1'] = {'forward': residual1_f, 'backward': residual1_b}

        layernorm1_f, layernorm_reduction_f, LN1_comm_bytes_f= self.get_layernorm_f(batch=batch_size, seq_len=seq_len, d_model=hidden_dim)
        layernorm1_b, layernorm_reduction_b, LN1_comm_bytes_b= self.get_layernorm_b(batch=batch_size, seq_len=seq_len, d_model=hidden_dim)
        transformer_results['layernorm1'] = {'forward': layernorm1_f + residual1_f, 'forward_reduction':layernorm_reduction_f, 'backward': layernorm1_b + residual1_b, 'backward_reduction': layernorm_reduction_b, 'comm_size_forward': LN1_comm_bytes_f, 'comm_size_backward': LN1_comm_bytes_b}
        

        residual2_f = self.get_residual_f(tensor_shape=gemm_ffn2)
        residual2_b = self.get_residual_b(tensor_shape=gemm_ffn2)
        # transformer_results['residual2'] = {'forward': residual2_f, 'backward': residual2_b}

        layernorm2_f, layernorm_reduction_f, LN2_comm_bytes_f = self.get_layernorm_f(batch=batch_size, seq_len=seq_len, d_model=hidden_dim)
        layernorm2_b, layernorm_reduction_b, LN2_comm_bytes_b = self.get_layernorm_b(batch=batch_size, seq_len=seq_len, d_model=hidden_dim)
        transformer_results['layernorm2'] = {'forward': layernorm2_f + residual2_f, 'forward_reduction':layernorm_reduction_f,'backward': layernorm2_b + residual2_b, 'backward_reduction': layernorm_reduction_b, 'comm_size_forward': LN2_comm_bytes_f, 'comm_size_backward': LN2_comm_bytes_b}
        
        gelu_f = self.get_gelu_f(tensor_shape=gemm_ffn1)
        gelu_b = self.get_gelu_b(tensor_shape=gemm_ffn1)
        transformer_results['gelu'] = {'forward': gelu_f, 'backward': gelu_b}

        linear_softmax_f = self.get_linear_softmax_f(gemm=gemm_linear)
        linear_softmax_b = self.get_linear_softmax_b(gemm=gemm_linear)
        transformer_results['linear_softmax'] = {'forward': linear_softmax_f, 'backward': linear_softmax_b}

        # Calculate MHA and FFN times directly from results dict
        mha_time_f = ( # not including reduction
            transformer_results['qkv_proj']['forward_gemm'] + transformer_results['attention_score']['forward_gemm'] +
            transformer_results['attention_output']['forward_gemm'] + transformer_results['output_proj']['forward_gemm'] +
            transformer_results['attention_scale_softmax']['forward']
        )
        
        
        mha_time_b = ( # not including reduction 
            transformer_results['qkv_proj']['backward_gemm'] + transformer_results['attention_score']['backward_gemm'] +
            transformer_results['attention_output']['backward_gemm'] + transformer_results['output_proj']['backward_gemm'] +
            transformer_results['attention_scale_softmax']['backward']
        )
        transformer_results['MHA'] = {'forward': mha_time_f, 'backward': mha_time_b, "forward_reduction": out_proj_reduction_f, "backward_reduction": qkv_proj_reduction_b, "comm_size_forward": out_proj_size, "comm_size_backward": qkv_proj_size_b }

        ffn_time_f = transformer_results['ffn1']['forward_gemm'] + transformer_results['ffn2']['forward_gemm'] + transformer_results['gelu']['forward']
        ffn_time_b = (
            transformer_results['ffn1']['backward_gemm'] + transformer_results['ffn2']['backward_gemm'] + transformer_results['gelu']['backward']
        )
        transformer_results['MLP'] = {'forward': ffn_time_f, 'backward': ffn_time_b, "forward_reduction": ffn2_reduction_f, "backward_reduction": ffn1_reduction_b, "comm_size_forward": ffn2_size, "comm_size_backward": ffn1_size_b }
        # Calculate transformer times directly
        
        transformer_time_f = (
            transformer_results['MHA']['forward'] + transformer_results['MHA']['forward_reduction'] + transformer_results['MLP']['forward'] + transformer_results['MLP']['forward_reduction'] +
            transformer_results['layernorm1']['forward'] + transformer_results['layernorm1']['forward_reduction']+ transformer_results['layernorm2']['forward']+ transformer_results['layernorm2']['forward_reduction']
        )
        transformer_time_b = (
            transformer_results['MHA']['backward'] + transformer_results['MHA']['backward_reduction'] + transformer_results['MLP']['backward'] + transformer_results['MLP']['backward_reduction'] +
            transformer_results['layernorm1']['backward'] + transformer_results['layernorm1']['backward_reduction']+ transformer_results['layernorm2']['backward']+ transformer_results['layernorm2']['backward_reduction']
        )
        output_csv = os.path.join(self.output_dir, "Transformer_breakdown_times.csv")
        with open(output_csv, "w") as f:
            data = [qkv_proj_gemm_f, attn_score_gemm_f, attn_out_gemm_f, out_proj_gemm_f,ffn1_gemm_f, ffn2_gemm_f,
                    attention_scale_softmax_f, layernorm1_f + residual1_f, layernorm2_f + residual2_f ,gelu_f,
                    out_proj_reduction_f, ffn2_reduction_f]
            LLM_util.save_transformer_breakdown_csv(data)
            
        # Write results to file
        output_file = os.path.join(self.output_dir, "Transformer_breakdown_results.txt")
        with open(output_file, "w") as f:
            f.write("Transformer Breakdown Results:\n")
            for key, value in transformer_results.items():
                f.write(f"{key}: {value}\n")
            f.write(f"Total Transformer Forward Time: {transformer_time_f}\n")
            f.write(f"Total Transformer Backward Time: {transformer_time_b}\n")


        node_breakdown = {
            'transformer_time_f': transformer_time_f,
            'transformer_time_b': transformer_time_b,
            'embedding_f': transformer_results['embedding']['forward'],
            'embedding_b': transformer_results['embedding']['backward'],
            'linear_softmax_f': transformer_results['linear_softmax']['forward'],
            'linear_softmax_b': transformer_results['linear_softmax']['backward']
        }

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
    ) -> Dict[str, Dict[str, Any]]:
        return {
            'transformer': {
                'size': reduction_sizes['total_size'],
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': local_comp['total_local']
            },
            'embedding': {
                'size': embedding_size,
                'type': 'all_reduce',
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0
            },
            'softmax': {
                'size': softmax_size,
                'type': 'all_reduce',
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

    def _populate_transformer_comm_metadata(
        self,
        entry: Dict[str, Any],
        metadata: Dict[str, Dict[str, Any]],
        # gemm_spec: Tuple[int, ...],
        comm_bytes_fwd: int,
        comm_bytes_bwd: int,
    ) -> None:
        """Attach tensor-parallel collectives for a GEMM to metadata and entry."""

        # tp_mode = self.t
        # kp1 = int(self.kp1) if self.kp1 else 1
        # kp2 = int(self.kp2) if self.kp2 else 1
        # # precision = self.precision

        # if not tp_mode or (kp1 <= 1 and kp2 <= 1):
        #     return
        if not self.tp or self.tp <= 1:
            return

        # _, effective_m, effective_k, effective_n = self._effective_dims(gemm_spec)

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

        seq_degree = self._sequence_parallel_degree()
        if self.tp > 1 and seq_degree == 1:
            #TODO: correct the communication type here
            add_comm('forward', 'all_reduce', 'all_reduce', comm_bytes_fwd, self.tp, 'tp')
            add_comm('backward', 'all_reduce', 'all_reduce', comm_bytes_bwd, self.tp, 'tp')
        elif self.tp > 1 and seq_degree > 1:
            if entry['name'] in ['layernorm1', 'layernorm2']:
                add_comm('forward', 'all_gather', 'all_gather', comm_bytes_fwd, seq_degree, 'tp')
                add_comm('backward', 'all_gather', 'all_gather', comm_bytes_bwd, seq_degree, 'tp')
            elif entry['name'] in ['MHA', 'MLP']:
                add_comm('forward', 'reduce_scatter', 'reduce_scatter', comm_bytes_fwd, seq_degree, 'tp')
                add_comm('backward', 'reduce_scatter', 'reduce_scatter', comm_bytes_bwd, seq_degree, 'tp')

        

    def _build_interconnect_params(self) -> Dict[str, Tuple[float, float]]:
        return {
            'dp': (self.IBD, self.LLD),
            'lp': (self.IBL, self.LLL),
            'tp': (self.IBTP, self.LLTP),
        }


    def _tp_degree(self) -> int:
        return max(1, int(self.tp))

    def _prepare_execution_graphs(
        self,
        *,
        node_breakdown: Dict[str, float],
        transformer_results: Dict[str, Dict[str, Any]],
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        vocab_size: int,
        include_pipeline_backward: bool,
        include_transformer_backward: bool,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None, # optional override, decode only.
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
            reduction_sizes = self.get_data_parallel_reduction_sizes(hidden_dim, ffn_dim)
            local_comp = self.get_data_parallel_local_computation(hidden_dim, ffn_dim)

        embedding_size = math.ceil(self.precision * vocab_size * hidden_dim) + math.ceil(self.precision * seq_len * hidden_dim)
        softmax_size = math.ceil(self.precision * hidden_dim * vocab_size)
        # Below, we fix pipeline comm sizes for decode
        attn_shape = (gemm_shapes or {}).get("attention_score")
        pipeline_seq_len = max(1, int(attn_shape[1])) if attn_shape and len(attn_shape) > 1 else seq_len

        cross_layer_bytes = self.get_inter_layer_comm_latency_llm(batch_size, hidden_dim, pipeline_seq_len)[1]

        comm_metadata = self._build_comm_metadata(
            reduction_sizes=reduction_sizes,
            local_comp=local_comp, #when doing all_reduce between dp nodes, there is local computation on each node to apply gradients
            embedding_size=embedding_size,
            softmax_size=softmax_size,
            cross_layer_bytes=cross_layer_bytes,
        )

        # shapes = LLM_util.process_gemm_shapes(
        #     batch_size,
        #     seq_len,
        #     hidden_dim,
        #     num_heads,
        #     ffn_dim,
        #     vocab_size,
        #     option="multiply_batch_into_m",
        # )
        
        # gemm_qkv_proj = shapes["qkv_proj"]
        # gemm_attention_score = shapes["attention_score"]
        # gemm_attention_output = shapes["attention_output"]
        # gemm_output_proj = shapes["output_proj"]
        # gemm_ffn1 = shapes["ffn1"]
        # gemm_ffn2 = shapes["ffn2"]
        # transformer_results = compute_all_gemm_and_node_times(
        #     self,
        #     batch_size,
        transformer_operation_entries: List[Dict[str, Any]] = []
        transformer_comm_metadata: Dict[str, Dict[str, Any]] = {}

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

        transformer_graph: Optional[Graph] = None
        transformer_forward_root: Optional[Any] = None
        transformer_backward_root: Optional[Any] = None

        transformer_comp_times = {
            "transformer": {
                "gemms": transformer_operation_entries,
                # "non_gemm": [
                #     {"operation": "scale_softmax", "duration": transformer_results['attention_scale_softmax']['forward']},
                    
                    # {"operation": "transformer_b", "duration": node_breakdown['transformer_time_b']},
                    # {"operation": "embedding_f", "duration": node_breakdown['embedding_f']},
                    # {"operation": "embedding_b", "duration": node_breakdown['embedding_b']},
                    # {"operation": "linear_softmax_f", "duration": node_breakdown['linear_softmax_f']},
                    # {"operation": "linear_softmax_b", "duration": node_breakdown['linear_softmax_b']},
                # ],
                # "tp_degree": tp_degree,
            }
        }

        transformer_graph = simulate_LLM.Graph(
            mode="transformer",
            dp=self.dp,
            lp=self.lp,
            tp=self.tp,
            # kp1=self.kp1,
            # kp2=self.kp2,
            # tp_mode=self.t,
            comp_times=transformer_comp_times,
            comm_metadata=transformer_comm_metadata,
            misc_metadata={},
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
            "kv_cache_fetch": node_breakdown.get('kv_cache_fetch', 0.0),
            "kv_cache_store": node_breakdown.get('kv_cache_store', 0.0),
            "cross_layer_f": 0.0,
            "cross_layer_b": 0.0,
        }
        misc_metadata = {
            "num_batch": self.mb,
            "num_layer": self.num_layers,
            "all_reduce": getattr(self, "all_reduce", "the end"),
        }
        if getattr(self, "kv_cache_fetch_overlap", False):
            misc_metadata["kv_cache_fetch_overlap"] = True

        pipeline_graph_obj = simulate_LLM.Graph(
            mode="pipeline",
            dp=self.dp,
            lp=self.lp,
            tp=self.tp,
            # kp1=self.kp1,
            # kp2=self.kp2,
            # tp_mode=self.t,
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
        ffn_mult = self.ffn_mult
        ffn_dim = self.hidden_dim * ffn_mult if ffn_mult else self.ffn_dim
        
        attention_type = self.attention_type
        kv_heads = self.kv_heads
        
        print("self.tp:", self.tp)

        # Adjust types and calculate node latencies
        self.readjust_type()

        # Get structured transformer results and node breakdown in one efficient call
        # if attention_type =='mha':
        transformer_results, node_breakdown = self.compute_all_gemm_and_node_times(batch_size, vocab_size, hidden_dim, seq_len, num_heads, kv_heads, ffn_dim )
        # elif attention_type =='gqa':
            
            




        print("Calculating LLM time...")

        # Create graph and calculate times

        print("simulating parallelism with dp = {}, lp = {}, total data batch = {}, "
            "for each dp node, data batch = {}, for each pipeline stage, data batch = {}".format(self.dp, self.lp, self.batch_size, self.miniB, self.microB))
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
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            vocab_size=vocab_size,
            include_pipeline_backward=True,
            include_transformer_backward=True,
        )

        self.transformer_graph = transformer_graph
        self.transformer_forward_root = transformer_forward_root
        self.transformer_backward_root = transformer_backward_root
        self.transformer_analytical_time_forward = node_breakdown['transformer_time_f']
        self.transformer_analytical_time_backward = node_breakdown['transformer_time_b']
        self.transformer_graph.save_graph(transformer_forward_root, "output_graph/", "transformer_graph_forward_initial")
        
        self.transformer_graph.save_graph(transformer_backward_root, "output_graph/", "transformer_graph_backward_initial")

        self.pipeline_graph = pipeline_graph_obj
        self.pipeline_root = graph_root
        self.pipeline_interconnect = interconnect_params
        self.pipeline_graph.save_graph(graph_root, "output_graph/", "initial_pipeline_graph")
        
        mode = self.execution_mode
        # start = time.perf_counter()
        # forward_graph, _ = self.pipeline_graph.extract_forward_graph(self.pipeline_root)
        # backward_graph= self.pipeline_graph.extract_backward_graph(self.pipeline_root)
        # elapsed = time.perf_counter() - start
        # print(f"extracting forward graph took {elapsed:.3f}s")
        # setattr(self, "forward_only_root", forward_graph)
        # setattr(self, "backward_only_root", backward_graph)
        # self.pipeline_graph.save_graph(forward_graph, "output_graph/", f"forward_path_before_dispatcher_{mode.value.lower()}")
        # self.pipeline_graph.save_graph(backward_graph, "output_graph/", f"backward_path_before_dispacher_{mode.value.lower()}")
        # dispather_forward = LLMExecutionDispatcher(
        #     time_calc=self,
        #     pipeline_graph=self.pipeline_graph,
        #     pipeline_root=forward_graph,
        #     interconnect_params=self.pipeline_interconnect,
        #     transformer_graph=self.transformer_graph,
        #     transformer_forward_root=self.transformer_forward_root,
        #     transformer_backward_root=self.transformer_backward_root,
        # )
        # dispatcher_backward = LLMExecutionDispatcher(
        #     time_calc=self,
        #     pipeline_graph=self.pipeline_graph,
        #     pipeline_root=backward_graph,
        #     interconnect_params=self.pipeline_interconnect,
        #     transformer_graph=self.transformer_graph,
        #     transformer_forward_root=self.transformer_forward_root,
        #     transformer_backward_root=self.transformer_backward_root,
        # )
        # result_forward = dispather_forward.run(self.execution_mode)
        # # result_backward = dispatcher_backward.run(self.execution_mode)
        # time_fw = result_forward.total_time
        # time_bw = result_backward.total_time
        # print(f"{self.execution_mode.value} pipeline forward execution time: {time_fw:.3f}s")
        # print(f"{self.execution_mode.value} pipeline backward execution time: {time_bw:.3f}s")
        # forward_graph = result_forward.graph_root
        # backward_graph = result_backward.graph_root
        
        
        # self.pipeline_graph.save_graph(forward_graph, "output_graph/", f"pipeline_graph_forward_{mode.value.lower()}")
        # self.pipeline_graph.save_graph(backward_graph, "output_graph/", f"pipeline_graph_backward_{mode.value.lower()}")
        

        
        
        
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
        

        # start = time.perf_counter()
        self.pipeline_graph.save_graph(pipeline_root, "output_graph/", f"pipeline_graph_{mode.value.lower()}")
        # elapsed = time.perf_counter() - start
        # print(f"save_graph took {elapsed:.3f}s")

        # self.pipeline_graph.save_graph(pipeline_root, "output_graph/", f"pipeline_graph_{mode.value.lower()}")
        
        
        # import time





        # self.transformer_forward_root = self.transformer_graph.convert_comm_sizes_to_times(
        #     self.transformer_forward_root,
        #     self.network_model,
        #     self.pipeline_interconnect,
        # )
        # self.transformer_backward_root = self.transformer_graph.convert_comm_sizes_to_times(
        #     self.transformer_backward_root,
        #     self.network_model,
        #     self.pipeline_interconnect,
        # )

        # graph_folder = self.output_dir.rstrip(os.sep) + os.sep
        # if self._generate_graphs and self.transformer_forward_root is not None:
        #     self.transformer_graph.save_graph(
        #         self.transformer_forward_root,
        #         graph_folder,
        #         "transformer_graph_forward",
        #     )
        # if self._generate_graphs and self.transformer_backward_root is not None:
        #     self.transformer_graph.save_graph(
        #         self.transformer_backward_root,
        #         graph_folder,
        #         "transformer_graph_backward",
        #     )
        # if self._generate_graphs and self.pipeline_root is not None:
        #     self.pipeline_graph.save_graph(
        #         self.pipeline_root,
        #         graph_folder,
        #         "pipeline_graph_post",
        #     )

        # debug helper. If set, print analytical transformer time and actual transformer time
        if self._generate_graphs:
            print(f"Analytical transformer forward time: {self.transformer_analytical_time_forward:.4f}s")
            print(f"Analytical transformer backward time: {self.transformer_analytical_time_backward:.4f}s")
            if self.transformer_astrasim_time_forward is not None and self.transformer_astrasim_time_backward is not None:
                print(f"Actual transformer forward time: {self.transformer_astrasim_time_forward:.4f}s")
                print(f"Actual transformer backward time: {self.transformer_astrasim_time_backward:.4f}s")

        self.tot_time = time_fw_bw

        output_file = os.path.join(self.output_dir, "LLM_time_results.txt")
        
        with open(output_file, "w") as f:
            f.write("\n\n==============================================\n")
            f.write("Performance Results\n")
            f.write("==============================================\n")
            # f.write("Forward Time: {0:.8f} {1}\n".format(time_fw * m, second))
            # f.write("Backward Time: {0:.8f} {1}\n".format(time_bw * m, second))
            f.write("Forward + Backward Time: {0:.8f} {1}\n".format(time_fw_bw * m, second))

            # f.write("Total Time: {0:.8f}\n".format(TC.getTime()))

        return time_fw_bw

    def get_time(self):
        return self.tot_time

    def getReductionTotal(self): #not considering congestion
        return getattr(self, "_reduction_total_llm", 0.0)


class LLMExecutionDispatcher:
    def __init__(
        self,
        time_calc: TimeCalculationLLM,
        pipeline_graph: Graph,
        pipeline_root: Any,
        interconnect_params: Dict[str, Tuple[float, float]],
        transformer_graph: Optional[Graph] = None,
        transformer_forward_root: Optional[Any] = None,
        transformer_backward_root: Optional[Any] = None,
    ) -> None:
        self.time_calc = time_calc
        self.pipeline_graph = pipeline_graph
        self.pipeline_root = pipeline_root
        self.interconnect_params = interconnect_params
        self.transformer_graph = transformer_graph
        self.transformer_forward_root = transformer_forward_root
        self.transformer_backward_root = transformer_backward_root
        self.flattened_root: Optional[Any] = None

    def run(self, mode: ExecutionMode) -> ExecutionResult:
        if mode == ExecutionMode.ANALYTICAL:
            print("Running analytical execution...")
            return self._run_pipeline_with_analytical_comm(ExecutionMode.ANALYTICAL)
        if mode == ExecutionMode.HYBRID:
            print("Running hybrid execution...")
            return self._run_hybrid()
        if mode == ExecutionMode.FULL_ASTRASIM_HIERARCHICAL:
            print("Running full AstraSim hierarchical execution...")
            return self._run_full_astrasim_hierarchical()
        if mode == ExecutionMode.FULL_ASTRASIM_FLATTENED:
            print("Running full AstraSim flattened execution...")
            return self._run_full_astrasim_flattened()
        raise ValueError(f"Unsupported execution mode: {mode}")

    def _run_pipeline_with_analytical_comm(self, declared_mode: ExecutionMode) -> ExecutionResult:
        if declared_mode == ExecutionMode.HYBRID:
            filename = "/hybrid_graph"
            timed_root = self.pipeline_graph.convert_comm_sizes_to_times(
                self.pipeline_root,
                self.time_calc.network_model,
                self.interconnect_params,
            )
        else: # must be "ANALYTICAL"
            filename = "/analytical_graph"
            timed_root = self.pipeline_graph.convert_comm_sizes_to_times(
                self.pipeline_root,
                self.time_calc.network_model,
                self.interconnect_params,
            )
            
        generate_graphs = _env_flag("DEEPFLOW_VISUALIZE_GRAPHS")
        if generate_graphs:
            self.pipeline_graph.save_graph(
                self.pipeline_root,
                self.time_calc.output_dir,
                filename,
            )

        # Persist timed root for any downstream consumer
        self.pipeline_root = timed_root
        total_time = self.pipeline_graph.simulate(timed_root)
        self.pipeline_graph.save_graph(timed_root, "output_graph/", f"pipeline_graph_{declared_mode.value.lower()}")
        # print("g")
        print(f"{declared_mode.value} pipeline execution time: {total_time:.3f}s")
        
        return ExecutionResult(total_time=total_time, graph_root=timed_root, mode=declared_mode)

    def _run_hybrid(self) -> ExecutionResult:
        generate_graphs = _env_flag("DEEPFLOW_VISUALIZE_GRAPHS")
        if generate_graphs:
            self.transformer_graph.save_graph(
                self.transformer_forward_root,
                self.time_calc.output_dir,
                "/hybrid_graph_transformer_forward",
            )
            self.transformer_graph.save_graph(
                self.transformer_backward_root,
                self.time_calc.output_dir,
                "/hybrid_graph_transformer_backward",
            )
            
            timed_root = self.transformer_graph.convert_comm_sizes_to_times(
                self.transformer_forward_root,
                self.time_calc.network_model,
                self.interconnect_params,
            )
            timed_root_b = self.transformer_graph.convert_comm_sizes_to_times(
                self.transformer_backward_root,
                self.time_calc.network_model,
                self.interconnect_params,
            )
            self.transformer_graph.save_graph(
                timed_root,
                self.time_calc.output_dir,
                "/hybrid_graph_transformer_deepflow_annotated",
            )
            self.transformer_graph.save_graph(
                timed_root_b,
                self.time_calc.output_dir,
                "/hybrid_graph_transformer_backward_deepflow_annotated",
            )
            timed_root_pipeline = self.pipeline_graph.convert_comm_sizes_to_times(
                self.pipeline_root,
                self.time_calc.network_model,
                self.interconnect_params,
            )
            self.pipeline_graph.save_graph(
                timed_root_pipeline,
                self.time_calc.output_dir,
                "/hybrid_graph_pipeline_deepflow_annotated",
            )
        transformer_time = self._run_transformer_astrasim(ExecutionMode.HYBRID)

        if transformer_time is not None:
            self._apply_transformer_time(transformer_time)
        return self._run_pipeline_with_analytical_comm(ExecutionMode.HYBRID)

    def _run_full_astrasim_hierarchical(self) -> ExecutionResult:
        transformer_time = self._run_transformer_astrasim(ExecutionMode.FULL_ASTRASIM_HIERARCHICAL)
        if transformer_time is not None:
            self._apply_transformer_time(transformer_time)

        dp_count = getattr(self.time_calc, "dp", 1) or 1
        if not self.pipeline_root:
            raise RuntimeError("Pipeline graph root is not available for AstraSim execution")

        # Use hierarchical artifact directory when persisting artifacts
        artifact_dir = self.time_calc.output_dir
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_hier")

        run_kwargs = {
            "persist_artifacts": self.time_calc.persist_astrasim_artifacts,
        }
        run_type = str(getattr(getattr(self.time_calc, "model", None), "run_type", "training")).lower()
        effective_dp = 1 if run_type == "inference" else max(1, getattr(self.time_calc, "dp", 1))
        if run_type == "inference":
            run_kwargs["dp_override"] = 1

        per_rank_sec, max_sec = run_astra_simulation_only_onepath(
            self.pipeline_root,
            self.time_calc,
            artifact_dir,
            **run_kwargs,
        )
        self.time_calc.pipeline_astrasim_per_rank = per_rank_sec
        self.time_calc.pipeline_astrasim_time = max_sec
        if max_sec <= 0:
            raise RuntimeError("AstraSim pipeline execution returned non-positive duration")
        return ExecutionResult(total_time=max_sec, graph_root=self.pipeline_root, mode=ExecutionMode.FULL_ASTRASIM_HIERARCHICAL)

    def _run_full_astrasim_flattened(self) -> ExecutionResult:
        if not self.pipeline_root:
            raise RuntimeError("Pipeline graph root is not available for flattening")
        if not self.transformer_graph:
            raise RuntimeError("Transformer graph metadata is required for flattening")

        # output_dir = "./astra_flattened_graph"
        # os.makedirs(output_dir, exist_ok=True)
        # base_path = os.path.join(output_dir, "pipeline_unflattened")
        # dot = visualize_graph(self.pipeline_root, filename=base_path)
        # try:
        #     dot.render(base_path, format="png", cleanup=True)
        # except Exception as exc:
        #     print(f"[WARN] Failed to render pipeline graph: {exc}")

        flattener = PipelineGraphFlattener(
            pipeline_graph=self.pipeline_graph,
            transformer_graph=self.transformer_graph,
        )
        flattened_root = flattener.build(self.pipeline_root)
        if flattened_root is None:
            raise RuntimeError("Pipeline flattening produced an empty graph")

        self.time_calc.flattened_pipeline_root = flattened_root
        if _env_flag("DEEPFLOW_VISUALIZE_GRAPHS") and self.pipeline_root is not None:
            self.pipeline_graph.save_graph(
                self.pipeline_root,
                self.time_calc.output_dir,
                "/pipeline_graph_pre",
            )
        self.pipeline_root = flattened_root
        self.pipeline_graph.save_graph(
            flattened_root,
            self.time_calc.output_dir,
            "pipeline_graph_flattened",
        )
        # output_dir = "./astra_flattened_graph"
        # os.makedirs(output_dir, exist_ok=True)
        # base_path = os.path.join(output_dir, "pipeline_flattened")
        # dot = visualize_graph(flattened_root, filename=base_path)
        # try:
        #     dot.render(base_path, format="png", cleanup=True)
        # except Exception as exc:  # pragma: no cover - visualization best-effort
        #     print(f"[WARN] Failed to render flattened pipeline graph: {exc}")

        unique_hw_ids = self._collect_hw_ids(flattened_root)
        if not unique_hw_ids:
            raise RuntimeError("Flattened pipeline graph exposes no compute nodes with hardware IDs")

        # Use flattened artifact directory when persisting artifacts
        artifact_dir = self.time_calc.output_dir
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_flat")

        run_kwargs = {
            "persist_artifacts": self.time_calc.persist_astrasim_artifacts,
        }
        run_type = str(getattr(getattr(self.time_calc, "model", None), "run_type", "training")).lower()
        effective_dp = 1 if run_type == "inference" else max(1, getattr(self.time_calc, "dp", 1))
        if run_type == "inference":
            run_kwargs["dp_override"] = 1

        per_rank_sec, max_sec = run_astra_simulation_only_onepath(
            flattened_root,
            self.time_calc,
            artifact_dir,
            **run_kwargs,
        )

        if not per_rank_sec:
            raise RuntimeError("AstraSim flattened execution returned no per-rank timings")

        expected_rank_count = effective_dp * len(unique_hw_ids)
        # Special case: If expected rank count is 1, then 2 is fine, but we prune the extra result
        # this is done, since astrasim backend only supports >1 ranks, so we generate extra fake result for that case.
        if expected_rank_count == 1:
            if len(per_rank_sec) > 2:
                raise RuntimeError(
                    "AstraSim rank count mismatch for flattened execution: "
                    f"expected {expected_rank_count}, got {len(per_rank_sec)}"
                )
            per_rank_sec = per_rank_sec[:1]

        if len(per_rank_sec) != expected_rank_count:
            raise RuntimeError(
                "AstraSim rank count mismatch for flattened execution: "
                f"expected {expected_rank_count}, got {len(per_rank_sec)}"
            )

        if max_sec <= 0:
            raise RuntimeError("AstraSim flattened execution returned non-positive duration")

        self.time_calc.pipeline_astrasim_per_rank = per_rank_sec
        self.time_calc.pipeline_astrasim_time = max_sec
        self.time_calc.flattened_astrasim_per_rank = per_rank_sec
        self.time_calc.flattened_astrasim_total = max_sec

        return ExecutionResult(
            total_time=max_sec,
            graph_root=flattened_root,
            mode=ExecutionMode.FULL_ASTRASIM_FLATTENED,
        )

    def _collect_hw_ids(self, root: Any) -> Set[int]:
        visited: Set[int] = set()
        hw_ids: Set[int] = set()

        def enqueue_children(obj: Any) -> None:
            for child in getattr(obj, "children", []):
                stack.append(child)

        stack: List[Any]
        if isinstance(root, (list, tuple)):
            stack = list(root)
        else:
            stack = [root]

        while stack:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in visited:
                continue
            visited.add(obj_id)

            if isinstance(obj, simulate_LLM.Node):
                hw_id = getattr(obj, "hw_id", None)
                if hw_id is not None and hw_id >= 0:
                    hw_ids.add(int(hw_id))

            enqueue_children(obj)

        return hw_ids



    def _run_transformer_astrasim(self, mode: ExecutionMode) -> Optional[TransformerTimings]:
        del mode  # mode currently unused but kept for signature consistency

        # Use hierarchical artifact directory when persisting artifacts for transformer simulation
        artifact_dir = self.time_calc.output_dir
        artifact_dir_fwd = artifact_dir
        artifact_dir_bwd = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_hier")
            artifact_dir_fwd = os.path.join(artifact_dir, "fwd")
            artifact_dir_bwd = os.path.join(artifact_dir, "bwd")
            os.makedirs(artifact_dir_fwd, exist_ok=True)
            os.makedirs(artifact_dir_bwd, exist_ok=True)

        fwd_per_rank = None
        bwd_per_rank = None
        fwd_max = 0
        bwd_max = 0
        if self.transformer_forward_root:
            fwd_per_rank, fwd_max = run_astra_simulation_only_onepath(
                self.transformer_forward_root,
                self.time_calc,
                artifact_dir_fwd,
                dp_override=1,
                persist_artifacts=self.time_calc.persist_astrasim_artifacts,
            )
        if self.transformer_backward_root:
            bwd_per_rank, bwd_max = run_astra_simulation_only_onepath(
                self.transformer_backward_root,
                self.time_calc,
                artifact_dir_bwd,
                dp_override=1,
                persist_artifacts=self.time_calc.persist_astrasim_artifacts,
            )

        self.time_calc.transformer_astrasim_per_rank_forward = fwd_per_rank
        self.time_calc.transformer_astrasim_per_rank_backward = bwd_per_rank
        self.time_calc.transformer_astrasim_time_forward = fwd_max
        self.time_calc.transformer_astrasim_time_backward = bwd_max

        if fwd_max < 0 or bwd_max < 0:
            raise RuntimeError("AstraSim transformer execution returned non-positive duration")

        return TransformerTimings(forward=fwd_max, backward=bwd_max)

    def _apply_transformer_time(self, timings: TransformerTimings) -> None:
        if timings.forward < 0 or timings.backward < 0:
            raise ValueError("AstraSim transformer times must be positive")

        comp_times = getattr(self.pipeline_graph, "comp_times", None)
        if isinstance(comp_times, dict):
            if "transformer_f" in comp_times:
                print(f"Original transformer forward time: {comp_times['transformer_f']:.3f}s, updating to AstraSim value: {timings.forward:.3f}s")
                comp_times["transformer_f"] = timings.forward
                # print(f"Updated transformer forward time to AstraSim value: {timings.forward:.3f}s")
            if "transformer_b" in comp_times:
                comp_times["transformer_b"] = timings.backward

        visited: Set[int] = set()
        roots: List[Any]
        if isinstance(self.pipeline_root, (list, tuple)):
            roots = list(self.pipeline_root)
        else:
            roots = [self.pipeline_root]

        for root in roots:
            self._assign_transformer_durations(root, visited, timings.forward, timings.backward)

    def _assign_transformer_durations(self, node: Any, visited: Set[int], forward_value: float, backward_value: float) -> None:
        if node is None:
            return
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)

        if isinstance(node, simulate_LLM.Node):
            if node.name == "transformer":
                node.duration = forward_value
            elif node.name == "transformer_b":
                node.duration = backward_value

        for child in getattr(node, "children", []):
            self._assign_transformer_durations(child, visited, forward_value, backward_value)

    def compute_decode_layer_time(
        self,
        *,
        gemm_results: Dict[str, Dict[str, float]],
        batch_size: int,
        seq_len: int,
        total_seq_len: int,
    ) -> Tuple[float, float]:
        head_dim = self.hidden_dim // self.num_heads
        attention_score_shape = (
            batch_size * self.num_heads,
            1,
            head_dim,
            total_seq_len,
        )
        attention_scale_softmax_f = self.get_scale_softmax_f(attention_score_shape)

        token_bytes = LLM_util.kv_cache_token_bytes(
            batch_size=batch_size,
            num_heads=self.num_heads,
            head_dim=head_dim,
            precision_bytes=self.kv_cache_precision,
        )
        kv_cache_fetch_time = self.roofline(
            0,
            token_bytes * total_seq_len,
            name="kv_cache_fetch",
        ) + self.O
        kv_cache_store_time = self.roofline(
            0,
            token_bytes,
            name="kv_cache_store",
        ) + self.O

        output_seq_len = 1

        output_proj_shape = (
            batch_size,
            output_seq_len,
            self.hidden_dim,
            self.hidden_dim,
        )
        residual1_f = self.get_residual_f(output_proj_shape)
        layernorm1_f = self.get_layernorm_f(output_proj_shape)

        ffn_dim = self.hidden_dim * self.ffn_mult if self.ffn_mult else self.ffn_dim

        ffn2_shape = (
            batch_size,
            output_seq_len,
            ffn_dim,
            self.hidden_dim,
        )
        residual2_f = self.get_residual_f(ffn2_shape)
        layernorm2_f = self.get_layernorm_f(ffn2_shape)

        linear_shape = (
            batch_size,
            output_seq_len,
            self.hidden_dim,
            self.vocab_size,
        )
        linear_softmax_f = self.get_linear_softmax_f(linear_shape)
        if "linear" in gemm_results:
            gemm_results["linear"]["forward"] = linear_softmax_f
        gemm_results["kv_cache_fetch"] = {
            "forward": kv_cache_fetch_time,
            "backward": 0.0,
            "forward_gemm": kv_cache_fetch_time,
            "forward_reduction": 0.0,
            "backward_gemm": 0.0,
            "backward_reduction": 0.0,
        }
        gemm_results["kv_cache_store"] = {
            "forward": kv_cache_store_time,
            "backward": 0.0,
            "forward_gemm": kv_cache_store_time,
            "forward_reduction": 0.0,
            "backward_gemm": 0.0,
            "backward_reduction": 0.0,
        }

        core_time = (
            gemm_results["qkv_proj"]["forward"]
            + gemm_results["attention_score"]["forward"]
            + attention_scale_softmax_f
            + gemm_results["attention_output"]["forward"]
            + gemm_results["output_proj"]["forward"]
            + residual1_f
            + layernorm1_f
            + gemm_results["ffn1"]["forward"]
            + gemm_results["ffn2"]["forward"]
            + residual2_f
            + layernorm2_f
        )

        layer_time = core_time

        total_transformer_time = layer_time * self.num_layers

        return total_transformer_time, linear_softmax_f
