# Hierarchical Mode Fault Bugs

## 1. Fault Projection Leak
Hierarchical execution runs AstraSim separately for the transformer (TP/CP axes) and pipeline (LP axis) graphs. Each run filters the hardware layout down to the axes needed for that graph, but `generate_astrasim_configs_from_hw()` still injects the original `network.faulty_links` list without remapping the endpoints into the reduced coordinate space. Any fault whose flattened IDs fall inside the smaller rank range now appears to live on *every* simulated axis subset. This reinterprets a single physical faulty link multiple times (e.g., once in the transformer run and again in the LP run), so hierarchical mode applies the same degradation repeatedly and slows far more than flattened execution.

## 2. Single-Replica Transformer Timing
The hierarchical pipeline measures AstraSim on exactly one transformer replica and copies that timing to every transformer node in the pipeline graph. That assumption holds only when all stages experience identical network conditions. Faulty links are physical, rank-specific degradations—different pipeline stages map to different hardware coordinates, so only the stages whose TP/CP cluster actually touches the faulty link should be slowed. Because hierarchical mode reuses the single measurement for every stage, *all* stages inherit the worst-case transformer time, even if the fault affects just one cluster. Consequently hierarchical mode always over-penalizes the transformer, regardless of where the fault truly sits.

## Remediation Plan (hierarchical & hybrid)
1. **Decode Fault Coordinates**
   - Reconstruct the global axis layout (`tp`, `cp`, `lp`, `dp`) exactly as flattened mode assigns hw_ids.
   - For every `faulty_links` entry, convert `src/dst` into `(tp,cp,lp,dp)` tuples, then re-linearize per axis subset. This projection will emit two remapped lists: one for TP/CP-only graphs (transformer) and one for LP/DP graphs (pipeline). Hybrid mode follows the same flow because it still invokes AstraSim for transformer+pipeline pieces separately.

2. **Route Faults to the Correct Graph**
   - Use the projected coordinates to classify each link by the highest dimension it touches (dim0 → transformer cluster, dim≥1 → pipeline/DP axes). Only pass transformer-dimension faults into the transformer AstraSim jobs; only pass higher-dimension faults into the pipeline job.

3. **Per-Cluster Transformer Simulations**
   - Identify the set of TP/CP clusters impacted by faults. Run AstraSim `(N + 1)` times per direction: one baseline (no transformer faults) plus one run per faulty cluster with its specific remapped fault list attached. Hybrid mode needs the same forward/backward pairs.
   - Cache the timing per faulty cluster (forward/backward) alongside the baseline.

4. **Annotate Pipeline Graph Nodes**
   - When applying transformer timings in hierarchical/hybrid modes, assign the baseline duration to transformer nodes whose cluster is fault-free, and assign the corresponding per-cluster duration to nodes mapped to faulty hardware. This ensures only the affected stages slow down.

5. **Pipeline Fault Injection**
   - Feed the LP/DP-projected fault list into the pipeline AstraSim execution. Because the endpoints are already remapped to the LP/DP-only axis ordering, the existing hierarchical pipeline simulation will see the correct degraded links.
