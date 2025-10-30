## Installation Guide

Prerequisite: Python 3

### Step 1. Clone the repository

```bash
git clone https://github.com/gkarfakis19/DeepFlow/
cd DeepFlow
```

### Step 2. Set up the environment

#### Option A: Using uv (recommended)

- `pip install uv`
- `uv venv [/path/to/new/virtual/environment]`
- `source [/path/to/new/virtual/environment]/bin/activate`
- `uv sync`

#### Option B: Using pip

- `python3 -m venv [/path/to/new/virtual/environment]`
- `source [/path/to/new/virtual/environment]/bin/activate`
- `pip install --upgrade pip`
- `pip install -r requirements.txt`

### Step 3. (Optional) Set up AstraSim for advanced network simulation

- `git submodule update --init --recursive`
- `git submodule update --remote -- astra-sim`
- `ASTRA_SIM=$(realpath ./astra-sim)`
- `cd ${ASTRA_SIM}`
- `./build/astra_analytical/build.sh`
- `cd ..`

### Troubleshooting

If you encounter protobuf failures while building AstraSim, try:

- `pip uninstall protobuf`
- `pip install protobuf==3.20.3`

### Step 4. Verify the installation

- `./examples/llm.sh` or `./examples/llm_inference.sh` (analytical backend, training & inference of Llama2-7B)
- `./examples/llm_astra.sh` or `./examples/llm_astra_inference.sh` (AstraSim backend, training & inference of Llama2-7B. Requires AstraSim installation)

## Execution Backend Configuration

DeepFlow supports four execution backends with different accuracy and performance characteristics. Configure the backend in your hardware config file under `execution_backend`.

### 1. Analytical DeepFlow (Default - no AstraSim needed)

- **Accuracy:** Very fast but inaccurate; only ring network model and no congestion modeling.
- **Configuration:**

```yaml
execution_backend:
  model: analytical
```

### 2. Hybrid (AstraSim needed)

- **Accuracy:** More accurate; models congestion in transformer blocks only but roughly 2-3× slower.
- **Execution:** DeepFlow executes the pipeline graph; AstraSim executes the transformer block graph.
- **Configuration:**

```yaml
execution_backend:
  model: astra
  astra:
    mode: hybrid
```

### 3. Full AstraSim Hierarchical (AstraSim needed)

- **Accuracy:** Even more accurate; models congestion in transformer and pipeline graphs separately. Assumes no congestion between pipeline/data parallelism and tensor parallelism (optimistic). Roughly as fast as Hybrid for small systems, increasingly slower for larger systems.
- **Execution:** AstraSim executes both pipeline and transformer block graphs separately.
- **Configuration:**

```yaml
execution_backend:
  model: astra
  astra:
    mode: full_astrasim_hierarchical
```

### 4. Full AstraSim Flattened (AstraSim needed)

- **Accuracy:** Most accurate; models congestion between all collectives with no separate network assumptions. Very slow for large systems but most comprehensive.
- **Execution:** AstraSim executes one big flattened graph combining pipeline and transformer operations.
- **Configuration:**

```yaml
execution_backend:
  model: astra
  astra:
    mode: full_astrasim_flattened
```

## Execution Modes

DeepFlow can be used in 6 different modes.

### Model Prediction Modes

1. **Performance Prediction Mode (GEMM)**
   - **When to use:** Distributed GEMM prediction.
   - **How:**
     - Specify GEMM parameters in `configs/model-config/GEMM.yaml`.
     - Specify hardware parameters in `configs/hardware-config/[config.yaml]`.
     - Run `python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/GEMM.yaml`.

2. **Performance Prediction Mode (LLM)**
   - **When to use:** End-to-end LLM prediction.
   - **How:**
     - Specify LLM parameters in `configs/model-config/LLM.yaml`.
     - Specify hardware parameters in `configs/hardware-config/[config.yaml]`.
     - Run `python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/LLM.yaml`.

3. **Performance Prediction Mode (LSTM End-to-End Application)**
   - **When to use:** End-to-end LSTM prediction.
   - **Note:** This mode has not been tested/validated with the AstraSim backend.
   - **How:**
     - Specify LSTM parameters in `configs/model-config/LSTM.yaml`.
     - Specify hardware parameters in `configs/hardware-config/[config.yaml]`.
     - Run `python run_perf.py --hardware_config configs/hardware-config/[config.yaml] --model_config configs/model-config/LSTM.yaml`.

LLM mode is a work in progress with limited validation. Results are saved under `output/<mode>`.

The following three modes are only tested/validated for LSTM (porting to LLM is very WIP):

4. **Performance Prediction Mode (using `main.py` standalone argument; somewhat equivalent to option 2, for running on Slurm)**
   - `python main.py stand_alone --exp_dir [/path/to/output/result/directory] --exp_config configs/[config.yaml]`

5. **Architecture Search for a Fixed Parallelism Strategy**
   - `python GD_search.py --exp_config configs/[config.yaml] --exp_dir [/path/to/output/directory] --debug False --index [index] --batch_size [batch] --hidden_dim [lstm_dim] --data_scale [dataset_scaling_factor] --dp [data parallel dim.] --lp [layer parallel dim.] --kp_type [0|1] --kp1 [kp1 dim.] --kp2 [kp2 dim.] --inter_derate [derate_factor_for_inter_package_bandwidth] --intra_derate [derate_factor_for_intra_package_bandwidth] --kp1_inter [False|True] --kp2_inter [False|True] --dp_inter [False|True] --lp_inter [False|True] --wafer_dim [package dim.]`
   - **Example:**
     `python GD_search.py --exp_config configs/exp_config.yaml --exp_dir output --debug False --index 40 --batch_size 256 --hidden_dim 19968 --data_scale 1 --dp 64 --lp 1 --kp_type 1 --kp1 1 --kp2 1 --inter_derate 0 --intra_derate 2 --kp1_inter False --kp2_inter False --dp_inter False --lp_inter False --wafer_dim 8`

6. **Architecture Search Mode for All Types of Parallelism Strategies**
   - `python main.py arch_search --exp_dir [/path/to/output/directory] --exp_config configs/[config.yaml]`

## AstraSim Artifact and Graph Visualization

DeepFlow can generate and visualize graphs, and when using the AstraSim network backend, can also generate and visualize network communication artifacts.

### Environment Flags

- `DEEPFLOW_VISUALIZE_GRAPHS=1`: Generate graph visualizations of DeepFlow computation graphs (no AstraSim artifact visualization).
- `DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS=1`: Enable artifact persistence to disk (for both AstraSim and DeepFlow artifacts).
- `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1`: Generate PNG visualizations and text dumps for persisted AstraSim ET files (very slow for many nodes).
- Do not set `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1` for multi-threaded runs.

### Artifact Output Locations

- Flattened execution mode: `output/LLM/astra_flat/`
- Hierarchical/Hybrid modes: `output/LLM/astra_hier/`

## Example case study: Tensor parallelism on inference runtime

1. Start by creating a model config from Hugging Face. For example, to pull the Qwen/Qwen2.5-3B settings:

```bash
python configs/model-config/hf_to_config.py Qwen/Qwen2.5-3B --run-type inference --batch-size 32 --seq-len 65536 --decode-len 1024 --use-flashattention true --flash-tile-size 256 -o configs/model-config/Qwen2.5-3B.yaml
```
 The config file for Qwen2.5-3B is now autamatically generated under `configs/model-config`

2. Use the example A100 hardware file and set tensor parallel degree to 1:


Edit `configs/hardware-config/a100_80GB_example.yaml` so that `system_hierarchy.num_devices_per_node: 1` while `scheduling_param.tp: 1`. This models inference on one GPU without tensor parallelism.

Run the inference estimation using the example configs:

```bash
python run_perf.py --hardware_config configs/hardware-config/a100_80GB_example.yaml --model_config configs/model-config/Qwen2.5-3B.yaml
```

3. To see the tensor-parallelism effect, modify the hardware file, switch to two devices per node and enable tensor parallelism:

Edit `configs/hardware-config/a100_80GB_example.yaml` so that `system_hierarchy.num_devices_per_node: 2` and `scheduling_param.tp: 2`. This models inference on one GPU with tensor parallelism degree of 2.

Re-run the same inference command with the updated hardware config.

Comparing the two runs will show how increasing tensor parallelism changes the predicted inference runtime for this model.

DeepFlow should report the following runtimes:  
- TP = 1 (single GPU): LLM inference time: 281.67s  
- TP = 2 (tensor degree of 2): LLM inference time: 157.17s



### Generated Files

- `.et` files: Chakra execution traces for AstraSim replay.
- `.png` files: Rendered PNG visualizations (when `DEEPFLOW_VISUALIZE_GRAPHS=1` or `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1`).
- `.txt` files: Human-readable text dumps of ET files (when `DEEPFLOW_PERSIST_ARTIFACT_VIZ=1`).

### Debugging: Usage Example

```bash
DEEPFLOW_PERSIST_ASTRASIM_ARTIFACTS=1 DEEPFLOW_VISUALIZE_GRAPHS=1 DEEPFLOW_PERSIST_ARTIFACT_VIZ=1 python run_perf.py \
  --hardware_config configs/hardware-config/a100_80GB.yaml \
  --model_config configs/model-config/LLM.yaml
```

## Tips

- AstraSim backend caches runs by default in `./astra_cache/`. To disable caching, set the environment variable `DEEPFLOW_ASTRA_CACHE_MODE` to `NO_CACHE` or `CACHE_READONLY` (for multi-threaded runs). `NO_CACHE` (or manual cache flushing) is necessary if the AstraSim binary itself is modified.
- Check the `configs` directory for architecture templates and technology node configurations.
