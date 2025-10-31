#!/bin/bash

# Runs performance model in LLM mode
# Setup model parameters in configs/model-config/LLM.yaml
# Setup hardware parameters (incl. parallelism/network) in configs/hardware-config/a100_80GB_no_parallelism.yaml

python run_perf.py --hardware_config configs/hardware-config/a100_80GB_no_parallelism.yaml --model_config configs/model-config/Llama2-7B.yaml
