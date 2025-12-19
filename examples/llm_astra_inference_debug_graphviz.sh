#!/bin/bash

# Runs performance model in LLM mode
# Setup model parameter in configs/model-config/Llama2-7B.yaml
# Setup hardware parameters (incl. parallelism/network) in configs/hardware-config/a100_80GB.yaml

RAPID_PERSIST_ASTRASIM_ARTIFACTS=1 RAPID_VISUALIZE_GRAPHS=1 RAPID_PERSIST_ARTIFACT_VIZ=1 python run_perf.py --hardware_config configs/hardware-config/a100_80GB.yaml --model_config configs/model-config/LLM_inf.yaml