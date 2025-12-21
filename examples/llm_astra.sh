#!/bin/bash

# Runs performance model in LLM mode
# Setup model parameter in configs/model-config/Llama2-7B.yaml
# Setup hardware parameters (incl. parallelism/network) in configs/hardware-config/a100_80GB.yaml

HW_CONFIG="configs/hardware-config/a100_80GB.yaml"
MODEL_CONFIG="configs/model-config/Llama2-7B.yaml"

if [[ -n "${RAPID_UV_RUN:-}" ]]; then
    run_cmd=(uv run run_perf.py)
else
    run_cmd=(python3 run_perf.py)
fi

"${run_cmd[@]}" --hardware_config "${HW_CONFIG}" --model_config "${MODEL_CONFIG}"
