#!/bin/bash

# Runs the first Korthikanti training case (GPT-22B, bs=4, mb=1, dp=1, tp=8, pp=1, cp=1, tp_sp=true)
# with all debug/visualization flags enabled.

RAPID_PERSIST_ASTRASIM_ARTIFACTS=1 \
RAPID_VISUALIZE_GRAPHS=1 \
RAPID_PERSIST_ARTIFACT_VIZ=1 \
uv run run_perf.py \
  --hardware_config validation_scripts/validation_configs/hardware-config/a100_80GB_korthikanti.yaml \
  --model_config validation_scripts/validation_configs/model-config/GPT_22_B.yaml
