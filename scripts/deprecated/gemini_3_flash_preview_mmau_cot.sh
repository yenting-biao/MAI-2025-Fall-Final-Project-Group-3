#!/bin/bash
set -e

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUBLAS_WORKSPACE_CONFIG=:16:8

AUDIO_TASKS=("MMAU")
RESPONSE_TASKS=("chain-of-thought")
IF_TASKS=("chain-of-thought")
EXAMPLES=(0 1 2 3 4 5 6 7 8)
MODEL_NAMES=("gemini-3-flash-preview")
SEEDS=(42)
OUTPUT_DIR="${OUTPUT_DIR:-model_responses/}"    

for model_name in "${MODEL_NAMES[@]}"; do
  for audio_task in "${AUDIO_TASKS[@]}"; do
    for response_task in "${RESPONSE_TASKS[@]}"; do
      for IF_task in "${IF_TASKS[@]}"; do
        for examples in "${EXAMPLES[@]}"; do
          for seed in "${SEEDS[@]}"; do
            python run.py \
              --model_name "${model_name}" \
              --audio_task "${audio_task}" \
              --response_task "${response_task}" \
              --IF_task "${IF_task}" \
              --seed "${seed}" \
              --examples "${examples}" \
              --output_dir "${OUTPUT_DIR}"
          done
        done
      done
    done
  done
done
