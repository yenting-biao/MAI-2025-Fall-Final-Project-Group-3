#!/bin/bash

set -e

MODEL_NAME="blsp-emo"
RESPONSE_TASKS=("chain-of-thought")
AUDIO_TASKS=("MMAU" "ASR" "SER" "GR")

for audio_task in "${AUDIO_TASKS[@]}"; do
    for response_task in "${RESPONSE_TASKS[@]}"; do
        python -m eval_scripts.eval_llm_judge \
            --model_name="${MODEL_NAME}" \
            --audio_task="${audio_task}" \
            --response_task="${response_task}" \
            --task_level
    done
done
