#!/bin/bash

MODEL_NAME="qwen"
RESPONSE_TASK=("creative_writing" "chain-of-thought")
AUDIO_TASKS=("ASR" "SER" "GR")

for audio_task in "${AUDIO_TASKS[@]}"; do
    for response_task in "${RESPONSE_TASK[@]}"; do
        python -m eval_scripts.eval_llm_judge \
            --model_name="${MODEL_NAME}" \
            --audio_task="${audio_task}" \
            --response_task="${response_task}"
    done
done
