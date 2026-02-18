#!/bin/bash

MODEL_NAME="$1"
RESPONSE_TASKS=("chain-of-thought")
AUDIO_TASKS=("ASR" "SER" "GR" "MMAU")

for audio_task in "${AUDIO_TASKS[@]}"; do
    for response_task in "${RESPONSE_TASKS[@]}"; do
        python -m eval_scripts.eval_llm_judge \
            --model_name="${MODEL_NAME}" \
            --audio_task="${audio_task}" \
            --response_task="${response_task}" \
            --no_output_constraints
    done
done
