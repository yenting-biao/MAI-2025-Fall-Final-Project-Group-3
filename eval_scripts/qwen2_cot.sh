#!/bin/bash

# Run single
# python -m eval_scripts.eval_llm_judge \
#     --input_response_data ./model_responses/qwen2/ASR/chain-of-thought/chain-of-thought/output_0-shot_20251211-004817.jsonl

# Run multiple
MODEL_NAME="qwen2"
RESPONSE_TASKS=("chain-of-thought")
AUDIO_TASKS=("ASR" "SER" "GR")

for audio_task in "${AUDIO_TASKS[@]}"; do
    for response_task in "${RESPONSE_TASKS[@]}"; do
        python -m eval_scripts.eval_llm_judge \
            --model_name="${MODEL_NAME}" \
            --audio_task="${audio_task}" \
            --response_task="${response_task}"
    done
done
