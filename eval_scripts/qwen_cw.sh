#!/bin/bash

MODEL_NAME="qwen"
RESPONSE_TASK="closed_ended_questions"
AUDIO_TASKS=("ASR" "SER" "GR")

for audio_task in "${AUDIO_TASKS[@]}"; do
    python -m eval_scripts.eval_llm_judge.py \
        --model_name="${model_name}" \
        --audio_task="${audio_task}" \
        --response_task="${response_task}" \
        --IF_task="${IF_task}" \
        --examples="${examples}"
done
