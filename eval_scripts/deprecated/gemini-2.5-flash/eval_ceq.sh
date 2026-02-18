#!/bin/bash

ROOT_DIR=$(pwd)
AUDIO_TASKS=("ASR" "SER" "GR" "MMAU")
RESPONSE_TASKS=("closed_ended_questions")
IF_TASKS=(
    "change_case:english_capital"
    "change_case:english_lowercase"
    "detectable_format:json_format"
    "startend:quotation"
    "detectable_format:title"
    "combination:repeat_prompt"
    "startend:end_checker"
)
EXAMPLES=(0 1 2 3 4 5 6 7 8)
MODEL_NAMES=("gemini-2.5-flash")

for model_name in "${MODEL_NAMES[@]}"; do
  for audio_task in "${AUDIO_TASKS[@]}"; do
    for response_task in "${RESPONSE_TASKS[@]}"; do
      for IF_task in "${IF_TASKS[@]}"; do
        for examples in "${EXAMPLES[@]}"; do
            python -m eval_scripts.evaluation_main \
                --model_name="${model_name}" \
                --audio_task="${audio_task}" \
                --response_task="${response_task}" \
                --IF_task="${IF_task}" \
                --examples="${examples}"
        done
      done
    done
  done
done


