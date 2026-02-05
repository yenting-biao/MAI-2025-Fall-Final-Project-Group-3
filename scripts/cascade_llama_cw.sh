#!/bin/bash
set -e

export CUBLAS_WORKSPACE_CONFIG=:16:8

AUDIO_TASKS=("ASR" "SER" "GR")
RESPONSE_TASKS=("creative_writing")
IF_TASKS=(
    "detectable_format:number_bullet_lists"
    "keywords:existence"
    "keywords:forbidden_words"
    "length_constraints:number_words"
    "length_constraints:number_sentences"
    "length_constraints:number_paragraphs"
)
EXAMPLES=(0 1 2 3 4 5 6 7 8)
MODEL_NAMES=("cascade_Llama-3_1-8B-Instruct")
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
