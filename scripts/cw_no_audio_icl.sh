#!/bin/bash
set -e

export CUBLAS_WORKSPACE_CONFIG=:4096:8
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
MODEL_NAMES=("$1") # qwen2, desta2_5, qwen25_omni, blsp-emo, gemini-2.5-flash, gemini-2.5-flash_no-thinking
SEEDS=(42)
OUTPUT_DIR="${OUTPUT_DIR:-model_responses/}"

for model_name in "${MODEL_NAMES[@]}"; do
  for audio_task in "${AUDIO_TASKS[@]}"; do
    for response_task in "${RESPONSE_TASKS[@]}"; do
      for IF_task in "${IF_TASKS[@]}"; do
        for examples in "${EXAMPLES[@]}"; do
          for seed in "${SEEDS[@]}"; do

            # if not asr, skip keywords:existence and keywords:forbidden_words
            if [[ "${audio_task}" != "ASR" && ( "${IF_task}" == "keywords:existence" || "${IF_task}" == "keywords:forbidden_words" ) ]]; then
              continue
            fi

            python run.py \
              --model_name "${model_name}" \
              --audio_task "${audio_task}" \
              --response_task "${response_task}" \
              --IF_task "${IF_task}" \
              --seed "${seed}" \
              --examples "${examples}" \
              --output_dir "${OUTPUT_DIR}" \
              --no_audio_icl # TODO: I am not sure if this setting is needed to run
          done
        done
      done
    done
  done
done
