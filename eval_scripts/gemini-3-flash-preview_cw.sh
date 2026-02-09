#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

ROOT_DIR="$(pwd)"
AUDIO_TASKS=(ASR SER GR)
RESPONSE_TASKS=(creative_writing)
IF_TASKS=(
  "detectable_format:number_bullet_lists"
  "length_constraints:number_words"
  "length_constraints:number_sentences"
  "length_constraints:number_paragraphs"
)
MODEL_NAMES=("gemini-3-flash-preview")

for model_name in "${MODEL_NAMES[@]}"; do
  for audio_task in "${AUDIO_TASKS[@]}"; do

    if [[ "$audio_task" == "ASR" ]]; then
      IF_TASKS_CURRENT=(
        "detectable_format:number_bullet_lists"
        "keywords:existence"
        "keywords:forbidden_words"
        "length_constraints:number_words"
        "length_constraints:number_sentences"
        "length_constraints:number_paragraphs"
      )
    else
      IF_TASKS_CURRENT=("${IF_TASKS[@]}")
    fi

    for response_task in "${RESPONSE_TASKS[@]}"; do
      for IF_task in "${IF_TASKS_CURRENT[@]}"; do

        # If your directories use underscores instead of colons, keep this line.
        # If your directories literally contain ':', remove this line and use IF_task directly.
        dir_IF="${IF_task//:/_}"

        files=( "${ROOT_DIR}/model_responses/${model_name}/${audio_task}/${response_task}/${dir_IF}/output_"*.jsonl )

        # No files → skip
        ((${#files[@]} == 0)) && continue

        for file in "${files[@]}"; do
          python3 -m eval_scripts.evaluation_main_speech_ifeval -i "$file"
        done
      done
    done
  done
done
