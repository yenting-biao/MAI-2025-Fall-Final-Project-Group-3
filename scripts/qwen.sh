AUDIO_TASKS=("ASR" "SER" "GR")
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
MODEL_NAMES=("qwen")
SEEDS=(42)
OUTPUT_DIR="model_responses/jayliao_lugia/"

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
