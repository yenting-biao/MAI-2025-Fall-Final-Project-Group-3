#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUBLAS_WORKSPACE_CONFIG=:16:8

audio_task="ASR"
# audio_task="SER"
# audio_task="GR"

# response_task="closed_ended_questions"
# response_task="chain-of-thought"
response_task="creative_writing"

# IF_task="change_case:english_capital"
# IF_task="change_case:english_lowercase"
# IF_task="detectable_format:json_format"
# IF_task="startend:quotation"
# IF_task="detectable_format:title"
# IF_task="combination:repeat_prompt"
# IF_task="startend:end_checker"

IF_task="detectable_format:number_bullet_lists"
# IF_task="keywords:existence"
# IF_task="keywords:forbidden_words"
# IF_task="length_constraints:number_words"
# IF_task="length_constraints:number_sentences"
# IF_task="length_constraints:number_paragraphs"

# IF_task="chain-of-thought"

examples=2
model_name="qwen"
seed=42
OUTPUT_DIR="${OUTPUT_DIR:-model_responses/test/}"

python run.py \
  --model_name "${model_name}" \
  --audio_task "${audio_task}" \
  --response_task "${response_task}" \
  --IF_task "${IF_task}" \
  --seed "${seed}" \
  --examples "${examples}" \
  --output_dir "${OUTPUT_DIR}" \
  --verbose \
  --debug \
  --debug_examples 5
