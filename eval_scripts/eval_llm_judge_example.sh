#!/bin/bash

# Run a single file

# python eval_scripts/eval_llm_judge.py \
#   --input_path model_responses/qwen/ASR/creative_writing/detectable_format_number_bullet_lists/output_0-shot_20251208-210047.jsonl \
#   --output_path model_responses/qwen/ASR/creative_writing/detectable_format_number_bullet_lists/reports/judge@output_0-shot_20251208-210047.jsonl

# python eval_scripts/eval_llm_judge.py \
#   --input_path model_responses/qwen/ASR/creative_writing/detectable_format_number_bullet_lists/output_4-shot_20251208-211013.jsonl \
#   --output_path model_responses/qwen/ASR/creative_writing/detectable_format_number_bullet_lists/reports/judge@output_4-shot_20251208-211013.jsonl

# Run multiple files within a single task
uv run python eval_scripts/eval_llm_judge.py \
  --model_name qwen \
  --audio_task ASR \
  --response_task creative_writing