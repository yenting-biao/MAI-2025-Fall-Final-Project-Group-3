#!/bin/bash

# ---------------------------
# Configuration
# ---------------------------

# Path to the Speech-IFEval root directory
ROOT_DIR="$(pwd)"

# Path to your input JSONL (model outputs)
# Modify this to the file you want to evaluate
model_name="qwen"
audio_task="ASR"
response_task="closed_ended_questions"
IF_task="change_case:english_capital"
examples=1

# Output directory for reports
OUTPUT_DIR="${ROOT_DIR}/eval_outputs/test"

# ---------------------------
# Run Evaluation
# ---------------------------

echo "Running evaluation..."
echo "Using:"
echo "  evaluation_main.py"
echo ""

python -m eval_scripts.evaluation_main \
    --model_name="${model_name}" \
    --audio_task="${audio_task}" \
    --response_task="${response_task}" \
    --IF_task="${IF_task}" \
    --examples="${examples}"

echo ""
echo "Done!"
echo "Report is saved under:"
echo "  ${OUTPUT_DIR}/reports/"
