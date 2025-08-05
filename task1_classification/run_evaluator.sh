#!/bin/bash

# Paths
pred_dir="/media/jaume/DATA/Data/fomo25/fomo-task1-val/predictions"
gt_dir="/home/jaume/Desktop/Code/container-validator/fake_data/fomo25/fomo-task1-val/labels"
eval_output="/media/jaume/DATA/Data/fomo25/fomo-task1-val/eval_results"
evaluator_script="/home/jaume/Desktop/Code/container-validator/task1_classification/evaluation/clf_evaluator.py"

mkdir -p "${eval_output}"

# Run the evaluator with default threshold and custom prefix
python "${evaluator_script}" \
    "${gt_dir}" \
    "${pred_dir}" \
    --output-dir "${eval_output}" \
    --prefix "infarct_eval" \
    --verbose