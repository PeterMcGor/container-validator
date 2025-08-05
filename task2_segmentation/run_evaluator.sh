#!/bin/bash

# Paths
pred_dir="/media/jaume/DATA/Data/fomo25/fomo-task2-val/predictions"
gt_dir="/home/jaume/Desktop/Code/container-validator/fake_data/fomo25/fomo-task2-val/labels"
eval_output="/media/jaume/DATA/Data/fomo25/fomo-task2-val/eval_results"
evaluator_script="/home/jaume/Desktop/Code/container-validator/task2_segmentation/evaluation/seg_evaluator.py"

mkdir -p "${eval_output}"

# Run the evaluator with default threshold and custom prefix
python "${evaluator_script}" \
    "${gt_dir}" \
    "${pred_dir}" \
    -l 0 1 \
    --output-dir "${eval_output}" \
    --verbose