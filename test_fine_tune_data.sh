#!/bin/bash

# data_path="/home/jaume/Desktop/Code/container-validator"
data_path="/media/jaume/DATA/Data/fomo25_finetune_data"

task1_img="/media/jaume/DATA/Data/SingularityImagesFOMO/classification.sif"
task2_img="/media/jaume/DATA/Data/SingularityImagesFOMO/segmentation.sif"
task3_img="/media/jaume/DATA/Data/SingularityImagesFOMO/brain_age.sif"

output_path1="/media/jaume/DATA/Data/fomo25/fomo-task1-val/predictions_validator_ft"
output_path2="/media/jaume/DATA/Data/fomo25/fomo-task2-val/predictions_validator_ft"
output_path3="/media/jaume/DATA/Data/fomo25/fomo-task3-val/predictions_validator_ft"

# # Command to run full validation on task 1 (infarct detection)
# python main.py --task task1 --container ${task1_img} --data-dir ${data_path}/fomo-task1/ --output-dir ${output_path1}

# # Command to run validation on task 2 (meningioma segmentation)
# python main.py --task task2 --container ${task2_img} --data-dir ${data_path}/fomo-task2/ --output-dir ${output_path2}

# # Command to run validation on task 3 (brain age estimation)
# python main.py --task task3 --container ${task3_img} --data-dir ${data_path}/fomo-task3/ --output-dir ${output_path3}

# Command to only validate the environment (without running the model)
# python main.py --task task1 --container ${task1_img} --data-dir ${data_path}/fomo-task1/ --output-dir output/task1/ --validate-env-only

# # Command to validate the environment using CPU only (skip GPU checks)
# python main.py --task task1 --container ${task1_img} --data-dir ${data_path}/fomo-task1/ --output-dir output/task1/ --validate-env-only --skip-gpu-check


# ---- Run evaluator scripts ----

# Task 1
pred_dir="/media/jaume/DATA/Data/fomo25/fomo-task1-val/predictions_validator_ft"
gt_dir="/media/jaume/DATA/Data/fomo25_finetune_data/fomo-task1/labels_flattened"
eval_output="/media/jaume/DATA/Data/fomo25/fomo-task1-val/eval_results_ft"
evaluator_script="/home/jaume/Desktop/Code/container-validator/task1_classification/evaluation/clf_evaluator.py"

mkdir -p "${eval_output}"

# Run the evaluator with default threshold and custom prefix
python "${evaluator_script}" \
    "${gt_dir}" \
    "${pred_dir}" \
    --output-dir "${eval_output}" \
    --prefix "infarct_eval" \
    --verbose

# Task 2
pred_dir="/media/jaume/DATA/Data/fomo25/fomo-task2-val/predictions_validator_ft"
gt_dir="/media/jaume/DATA/Data/fomo25_finetune_data/fomo-task2/labels_flattened"
eval_output="/media/jaume/DATA/Data/fomo25/fomo-task2-val/eval_results_ft"
evaluator_script="/home/jaume/Desktop/Code/container-validator/task2_segmentation/evaluation/seg_evaluator.py"

mkdir -p "${eval_output}"

# Run the evaluator with default threshold and custom prefix
python "${evaluator_script}" \
    "${gt_dir}" \
    "${pred_dir}" \
    -l 0 1 \
    --output-dir "${eval_output}" \
    --verbose


# Task 3
pred_dir="/media/jaume/DATA/Data/fomo25/fomo-task3-val/predictions_validator_ft"
gt_dir="/media/jaume/DATA/Data/fomo25_finetune_data/fomo-task3/labels_flattened"
eval_output="/media/jaume/DATA/Data/fomo25/fomo-task3-val/eval_results_ft"
evaluator_script="/home/jaume/Desktop/Code/container-validator/task3_regression/evaluation/reg_evaluator.py"

mkdir -p "${eval_output}"

# Run the evaluator with default threshold and custom prefix
python "${evaluator_script}" \
    "${gt_dir}" \
    "${pred_dir}" \
    --output-dir "${eval_output}" \
    --prefix "regression_results" \
    --verbose