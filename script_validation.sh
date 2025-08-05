#!/bin/bash

data_path="/home/jaume/Desktop/Code/container-validator"

task1_img="/media/jaume/DATA/Data/SingularityImagesFOMO/classification.sif"
task2_img=""
task3_img="/media/jaume/DATA/Data/SingularityImagesFOMO/brain_age.sif"

output_path1="/media/jaume/DATA/Data/fomo25/fomo-task1-val/predictions_validator"
output_path2=""
output_path3="/media/jaume/DATA/Data/fomo25/fomo-task3-val/predictions_validator"

# Command to run full validation on task 1 (infarct detection)
python main.py --task task1 --container ${task1_img} --data-dir ${data_path}/fake_data/fomo25/fomo-task1-val/ --output-dir ${output_path1}

# Command to run validation on task 2 (meningioma segmentation)
# python main.py --task task2 --container ${task2_img} --data-dir ${data_path}/fake_data/fomo25/fomo-task2-val/ --output-dir ${output_path2}

# Command to run validation on task 3 (brain age estimation)
python main.py --task task3 --container ${task3_img} --data-dir ${data_path}/fake_data/fomo25/fomo-task3-val/ --output-dir ${output_path3}

# Command to only validate the environment (without running the model)
# python main.py --task task1 --container ${task1_img} --data-dir ${data_path}/fake_data/fomo25/fomo-task1-val/ --output-dir output/task1/ --validate-env-only

# # Command to validate the environment using CPU only (skip GPU checks)
# python main.py --task task1 --container ${task1_img} --data-dir ${data_path}/fake_data/fomo25/fomo-task1-val/ --output-dir output/task1/ --validate-env-only --skip-gpu-check