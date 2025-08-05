#!/bin/bash

img_path="/media/jaume/DATA/Data/SingularityImagesFOMO/classification.sif"

subject_id="sub_4"
input_data="/home/jaume/Desktop/Code/container-validator/fake_data/fomo25/fomo-task1-val/preprocessed/sub_4/ses_1"
output_path="/media/jaume/DATA/Data/fomo25/fomo-task1-val/predictions"
mkdir -p ${output_path}

predict_script="/home/jaume/Desktop/Code/container-validator/task1_classification/predict.py"

# --bind ${predict_script}:/app/predict.py \
apptainer run --bind ${input_data}:/input:ro \
    --bind ${output_path}:/output \
    --nv \
    ${img_path} \
    --flair /input/flair.nii.gz \
    --adc /input/adc.nii.gz \
    --dwi_b1000 /input/dwi_b1000.nii.gz \
    --t2s /input/t2s.nii.gz \
    --output /output/${subject_id}.txt


# ===== Loop =====
mg_path="/media/jaume/DATA/Data/SingularityImagesFOMO/classification.sif"
input_root="/home/jaume/Desktop/Code/container-validator/fake_data/fomo25/fomo-task1-val/preprocessed"
output_root="/media/jaume/DATA/Data/fomo25/fomo-task1-val/predictions"

# Subject IDs to process
subjects=("sub_4" "sub_14" "sub_15")

for subject_id in "${subjects[@]}"; do
    input_data="${input_root}/${subject_id}/ses_1"
    output_path="${output_root}"
    mkdir -p "${output_path}"

    echo "Running prediction for ${subject_id}..."

    apptainer run \
        --bind "${input_data}:/input:ro" \
        --bind "${output_path}:/output" \
        --nv \
        "${img_path}" \
        --flair /input/flair.nii.gz \
        --adc /input/adc.nii.gz \
        --dwi_b1000 /input/dwi_b1000.nii.gz \
        --t2s /input/t2s.nii.gz \
        --output /output/${subject_id}.txt

    echo "Finished ${subject_id}"
    echo
done
