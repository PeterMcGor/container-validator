#!/bin/bash

img_path="/media/jaume/DATA/Data/SingularityImagesFOMO/segmentation.sif"

subject_id="sub_6"
input_data="/home/jaume/Desktop/Code/container-validator/fake_data/fomo25/fomo-task2-val/preprocessed/sub_6/ses_1"
output_path="/media/jaume/DATA/Data/fomo25/fomo-task2-val/predictions"
mkdir -p ${output_path}

predict_script="/home/jaume/Desktop/Code/container-validator/task2_segmentation/predict.py"

# # --bind ${predict_script}:/app/predict.py \
# apptainer run --bind ${input_data}:/input:ro \
#     --bind ${output_path}:/output \
#     --nv \
#     ${img_path} \
#     --flair /input/flair.nii.gz \
#     --dwi_b1000 /input/dwi_b1000.nii.gz \
#     --t2s /input/t2s.nii.gz \
#     --output /output/${subject_id}.nii.gz


# ===== Loop =====
img_path="/media/jaume/DATA/Data/SingularityImagesFOMO/segmentation.sif"
input_root="/home/jaume/Desktop/Code/container-validator/fake_data/fomo25/fomo-task2-val/preprocessed"
output_root="/media/jaume/DATA/Data/fomo25/fomo-task2-val/predictions"

# Subject IDs to process
subjects=("sub_6" "sub_7") #"sub_17")

for subject_id in "${subjects[@]}"; do
    input_data="${input_root}/${subject_id}/ses_1"
    output_path="${output_root}"
    mkdir -p "${output_path}"

    echo "Running prediction for ${subject_id}..."

    apptainer run \
        --bind "${input_data}:/input:ro" \
        --bind "${output_path}:/output" \
        --bind ${predict_script}:/app/predict.py \
        --nv \
        "${img_path}" \
        --flair /input/flair.nii.gz \
        --dwi_b1000 /input/dwi_b1000.nii.gz \
        --t2s /input/t2s.nii.gz \
        --output /output/${subject_id}.nii.gz

    echo "Finished ${subject_id}"
    echo
done

subject_id="sub_17"
input_data="${input_root}/${subject_id}/ses_1"
output_path="${output_root}"
mkdir -p "${output_path}"

echo "Running prediction for ${subject_id}..."

apptainer run \
    --bind "${input_data}:/input:ro" \
    --bind "${output_path}:/output" \
    --bind ${predict_script}:/app/predict.py \
    --nv \
    "${img_path}" \
    --flair /input/flair.nii.gz \
    --dwi_b1000 /input/dwi_b1000.nii.gz \
    --swi /input/swi.nii.gz \
    --output /output/${subject_id}.nii.gz

echo "Finished ${subject_id}"