#!/bin/bash

img_path="/media/jaume/DATA/Data/SingularityImagesFOMO_DINO/brain_age.sif"
# img_path="/media/jaume/DATA/Data/SingularityImagesFOMO_DINO/brain_age_dino_em.sif"

subject_id="sub_2"
input_data="/home/jaume/Desktop/Code/container-validator/fake_data/fomo25/fomo-task3-val/preprocessed/sub_2/ses_1"
output_path="/media/jaume/DATA/Data/fomo25/fomo-task3-val/predictions"
mkdir -p ${output_path}

predict_script="/home/jaume/Desktop/Code/container-validator_dino/task3_regression/predict.py"
echo "Running prediction for ${subject_id}..."
start_time=$(date +%s)

# --bind ${predict_script}:/app/predict.py \
apptainer run --bind ${input_data}:/input:ro \
    --bind ${predict_script}:/app/predict.py \
    --bind ${output_path}:/output \
    --nv \
    ${img_path} \
    --t1 /input/t1.nii.gz \
    --t2 /input/t2.nii.gz \
    --output /output/${subject_id}.txt

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Finished ${subject_id} in ${elapsed} seconds"
# # ===== Loop =====
# mg_path="/media/jaume/DATA/Data/SingularityImagesFOMO/classification.sif"
# input_root="/home/jaume/Desktop/Code/container-validator/fake_data/fomo25/fomo-task3-val/preprocessed"
# output_root="/media/jaume/DATA/Data/fomo25/fomo-task3-val/predictions"

# # Subject IDs to process
# subjects=("sub_2" "sub_58" "sub_81")

# for subject_id in "${subjects[@]}"; do
#     input_data="${input_root}/${subject_id}/ses_1"
#     output_path="${output_root}"
#     mkdir -p "${output_path}"

#     echo "Running prediction for ${subject_id}..."
#     start_time=$(date +%s)

#     apptainer run \
#         --bind "${input_data}:/input:ro" \
#         --bind "${output_path}:/output" \
#         --nv \
#         "${img_path}" \
#         --t1 /input/t1.nii.gz \
#         --t2 /input/t2.nii.gz \
#         --output /output/${subject_id}.txt

#     end_time=$(date +%s)
#     elapsed=$((end_time - start_time))

#     echo "Finished ${subject_id} in ${elapsed} seconds"
# done
