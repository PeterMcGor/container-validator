#!/bin/sh

save_path="/media/jaume/DATA/Data/SingularityImagesFOMO_DINO"
mkdir -p $save_path

src_path="/home/jaume/Desktop/Code/container-validator_dino/task2_segmentation"

apptainer build --fakeroot --arch amd64 ${save_path}/segmentation.sif ${src_path}/Apptainer.def