#!/bin/sh

save_path="/media/jaume/DATA/Data/SingularityImagesFOMO"
mkdir -p $save_path

src_path="/home/jaume/Desktop/Code/container-validator/task2_segmentation"

apptainer build --fakeroot ${save_path}/segmentation.sif ${src_path}/Apptainer.def