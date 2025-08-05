#!/bin/sh

save_path="/media/jaume/DATA/Data/SingularityImagesFOMO"
mkdir -p $save_path

src_path="/home/jaume/Desktop/Code/container-validator/task1_classification"

apptainer build --fakeroot ${save_path}/classification.sif ${src_path}/Apptainer.def