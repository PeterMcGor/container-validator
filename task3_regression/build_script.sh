#!/bin/sh

save_path="/media/jaume/DATA/Data/SingularityImagesFOMO"
mkdir -p $save_path

src_path="/home/jaume/Desktop/Code/container-validator/task3_regression"

apptainer build --fakeroot --arch amd64 ${save_path}/brain_age.sif ${src_path}/Apptainer.def