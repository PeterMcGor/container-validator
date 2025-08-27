#!/bin/sh

save_path="/media/jaume/DATA/Data/SingularityImagesFOMO_DINO"
mkdir -p $save_path

src_path="/home/jaume/Desktop/Code/container-validator_dino"

apptainer build --fakeroot --arch amd64 ${save_path}/dino3d_base.sif ${src_path}/Apptainer.def
# apptainer build --fakeroot /path/to/save/your/container.sif path/to/Apptainer.def --arch amd64