import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root) # I need to do this because the original developers decided that having a repo name with '-' its amazing
from task2_segmentation.nnunet_utils.create_splits import create_nnunet_splits_from_csv
from task2_segmentation.nnunet_utils.create_folds_for_dino3d import create_dino_datasets_from_splits


# Example usage
if __name__ == "__main__":
    #csv_file = "t1_development_folds_bins-3_seed-42.csv"
    #splits = create_nnunet_splits_from_csv(csv_file, output_filepath='splits_final_no_test.json')


    # Example paths - update according to your setup
    preprocessed_data = "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task001_FOMO2"
    splits_input = "/home/jovyan/workspace/container-validator/task2_segmentation/splits/nnunet_experiments/splits_final_no_test.json"  # Can be either JSON or CSV file
    output_directory = "/home/jovyan/workspace/container-validator/task2_segmentation/splits/dino_experiments/"
    extracted_directory = "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task002_FOMO2_extracted_modalities"
    
    fomo_modalities = {
        0: {"name": "dwi", "description": "DWI (Diffusion Weighted Imaging)"},
        1: {"name": "flair", "description": "T2FLAIR"},
        2: {"name": "ADC", "description": "ADC (Apparent Diffusion Coefficient)"},
        3: {"name": "swi/t2star", "description": "SWI_OR_T2STAR (Susceptibility Weighted Imaging or T2*)"},
        4: {"name": "label", "description": "Segmentation Label"}
    }
    create_dino_datasets_from_splits(
         preprocessed_data_path=preprocessed_data,
         splits_source="splits_final.json",  # JSON file
         output_dir=output_directory,
         extracted_data_dir=extracted_directory,
         experiment_name="single_experiment_always_extract",
         file_prefix="FOMO2_sub_",
         num_image_channels=3,
         use_val_as_test=True,
         modality_info=fomo_modalities,  # 4 channels automatically detected
         extraction_strategy="always"  # Always re-extract even if files exist
     )