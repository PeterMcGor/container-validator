import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root) # I need to do this because the original developers decided that having a repo name with '-' its amazing
from task2_segmentation.nnunet_utils.create_splits import create_nnunet_splits_from_csv
from task2_segmentation.nnunet_utils.create_folds_for_dino3d import create_dino_folds


# Example usage
if __name__ == "__main__":
    #csv_file = "t1_development_folds_bins-3_seed-42.csv"
    #splits = create_nnunet_splits_from_csv(csv_file, output_filepath='splits_final_no_test.json')


    # Example paths - update according to your setup
    preprocessed_data = "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task001_FOMO1"
    splits_input = "/home/jovyan/workspace/container-validator/task1_classification/splits/splits_final_no_test.json"  # Can be either JSON or CSV file
    output_directory = "/home/jovyan/workspace/container-validator/task1_classification/splits/dino_experiments/"
    extracted_directory = "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task001_FOMO1_extracted_modalities"
       
    # Define modality configurations for extracted files
    task1_3ch_modalities = {
        0: {"name": "dwi", "description": "DWI"},
        1: {"name": "flair", "description": "T2FLAIR"}, 
        2: {"name": "adc", "description": "ADC"}  
    }
    
    task1_4ch_modalities = {
        0: {"name": "dwi", "description": "DWI"},
        1: {"name": "flair", "description": "T2FLAIR"},
        2: {"name": "adc", "description": "ADC"},  
        3: {"name": "swi0t2star", "description": "SWI_OR_T2STAR"} 
    } 


    create_dino_folds(
        splits_source=splits_input,
        data_dir=preprocessed_data,
        output_dir=output_directory,
        extracted_data_dir=extracted_directory,
        experiment_name="fomo-task1-3ch-mimic",
        file_prefix="FOMO1_sub_",
        modality_info=task1_3ch_modalities,
        label_extension=".txt",
        extraction_strategy="if_missing"
    )