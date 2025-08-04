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
    preprocessed_data = "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task003_FOMO3"
    splits_input = "/home/jovyan/workspace/container-validator/task3_regression/splits/splits_final_no_test.json"  # Can be either JSON or CSV file
    output_directory = "/home/jovyan/workspace/container-validator/task3_regression/splits/dino_experiments/"
    extracted_directory = "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task003_FOMO3_extracted_modalities"
       
    # Define modality configurations for extracted files
    task3_1ch_modalities = {
        0: {"name": "t1", "description": "t1"}
    }

    task3_2ch_modalities = {
        0: {"name": "t1", "description": "T1"},
        1: {"name": "t2", "description": "T2"} 
    }
    

    create_dino_folds(
        splits_source=splits_input,
        data_dir=preprocessed_data,
        output_dir=output_directory,
        extracted_data_dir=extracted_directory,
        experiment_name="fomo-task3-1ch-mimic",
        file_prefix="FOMO3_",
        modality_info=task3_1ch_modalities,
        label_extension=".txt",
        extraction_strategy="if_missing"
    )