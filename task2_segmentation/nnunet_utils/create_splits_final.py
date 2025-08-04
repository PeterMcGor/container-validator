import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root) # I need to do this because the original developers decided that having a repo name with '-' its amazing
from task2_segmentation.nnunet_utils.create_splits import create_nnunet_splits_from_csv, validate_splits


# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = "/media/secondary/fomo-fine-tuning-datasets/fomo-task3/t3_development_folds_bins-6-2-2_seed-42.csv"
    
    # Create the splits
    splits = create_nnunet_splits_from_csv(csv_file, output_filepath='/home/petermcgor/Documents/Projects/FOMO_challenge/container-validator/task3_regression/splits/splits_final_no_test.json')
    
    # Validate the splits
    validate_splits(splits, csv_file)
