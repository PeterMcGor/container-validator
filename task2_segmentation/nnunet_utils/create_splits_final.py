import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root) # I need to do this because the original developers decided that having a repo name with '-' its amazing
from task2_segmentation.nnunet_utils.create_splits import create_nnunet_splits_from_csv, validate_splits


# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file = "../splits/cv_splits_bins-3_seed-42_no-test.csv"
    
    # Create the splits
    splits = create_nnunet_splits_from_csv(csv_file, output_filepath='../splits/nnunet_experiments/splits_final_no_test.json')
    
    # Validate the splits
    validate_splits(splits, csv_file)
