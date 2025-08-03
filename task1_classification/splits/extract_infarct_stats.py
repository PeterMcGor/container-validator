import glob
import os
import sys

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)  # Add project root to sys.path
from task2_segmentation.nnunet_utils.extract_lesions_stats import extract_label_stats



def extract_number_from_file(file_path):
    """
    Extract a number from a text file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        int: The number extracted from the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file doesn't contain a valid number
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()
            return int(content)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found")
    except ValueError:
        raise ValueError(f"File content '{content}' is not a valid number")


if __name__ == "__main__":
    main_dir = '/media/secondary/fomo-fine-tuning-datasets/fomo-task1/'
    main_labels_dir = os.path.join(main_dir,'labels/sub*/ses*/*.txt')
    save_cls_path = os.path.join(main_dir,"infarct_stats.csv")
    save_summary_path = os.path.join(main_dir,"infarct_summary_stats.csv")
    cl_mask_paths = glob.glob(main_labels_dir)
    cl_list = []
    cl_summary_list = []

    print(f"Found {len(cl_mask_paths)} label files")
    print(f"Pattern used: {main_labels_dir}")
    
    if not cl_mask_paths:
        print("No files found! Check your directory structure and glob pattern.")
        exit()

    for label_path in cl_mask_paths:  # Renamed variable to avoid confusion
        # Extract filename and folder name
        fields = label_path.split('/')
        session = fields[-2].split('_')[1] if '_' in fields[-2] else fields[-2]
        subject_id = fields[-3]
        t2s_or_swi = os.path.exists(label_path.replace('labels', 'preprocessed').replace('label.txt','t2s.nii.gz'))  # Check if T2s or SWI exists
        print(f"Processing: {label_path}")
        print(f"Subject: {subject_id}, Session: {session}, T2s_exists: {t2s_or_swi}")
    
        # Get the mask file path
        mask = label_path.replace('label.txt', 'seg.nii.gz')
        print(f"Looking for mask: {mask}")
        
        if os.path.exists(mask):
            try:
                label_stats, summary_stats = extract_label_stats(mask)
                print(f"Number of lesions: {summary_stats.get('N_lesions', 0)}")
                print(f"Total lesion volume: {summary_stats.get('Total_volume', 0):.2f}")
            except Exception as e:
                print(f"Error in extract_label_stats: {e}")
                label_stats = []
                summary_stats = {}
        else:
            print(f"Mask file not found: {mask}")
            label_stats = []
            summary_stats = {}
            print("No lesions")
        
        # Extract label number from text file
        label_txt_path = mask.replace('seg.nii.gz','label.txt')
        label_number = extract_number_from_file(label_txt_path)  # Renamed variable
        summary_stats['label'] = label_number
       
        print("--------")
        
        # Create a dictionary with common metadata
        metadata = {
            'subject_id': subject_id,
            'TP': session,
            'filename': mask,
            't2s_or_swi': t2s_or_swi
        }
        
        # Update label_stats and summary_stats with metadata
        if isinstance(label_stats, list):
            for stats in label_stats:
                if isinstance(stats, dict):
                    stats.update(metadata)
            cl_list.extend(label_stats)  # Use extend instead of +=
        
        # Update summary_stats with metadata
        summary_stats.update(metadata)
        cl_summary_list.append(summary_stats)
    
    print(f"Total records in cl_list: {len(cl_list)}")
    print(f"Total records in cl_summary_list: {len(cl_summary_list)}")
    
    # Create DataFrames
    cl_df = pd.DataFrame(cl_list)
    cl_summary_df = pd.DataFrame(cl_summary_list)
    
    print("DataFrame shapes:")
    print(f"cl_df: {cl_df.shape}")
    print(f"cl_summary_df: {cl_summary_df.shape}")
    
    # Handle empty values - use fillna for NaN and replace for empty strings
    cl_df = cl_df.fillna(0).replace('', 0)
    cl_summary_df = cl_summary_df.fillna(0).replace('', 0)
    
    # Save to CSV
    try:
        cl_df.to_csv(save_cls_path, index=False)
        cl_summary_df.to_csv(save_summary_path, index=False)
        print(f"Files saved successfully:")
        print(f"- {save_cls_path}")
        print(f"- {save_summary_path}")
    except Exception as e:
        print(f"Error saving files: {e}")