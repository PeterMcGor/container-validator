import os 
import sys
import glob
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)  # Add project root to sys.path
from task1_classification.splits.extract_infarct_stats import extract_number_from_file
from task2_segmentation.nnunet_utils.extract_lesions_stats import extract_image_basics

if __name__ == "__main__":
    main_dir = '/media/secondary/fomo-fine-tuning-datasets/fomo-task3/'
    main_labels_dir = os.path.join(main_dir,'labels/sub*/ses*/*.txt')
    save_cls_path = os.path.join(main_dir,"brain-age_stats.csv")
    save_summary_path = os.path.join(main_dir,"brain-age_summary_stats.csv")
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
        #t2s_or_swi = os.path.exists(label_path.replace('labels', 'preprocessed').replace('label.txt','t2s.nii.gz'))  # Check if T2s or SWI exists
        print(f"Processing: {label_path}")
        print(f"Subject: {subject_id}, Session: {session}")
    
        # Get the mask file path
        t1 = label_path.replace('labels', 'preprocessed').replace('label.txt','t1.nii.gz') # I use the T1. Images are registered
        print(f"Looking for t1: {t1}") 
        
        if os.path.exists(t1):
            try:
                summary_stats = extract_image_basics(t1)
            except Exception as e:
                print(f"Error in extract_label_stats: {e}")
                
        else:
            print(f"T1 file not found: {t1}")
            summary_stats = {}
            print("No t1")
        
        # Extract age from text file  
        summary_stats['age'] = extract_number_from_file(label_path) 
       
        print("--------")
        
        # Create a dictionary with common metadata
        metadata = {
            'subject_id': subject_id,
            'TP': session,
            'filename': t1,
        }
        
        # Update summary_stats with metadata
        summary_stats.update(metadata)
        cl_summary_list.append(summary_stats)
    
    print(f"Total records in cl_summary_list: {len(cl_summary_list)}")
    
    # Create DataFrames
    cl_summary_df = pd.DataFrame(cl_summary_list)
    
    print("DataFrame shapes:")
    print(f"cl_summary_df: {cl_summary_df.shape}")
    
   
    # Save to CSV
    try:
        cl_summary_df.to_csv(save_summary_path, index=False)
        print(f"Files saved successfully:")
        print(f"- {save_summary_path}")
    except Exception as e:
        print(f"Error saving files: {e}")