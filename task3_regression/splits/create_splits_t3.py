import sys
import os

import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root) # I need to do this because the original developers decided that having a repo name with '-' its amazing
from task2_segmentation.nnunet_utils.create_splits import flexible_stratified_split

if __name__ == "__main__":
    

    random_seed = 42
    test_size = 0.2
    n_bins = [6,2,2] # This actually will depends on the data a lot 
    cl_data_path = '/media/secondary/fomo-fine-tuning-datasets/fomo-task3/brain-age_summary_stats.csv'
    save_dir = '/media/secondary/fomo-fine-tuning-datasets/fomo-task3'

    cl_data = pd.read_csv(cl_data_path) 
    
    #Create two dataframes. UCL data as out-of-domain
    cl_data['domain'] = 'development'
    # Create the dataframe for training/test in domain and out-of-domain
    developement_df = cl_data[cl_data['domain'] == 'development']

    dev_df = flexible_stratified_split(developement_df, test_size=test_size, n_bins=n_bins, random_state=random_seed, group_cols=['subject_id'],
                            stratify_cols=['age','Spacing_X','Spacing_Z'],
                            continuous_cols=['age','Spacing_X','Spacing_Z'], folds=5)
    dev_df.to_csv(os.path.join(save_dir, f"t3_development_folds_bins-{str(n_bins)}_seed-{random_seed}.csv"), index=False)