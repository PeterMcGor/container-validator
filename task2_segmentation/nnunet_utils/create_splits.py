import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

def merge_single_sample_groups(group_stats, count_on='strata', continuous_cols=['N_lesions', 'Total_volume'], filter_cols='site_exp'):
    # Identify groups with only one member
    strata_counts = group_stats[count_on].value_counts()
    single_sample_groups = strata_counts[strata_counts == 1].index.tolist()
    
    if not single_sample_groups:
        return group_stats  # No single sample groups to merge
    # add warning messages with the groups to merge
    print(f"WARNING: Merging single sample groups: {single_sample_groups}.")

    # Calculate means for continuous columns for each strata
    means = group_stats.groupby([count_on]+[filter_cols])[continuous_cols ].mean().reset_index()
    
    # Get the row of the single sample group
    group = single_sample_groups[0]
    single_sample_row = group_stats[group_stats[count_on] == group]
    site_exp_value = single_sample_row[filter_cols].values[0]
        
    # Find the nearest group based on the mean of continuous variables with the same site_exp
    nearest_group = means.loc[(means[filter_cols] == site_exp_value) & (means[count_on] != group)].copy() 
        
    if nearest_group.empty:
        print(f"No suitable nearest group found for single sample group '{group}' with site_exp '{site_exp_value}'.")
        # continue  # Skip merging if no suitable group is found

    # TODO: The distance function is not ideal without normalization at least. Volume scale is much bigger than N_lesions. Change!    
    nearest_group['distance'] = ((nearest_group[continuous_cols] - single_sample_row[continuous_cols].values)**2).sum(axis=1)
    nearest_group = nearest_group.loc[nearest_group['distance'].idxmin(), count_on]
        
    # Update the strata of the single sample group to the nearest group
    group_stats.loc[group_stats[count_on] == group, count_on] = nearest_group

    return merge_single_sample_groups(group_stats)



def flexible_stratified_split(df, 
                            group_cols=['subject_id', 'site_exp'],
                            stratify_cols=['site_exp', 'N_lesions', 'Total_volume'],
                            continuous_cols=['N_lesions', 'Total_volume'],
                            test_size=0.3, 
                            random_state=42, 
                            n_bins=5, 
                            folds = 5):
    """
    Split dataset using sklearn's train_test_split while preserving distributions
    and keeping grouped data together.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset
    group_cols : list
        Columns that define groups that should stay together (e.g., ['subject_id', 'site_exp'])
    stratify_cols : list
        Columns to consider for stratification (e.g., ['site_exp', 'N_lesions', 'Total_volume'])
    continuous_cols : list
        Columns that need discretization for stratification (e.g., ['N_lesions', 'Total_volume'])
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Random state for reproducibility
    n_bins : int, default=5
        Number of bins for continuous variables
    
    Returns:
    --------
    train_df : pandas.DataFrame
        Training dataset
    test_df : pandas.DataFrame
        Test dataset
    """
        # Split the original dataframe based on groups
    def create_group_key(row):
        return tuple(row[col] for col in group_cols)
    
    # Validate inputs
    if group_cols:
        if not all(col in df.columns for col in group_cols):
            raise ValueError("All grouping columns must be in the dataframe")
        if not all(col in df.columns for col in stratify_cols):
            raise ValueError("All stratification columns must be in the dataframe")
        if not all(col in df.columns for col in continuous_cols):
            raise ValueError("All continuous columns must be in the dataframe")
    
    # First, create stratification labels at group level
    # For continuous variables, we'll use mean
    agg_dict = {col: 'mean' for col in continuous_cols}
    # For categorical variables in stratify_cols but not in continuous_cols, we'll use first value
    cat_cols = [col for col in stratify_cols if col not in continuous_cols]
    agg_dict.update({col: 'first' for col in cat_cols})
    
    # Get group-level statistics. I need to keep the gropup cols in the dataframe
    group_stats = df.groupby(group_cols).agg(agg_dict).reset_index() if group_cols else df.copy()
    
    # Bin continuous variables
    if continuous_cols:
        kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        for col in continuous_cols:
            group_stats[f'{col}_bin'] = kbd.fit_transform(group_stats[[col]]).astype(int)
    
    # Create combined stratification label
    strata_components = []
    for col in stratify_cols:
        if col in continuous_cols:
            strata_components.append(f'{col}_bin')
        else:
            strata_components.append(col)
    print("Strata components:", strata_components)
 
    group_stats['strata'] = group_stats[strata_components].astype(str).agg('_'.join, axis=1)
    print("Strata created:", group_stats)
    
    # Avoid gorups with a single sample
    #group_stats = merge_single_sample_groups(group_stats, continuous_cols=['Total_volume'], filter_cols=[]) 

    # Check for minimum class size in strata
    strata_counts = group_stats['strata'].value_counts()
    if strata_counts.min() < 2:
        # Identify groups with fewer than 2 members
        low_count_groups = strata_counts[strata_counts < 2].index.tolist()
        raise ValueError(f"One or more classes in 'strata' have fewer than 2 members: {low_count_groups}. Cannot perform stratified split.")
    
    # Split groups while stratifying
    if folds: 
        fold=0
        df['cv_fold'] = -1
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        for train_index, test_index in skf.split(group_stats, group_stats['strata']):
            train_groups = group_stats.iloc[train_index]
            test_groups = group_stats.iloc[test_index]
            
            # Split the original dataframe based on groups
            train_mask = df.apply(lambda x: create_group_key(x) in set(create_group_key(row) for _, row in train_groups.iterrows()), axis=1)
            train_df = df[train_mask].copy()
            test_df = df[~train_mask].copy()
            df.loc[~train_mask,'cv_fold'] = fold
            fold+=1
        return df
        
    else:
        train_groups, test_groups = train_test_split(
            group_stats,
            test_size=test_size,
            random_state=random_state,
            stratify=group_stats['strata']
        )
    

        train_group_keys = set(create_group_key(row) for _, row in train_groups.iterrows())
        #test_group_keys = set(create_group_key(row) for _, row in test_groups.iterrows())
        
        train_mask = df.apply(lambda x: create_group_key(x) in train_group_keys, axis=1)
        train_df = df[train_mask].copy()
        test_df = df[~train_mask].copy()
        
        # Print distribution statistics
        print("\nDistribution Statistics:")
        
        # For categorical variables
        for col in set(stratify_cols) - set(continuous_cols):
            print(f"\n{col} Distribution (%):")
            train_dist = train_df[col].value_counts(normalize=True) * 100
            test_dist = test_df[col].value_counts(normalize=True) * 100
            dist_stats = pd.DataFrame({
                'Train %': train_dist,
                'Test %': test_dist
            }).round(2)
            print(dist_stats)
        
        # For continuous variables
        for col in continuous_cols:
            print(f"\n{col} Distribution:")
            print("\nTrain:")
            print(train_df[col].describe().round(2))
            print("\nTest:")
            print(test_df[col].describe().round(2))
        
        print("\nGroup counts:")
        train_groups = set(create_group_key(row) for _, row in train_df.iterrows())
        test_groups = set(create_group_key(row) for _, row in test_df.iterrows())
        print(f"Train unique groups: {len(train_groups)}")
        print(f"Test unique groups: {len(test_groups)}")
    
    return train_df, test_df



def create_nnunet_splits_from_csv(csv_filepath, output_filepath='splits_final.json'):
    """
    Create a splits_final.json file from a CSV with cv_fold information.
    
    Parameters:
    csv_filepath (str): Path to the input CSV file
    output_filepath (str): Path for the output JSON file
    
    Returns:
    list: The splits data structure
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_filepath)
    
    # Get unique fold values and sort them
    unique_folds = sorted(df['cv_fold'].unique())
    
    # Initialize the splits list
    splits = []
    
    # For each fold, create train/val split
    for fold in unique_folds:
        # Subjects in current fold go to validation
        val_subjects = df[df['cv_fold'] == fold]['subject_id'].tolist()
        
        # All other subjects go to training
        train_subjects = df[df['cv_fold'] != fold]['subject_id'].tolist()
        
        # Sort the lists for consistency
        val_subjects.sort()
        train_subjects.sort()
        
        # Create the fold dictionary
        fold_dict = {
            "train": train_subjects,
            "val": val_subjects
        }
        
        splits.append(fold_dict)
    
    # Save to JSON file
    with open(output_filepath, 'w') as f:
        json.dump(splits, f, indent=4)
    
    # Print summary
    print(f"Created {len(splits)} folds:")
    for i, split in enumerate(splits):
        print(f"Fold {i}: {len(split['train'])} train, {len(split['val'])} val subjects")
        print(f"  Val subjects: {split['val']}")
    
    print(f"\nSplits saved to: {output_filepath}")
    
    return splits

def validate_splits(splits, csv_filepath):
    """
    Validate that the splits cover all subjects correctly.
    
    Parameters:
    splits (list): The splits data structure
    csv_filepath (str): Path to the original CSV file
    """
    
    df = pd.read_csv(csv_filepath)
    all_subjects = set(df['subject_id'].unique())
    
    print(f"\nValidation:")
    print(f"Total subjects in CSV: {len(all_subjects)}")
    
    for i, split in enumerate(splits):
        train_set = set(split['train'])
        val_set = set(split['val'])
        
        # Check no overlap between train and val
        overlap = train_set.intersection(val_set)
        if overlap:
            print(f"❌ Fold {i}: Overlap found - {overlap}")
        else:
            print(f"✅ Fold {i}: No overlap between train/val")
        
        # Check all subjects are covered
        fold_subjects = train_set.union(val_set)
        if fold_subjects == all_subjects:
            print(f"✅ Fold {i}: All subjects covered")
        else:
            missing = all_subjects - fold_subjects
            extra = fold_subjects - all_subjects
            if missing:
                print(f"❌ Fold {i}: Missing subjects - {missing}")
            if extra:
                print(f"❌ Fold {i}: Extra subjects - {extra}")


if __name__ == "__main__":
    import os

    random_seed = 42
    test_size = 0.22
    n_bins = 3 # This actually will depends on the data a lot
    cl_data_path = '/media/secondary/fomo-fine-tuning-datasets/fomo-task2/meningioma_stats.csv'
    save_dir = '/media/secondary/fomo-fine-tuning-datasets/fomo-task2'

    cl_data = pd.read_csv(cl_data_path) 
    
    #Create two dataframes. UCL data as out-of-domain
    cl_data['domain'] = 'development'
    #cl_data.loc[cl_data['in_split4'].str.contains('ucl'), 'domain'] = 'deployment'
    # Create the dataframe for training/test in domain and out-of-domain
    developement_df = cl_data[cl_data['domain'] == 'development']
    #deployment_df = cl_data[cl_data['domain'] == 'deployment']

    train_dev_df, test_dev_df = flexible_stratified_split(developement_df, test_size=test_size, n_bins=n_bins, random_state=random_seed, group_cols=['subject_id'],
                            stratify_cols=['Total_volume','t2s_or_swi'],
                            continuous_cols=['Total_volume'], folds=False)
    #dev_df = flexible_stratified_split(developement_df, test_size=test_size, n_bins=n_bins, random_state=random_seed, group_cols=['subject_id'],
    #                        stratify_cols=['Total_volume','t2s_or_swi'],
    #                        continuous_cols=['Total_volume'], folds=5)
    train_dev_df['train'] = True
    test_dev_df['train'] = False
    #deployment_df['train'] = False
    #summary_df = pd.concat([train_dev_df, test_dev_df, deployment_df])
    summary_df = pd.concat([train_dev_df, test_dev_df])
    summary_df.to_csv(os.path.join(save_dir, f"meningioma_train_test_splits_bins-{n_bins}_seed-{random_seed}.csv"), index=False)
    #dev_df.to_csv(os.path.join(save_dir, f"train_test_splits_bins-{n_bins}_seed-{random_seed}.csv"), index=False)
    n_bins = 2
    dev_df = flexible_stratified_split(train_dev_df, test_size=test_size, n_bins=n_bins, random_state=random_seed, group_cols=['subject_id'],
                            stratify_cols=['Total_volume','t2s_or_swi'],
                            continuous_cols=['Total_volume'], folds=5)
    dev_df.to_csv(os.path.join(save_dir, f"golds_splits_bins-{n_bins}_seed-{random_seed}.csv"), index=False)
