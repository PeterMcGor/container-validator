import json
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional

def create_nnunet_splits_from_csv(csv_filepath, output_filepath='splits_final.json'):
    """
    Create a splits_final.json file from a CSV with cv_fold information.
    """
    df = pd.read_csv(csv_filepath)
    unique_folds = sorted(df['cv_fold'].unique())
    
    splits = []
    for fold in unique_folds:
        val_subjects = df[df['cv_fold'] == fold]['subject_id'].tolist()
        train_subjects = df[df['cv_fold'] != fold]['subject_id'].tolist()
        
        splits.append({
            "train": sorted(train_subjects),
            "val": sorted(val_subjects)
        })
    
    with open(output_filepath, 'w') as f:
        json.dump(splits, f, indent=4)
    
    return splits

def create_dino_datasets_from_splits(
    preprocessed_data_path: str,
    splits_source: str,  # Can be JSON file or CSV file
    output_dir: str,
    extracted_data_dir: str,
    experiment_name: str,
    file_prefix: str = "FOMO2_sub_",
    num_image_channels: int = 3,
    use_val_as_test: bool = True,
    subject_id_column: str = "subject_id",
    cv_fold_column: str = "cv_fold",
    modality_info: Dict[int, Dict[str, str]] = None,
    extraction_strategy: str = "if_missing"
):
    """
    Convert NPY/PKL dataset to 3D DINO format for cross-validation using flexible splits input.
    
    Args:
        preprocessed_data_path: Path to folder containing .npy/.pkl files
        splits_source: Path to either:
                      - splits_final.json file (with train/val structure)
                      - CSV file (with cv_fold column)
        output_dir: Directory to save the experiment folder and JSON files
        extracted_data_dir: Directory to save extracted modality .npy files
        experiment_name: Name of experiment (used for folder name and JSON prefix)
        file_prefix: Prefix of your .npy/.pkl files (e.g., "FOMO2_sub_")
        num_image_channels: Number of image channels to use (2 or 3)
        use_val_as_test: If True, use validation subjects as test subjects when no separate test set
        subject_id_column: Column name for subject IDs in CSV (default: "subject_id")
        cv_fold_column: Column name for CV fold in CSV (default: "cv_fold")
        modality_info: Dictionary mapping channel index to {"name": str, "description": str}
                      If None, uses default FOMO setup (dwi, flair, t2star, label)
        extraction_strategy: When to extract files:
                           - "always": Always extract (overwrite existing)
                           - "if_empty": Only if extraction folder is empty  
                           - "if_missing": Only extract files that don't exist (case by case)
    """
    
    # Determine if input is JSON or CSV and load splits accordingly
    splits_path = Path(splits_source)
    
    if splits_path.suffix.lower() == '.csv':
        print(f"📊 Input detected as CSV file: {splits_source}")
        print(f"   Converting CSV to splits using columns: {subject_id_column}, {cv_fold_column}")
        
        # Convert CSV to splits format
        df = pd.read_csv(splits_source)
        unique_folds = sorted(df[cv_fold_column].unique())
        
        splits = []
        for fold in unique_folds:
            val_subjects = df[df[cv_fold_column] == fold][subject_id_column].tolist()
            train_subjects = df[df[cv_fold_column] != fold][subject_id_column].tolist()
            
            splits.append({
                "train": sorted([str(s) for s in train_subjects]),
                "val": sorted([str(s) for s in val_subjects])
            })
        
        print(f"✅ Created {len(splits)} folds from CSV")
        
    elif splits_path.suffix.lower() == '.json':
        print(f"📊 Input detected as JSON file: {splits_source}")
        
        # Load existing JSON splits
        with open(splits_source, 'r') as f:
            splits = json.load(f)
            
        print(f"✅ Loaded {len(splits)} folds from JSON")
        
    else:
        raise ValueError(f"Unsupported file format: {splits_path.suffix}. Use .json or .csv")
    
    # Setup paths
    data_path = Path(preprocessed_data_path)
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create extracted data directory
    extracted_path = Path(extracted_data_dir)
    extracted_path.mkdir(parents=True, exist_ok=True)
    
    # Get all available subjects from the preprocessed data
    npy_files = list(data_path.glob(f"{file_prefix}*.npy"))
    available_subjects = []
    
    for npy_file in npy_files:
        subject_id = npy_file.name.replace(file_prefix, "").replace(".npy", "")
        pkl_file = data_path / f"{file_prefix}{subject_id}.pkl"
        
        if pkl_file.exists():
            available_subjects.append(subject_id)
        else:
            print(f"Warning: Missing PKL file for {subject_id}")
    
    print(f"Found {len(available_subjects)} subjects with both .npy and .pkl files")
    print(f"Available subjects: {sorted(available_subjects)}")
    print(f"Creating {num_image_channels}-channel experiment with {len(splits)} folds")
    
    # Extract individual modality channels
    print(f"\n🔄 Extracting individual modality channels...")
    
    # Use the provided modality_info or set default
    if modality_info is None:
        modality_info = {
            0: {"name": "dwi", "description": "DWI (Diffusion Weighted Imaging)"},
            1: {"name": "flair", "description": "T2FLAIR"},
            2: {"name": "t2star", "description": "SWI_OR_T2STAR (Susceptibility Weighted Imaging or T2*)"},
            3: {"name": "label", "description": "Segmentation Label"}
        }
    
    extract_modality_channels_from_npy(
        data_path, 
        extracted_path, 
        file_prefix, 
        available_subjects,
        modality_info=modality_info,
        extraction_strategy=extraction_strategy
    )
    
    # Process each fold
    for fold_idx in range(len(splits)):
        fold_data = splits[fold_idx]
        
        # Get training and validation subject IDs for this fold
        train_subjects = [extract_subject_id(str(case)) for case in fold_data["train"]]
        val_subjects = [extract_subject_id(str(case)) for case in fold_data["val"]]
        
        # Filter to only include subjects that have preprocessed data
        train_subjects = [s for s in train_subjects if s in available_subjects]
        val_subjects = [s for s in val_subjects if s in available_subjects]
        
        # Handle test subjects
        if use_val_as_test:
            test_subjects = val_subjects.copy()  # Use validation subjects as test subjects
            print(f"\nFold {fold_idx}: Using validation subjects as test subjects")
        else:
            test_subjects = []  # No test subjects
            print(f"\nFold {fold_idx}: No test subjects (empty test set)")
        
        print(f"  Train subjects: {len(train_subjects)} - {train_subjects}")
        print(f"  Val subjects: {len(val_subjects)} - {val_subjects}")
        print(f"  Test subjects: {len(test_subjects)} - {test_subjects}")
        
        # Create DINO format dataset
        dino_dataset = {
            "training": [],
            "validation": [],
            "test": []
        }
        
        # Add training data
        for subject in train_subjects:
            sample_data = get_modality_channel_paths(extracted_path, file_prefix, subject, data_path, num_image_channels, modality_info)
            if sample_data:
                dino_dataset["training"].append(sample_data)
        
        # Add validation data
        for subject in val_subjects:
            sample_data = get_modality_channel_paths(extracted_path, file_prefix, subject, data_path, num_image_channels, modality_info)
            if sample_data:
                dino_dataset["validation"].append(sample_data)
        
        # Add test data
        for subject in test_subjects:
            sample_data = get_modality_channel_paths(extracted_path, file_prefix, subject, data_path, num_image_channels, modality_info)
            if sample_data:
                dino_dataset["test"].append(sample_data)
        
        # Save fold dataset
        output_file = output_path / f"{experiment_name}_fold_{fold_idx}.json"
        with open(output_file, 'w') as f:
            json.dump(dino_dataset, f, indent=2)
        
        print(f"  Saved to: {output_file}")
        print(f"  Training samples: {len(dino_dataset['training'])}")
        print(f"  Validation samples: {len(dino_dataset['validation'])}")
        print(f"  Test samples: {len(dino_dataset['test'])}")

def create_both_experiments_from_splits(
    preprocessed_data_path: str,
    splits_source: str,  # Can be JSON file or CSV file
    base_output_dir: str,
    extracted_data_dir: str,
    experiment1_name: str = "2channel_dwi_flair",
    experiment2_name: str = "3channel_dwi_flair_swi/t2star",
    file_prefix: str = "FOMO2_sub_",
    use_val_as_test: bool = True,
    subject_id_column: str = "subject_id",
    cv_fold_column: str = "cv_fold",
    modality_info: Dict[int, Dict[str, str]] = None,
    extraction_strategy: str = "if_missing"
):
    """
    Create both 2-channel and 3-channel experiments from flexible splits input.
    
    Args:
        modality_info: Dictionary mapping channel index to {"name": str, "description": str}
                      If None, uses default FOMO setup (dwi, flair, t2star, label)
        extraction_strategy: When to extract files:
                           - "always": Always extract (overwrite existing)
                           - "if_empty": Only if extraction folder is empty  
                           - "if_missing": Only extract files that don't exist (case by case)
    """
    
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Determine input type
    splits_path = Path(splits_source)
    input_type = "CSV" if splits_path.suffix.lower() == '.csv' else "JSON"
    
    # Use the provided modality_info or set default
    if modality_info is None:
        modality_info = {
            0: {"name": "dwi", "description": "DWI (Diffusion Weighted Imaging)"},
            1: {"name": "flair", "description": "T2FLAIR"},
            2: {"name": "t2star", "description": "SWI_OR_T2STAR (Susceptibility Weighted Imaging or T2*)"},
            3: {"name": "label", "description": "Segmentation Label"}
        }
    
    print("🔬 Creating BOTH experiments from flexible splits input:")
    print(f"  📁 Input: {splits_source} ({input_type} format)")
    print(f"  📊 Experiment 1: {experiment1_name} (image1=DWI + image2=T2FLAIR + label)")
    print(f"  📊 Experiment 2: {experiment2_name} (image1=DWI + image2=T2FLAIR + image3=T2STAR + label)")
    print(f"  💾 Extracted data will be saved to: {extracted_data_dir}")
    print(f"  🧪 Test strategy: {'Use validation as test' if use_val_as_test else 'Empty test set'}")
    print(f"  🔄 Extraction strategy: {extraction_strategy}")
    print("=" * 80)
    
    # Experiment 1: 2-channel (image1=dwi + image2=flair)
    print(f"\n🧪 EXPERIMENT 1: {experiment1_name}")
    print("-" * 50)
    
    create_dino_datasets_from_splits(
        preprocessed_data_path=preprocessed_data_path,
        splits_source=splits_source,
        output_dir=base_output_dir,
        extracted_data_dir=extracted_data_dir,
        experiment_name=experiment1_name,
        file_prefix=file_prefix,
        num_image_channels=2,
        use_val_as_test=use_val_as_test,
        subject_id_column=subject_id_column,
        cv_fold_column=cv_fold_column,
        modality_info=modality_info,
        extraction_strategy=extraction_strategy
    )
    
    # Experiment 2: 3-channel (image1=dwi + image2=flair + image3=t2star)
    print(f"\n🧪 EXPERIMENT 2: {experiment2_name}")
    print("-" * 50)
    
    create_dino_datasets_from_splits(
        preprocessed_data_path=preprocessed_data_path,
        splits_source=splits_source,
        output_dir=base_output_dir,
        extracted_data_dir=extracted_data_dir,
        experiment_name=experiment2_name,
        file_prefix=file_prefix,
        num_image_channels=3,
        use_val_as_test=use_val_as_test,
        subject_id_column=subject_id_column,
        cv_fold_column=cv_fold_column,
        modality_info=modality_info,
        extraction_strategy=extraction_strategy
    )
    
    print("\n" + "=" * 80)
    print("✅ BOTH EXPERIMENTS COMPLETED!")
    print(f"📁 Input: {splits_source} ({input_type})")
    print("📁 Output structure:")
    print(f"   {extracted_data_dir}/                     # Extracted modality files")
    print("   ├── FOMO2_sub_X_dwi.npy              # DWI")
    print("   ├── FOMO2_sub_X_flair.npy            # T2FLAIR")
    print("   ├── FOMO2_sub_X_swi.npy              # SWI_OR_T2STAR")
    print("   └── FOMO2_sub_X_label.npy            # Segmentation")
    print(f"   {base_output_dir}/")
    print(f"   ├── {experiment1_name}/")
    print(f"   │   ├── {experiment1_name}_fold_0.json")
    print("   │   ├── ... (fold files)")
    print(f"   └── {experiment2_name}/")
    print(f"       ├── {experiment2_name}_fold_0.json")
    print("       └── ... (fold files)")

# Keep the existing helper functions
def extract_modality_channels_from_npy(
    data_path: Path, 
    extracted_path: Path, 
    file_prefix: str, 
    subjects: List[str],
    modality_info: Dict[int, Dict[str, str]] = None,
    extraction_strategy: str = "if_missing"
):
    """
    Extract each channel as a separate .npy file with modality names.
    
    Args:
        data_path: Path to folder containing combined .npy files
        extracted_path: Path to save extracted modality files
        file_prefix: Prefix of .npy files
        subjects: List of subject IDs to process
        modality_info: Dictionary mapping channel index to {"name": str, "description": str}
                      If None, uses default FOMO setup
        extraction_strategy: When to extract files:
                           - "always": Always extract (overwrite existing)
                           - "if_empty": Only if extraction folder is empty
                           - "if_missing": Only extract files that don't exist (case by case)
    """
    
    # Default modality info for FOMO experiments
    if modality_info is None:
        modality_info = {
            0: {"name": "dwi", "description": "DWI (Diffusion Weighted Imaging)"},
            1: {"name": "flair", "description": "T2FLAIR"},
            2: {"name": "t2star", "description": "SWI_OR_T2STAR (Susceptibility Weighted Imaging or T2*)"},
            3: {"name": "label", "description": "Segmentation Label"}
        }
    
    expected_channels = len(modality_info)
    
    print(f"  Extracting modality channels (strategy: {extraction_strategy}):")
    for i, info in modality_info.items():
        print(f"    Channel {i}: {info['name']}.npy ({info['description']})")
    
    # Check extraction strategy
    if extraction_strategy == "if_empty":
        # Check if any extracted files exist
        existing_files = list(extracted_path.glob(f"{file_prefix}*_*.npy"))
        if existing_files:
            print(f"  ✅ Found {len(existing_files)} existing extracted files, skipping extraction...")
            return
    
    subjects_processed = 0
    subjects_skipped = 0
    
    for subject_id in subjects:
        npy_path = data_path / f"{file_prefix}{subject_id}.npy"
        
        # Check if extraction is needed for this subject (if strategy is "if_missing")
        if extraction_strategy == "if_missing":
            subject_files_exist = True
            for channel_idx in modality_info.keys():
                modality_name = modality_info[channel_idx]["name"]
                output_path = extracted_path / f"{file_prefix}{subject_id}_{modality_name}.npy"
                if not output_path.exists():
                    subject_files_exist = False
                    break
            
            if subject_files_exist:
                subjects_skipped += 1
                continue  # Skip this subject, all files already exist
        
        try:
            combined_data = np.load(npy_path, allow_pickle=True)
            
            if len(combined_data) != expected_channels:
                print(f"Warning: Expected {expected_channels} channels, got {len(combined_data)} for {subject_id}")
                continue
            
            # Extract each channel
            for channel_idx, data_array in enumerate(combined_data):
                if channel_idx not in modality_info:
                    print(f"Warning: No modality info for channel {channel_idx}, skipping...")
                    continue
                    
                modality_name = modality_info[channel_idx]["name"]
                output_path = extracted_path / f"{file_prefix}{subject_id}_{modality_name}.npy"
                
                # Save the extracted channel
                np.save(output_path, data_array)
                print(f"  Extracted {modality_name}: {output_path.name} - Shape: {data_array.shape}")
            
            subjects_processed += 1
                
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
    
    print(f"  📊 Extraction summary: {subjects_processed} subjects processed, {subjects_skipped} subjects skipped")

def get_modality_channel_paths(
    extracted_path: Path, 
    file_prefix: str, 
    subject_id: str, 
    data_path: Path, 
    num_image_channels: int = 3,
    modality_info: Dict[int, Dict[str, str]] = None
) -> Optional[Dict[str, str]]:
    """Get paths to individual modality files in DINO format."""
    
    sample_data = {}
    
    # Use provided modality_info or default
    if modality_info is None:
        modality_info = {
            0: {"name": "dwi", "description": "DWI (Diffusion Weighted Imaging)"},
            1: {"name": "flair", "description": "T2FLAIR"},
            2: {"name": "t2star", "description": "SWI_OR_T2STAR (Susceptibility Weighted Imaging or T2*)"},
            3: {"name": "label", "description": "Segmentation Label"}
        }
    
    # Create mapping from image index to modality name
    modality_mapping = {}
    image_channel_idx = 0
    for channel_idx, info in modality_info.items():
        if info["name"] != "label":  # Skip label channel for image mapping
            image_channel_idx += 1
            modality_mapping[image_channel_idx] = info["name"]
    
    # Add image channels
    for i in range(1, num_image_channels + 1):
        if i not in modality_mapping:
            print(f"Warning: No modality mapping for image{i}")
            return None
            
        modality_name = modality_mapping[i]
        image_key = f"image{i}"
        image_path = extracted_path / f"{file_prefix}{subject_id}_{modality_name}.npy"
        
        if not image_path.exists():
            print(f"Warning: {modality_name} file not found: {image_path}")
            return None
        
        sample_data[image_key] = str(image_path)
    
    # Add label - find label modality name
    label_modality_name = None
    for channel_idx, info in modality_info.items():
        if info["name"] == "label":
            label_modality_name = "label"
            break
    
    if label_modality_name is None:
        print(f"Warning: No label modality found in modality_info")
        return None
    
    label_path = extracted_path / f"{file_prefix}{subject_id}_{label_modality_name}.npy"
    if not label_path.exists():
        print(f"Warning: Label file not found: {label_path}")
        return None
    
    sample_data["label"] = str(label_path)
    
    return sample_data

def extract_subject_id(nnunet_case_name: str) -> str:
    """Extract subject ID from case name."""
    case_name = str(nnunet_case_name).replace(".nii.gz", "")
    
    if "sub_" in case_name:
        return case_name.split("sub_")[-1]
    else:
        return case_name

# Example usage
if __name__ == "__main__":
    # Example paths - update according to your setup
    preprocessed_data = "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task002_FOMO2"
    splits_input = "/home/jovyan/workspace/container-validator/task2_segmentation/splits/nnunet_experiments/splits_final_no_test.json"  # Can be either JSON or CSV file
    output_directory = "/home/jovyan/workspace/container-validator/task2_segmentation/splits/dino_experiments/"
    extracted_directory = "/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task002_FOMO2_extracted_modalities"
    
    # Default FOMO modality configuration
    fomo_modalities = {
        0: {"name": "dwi", "description": "DWI (Diffusion Weighted Imaging)"},
        1: {"name": "flair", "description": "T2FLAIR"},
        2: {"name": "swi", "description": "SWI_OR_T2STAR (Susceptibility Weighted Imaging or T2*)"},
        3: {"name": "label", "description": "Segmentation Label"}
    }
    
    # Custom modality configuration example (for different experiments)
    custom_modalities = {
        0: {"name": "t1", "description": "T1-weighted"},
        1: {"name": "t2", "description": "T2-weighted"},
        2: {"name": "flair", "description": "FLAIR"},
        3: {"name": "swi", "description": "Susceptibility Weighted Imaging"},
        4: {"name": "label", "description": "Segmentation Label"}
    }
    
    # Option 1: Use CSV file with default FOMO setup
    create_both_experiments_from_splits(
        preprocessed_data_path=preprocessed_data,
        splits_source=splits_input,  # CSV file
        base_output_dir=output_directory,
        extracted_data_dir=extracted_directory,
        experiment1_name="fomo-task2_2channels_no_test_mimic",
        experiment2_name="fomo-task2_3channels_no_test_mimic", 
        file_prefix="FOMO2_sub_",
        use_val_as_test=True,
        extraction_strategy="if_missing",  # Only extract missing files
        modality_info=fomo_modalities  # Use default FOMO setup (4 channels)
    )
    
    # Option 2: Custom experiment with different modalities
    # create_both_experiments_from_splits(
    #     preprocessed_data_path=preprocessed_data,
    #     splits_source="custom_splits.csv",
    #     base_output_dir=output_directory,
    #     extracted_data_dir=extracted_directory,
    #     experiment1_name="custom_2channel_t1_t2",
    #     experiment2_name="custom_3channel_t1_t2_flair",
    #     file_prefix="CUSTOM_sub_",
    #     modality_info=custom_modalities,  # 5 channels automatically detected
    #     extraction_strategy="always"  # Always re-extract
    # )
    
    # Option 3: Single experiment with specific extraction strategy
    # create_dino_datasets_from_splits(
    #     preprocessed_data_path=preprocessed_data,
    #     splits_source="splits_final.json",  # JSON file
    #     output_dir=output_directory,
    #     extracted_data_dir=extracted_directory,
    #     experiment_name="single_experiment_always_extract",
    #     file_prefix="FOMO2_sub_",
    #     num_image_channels=3,
    #     use_val_as_test=True,
    #     modality_info=fomo_modalities,  # 4 channels automatically detected
    #     extraction_strategy="always"  # Always re-extract even if files exist
    # )