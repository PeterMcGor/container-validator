import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional

def create_dino_folds(
    splits_source: str,
    data_dir: str,  # Directory with combined .npy files
    output_dir: str,
    extracted_data_dir: str,  # Directory to save extracted modality files
    experiment_name: str,
    file_prefix: str,
    modality_info: Dict[int, Dict[str, str]],  # REQUIRED - no default
    use_val_as_test: bool = True,
    subject_id_column: str = "subject_id",
    cv_fold_column: str = "cv_fold",
    label_extension: str = ".txt",
    extraction_strategy: str = "if_missing"
):
    """
    Create DINO datasets with automatic extraction from combined files.
    
    Process:
    1. Extract modalities from combined FOMO1_sub_X.npy → FOMO1_sub_X_dwi.npy, FOMO1_sub_X_flair.npy, etc.
    2. Create DINO JSON with separate file paths for each modality
    
    Args:
        splits_source: Path to splits_final.json or CSV file
        data_dir: Directory containing combined .npy files (e.g., FOMO1_sub_X.npy)
        output_dir: Directory to save experiment JSON files
        extracted_data_dir: Directory to save extracted modality files
        experiment_name: Name of experiment
        file_prefix: Prefix of files (e.g., "FOMO1_sub_")
        modality_info: REQUIRED dictionary defining image channels:
                      {0: {"name": "dwi"}, 1: {"name": "flair"}, 2: {"name": "adc"}}
        use_val_as_test: If True, use validation subjects as test subjects
        subject_id_column: Column name for subject IDs in CSV
        cv_fold_column: Column name for CV fold in CSV
        label_extension: Extension for label files (.txt, .npy, etc.)
        extraction_strategy: "always", "if_empty", or "if_missing"
    """
    
    # Auto-detect number of image channels from modality_info
    num_image_channels = len(modality_info)
    
    print(f"🔬 Creating {num_image_channels}-channel experiment: {experiment_name}")
    print(f"📋 Modalities: {[info['name'] for info in modality_info.values()]}")
    print(f"📁 Data dir: {data_dir}")
    print(f"📁 Extracted dir: {extracted_data_dir}")
    print(f"🏷️  Label extension: {label_extension}")
    print("-" * 80)
    
    # Load splits
    splits = load_splits(splits_source, subject_id_column, cv_fold_column)
    
    # Setup paths
    data_path = Path(data_dir)
    extracted_path = Path(extracted_data_dir)
    extracted_path.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get available subjects from combined files
    available_subjects = get_available_subjects_combined(data_path, file_prefix, label_extension)
    
    print(f"Found {len(available_subjects)} subjects with combined .npy and label files")
    print(f"Available subjects: {available_subjects}")
    
    # Extract modality channels
    print(f"\n🔄 Extracting modality channels...")
    extract_modality_channels(
        data_path, 
        extracted_path, 
        file_prefix, 
        available_subjects,
        modality_info,
        extraction_strategy
    )
    
    # Process each fold
    for fold_idx in range(len(splits)):
        fold_data = splits[fold_idx]
        
        # Get training and validation subject IDs for this fold
        train_subjects = [extract_subject_id(str(case)) for case in fold_data["train"]]
        val_subjects = [extract_subject_id(str(case)) for case in fold_data["val"]]
        
        # Filter to only include subjects that have all required files
        train_subjects = [s for s in train_subjects if s in available_subjects]
        val_subjects = [s for s in val_subjects if s in available_subjects]
        
        # Handle test subjects
        if use_val_as_test:
            test_subjects = val_subjects.copy()
            print(f"\nFold {fold_idx}: Using validation subjects as test subjects")
        else:
            test_subjects = []
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
            sample_data = get_extracted_sample_data(data_path, extracted_path, file_prefix, subject, 
                                                  modality_info, label_extension)
            if sample_data:
                dino_dataset["training"].append(sample_data)
        
        # Add validation data
        for subject in val_subjects:
            sample_data = get_extracted_sample_data(data_path, extracted_path, file_prefix, subject,
                                                  modality_info, label_extension)
            if sample_data:
                dino_dataset["validation"].append(sample_data)
        
        # Add test data
        for subject in test_subjects:
            sample_data = get_extracted_sample_data(data_path, extracted_path, file_prefix, subject,
                                                  modality_info, label_extension)
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

def extract_modality_channels(
    data_path: Path, 
    extracted_path: Path, 
    file_prefix: str, 
    subjects: List[str],
    modality_info: Dict[int, Dict[str, str]],
    extraction_strategy: str = "if_missing"
):
    """
    Extract each channel as a separate .npy file with modality names.
    """
    
    expected_channels = len(modality_info)
    
    print(f"  Extracting modality channels (strategy: {extraction_strategy}):")
    for i, info in modality_info.items():
        print(f"    Channel {i}: {info['name']}.npy ({info.get('description', '')})")
    
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
            for channel_idx, info in modality_info.items():
                modality_name = info["name"]
                output_path = extracted_path / f"{file_prefix}{subject_id}_{modality_name}.npy"
                if not output_path.exists():
                    subject_files_exist = False
                    break
            
            if subject_files_exist:
                subjects_skipped += 1
                continue  # Skip this subject, all files already exist
        
        try:
            # Load the multi-channel data
            combined_data = np.load(npy_path, allow_pickle=True)
            
            if len(combined_data) != expected_channels:
                print(f"Warning: Expected {expected_channels} channels, got {len(combined_data)} for {subject_id}")
                #continue
            
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

def get_available_subjects_combined(data_path: Path, file_prefix: str, label_extension: str) -> List[str]:
    """Get available subjects from combined .npy files."""
    
    # Look for combined .npy files
    npy_files = list(data_path.glob(f"{file_prefix}*.npy"))
    available_subjects = []
    
    for npy_file in npy_files:
        # Extract subject ID: FOMO1_sub_10.npy -> 10
        subject_id = npy_file.name.replace(file_prefix, "").replace(".npy", "")
        
        # Check if label file exists
        label_file = data_path / f"{file_prefix}{subject_id}{label_extension}"
        if label_file.exists():
            available_subjects.append(subject_id)
        else:
            print(f"Warning: Missing label file for {subject_id}: {label_file}")
    
    return sorted(available_subjects)

def get_extracted_sample_data(
    data_path: Path,
    extracted_path: Path, 
    file_prefix: str, 
    subject_id: str,
    modality_info: Dict[int, Dict[str, str]],
    label_extension: str
) -> Optional[Dict[str, str]]:
    """Get sample data paths using extracted modality files."""
    
    sample_data = {}
    
    # Use separate extracted modality files
    for i in range(len(modality_info)):
        if i not in modality_info:
            print(f"Warning: No modality info for channel {i}")
            return None
        
        modality_name = modality_info[i]["name"]
        modality_file = extracted_path / f"{file_prefix}{subject_id}_{modality_name}.npy"
        
        if not modality_file.exists():
            print(f"Warning: {modality_name} file not found: {modality_file}")
            return None
        
        sample_data[f"image{i+1}"] = str(modality_file)
    
    # Add label file (from original data directory)
    label_file = data_path / f"{file_prefix}{subject_id}{label_extension}"
    if not label_file.exists():
        print(f"Warning: Label file not found: {label_file}")
        return None
    
    sample_data["label"] = str(label_file)
    
    return sample_data

def load_splits(splits_source: str, subject_id_column: str, cv_fold_column: str):
    """Load splits from JSON or CSV file."""
    splits_path = Path(splits_source)
    
    if splits_path.suffix.lower() == '.csv':
        print(f"📊 Input detected as CSV file: {splits_source}")
        
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
        
        with open(splits_source, 'r') as f:
            splits = json.load(f)
            
        print(f"✅ Loaded {len(splits)} folds from JSON")
    else:
        raise ValueError(f"Unsupported file format: {splits_path.suffix}. Use .json or .csv")
    
    return splits

def extract_subject_id(case_name: str) -> str:
    """Extract subject ID from case name."""
    case_name = str(case_name).replace(".nii.gz", "")
    
    if "sub_" in case_name:
        return case_name.split("sub_")[-1]
    else:
        return case_name

def create_both_experiments_with_extraction(
    splits_source: str,
    data_dir: str,
    base_output_dir: str,
    extracted_data_dir: str,
    experiment_base_name: str,
    file_prefix: str,
    modality_info_3ch: Dict[int, Dict[str, str]],
    modality_info_4ch: Dict[int, Dict[str, str]],
    label_extension: str = ".txt",
    use_val_as_test: bool = True,
    extraction_strategy: str = "if_missing"
):
    """Create both 3-channel and 4-channel experiments with extraction."""
    
    print(f"🔬 Creating both 3ch and 4ch experiments with extraction:")
    print(f"  📁 Data: {data_dir}")
    print(f"  📁 Extracted: {extracted_data_dir}")
    print(f"  🏷️  Label: {label_extension}")
    print(f"  🔄 Extraction: {extraction_strategy}")
    print(f"  📋 3ch: {[info['name'] for info in modality_info_3ch.values()]}")
    print(f"  📋 4ch: {[info['name'] for info in modality_info_4ch.values()]}")
    print("=" * 80)
    
    # 3-channel experiment
    print(f"\n🧪 EXPERIMENT 1: {experiment_base_name}_3ch")
    
    create_dino_with_extraction(
        splits_source=splits_source,
        data_dir=data_dir,
        output_dir=base_output_dir,
        extracted_data_dir=extracted_data_dir,
        experiment_name=f"{experiment_base_name}_3ch",
        file_prefix=file_prefix,
        modality_info=modality_info_3ch,
        use_val_as_test=use_val_as_test,
        label_extension=label_extension,
        extraction_strategy=extraction_strategy
    )
    
    # 4-channel experiment
    print(f"\n🧪 EXPERIMENT 2: {experiment_base_name}_4ch")
    
    create_dino_with_extraction(
        splits_source=splits_source,
        data_dir=data_dir,
        output_dir=base_output_dir,
        extracted_data_dir=extracted_data_dir,
        experiment_name=f"{experiment_base_name}_4ch",
        file_prefix=file_prefix,
        modality_info=modality_info_4ch,
        use_val_as_test=use_val_as_test,
        label_extension=label_extension,
        extraction_strategy=extraction_strategy
    )

# Example usage
if __name__ == "__main__":
    
    # Task001: Extract from combined .npy files + .txt labels
    print("🔬 TASK001: Extract from combined files + text labels")
    print("=" * 60)
    
    # Define modality configurations explicitly
    task1_3ch_modalities = {
        0: {"name": "dwi", "description": "DWI"},
        1: {"name": "flair", "description": "T2FLAIR"},
        2: {"name": "adc", "description": "ADC"}
    }
    
    task1_4ch_modalities = {
        0: {"name": "dwi", "description": "DWI"},
        1: {"name": "flair", "description": "T2FLAIR"},
        2: {"name": "adc", "description": "ADC"},
        3: {"name": "swi_or_t2star", "description": "SWI_OR_T2STAR"}
    }
    
    create_both_experiments_with_extraction(
        splits_source="/home/jovyan/workspace/container-validator/task2_segmentation/splits/nnunet_experiments/splits_final_no_test.json",
        data_dir="/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task001_FOMO1/",
        base_output_dir="/home/jovyan/workspace/container-validator/task1_classification/splits/dino_experiments/",
        extracted_data_dir="/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task001_FOMO1_extracted_modalities/",
        experiment_base_name="fomo-task1",
        file_prefix="FOMO1_sub_",
        modality_info_3ch=task1_3ch_modalities,
        modality_info_4ch=task1_4ch_modalities,
        label_extension=".txt",
        extraction_strategy="if_missing"  # Only extract missing files
    )
    
    # Single experiment example  
    print("\n🔬 SINGLE EXPERIMENT: Custom 3-channel with extraction")
    print("=" * 60)
    
    custom_modalities = {
        0: {"name": "dwi", "description": "DWI"},
        1: {"name": "flair", "description": "T2FLAIR"},
        2: {"name": "adc", "description": "ADC"}
    }
    
    create_dino_folds(
        splits_source="/home/jovyan/workspace/container-validator/task2_segmentation/splits/nnunet_experiments/splits_final_no_test.json",
        data_dir="/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task001_FOMO1/",
        output_dir="/home/jovyan/workspace/container-validator/task1_classification/splits/dino_experiments/",
        extracted_data_dir="/home/jovyan/shared/pedro-maciasgordaliza/fomo25/finetuning_data_preprocess/mimic-pretreaining-preprocessing/Task001_FOMO1_extracted_modalities/",
        experiment_name="fomo-task1-3ch-mimic",
        file_prefix="FOMO1_sub_",
        modality_info=custom_modalities,
        label_extension=".txt",
        extraction_strategy="if_missing"
    )