# FOMO Challenge Task 2 - Data Splits

This directory contains the definition of data splits for Task 2 segmentation experiments in the FOMO challenge. The splits are designed to stratify the few-shot data into train/validation/test sets for initial experiments, ensuring consistent subject allocation across different model architectures and preprocessing approaches.

## Directory Structure

```
task2_segmentation/splits/
├── dino_experiments/
│   ├── fomo-task2_2channels_mimic/
│   │   ├── fomo-task2_2channels_mimic_fold_0.json
│   │   ├── fomo-task2_2channels_mimic_fold_1.json
│   │   ├── fomo-task2_2channels_mimic_fold_2.json
│   │   ├── fomo-task2_3channels_mimic_fold_3.json
│   │   └── fomo-task2_2channels_mimic_fold_4.json
│   └── fomo-task2_3channels_mimic/
│       ├── fomo-task2_3channels_mimic_fold_0.json
│       ├── fomo-task2_3channels_mimic_fold_1.json
│       ├── fomo-task2_3channels_mimic_fold_2.json
│       ├── fomo-task2_3channels_mimic_fold_3.json
│       └── fomo-task2_3channels_mimic_fold_4.json
├── nnunet_experiments/
│   └── splits_final.json
└── README.md
```

## Split Definitions

### Initial Experiments (Train/Validation/Test)

The splits defined in this directory are used for the initial experimental phase where data is divided into three sets:
- **Training set**: Used for model training
- **Validation set**: Used for hyperparameter tuning and model selection
- **Test set**: Used for final evaluation

### Cross-Validation Strategy

All experiments use **5-fold cross-validation** to ensure robust evaluation. The subject allocation is consistent across all model architectures and preprocessing approaches.

**Important**: The same subjects are always assigned to the same folds across different experiments. What changes between experiment types is:
- Data routes/paths
- Data formats and preprocessing pipelines
- Model-specific configuration requirements

### Experiment Types

#### DINO Experiments (`dino_experiments/`)
- **2-channel variant** (`fomo-task2_2channels_mimic/`): Splits for 2-channel input DINO models
- **3-channel variant** (`fomo-task2_3channels_mimic/`): Splits for 3-channel input DINO models
- Each variant contains 5 fold definition files (fold_0 through fold_4)

#### nnU-Net Experiments (`nnunet_experiments/`)
- **splits_final.json**: Consolidated split definition compatible with nnU-Net framework requirements

## Future Experiments (Train/Validation Only)

<!-- TODO: Add specification for splitting when only train and validation sets are employed -->

### Placeholder for Train/Validation Splits

*This section will be updated with specifications for experiments that use only training and validation sets (no separate test set). The splitting strategy will maintain the same subject allocation across folds to ensure consistency with initial experiments.*

**Key principles for future splits:**
- Maintain subject consistency across folds
- Adapt data routes and formats as needed for different model architectures
- Preserve stratification strategy for few-shot learning scenarios

---

## Usage Notes

1. **Consistency**: All split files ensure the same subjects appear in corresponding folds across different experiment types
2. **Extensibility**: New experiment types can be added by creating new subdirectories with appropriate split definitions
3. **Format Compatibility**: Each subdirectory contains splits formatted specifically for the corresponding model framework (DINO, nnU-Net, etc.)

## File Format

Each JSON split file contains subject IDs organized by fold, maintaining the stratified sampling approach required for few-shot learning in medical image segmentation tasks.