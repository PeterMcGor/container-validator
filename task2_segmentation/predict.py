#!/usr/bin/env python3
"""
FOMO25 Challenge - Task 2: Binary Segmentation
"""
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

from fomo25.src.inference.predict import load_modalities
from fomo25.src.data.task_configs import task2_config
from fomo25.src.models.supervised_seg import SupervisedSegModel

from yucca.functional.preprocessing import (
    preprocess_case_for_inference,
    reverse_preprocessing,
)


# Task-specific hardcoded configuration
predict_config = {
    # Import values from task_configs
    **task2_config,
    # Add inference-specific configs
    "model_path": "/app/models/Task002_FOMO2/mmunetvae/version_0/checkpoints/best_model.ckpt",
    "patch_size": (64, 64, 64),
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FOMO25 Task 2 Binary Segmentation")
    
    # Input paths for each modality
    parser.add_argument("--flair", type=str, help="Path to T2 FLAIR image")
    parser.add_argument("--dwi_b1000", type=str, help="Path to DWI b1000 image")
    parser.add_argument("--t2s", type=str, help="Path to T2* image (optional)")
    parser.add_argument("--swi", type=str, help="Path to SWI image (optional)")
    
    # Output path for segmentation mask
    parser.add_argument("--output", type=str, required=True, help="Path to save segmentation NIfTI")
    
    return parser.parse_args()

def predict_segmentation(args):
    """
    Generate binary segmentation mask based on the provided modalities.
    
    Returns:
        tuple: (segmentation_mask, reference_image) where:
            - segmentation_mask: numpy array with binary mask (0 or 1)
            - reference_image: nibabel image object for metadata
    """
    
    # Load a reference image to get shape and metadata
    reference_img = None
    for modality in ['flair', 'dwi_b1000', 't2s', 'swi']:
        path = getattr(args, modality)
        if path and Path(path).exists():
            reference_img = nib.load(path)
            break
    
    if reference_img is None:
        raise ValueError("No valid modality found")
    
    # Get image shape for creating the mask
    shape = reference_img.shape
    
    #########################################################################
    # PLACEHOLDER: ADD YOUR SEGMENTATION INFERENCE CODE HERE
    #########################################################################
    # 
    # Available image paths:
    #   - args.flair: T2 FLAIR image path
    #   - args.dwi_b1000: DWI b1000 image path
    #   - args.t2s: T2* image path (may be None)
    #   - args.swi: SWI image path (may be None)
    #
    # Example steps you might implement:
    #   1. Load the images you need (not all are required)
    #   2. Preprocess the images (normalize, resample, register, etc.)
    #   3. Load your trained segmentation model
    #   4. Run inference to get predictions
    #   5. Post-process predictions (threshold, clean up, etc.)
    #   6. Return binary mask (0 or 1 values)
    #
    # Example (replace with your actual code):
    #   model = load_your_segmentation_model()
    #   images = load_and_preprocess_images(args)
    #   prediction = model.predict(images)
    #   binary_mask = (prediction > 0.5).astype(np.uint8)
    #
    #########################################################################
    
    # Dummy segmentation - REPLACE THIS WITH YOUR ACTUAL PREDICTION
    # segmentation_mask = np.zeros(shape, dtype=np.uint8)
    
    # # Determine modality
    # if "dwi" in file.lower():
    #     modality_index = 0  # DWI
    # elif "flair" in file.lower():
    #     modality_index = 1  # T2FLAIR
    # elif "swi" in file.lower() or "t2s" in file.lower():
    #     modality_index = 2  # SWI_OR_T2STAR
    # else:
    #     return f"Warning: Skipping file {file}"
    
    # Map arguments to modality paths in expected order from task 
    if args.swi is not None:
        modality_paths = [args.dwi_b1000, args.flair, args.swi]
    elif args.t2s is not None:
        modality_paths = [args.dwi_b1000, args.flair, args.t2s]
    else:
        raise ValueError("At least one of SWI or T2* must be provided")

    # Load input images
    images = load_modalities(modality_paths)

    # Extract configuration parameters
    task_type = predict_config["task_type"]
    crop_to_nonzero = predict_config["crop_to_nonzero"]
    norm_op = predict_config["norm_op"]
    num_classes = predict_config["num_classes"]
    keep_aspect_ratio = predict_config.get("keep_aspect_ratio", True)
    patch_size = predict_config["patch_size"]
    model_path = predict_config["model_path"]

    # Define preprocessing parameters
    normalization_scheme = [norm_op] * len(modality_paths)
    target_spacing = [1.0, 1.0, 1.0]  # Isotropic 1mm spacing
    target_orientation = "RAS"

    # Apply preprocessing
    case_preprocessed, case_properties = preprocess_case_for_inference(
        crop_to_nonzero=crop_to_nonzero,
        images=images,
        intensities=None,  # Use default intensity normalization
        normalization_scheme=normalization_scheme,
        patch_size=patch_size,
        target_size=None,  # We use target_spacing instead
        target_spacing=target_spacing,
        target_orientation=target_orientation,
        allow_missing_modalities=False,
        keep_aspect_ratio=keep_aspect_ratio,
        transpose_forward=[0, 1, 2],  # Standard transpose order
    )

    # Load the model checkpoint directly with Lightning
    model = SupervisedSegModel.load_from_checkpoint(checkpoint_path=model_path)    

    # Set model to evaluation mode
    model.eval()

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    case_preprocessed = case_preprocessed.to(device)

    # Run inference
    with torch.no_grad():
       # Run the forward pass
        overlap = 0.5  # Standard overlap for sliding window

        # Get prediction
        predictions = model.model.predict(
            data=case_preprocessed,
            mode="3D",
            mirror=False,  # No test-time augmentation
            overlap=overlap,
            patch_size=patch_size,
            sliding_window_prediction=True,
            device=device,
        )
        
    # Reverse preprocessing
    transpose_forward = [0, 1, 2]
    transpose_backward = [0, 1, 2]

    predictions_original, _ = reverse_preprocessing(
        crop_to_nonzero=crop_to_nonzero,
        images=predictions,
        image_properties=case_properties,
        n_classes=num_classes,
        transpose_forward=transpose_forward,
        transpose_backward=transpose_backward,
    )

    # For segmentation, apply argmax
    # print(predictions_original.shape)
    segmentation_mask = np.argmax(predictions_original[0], axis=0)
    segmentation_mask = (segmentation_mask > 0).astype(np.uint8)
    # print(segmentation_mask.shape)
    # print("Unique values in segmentation:", np.unique(segmentation_mask))

    return segmentation_mask, reference_img

def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Generate segmentation
    segmentation_mask, reference_img = predict_segmentation(args)
    
    # Create NIfTI image with segmentation mask
    # Uses the reference image's affine matrix and header for proper spatial alignment
    output_img = nib.Nifti1Image(
        segmentation_mask, 
        reference_img.affine, 
        reference_img.header
    )
    
    # Save segmentation mask
    nib.save(output_img, args.output)
    
    return 0

if __name__ == "__main__":
    exit(main())