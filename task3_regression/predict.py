#!/usr/bin/env python3
"""
FOMO25 Challenge - Task 3: Brain Age Prediction (Regression)
"""
import argparse
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F

from fomo25.src.inference.predict import load_modalities
from fomo25.src.data.task_configs import task3_config
from fomo25.src.models.supervised_reg import SupervisedRegModel

from yucca.functional.preprocessing import (
    preprocess_case_for_inference,
    reverse_preprocessing,
)


# Task-specific hardcoded configuration
predict_config = {
    # Import values from task_configs
    **task3_config,
    # Add inference-specific configs
    "model_path": "/app/models/Task003_FOMO3/mmunetvae/version_0/checkpoints/best_model.ckpt",
    "patch_size": (64, 64, 64),
}



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FOMO25 Task 3 Brain Age Prediction")
    
    # Input paths for T1 and T2 modalities
    parser.add_argument("--t1", type=str, help="Path to T1-weighted image")
    parser.add_argument("--t2", type=str, help="Path to T2-weighted image")
    
    # Output path for predictions
    parser.add_argument("--output", type=str, required=True, help="Path to save output CSV")
    
    return parser.parse_args()

def predict_age(args):
    """
    Predict brain age based on T1 and T2 modalities.
    
    Returns:
        float: Predicted brain age in years
    """
    
    #########################################################################
    # PLACEHOLDER: ADD YOUR BRAIN AGE PREDICTION CODE HERE
    #########################################################################
    # 
    # Available image paths:
    #   - args.t1: T1-weighted image path
    #   - args.t2: T2-weighted image path
    #
    # Example steps you might implement:
    #   1. Load T1 and T2 images
    #   2. Preprocess images (normalize, skull-strip, register, etc.)
    #   3. Extract features or prepare input for your model
    #   4. Load your trained regression model
    #   5. Run inference to predict age
    #   6. Return predicted age value
    #
    # Example (replace with your actual code):
    #   model = load_your_age_prediction_model()
    #   t1_image = load_and_preprocess_image(args.t1)
    #   t2_image = load_and_preprocess_image(args.t2)
    #   features = extract_features(t1_image, t2_image)
    #   predicted_age = model.predict(features)
    #
    #########################################################################
    
    # Dummy age prediction - REPLACE THIS WITH YOUR ACTUAL PREDICTION
    # predicted_age = 45.0
    
    # Map arguments to modality paths in expected order from task     
    modality_paths = [args.t1, args.t2]    

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
    model = SupervisedRegModel.load_from_checkpoint(checkpoint_path=model_path)    

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
            
    # For regression, just take the raw prediction
    predicted_age = predictions[0, 0]
    print(predictions)

    return predicted_age

def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Get age prediction
    predicted_age = predict_age(args)
    
    # Create output TXT file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"{predicted_age:.2f}\n")
    
    return 0

if __name__ == "__main__":
    exit(main())