#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# from fomo25.src.inference.predict import load_modalities
from fomo25.src.inference.predict import load_modalities
from fomo25.src.data.task_configs import task1_config
from fomo25.src.models.supervised_cls import SupervisedClsModel

from yucca.functional.preprocessing import (
    preprocess_case_for_inference,
    reverse_preprocessing,
)


# Task-specific hardcoded configuration
predict_config = {
    # Import values from task_configs
    **task1_config,
    # Add inference-specific configs
    "model_path": "/app/models/Task001_FOMO1/mmunetvae/version_0/checkpoints/best_model.ckpt",
    "patch_size": (64, 64, 64),
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FOMO25 Task 1 - Infarct Classification")
    
    # Input paths for each modality
    parser.add_argument("--flair", type=str, help="Path to T2 FLAIR image")
    parser.add_argument("--adc", type=str, help="Path to ADC image")
    parser.add_argument("--dwi_b1000", type=str, help="Path to DWI b1000 image")
    parser.add_argument("--t2s", type=str, help="Path to T2* image (optional)")
    parser.add_argument("--swi", type=str, help="Path to SWI image (optional)")
    
    # Output path for predictions
    parser.add_argument("--output", type=str, required=True, help="Path to save output .txt file")
    
    return parser.parse_args()

def predict(args):
    """
    Predict infarct probability based on the provided modalities.
    
    Returns:
        float: Probability of positive class (infarct presence) between 0 and 1
    """
    
    #########################################################################
    # PLACEHOLDER: ADD YOUR INFERENCE CODE HERE
    #########################################################################
    # 
    # Available image paths:
    #   - args.flair: T2 FLAIR image path
    #   - args.adc: ADC image path  
    #   - args.dwi_b1000: DWI b1000 image path
    #   - args.t2s: T2* image path (may be None)
    #   - args.swi: SWI image path (may be None)
    #
    # Example steps you might implement:
    #   1. Load the images you need (not all 4 are required)
    #   2. Preprocess the images (normalize, resample, etc.)
    #   3. Load your trained model
    #   4. Run inference
    #   5. Return probability of positive class
    #
    # Example (replace with your actual code):
    #   model = load_your_model()
    #   images = load_and_preprocess_images(args)
    #   probability = model.predict(images)
    #
    #########################################################################
    
    # Dummy probability - REPLACE THIS WITH YOUR ACTUAL PREDICTION
    # probability = 0.75  # Example probability, should be between 0 and 1
    
    # NOTE: Remember the order [ preprocess task 1 ]
    # if "dwi" in file:
    #     modality_index = 0  # DWI
    # elif "flair" in file:
    #     modality_index = 1  # T2FLAIR
    # elif "adc" in file:
    #     modality_index = 2  # ADC
    # elif "swi" in file or "t2s" in file:
    #     modality_index = 3  # SWI_OR_T2STAR
    # else:
    #     continue

    # Map arguments to modality paths in expected order from task 
    if args.swi is not None:
        modality_paths = [args.dwi_b1000, args.flair, args.adc, args.swi]
    elif args.t2s is not None:
        modality_paths = [args.dwi_b1000, args.flair, args.adc, args.t2s]
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
    model = SupervisedClsModel.load_from_checkpoint(checkpoint_path=model_path)    

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
        
    # For classification, apply softmax and take argmax        
    pred_probs = F.softmax(predictions, dim=1)
    pred_label = pred_probs.argmax().item()

    # Probability of positive class (infarct presence)
    # print(predictions)
    # print(predictions.shape)
    # print(pred_probs)
    # print(pred_probs.shape)
    probability = pred_probs[0][1].item()  # Assuming class 1 is positive

    # predictions_softmax = torch.nn.functional.softmax(
    #     torch.from_numpy(predictions_original), dim=1
    # )
    # prediction_final = torch.argmax(predictions_softmax, dim=1)[0].numpy()    

    # Save the prediction
    # save_prediction(prediction_final, images[0], output_path)

    return probability

def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Get prediction probability
    probability = predict(args)
    
    # Save probability in a text file called <subject_id>.txt
    subject_id = Path(args.output).stem  # Extract subject ID from output path
    output_file = Path(args.output).parent / f"{subject_id}.txt"
    with open(output_file, 'w') as f:
        f.write(f"{probability:.3f}")

    # # And the prediction label
    # if probability >= 0.5:
    #     prediction_label = "infarct"
    # else:
    #     prediction_label = "no_infarct"
    
    return 0

if __name__ == "__main__":
    exit(main())