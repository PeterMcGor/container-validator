#!/usr/bin/env python3
"""
Generate fake predictions for testing the classification evaluator
Mimics the output format expected from the challenge prediction script
"""

import argparse
import os
import numpy as np
from pathlib import Path
from typing import List


def generate_fake_predictions(gt_dir: str, output_dir: str, format_type: str = "directory", 
                            seed: int = 42, realistic: bool = True):
    """
    Generate fake predictions based on ground truth structure
    
    Args:
        gt_dir: Ground truth directory with subject folders
        output_dir: Where to save fake predictions
        format_type: "directory" for sub_XX/prediction.txt or "flat" for sub_XX.txt
        seed: Random seed for reproducibility
        realistic: If True, make predictions somewhat correlated with ground truth
    """
    
    np.random.seed(seed)
    gt_path = Path(gt_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    subjects = []
    gt_labels = []
    
    # Load ground truth to make realistic predictions
    for subject_dir in gt_path.iterdir():
        if subject_dir.is_dir():
            subject_id = subject_dir.name
            label_file = subject_dir / "label.txt"
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    label = f.read().strip()
                    
                # Parse label
                if label.lower() in ['true', '1', 'positive', 'infarct']:
                    gt_label = 1
                elif label.lower() in ['false', '0', 'negative', 'no_infarct']:
                    gt_label = 0
                else:
                    try:
                        gt_label = int(float(label))
                    except ValueError:
                        print(f"Warning: Could not parse label '{label}' for {subject_id}, skipping")
                        continue
                
                subjects.append(subject_id)
                gt_labels.append(gt_label)
    
    print(f"Found {len(subjects)} subjects with ground truth labels")
    print(f"Positive cases: {sum(gt_labels)}, Negative cases: {len(gt_labels) - sum(gt_labels)}")
    
    # Generate predictions
    predictions = []
    for i, (subject_id, gt_label) in enumerate(zip(subjects, gt_labels)):
        
        if realistic:
            # Generate somewhat realistic predictions
            if gt_label == 1:  # Positive case
                # Higher probability for positive cases, but with some noise
                base_prob = np.random.uniform(0.6, 0.95)
                noise = np.random.normal(0, 0.1)
                prob = np.clip(base_prob + noise, 0.0, 1.0)
            else:  # Negative case
                # Lower probability for negative cases, but with some noise
                base_prob = np.random.uniform(0.05, 0.4)
                noise = np.random.normal(0, 0.1)
                prob = np.clip(base_prob + noise, 0.0, 1.0)
        else:
            # Completely random predictions
            prob = np.random.uniform(0.0, 1.0)
        
        predictions.append(prob)
        
        # Save prediction file
        if format_type == "directory":
            # Create subject directory and save prediction.txt
            subject_pred_dir = output_path / subject_id
            subject_pred_dir.mkdir(exist_ok=True)
            pred_file = subject_pred_dir / "prediction.txt"
        else:  # flat format
            # Save directly as sub_XX.txt
            pred_file = output_path / f"{subject_id}.txt"
        
        with open(pred_file, 'w') as f:
            f.write(f"{prob:.6f}")
    
    print(f"Generated {len(predictions)} fake predictions in '{format_type}' format")
    print(f"Saved to: {output_path}")
    
    # Print some statistics
    print(f"\nPrediction Statistics:")
    print(f"  Min: {min(predictions):.4f}")
    print(f"  Max: {max(predictions):.4f}")
    print(f"  Mean: {np.mean(predictions):.4f}")
    print(f"  Std: {np.std(predictions):.4f}")
    
    return subjects, gt_labels, predictions


def main():
    parser = argparse.ArgumentParser(
        description='Generate fake predictions for testing classification evaluator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate realistic predictions in directory format
  python generate_fake_predictions.py fake_data/fomo25/fomo-task1-val/labels fake_predictions_dir/

  # Generate random predictions in flat format  
  python generate_fake_predictions.py fake_data/fomo25/fomo-task1-val/labels fake_predictions_flat/ --format flat --random

  # Custom seed for reproducibility
  python generate_fake_predictions.py fake_data/fomo25/fomo-task1-val/labels fake_predictions/ --seed 123
        """
    )
    
    parser.add_argument('gt_dir', type=str,
                       help='Ground truth directory (with subject folders containing label.txt)')
    parser.add_argument('output_dir', type=str,
                       help='Output directory for fake predictions')
    
    parser.add_argument('--format', type=str, choices=['directory', 'flat'], default='directory',
                       help='Output format: "directory" for sub_XX/prediction.txt or "flat" for sub_XX.txt (default: directory)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--random', action='store_true',
                       help='Generate completely random predictions (default: somewhat realistic)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.gt_dir):
        print(f"Error: Ground truth directory '{args.gt_dir}' does not exist.")
        return 1
    
    try:
        # Generate fake predictions
        subjects, gt_labels, predictions = generate_fake_predictions(
            gt_dir=args.gt_dir,
            output_dir=args.output_dir,
            format_type=args.format,
            seed=args.seed,
            realistic=not args.random
        )
        
        if args.verbose:
            print(f"\nDetailed Results:")
            for subj, gt, pred in zip(subjects, gt_labels, predictions):
                print(f"  {subj}: GT={gt}, Pred={pred:.4f}")
        
        print(f"\n✓ Fake predictions generated successfully!")
        print(f"Now you can test the evaluator with:")
        print(f"python classification_evaluator.py {args.gt_dir} {args.output_dir} -o evaluation_results/ -v")
        
        return 0
        
    except Exception as e:
        print(f"Error generating fake predictions: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())