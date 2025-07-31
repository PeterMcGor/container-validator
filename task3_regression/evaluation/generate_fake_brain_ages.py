#!/usr/bin/env python3
"""
Generate fake brain age predictions for testing the regression evaluator
Mimics the output format expected from the challenge prediction script
"""

import argparse
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple


def generate_fake_brain_ages(gt_dir: str, output_dir: str, format_type: str = "directory", 
                           seed: int = 42, noise_level: float = 3.0, bias: float = 0.0):
    """
    Generate fake brain age predictions based on ground truth ages
    
    Args:
        gt_dir: Ground truth directory with subject folders
        output_dir: Where to save fake predictions
        format_type: "directory" for sub_XX/prediction.txt or "flat" for sub_XX.txt
        seed: Random seed for reproducibility
        noise_level: Standard deviation of noise to add (in years)
        bias: Systematic bias to add (positive = overestimate ages)
    """
    
    np.random.seed(seed)
    gt_path = Path(gt_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    subjects = []
    gt_ages = []
    
    # Load ground truth ages
    for subject_dir in gt_path.iterdir():
        if subject_dir.is_dir():
            subject_id = subject_dir.name
            label_file = subject_dir / "label.txt"
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    try:
                        age = float(f.read().strip())
                        subjects.append(subject_id)
                        gt_ages.append(age)
                    except ValueError:
                        print(f"Warning: Could not parse age for {subject_id}, skipping")
                        continue
    
    print(f"Found {len(subjects)} subjects with ground truth ages")
    print(f"Age range: {min(gt_ages):.1f} - {max(gt_ages):.1f} years")
    print(f"Mean age: {np.mean(gt_ages):.1f} ± {np.std(gt_ages):.1f} years")
    
    # Generate predictions
    predictions = []
    for i, (subject_id, true_age) in enumerate(zip(subjects, gt_ages)):
        
        # Add noise and bias to true age
        noise = np.random.normal(0, noise_level)
        predicted_age = true_age + bias + noise
        
        # Ensure reasonable age bounds (typically 0-100 for brain age)
        predicted_age = np.clip(predicted_age, 0, 120)
        
        predictions.append(predicted_age)
        
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
            f.write(f"{predicted_age:.1f}")
    
    print(f"Generated {len(predictions)} fake brain age predictions in '{format_type}' format")
    print(f"Saved to: {output_path}")
    
    # Calculate and print prediction statistics
    predictions = np.array(predictions)
    gt_ages = np.array(gt_ages)
    abs_errors = np.abs(predictions - gt_ages)
    
    print(f"\nPrediction Statistics:")
    print(f"  Predicted age range: {np.min(predictions):.1f} - {np.max(predictions):.1f} years")
    print(f"  Mean predicted age: {np.mean(predictions):.1f} ± {np.std(predictions):.1f} years")
    print(f"  Mean Absolute Error: {np.mean(abs_errors):.2f} years")
    print(f"  RMSE: {np.sqrt(np.mean((predictions - gt_ages)**2)):.2f} years")
    print(f"  Correlation with GT: {np.corrcoef(gt_ages, predictions)[0,1]:.4f}")
    print(f"  Mean Bias: {np.mean(predictions - gt_ages):.2f} years")
    
    return subjects, gt_ages, predictions


def create_sample_ground_truth(output_dir: str, n_subjects: int = 5, age_range: Tuple[int, int] = (20, 80), seed: int = 42):
    """
    Create sample ground truth data for testing (if no real fake data available)
    """
    np.random.seed(seed)
    output_path = Path(output_dir)
    labels_dir = output_path / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {n_subjects} sample subjects with ages {age_range[0]}-{age_range[1]} years")
    
    for i in range(n_subjects):
        subject_id = f"sub_{i+1:02d}"
        subject_dir = labels_dir / subject_id
        subject_dir.mkdir(exist_ok=True)
        
        # Generate random age
        age = np.random.uniform(age_range[0], age_range[1])
        
        # Save age to label.txt
        label_file = subject_dir / "label.txt"
        with open(label_file, 'w') as f:
            f.write(f"{age:.1f}")
    
    print(f"Sample ground truth data created in: {labels_dir}")
    return str(labels_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Generate fake brain age predictions for testing regression evaluator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate realistic predictions with some noise
  python generate_fake_brain_ages.py fake_data/fomo25/fomo-task3-val/labels fake_brain_age_preds/

  # Generate predictions with higher noise and bias
  python generate_fake_brain_ages.py fake_data/fomo25/fomo-task3-val/labels noisy_preds/ --noise 5.0 --bias 2.0

  # Generate predictions in flat format
  python generate_fake_brain_ages.py fake_data/fomo25/fomo-task3-val/labels flat_preds/ --format flat

  # Create sample GT data and predictions (for testing without real fake data)
  python generate_fake_brain_ages.py --create-sample-gt sample_data/ --n-subjects 10
        """
    )
    
    parser.add_argument('gt_dir', type=str, nargs='?',
                       help='Ground truth directory (with subject folders containing label.txt)')
    parser.add_argument('output_dir', type=str, nargs='?',
                       help='Output directory for fake predictions')
    
    parser.add_argument('--format', type=str, choices=['directory', 'flat'], default='directory',
                       help='Output format: "directory" for sub_XX/prediction.txt or "flat" for sub_XX.txt (default: directory)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--noise', type=float, default=3.0,
                       help='Noise level (std dev in years) to add to predictions (default: 3.0)')
    parser.add_argument('--bias', type=float, default=0.0,
                       help='Systematic bias (years) to add to predictions (default: 0.0)')
    
    # Option to create sample data
    parser.add_argument('--create-sample-gt', type=str,
                       help='Create sample ground truth data in specified directory')
    parser.add_argument('--n-subjects', type=int, default=5,
                       help='Number of sample subjects to create (default: 5)')
    parser.add_argument('--age-range', type=int, nargs=2, default=[20, 80],
                       help='Age range for sample subjects (default: 20 80)')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Handle sample GT creation
    if args.create_sample_gt:
        gt_dir = create_sample_ground_truth(
            args.create_sample_gt, 
            args.n_subjects, 
            tuple(args.age_range),
            args.seed
        )
        
        # If no output_dir specified, create predictions in the same location
        if not args.output_dir:
            output_dir = str(Path(args.create_sample_gt) / "predictions")
        else:
            output_dir = args.output_dir
            
        # Generate predictions for the sample data
        subjects, gt_ages, predictions = generate_fake_brain_ages(
            gt_dir=gt_dir,
            output_dir=output_dir,
            format_type=args.format,
            seed=args.seed,
            noise_level=args.noise,
            bias=args.bias
        )
        
        print(f"\n✓ Sample data and predictions created!")
        print(f"Now you can test the evaluator with:")
        print(f"python regression_evaluator.py {gt_dir} {output_dir} -o evaluation_results/ -v")
        
        return 0
    
    # Regular prediction generation
    if not args.gt_dir or not args.output_dir:
        parser.error("gt_dir and output_dir are required unless using --create-sample-gt")
    
    # Validate inputs
    if not os.path.exists(args.gt_dir):
        print(f"Error: Ground truth directory '{args.gt_dir}' does not exist.")
        print("Use --create-sample-gt to create sample data for testing.")
        return 1
    
    try:
        # Generate fake predictions
        subjects, gt_ages, predictions = generate_fake_brain_ages(
            gt_dir=args.gt_dir,
            output_dir=args.output_dir,
            format_type=args.format,
            seed=args.seed,
            noise_level=args.noise,
            bias=args.bias
        )
        
        if args.verbose:
            print(f"\nDetailed Results:")
            for subj, gt, pred in zip(subjects, gt_ages, predictions):
                error = abs(pred - gt)
                print(f"  {subj}: True={gt:.1f}, Pred={pred:.1f}, AE={error:.1f}")
        
        print(f"\n✓ Fake brain age predictions generated successfully!")
        print(f"Now you can test the evaluator with:")
        print(f"python regression_evaluator.py {args.gt_dir} {args.output_dir} -o evaluation_results/ -v")
        
        return 0
        
    except Exception as e:
        print(f"Error generating fake brain age predictions: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())