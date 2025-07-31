#!/usr/bin/env python3
"""
Brain Age Regression Evaluator for Task 3: Brain Age Estimation
Comprehensive evaluation with Absolute Error, Correlation Coefficient, and additional metrics
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class RegressionEvaluator:
    """
    Evaluator for regression tasks (brain age estimation) with comprehensive metrics
    """
    
    def __init__(self, gt_dir: str, pred_dir: str):
        """
        Initialize evaluator
        
        Args:
            gt_dir: Directory with ground truth labels (ages)
            pred_dir: Directory with predictions (ages)
        """
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)
        self.results = {}
        
    def load_ground_truth(self) -> Dict[str, float]:
        """Load ground truth ages from label.txt files"""
        gt_labels = {}
        
        for subject_dir in self.gt_dir.iterdir():
            if subject_dir.is_dir():
                subject_id = subject_dir.name
                label_file = subject_dir / "label.txt"
                
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        try:
                            age = float(f.read().strip())
                            gt_labels[subject_id] = age
                        except ValueError:
                            print(f"Warning: Could not parse age for {subject_id}")
                            continue
        
        return gt_labels
    
    def load_predictions(self) -> Dict[str, float]:
        """Load predictions from prediction files - handles both directory and flat structures"""
        predictions = {}
        
        # Method 1: Check for directory structure (sub_XX/prediction.txt)
        for subject_dir in self.pred_dir.iterdir():
            if subject_dir.is_dir():
                subject_id = subject_dir.name
                
                # Try different possible prediction file names
                possible_files = ["prediction.txt", "pred.txt", "age.txt", "output.txt"]
                pred_file = None
                
                for filename in possible_files:
                    potential_file = subject_dir / filename
                    if potential_file.exists():
                        pred_file = potential_file
                        break
                
                if pred_file is None:
                    # Try any .txt file in the directory
                    txt_files = list(subject_dir.glob("*.txt"))
                    if txt_files:
                        pred_file = txt_files[0]
                
                if pred_file and pred_file.exists():
                    with open(pred_file, 'r') as f:
                        try:
                            pred_age = float(f.read().strip())
                            predictions[subject_id] = pred_age
                        except ValueError:
                            print(f"Warning: Could not parse prediction for {subject_id}")
                            continue
        
        # Method 2: Check for flat structure (sub_XX.txt files directly in pred_dir)  
        if not predictions:  # Only try this if no directory structure found
            for pred_file in self.pred_dir.glob("*.txt"):
                subject_id = pred_file.stem  # Get filename without extension
                
                try:
                    with open(pred_file, 'r') as f:
                        pred_age = float(f.read().strip())
                        predictions[subject_id] = pred_age
                except ValueError:
                    print(f"Warning: Could not parse prediction for {subject_id}")
                    continue
        
        return predictions
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        
        metrics = {}
        
        # Primary metrics (required by challenge)
        abs_errors = np.abs(y_true - y_pred)
        metrics['Mean_Absolute_Error'] = np.mean(abs_errors)
        metrics['MAE'] = metrics['Mean_Absolute_Error']  # Alias
        
        # Correlation coefficients
        try:
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)
            metrics['Pearson_Correlation'] = pearson_corr
            metrics['Pearson_p_value'] = pearson_p
        except:
            metrics['Pearson_Correlation'] = np.nan
            metrics['Pearson_p_value'] = np.nan
            
        try:
            spearman_corr, spearman_p = spearmanr(y_true, y_pred)
            metrics['Spearman_Correlation'] = spearman_corr
            metrics['Spearman_p_value'] = spearman_p
        except:
            metrics['Spearman_Correlation'] = np.nan
            metrics['Spearman_p_value'] = np.nan
        
        # Additional regression metrics
        metrics['Root_Mean_Squared_Error'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['RMSE'] = metrics['Root_Mean_Squared_Error']  # Alias
        
        try:
            metrics['R_squared'] = r2_score(y_true, y_pred)
        except:
            metrics['R_squared'] = np.nan
            
        # Bias and variance metrics
        residuals = y_pred - y_true
        metrics['Mean_Bias'] = np.mean(residuals)
        metrics['Std_Residuals'] = np.std(residuals)
        metrics['Mean_Squared_Error'] = mean_squared_error(y_true, y_pred)
        
        # Percentile-based metrics
        metrics['Median_Absolute_Error'] = np.median(abs_errors)
        metrics['90th_Percentile_AE'] = np.percentile(abs_errors, 90)
        metrics['95th_Percentile_AE'] = np.percentile(abs_errors, 95)
        metrics['Max_Absolute_Error'] = np.max(abs_errors)
        
        # Age-specific metrics (useful for brain age)
        metrics['Mean_Predicted_Age'] = np.mean(y_pred)
        metrics['Mean_True_Age'] = np.mean(y_true)
        metrics['Std_Predicted_Age'] = np.std(y_pred)
        metrics['Std_True_Age'] = np.std(y_true)
        
        # Percentage of predictions within certain error bounds
        metrics['Within_1_Year'] = np.mean(abs_errors <= 1.0) * 100  # Percentage
        metrics['Within_2_Years'] = np.mean(abs_errors <= 2.0) * 100
        metrics['Within_5_Years'] = np.mean(abs_errors <= 5.0) * 100
        metrics['Within_10_Years'] = np.mean(abs_errors <= 10.0) * 100
        
        return metrics
    
    def evaluate(self) -> Dict:
        """Run complete evaluation"""
        
        # Load data
        print("Loading ground truth ages...")
        gt_ages = self.load_ground_truth()
        print(f"Loaded {len(gt_ages)} ground truth ages")
        
        print("Loading predicted ages...")
        pred_ages = self.load_predictions()
        print(f"Loaded {len(pred_ages)} predicted ages")
        
        # Find common subjects
        common_subjects = set(gt_ages.keys()) & set(pred_ages.keys())
        print(f"Found {len(common_subjects)} subjects with both GT and predictions")
        
        if len(common_subjects) == 0:
            raise ValueError("No subjects found with both ground truth and predictions")
        
        # Prepare arrays
        subject_list = list(common_subjects)
        y_true = np.array([gt_ages[subj] for subj in subject_list])
        y_pred = np.array([pred_ages[subj] for subj in subject_list])
        
        # Calculate metrics
        print("Calculating regression metrics...")
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Add dataset info
        metrics['total_subjects'] = len(common_subjects)
        metrics['min_true_age'] = float(np.min(y_true))
        metrics['max_true_age'] = float(np.max(y_true))
        metrics['age_range'] = float(np.max(y_true) - np.min(y_true))
        
        # Store detailed results
        abs_errors = np.abs(y_true - y_pred)
        residuals = y_pred - y_true
        
        self.results = {
            'metrics': metrics,
            'per_subject_results': [
                {
                    'subject_id': subj,
                    'true_age': float(gt_ages[subj]),
                    'predicted_age': float(pred_ages[subj]),
                    'absolute_error': float(abs_errors[i]),
                    'residual': float(residuals[i])  # predicted - true (positive = overestimate)
                }
                for i, subj in enumerate(subject_list)
            ]
        }
        
        return self.results
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get metrics as a pandas DataFrame"""
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        
        metrics_data = []
        for metric_name, value in self.results['metrics'].items():
            metrics_data.append({
                'metric': metric_name,
                'value': value
            })
        
        return pd.DataFrame(metrics_data)
    
    def get_per_subject_dataframe(self) -> pd.DataFrame:
        """Get per-subject results as a pandas DataFrame"""
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        
        return pd.DataFrame(self.results['per_subject_results'])
    
    def save_results(self, output_dir: str, prefix: str = "regression_results"):
        """Save results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics CSV
        metrics_df = self.get_metrics_dataframe()
        metrics_csv_path = output_path / f"{prefix}_metrics.csv"
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"✓ Metrics saved to: {metrics_csv_path}")
        
        # Save per-subject results CSV
        per_subject_df = self.get_per_subject_dataframe()
        per_subject_csv_path = output_path / f"{prefix}_per_subject.csv"
        per_subject_df.to_csv(per_subject_csv_path, index=False)
        print(f"✓ Per-subject results saved to: {per_subject_csv_path}")
        
        # Save JSON
        json_path = output_path / f"{prefix}.json"
        with open(json_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for key, value in self.results.items():
                if key == 'metrics':
                    json_results[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                       for k, v in value.items()}
                else:
                    json_results[key] = value
            json.dump(json_results, f, indent=2)
        print(f"✓ JSON results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate brain age regression predictions for Task 3: Brain Age Estimation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python regression_evaluator.py /path/to/gt_ages /path/to/predictions -o /path/to/output/

  # Custom prefix and verbose output
  python regression_evaluator.py /path/to/gt_ages /path/to/predictions -o /path/to/output/ --prefix brain_age_eval -v

  # Using fake data for testing
  python regression_evaluator.py fake_data/fomo25/fomo-task3-val/labels fake_predictions/ -o results/ -v
        """
    )
    
    # Positional arguments
    parser.add_argument('gt_dir', type=str,
                       help='Directory containing ground truth ages (with subject folders containing label.txt)')
    parser.add_argument('pred_dir', type=str,
                       help='Directory containing age predictions (with subject folders containing prediction files)')
    
    # Required arguments
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                       help='Output directory for results')
    
    # Optional arguments
    parser.add_argument('--prefix', type=str, default='regression_results',
                       help='Prefix for output files (default: regression_results)')
    parser.add_argument('--keep-json', action='store_true',
                       help='Keep JSON results file (default: True)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.gt_dir):
        print(f"Error: Ground truth directory '{args.gt_dir}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.pred_dir):
        print(f"Error: Predictions directory '{args.pred_dir}' does not exist.")
        sys.exit(1)
    
    try:
        # Initialize evaluator
        evaluator = RegressionEvaluator(
            gt_dir=args.gt_dir,
            pred_dir=args.pred_dir
        )
        
        if args.verbose:
            print(f"Ground truth directory: {args.gt_dir}")
            print(f"Predictions directory: {args.pred_dir}")
            print(f"Output directory: {args.output_dir}")
            print()
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Save results
        evaluator.save_results(args.output_dir, args.prefix)
        
        # Print summary
        metrics = results['metrics']
        print(f"\n📊 Brain Age Evaluation Summary:")
        print(f"  Total subjects: {metrics['total_subjects']}")
        print(f"  Age range: {metrics['min_true_age']:.1f} - {metrics['max_true_age']:.1f} years")
        print(f"  Mean Absolute Error: {metrics['Mean_Absolute_Error']:.2f} years")
        print(f"  RMSE: {metrics['RMSE']:.2f} years")
        print(f"  Pearson Correlation: {metrics['Pearson_Correlation']:.4f}")
        print(f"  R²: {metrics['R_squared']:.4f}")
        print(f"  Mean Bias: {metrics['Mean_Bias']:.2f} years")
        
        if args.verbose:
            print(f"\n🎯 Accuracy within bounds:")
            print(f"  Within 1 year: {metrics['Within_1_Year']:.1f}%")
            print(f"  Within 2 years: {metrics['Within_2_Years']:.1f}%")
            print(f"  Within 5 years: {metrics['Within_5_Years']:.1f}%")
            print(f"  Within 10 years: {metrics['Within_10_Years']:.1f}%")
            
            print(f"\n📈 Age Statistics:")
            print(f"  True ages: {metrics['Mean_True_Age']:.1f} ± {metrics['Std_True_Age']:.1f}")
            print(f"  Predicted ages: {metrics['Mean_Predicted_Age']:.1f} ± {metrics['Std_Predicted_Age']:.1f}")
        
        print("\n🎉 Brain age evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()