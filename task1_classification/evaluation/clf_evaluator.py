#!/usr/bin/env python3
"""
Classification Evaluator for Task 1: Infarct Detection
Comprehensive evaluation with AUROC and additional metrics
"""

import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, balanced_accuracy_score,
    matthews_corrcoef, classification_report
)
import warnings
warnings.filterwarnings('ignore')


class ClassificationEvaluator:
    """
    Evaluator for binary classification tasks with comprehensive metrics
    """
    
    def __init__(self, gt_dir: str, pred_dir: str, threshold: float = 0.5):
        """
        Initialize evaluator
        
        Args:
            gt_dir: Directory with ground truth labels
            pred_dir: Directory with predictions  
            threshold: Threshold for converting probabilities to binary predictions
        """
        self.gt_dir = Path(gt_dir)
        self.pred_dir = Path(pred_dir)
        self.threshold = threshold
        self.results = {}
        
    def load_ground_truth(self) -> Dict[str, int]:
        """Load ground truth labels from label.txt files"""
        gt_labels = {}
        
        for subject_dir in self.gt_dir.iterdir():
            if subject_dir.is_dir():
                subject_id = subject_dir.name
                label_file = subject_dir / "label.txt"
                
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        label = f.read().strip()
                        # Handle different label formats
                        if label.lower() in ['true', '1', 'positive', 'infarct']:
                            gt_labels[subject_id] = 1
                        elif label.lower() in ['false', '0', 'negative', 'no_infarct']:
                            gt_labels[subject_id] = 0
                        else:
                            try:
                                gt_labels[subject_id] = int(float(label))
                            except ValueError:
                                print(f"Warning: Could not parse label '{label}' for {subject_id}")
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
                possible_files = ["prediction.txt", "pred.txt", "score.txt", "probability.txt"]
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
                            pred_value = float(f.read().strip())
                            predictions[subject_id] = pred_value
                        except ValueError:
                            print(f"Warning: Could not parse prediction for {subject_id}")
                            continue
        
        # Method 2: Check for flat structure (sub_XX.txt files directly in pred_dir)
        if not predictions:  # Only try this if no directory structure found
            for pred_file in self.pred_dir.glob("*.txt"):
                subject_id = pred_file.stem  # Get filename without extension
                
                try:
                    with open(pred_file, 'r') as f:
                        pred_value = float(f.read().strip())
                        predictions[subject_id] = pred_value
                except ValueError:
                    print(f"Warning: Could not parse prediction for {subject_id}")
                    continue
        
        return predictions
    
    def calculate_metrics(self, y_true: np.ndarray, y_scores: np.ndarray, 
                         y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive classification metrics"""
        
        metrics = {}
        
        # Primary metrics
        try:
            metrics['AUROC'] = roc_auc_score(y_true, y_scores)
        except ValueError as e:
            print(f"Warning: Could not calculate AUROC: {e}")
            metrics['AUROC'] = np.nan
            
        try:
            metrics['AUPRC'] = average_precision_score(y_true, y_scores)
        except ValueError as e:
            print(f"Warning: Could not calculate AUPRC: {e}")
            metrics['AUPRC'] = np.nan
        
        # Threshold-based metrics
        metrics['Accuracy'] = accuracy_score(y_true, y_pred)
        metrics['Balanced_Accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['Sensitivity'] = metrics['Recall']  # Same as recall
        metrics['F1_Score'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
        
        # Confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['True_Positives'] = int(tp)
        metrics['True_Negatives'] = int(tn)
        metrics['False_Positives'] = int(fp)
        metrics['False_Negatives'] = int(fn)
        
        # Additional derived metrics
        metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['PPV'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        metrics['FPR'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        metrics['FNR'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
        
        return metrics
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """Find optimal threshold using different criteria"""
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        
        # Youden's J statistic (maximize TPR - FPR)
        youden_j = tpr - fpr
        optimal_idx_youden = np.argmax(youden_j)
        optimal_threshold_youden = thresholds[optimal_idx_youden]
        
        # F1 score optimization
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx_f1 = np.argmax(f1_scores)
        optimal_threshold_f1 = pr_thresholds[optimal_idx_f1] if optimal_idx_f1 < len(pr_thresholds) else 0.5
        
        return {
            'optimal_threshold_youden': optimal_threshold_youden,
            'optimal_threshold_f1': optimal_threshold_f1,
            'youden_j_score': youden_j[optimal_idx_youden],
            'max_f1_score': f1_scores[optimal_idx_f1]
        }
    
    def evaluate(self) -> Dict:
        """Run complete evaluation"""
        
        # Load data
        print("Loading ground truth labels...")
        gt_labels = self.load_ground_truth()
        print(f"Loaded {len(gt_labels)} ground truth labels")
        
        print("Loading predictions...")
        predictions = self.load_predictions()
        print(f"Loaded {len(predictions)} predictions")
        
        # Find common subjects
        common_subjects = set(gt_labels.keys()) & set(predictions.keys())
        print(f"Found {len(common_subjects)} subjects with both GT and predictions")
        
        if len(common_subjects) == 0:
            raise ValueError("No subjects found with both ground truth and predictions")
        
        # Prepare arrays
        y_true = np.array([gt_labels[subj] for subj in common_subjects])
        y_scores = np.array([predictions[subj] for subj in common_subjects])
        y_pred = (y_scores >= self.threshold).astype(int)
        
        subject_list = list(common_subjects)
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = self.calculate_metrics(y_true, y_scores, y_pred)
        
        # Find optimal thresholds
        optimal_thresholds = self.find_optimal_threshold(y_true, y_scores)
        metrics.update(optimal_thresholds)
        
        # Add dataset info
        metrics['total_subjects'] = len(common_subjects)
        metrics['positive_cases'] = int(np.sum(y_true))
        metrics['negative_cases'] = int(len(y_true) - np.sum(y_true))
        metrics['threshold_used'] = self.threshold
        
        # Store detailed results
        self.results = {
            'metrics': metrics,
            'per_subject_results': [
                {
                    'subject_id': subj,
                    'ground_truth': int(gt_labels[subj]),
                    'prediction_score': float(predictions[subj]),
                    'prediction_binary': int((predictions[subj] >= self.threshold)),
                    'correct': int(gt_labels[subj] == (predictions[subj] >= self.threshold))
                }
                for subj in subject_list
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
    
    def save_results(self, output_dir: str, prefix: str = "classification_results"):
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
        description='Evaluate binary classification predictions for Task 1: Infarct Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python classification_evaluator.py /path/to/gt_labels /path/to/predictions -o /path/to/output/

  # Custom threshold and prefix
  python classification_evaluator.py /path/to/gt_labels /path/to/predictions -o /path/to/output/ -t 0.6 --prefix infarct_eval

  # Verbose output
  python classification_evaluator.py /path/to/gt_labels /path/to/predictions -o /path/to/output/ -v
        """
    )
    
    # Positional arguments
    parser.add_argument('gt_dir', type=str,
                       help='Directory containing ground truth labels (with subject folders containing label.txt)')
    parser.add_argument('pred_dir', type=str,
                       help='Directory containing predictions (with subject folders containing prediction files)')
    
    # Required arguments
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                       help='Output directory for results')
    
    # Optional arguments
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                       help='Threshold for binary classification (default: 0.5)')
    parser.add_argument('--prefix', type=str, default='classification_results',
                       help='Prefix for output files (default: classification_results)')
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
        evaluator = ClassificationEvaluator(
            gt_dir=args.gt_dir,
            pred_dir=args.pred_dir,
            threshold=args.threshold
        )
        
        if args.verbose:
            print(f"Ground truth directory: {args.gt_dir}")
            print(f"Predictions directory: {args.pred_dir}")
            print(f"Classification threshold: {args.threshold}")
            print(f"Output directory: {args.output_dir}")
            print()
        
        # Run evaluation
        results = evaluator.evaluate()
        
        # Save results
        evaluator.save_results(args.output_dir, args.prefix)
        
        # Print summary
        metrics = results['metrics']
        print(f"\n📊 Evaluation Summary:")
        print(f"  Total subjects: {metrics['total_subjects']}")
        print(f"  Positive cases: {metrics['positive_cases']}")
        print(f"  Negative cases: {metrics['negative_cases']}")
        print(f"  AUROC: {metrics['AUROC']:.4f}")
        print(f"  AUPRC: {metrics['AUPRC']:.4f}")
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  F1-Score: {metrics['F1_Score']:.4f}")
        print(f"  Sensitivity: {metrics['Sensitivity']:.4f}")
        print(f"  Specificity: {metrics['Specificity']:.4f}")
        
        if args.verbose:
            print(f"\n🎯 Optimal Thresholds:")
            print(f"  Youden's J: {metrics['optimal_threshold_youden']:.4f} (J={metrics['youden_j_score']:.4f})")
            print(f"  Max F1: {metrics['optimal_threshold_f1']:.4f} (F1={metrics['max_f1_score']:.4f})")
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()