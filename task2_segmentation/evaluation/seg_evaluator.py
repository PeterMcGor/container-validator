
import argparse
import os
import sys

from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder_simple

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root) # I need to do this because the original developers decided that having a repo name with '-' its amazing
from task2_segmentation.nnunet_utils.utils import nnUNetSumReader

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate segmentation predictions and export results to CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with labels 1 and 2
  python seg_evaluator_script.py /path/to/gt_folder /path/to/pred_folder -l 1 2 -o /path/to/output/

  # With ignore label and custom number of processes
  python seg_evaluator_script.py /path/to/gt_folder /path/to/pred_folder -l 1 2 -il 0 -np 8 -o /path/to/output/

  # Keep JSON file and use custom output names
  python seg_evaluator_script.py /path/to/gt_folder /path/to/pred_folder -l 1 2 -o /path/to/output/ --keep-json --per-case-csv results.csv --summary-csv summary.csv
        """
    )
    
    # Positional arguments
    parser.add_argument('gt_folder', type=str, 
                       help='Path to folder with ground truth segmentations')
    parser.add_argument('pred_folder', type=str, 
                       help='Path to folder with predicted segmentations')
    
    # Required arguments
    parser.add_argument('-l', '--labels', type=int, nargs='+', required=True,
                       help='List of labels to evaluate (e.g., -l 1 2 3)')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                       help='Output directory for CSV files')
    
    # Optional arguments
    parser.add_argument('-il', '--ignore-label', type=int, default=None,
                       help='Label to ignore during evaluation (default: None)')
    parser.add_argument('-np', '--num-processes', type=int, default=8,
                       help='Number of processes to use for evaluation (default: 8)')
    parser.add_argument('--chill', action='store_true',
                       help='Don\'t crash if pred_folder is missing some files from gt_folder')
    
    # Output file naming
    parser.add_argument('--per-case-csv', type=str, default='per_case_results.csv',
                       help='Name of the per-case results CSV file (default: per_case_results.csv)')
    parser.add_argument('--summary-csv', type=str, default='summary_metrics.csv',
                       help='Name of the summary metrics CSV file (default: summary_metrics.csv)')
    parser.add_argument('--json-file', type=str, default='evaluation_results.json',
                       help='Name of the intermediate JSON file (default: evaluation_results.json)')
    
    # JSON handling
    parser.add_argument('--keep-json', action='store_true',
                       help='Keep the intermediate JSON file (default: delete after CSV export)')
    
    # Verbose output
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.gt_folder):
        print(f"Error: Ground truth folder '{args.gt_folder}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(args.pred_folder):
        print(f"Error: Prediction folder '{args.pred_folder}' does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define file paths
    json_path = os.path.join(args.output_dir, args.json_file)
    per_case_csv_path = os.path.join(args.output_dir, args.per_case_csv)
    summary_csv_path = os.path.join(args.output_dir, args.summary_csv)
    
    if args.verbose:
        print(f"Ground truth folder: {args.gt_folder}")
        print(f"Prediction folder: {args.pred_folder}")
        print(f"Labels to evaluate: {args.labels}")
        print(f"Ignore label: {args.ignore_label}")
        print(f"Number of processes: {args.num_processes}")
        print(f"Output directory: {args.output_dir}")
        print(f"JSON file: {json_path}")
        print(f"Per-case CSV: {per_case_csv_path}")
        print(f"Summary CSV: {summary_csv_path}")
        print()
    
    try:
        # Step 1: Run nnUNet evaluation
        print("Running nnUNet evaluation...")
        compute_metrics_on_folder_simple(
            folder_ref=args.gt_folder,
            folder_pred=args.pred_folder,
            labels=args.labels,
            output_file=json_path,
            num_processes=args.num_processes,
            ignore_label=args.ignore_label,
            chill=args.chill
        )
        
        if args.verbose:
            print(f"✓ Evaluation completed. Results saved to: {json_path}")
        
        # Step 2: Convert JSON to CSV
        print("Converting results to CSV format...")
        reader = nnUNetSumReader(json_path)
        
        # Export both CSV files
        reader.export_both_csvs(per_case_csv_path, summary_csv_path)
        
        print(f"✓ Per-case results exported to: {per_case_csv_path}")
        print(f"✓ Summary metrics exported to: {summary_csv_path}")
        
        # Print summary statistics
        per_case_df = reader.get_data_frame()
        summary_df = reader.get_summary_dataframe()
        
        print(f"\nSummary:")
        print(f"  - Total cases evaluated: {len(per_case_df) // len(args.labels)}")
        print(f"  - Labels evaluated: {sorted(args.labels)}")
        print(f"  - Total rows in per-case CSV: {len(per_case_df)}")
        print(f"  - Total rows in summary CSV: {len(summary_df)}")
        
        if args.verbose and len(summary_df) > 0:
            print(f"\nMean Dice scores by label:")
            summary_mean = summary_df[summary_df['metric_type'] == 'mean']
            for _, row in summary_mean.iterrows():
                print(f"  - Label {row['label']}: {row['Dice']:.4f}")
        
        # Step 3: Handle JSON file
        if not args.keep_json:
            os.remove(json_path)
            if args.verbose:
                print(f"✓ Intermediate JSON file removed: {json_path}")
        else:
            print(f"✓ JSON file kept: {json_path}")
            
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        sys.exit(1)
    
    print("\n🎉 Evaluation completed successfully!")


if __name__ == '__main__':
    main()