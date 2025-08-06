import os
import shutil
from pathlib import Path
import argparse


def flatten_structure(base_dir: str, dest_dir: str):
    base_dir = Path(base_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Loop through all subject directories
    for subject_dir in base_dir.glob("sub_*"):
        subject_name = subject_dir.name
        ses_dir = subject_dir / "ses_1"

        if ses_dir.exists():
            # Handle seg.nii.gz
            nii_file = ses_dir / "seg.nii.gz"
            if nii_file.exists():
                dest_file = dest_dir / f"{subject_name}.nii.gz"
                shutil.copy2(nii_file, dest_file)
                print(f"Copied: {nii_file} -> {dest_file}")

            # Handle label.txt
            txt_file = ses_dir / "label.txt"
            if txt_file.exists():
                # dest_file = dest_dir / f"{subject_name}.txt"
                dest_subject_dir = dest_dir / subject_name
                dest_subject_dir.mkdir(exist_ok=True)
                dest_file = dest_subject_dir / f"label.txt"
                shutil.copy2(txt_file, dest_file)
                print(f"Copied: {txt_file} -> {dest_file}")


def main():
    parser = argparse.ArgumentParser(description="Flatten subject folder structure.")
    parser.add_argument("--input", "-i", required=True, help="Path to base input folder (e.g., task_x/labels)")
    parser.add_argument("--output", "-o", required=True, help="Path to destination output folder")

    args = parser.parse_args()
    flatten_structure(args.input, args.output)


if __name__ == "__main__":
    main()
