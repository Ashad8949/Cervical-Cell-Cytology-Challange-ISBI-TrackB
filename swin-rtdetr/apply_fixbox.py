"""
Apply fixbox to Swin-RTDETR submissions.
Forces all predicted boxes to exactly 100x100 pixels (matching GT).

All GT boxes in RIVA Track B are exactly 100×100 pixels.
Model predictions vary in size — fixing them removes size noise
and improves IoU at stricter thresholds (mAP50-95).

Usage:
  python apply_fixbox.py
  python apply_fixbox.py --box_size 100
  python apply_fixbox.py --input_dir ../submissions/swin-rtdetr --output_dir ../submissions/swin-rtdetr-fixbox
"""

import os
import glob
import argparse
import pandas as pd

# Defaults
DEFAULT_BOX_SIZE = 100
DEFAULT_INPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "submissions", "swin-rtdetr"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "submissions", "swin-rtdetr-fixbox"
)


def apply_fixbox(input_csv: str, output_csv: str, box_size: int = 100):
    """
    Read a submission CSV, force width/height to box_size, and save.
    Center coordinates (x, y) remain unchanged.
    """
    df = pd.read_csv(input_csv)
    original_count = len(df)

    if "width" in df.columns and "height" in df.columns:
        df["width"] = box_size
        df["height"] = box_size
    else:
        print(f"  WARNING: {input_csv} missing width/height columns, skipping.")
        return

    # Re-index IDs
    df["id"] = range(len(df))

    df.to_csv(output_csv, index=False, float_format="%.6f")

    print(f"  {os.path.basename(input_csv):>50s}  ->  {os.path.basename(output_csv)}")
    print(f"    {original_count:,} predictions, boxes forced to {box_size}x{box_size}")


def main():
    parser = argparse.ArgumentParser(description="Apply fixbox to submissions")
    parser.add_argument("--input_dir", type=str, default=DEFAULT_INPUT_DIR,
                       help="Directory containing submission CSVs")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory for fixbox CSVs")
    parser.add_argument("--box_size", type=int, default=DEFAULT_BOX_SIZE,
                       help="Fixed box size in pixels (default: 100)")
    parser.add_argument("--input_file", type=str, default=None,
                       help="Process a single CSV file instead of a directory")
    args = parser.parse_args()

    print("=" * 80)
    print("  APPLYING FIXBOX TO SWIN-RTDETR SUBMISSIONS")
    print("=" * 80)
    print(f"  Box size: {args.box_size}x{args.box_size}")
    print(f"  Rationale: All GT boxes in RIVA are exactly {args.box_size}x{args.box_size}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_file:
        # Process single file
        basename = os.path.basename(args.input_file)
        name, ext = os.path.splitext(basename)
        output_file = os.path.join(args.output_dir, f"{name}_fixbox{ext}")
        apply_fixbox(args.input_file, output_file, args.box_size)
    else:
        # Process all CSVs in input_dir
        csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))

        if not csv_files:
            print(f"\n  No CSV files found in {args.input_dir}")
            return

        print(f"\n  Found {len(csv_files)} CSV files in {args.input_dir}\n")

        for csv_path in csv_files:
            basename = os.path.basename(csv_path)
            name, ext = os.path.splitext(basename)
            output_file = os.path.join(args.output_dir, f"{name}_fixbox{ext}")
            apply_fixbox(csv_path, output_file, args.box_size)
            print()

    # Summary
    print("=" * 80)
    print("  FIXBOX COMPLETE")
    print("=" * 80)
    output_files = sorted(glob.glob(os.path.join(args.output_dir, "*.csv")))
    print(f"\n  {len(output_files)} files written to: {args.output_dir}\n")
    for f in output_files:
        size = os.path.getsize(f)
        print(f"    {os.path.basename(f):>55s}  ({size:>10,} bytes)")

    print(f"\n  Recommended submissions:")
    # Find ensemble conf_0.0500 fixbox files
    for keyword in ["ensemble_conf_0.0500", "fold1_conf_0.0500", "fold2_conf_0.0500"]:
        matches = [f for f in output_files if keyword in os.path.basename(f)]
        for m in matches:
            print(f"    -> {os.path.basename(m)}")

    print("=" * 80)


if __name__ == "__main__":
    main()
