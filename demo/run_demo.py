#!/usr/bin/env python3
"""
Demo script showing how to use the lerobot dataset validator.

This script can be run without installing the package by adding the parent
directory to the Python path.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import without installing
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from lerobot_validator import LerobotDatasetValidator, compute_gcp_path
from lerobot_validator.gcp_path import format_upload_instructions


def main():
    """Run the demo validation."""
    # Get paths
    demo_dir = Path(__file__).parent
    dataset_path = demo_dir / "sample_dataset"

    print("=" * 80)
    print("Lerobot Dataset Validator - Demo")
    print("=" * 80)
    print()
    print(f"Dataset path:    {dataset_path}")
    print(f"Metadata CSV:    {dataset_path}/meta/custom_metadata.csv")
    print(f"Annotation JSON: {dataset_path}/meta/custom_annotation.json")
    print()
    print("Running validation...")
    print()

    # Create validator (assuming eval data for demo)
    validator = LerobotDatasetValidator(
        dataset_path=dataset_path,
        is_eval_data=True,
    )

    # Run validation
    validation_passed = validator.validate()

    # Print results
    validator.print_results()
    print()

    if validation_passed:
        print("=" * 80)
        print("Computing GCP upload path...")
        print()

        # Compute GCP path (assuming this is teleop data for demo)
        gcp_path = compute_gcp_path(
            dataset_name="demo-dataset",
            bucket_name="demo-bucket",
            data_type="teleop",
            version="v1.0.0",
        )

        print(format_upload_instructions(gcp_path, dataset_path))
        print("=" * 80)
        print()
        print("✓ Demo completed successfully!")
        print()
        print("Try modifying the demo files to see validation errors:")
        print("  1. Remove a column from custom_metadata.csv")
        print("  2. Add human_intervention to demo_ep_002 (non-eval episode)")
        print("  3. Use time values > 15s for demo_ep_001 annotations")
        print()
        return 0
    else:
        print("=" * 80)
        print("❌ Validation failed!")
        print()
        print("This demo should pass validation. If you see errors,")
        print("there may be an issue with the demo files or the library.")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

