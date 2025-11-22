"""
Command-line interface for the lerobot dataset validator using tyro.
"""

import sys
from typing import Literal, Optional

import tyro
from cloudpathlib import CloudPath, AnyPath

from lerobot_validator.validator import LerobotDatasetValidator
from lerobot_validator.gcp_path import compute_gcp_path, format_upload_instructions


def validate(
    dataset_path: str,
    data_type: Literal["teleop", "eval"],
):
    """
    Validate lerobot dataset metadata and annotations.

    The validator expects to find these files in the dataset's meta folder:
        - {dataset_path}/meta/custom_metadata.csv (required)
        - {dataset_path}/meta/custom_annotation.json (optional)

    Args:
        dataset_path: Path to the lerobot dataset directory (supports both local paths and GCP URIs like gs://bucket/path)
        data_type: Data type - must be "teleop" or "eval" (required). 
                   Teleop data (training) should have is_eval_episode=False, eval data should have is_eval_episode=True.
    """
    # Convert to AnyPath - supports both local Path and CloudPath (gs://)
    dataset_path_obj = AnyPath(dataset_path)
    
    # Determine path type for display
    path_type = "GCP URI" if isinstance(dataset_path_obj, CloudPath) else "Local path"

    # Run validation
    print("=" * 80)
    print("Lerobot Dataset Validator")
    print("=" * 80)
    print()
    print(f"Dataset path:    {dataset_path_obj} ({path_type})")
    print(f"Metadata CSV:    {dataset_path_obj}/meta/custom_metadata.csv (required)")
    print(f"Annotation JSON: {dataset_path_obj}/meta/custom_annotation.json (optional)")
    print(f"Data type:       {data_type.capitalize()}")
    print()
    print("Running validation...")
    print()

    # Map data_type to is_eval_data: eval=True, teleop=False
    is_eval_data = (data_type.lower() == "eval")
    
    validator = LerobotDatasetValidator(
        dataset_path=dataset_path_obj,
        is_eval_data=is_eval_data,
    )

    validation_passed = validator.validate()

    # Print results
    validator.print_results()
    print()

    if not validation_passed:
        print("=" * 80)
        print("❌ Validation failed. Please fix the errors and try again.")
        print("=" * 80)
        sys.exit(1)

    print("=" * 80)
    print("✅ Validation passed!")
    print("=" * 80)
    sys.exit(0)


def compute_upload_path(
    dataset_path: str,
    dataset_name: str,
    bucket_name: str,
    data_type: Literal["teleop", "eval"],
    dataset_version: Optional[str] = None,
    custom_folder_prefix: Optional[str] = None,
    skip_validation: bool = False,
):
    """
    Compute GCP upload path for a dataset (with optional validation).

    Args:
        dataset_path: Path to the lerobot dataset directory (supports both local paths and GCP URIs like gs://bucket/path)
        dataset_name: Name of the dataset
        bucket_name: GCS bucket name for upload destination (required)
        data_type: Data type - must be "teleop" or "eval" (required). 
                   Teleop data (training) should have is_eval_episode=False, eval data should have is_eval_episode=True.
        dataset_version: Dataset version (default: current timestamp)
        custom_folder_prefix: Custom folder prefix for GCP path (can include nested folders, e.g., 'foo/bar')
        skip_validation: Skip validation and only compute the path (default: False)
    """
    # Convert to AnyPath
    dataset_path_obj = AnyPath(dataset_path)
    path_type = "GCP URI" if isinstance(dataset_path_obj, CloudPath) else "Local path"

    print("=" * 80)
    print("GCP Upload Path Calculator")
    print("=" * 80)
    print()
    print(f"Dataset path:    {dataset_path_obj} ({path_type})")
    print(f"Dataset name:    {dataset_name}")
    print(f"Bucket:          {bucket_name}")
    print(f"Data type:       {data_type.capitalize()}")
    if dataset_version:
        print(f"Version:         {dataset_version}")
    if custom_folder_prefix:
        print(f"Folder prefix:   {custom_folder_prefix}")
    print()

    # Run validation first unless skipped
    if not skip_validation:
        print("Running validation...")
        print()
        
        is_eval_data = (data_type.lower() == "eval")
        
        validator = LerobotDatasetValidator(
            dataset_path=dataset_path_obj,
            is_eval_data=is_eval_data,
        )

        validation_passed = validator.validate()
        validator.print_results()
        print()

        if not validation_passed:
            print("=" * 80)
            print("❌ Validation failed. Please fix the errors before uploading.")
            print("=" * 80)
            sys.exit(1)

        print("✅ Validation passed!")
        print()

    # Compute GCP path
    print("=" * 80)
    
    gcp_path = compute_gcp_path(
        dataset_name=dataset_name,
        bucket_name=bucket_name,
        data_type=data_type,
        version=dataset_version,
        custom_folder_prefix=custom_folder_prefix,
    )

    print(format_upload_instructions(gcp_path, dataset_path_obj))
    print("=" * 80)

    sys.exit(0)


# Create CLI with multiple commands
cli = tyro.extras.subcommand_cli_from_dict(
    {
        "validate": validate,
        "compute-path": compute_upload_path,
    },
    description="Lerobot Dataset Validator and Upload Path Calculator",
)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()

