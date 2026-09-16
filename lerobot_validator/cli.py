"""
Command-line interface for the lerobot dataset validator using tyro.
"""

import sys
from typing import Literal, Optional

import tyro
from lerobot_validator.validator import LerobotDatasetValidator
from lerobot_validator.gcp_path import compute_gcp_path, format_upload_instructions
from lerobot_validator.storage import dataset_path as make_dataset_path
from lerobot_validator.storage import path_kind


def validate(
    dataset_path: str,
    data_type: Literal["teleop", "eval"],
    s3_endpoint_url: Optional[str] = None,
    s3_region: Optional[str] = None,
    video_samples: int = 5,
):
    """
    Validate lerobot dataset metadata and annotations.

    The validator expects to find these files in the dataset's meta folder:
        - {dataset_path}/meta/custom_metadata.csv (required)
        - {dataset_path}/meta/custom_annotation.json (optional)

    Args:
        dataset_path: Local, GCS, or S3-compatible dataset path.
        data_type: Data type - must be "teleop" or "eval" (required). 
                   Teleop data (training) should have is_eval_episode=False, eval data should have is_eval_episode=True.
        s3_endpoint_url: Custom S3-compatible endpoint, for example an OCI Object Storage compatibility endpoint.
        s3_region: Region used for S3 request signing.
        video_samples: Number of evenly distributed episodes to probe. Set to 0 to skip video decoding checks.
    """
    dataset_path_obj = make_dataset_path(
        dataset_path,
        s3_endpoint_url=s3_endpoint_url,
        s3_region=s3_region,
    )

    # Run validation
    print("=" * 80)
    print("Lerobot Dataset Validator")
    print("=" * 80)
    print()
    print(f"Dataset path:    {dataset_path_obj} ({path_kind(dataset_path_obj)})")
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
        video_samples=video_samples,
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
    s3_endpoint_url: Optional[str] = None,
    s3_region: Optional[str] = None,
    video_samples: int = 5,
):
    """
    Compute GCP upload path for a dataset (with optional validation).

    Args:
        dataset_path: Local, GCS, or S3-compatible dataset path.
        dataset_name: Name of the dataset
        bucket_name: GCS bucket name for upload destination (required)
        data_type: Data type - must be "teleop" or "eval" (required). 
                   Teleop data (training) should have is_eval_episode=False, eval data should have is_eval_episode=True.
        dataset_version: Dataset version (default: current timestamp)
        custom_folder_prefix: Custom folder prefix for GCP path (can include nested folders, e.g., 'foo/bar')
        skip_validation: Skip validation and only compute the path (default: False)
        s3_endpoint_url: Custom S3-compatible endpoint used to read dataset_path.
        s3_region: Region used for S3 request signing.
        video_samples: Number of evenly distributed episodes to probe. Set to 0 to skip video decoding checks.
    """
    dataset_path_obj = make_dataset_path(
        dataset_path,
        s3_endpoint_url=s3_endpoint_url,
        s3_region=s3_region,
    )

    print("=" * 80)
    print("GCP Upload Path Calculator")
    print("=" * 80)
    print()
    print(f"Dataset path:    {dataset_path_obj} ({path_kind(dataset_path_obj)})")
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
            video_samples=video_samples,
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
