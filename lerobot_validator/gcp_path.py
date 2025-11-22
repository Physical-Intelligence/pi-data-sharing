"""
GCP path computation logic.
"""

from pathlib import Path
from typing import Literal, Optional, Union
from datetime import datetime
from cloudpathlib import CloudPath


def compute_gcp_path(
    dataset_name: str,
    bucket_name: str,
    data_type: Literal["teleop", "eval"],
    version: Optional[str] = None,
    custom_folder_prefix: Optional[str] = None,
) -> str:
    """
    Compute the GCP path where the dataset should be uploaded.

    Args:
        dataset_name: Name of the dataset
        bucket_name: GCS bucket name (required)
        data_type: Data type - must be "teleop" or "eval" (required)
        version: Optional version string. If not provided, uses current timestamp
        custom_folder_prefix: Optional custom folder prefix (can include nested folders like "foo/bar")

    Returns:
        GCP path in the format: 
        gs://bucket/[custom_prefix/]dataset/version/data_type/
    """
    # Validate data_type
    if data_type.lower() not in ["teleop", "eval"]:
        raise ValueError(f"data_type must be 'teleop' or 'eval', got: {data_type}")
    
    # Sanitize inputs
    dataset_name = dataset_name.lower().replace(" ", "-").replace("_", "-")
    data_type = data_type.lower()

    # Generate version if not provided
    if version is None:
        version = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        version = version.replace(" ", "-").replace("_", "-")

    # Build path components
    path_parts = [f"gs://{bucket_name}"]
    
    # Add custom folder prefix if provided
    if custom_folder_prefix:
        # Remove leading/trailing slashes and sanitize
        prefix = custom_folder_prefix.strip("/")
        path_parts.append(prefix)
    
    # Add standard path components
    path_parts.extend([dataset_name, version, data_type])
    
    # Construct the GCP path
    gcp_path = "/".join(path_parts) + "/"

    return gcp_path


def format_upload_instructions(gcp_path: str, local_dataset_path: Union[str, Path, CloudPath]) -> str:
    """
    Format upload instructions.

    Args:
        gcp_path: The computed GCP path
        local_dataset_path: Local path to the dataset (can be Path or CloudPath)

    Returns:
        Formatted upload instructions
    """
    # Handle CloudPath - if it's already on GCP, provide different instructions
    if isinstance(local_dataset_path, CloudPath):
        instructions = f"""
Upload Instructions
==================

Your dataset is on GCP at:
{local_dataset_path}

Target Destination Path:
{gcp_path}

To copy/move within GCP, run:

    gsutil -m cp -r {local_dataset_path}/* {gcp_path}

Or to move:

    gsutil -m mv {local_dataset_path}/* {gcp_path}

Notes:
- The -m flag enables parallel operations for faster transfer
- Make sure you have authenticated with: gcloud auth login
- Ensure you have read/write permissions to both buckets
"""
    else:
        instructions = f"""
Upload Instructions
==================

Your dataset is ready to upload!

GCP Destination Path:
{gcp_path}

To upload your dataset, run:

    gsutil -m cp -r {local_dataset_path}/* {gcp_path}

Notes:
- The -m flag enables parallel uploads for faster transfer
- Make sure you have authenticated with: gcloud auth login
- Ensure you have write permissions to the bucket
- The upload may take some time depending on dataset size
"""
    
    instructions += """
For more information on gsutil, see:
https://cloud.google.com/storage/docs/gsutil
"""
    return instructions

