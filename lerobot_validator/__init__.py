"""
lerobot-dataset-validator

A lightweight library for validating lerobot dataset metadata and annotations,
and computing GCP upload paths.
"""

__version__ = "0.1.0"

from lerobot_validator.validator import LerobotDatasetValidator
from lerobot_validator.gcp_path import compute_gcp_path
from lerobot_validator.v3_metadata_checker import LerobotV3MetadataChecker, ensure_tasks_parquet
from lerobot_validator.v3_checks import Issue, validate_v3_dataset

__all__ = [
    "LerobotDatasetValidator",
    "compute_gcp_path",
    "LerobotV3MetadataChecker",
    "ensure_tasks_parquet",
    "Issue",
    "validate_v3_dataset",
]
