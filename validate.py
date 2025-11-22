#!/usr/bin/env python3
"""
Lerobot Dataset Validator - Main entry point.

This script provides two commands:
1. validate: Validate dataset metadata and annotations
2. compute-path: Compute GCP upload path (with optional validation)

The validator expects to find these files in the dataset's meta folder:
    - {dataset_path}/meta/custom_metadata.csv (required)
    - {dataset_path}/meta/custom_annotation.json (optional)

Usage:
    # Validate only
    python validate.py validate --dataset-path ./dataset --data-type teleop

    # Compute upload path with validation
    python validate.py compute-path --dataset-path ./dataset --dataset-name my-dataset \
        --bucket-name my-bucket --data-type eval

For help:
    python validate.py --help
    python validate.py validate --help
    python validate.py compute-path --help
"""

from lerobot_validator.cli import main

if __name__ == "__main__":
    main()
