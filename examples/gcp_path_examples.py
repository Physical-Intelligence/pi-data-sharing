#!/usr/bin/env python3
"""
Examples demonstrating different GCP path computation options.
"""

from lerobot_validator import compute_gcp_path


def example_basic_path():
    """Basic GCP path computation."""
    print("=" * 60)
    print("Example 1: Basic GCP Path")
    print("=" * 60)
    
    path = compute_gcp_path(
        dataset_name="pick-and-place",
        bucket_name="my-bucket",
        version="v1.0",
    )
    
    print(f"Path: {path}")
    print()
    # Output: gs://my-bucket/pick-and-place/v1.0/


def example_custom_folder_prefix():
    """GCP path with custom folder prefix."""
    print("=" * 60)
    print("Example 2: Custom Folder Prefix")
    print("=" * 60)
    
    path = compute_gcp_path(
        dataset_name="pick-and-place",
        bucket_name="my-bucket",
        version="v1.0",
        custom_folder_prefix="my-project/datasets",
    )
    
    print(f"Path: {path}")
    print()
    # Output: gs://my-bucket/my-project/datasets/pick-and-place/v1.0/


def example_with_teleop_data_type():
    """GCP path for teleop data."""
    print("=" * 60)
    print("Example 3: Teleop Data Path")
    print("=" * 60)
    
    teleop_path = compute_gcp_path(
        dataset_name="pick-and-place",
        bucket_name="my-bucket",
        data_type="teleop",
        version="v1.0",
    )
    
    print(f"Teleop path: {teleop_path}")
    print()
    # Output: gs://my-bucket/pick-and-place/v1.0/teleop/


def example_with_autonomous_data_type():
    """GCP path for autonomous data."""
    print("=" * 60)
    print("Example 4: Autonomous Data Path")
    print("=" * 60)
    
    autonomous_path = compute_gcp_path(
        dataset_name="pick-and-place",
        bucket_name="my-bucket",
        data_type="autonomous",
        version="v1.0",
    )
    
    print(f"Autonomous path: {autonomous_path}")
    print()
    # Output: gs://my-bucket/pick-and-place/v1.0/autonomous/


def example_custom_prefix_with_data_type():
    """Custom prefix with data type."""
    print("=" * 60)
    print("Example 5: Custom Prefix + Data Type")
    print("=" * 60)
    
    path = compute_gcp_path(
        dataset_name="pick-and-place",
        bucket_name="my-bucket",
        data_type="teleop",
        version="v1.0",
        custom_folder_prefix="project-alpha/robot-data",
    )
    
    print(f"Path: {path}")
    print()
    # Output: gs://my-bucket/project-alpha/robot-data/pick-and-place/v1.0/teleop/


def example_different_buckets():
    """GCP paths with different buckets."""
    print("=" * 60)
    print("Example 6: Different Buckets")
    print("=" * 60)
    
    path1 = compute_gcp_path(
        dataset_name="pick-and-place",
        bucket_name="production-bucket",
        data_type="teleop",
        version="v1.0",
    )
    
    path2 = compute_gcp_path(
        dataset_name="pick-and-place",
        bucket_name="staging-bucket",
        data_type="autonomous",
        version="v1.0",
    )
    
    print(f"Production: {path1}")
    print(f"Staging:    {path2}")
    print()
    # Production: gs://production-bucket/pick-and-place/v1.0/teleop/
    # Staging:    gs://staging-bucket/pick-and-place/v1.0/autonomous/


def example_all_options():
    """Using all options together."""
    print("=" * 60)
    print("Example 7: All Options Combined")
    print("=" * 60)
    
    path = compute_gcp_path(
        dataset_name="pick-and-place",
        bucket_name="my-custom-bucket",
        data_type="teleop",
        version="v2.1.0",
        custom_folder_prefix="experiments/batch-42",
    )
    
    print(f"Path: {path}")
    print()
    # Output: gs://my-custom-bucket/experiments/batch-42/pick-and-place/v2.1.0/teleop/


if __name__ == "__main__":
    """Run all examples."""
    print("\n")
    print("GCP Path Computation Examples")
    print("=" * 60)
    print("\n")
    
    example_basic_path()
    example_custom_folder_prefix()
    example_with_teleop_data_type()
    example_with_autonomous_data_type()
    example_custom_prefix_with_data_type()
    example_different_buckets()
    example_all_options()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print()

