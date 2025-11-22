"""Tests for GCP path computation."""

from pathlib import Path

import pytest

from lerobot_validator.gcp_path import compute_gcp_path, format_upload_instructions


def test_compute_gcp_path_basic():
    """Test basic GCP path computation."""
    path = compute_gcp_path(
        dataset_name="test_dataset",
        bucket_name="test-bucket",
        data_type="teleop",
        version="v1.0",
    )

    assert path.startswith("gs://test-bucket/")
    assert "test-dataset" in path
    assert "v1.0" in path
    assert "teleop" in path
    assert path.endswith("/")


def test_compute_gcp_path_with_custom_bucket():
    """Test GCP path computation with custom bucket."""
    path = compute_gcp_path(
        dataset_name="dataset",
        bucket_name="custom-bucket",
        data_type="eval",
        version="v1",
    )

    assert path.startswith("gs://custom-bucket/")
    assert "eval" in path


def test_compute_gcp_path_with_custom_prefix_only():
    """Test GCP path computation with just custom prefix."""
    path = compute_gcp_path(
        dataset_name="dataset",
        bucket_name="test-bucket",
        data_type="teleop",
        version="v1",
        custom_folder_prefix="my/prefix",
    )

    assert "my/prefix" in path
    assert "gs://test-bucket/my/prefix/dataset/v1/teleop/" == path


def test_compute_gcp_path_sanitization():
    """Test that dataset names are sanitized."""
    path = compute_gcp_path(
        dataset_name="Dataset_With_Underscores",
        bucket_name="test-bucket",
        data_type="teleop",
        version="v1.0",
    )

    # Spaces and underscores should be replaced with hyphens
    assert "dataset-with-underscores" in path


def test_compute_gcp_path_auto_version():
    """Test that automatic version is generated when not provided."""
    path = compute_gcp_path(
        dataset_name="dataset",
        bucket_name="test-bucket",
        data_type="teleop",
    )

    # Should contain a timestamp-based version
    assert "gs://test-bucket/" in path
    assert "dataset" in path
    assert "teleop" in path


def test_format_upload_instructions():
    """Test upload instructions formatting."""
    gcp_path = "gs://bucketdataset/v1/"
    local_path = Path("/path/to/dataset")

    instructions = format_upload_instructions(gcp_path, local_path)

    assert gcp_path in instructions
    assert str(local_path) in instructions
    assert "gsutil" in instructions
    assert "-m cp -r" in instructions


def test_compute_gcp_path_with_custom_prefix():
    """Test GCP path computation with custom folder prefix."""
    path = compute_gcp_path(
        dataset_name="dataset",
        bucket_name="test-bucket",
        data_type="eval",
        version="v1",
        custom_folder_prefix="my/custom/folder",
    )

    assert "my/custom/folder" in path
    assert "eval" in path
    assert path.startswith("gs://test-bucket/my/custom/folder/")


def test_compute_gcp_path_with_data_type():
    """Test GCP path computation with data type."""
    teleop_path = compute_gcp_path(
        dataset_name="dataset",
        bucket_name="test-bucket",
        version="v1",
        data_type="teleop",
    )

    eval_path = compute_gcp_path(
        dataset_name="dataset",
        bucket_name="test-bucket",
        version="v1",
        data_type="eval",
    )

    assert teleop_path.endswith("/teleop/")
    assert eval_path.endswith("/eval/")
    assert "teleop" in teleop_path
    assert "eval" in eval_path


def test_compute_gcp_path_with_custom_prefix_and_data_type():
    """Test GCP path with both custom prefix and data type."""
    path = compute_gcp_path(
        dataset_name="dataset",
        bucket_name="test-bucket",
        data_type="teleop",
        version="v1",
        custom_folder_prefix="custom/path",
    )

    assert "custom/path" in path
    assert path.endswith("/teleop/")
    # Format: gs://bucket/custom/path/dataset/version/teleop/
    assert "gs://test-bucket/custom/path/dataset/v1/teleop/" == path


def test_compute_gcp_path_invalid_data_type():
    """Test that invalid data type raises error."""
    with pytest.raises(ValueError, match="data_type must be 'teleop' or 'eval'"):
        compute_gcp_path(
            dataset_name="dataset",
            bucket_name="test-bucket",
            data_type="invalid",
            version="v1",
        )


def test_compute_gcp_path_teleop_vs_eval():
    """Test that teleop and eval data types produce different paths."""
    teleop_path = compute_gcp_path(
        dataset_name="dataset",
        bucket_name="test-bucket",
        data_type="teleop",
        version="v1",
    )

    eval_path = compute_gcp_path(
        dataset_name="dataset",
        bucket_name="test-bucket",
        data_type="eval",
        version="v1",
    )

    assert teleop_path.endswith("/teleop/")
    assert eval_path.endswith("/eval/")
    assert teleop_path != eval_path
    # Base path should be the same
    assert teleop_path.replace("/teleop/", "/") == eval_path.replace("/eval/", "/")

