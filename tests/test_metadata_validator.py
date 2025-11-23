"""Tests for metadata CSV validator."""

import tempfile
from pathlib import Path

import pandas as pd

from lerobot_validator.metadata_validator import MetadataValidator
from lerobot_validator.schemas import REQUIRED_METADATA_COLUMNS


def test_valid_metadata():
    """Test validation with valid metadata CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "custom_metadata.csv"

        # Create valid metadata with new schema
        df = pd.DataFrame(
            {
                "episode_index": [0, 1, 2],
                "operator_id": ["op1", "op1", "op2"],
                "is_eval_episode": [True, False, True],
                "episode_id": ["ep_001", "ep_002", "ep_003"],
                "start_timestamp": [1730455200, 1730458800, 1730462400],  # UTC seconds
                "checkpoint_path": [
                    "gs://my-bucket/policies/policy_v1.pth",
                    "",  # Non-eval episode shouldn't have checkpoint
                    "gs://my-bucket/policies/policy_v2.pth",
                ],
                "success": [True, False, True],
                "station_id": ["station_1", "station_1", "station_2"],
                "robot_id": ["robot_alpha", "robot_alpha", "robot_beta"],
            }
        )
        df.to_csv(metadata_path, index=False)

        # Validate
        validator = MetadataValidator(metadata_path)
        assert validator.validate() is True
        assert len(validator.get_errors()) == 0


def test_missing_columns():
    """Test validation with missing required columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "custom_metadata.csv"

        # Create metadata missing some columns
        df = pd.DataFrame(
            {
                "episode_index": [0, 1],
                "episode_id": ["ep_001", "ep_002"],
                "success": [True, False],
            }
        )
        df.to_csv(metadata_path, index=False)

        # Validate
        validator = MetadataValidator(metadata_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert len(errors) > 0
        assert any("Missing required columns" in err for err in errors)


def test_unexpected_columns():
    """Test validation with unexpected columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "custom_metadata.csv"

        # Create metadata with all required columns plus extra ones
        data = {col: [0, 1] for col in REQUIRED_METADATA_COLUMNS}
        data["unexpected_column"] = ["val1", "val2"]
        df = pd.DataFrame(data)
        df.to_csv(metadata_path, index=False)

        # Validate
        validator = MetadataValidator(metadata_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("Unexpected columns" in err for err in errors)


def test_duplicate_episode_ids():
    """Test validation with duplicate episode IDs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "custom_metadata.csv"

        # Create metadata with duplicate episode_id
        df = pd.DataFrame(
            {
                "episode_index": [0, 1, 2],
                "operator_id": ["op1", "op1", "op2"],
                "is_eval_episode": [True, False, True],
                "episode_id": ["ep_001", "ep_001", "ep_003"],  # Duplicate
                "start_timestamp": [1730455200, 1730458800, 1730462400],
                "checkpoint_path": [
                    "gs://bucket/path/policy_v1.pth",
                    "",
                    "gs://bucket/path/policy_v2.pth",
                ],
                "success": [True, False, True],
                "station_id": ["station_1", "station_1", "station_2"],
                "robot_id": ["robot_1", "robot_1", "robot_2"],
            }
        )
        df.to_csv(metadata_path, index=False)

        # Validate
        validator = MetadataValidator(metadata_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("duplicate" in err.lower() for err in errors)


def test_file_not_found():
    """Test validation with non-existent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "nonexistent.csv"
        validator = MetadataValidator(metadata_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("not found" in err for err in errors)


def test_checkpoint_path_on_non_eval():
    """Test that checkpoint_path on non-eval episode fails validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "custom_metadata.csv"

        df = pd.DataFrame(
            {
                "episode_index": [0],
                "operator_id": ["op1"],
                "is_eval_episode": [False],  # Non-eval
                "episode_id": ["ep_001"],
                "start_timestamp": [1730455200],
                "checkpoint_path": ["gs://bucket/path/policy.pth"],  # Should not be set
                "success": [True],
                "station_id": ["station_1"],
                "robot_id": ["robot_1"],
            }
        )
        df.to_csv(metadata_path, index=False)

        validator = MetadataValidator(metadata_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("should not be specified for non-eval episodes" in err for err in errors)


def test_invalid_gcs_uri_missing_gs_prefix():
    """Test validation with invalid GCS URI (missing gs:// prefix)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "custom_metadata.csv"

        df = pd.DataFrame(
            {
                "episode_index": [0],
                "operator_id": ["op1"],
                "is_eval_episode": [True],
                "episode_id": ["ep_001"],
                "start_timestamp": [1730455200],
                "checkpoint_path": ["s3://bucket/path/policy.pth"],  # Wrong prefix
                "success": [True],
                "station_id": ["station_1"],
                "robot_id": ["robot_1"],
            }
        )
        df.to_csv(metadata_path, index=False)

        validator = MetadataValidator(metadata_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("must start with 'gs://'" in err for err in errors)


def test_invalid_gcs_uri_missing_bucket():
    """Test validation with invalid GCS URI (missing bucket name)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "custom_metadata.csv"

        df = pd.DataFrame(
            {
                "episode_index": [0],
                "operator_id": ["op1"],
                "is_eval_episode": [True],
                "episode_id": ["ep_001"],
                "start_timestamp": [1730455200],
                "checkpoint_path": ["gs://"],  # Missing bucket
                "success": [True],
                "station_id": ["station_1"],
                "robot_id": ["robot_1"],
            }
        )
        df.to_csv(metadata_path, index=False)

        validator = MetadataValidator(metadata_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("missing bucket name" in err for err in errors)


def test_invalid_gcs_uri_missing_path():
    """Test validation with invalid GCS URI (missing path after bucket)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "custom_metadata.csv"

        df = pd.DataFrame(
            {
                "episode_index": [0],
                "operator_id": ["op1"],
                "is_eval_episode": [True],
                "episode_id": ["ep_001"],
                "start_timestamp": [1730455200],
                "checkpoint_path": ["gs://my-bucket"],  # Missing path
                "success": [True],
                "station_id": ["station_1"],
                "robot_id": ["robot_1"],
            }
        )
        df.to_csv(metadata_path, index=False)

        validator = MetadataValidator(metadata_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("must include path after bucket" in err for err in errors)


def test_valid_gcs_uri_formats():
    """Test validation with various valid GCS URI formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "custom_metadata.csv"

        df = pd.DataFrame(
            {
                "episode_index": [0, 1, 2],
                "operator_id": ["op1", "op1", "op2"],
                "is_eval_episode": [True, False, True],
                "episode_id": ["ep_001", "ep_002", "ep_003"],
                "start_timestamp": [1730455200, 1730458800, 1730462400],
                "checkpoint_path": [
                    "gs://bucket/path/to/policy.pth",
                    "",  # Empty for non-eval
                    "gs://another-bucket/policy_v1.pth",
                ],
                "success": [True, False, True],
                "station_id": ["station_1", "station_1", "station_2"],
                "robot_id": ["robot_1", "robot_1", "robot_2"],
            }
        )
        df.to_csv(metadata_path, index=False)

        validator = MetadataValidator(metadata_path)
        assert validator.validate() is True
        assert len(validator.get_errors()) == 0
