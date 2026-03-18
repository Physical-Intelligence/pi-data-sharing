"""Tests for is_eval_data consistency check."""

import json
import tempfile
from pathlib import Path

import pandas as pd

from lerobot_validator.validator import LerobotDatasetValidator


def create_test_dataset(tmpdir):
    """Create a minimal test dataset structure."""
    dataset_path = Path(tmpdir) / "dataset"
    dataset_path.mkdir()

    # Create meta folder
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir()

    # Create data folder with a valid data chunk parquet
    data_dir = dataset_path / "data"
    chunk_dir = data_dir / "chunk-000"
    chunk_dir.mkdir(parents=True)
    pd.DataFrame({
        "episode_index": [0, 0, 1, 1],
        "timestamp": [0.0, 0.033, 0.0, 0.033],
    }).to_parquet(chunk_dir / "episode_000000.parquet", index=False)

    # Create info.json in meta folder
    info = {
        "fps": 30,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "action": {"dtype": "float32", "shape": [7]},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f)

    # Create tasks.parquet
    pd.DataFrame({"task_index": [0], "task": ["default"]}).to_parquet(
        meta_dir / "tasks.parquet", index=False
    )

    # Create episodes.parquet with required v3 columns
    pd.DataFrame({
        "episode_index": [0, 1],
        "data/chunk_index": [0, 0],
        "data/file_index": [0, 1],
        "tasks": [["default"], ["default"]],
    }).to_parquet(meta_dir / "episodes.parquet", index=False)

    return dataset_path


def test_is_eval_data_consistency_pass():
    """Test that validation passes when is_eval_episode matches is_eval_data flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = create_test_dataset(tmpdir)

        # Create metadata with all is_eval_episode=True
        metadata_path = dataset_path / "meta" / "custom_metadata.csv"
        df = pd.DataFrame(
            {
                "episode_index": [0, 1],
                "operator_id": ["op1", "op1"],
                "is_eval_episode": [True, True],  # All True
                "episode_id": ["ep_001", "ep_002"],
                "start_timestamp": [1730455200, 1730458800],  # UTC seconds
                "checkpoint_path": [
                    "gs://bucket/policies/policy_v1.pth",
                    "gs://bucket/policies/policy_v1.pth",
                ],
                "success": [True, False],
                "station_id": ["station_1", "station_1"],
                "robot_id": ["robot_1", "robot_1"],
            }
        )
        df.to_csv(metadata_path, index=False)

        # Create valid annotation with new schema
        annotation_path = dataset_path / "meta" / "custom_annotation.json"
        annotations = {
            "episodes": [
                {"episode_id": "ep_001", "spans": [], "extras": {}},
                {"episode_id": "ep_002", "spans": [], "extras": {}},
            ]
        }
        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

        # Validate with is_eval_data=True (should match)
        validator = LerobotDatasetValidator(
            dataset_path, is_eval_data=True
        )
        assert validator.validate() is True
        assert len(validator.get_errors()) == 0


def test_is_eval_data_consistency_fail():
    """Test that validation fails when is_eval_episode doesn't match is_eval_data flag."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = create_test_dataset(tmpdir)

        # Create metadata with mixed is_eval_episode values
        metadata_path = dataset_path / "meta" / "custom_metadata.csv"
        df = pd.DataFrame(
            {
                "episode_index": [0, 1],
                "operator_id": ["op1", "op1"],
                "is_eval_episode": [True, False],  # Mixed values
                "episode_id": ["ep_001", "ep_002"],
                "start_timestamp": [1730455200, 1730458800],
                "checkpoint_path": [
                    "gs://bucket/policies/policy_v1.pth",
                    "",
                ],
                "success": [True, False],
                "station_id": ["station_1", "station_1"],
                "robot_id": ["robot_1", "robot_1"],
            }
        )
        df.to_csv(metadata_path, index=False)

        # Create valid annotation with new schema
        annotation_path = dataset_path / "meta" / "custom_annotation.json"
        annotations = {
            "episodes": [
                {"episode_id": "ep_001", "spans": [], "extras": {}},
                {"episode_id": "ep_002", "spans": [], "extras": {}},
            ]
        }
        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

        # Validate with is_eval_data=True (should fail - not all episodes are eval)
        validator = LerobotDatasetValidator(
            dataset_path, is_eval_data=True
        )
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("mismatched is_eval_episode values" in err for err in errors)


def test_is_eval_data_training_consistency():
    """Test training data consistency check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = create_test_dataset(tmpdir)

        # Create metadata with all is_eval_episode=False
        metadata_path = dataset_path / "meta" / "custom_metadata.csv"
        df = pd.DataFrame(
            {
                "episode_index": [0, 1],
                "operator_id": ["op1", "op1"],
                "is_eval_episode": [False, False],  # All False
                "episode_id": ["ep_001", "ep_002"],
                "start_timestamp": [1730455200, 1730458800],
                "checkpoint_path": ["", ""],  # No checkpoints for training
                "success": [True, False],
                "station_id": ["station_1", "station_1"],
                "robot_id": ["robot_1", "robot_1"],
            }
        )
        df.to_csv(metadata_path, index=False)

        # Create valid annotation with new schema (annotation file is optional, but including it here)
        annotation_path = dataset_path / "meta" / "custom_annotation.json"
        annotations = {
            "episodes": [
                {"episode_id": "ep_001", "spans": [], "extras": {}},
                {"episode_id": "ep_002", "spans": [], "extras": {}},
            ]
        }
        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

        # Validate with is_eval_data=False (should pass)
        validator = LerobotDatasetValidator(
            dataset_path, is_eval_data=False
        )
        assert validator.validate() is True


def test_is_eval_data_not_provided():
    """Test that validation works when is_eval_data is not provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = create_test_dataset(tmpdir)

        # Create metadata with mixed values
        metadata_path = dataset_path / "meta" / "custom_metadata.csv"
        df = pd.DataFrame(
            {
                "episode_index": [0, 1],
                "operator_id": ["op1", "op1"],
                "is_eval_episode": [True, False],  # Mixed - should be OK without flag
                "episode_id": ["ep_001", "ep_002"],
                "start_timestamp": [1730455200, 1730458800],
                "checkpoint_path": [
                    "gs://bucket/policies/policy_v1.pth",
                    "",
                ],
                "success": [True, False],
                "station_id": ["station_1", "station_1"],
                "robot_id": ["robot_1", "robot_1"],
            }
        )
        df.to_csv(metadata_path, index=False)

        # Create valid annotation (no intervention for non-eval) - annotation is optional
        annotation_path = dataset_path / "meta" / "custom_annotation.json"
        annotations = {
            "episodes": [
                {"episode_id": "ep_001", "spans": [], "extras": {}},
                {"episode_id": "ep_002", "spans": [], "extras": {}},
            ]
        }
        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

        # Validate without is_eval_data (should pass - no consistency check)
        validator = LerobotDatasetValidator(dataset_path)
        assert validator.validate() is True
