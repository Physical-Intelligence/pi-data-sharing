"""Integration tests for the full validator."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from lerobot_validator.validator import LerobotDatasetValidator


def create_test_dataset(tmpdir):
    """Create a minimal test dataset structure."""
    dataset_path = Path(tmpdir) / "dataset"
    dataset_path.mkdir()
    
    # Create meta folder
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir()

    # Create info.json with task and fps
    info = {
        "task": "pick_and_place",
        "fps": 30,
        "episodes": {
            "ep_001": {"duration": 10.0, "num_frames": 300},
            "ep_002": {"duration": 5.0, "num_frames": 150},
        },
    }
    with open(dataset_path / "info.json", "w") as f:
        json.dump(info, f)

    return dataset_path


def test_full_validation_success():
    """Test full validation with all components valid."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset
        dataset_path = create_test_dataset(tmpdir)

        # Create valid metadata in meta folder with new schema
        metadata_path = dataset_path / "meta" / "custom_metadata.csv"
        df = pd.DataFrame(
            {
                "episode_index": [0, 1],
                "operator_id": ["op1", "op1"],
                "is_eval_episode": [True, False],
                "episode_id": ["ep_001", "ep_002"],
                "start_timestamp": [1730455200, 1730458800],  # UTC seconds
                "checkpoint_path": [
                    "gs://bucket/policies/policy_v1.pth",
                    "",  # No checkpoint for non-eval
                ],
                "success": [True, False],
                "station_id": ["station_1", "station_1"],
                "robot_id": ["robot_1", "robot_1"],
            }
        )
        df.to_csv(metadata_path, index=False)

        # Create valid annotation in meta folder with new schema
        annotation_path = dataset_path / "meta" / "custom_annotation.json"
        annotations = {
            "episodes": [
                {
                    "episode_id": "ep_001",
                    "spans": [
                        {"start_time": 1.0, "end_time": 2.5, "label": "human_intervention"},
                        {"start_time": 5.0, "end_time": 7.0, "label": "human_intervention"},
                        {"start_time": 0.0, "end_time": 3.0, "label": "grasp"},
                    ],
                    "extras": {},
                },
                {
                    "episode_id": "ep_002",
                    "spans": [],  # No intervention for non-eval
                    "extras": {},
                },
            ]
        }
        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

        # Validate
        validator = LerobotDatasetValidator(dataset_path)
        assert validator.validate() is True
        assert len(validator.get_errors()) == 0


def test_intervention_non_eval_episode():
    """Test that intervention on non-eval episode fails validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset
        dataset_path = create_test_dataset(tmpdir)

        # Create metadata in meta folder
        metadata_path = dataset_path / "meta" / "custom_metadata.csv"
        df = pd.DataFrame(
            {
                "episode_index": [0],
                "operator_id": ["op1"],
                "is_eval_episode": [False],  # NOT an eval episode
                "episode_id": ["ep_001"],
                "start_timestamp": [1730455200],
                "checkpoint_path": [""],
                "success": [True],
                "station_id": ["station_1"],
                "robot_id": ["robot_1"],
            }
        )
        df.to_csv(metadata_path, index=False)

        # Create annotation in meta folder with intervention for non-eval episode
        annotation_path = dataset_path / "meta" / "custom_annotation.json"
        annotations = {
            "episodes": [
                {
                    "episode_id": "ep_001",
                    "spans": [
                        {"start_time": 1.0, "end_time": 2.5, "label": "human_intervention"}
                    ],
                    "extras": {},
                }
            ]
        }
        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

        # Validate
        validator = LerobotDatasetValidator(dataset_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("is_eval_episode=False" in err for err in errors)


def test_intervention_exceeds_boundaries():
    """Test that intervention times exceeding episode duration fail validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset
        dataset_path = create_test_dataset(tmpdir)

        # Create metadata in meta folder
        metadata_path = dataset_path / "meta" / "custom_metadata.csv"
        df = pd.DataFrame(
            {
                "episode_index": [0],
                "operator_id": ["op1"],
                "is_eval_episode": [True],
                "episode_id": ["ep_001"],
                "start_timestamp": [1730455200],
                "checkpoint_path": ["gs://bucket/policies/policy_v1.pth"],
                "success": [True],
                "station_id": ["station_1"],
                "robot_id": ["robot_1"],
            }
        )
        df.to_csv(metadata_path, index=False)

        # Create annotation in meta folder with intervention exceeding episode duration (10s)
        annotation_path = dataset_path / "meta" / "custom_annotation.json"
        annotations = {
            "episodes": [
                {
                    "episode_id": "ep_001",
                    "spans": [
                        {"start_time": 1.0, "end_time": 15.0, "label": "human_intervention"}  # Exceeds 10s
                    ],
                    "extras": {},
                }
            ]
        }
        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

        # Validate
        validator = LerobotDatasetValidator(dataset_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("exceeds episode duration" in err for err in errors)
