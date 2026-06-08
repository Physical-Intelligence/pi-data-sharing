"""Integration tests for the full validator."""

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

    # Create info.json in meta folder (lerobot stores it there)
    info = {
        "fps": 30,
        "codebase_version": "v3.0",
        "data_path": "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet",
        "features": {
            "action": {"dtype": "float32", "shape": [7]},
        },
        "episodes": {
            "ep_001": {"duration": 10.0, "num_frames": 300},
            "ep_002": {"duration": 5.0, "num_frames": 150},
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


def test_full_umi_validation_success() -> None:
    """Test full validation for an unlabeled UMI dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = create_test_dataset(tmpdir)
        meta_dir = dataset_path / "meta"

        with open(meta_dir / "info.json") as f:
            info = json.load(f)
        image = {"dtype": "video", "shape": [480, 640, 3]}
        image_features = {
            "observation/base_0_camera/rgb/image": dict(image),
        }
        tracking_features = {
            "observation/left/position": {"dtype": "float32", "shape": [3]},
            "observation/left/quaternion_xyzw": {"dtype": "float32", "shape": [4]},
            "observation/left/gripper": {"dtype": "float32", "shape": [1]},
            "observation/right/position": {"dtype": "float32", "shape": [3]},
            "observation/right/quaternion_xyzw": {"dtype": "float32", "shape": [4]},
            "observation/right/gripper": {"dtype": "float32", "shape": [1]},
        }
        timestamp_features = {
            f"{name}/timestamp": {"dtype": "int64", "shape": [1]}
            for name in tracking_features
        }
        info["features"] = image_features | tracking_features | timestamp_features
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f)

        episodes = pd.read_parquet(meta_dir / "episodes.parquet")
        for feature_name in image_features:
            episodes[f"videos/{feature_name}/chunk_index"] = [0, 0]
            episodes[f"videos/{feature_name}/from_timestamp"] = [0.0, 0.0]
        episodes.to_parquet(meta_dir / "episodes.parquet", index=False)

        pd.DataFrame(
            {
                "episode_index": [0, 1],
                "operator_id": ["op1", "op1"],
                "is_eval_episode": [False, False],
                "episode_id": ["umi_001", "umi_002"],
                "start_timestamp": [1730455200, 1730458800],
                "station_id": ["station_1", "station_1"],
                "success": [None, None],
                "provider_batch": ["pilot", "pilot"],
            }
        ).to_csv(meta_dir / "custom_metadata.csv", index=False)

        validator = LerobotDatasetValidator(
            dataset_path,
            is_eval_data=False,
            dataset_profile="umi",
        )

        assert validator.validate() is True
        assert validator.get_errors() == []


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
