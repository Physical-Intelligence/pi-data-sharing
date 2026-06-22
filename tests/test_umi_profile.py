"""Tests for UMI-specific metadata and feature validation."""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from lerobot_validator.metadata_validator import MetadataValidator
from lerobot_validator.v3_checks import validate_umi_features


def test_umi_metadata_allows_unlabeled_non_robot_rows(tmp_path: Path) -> None:
    metadata_path = tmp_path / "custom_metadata.csv"
    pd.DataFrame(
        {
            "episode_index": [0],
            "operator_id": ["collector@example.com"],
            "is_eval_episode": [False],
            "episode_id": ["umi_001"],
            "start_timestamp": [1780591657.0],
            "station_id": ["kitchen_1"],
            "success": [None],
            "provider_batch": ["pilot"],
        }
    ).to_csv(metadata_path, index=False)

    validator = MetadataValidator(metadata_path, dataset_profile="umi")

    assert validator.validate() is True
    assert validator.get_errors() == []


def test_robot_metadata_profile_stays_strict(tmp_path: Path) -> None:
    metadata_path = tmp_path / "custom_metadata.csv"
    pd.DataFrame(
        {
            "episode_index": [0],
            "operator_id": ["collector@example.com"],
            "is_eval_episode": [False],
            "episode_id": ["umi_001"],
            "start_timestamp": [1780591657.0],
            "station_id": ["kitchen_1"],
        }
    ).to_csv(metadata_path, index=False)

    validator = MetadataValidator(metadata_path)

    assert validator.validate() is False
    assert any("Missing required columns" in error for error in validator.get_errors())


def test_umi_features_accept_public_contract_with_observation_prefix(
    tmp_path: Path,
) -> None:
    dataset_path = _write_info(tmp_path, _canonical_umi_features())

    assert validate_umi_features(dataset_path) == []


def test_umi_features_allow_optional_wrist_cameras_and_camera_tracking_to_be_omitted(
    tmp_path: Path,
) -> None:
    features = _canonical_umi_features()
    for name in list(features):
        if "wrist_0_camera" in name or "base_0_camera/position" in name or "base_0_camera/quaternion" in name:
            del features[name]
    dataset_path = _write_info(tmp_path, features)

    assert validate_umi_features(dataset_path) == []


def test_umi_features_report_missing_fields_and_wrong_pose_shapes(
    tmp_path: Path,
) -> None:
    features = _canonical_umi_features()
    del features["observation/right/gripper"]
    features["observation/left/position"]["shape"] = [4]
    features["observation/base_0_camera/intrinsics"]["shape"] = [4, 4]
    dataset_path = _write_info(tmp_path, features)

    issues = validate_umi_features(dataset_path)

    assert any("right/gripper" in issue.message for issue in issues)
    assert any(
        "left/position" in issue.message and "shape [3]" in issue.message
        for issue in issues
    )
    assert any("base_0_camera/intrinsics" in issue.message and "shape [3, 3]" in issue.message for issue in issues)


def test_umi_features_warn_missing_tracking_timestamps(tmp_path: Path) -> None:
    features = _canonical_umi_features()
    del features["observation/left/gripper/timestamp"]
    dataset_path = _write_info(tmp_path, features)

    issues = validate_umi_features(dataset_path)

    assert len(issues) == 1
    assert issues[0].level == "warning"
    assert "left/gripper/timestamp" in issues[0].message


def _write_info(tmp_path: Path, features: Dict[str, Dict[str, Any]]) -> Path:
    dataset_path = tmp_path / "dataset"
    meta_path = dataset_path / "meta"
    meta_path.mkdir(parents=True)
    (meta_path / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "features": features,
            }
        )
    )
    return dataset_path


def _canonical_umi_features() -> Dict[str, Dict[str, Any]]:
    image = {"dtype": "video", "shape": [480, 640, 3]}
    tracked = {
        "observation/left/position": {"dtype": "float32", "shape": [3]},
        "observation/left/quaternion_xyzw": {"dtype": "float32", "shape": [4]},
        "observation/left/gripper": {"dtype": "float32", "shape": [1]},
        "observation/right/position": {"dtype": "float32", "shape": [3]},
        "observation/right/quaternion_xyzw": {"dtype": "float32", "shape": [4]},
        "observation/right/gripper": {"dtype": "float32", "shape": [1]},
        "observation/base_0_camera/position": {"dtype": "float32", "shape": [3]},
        "observation/base_0_camera/quaternion_xyzw": {
            "dtype": "float32",
            "shape": [4],
        },
    }
    timestamps = {
        f"{name}/timestamp": {"dtype": "int64", "shape": [1]}
        for name in tracked
    }
    return {
        "observation/base_0_camera/rgb/image": dict(image),
        "observation/left_wrist_0_camera/rgb/image": dict(image),
        "observation/right_wrist_0_camera/rgb/image": dict(image),
        "observation/base_0_camera/intrinsics": {
            "dtype": "float32",
            "shape": [3, 3],
        },
        **tracked,
        **timestamps,
    }
