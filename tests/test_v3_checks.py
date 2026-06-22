"""Tests for P0 v3 validators (lerobot_validator.v3_checks)."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest import mock

import pandas as pd

from lerobot_validator.v3_checks import (
    Issue,
    validate_codebase_version,
    validate_custom_metadata_csv,
    validate_feature_dtypes,
    validate_feature_shapes,
    validate_start_timestamp,
    validate_tasks_format,
    validate_timestamps,
    validate_v3_dataset,
    validate_video_frame_count,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(tmpdir: str) -> Path:
    """Create a minimal dataset directory skeleton."""
    root = Path(tmpdir) / "dataset"
    root.mkdir()
    (root / "meta").mkdir()
    (root / "data").mkdir()
    return root


def _write_info(root: Path, info: Dict[str, Any]) -> None:
    with open(root / "meta" / "info.json", "w") as f:
        json.dump(info, f)


def _minimal_info(**overrides: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "fps": 30,
        "codebase_version": "v3.0",
        "chunks_size": 1000,
        "features": {
            "observation.images.top": {
                "dtype": "video",
                "shape": [480, 640, 3],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
            },
        },
    }
    info.update(overrides)
    return info


def _write_tasks_parquet(root: Path) -> None:
    pd.DataFrame({"task_index": [0], "task": ["default"]}).to_parquet(
        root / "meta" / "tasks.parquet", index=False
    )


def _write_custom_metadata(root: Path, df: pd.DataFrame) -> None:
    df.to_csv(root / "meta" / "custom_metadata.csv", index=False)


def _valid_metadata_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_index": [0, 1],
            "operator_id": ["op1", "op1"],
            "is_eval_episode": [False, False],
            "episode_id": ["ep_001", "ep_002"],
            "start_timestamp": [1730455200.0, 1730458800.0],
            "checkpoint_path": ["", ""],
            "success": [True, False],
            "station_id": ["station_1", "station_1"],
            "robot_id": ["robot_1", "robot_1"],
        }
    )


def _errors(issues: List[Issue]) -> List[Issue]:
    return [i for i in issues if i.level == "error"]


def _warnings(issues: List[Issue]) -> List[Issue]:
    return [i for i in issues if i.level == "warning"]


# ===================================================================
# V1: validate_tasks_format
# ===================================================================


class TestValidateTasksFormat:
    def test_parquet_present_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())
            _write_tasks_parquet(root)

            issues = validate_tasks_format(root)
            assert len(_errors(issues)) == 0

    def test_neither_file_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())

            issues = validate_tasks_format(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "tasks.parquet not found" in errors[0].message

    def test_jsonl_only_warns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())
            (root / "meta" / "tasks.jsonl").write_text(
                '{"task_index": 0, "task": "pick"}\n'
            )

            issues = validate_tasks_format(root)
            assert len(_errors(issues)) == 0
            warnings = _warnings(issues)
            assert len(warnings) == 1
            assert "tasks.jsonl" in warnings[0].message

    def test_both_files_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())
            _write_tasks_parquet(root)
            (root / "meta" / "tasks.jsonl").write_text(
                '{"task_index": 0, "task": "pick"}\n'
            )

            issues = validate_tasks_format(root)
            assert len(issues) == 0


# ===================================================================
# V2: validate_codebase_version
# ===================================================================


class TestValidateCodebaseVersion:
    def test_v3_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info(codebase_version="v3.0"))

            issues = validate_codebase_version(root)
            assert len(issues) == 0

    def test_v3_minor_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info(codebase_version="v3.1.2"))

            issues = validate_codebase_version(root)
            assert len(issues) == 0

    def test_v2_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info(codebase_version="v2.1"))

            issues = validate_codebase_version(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "v3." in errors[0].message

    def test_missing_version_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            info = _minimal_info()
            del info["codebase_version"]
            _write_info(root, info)

            issues = validate_codebase_version(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "missing" in errors[0].message.lower()

    def test_no_info_json_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            # no info.json at all

            issues = validate_codebase_version(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "info.json" in errors[0].message


# ===================================================================
# V5: validate_feature_shapes
# ===================================================================


class TestValidateFeatureShapes:
    def test_valid_shapes_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())

            issues = validate_feature_shapes(root)
            assert len(issues) == 0

    def test_empty_shape_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            info = _minimal_info()
            info["features"]["action"]["shape"] = []
            _write_info(root, info)

            issues = validate_feature_shapes(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "empty shape" in errors[0].message
            assert "action" in errors[0].message

    def test_scalar_shape_1_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            info = _minimal_info()
            info["features"]["scalar_feat"] = {"dtype": "float32", "shape": [1]}
            _write_info(root, info)

            issues = validate_feature_shapes(root)
            assert len(issues) == 0

    def test_image_feature_2d_shape_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            info = _minimal_info()
            info["features"]["observation.images.top"]["shape"] = [640, 480]
            _write_info(root, info)

            issues = validate_feature_shapes(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "3-element shape" in errors[0].message

    def test_video_feature_4d_shape_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            info = _minimal_info()
            info["features"]["observation.images.top"]["shape"] = [1, 480, 640, 3]
            _write_info(root, info)

            issues = validate_feature_shapes(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "3-element shape" in errors[0].message

    def test_image_dtype_3d_shape_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            info = _minimal_info()
            info["features"]["cam"] = {"dtype": "image", "shape": [480, 640, 3]}
            _write_info(root, info)

            issues = validate_feature_shapes(root)
            assert len(issues) == 0

    def test_no_info_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            # no info.json

            issues = validate_feature_shapes(root)
            assert len(issues) == 0


# ===================================================================
# V7: validate_timestamps
# ===================================================================


class TestValidateTimestamps:
    def test_relative_timestamps_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())
            chunk_dir = root / "data" / "chunk-000"
            chunk_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "episode_index": [0, 0, 0],
                    "timestamp": [0.0, 0.033, 0.066],
                }
            ).to_parquet(chunk_dir / "episode_000000.parquet", index=False)

            issues = validate_timestamps(root)
            assert len(issues) == 0

    def test_absolute_timestamps_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())
            chunk_dir = root / "data" / "chunk-000"
            chunk_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "episode_index": [0, 0],
                    "timestamp": [1_700_000_000.0, 1_700_000_000.033],
                }
            ).to_parquet(chunk_dir / "episode_000000.parquet", index=False)

            issues = validate_timestamps(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "absolute Unix epoch" in errors[0].message

    def test_non_monotonic_warns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())
            chunk_dir = root / "data" / "chunk-000"
            chunk_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "episode_index": [0, 0, 0],
                    "timestamp": [0.0, 0.066, 0.033],  # non-monotonic
                }
            ).to_parquet(chunk_dir / "episode_000000.parquet", index=False)

            issues = validate_timestamps(root)
            warnings = _warnings(issues)
            assert len(warnings) >= 1
            assert any("non-monotonically" in w.message for w in warnings)

    def test_large_starting_offset_warns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())
            chunk_dir = root / "data" / "chunk-000"
            chunk_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "episode_index": [0, 0],
                    "timestamp": [5.0, 5.033],  # starts at 5s, not near 0
                }
            ).to_parquet(chunk_dir / "episode_000000.parquet", index=False)

            issues = validate_timestamps(root)
            warnings = _warnings(issues)
            assert len(warnings) >= 1
            assert any("starts at timestamp" in w.message for w in warnings)

    def test_no_data_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            # data dir is empty (no parquet files)

            issues = validate_timestamps(root)
            assert len(issues) == 0

    def test_no_episode_index_column_still_checks_absolute(self):
        """Even without episode_index column, absolute timestamps should be caught."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())
            chunk_dir = root / "data" / "chunk-000"
            chunk_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": [1_700_000_000.0, 1_700_000_000.033],
                }
            ).to_parquet(chunk_dir / "episode_000000.parquet", index=False)

            issues = validate_timestamps(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "absolute Unix epoch" in errors[0].message


# ===================================================================
# V11: validate_custom_metadata_csv
# ===================================================================


class TestValidateCustomMetadataCsv:
    def test_valid_metadata_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_custom_metadata(root, _valid_metadata_df())

            issues = validate_custom_metadata_csv(root)
            assert len(_errors(issues)) == 0

    def test_missing_file_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)

            issues = validate_custom_metadata_csv(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "not found" in errors[0].message

    def test_missing_episode_index_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            df = pd.DataFrame(
                {
                    "episode_id": ["ep_001", "ep_002"],
                    "operator_id": ["op1", "op1"],
                }
            )
            _write_custom_metadata(root, df)

            issues = validate_custom_metadata_csv(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "episode_index" in errors[0].message

    def test_missing_episode_id_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            df = pd.DataFrame(
                {
                    "episode_index": [0, 1],
                    "operator_id": ["op1", "op1"],
                }
            )
            _write_custom_metadata(root, df)

            issues = validate_custom_metadata_csv(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "episode_id" in errors[0].message

    def test_null_episode_id_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            df = _valid_metadata_df()
            df.loc[0, "episode_id"] = None
            _write_custom_metadata(root, df)

            issues = validate_custom_metadata_csv(root)
            errors = _errors(issues)
            assert any("null" in e.message for e in errors)

    def test_duplicate_episode_id_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            df = _valid_metadata_df()
            df.loc[1, "episode_id"] = "ep_001"  # duplicate
            _write_custom_metadata(root, df)

            issues = validate_custom_metadata_csv(root)
            errors = _errors(issues)
            assert any("duplicate" in e.message for e in errors)

    def test_missing_optional_columns_warns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            df = pd.DataFrame(
                {
                    "episode_index": [0, 1],
                    "episode_id": ["ep_001", "ep_002"],
                }
            )
            _write_custom_metadata(root, df)

            issues = validate_custom_metadata_csv(root)
            warnings = _warnings(issues)
            assert len(warnings) >= 1
            assert any("optional columns" in w.message for w in warnings)

    def test_all_columns_present_no_warnings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_custom_metadata(root, _valid_metadata_df())

            issues = validate_custom_metadata_csv(root)
            warnings = _warnings(issues)
            assert len(warnings) == 0


# ===================================================================
# V12: validate_start_timestamp
# ===================================================================


class TestValidateStartTimestamp:
    def test_valid_timestamps_pass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_custom_metadata(root, _valid_metadata_df())

            issues = validate_start_timestamp(root)
            assert len(issues) == 0

    def test_null_timestamp_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            df = _valid_metadata_df()
            df.loc[0, "start_timestamp"] = None
            _write_custom_metadata(root, df)

            issues = validate_start_timestamp(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "missing/null" in errors[0].message

    def test_below_threshold_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            df = _valid_metadata_df()
            df.loc[0, "start_timestamp"] = 100.0  # relative offset, not epoch
            _write_custom_metadata(root, df)

            issues = validate_start_timestamp(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "below year-2000 threshold" in errors[0].message

    def test_above_max_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            df = _valid_metadata_df()
            df.loc[0, "start_timestamp"] = 5_000_000_000.0  # year ~2128
            _write_custom_metadata(root, df)

            issues = validate_start_timestamp(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "above year-2100 threshold" in errors[0].message

    def test_non_numeric_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            df = _valid_metadata_df()
            df["start_timestamp"] = df["start_timestamp"].astype(str)
            df.loc[0, "start_timestamp"] = "not-a-number"
            _write_custom_metadata(root, df)

            issues = validate_start_timestamp(root)
            errors = _errors(issues)
            assert len(errors) == 1
            assert "not a valid float" in errors[0].message

    def test_missing_csv_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            # no custom_metadata.csv

            issues = validate_start_timestamp(root)
            assert len(issues) == 0

    def test_missing_column_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            df = pd.DataFrame(
                {
                    "episode_index": [0],
                    "episode_id": ["ep_001"],
                }
            )
            _write_custom_metadata(root, df)

            issues = validate_start_timestamp(root)
            assert len(issues) == 0


# ===================================================================
# validate_v3_dataset (combined runner)
# ===================================================================


class TestValidateV3Dataset:
    def test_fully_valid_dataset_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())
            _write_tasks_parquet(root)
            _write_custom_metadata(root, _valid_metadata_df())

            # Write relative timestamps
            chunk_dir = root / "data" / "chunk-000"
            chunk_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "episode_index": [0, 0],
                    "timestamp": [0.0, 0.033],
                }
            ).to_parquet(chunk_dir / "episode_000000.parquet", index=False)

            issues = validate_v3_dataset(root)
            errors = _errors(issues)
            assert len(errors) == 0

    def test_multiple_issues_collected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            # No info.json -> V2 error
            # No tasks.parquet -> V1 error
            # No custom_metadata.csv -> V11 error

            issues = validate_v3_dataset(root)
            errors = _errors(issues)
            # Should have at least V1 + V2 + V11 errors
            assert len(errors) >= 3

    def test_issue_str_representation(self):
        issue = Issue(level="error", validator="test_validator", message="test message")
        assert str(issue) == "[error] test_validator: test message"


# ===================================================================
# V13: validate_video_frame_count
# ===================================================================


class TestValidateVideoFrameCount:
    def test_no_info_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            issues = validate_video_frame_count(root)
            assert len(issues) == 0

    def test_no_video_features_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            info = _minimal_info()
            # Replace video feature with a non-video one.
            info["features"] = {"action": {"dtype": "float32", "shape": [7]}}
            _write_info(root, info)

            issues = validate_video_frame_count(root)
            assert len(issues) == 0

    def test_no_data_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())
            # Remove data dir.
            (root / "data").rmdir()

            issues = validate_video_frame_count(root)
            assert len(issues) == 0

    def test_no_parquet_files_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())
            # data/ exists but is empty.
            issues = validate_video_frame_count(root)
            assert len(issues) == 0


def _setup_video_dataset(
    root: Path,
    *,
    num_episodes: int = 3,
    frames_per_episode: int = 100,
    video_keys: Optional[List[str]] = None,
) -> None:
    """Create a dataset skeleton with parquet data and placeholder video files."""
    if video_keys is None:
        video_keys = ["observation.images.top"]

    features: Dict[str, Any] = {
        vk: {"dtype": "video", "shape": [480, 640, 3]} for vk in video_keys
    }
    features["action"] = {"dtype": "float32", "shape": [7]}

    info = _minimal_info(
        features=features,
        video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:06d}.mp4",
    )
    _write_info(root, info)

    rows = []
    for ep in range(num_episodes):
        rows.extend([{"episode_index": ep}] * frames_per_episode)
    df = pd.DataFrame(rows)
    chunk_dir = root / "data" / "chunk-000"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(chunk_dir / "train-00000.parquet"), index=False)

    episodes_rows = []
    for ep in range(num_episodes):
        row: Dict[str, Any] = {"episode_index": ep}
        for vk in video_keys:
            row[f"videos/{vk}/chunk_index"] = 0
            row[f"videos/{vk}/file_index"] = ep
        episodes_rows.append(row)
    episodes_df = pd.DataFrame(episodes_rows)
    (root / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(str(root / "meta" / "episodes" / "part-0.parquet"), index=False)

    for vk in video_keys:
        for ep in range(num_episodes):
            vdir = root / "videos" / vk / "chunk-000"
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / f"file-{ep:06d}.mp4").write_bytes(b"fake")


def _make_subprocess_side_effect(
    *,
    truncated_videos: Optional[Set[str]] = None,
    failed_probes: Optional[Set[str]] = None,
    actual_frame_count: int = 100,
    container_duration: float = 60.0,
) -> Any:
    """Return a side_effect for subprocess.run that simulates ffprobe/ffmpeg."""
    truncated_videos = truncated_videos or set()
    failed_probes = failed_probes or set()

    def side_effect(cmd: List[str], **kwargs: Any) -> "subprocess.CompletedProcess[str]":
        exe = cmd[0]
        input_file = None
        for i, arg in enumerate(cmd):
            if arg == "-i" and i + 1 < len(cmd):
                input_file = cmd[i + 1]
                break
        if input_file is None:
            input_file = cmd[-1]

        video_name = Path(input_file).name if input_file else ""

        if exe == "ffprobe" and "-count_frames" in cmd:
            if video_name in failed_probes:
                return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="error")
            return subprocess.CompletedProcess(cmd, returncode=0, stdout=str(actual_frame_count), stderr="")

        if exe == "ffprobe" and any("format=duration" in a for a in cmd):
            return subprocess.CompletedProcess(
                cmd, returncode=0, stdout=str(container_duration), stderr=""
            )

        if exe == "ffmpeg":
            if video_name in truncated_videos:
                return subprocess.CompletedProcess(
                    cmd, returncode=1, stdout="", stderr="[h264] no frame!"
                )
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    return side_effect


@mock.patch("lerobot_validator.v3_checks.subprocess.run")
def test_truncated_video_detected(mock_run: mock.MagicMock) -> None:
    mock_run.side_effect = _make_subprocess_side_effect(
        truncated_videos={"file-000001.mp4"},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _setup_video_dataset(root, num_episodes=3)

        issues = validate_video_frame_count(root)

        errors = _errors(issues)
        assert any("Truncated video" in e.message for e in errors)


@mock.patch("lerobot_validator.v3_checks.subprocess.run")
def test_probe_failure_produces_warning(mock_run: mock.MagicMock) -> None:
    mock_run.side_effect = _make_subprocess_side_effect(
        failed_probes={"file-000000.mp4"},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _setup_video_dataset(root, num_episodes=3)

        issues = validate_video_frame_count(root)

        warnings = _warnings(issues)
        assert any("could not probe frame count" in w.message for w in warnings)


@mock.patch("lerobot_validator.v3_checks.subprocess.run")
def test_all_video_keys_checked(mock_run: mock.MagicMock) -> None:
    probed_files: List[str] = []
    original = _make_subprocess_side_effect()

    def tracking_side_effect(cmd: List[str], **kwargs: Any) -> "subprocess.CompletedProcess[str]":
        if cmd[0] == "ffprobe" and "-count_frames" in cmd:
            probed_files.append(cmd[-1])
        return original(cmd, **kwargs)

    mock_run.side_effect = tracking_side_effect

    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _setup_video_dataset(
            root,
            num_episodes=1,
            video_keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
        )

        validate_video_frame_count(root)

    assert len(probed_files) == 3
    probed_keys = {Path(p).parent.parent.name for p in probed_files}
    assert probed_keys == {"cam_high", "cam_left_wrist", "cam_right_wrist"}


@mock.patch("lerobot_validator.v3_checks.subprocess.run")
def test_excessive_frame_drop_errors(mock_run: mock.MagicMock) -> None:
    mock_run.side_effect = _make_subprocess_side_effect(actual_frame_count=10)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _setup_video_dataset(root, num_episodes=2, frames_per_episode=100)

        issues = validate_video_frame_count(root)

        errors = _errors(issues)
        assert any("excessive dropped frames" in e.message for e in errors)


# ===================================================================
# V14: validate_feature_dtypes
# ===================================================================


class TestValidateFeatureDtypes:
    def test_no_string_features_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            _write_info(root, _minimal_info())

            issues = validate_feature_dtypes(root)
            assert len(issues) == 0

    def test_string_feature_warns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            info = _minimal_info()
            info["features"]["instruction.text"] = {"dtype": "string", "shape": [1]}
            _write_info(root, info)

            issues = validate_feature_dtypes(root)
            warnings = _warnings(issues)
            assert len(warnings) == 1
            assert "instruction.text" in warnings[0].message
            assert "string" in warnings[0].message.lower()

    def test_multiple_string_features_single_warning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            info = _minimal_info()
            info["features"]["instruction.text"] = {"dtype": "string", "shape": [1]}
            info["features"]["observation.meta.tool"] = {"dtype": "string", "shape": [1]}
            _write_info(root, info)

            issues = validate_feature_dtypes(root)
            warnings = _warnings(issues)
            assert len(warnings) == 1
            assert "instruction.text" in warnings[0].message
            assert "observation.meta.tool" in warnings[0].message

    def test_no_info_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _make_dataset(tmpdir)
            issues = validate_feature_dtypes(root)
            assert len(issues) == 0
