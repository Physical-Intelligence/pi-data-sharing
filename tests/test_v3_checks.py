"""Tests for P0 v3 validators (lerobot_validator.v3_checks)."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from lerobot_validator.v3_checks import (
    Issue,
    validate_codebase_version,
    validate_custom_metadata_csv,
    validate_feature_shapes,
    validate_start_timestamp,
    validate_tasks_format,
    validate_timestamps,
    validate_v3_dataset,
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
