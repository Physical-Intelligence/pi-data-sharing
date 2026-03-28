"""Tests for LeRobot v3 metadata checker."""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from lerobot_validator.v3_metadata_checker import LerobotV3MetadataChecker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(tmpdir: str) -> Path:
    root = Path(tmpdir) / "dataset"
    root.mkdir()
    (root / "meta").mkdir()
    (root / "data").mkdir()
    return root


def _write_info(root: Path, info: Dict[str, Any]) -> None:
    with open(root / "meta" / "info.json", "w") as f:
        json.dump(info, f)


def _minimal_info() -> Dict[str, Any]:
    return {
        "fps": 30,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
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


def _write_tasks_parquet(root: Path) -> None:
    pd.DataFrame({"task_index": [0], "task": ["default"]}).to_parquet(
        root / "meta" / "tasks.parquet", index=False
    )


def _write_episodes_parquet(root: Path) -> None:
    pd.DataFrame(
        {
            "episode_index": [0, 1],
            "data/chunk_index": [0, 0],
            "data/file_index": [0, 1],
            "tasks": [["default"], ["default"]],
        }
    ).to_parquet(root / "meta" / "episodes.parquet", index=False)


# ---------------------------------------------------------------------------
# Check 1: tasks.parquet
# ---------------------------------------------------------------------------


def test_tasks_parquet_present_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        _write_tasks_parquet(root)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert not any("tasks.parquet" in e for e in checker.get_errors())


def test_tasks_jsonl_only_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        (root / "meta" / "tasks.jsonl").write_text(
            '{"task_index": 0, "task": "pick"}\n'
        )

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any("tasks.jsonl" in e for e in checker.get_errors())


def test_no_tasks_file_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any("tasks.parquet" in e for e in checker.get_errors())


# ---------------------------------------------------------------------------
# Check 2: episodes parquet columns
# ---------------------------------------------------------------------------


def test_episodes_parquet_missing_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any("episodes.parquet" in e for e in checker.get_errors())


def test_episodes_parquet_missing_column_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        pd.DataFrame(
            {
                "episode_index": [0],
                "data/chunk_index": [0],
                "data/file_index": [0],
                # missing "tasks" column
            }
        ).to_parquet(root / "meta" / "episodes.parquet", index=False)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any(
            "tasks" in e and "missing" in e.lower()
            for e in checker.get_errors()
        )


def test_episodes_parquet_all_columns_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        _write_tasks_parquet(root)
        _write_episodes_parquet(root)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert not any("episodes.parquet" in e for e in checker.get_errors())


# ---------------------------------------------------------------------------
# Check 3: feature shapes
# ---------------------------------------------------------------------------


def test_empty_feature_shape_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        info = _minimal_info()
        info["features"]["action"]["shape"] = []
        _write_info(root, info)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any("empty shape" in e for e in checker.get_errors())


def test_valid_scalar_shape_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        info = _minimal_info()
        info["features"]["action"]["shape"] = [1]
        _write_info(root, info)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert not any("empty shape" in e for e in checker.get_errors())


def test_missing_shape_key_on_numeric_feature_fails():
    """A numeric feature with no 'shape' key at all should be flagged."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        info = _minimal_info()
        # Add a feature with dtype but no shape key
        info["features"]["observation.state"] = {"dtype": "int64"}
        _write_info(root, info)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        errors = checker.get_errors()
        assert any(
            "observation.state" in e and "missing 'shape'" in e
            for e in errors
        ), f"Expected missing shape error for observation.state, got: {errors}"


def test_missing_shape_key_on_metadata_column_passes():
    """Metadata columns (episode_index, timestamp, etc.) don't need shape."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        info = _minimal_info()
        info["features"]["episode_index"] = {"dtype": "int64"}
        info["features"]["timestamp"] = {"dtype": "float64"}
        info["features"]["frame_index"] = {"dtype": "int64"}
        info["features"]["index"] = {"dtype": "int64"}
        info["features"]["task_index"] = {"dtype": "int64"}
        _write_info(root, info)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        errors = checker.get_errors()
        assert not any(
            "missing 'shape'" in e for e in errors
        ), f"Metadata columns should not require shape, got: {errors}"


def test_missing_shape_key_on_video_feature_passes():
    """Video features don't need shape checked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        info = _minimal_info()
        # Remove shape from the video feature
        info["features"]["observation.images.top"] = {"dtype": "video"}
        _write_info(root, info)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        errors = checker.get_errors()
        assert not any(
            "observation.images.top" in e and "missing 'shape'" in e
            for e in errors
        ), f"Video features should not require shape, got: {errors}"


# ---------------------------------------------------------------------------
# Check 4: path templates
# ---------------------------------------------------------------------------


def test_data_path_missing_placeholder_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        info = _minimal_info()
        # missing {episode_chunk}
        info["data_path"] = "data/chunk-000/episode_{episode_index:06d}.parquet"
        _write_info(root, info)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any(
            "episode_chunk" in e and "data_path" in e
            for e in checker.get_errors()
        )


def test_video_path_missing_video_key_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        info = _minimal_info()
        info["video_path"] = (
            "videos/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
        )
        _write_info(root, info)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any(
            "video_key" in e and "video_path" in e
            for e in checker.get_errors()
        )


def test_valid_path_templates_pass():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert not any("template" in e for e in checker.get_errors())


# ---------------------------------------------------------------------------
# Check 5: video files exist
# ---------------------------------------------------------------------------


def test_missing_video_files_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        _write_tasks_parquet(root)
        _write_episodes_parquet(root)
        # no actual video files

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any("Missing video files" in e for e in checker.get_errors())


def test_present_video_files_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        _write_tasks_parquet(root)
        _write_episodes_parquet(root)

        for ep_idx in [0, 1]:
            vdir = (
                root
                / "videos"
                / "observation.images.top"
                / "chunk-000"
            )
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / f"episode_{ep_idx:06d}.mp4").write_bytes(b"\x00")

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert not any(
            "Missing video files" in e for e in checker.get_errors()
        )


# ---------------------------------------------------------------------------
# Check 6: timestamp consistency
# ---------------------------------------------------------------------------


def test_relative_timestamps_consistent_passes():
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

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert not any(
            "Timestamp inconsistency" in e for e in checker.get_errors()
        )


def test_mixed_timestamp_modes_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        chunk_dir = root / "data" / "chunk-000"
        chunk_dir.mkdir(parents=True)
        # relative
        pd.DataFrame(
            {
                "episode_index": [0, 0],
                "timestamp": [0.0, 0.033],
            }
        ).to_parquet(chunk_dir / "episode_000000.parquet", index=False)
        # absolute (Unix epoch)
        pd.DataFrame(
            {
                "episode_index": [1, 1],
                "timestamp": [1_700_000_000.0, 1_700_000_000.033],
            }
        ).to_parquet(chunk_dir / "episode_000001.parquet", index=False)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any(
            "Timestamp inconsistency" in e for e in checker.get_errors()
        )


# ---------------------------------------------------------------------------
# Check 7: episode contiguity
# ---------------------------------------------------------------------------


def test_contiguous_episodes_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        chunk_dir = root / "data" / "chunk-000"
        chunk_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "episode_index": [0, 0, 0, 1, 1, 1],
                "timestamp": [0.0, 0.033, 0.066, 0.0, 0.033, 0.066],
            }
        ).to_parquet(chunk_dir / "episode_000000.parquet", index=False)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert not any(
            "Non-contiguous" in e for e in checker.get_errors()
        )


def test_non_contiguous_episodes_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        chunk_dir = root / "data" / "chunk-000"
        chunk_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "episode_index": [0, 1, 0],  # non-contiguous
                "timestamp": [0.0, 0.0, 0.033],
            }
        ).to_parquet(chunk_dir / "episode_000000.parquet", index=False)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any(
            "Non-contiguous" in e for e in checker.get_errors()
        )


# ---------------------------------------------------------------------------
# Check 8: no video struct columns in data parquet
# ---------------------------------------------------------------------------


def test_data_parquet_without_video_columns_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        chunk_dir = root / "data" / "chunk-000"
        chunk_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "action": [[0.1] * 7],
                "episode_index": [0],
                "timestamp": [0.0],
            }
        ).to_parquet(chunk_dir / "episode_000000.parquet", index=False)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert not any(
            "video feature columns" in e for e in checker.get_errors()
        )


def test_data_parquet_with_video_struct_columns_fails():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        chunk_dir = root / "data" / "chunk-000"
        chunk_dir.mkdir(parents=True)
        # Simulate a data parquet that erroneously includes the video
        # feature key as a struct column (path + timestamp).
        pd.DataFrame(
            {
                "action": [[0.1] * 7],
                "observation.images.top": [
                    {"path": "videos/top/chunk-000/ep_000000.mp4", "timestamp": 0.0}
                ],
                "episode_index": [0],
                "timestamp": [0.0],
            }
        ).to_parquet(chunk_dir / "episode_000000.parquet", index=False)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any(
            "video feature columns" in e for e in checker.get_errors()
        )


# ---------------------------------------------------------------------------
# Check 9: episode video metadata columns
# ---------------------------------------------------------------------------


def test_episode_parquet_with_video_metadata_passes():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        pd.DataFrame(
            {
                "episode_index": [0, 1],
                "data/chunk_index": [0, 0],
                "data/file_index": [0, 1],
                "tasks": [["default"], ["default"]],
                "videos/observation.images.top/chunk_index": [0, 0],
                "videos/observation.images.top/from_timestamp": [0.0, 0.0],
            }
        ).to_parquet(root / "meta" / "episodes.parquet", index=False)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert not any(
            "video metadata columns" in e for e in checker.get_errors()
        )


def test_episode_parquet_missing_video_metadata_fails():
    """Episode parquet without videos/{key}/chunk_index and from_timestamp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        _write_episodes_parquet(root)  # Has data/* cols but no videos/* cols

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any(
            "video metadata columns" in e for e in checker.get_errors()
        )


def test_episode_parquet_chunked_dir_missing_video_metadata_fails():
    """Same check works for chunked episodes/ directory layout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = _make_dataset(tmpdir)
        _write_info(root, _minimal_info())
        ep_dir = root / "meta" / "episodes" / "chunk-000"
        ep_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "episode_index": [0],
                "data/chunk_index": [0],
                "data/file_index": [0],
                "tasks": [["default"]],
            }
        ).to_parquet(ep_dir / "file-000.parquet", index=False)

        checker = LerobotV3MetadataChecker(root)
        checker.validate()
        assert any(
            "video metadata columns" in e for e in checker.get_errors()
        )
