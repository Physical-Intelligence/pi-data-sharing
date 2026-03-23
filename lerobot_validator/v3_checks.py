"""
P0 validators for LeRobot v3 datasets.

Each validator function takes a dataset path and returns a list of Issue objects.
Issues have a level ("error" or "warning") and a descriptive message.

Validators:
  V1:  validate_tasks_format       -- meta/tasks.parquet vs tasks.jsonl
  V2:  validate_codebase_version   -- info.json codebase_version starts with "v3."
  V5:  validate_feature_shapes     -- reject shape=[], image features need 3-element shape
  V7:  validate_timestamps         -- reject absolute Unix epoch timestamps in data parquet
  V11: validate_custom_metadata_csv -- required columns, no null/duplicate episode_ids
  V12: validate_start_timestamp    -- start_timestamp must be plausible Unix epoch floats
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from cloudpathlib import AnyPath, CloudPath

logger = logging.getLogger(__name__)

# Timestamps at or above this value are treated as absolute Unix epoch (year 2000+).
_UNIX_EPOCH_THRESHOLD = 946_684_800.0

# Upper bound for plausible Unix epoch timestamps (year 2100).
_UNIX_EPOCH_MAX = 4_102_444_800.0

# Required columns in custom_metadata.csv (minimum set for rejection).
_REQUIRED_METADATA_COLUMNS = ["episode_index", "episode_id"]


@dataclass
class Issue:
    """A single validation finding."""

    level: str  # "error" or "warning"
    validator: str  # e.g. "validate_tasks_format"
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.validator}: {self.message}"


# ---------------------------------------------------------------------------
# V1: validate_tasks_format
# ---------------------------------------------------------------------------


def validate_tasks_format(dataset_path: Union[str, Path, CloudPath]) -> List[Issue]:
    """Check that meta/tasks.parquet exists; warn if only tasks.jsonl is present.

    - Error: neither tasks.parquet nor tasks.jsonl exists.
    - Warning: tasks.jsonl exists but tasks.parquet does not (old format).
    - Pass: tasks.parquet exists.
    """
    root = _to_path(dataset_path)
    meta = root / "meta"
    issues: List[Issue] = []

    has_parquet = (meta / "tasks.parquet").exists()
    has_jsonl = (meta / "tasks.jsonl").exists()

    if not has_parquet and not has_jsonl:
        issues.append(
            Issue(
                level="error",
                validator="validate_tasks_format",
                message=(
                    "meta/tasks.parquet not found. "
                    "LeRobot v3 datasets must include a tasks.parquet file."
                ),
            )
        )
    elif has_jsonl and not has_parquet:
        issues.append(
            Issue(
                level="warning",
                validator="validate_tasks_format",
                message=(
                    "meta/tasks.parquet not found but meta/tasks.jsonl is present. "
                    "The converter will auto-convert, but you should migrate to "
                    "tasks.parquet before uploading."
                ),
            )
        )

    return issues


# ---------------------------------------------------------------------------
# V2: validate_codebase_version
# ---------------------------------------------------------------------------


def validate_codebase_version(dataset_path: Union[str, Path, CloudPath]) -> List[Issue]:
    """Check that info.json contains codebase_version starting with 'v3.'.

    - Error: codebase_version is missing or does not start with 'v3.'.
    """
    root = _to_path(dataset_path)
    issues: List[Issue] = []
    info = _load_info(root)

    if info is None:
        issues.append(
            Issue(
                level="error",
                validator="validate_codebase_version",
                message="meta/info.json not found or not valid JSON.",
            )
        )
        return issues

    version = info.get("codebase_version")
    if version is None:
        issues.append(
            Issue(
                level="error",
                validator="validate_codebase_version",
                message="meta/info.json is missing 'codebase_version' field.",
            )
        )
    elif not str(version).startswith("v3."):
        issues.append(
            Issue(
                level="error",
                validator="validate_codebase_version",
                message=(
                    f"codebase_version is '{version}' but must start with 'v3.'. "
                    "Only LeRobot v3 datasets are supported."
                ),
            )
        )

    return issues


# ---------------------------------------------------------------------------
# V5: validate_feature_shapes
# ---------------------------------------------------------------------------


def validate_feature_shapes(dataset_path: Union[str, Path, CloudPath]) -> List[Issue]:
    """Check feature shapes in info.json.

    - Error: a feature has shape=[] (zero-dimensional).
    - Error: an image/video feature does not have a 3-element shape.
    """
    root = _to_path(dataset_path)
    issues: List[Issue] = []
    info = _load_info(root)

    if info is None:
        return issues

    features = info.get("features", {})
    if not isinstance(features, dict):
        return issues

    for name, defn in features.items():
        if not isinstance(defn, dict):
            continue

        shape = defn.get("shape")
        dtype = defn.get("dtype", "")

        # Reject 0-D shapes
        if isinstance(shape, list) and len(shape) == 0:
            issues.append(
                Issue(
                    level="error",
                    validator="validate_feature_shapes",
                    message=(
                        f"Feature '{name}' has an empty shape (shape: []). "
                        "Scalar features should use shape: [1]."
                    ),
                )
            )
            continue

        # Image/video features must have exactly 3 dimensions (H, W, C) or (C, H, W)
        if dtype in ("video", "image") and isinstance(shape, list) and len(shape) != 3:
            issues.append(
                Issue(
                    level="error",
                    validator="validate_feature_shapes",
                    message=(
                        f"Feature '{name}' (dtype='{dtype}') has shape {shape} "
                        f"but image/video features must have a 3-element shape "
                        f"(e.g. [H, W, C])."
                    ),
                )
            )

    return issues


# ---------------------------------------------------------------------------
# V7: validate_timestamps
# ---------------------------------------------------------------------------


def validate_timestamps(dataset_path: Union[str, Path, CloudPath]) -> List[Issue]:
    """Check that data parquet timestamps are relative, not absolute Unix epoch.

    Reads the first data parquet file and checks the first timestamp value.

    - Error: timestamps are absolute Unix epoch (>= 946684800.0).
    - Warning: timestamps are not monotonically increasing within an episode.
    - Warning: non-zero starting offset within an episode (> 1 second).
    """
    root = _to_path(dataset_path)
    issues: List[Issue] = []

    data_dir = root / "data"
    if not data_dir.exists():
        return issues

    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        return issues

    # Read the first data parquet file
    pf = parquet_files[0]
    try:
        df = pd.read_parquet(str(pf), columns=["timestamp", "episode_index"])
    except Exception:
        try:
            df = pd.read_parquet(str(pf), columns=["timestamp"])
        except Exception:
            return issues

    if df.empty or "timestamp" not in df.columns:
        return issues

    # Check for absolute Unix epoch timestamps
    first_ts = float(df["timestamp"].iloc[0])
    if first_ts >= _UNIX_EPOCH_THRESHOLD:
        issues.append(
            Issue(
                level="error",
                validator="validate_timestamps",
                message=(
                    f"Timestamps appear to be absolute Unix epoch values "
                    f"(first value: {first_ts}). LeRobot v3 requires "
                    f"per-episode-relative timestamps starting near 0.0. "
                    f"Absolute timestamps cause video decode failures."
                ),
            )
        )
        return issues  # No point checking monotonicity if timestamps are wrong type

    # Check per-episode properties if episode_index is available
    if "episode_index" in df.columns:
        for ep_idx, ep_df in df.groupby("episode_index"):
            ts = ep_df["timestamp"].values

            # Warn if starting offset > 1 second
            if len(ts) > 0 and ts[0] > 1.0:
                issues.append(
                    Issue(
                        level="warning",
                        validator="validate_timestamps",
                        message=(
                            f"Episode {ep_idx} starts at timestamp {ts[0]:.3f}s "
                            f"(expected near 0.0)."
                        ),
                    )
                )

            # Warn if not monotonically increasing
            if len(ts) > 1:
                diffs = ts[1:] - ts[:-1]
                if (diffs < 0).any():
                    issues.append(
                        Issue(
                            level="warning",
                            validator="validate_timestamps",
                            message=(
                                f"Episode {ep_idx} has non-monotonically "
                                f"increasing timestamps."
                            ),
                        )
                    )

    return issues


# ---------------------------------------------------------------------------
# V11: validate_custom_metadata_csv
# ---------------------------------------------------------------------------


def validate_custom_metadata_csv(dataset_path: Union[str, Path, CloudPath]) -> List[Issue]:
    """Check that meta/custom_metadata.csv exists and has required columns.

    - Error: file missing.
    - Error: required columns (episode_index, episode_id) absent.
    - Error: null episode_id values.
    - Error: duplicate episode_id values.
    - Warning: other expected columns missing.
    """
    root = _to_path(dataset_path)
    issues: List[Issue] = []

    csv_path = root / "meta" / "custom_metadata.csv"
    if not csv_path.exists():
        issues.append(
            Issue(
                level="error",
                validator="validate_custom_metadata_csv",
                message="meta/custom_metadata.csv not found.",
            )
        )
        return issues

    try:
        df = pd.read_csv(str(csv_path))
    except Exception as exc:
        issues.append(
            Issue(
                level="error",
                validator="validate_custom_metadata_csv",
                message=f"Failed to read meta/custom_metadata.csv: {exc}",
            )
        )
        return issues

    # Check required columns
    missing_required = [c for c in _REQUIRED_METADATA_COLUMNS if c not in df.columns]
    if missing_required:
        issues.append(
            Issue(
                level="error",
                validator="validate_custom_metadata_csv",
                message=(
                    f"meta/custom_metadata.csv is missing required columns: "
                    f"{missing_required}"
                ),
            )
        )
        return issues  # Cannot do further checks without required columns

    # Check for null episode_id values
    null_ids = df[df["episode_id"].isna()]
    if len(null_ids) > 0:
        issues.append(
            Issue(
                level="error",
                validator="validate_custom_metadata_csv",
                message=(
                    f"episode_id has null values at rows: "
                    f"{null_ids.index.tolist()}"
                ),
            )
        )

    # Check for duplicate episode_id values
    duplicates = df[df["episode_id"].duplicated(keep=False)]
    if len(duplicates) > 0:
        dup_ids = duplicates["episode_id"].unique().tolist()
        issues.append(
            Issue(
                level="error",
                validator="validate_custom_metadata_csv",
                message=(
                    f"episode_id has duplicate values: {dup_ids}"
                ),
            )
        )

    # Warn about other expected columns that are missing
    all_expected = [
        "episode_index",
        "operator_id",
        "is_eval_episode",
        "episode_id",
        "start_timestamp",
        "checkpoint_path",
        "success",
        "station_id",
        "robot_id",
    ]
    missing_optional = [
        c for c in all_expected if c not in df.columns and c not in _REQUIRED_METADATA_COLUMNS
    ]
    if missing_optional:
        issues.append(
            Issue(
                level="warning",
                validator="validate_custom_metadata_csv",
                message=(
                    f"meta/custom_metadata.csv is missing optional columns: "
                    f"{missing_optional}"
                ),
            )
        )

    return issues


# ---------------------------------------------------------------------------
# V12: validate_start_timestamp
# ---------------------------------------------------------------------------


def validate_start_timestamp(dataset_path: Union[str, Path, CloudPath]) -> List[Issue]:
    """Check that start_timestamp values are plausible Unix epoch floats.

    - Error: value is not a valid float.
    - Error: value is below year-2000 threshold (likely relative, not absolute).
    - Error: value is above year-2100 threshold.
    - Error: value is null/missing.
    """
    root = _to_path(dataset_path)
    issues: List[Issue] = []

    csv_path = root / "meta" / "custom_metadata.csv"
    if not csv_path.exists():
        return issues  # V11 already reports this

    try:
        df = pd.read_csv(str(csv_path))
    except Exception:
        return issues  # V11 already reports this

    if "start_timestamp" not in df.columns:
        return issues  # V11 warns about missing columns

    invalid: List[str] = []
    for idx, row in df.iterrows():
        ts = row.get("start_timestamp")
        episode_id = row.get("episode_id", f"row_{idx}")

        if pd.isna(ts):
            invalid.append(
                f"  Row {idx} (episode '{episode_id}'): "
                f"start_timestamp is missing/null"
            )
            continue

        try:
            ts_float = float(ts)
        except (ValueError, TypeError):
            invalid.append(
                f"  Row {idx} (episode '{episode_id}'): "
                f"'{ts}' is not a valid float"
            )
            continue

        if ts_float < _UNIX_EPOCH_THRESHOLD:
            invalid.append(
                f"  Row {idx} (episode '{episode_id}'): "
                f"{ts_float} is below year-2000 threshold ({_UNIX_EPOCH_THRESHOLD}); "
                f"likely a relative offset, not an absolute Unix timestamp"
            )
        elif ts_float > _UNIX_EPOCH_MAX:
            invalid.append(
                f"  Row {idx} (episode '{episode_id}'): "
                f"{ts_float} is above year-2100 threshold ({_UNIX_EPOCH_MAX})"
            )

    if invalid:
        issues.append(
            Issue(
                level="error",
                validator="validate_start_timestamp",
                message=(
                    "start_timestamp must be a valid Unix epoch float "
                    f"(range {_UNIX_EPOCH_THRESHOLD} to {_UNIX_EPOCH_MAX}):\n"
                    + "\n".join(invalid)
                ),
            )
        )

    return issues


# ---------------------------------------------------------------------------
# Convenience: run all P0 validators
# ---------------------------------------------------------------------------

_P0_VALIDATORS = [
    validate_tasks_format,
    validate_codebase_version,
    validate_feature_shapes,
    validate_timestamps,
    validate_custom_metadata_csv,
    validate_start_timestamp,
]


def validate_v3_dataset(
    dataset_path: Union[str, Path, CloudPath],
    thorough: bool = False,
) -> List[Issue]:
    """Run all P0 validators and return a combined list of issues.

    Args:
        dataset_path: Path to the lerobot dataset directory.
        thorough: Reserved for future P2 checks that require video probing.

    Returns:
        A list of Issue objects (errors and warnings).
    """
    all_issues: List[Issue] = []
    for validator_fn in _P0_VALIDATORS:
        try:
            all_issues.extend(validator_fn(dataset_path))
        except Exception as exc:
            logger.warning("Validator %s raised: %s", validator_fn.__name__, exc)
            all_issues.append(
                Issue(
                    level="error",
                    validator=validator_fn.__name__,
                    message=f"Validator raised an unexpected exception: {exc}",
                )
            )
    return all_issues


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_path(dataset_path: Union[str, Path, CloudPath]) -> Any:
    """Convert a string or Path to an AnyPath."""
    if isinstance(dataset_path, str):
        return AnyPath(dataset_path)
    return dataset_path


def _load_info(root: Any) -> Optional[Dict[str, Any]]:
    """Load meta/info.json and return the parsed dict, or None on failure."""
    info_file = root / "meta" / "info.json"
    if not info_file.exists():
        return None
    try:
        with info_file.open("r") as f:
            return json.load(f)
    except Exception:
        return None
