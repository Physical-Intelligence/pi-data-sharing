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

from lerobot_validator.schemas import REQUIRED_METADATA_COLUMNS

logger = logging.getLogger(__name__)

# Timestamps at or above this value are treated as absolute Unix epoch (year 2000+).
UNIX_EPOCH_THRESHOLD = 946_684_800.0

# Upper bound for plausible Unix epoch timestamps (year 2100).
UNIX_EPOCH_MAX = 4_102_444_800.0

# Minimum columns required for the converter to function at all.
_MIN_REQUIRED_COLUMNS = ["episode_index", "episode_id"]


@dataclass
class Issue:
    """A single validation finding."""

    level: str  # "error" or "warning"
    validator: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.validator}: {self.message}"

    @staticmethod
    def error(validator: str, message: str) -> "Issue":
        return Issue(level="error", validator=validator, message=message)

    @staticmethod
    def warning(validator: str, message: str) -> "Issue":
        return Issue(level="warning", validator=validator, message=message)


def validate_tasks_format(dataset_path: Union[str, Path, CloudPath]) -> List[Issue]:
    """Check that meta/tasks.parquet exists; warn if only tasks.jsonl is present."""
    root = _to_path(dataset_path)
    meta = root / "meta"
    issues: List[Issue] = []

    has_parquet = (meta / "tasks.parquet").exists()
    has_jsonl = (meta / "tasks.jsonl").exists()

    if not has_parquet and not has_jsonl:
        issues.append(
            Issue.error(
                "validate_tasks_format",
                "meta/tasks.parquet not found. "
                "LeRobot v3 datasets must include a tasks.parquet file.",
            )
        )
    elif has_jsonl and not has_parquet:
        issues.append(
            Issue.warning(
                "validate_tasks_format",
                "meta/tasks.parquet not found but meta/tasks.jsonl is present. "
                "The converter will auto-convert, but you should migrate to "
                "tasks.parquet before uploading.",
            )
        )

    return issues


def validate_codebase_version(dataset_path: Union[str, Path, CloudPath]) -> List[Issue]:
    """Check that info.json contains codebase_version starting with 'v3.'."""
    root = _to_path(dataset_path)
    issues: List[Issue] = []
    info = _load_info(root)

    if info is None:
        issues.append(Issue.error("validate_codebase_version", "meta/info.json not found or not valid JSON."))
        return issues

    version = info.get("codebase_version")
    if version is None:
        issues.append(Issue.error("validate_codebase_version", "meta/info.json is missing 'codebase_version' field."))
    elif not str(version).startswith("v3."):
        issues.append(
            Issue.error(
                "validate_codebase_version",
                f"codebase_version is '{version}' but must start with 'v3.'. "
                "Only LeRobot v3 datasets are supported.",
            )
        )

    return issues


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

        if isinstance(shape, list) and len(shape) == 0:
            issues.append(
                Issue.error(
                    "validate_feature_shapes",
                    f"Feature '{name}' has an empty shape (shape: []). "
                    "Scalar features should use shape: [1].",
                )
            )
            continue

        if dtype in ("video", "image") and isinstance(shape, list) and len(shape) != 3:
            issues.append(
                Issue.error(
                    "validate_feature_shapes",
                    f"Feature '{name}' (dtype='{dtype}') has shape {shape} "
                    f"but image/video features must have a 3-element shape "
                    f"(e.g. [H, W, C]).",
                )
            )

    return issues


def validate_timestamps(dataset_path: Union[str, Path, CloudPath]) -> List[Issue]:
    """Check that data parquet timestamps are relative, not absolute Unix epoch.

    Reads only the first data parquet file. Checks the first timestamp to
    determine if values are absolute, then samples the first episode for
    monotonicity and starting offset.
    """
    root = _to_path(dataset_path)
    issues: List[Issue] = []

    data_dir = root / "data"
    if not data_dir.exists():
        return issues

    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    if not parquet_files:
        return issues

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

    first_ts = float(df["timestamp"].iloc[0])
    if first_ts >= UNIX_EPOCH_THRESHOLD:
        issues.append(
            Issue.error(
                "validate_timestamps",
                f"Timestamps appear to be absolute Unix epoch values "
                f"(first value: {first_ts}). LeRobot v3 requires "
                f"per-episode-relative timestamps starting near 0.0. "
                f"Absolute timestamps cause video decode failures.",
            )
        )
        return issues

    # Check only the first episode for monotonicity/offset (avoid processing entire file).
    if "episode_index" in df.columns:
        first_ep = df["episode_index"].iloc[0]
        ep_df = df[df["episode_index"] == first_ep]
        ts = ep_df["timestamp"].values

        if len(ts) > 0 and ts[0] > 1.0:
            issues.append(
                Issue.warning(
                    "validate_timestamps",
                    f"Episode {first_ep} starts at timestamp {ts[0]:.3f}s (expected near 0.0).",
                )
            )

        if len(ts) > 1:
            diffs = ts[1:] - ts[:-1]
            if (diffs < 0).any():
                issues.append(
                    Issue.warning(
                        "validate_timestamps",
                        f"Episode {first_ep} has non-monotonically increasing timestamps.",
                    )
                )

    return issues


def validate_custom_metadata_csv(
    dataset_path: Union[str, Path, CloudPath],
    _df_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[Issue]:
    """Check that meta/custom_metadata.csv exists and has required columns.

    If _df_cache is provided, the loaded DataFrame is stored under "csv" so
    downstream validators (e.g. validate_start_timestamp) can reuse it.
    """
    root = _to_path(dataset_path)
    issues: List[Issue] = []

    csv_path = root / "meta" / "custom_metadata.csv"
    if not csv_path.exists():
        issues.append(Issue.error("validate_custom_metadata_csv", "meta/custom_metadata.csv not found."))
        return issues

    try:
        df = pd.read_csv(str(csv_path))
    except Exception as exc:
        issues.append(
            Issue.error("validate_custom_metadata_csv", f"Failed to read meta/custom_metadata.csv: {exc}")
        )
        return issues

    if _df_cache is not None:
        _df_cache["csv"] = df

    missing_required = [c for c in _MIN_REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        issues.append(
            Issue.error(
                "validate_custom_metadata_csv",
                f"meta/custom_metadata.csv is missing required columns: {missing_required}",
            )
        )
        return issues

    null_ids = df[df["episode_id"].isna()]
    if len(null_ids) > 0:
        issues.append(
            Issue.error(
                "validate_custom_metadata_csv",
                f"episode_id has null values at rows: {null_ids.index.tolist()}",
            )
        )

    duplicates = df[df["episode_id"].duplicated(keep=False)]
    if len(duplicates) > 0:
        dup_ids = duplicates["episode_id"].unique().tolist()
        issues.append(
            Issue.error("validate_custom_metadata_csv", f"episode_id has duplicate values: {dup_ids}")
        )

    # Warn about expected columns from the full schema that are missing.
    missing_optional = [c for c in REQUIRED_METADATA_COLUMNS if c not in df.columns and c not in _MIN_REQUIRED_COLUMNS]
    if missing_optional:
        issues.append(
            Issue.warning(
                "validate_custom_metadata_csv",
                f"meta/custom_metadata.csv is missing optional columns: {missing_optional}",
            )
        )

    return issues


def validate_start_timestamp(
    dataset_path: Union[str, Path, CloudPath],
    _df_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[Issue]:
    """Check that start_timestamp values are plausible Unix epoch floats.

    Reuses the DataFrame from _df_cache if available (populated by
    validate_custom_metadata_csv), avoiding a redundant CSV read.
    """
    root = _to_path(dataset_path)
    issues: List[Issue] = []

    df = _df_cache.get("csv") if _df_cache else None
    if df is None:
        csv_path = root / "meta" / "custom_metadata.csv"
        if not csv_path.exists():
            return issues
        try:
            df = pd.read_csv(str(csv_path))
        except Exception:
            return issues

    if "start_timestamp" not in df.columns:
        return issues

    invalid: List[str] = []
    for idx, row in df.iterrows():
        ts = row.get("start_timestamp")
        episode_id = row.get("episode_id", f"row_{idx}")

        if pd.isna(ts):
            invalid.append(f"  Row {idx} (episode '{episode_id}'): start_timestamp is missing/null")
            continue

        try:
            ts_float = float(ts)
        except (ValueError, TypeError):
            invalid.append(f"  Row {idx} (episode '{episode_id}'): '{ts}' is not a valid float")
            continue

        if ts_float < UNIX_EPOCH_THRESHOLD:
            invalid.append(
                f"  Row {idx} (episode '{episode_id}'): "
                f"{ts_float} is below year-2000 threshold ({UNIX_EPOCH_THRESHOLD}); "
                f"likely a relative offset, not an absolute Unix timestamp"
            )
        elif ts_float > UNIX_EPOCH_MAX:
            invalid.append(
                f"  Row {idx} (episode '{episode_id}'): "
                f"{ts_float} is above year-2100 threshold ({UNIX_EPOCH_MAX})"
            )

    if invalid:
        issues.append(
            Issue.error(
                "validate_start_timestamp",
                "start_timestamp must be a valid Unix epoch float "
                f"(range {UNIX_EPOCH_THRESHOLD} to {UNIX_EPOCH_MAX}):\n" + "\n".join(invalid),
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
) -> List[Issue]:
    """Run all P0 validators and return a combined list of issues.

    Args:
        dataset_path: Path to the lerobot dataset directory.

    Returns:
        A list of Issue objects (errors and warnings).
    """
    all_issues: List[Issue] = []
    # Shared cache so V12 reuses the CSV loaded by V11.
    df_cache: Dict[str, pd.DataFrame] = {}
    for validator_fn in _P0_VALIDATORS:
        try:
            import inspect

            sig = inspect.signature(validator_fn)
            if "_df_cache" in sig.parameters:
                all_issues.extend(validator_fn(dataset_path, _df_cache=df_cache))
            else:
                all_issues.extend(validator_fn(dataset_path))
        except Exception as exc:
            logger.warning("Validator %s raised: %s", validator_fn.__name__, exc)
            all_issues.append(
                Issue.error(validator_fn.__name__, f"Validator raised an unexpected exception: {exc}")
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
