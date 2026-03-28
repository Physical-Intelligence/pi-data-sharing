"""
Validator for LeRobot v3 dataset metadata integrity.

Checks that the dataset conforms to the LeRobot v3 specification:
  1. tasks.parquet exists (auto-converts from tasks.jsonl if possible)
  2. Episodes parquet has required columns
  3. Feature shapes in info.json are non-empty (scalars need shape [1])
  4. File path templates in info.json use standard placeholders
  5. chunks_size present in info.json
  6. Video feature keys in info.json match episodes parquet columns
  7. Video files referenced by the dataset exist
  8. Timestamp consistency across data parquet files
  9. Episode row contiguity in data parquet files
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from cloudpathlib import AnyPath, CloudPath

logger = logging.getLogger(__name__)

# Columns that must exist in every episodes parquet file.
REQUIRED_EPISODES_COLUMNS: List[str] = [
    "data/chunk_index",
    "data/file_index",
    "tasks",
]

# Standard placeholders expected in info.json path templates.
REQUIRED_DATA_PATH_PLACEHOLDERS: Set[str] = {
    "{episode_chunk",
    "{episode_index",
}
REQUIRED_VIDEO_PATH_PLACEHOLDERS: Set[str] = {
    "{episode_chunk",
    "{episode_index",
    "{video_key",
}

# Timestamps below this threshold are treated as relative (seconds from
# episode start).  Values above are treated as absolute Unix timestamps.
# No robot episode is longer than ~11.5 days (1e6 seconds).
_ABSOLUTE_TIMESTAMP_THRESHOLD = 1_000_000.0


def ensure_tasks_parquet(dataset_path: Union[str, Path]) -> None:
    """Convert meta/tasks.jsonl to meta/tasks.parquet if the parquet is missing.

    Some LeRobot v3 datasets ship with tasks.jsonl instead of tasks.parquet.
    The lerobot library expects the parquet file, so this converts on the fly.
    No-op if tasks.parquet already exists or tasks.jsonl is absent.
    """
    meta = Path(dataset_path) / "meta"
    parquet_path = meta / "tasks.parquet"
    jsonl_path = meta / "tasks.jsonl"

    if parquet_path.exists() or not jsonl_path.exists():
        return

    rows = [json.loads(line) for line in jsonl_path.read_text().strip().splitlines()]
    pd.DataFrame(rows).to_parquet(str(parquet_path), index=False)
    logger.info("Converted %s to %s (%d tasks)", jsonl_path, parquet_path, len(rows))


class LerobotV3MetadataChecker:
    """
    Validates structural metadata for LeRobot v3 datasets.

    Usage::

        checker = LerobotV3MetadataChecker(dataset_path)
        passed  = checker.validate()
        errors  = checker.get_errors()
    """

    def __init__(self, dataset_path: Union[str, Path, CloudPath]) -> None:
        if isinstance(dataset_path, str):
            self.dataset_path = AnyPath(dataset_path)
        else:
            self.dataset_path = dataset_path
        self.errors: List[str] = []
        self._info: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Run all v3 metadata checks.  Returns True when all pass."""
        self.errors = []
        self._info = None

        if not self.dataset_path.exists():
            self.errors.append(
                f"Dataset directory not found: {self.dataset_path}"
            )
            return False

        self._load_info()

        self._check_tasks_parquet()
        self._check_episodes_parquet_columns()
        self._check_feature_shapes()
        self._check_path_templates()
        self._check_chunks_size()
        self._check_video_key_consistency()
        self._check_video_files_exist()
        self._check_timestamp_consistency()
        self._check_episode_contiguity()

        return len(self.errors) == 0

    def get_errors(self) -> List[str]:
        """Return accumulated error messages."""
        return self.errors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _meta_dir(self) -> Any:
        return self.dataset_path / "meta"

    def _data_dir(self) -> Any:
        return self.dataset_path / "data"

    def _load_info(self) -> None:
        info_file = self._meta_dir() / "info.json"
        if not info_file.exists():
            return  # Already reported by LerobotDatasetChecker
        try:
            with info_file.open("r") as f:
                self._info = json.load(f)
        except json.JSONDecodeError as exc:
            self.errors.append(f"meta/info.json is not valid JSON: {exc}")
        except Exception as exc:
            self.errors.append(f"Failed to read meta/info.json: {exc}")

    def _get_video_keys(self) -> List[str]:
        """Return feature keys whose dtype is video or image."""
        if self._info is None:
            return []
        features = self._info.get("features", {})
        return [
            name
            for name, defn in features.items()
            if isinstance(defn, dict)
            and defn.get("dtype") in ("video", "image")
        ]

    # ------------------------------------------------------------------
    # Check 1: tasks.parquet
    # ------------------------------------------------------------------

    def _check_tasks_parquet(self) -> None:
        meta = self._meta_dir()
        has_parquet = (meta / "tasks.parquet").exists()
        has_jsonl = (meta / "tasks.jsonl").exists()

        if has_jsonl and not has_parquet:
            # Auto-convert tasks.jsonl -> tasks.parquet for local paths.
            if not isinstance(meta, CloudPath):
                try:
                    ensure_tasks_parquet(self.dataset_path)
                    logger.info(
                        "Auto-converted meta/tasks.jsonl to meta/tasks.parquet"
                    )
                    return
                except Exception as exc:
                    self.errors.append(
                        f"meta/tasks.parquet not found and auto-conversion "
                        f"from tasks.jsonl failed: {exc}"
                    )
                    return
            self.errors.append(
                "meta/tasks.parquet not found but meta/tasks.jsonl is present. "
                "LeRobot v3 requires tasks.parquet — convert tasks.jsonl "
                "to parquet before uploading."
            )
        elif not has_parquet and not has_jsonl:
            self.errors.append(
                "meta/tasks.parquet not found. "
                "LeRobot v3 datasets must include a tasks.parquet file."
            )

    # ------------------------------------------------------------------
    # Check 2: episodes parquet columns
    # ------------------------------------------------------------------

    def _check_episodes_parquet_columns(self) -> None:
        episodes_file = self._meta_dir() / "episodes.parquet"
        if not episodes_file.exists():
            self.errors.append(
                "meta/episodes.parquet not found. "
                "LeRobot v3 datasets must include an episodes.parquet file."
            )
            return

        try:
            df = pd.read_parquet(str(episodes_file))
        except Exception as exc:
            self.errors.append(
                f"Failed to read meta/episodes.parquet: {exc}"
            )
            return

        missing = [
            col for col in REQUIRED_EPISODES_COLUMNS if col not in df.columns
        ]
        if missing:
            self.errors.append(
                f"meta/episodes.parquet is missing required columns: {missing}. "
                f"Present columns: {sorted(df.columns.tolist())}"
            )

    # ------------------------------------------------------------------
    # Check 3: feature shapes
    # ------------------------------------------------------------------

    # Metadata columns that don't require a shape entry in info.json.
    _METADATA_FEATURE_NAMES: set = {
        "episode_index",
        "frame_index",
        "index",
        "timestamp",
        "task_index",
    }

    # Numeric dtypes that should always carry a shape declaration.
    _NUMERIC_DTYPES: set = {
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
        "bfloat16",
    }

    def _check_feature_shapes(self) -> None:
        if self._info is None:
            return
        features = self._info.get("features", {})
        if not isinstance(features, dict):
            return

        for name, defn in features.items():
            if not isinstance(defn, dict):
                continue

            dtype = defn.get("dtype", "")

            # Skip video/image features and metadata columns.
            if dtype in ("video", "image"):
                continue
            if name in self._METADATA_FEATURE_NAMES:
                continue

            shape = defn.get("shape")

            if isinstance(shape, list) and len(shape) == 0:
                self.errors.append(
                    f"Feature '{name}' has an empty shape (shape: []). "
                    "Scalar features should use shape: [1]."
                )
            elif shape is None and dtype in self._NUMERIC_DTYPES:
                self.errors.append(
                    f"Feature '{name}' is missing 'shape'. "
                    "Scalar features should use shape: [1]."
                )

    # ------------------------------------------------------------------
    # Check 4: path templates
    # ------------------------------------------------------------------

    def _check_path_templates(self) -> None:
        if self._info is None:
            return

        data_path = self._info.get("data_path")
        if data_path is not None:
            self._validate_template(
                str(data_path),
                "data_path",
                REQUIRED_DATA_PATH_PLACEHOLDERS,
            )

        video_path = self._info.get("video_path")
        if video_path is not None:
            self._validate_template(
                str(video_path),
                "video_path",
                REQUIRED_VIDEO_PATH_PLACEHOLDERS,
            )

    def _validate_template(
        self,
        template: str,
        field: str,
        required: Set[str],
    ) -> None:
        # Extract placeholder stems (everything before the first colon or })
        found_stems = set()
        for match in re.finditer(r"\{([^}:]+)", template):
            found_stems.add("{" + match.group(1))

        missing = required - found_stems
        if missing:
            self.errors.append(
                f"info.json '{field}' template is missing required "
                f"placeholders {sorted(missing)}. "
                f"Template: '{template}'"
            )

    # ------------------------------------------------------------------
    # Check 5: video files exist
    # ------------------------------------------------------------------

    def _check_video_files_exist(self) -> None:
        if self._info is None:
            return

        video_path_tpl = self._info.get("video_path")
        if video_path_tpl is None:
            return

        video_keys = self._get_video_keys()
        if not video_keys:
            return

        episodes_file = self._meta_dir() / "episodes.parquet"
        if not episodes_file.exists():
            return  # Already flagged in check 2

        try:
            episodes_df = pd.read_parquet(str(episodes_file))
        except Exception:
            return

        if "episode_index" not in episodes_df.columns:
            return

        chunks_size = self._info.get("chunks_size", 1000)
        missing_files: List[str] = []

        for _, row in episodes_df.iterrows():
            ep_idx = int(row["episode_index"])
            ep_chunk = int(
                row.get("data/chunk_index", ep_idx // chunks_size)
            )

            for vkey in video_keys:
                try:
                    rendered = video_path_tpl.format(
                        episode_chunk=ep_chunk,
                        episode_index=ep_idx,
                        video_key=vkey,
                    )
                except KeyError:
                    continue  # Template uses non-standard placeholders

                if not (self.dataset_path / rendered).exists():
                    missing_files.append(rendered)

            if len(missing_files) > 10:
                missing_files.append("... (truncated)")
                break

        if missing_files:
            self.errors.append(
                f"Missing video files ({len(missing_files)} not found):\n"
                + "\n".join(f"  {p}" for p in missing_files)
            )

    # ------------------------------------------------------------------
    # Check 6: timestamp consistency
    # ------------------------------------------------------------------

    def _check_timestamp_consistency(self) -> None:
        data_dir = self._data_dir()
        if not data_dir.exists():
            return

        parquet_files = sorted(data_dir.glob("**/*.parquet"))
        if not parquet_files:
            return

        overall_mode: Optional[bool] = None  # True=absolute, False=relative
        inconsistent: List[str] = []

        for pf in parquet_files:
            try:
                df = pd.read_parquet(str(pf), columns=["timestamp"])
            except Exception:
                continue
            if df.empty:
                continue

            has_abs = bool(
                (df["timestamp"] > _ABSOLUTE_TIMESTAMP_THRESHOLD).any()
            )
            has_rel = bool(
                (df["timestamp"] <= _ABSOLUTE_TIMESTAMP_THRESHOLD).any()
            )

            if has_abs and has_rel:
                inconsistent.append(
                    f"  {pf.name}: mix of absolute and relative timestamps"
                )
                continue

            file_mode = has_abs
            if overall_mode is None:
                overall_mode = file_mode
            elif overall_mode != file_mode:
                inconsistent.append(
                    f"  {pf.name}: "
                    f"{'absolute' if file_mode else 'relative'} timestamps "
                    f"but earlier chunks use "
                    f"{'absolute' if overall_mode else 'relative'}"
                )

        if inconsistent:
            self.errors.append(
                "Timestamp inconsistency in data parquet files. "
                "All files must use either relative (seconds from episode "
                "start) or absolute (Unix epoch) timestamps:\n"
                + "\n".join(inconsistent)
            )

    # ------------------------------------------------------------------
    # Check 7: episode contiguity
    # ------------------------------------------------------------------

    def _check_episode_contiguity(self) -> None:
        data_dir = self._data_dir()
        if not data_dir.exists():
            return

        parquet_files = sorted(data_dir.glob("**/*.parquet"))
        if not parquet_files:
            return

        non_contiguous: List[str] = []

        for pf in parquet_files:
            try:
                df = pd.read_parquet(str(pf), columns=["episode_index"])
            except Exception:
                continue
            if df.empty:
                continue

            seen: set[int] = set()
            prev: Optional[int] = None
            for ep_idx in df["episode_index"]:
                ep_idx = int(ep_idx)
                if ep_idx != prev:
                    if ep_idx in seen:
                        non_contiguous.append(
                            f"  {pf.name}: episode_index={ep_idx} "
                            f"appears non-contiguously"
                        )
                        break
                    seen.add(ep_idx)
                    prev = ep_idx

        if non_contiguous:
            self.errors.append(
                "Non-contiguous episode rows in data parquet files. "
                "All rows for each episode_index must be grouped "
                "together:\n" + "\n".join(non_contiguous)
            )

    # ------------------------------------------------------------------
    # Check 8: chunks_size in info.json
    # ------------------------------------------------------------------

    def _check_chunks_size(self) -> None:
        if self._info is None:
            return
        if "chunks_size" not in self._info:
            self.errors.append(
                "info.json is missing 'chunks_size'. "
                "LeRobot v3 datasets must specify chunks_size (typically 1000) "
                "indicating how many episodes are grouped per chunk directory."
            )

    # ------------------------------------------------------------------
    # Check 9: video feature keys match episodes parquet columns
    # ------------------------------------------------------------------

    def _check_video_key_consistency(self) -> None:
        """Verify that video feature keys in info.json match episodes parquet columns.

        The lerobot library looks up video chunk/file indices from the episodes
        parquet using ``videos/{feature_key}/chunk_index``.  If the feature key
        doesn't match the column prefix, episode loading fails at runtime.
        """
        if self._info is None:
            return

        video_keys = self._get_video_keys()
        if not video_keys:
            return

        # Load one episodes parquet to get column names.
        episodes_dir = self._meta_dir() / "episodes"
        if not episodes_dir.exists():
            return

        ep_files = sorted(episodes_dir.glob("**/*.parquet"))
        if not ep_files:
            return

        try:
            ep_df = pd.read_parquet(str(ep_files[0]))
        except Exception:
            return

        ep_columns = set(ep_df.columns)

        for vkey in video_keys:
            expected_col = f"videos/{vkey}/chunk_index"
            if expected_col not in ep_columns:
                # Check if a shorter prefix matches (common mismatch: key
                # has a trailing modality suffix like /image).
                candidates = [
                    c for c in ep_columns
                    if c.startswith("videos/") and c.endswith("/chunk_index")
                ]
                suggestion = ""
                for cand in candidates:
                    prefix = cand.removeprefix("videos/").removesuffix("/chunk_index")
                    if vkey.startswith(prefix):
                        suggestion = (
                            f" Did you mean '{prefix}'? The episodes parquet "
                            f"has '{cand}'."
                        )
                        break
                self.errors.append(
                    f"Video feature key '{vkey}' in info.json does not match "
                    f"episodes parquet columns. Expected column "
                    f"'{expected_col}' but it was not found.{suggestion}"
                )
