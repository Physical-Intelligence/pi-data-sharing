"""
Validator for LeRobot v3 dataset metadata integrity.

Checks that the dataset conforms to the LeRobot v3 specification:
  1. tasks.parquet exists (flags if only tasks.jsonl is present)
  2. Episodes parquet has required columns
  3. Feature shapes in info.json are non-empty
  4. File path templates in info.json use standard placeholders
  5. Video files referenced by the dataset exist
  6. Timestamp consistency across data parquet files
  7. Episode row contiguity in data parquet files
  8. Data parquet files must not contain video struct columns
  9. Episodes parquet must include per-video-key metadata columns
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
    "{chunk_index",
    "{file_index",
}
REQUIRED_VIDEO_PATH_PLACEHOLDERS: Set[str] = {
    "{chunk_index",
    "{file_index",
    "{video_key",
}

# Timestamps below this threshold are treated as relative (seconds from
# episode start).  Values above are treated as absolute Unix timestamps.
# No robot episode is longer than ~11.5 days (1e6 seconds).
_ABSOLUTE_TIMESTAMP_THRESHOLD = 1_000_000.0


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
        self._episodes_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Run all v3 metadata checks.  Returns True when all pass."""
        self.errors = []
        self._info = None
        self._episodes_df = None

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
        self._check_video_files_exist()
        self._check_timestamp_consistency()
        self._check_episode_contiguity()
        self._check_no_video_columns_in_data_parquet()
        self._check_episode_video_metadata_columns()

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

    def _data_parquet_files(self) -> List[Any]:
        data_dir = self._data_dir()
        if not data_dir.exists():
            return []
        return sorted(data_dir.glob("**/*.parquet"))

    def _load_episodes_df(self) -> Optional[pd.DataFrame]:
        if self._episodes_df is not None:
            return self._episodes_df

        episodes_dir = self._meta_dir() / "episodes"
        if episodes_dir.exists():
            try:
                self._episodes_df = pd.read_parquet(str(episodes_dir))
                return self._episodes_df
            except Exception as exc:
                logger.warning("Failed to read %s: %s", episodes_dir, exc)
                return None

        episodes_file = self._meta_dir() / "episodes.parquet"
        if episodes_file.exists():
            try:
                self._episodes_df = pd.read_parquet(str(episodes_file))
                return self._episodes_df
            except Exception as exc:
                logger.warning("Failed to read %s: %s", episodes_file, exc)
                return None

        return None

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
        df = self._load_episodes_df()
        if df is None:
            self.errors.append(
                "Episodes parquet not found (checked meta/episodes.parquet "
                "and meta/episodes/ directory). "
                "LeRobot v3 datasets must include an episodes parquet."
            )
            return

        missing = [
            col for col in REQUIRED_EPISODES_COLUMNS if col not in df.columns
        ]
        if missing:
            self.errors.append(
                f"Episodes parquet is missing required columns: {missing}. "
                f"Present columns: {sorted(df.columns.tolist())}"
            )

    # ------------------------------------------------------------------
    # Check 3: feature shapes
    # ------------------------------------------------------------------

    def _check_feature_shapes(self) -> None:
        if self._info is None:
            return
        features = self._info.get("features", {})
        if not isinstance(features, dict):
            return

        for name, defn in features.items():
            if not isinstance(defn, dict):
                continue
            shape = defn.get("shape")
            if isinstance(shape, list) and len(shape) == 0:
                self.errors.append(
                    f"Feature '{name}' has an empty shape (shape: []). "
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

        episodes_df = self._load_episodes_df()
        if episodes_df is None:
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
                        chunk_index=ep_chunk,
                        file_index=ep_idx,
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
        parquet_files = self._data_parquet_files()
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
        parquet_files = self._data_parquet_files()
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
    # Check 8: no video struct columns in data parquet
    # ------------------------------------------------------------------

    def _check_no_video_columns_in_data_parquet(self) -> None:
        """Data parquet files must not contain video feature columns.

        Video features (dtype="video" in info.json) are stored as separate
        MP4 files.  If the data parquet also contains these columns (typically
        as struct<path, timestamp>), the LeRobot dataset loader will fail with
        a CastError because the column names don't match the expected schema.
        """
        video_keys = set(self._get_video_keys())
        if not video_keys:
            return

        parquet_files = self._data_parquet_files()
        if not parquet_files:
            return

        # Only need to check the first file — schema is consistent across chunks.
        pf = parquet_files[0]
        try:
            df = pd.read_parquet(str(pf), columns=None, nrows=0)
        except TypeError:
            # Older pandas versions don't support nrows; read full file.
            df = pd.read_parquet(str(pf))
        except Exception:
            return

        offending = sorted(video_keys & set(df.columns))
        if offending:
            self.errors.append(
                f"Data parquet files contain video feature columns: "
                f"{offending}. Video features (dtype='video' in info.json) "
                f"must NOT appear as columns in data parquet files — they "
                f"are stored as separate MP4 files. Remove these columns "
                f"from the data parquet."
            )

    # ------------------------------------------------------------------
    # Check 9: episode video metadata columns
    # ------------------------------------------------------------------

    def _check_episode_video_metadata_columns(self) -> None:
        """Episode parquet must include per-video-key metadata columns.

        For each video feature key, the episode parquet should contain
        ``videos/{key}/chunk_index`` and ``videos/{key}/from_timestamp``
        so the dataset loader can resolve the correct video file and
        starting timestamp for each episode.
        """
        video_keys = self._get_video_keys()
        if not video_keys:
            return

        df = self._load_episodes_df()
        if df is None:
            return

        missing: List[str] = []
        for vkey in video_keys:
            for suffix in ("chunk_index", "from_timestamp"):
                col = f"videos/{vkey}/{suffix}"
                if col not in df.columns:
                    missing.append(col)

        if missing:
            self.errors.append(
                f"Episode parquet is missing video metadata columns: "
                f"{missing}. For each video feature, the episode parquet "
                f"must include 'videos/{{key}}/chunk_index' and "
                f"'videos/{{key}}/from_timestamp' columns so the dataset "
                f"loader can resolve video files and timestamps."
            )
