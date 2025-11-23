"""
Validator for custom_metadata.csv file.
"""

from pathlib import Path
from typing import List, Dict, Any, Union

import pandas as pd
from cloudpathlib import CloudPath, AnyPath

from lerobot_validator.schemas import REQUIRED_METADATA_COLUMNS


class MetadataValidator:
    """Validates the custom_metadata.csv file."""

    def __init__(self, metadata_path: Union[str, Path, CloudPath]):
        """
        Initialize the metadata validator.

        Args:
            metadata_path: Path to the custom_metadata.csv file (supports local or cloud paths)
        """
        if isinstance(metadata_path, str):
            self.metadata_path = AnyPath(metadata_path)
        else:
            self.metadata_path = metadata_path
        self.df = None
        self.errors: List[str] = []

    def validate(self) -> bool:
        """
        Validate the metadata CSV file.

        Returns:
            True if validation passes, False otherwise
        """
        self.errors = []

        # Check if file exists
        if not self.metadata_path.exists():
            self.errors.append(f"Metadata file not found: {self.metadata_path}")
            return False

        try:
            # Load CSV - pandas can read from cloud paths directly via cloudpathlib
            # Convert CloudPath to string for pandas compatibility
            path_str = str(self.metadata_path)
            self.df = pd.read_csv(path_str)
        except Exception as e:
            self.errors.append(f"Failed to read CSV file: {e}")
            return False

        # Check for required columns
        self._check_required_columns()

        # Check for unexpected columns
        self._check_unexpected_columns()

        # Check data types and values
        self._check_data_validity()

        return len(self.errors) == 0

    def _check_required_columns(self) -> None:
        """Check that all required columns are present."""
        missing_columns = set(REQUIRED_METADATA_COLUMNS) - set(self.df.columns)
        if missing_columns:
            self.errors.append(
                f"Missing required columns in metadata CSV: {sorted(missing_columns)}"
            )

    def _check_unexpected_columns(self) -> None:
        """Check for unexpected columns in the CSV."""
        unexpected_columns = set(self.df.columns) - set(REQUIRED_METADATA_COLUMNS)
        if unexpected_columns:
            self.errors.append(
                f"Unexpected columns found in metadata CSV: {sorted(unexpected_columns)}. "
                f"Only the following columns are allowed: {REQUIRED_METADATA_COLUMNS}"
            )

    def _check_data_validity(self) -> None:
        """Check validity of data in the CSV."""
        if self.df is None or len(self.errors) > 0:
            return

        # Check is_eval_episode is boolean
        if "is_eval_episode" in self.df.columns:
            non_bool_values = self.df[
                ~self.df["is_eval_episode"].isin([True, False, "True", "False", "true", "false", 0, 1])
            ]
            if len(non_bool_values) > 0:
                self.errors.append(
                    f"Column 'is_eval_episode' must contain boolean values. "
                    f"Found invalid values at rows: {non_bool_values.index.tolist()}"
                )

        # Check success is boolean
        if "success" in self.df.columns:
            non_bool_values = self.df[
                ~self.df["success"].isin([True, False, "True", "False", "true", "false", 0, 1])
            ]
            if len(non_bool_values) > 0:
                self.errors.append(
                    f"Column 'success' must contain boolean values. "
                    f"Found invalid values at rows: {non_bool_values.index.tolist()}"
                )

        # Check for missing episode_id
        if "episode_id" in self.df.columns:
            missing_ids = self.df[self.df["episode_id"].isna()]
            if len(missing_ids) > 0:
                self.errors.append(
                    f"Column 'episode_id' cannot have missing values. "
                    f"Found missing values at rows: {missing_ids.index.tolist()}"
                )

            # Check for duplicate episode_id
            duplicates = self.df[self.df["episode_id"].duplicated(keep=False)]
            if len(duplicates) > 0:
                dup_ids = duplicates["episode_id"].unique().tolist()
                self.errors.append(
                    f"Column 'episode_id' must have unique values. "
                    f"Found duplicates: {dup_ids}"
                )

        # Check start_timestamp is valid UTC seconds (epoch time)
        if "start_timestamp" in self.df.columns:
            self._check_start_timestamp_format()

        # Check checkpoint_path validation rules
        if "checkpoint_path" in self.df.columns and "is_eval_episode" in self.df.columns:
            self._check_checkpoint_path_rules()

    def _check_start_timestamp_format(self) -> None:
        """Check that start_timestamp is a valid UTC timestamp in seconds (epoch time)."""
        invalid_timestamps = []
        
        for idx, row in self.df.iterrows():
            timestamp = row.get("start_timestamp")
            
            # Skip if missing (will be caught by required columns check)
            if pd.isna(timestamp):
                continue
            
            # Try to convert to float (epoch time should be numeric)
            try:
                timestamp_float = float(timestamp)
                
                # Sanity check: Unix epoch started Jan 1, 1970
                # Reasonable range: Jan 1, 2000 (946684800) to Jan 1, 2100 (4102444800)
                MIN_TIMESTAMP = 946684800  # Jan 1, 2000 00:00:00 UTC
                MAX_TIMESTAMP = 4102444800  # Jan 1, 2100 00:00:00 UTC
                
                if timestamp_float < MIN_TIMESTAMP or timestamp_float > MAX_TIMESTAMP:
                    episode_id = row.get("episode_id", f"row_{idx}")
                    invalid_timestamps.append(
                        (idx, episode_id, timestamp, 
                         f"timestamp {timestamp_float} is outside reasonable range (2000-2100)")
                    )
            except (ValueError, TypeError):
                episode_id = row.get("episode_id", f"row_{idx}")
                invalid_timestamps.append(
                    (idx, episode_id, timestamp, 
                     f"not a valid numeric timestamp (expected UTC seconds since epoch)")
                )
        
        if invalid_timestamps:
            error_details = []
            for idx, episode_id, timestamp, reason in invalid_timestamps:
                error_details.append(f"  Row {idx} (episode '{episode_id}'): '{timestamp}' - {reason}")
            
            self.errors.append(
                f"Column 'start_timestamp' must contain valid UTC timestamps in seconds (Unix epoch time).\n"
                + "\n".join(error_details)
            )

    def _check_checkpoint_path_rules(self) -> None:
        """
        Check that checkpoint_path is only specified for eval episodes.
        Also validates that it's a valid GCS URI when present.
        """
        invalid_non_eval = []
        invalid_uris = []
        
        for idx, row in self.df.iterrows():
            is_eval = row.get("is_eval_episode")
            checkpoint_path = row.get("checkpoint_path")
            
            # Convert is_eval to boolean
            is_eval_bool = is_eval in [True, "True", "true", 1]
            
            # Check if checkpoint_path is specified for non-eval episode
            if not is_eval_bool and pd.notna(checkpoint_path):
                checkpoint_str = str(checkpoint_path).strip()
                if checkpoint_str:  # Not empty string
                    episode_id = row.get("episode_id", f"row_{idx}")
                    invalid_non_eval.append((idx, episode_id))
            
            # If eval episode with checkpoint_path, validate it's a valid GCS URI
            if is_eval_bool and pd.notna(checkpoint_path):
                checkpoint_str = str(checkpoint_path).strip()
                
                # Check if it starts with gs://
                if not checkpoint_str.startswith("gs://"):
                    episode_id = row.get("episode_id", f"row_{idx}")
                    invalid_uris.append((idx, episode_id, checkpoint_str, "must start with 'gs://'"))
                    continue
                
                # Check if it has a bucket name (something after gs://)
                path_after_gs = checkpoint_str[5:]  # Remove 'gs://'
                if not path_after_gs or path_after_gs.startswith("/"):
                    episode_id = row.get("episode_id", f"row_{idx}")
                    invalid_uris.append((idx, episode_id, checkpoint_str, "missing bucket name"))
                    continue
                
                # Check if it has at least bucket/path structure
                if "/" not in path_after_gs:
                    episode_id = row.get("episode_id", f"row_{idx}")
                    invalid_uris.append((idx, episode_id, checkpoint_str, "must include path after bucket"))
        
        # Report errors for checkpoint_path on non-eval episodes
        if invalid_non_eval:
            error_details = []
            for idx, episode_id in invalid_non_eval:
                error_details.append(f"  Row {idx} (episode '{episode_id}')")
            
            self.errors.append(
                f"Column 'checkpoint_path' should not be specified for non-eval episodes (is_eval_episode=False).\n"
                f"Found checkpoint_path values on {len(invalid_non_eval)} non-eval episode(s):\n"
                + "\n".join(error_details)
            )
        
        # Report invalid GCS URIs
        if invalid_uris:
            error_details = []
            for idx, episode_id, uri, reason in invalid_uris:
                error_details.append(f"  Row {idx} (episode '{episode_id}'): '{uri}' - {reason}")
            
            self.errors.append(
                "Column 'checkpoint_path' must contain valid GCS URIs (format: gs://bucket/path/to/checkpoint).\n"
                + "\n".join(error_details)
            )

    def get_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.errors

    def get_metadata_df(self) -> pd.DataFrame:
        """Get the loaded metadata DataFrame."""
        return self.df

