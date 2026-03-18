"""
Main validator that orchestrates all checks.
"""

from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
from cloudpathlib import CloudPath, AnyPath

from lerobot_validator.metadata_validator import MetadataValidator
from lerobot_validator.annotation_validator import AnnotationValidator
from lerobot_validator.lerobot_checks import LerobotDatasetChecker
from lerobot_validator.v3_metadata_checker import LerobotV3MetadataChecker


class LerobotDatasetValidator:
    """
    Main validator for lerobot datasets with custom metadata and annotations.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path, CloudPath],
        is_eval_data: Optional[bool] = None,
    ):
        """
        Initialize the validator.

        Args:
            dataset_path: Path to the lerobot dataset directory (supports local Path or GCP CloudPath)
            is_eval_data: Optional flag indicating if this is eval data (True) or training data (False).
                         If provided, validates that all episodes have matching is_eval_episode field.
        
        The validator expects to find these files in the dataset's meta folder:
            - {dataset_path}/meta/custom_metadata.csv
            - {dataset_path}/meta/custom_annotation.json
        """
        # Convert to AnyPath to support both local and cloud paths
        if isinstance(dataset_path, str):
            self.dataset_path = AnyPath(dataset_path)
        else:
            self.dataset_path = dataset_path
        self.is_eval_data = is_eval_data

        # Construct expected paths in meta folder
        meta_dir = self.dataset_path / "meta"
        self.metadata_path = meta_dir / "custom_metadata.csv"
        self.annotation_path = meta_dir / "custom_annotation.json"

        self.metadata_validator = MetadataValidator(self.metadata_path)
        self.annotation_validator = AnnotationValidator(self.annotation_path)
        self.lerobot_checker = LerobotDatasetChecker(self.dataset_path)
        self.v3_checker = LerobotV3MetadataChecker(self.dataset_path)

        self.errors: List[str] = []

    def validate(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if all validations pass, False otherwise
        """
        self.errors = []

        # Run individual validators
        metadata_valid = self.metadata_validator.validate()
        annotation_valid = self.annotation_validator.validate()
        lerobot_valid = self.lerobot_checker.validate()
        v3_valid = self.v3_checker.validate()

        # Collect errors
        self.errors.extend(self.metadata_validator.get_errors())
        self.errors.extend(self.annotation_validator.get_errors())
        self.errors.extend(self.lerobot_checker.get_errors())
        self.errors.extend(self.v3_checker.get_errors())

        # If basic validations pass and annotations exist, run cross-validation
        if metadata_valid and annotation_valid and self.annotation_validator.get_annotations():
            self._cross_validate()

        # Check is_eval_data consistency if provided
        if metadata_valid and self.is_eval_data is not None:
            self._check_is_eval_data_consistency()

        return len(self.errors) == 0

    def _cross_validate(self) -> None:
        """
        Perform cross-validation between metadata, annotations, and lerobot dataset.
        """
        metadata_df = self.metadata_validator.get_metadata_df()
        annotations = self.annotation_validator.get_annotations()

        if metadata_df is None or annotations is None:
            return

        # Check 1: Intervention should only exist for eval episodes
        self._check_intervention_eval_only(metadata_df, annotations)

        # Check 2: Intervention times should be within episode boundaries
        self._check_intervention_boundaries(metadata_df, annotations)

    def _check_intervention_eval_only(
        self, metadata_df: pd.DataFrame, annotations: dict
    ) -> None:
        """
        Check that human intervention only exists for eval episodes.
        """
        # Convert is_eval_episode to boolean
        if "is_eval_episode" not in metadata_df.columns:
            return

        metadata_df["is_eval_episode_bool"] = metadata_df["is_eval_episode"].isin(
            [True, "True", "true", 1]
        )

        # Create mapping of episode_id to is_eval_episode
        eval_map = {}
        if "episode_id" in metadata_df.columns:
            eval_map = dict(
                zip(
                    metadata_df["episode_id"],
                    metadata_df["is_eval_episode_bool"],
                )
            )

        # Check each episode with human intervention spans
        if "episodes" not in annotations:
            return

        for episode in annotations["episodes"]:
            episode_id = episode.get("episode_id")
            if not episode_id:
                continue

            # Check for human_intervention spans
            spans = episode.get("spans", [])
            has_intervention = any(
                span.get("label") == "human_intervention" for span in spans
            )

            if has_intervention:
                if episode_id in eval_map:
                    if not eval_map[episode_id]:
                        self.errors.append(
                            f"Episode '{episode_id}' has human_intervention span but "
                            f"is_eval_episode=False. Human intervention should only "
                            f"exist for eval episodes."
                        )
                else:
                    self.errors.append(
                        f"Episode '{episode_id}' found in annotations but not in metadata CSV"
                    )

    def _check_intervention_boundaries(
        self, metadata_df: pd.DataFrame, annotations: dict
    ) -> None:
        """
        Check that intervention times are within episode boundaries.
        """
        if "episodes" not in annotations:
            return

        for episode in annotations["episodes"]:
            episode_id = episode.get("episode_id")
            if not episode_id:
                continue

            # Try to get duration from lerobot dataset
            duration = self.lerobot_checker.get_episode_duration(episode_id)

            # If not found, try to compute from metadata timestamps (if they exist)
            if duration is None and "episode_id" in metadata_df.columns:
                episode_row = metadata_df[metadata_df["episode_id"] == episode_id]
                if len(episode_row) > 0:
                    # Note: The new schema doesn't have end_timestamp, so we rely on lerobot dataset
                    pass

            # If we have duration, validate span times
            if duration is not None:
                spans = episode.get("spans", [])
                for idx, span in enumerate(spans):
                    end_time = span.get("end_time")
                    if end_time is not None and end_time > duration:
                        label = span.get("label", "unknown")
                        self.errors.append(
                            f"Episode '{episode_id}': spans[{idx}] (label='{label}') "
                            f"end_time ({end_time}s) exceeds episode duration ({duration}s)"
                        )

    def _check_is_eval_data_consistency(self) -> None:
        """
        Check that all episodes have is_eval_episode matching the is_eval_data flag.
        """
        metadata_df = self.metadata_validator.get_metadata_df()
        
        if metadata_df is None or "is_eval_episode" not in metadata_df.columns:
            return
        
        # Convert is_eval_episode to boolean
        metadata_df["is_eval_episode_bool"] = metadata_df["is_eval_episode"].isin(
            [True, "True", "true", 1]
        )
        
        # Check each episode
        mismatches = []
        for idx, row in metadata_df.iterrows():
            episode_id = row.get("episode_id", f"row_{idx}")
            is_eval = row["is_eval_episode_bool"]
            
            if is_eval != self.is_eval_data:
                mismatches.append((idx, episode_id, is_eval))
        
        if mismatches:
            expected_value = "True" if self.is_eval_data else "False"
            data_type = "eval" if self.is_eval_data else "teleop"
            
            error_details = []
            for idx, episode_id, actual_value in mismatches:
                actual_str = "True" if actual_value else "False"
                error_details.append(
                    f"  Row {idx} (episode '{episode_id}'): is_eval_episode={actual_str}, expected {expected_value}"
                )
            
            self.errors.append(
                f"Dataset is marked as {data_type} data (--data-type={data_type}), "
                f"but {len(mismatches)} episode(s) have mismatched is_eval_episode values:\n"
                + "\n".join(error_details)
            )

    def get_errors(self) -> List[str]:
        """Get all validation errors."""
        return self.errors

    def print_results(self) -> None:
        """Print validation results."""
        if len(self.errors) == 0:
            print("✓ All validations passed!")
        else:
            print(f"✗ Validation failed with {len(self.errors)} error(s):\n")
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")

