"""
Validator for custom_annotation.json file.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Union

import jsonschema
from cloudpathlib import CloudPath, AnyPath

from lerobot_validator.schemas import ANNOTATION_JSON_SCHEMA


class AnnotationValidator:
    """Validates the custom_annotation.json file."""

    def __init__(self, annotation_path: Union[str, Path, CloudPath]):
        """
        Initialize the annotation validator.

        Args:
            annotation_path: Path to the custom_annotation.json file (supports local or cloud paths)
        """
        if isinstance(annotation_path, str):
            self.annotation_path = AnyPath(annotation_path)
        else:
            self.annotation_path = annotation_path
        self.annotations = None
        self.errors: List[str] = []

    def validate(self) -> bool:
        """
        Validate the annotation JSON file.

        Returns:
            True if validation passes, False otherwise
        """
        self.errors = []

        # Check if file exists - if not, it's optional, so return True
        if not self.annotation_path.exists():
            # Annotation file is optional, so no error
            return True

        try:
            # Load JSON - cloudpathlib handles both local and cloud paths
            with self.annotation_path.open("r") as f:
                self.annotations = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            self.errors.append(f"Failed to read JSON file: {e}")
            return False

        # Validate against schema
        self._validate_schema()

        # Additional validations
        self._validate_time_intervals()

        return len(self.errors) == 0

    def _validate_schema(self) -> None:
        """Validate JSON against the expected schema."""
        try:
            jsonschema.validate(instance=self.annotations, schema=ANNOTATION_JSON_SCHEMA)
        except jsonschema.ValidationError as e:
            self.errors.append(f"JSON schema validation failed: {e.message}")
        except Exception as e:
            self.errors.append(f"Unexpected error during schema validation: {e}")

    def _validate_time_intervals(self) -> None:
        """Validate time intervals in annotations."""
        if not self.annotations or "episodes" not in self.annotations:
            return

        for episode in self.annotations["episodes"]:
            episode_id = episode.get("episode_id", "unknown")
            
            # Validate spans
            if "spans" in episode:
                for idx, span in enumerate(episode["spans"]):
                    start = span.get("start_time")
                    end = span.get("end_time")
                    
                    if start is not None and start < 0:
                        self.errors.append(
                            f"Episode '{episode_id}': spans[{idx}] "
                            f"has negative start_time: {start}"
                        )
                    if end is not None and end < 0:
                        self.errors.append(
                            f"Episode '{episode_id}': spans[{idx}] "
                            f"has negative end_time: {end}"
                        )
                    if start is not None and end is not None and start > end:
                        self.errors.append(
                            f"Episode '{episode_id}': spans[{idx}] "
                            f"has start_time ({start}) > end_time ({end})"
                        )

    def get_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.errors

    def get_annotations(self) -> Dict[str, Any]:
        """Get the loaded annotations dictionary."""
        return self.annotations

