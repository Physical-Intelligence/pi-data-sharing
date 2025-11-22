"""Tests for annotation JSON validator."""

import json
import tempfile
from pathlib import Path

import pytest

from lerobot_validator.annotation_validator import AnnotationValidator


def test_valid_annotation():
    """Test validation with valid annotation JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        annotation_path = Path(tmpdir) / "custom_annotation.json"

        # Create valid annotation with new schema
        annotations = {
            "episodes": [
                {
                    "episode_id": "ep_001",
                    "spans": [
                        {"start_time": 1.0, "end_time": 2.5, "label": "human_intervention"},
                        {"start_time": 5.0, "end_time": 7.0, "label": "human_intervention"},
                        {"start_time": 0.0, "end_time": 3.0, "label": "grasp"},
                        {"start_time": 3.0, "end_time": 8.0, "label": "move"},
                    ],
                    "extras": {"notes": "This is a test episode"},
                },
                {
                    "episode_id": "ep_002",
                    "spans": [],
                    "extras": {},
                },
            ]
        }

        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

        # Validate
        validator = AnnotationValidator(annotation_path)
        assert validator.validate() is True
        assert len(validator.get_errors()) == 0


def test_optional_fields():
    """Test validation with optional fields missing (should pass)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        annotation_path = Path(tmpdir) / "custom_annotation.json"

        # Only spans provided, extras missing
        annotations = {
            "episodes": [
                {
                    "episode_id": "ep_001",
                    "spans": [
                        {"start_time": 1.0, "end_time": 2.5, "label": "human_intervention"}
                    ],
                }
            ]
        }

        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

        # Validate - should pass since spans and extras are optional
        validator = AnnotationValidator(annotation_path)
        assert validator.validate() is True
        assert len(validator.get_errors()) == 0


def test_invalid_time_intervals():
    """Test validation with invalid time intervals."""
    with tempfile.TemporaryDirectory() as tmpdir:
        annotation_path = Path(tmpdir) / "custom_annotation.json"

        # Start time > end time
        annotations = {
            "episodes": [
                {
                    "episode_id": "ep_001",
                    "spans": [
                        {"start_time": 5.0, "end_time": 2.0, "label": "human_intervention"}  # Invalid
                    ],
                    "extras": {},
                }
            ]
        }

        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

        # Validate
        validator = AnnotationValidator(annotation_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("start_time" in err.lower() and "end_time" in err.lower() for err in errors)


def test_negative_times():
    """Test validation with negative times."""
    with tempfile.TemporaryDirectory() as tmpdir:
        annotation_path = Path(tmpdir) / "custom_annotation.json"

        # Negative time values
        annotations = {
            "episodes": [
                {
                    "episode_id": "ep_001",
                    "spans": [
                        {"start_time": -1.0, "end_time": 2.0, "label": "grasp"}  # Invalid
                    ],
                    "extras": {},
                }
            ]
        }

        with open(annotation_path, "w") as f:
            json.dump(annotations, f)

        # Validate
        validator = AnnotationValidator(annotation_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("negative" in err.lower() for err in errors)


def test_invalid_json():
    """Test validation with malformed JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        annotation_path = Path(tmpdir) / "custom_annotation.json"

        # Write invalid JSON
        with open(annotation_path, "w") as f:
            f.write("{invalid json")

        # Validate
        validator = AnnotationValidator(annotation_path)
        assert validator.validate() is False
        errors = validator.get_errors()
        assert any("json" in err.lower() for err in errors)


def test_file_not_found():
    """Test validation with non-existent file - should pass since annotation is optional."""
    with tempfile.TemporaryDirectory() as tmpdir:
        annotation_path = Path(tmpdir) / "nonexistent_annotation.json"
        # File doesn't exist but path is accessible
        validator = AnnotationValidator(annotation_path)
        assert validator.validate() is True  # Annotation file is optional
        errors = validator.get_errors()
        assert len(errors) == 0  # No errors since file is optional
