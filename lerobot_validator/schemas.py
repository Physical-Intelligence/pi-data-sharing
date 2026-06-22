"""
Schema definitions for metadata CSV and annotation JSON files.
"""

from typing import Dict, Any, List, Literal, Optional

# Expected columns in custom_metadata.csv (on top of what's already in lerobot dataset)
REQUIRED_METADATA_COLUMNS = [
    "episode_index",
    "operator_id",
    "is_eval_episode",
    "episode_id",
    "start_timestamp",
    "checkpoint_path",
    "success",
    "station_id",  # the 'scene' or the table the robot is attached to
    "robot_id",    # the robot hardware
]

DatasetProfile = Literal["robot", "umi"]

# UMI datasets are human demonstrations, so they do not require robot/eval-only
# metadata such as checkpoint_path, success, or robot_id.
UMI_REQUIRED_METADATA_COLUMNS = [
    "episode_index",
    "operator_id",
    "is_eval_episode",
    "episode_id",
    "start_timestamp",
    "station_id",
]

REQUIRED_METADATA_COLUMNS_BY_PROFILE: Dict[DatasetProfile, List[str]] = {
    "robot": REQUIRED_METADATA_COLUMNS,
    "umi": UMI_REQUIRED_METADATA_COLUMNS,
}

# UMI partners may include provider-specific metadata fields. The ingestion
# pipeline preserves or ignores these fields, so the validator should not
# reject an otherwise valid UMI dataset for including them.
ALLOWED_METADATA_COLUMNS_BY_PROFILE: Dict[DatasetProfile, Optional[List[str]]] = {
    "robot": REQUIRED_METADATA_COLUMNS,
    "umi": None,
}

# Required fields in the lerobot dataset itself
REQUIRED_LEROBOT_FIELDS = [
    "fps",   # fps field in info.json (frequency of data collection)
]

# JSON schema for custom_annotation.json
ANNOTATION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["episodes"],
    "properties": {
        "episodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["episode_id"],
                "properties": {
                    "episode_id": {"type": "string"},
                    "spans": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["start_time", "end_time", "label"],
                            "properties": {
                                "start_time": {"type": "number"},  # relative seconds from start
                                "end_time": {"type": "number"},    # relative seconds from start
                                "label": {"type": "string"},       # e.g., "human_intervention" or custom labels
                            },
                            "additionalProperties": False,
                        },
                    },
                    "extras": {
                        "type": "object",
                        # extras allows arbitrary key-value pairs for annotations not captured in existing spec
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}
