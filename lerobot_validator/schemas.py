"""
Schema definitions for metadata CSV and annotation JSON files.
"""

from typing import Dict, Any

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

