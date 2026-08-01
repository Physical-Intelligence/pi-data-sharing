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


ATTRIBUTE_VALUE_SCHEMA: Dict[str, Any] = {
    "anyOf": [
        {"type": "boolean"},
        {"type": "integer", "minimum": -(2**63), "maximum": 2**63 - 1},
        {"type": "number"},
        {"type": "string", "maxLength": 16_384},
    ]
}


def _current_span_schema(kind: str, data_schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["kind", "start_time", "end_time", "data"],
        "properties": {
            "kind": {"const": kind},
            "start_time": {"type": "number"},
            "end_time": {"type": "number"},
            "data": data_schema,
        },
        "additionalProperties": False,
    }


_OUTCOME_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["status"],
    "properties": {
        "status": {"enum": ["success", "failure", "aborted", "cancelled"]},
        "reason": {"type": "string", "maxLength": 16_384},
    },
    "additionalProperties": False,
}

_LANGUAGE_SEGMENT_DATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "annotation_phase": {"enum": ["pre", "post"]},
        "task": {"type": "string", "maxLength": 16_384},
        "text": {"type": "string", "maxLength": 16_384},
        "subtask": {"type": "string", "maxLength": 16_384},
        "outcome": _OUTCOME_SCHEMA,
    },
    "anyOf": [{"required": ["task"]}, {"required": ["text"]}],
    "additionalProperties": False,
}

_SEGMENT_OUTCOME_DATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["status"],
    "properties": _OUTCOME_SCHEMA["properties"],
    "additionalProperties": False,
}

_CONTROL_SOURCE_DATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["value"],
    "properties": {"value": {"enum": ["human", "policy", "none"]}},
    "additionalProperties": False,
}

_RUNTIME_STATE_DATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["name", "value"],
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1024,
            "pattern": r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$",
        },
        "value": {"type": "string", "minLength": 1, "maxLength": 16_384},
    },
    "additionalProperties": False,
}

_FEEDBACK_SIGNAL_DATA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["channel", "value"],
    "properties": {
        "channel": {"enum": ["reward", "post_hoc_reward"]},
        "value": {"enum": ["positive", "negative"]},
    },
    "additionalProperties": False,
}


ANNOTATION_JSON_SCHEMA_CURRENT: Dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "episodes"],
    "properties": {
        "schema_version": {"const": "2.0"},
        "episodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["episode_id"],
                "properties": {
                    "episode_id": {"type": "string"},
                    "attributes": {
                        "type": "object",
                        "maxProperties": 256,
                        "propertyNames": {"minLength": 1, "maxLength": 256},
                        "additionalProperties": ATTRIBUTE_VALUE_SCHEMA,
                    },
                    "spans": {
                        "type": "array",
                        "maxItems": 1000,
                        "items": {
                            "oneOf": [
                                _current_span_schema(
                                    "language_segment", _LANGUAGE_SEGMENT_DATA_SCHEMA
                                ),
                                _current_span_schema(
                                    "segment_outcome", _SEGMENT_OUTCOME_DATA_SCHEMA
                                ),
                                _current_span_schema("control_source", _CONTROL_SOURCE_DATA_SCHEMA),
                                _current_span_schema("runtime_state", _RUNTIME_STATE_DATA_SCHEMA),
                                _current_span_schema(
                                    "feedback_signal", _FEEDBACK_SIGNAL_DATA_SCHEMA
                                ),
                            ]
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}
