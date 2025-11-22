# Lerobot Dataset Validator

A lightweight library for validating lerobot dataset metadata and annotations, and computing GCP upload paths.

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Authentication (for GCP paths)

```bash
gcloud auth login
gcloud auth application-default login
```

### 3. Validate Your Dataset

```bash
# Validate training/teleop data
python validate.py validate \
  --dataset-path ./my-dataset \
  --data-type teleop

# Validate evaluation data
python validate.py validate \
  --dataset-path ./my-dataset \
  --data-type eval
```

### 4. Get Upload Instructions

```bash
python validate.py compute-path \
  --dataset-path ./my-dataset \
  --dataset-name my-robot-data \
  --bucket-name my-gcs-bucket \
  --data-type eval
```

## Features

- ✅ Validates `custom_metadata.csv` with required columns
- ✅ Validates `custom_annotation.json` structure (optional file)
- ✅ Checks lerobot dataset for task and fps
- ✅ Cross-validates: intervention only for eval episodes, time boundaries
- ✅ Computes GCP upload paths with custom prefixes
- ✅ Supports both local paths and GCP URIs (gs://)
- ✅ Two separate CLI commands: `validate` and `compute-path`
- ✅ Type-safe CLI using tyro

## Dataset Structure

Your dataset should have:
```
my-dataset/
├── info.json                    # Must contain "task" and "fps"
└── meta/
    ├── custom_metadata.csv      # Episode metadata (required)
    └── custom_annotation.json   # Episode annotations (optional)
```

## Required Files

### info.json

Must contain:
- `task`: Task string for the dataset
- `fps`: Data collection frequency (frames per second)

Example:
```json
{
  "task": "pick_and_place",
  "fps": 30
}
```

### custom_metadata.csv

Must have exactly these columns:

| Column | Type | Description |
|--------|------|-------------|
| `episode_index` | int | Episode number |
| `operator_id` | string | Operator identifier |
| `is_eval_episode` | boolean | True for eval, False for training |
| `episode_id` | string | Unique episode identifier |
| `start_timestamp` | float | UTC seconds (Unix epoch time) |
| `checkpoint_path` | string | GCS URI (only for eval episodes) |
| `success` | boolean | Whether episode was successful |
| `station_id` | string | Station/scene identifier |
| `robot_id` | string | Robot hardware identifier |

Example:
```csv
episode_index,operator_id,is_eval_episode,episode_id,start_timestamp,checkpoint_path,success,station_id,robot_id
0,operator_alice,True,ep_001,1730455200,gs://my-bucket/checkpoints/policy_v1.0.pth,True,station_01,robot_alpha
1,operator_bob,False,ep_002,1730458800,,True,station_01,robot_alpha
```

**Important Notes:**
- `start_timestamp` must be UTC seconds (Unix epoch time), not ISO format
  - ✅ Valid: `1730455200`
  - ❌ Invalid: `2024-11-01T10:00:00`
- `checkpoint_path` should only be set for eval episodes (is_eval_episode=True)
- `checkpoint_path` must be a valid GCS URI format: `gs://bucket/path/to/checkpoint`

See `examples/example_dataset/meta/custom_metadata.csv` for a complete example.

### custom_annotation.json (Optional)

Must follow this structure:

```json
{
  "episodes": [
    {
      "episode_id": "ep_001",
      "spans": [
        {"start_time": 0.0, "end_time": 5.0, "label": "grasp"},
        {"start_time": 2.0, "end_time": 3.0, "label": "human_intervention"}
      ],
      "extras": {"notes": "optional metadata"}
    }
  ]
}
```

- `spans`: List of time-based annotations with start_time, end_time, and label
  - `start_time` and `end_time` are relative seconds from episode start (just like timestamps in LeRobot data)
  - Use label `"human_intervention"` for human interventions during policy rollout
- `extras`: Free-form metadata (optional)

**Note:** This file is optional. If missing, validation will still pass.

See `examples/example_dataset/meta/custom_annotation.json` for a complete example.

## CLI Commands

The validator provides two separate commands:

### 1. `validate` - Validate Dataset Only

Validates dataset metadata and annotations without computing upload paths.

```bash
python validate.py validate \
  --dataset-path PATH \
  --data-type TYPE
```

**Arguments:**
- `--dataset-path`: Path to dataset directory (local or GCP URI like gs://bucket/path)
- `--data-type`: Either `teleop` (training) or `eval` (evaluation)

**Examples:**
```bash
# Validate local training data
python validate.py validate --dataset-path ./my-dataset --data-type teleop

# Validate GCP evaluation data
python validate.py validate --dataset-path gs://my-bucket/datasets/my-dataset --data-type eval
```

### 2. `compute-path` - Compute GCP Upload Path

Computes the GCP upload path for a dataset (validates by default).

```bash
python validate.py compute-path \
  --dataset-path PATH \
  --dataset-name NAME \
  --bucket-name BUCKET \
  --data-type TYPE \
  [--dataset-version VERSION] \
  [--custom-folder-prefix PREFIX] \
  [--skip-validation]
```

**Arguments:**
- `--dataset-path`: Path to dataset directory (required)
- `--dataset-name`: Dataset name for GCP path (required)
- `--bucket-name`: GCS bucket name (required)
- `--data-type`: Either `teleop` or `eval` (required)
- `--dataset-version`: Version string (optional, default: timestamp)
- `--custom-folder-prefix`: Custom folder prefix (optional, e.g., "experiments/phase-1")
- `--skip-validation`: Skip validation, only compute path (optional)

**Examples:**
```bash
# Compute path with validation
python validate.py compute-path \
  --dataset-path ./my-dataset \
  --dataset-name robot-manipulation \
  --bucket-name production-data \
  --data-type eval

# With custom version and prefix
python validate.py compute-path \
  --dataset-path gs://source-bucket/datasets/my-dataset \
  --dataset-name robot-manipulation \
  --bucket-name target-bucket \
  --data-type teleop \
  --dataset-version v2.1.0 \
  --custom-folder-prefix experiments/phase-1

# Skip validation (faster, but not recommended)
python validate.py compute-path \
  --dataset-path ./my-dataset \
  --dataset-name my-data \
  --bucket-name my-bucket \
  --data-type eval \
  --skip-validation
```

### Get Help

```bash
python validate.py --help
python validate.py validate --help
python validate.py compute-path --help
```

## Data Types

The validator uses two data types:

| Data Type | Description | is_eval_episode |
|-----------|-------------|-----------------|
| `teleop` | Training/teleoperation data | False |
| `eval` | Evaluation/policy rollout data | True |

**Important:** All episodes in a dataset must have matching `is_eval_episode` values that correspond to the specified data type.

## Validation Rules

### Metadata CSV
- All required columns must be present
- No extra columns allowed
- `episode_id` must be unique
- `is_eval_episode` and `success` must be boolean
- `start_timestamp` must be UTC seconds (Unix epoch time) in range 2000-2100
- `checkpoint_path` must be a valid GCS URI (gs://bucket/path) when specified
- `checkpoint_path` should only be set for eval episodes (is_eval_episode=True)

### Annotation JSON (if present)
- Must follow the required schema structure
- `spans` with `start_time < end_time` (timestamps are relative seconds from episode start)
- No negative time values allowed
- Proper JSON structure required

### Lerobot Dataset
- Must contain `task` field in `info.json`
- Must contain `fps` field in `info.json`

### Cross-Validation
1. **Human intervention constraint**: Spans with label `"human_intervention"` only allowed for eval episodes (is_eval_episode=True)
2. **Time boundary constraint**: All span times must be ≤ episode duration
3. **Data type consistency**: 
   - `--data-type teleop`: All episodes must have is_eval_episode=False
   - `--data-type eval`: All episodes must have is_eval_episode=True
4. **Checkpoint path constraint**: checkpoint_path should not be specified for non-eval episodes

## GCP Path Format

The computed GCP path follows this format:

```
gs://bucket/[custom_prefix/]dataset/version/data_type/
```

Examples:
- Eval data: `gs://my-bucket/dataset/v1.0/eval/`
- Teleop data: `gs://my-bucket/dataset/v1.0/teleop/`
- With prefix: `gs://my-bucket/experiments/batch-1/dataset/v1.0/eval/`

## CloudPath Support

The validator supports both local filesystem paths and GCP URIs:

```bash
# Local path
python validate.py validate --dataset-path ./my-dataset --data-type eval

# GCP URI
python validate.py validate --dataset-path gs://my-bucket/datasets/my-dataset --data-type eval
```

All file operations work transparently with both path types.

## Python API

```python
from pathlib import Path
from cloudpathlib import AnyPath
from lerobot_validator import LerobotDatasetValidator, compute_gcp_path

# Validate (expects files in dataset/meta/ folder)
# Supports both local Path and CloudPath
dataset_path = AnyPath("./dataset")  # or gs://bucket/dataset

validator = LerobotDatasetValidator(
    dataset_path=dataset_path,
    is_eval_data=True,  # True for eval, False for teleop
)

if validator.validate():
    print("✓ Validation passed!")
    
    # Compute GCP path
    gcp_path = compute_gcp_path(
        dataset_name="my-dataset",
        bucket_name="my-gcs-bucket",
        data_type="eval",  # "teleop" or "eval"
        custom_folder_prefix="experiments/run-1",  # Optional
    )
    print(f"Upload to: {gcp_path}")
else:
    for error in validator.get_errors():
        print(f"Error: {error}")
```

## Common Errors

**"Missing required columns in metadata CSV"**
- Add all required columns: episode_index, operator_id, is_eval_episode, episode_id, start_timestamp, checkpoint_path, success, station_id, robot_id

**"Unexpected columns found"**
- Remove extra columns not in the required list

**"Column 'start_timestamp' must contain valid UTC timestamps in seconds"**
- Use Unix epoch time (e.g., 1730455200), not ISO format (2024-11-01T10:00:00)
- Valid range: Year 2000 to 2100

**"Episode has human_intervention span but is_eval_episode=False"**
- Human interventions only allowed for eval episodes
- Either set is_eval_episode=True or remove the intervention spans

**"checkpoint_path should not be specified for non-eval episodes"**
- Only set checkpoint_path for eval episodes (is_eval_episode=True)
- Leave it empty for training/teleop episodes

**"Intervention time exceeds episode duration"**
- Check that all span end_time values are within episode length

**"No task string found in lerobot dataset"**
- Ensure info.json contains "task" field

**"Missing 'fps' field in info.json"**
- Add "fps" field to info.json to specify data collection frequency

**"path_to_policy_checkpoint must contain valid GCS URIs"**
- Use format `gs://bucket/path/to/checkpoint.pth`
- Make sure URIs start with `gs://`
- Include both bucket name and path

**"Dataset is marked as eval/teleop data but episodes have mismatched is_eval_episode values"**
- Eval data: All episodes should have is_eval_episode=True
- Teleop data: All episodes should have is_eval_episode=False

## Examples

Complete example datasets are provided:
- `demo/sample_dataset/` - Working demo with 3 episodes
- `examples/example_dataset/` - Reference implementation

To run the demo:
```bash
python validate.py validate \
  --dataset-path demo/sample_dataset \
  --data-type eval
```

## Development

### Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

### Code Formatting

```bash
black lerobot_validator/ tests/
isort lerobot_validator/ tests/
mypy lerobot_validator/
```

## Project Structure

```
pi-data-sharing/
├── validate.py              # Main entry point
├── lerobot_validator/       # Core library
│   ├── cli.py              # CLI commands (validate, compute-path)
│   ├── gcp_path.py         # GCP path computation
│   ├── metadata_validator.py
│   ├── annotation_validator.py
│   ├── lerobot_checks.py
│   ├── validator.py
│   └── schemas.py
├── tests/                   # Test suite
├── examples/                # Example CSV and JSON files
├── demo/                    # Working demo
├── TIMESTAMP_VALIDATION.md  # Timestamp format guide
└── README.md               # This file
```

## Timestamp Format

The `start_timestamp` field must be in **UTC seconds** (Unix epoch time):

```python
from datetime import datetime

# Convert ISO to UTC seconds
iso_time = "2024-11-01T14:00:00"
utc_seconds = int(datetime.fromisoformat(iso_time).timestamp())
print(utc_seconds)  # 1730469600
```

See `TIMESTAMP_VALIDATION.md` for detailed information and conversion examples.

## License

Apache-2.0

## Support

- **Examples**: See `examples/` and `demo/` directories
- **Documentation**: See `TIMESTAMP_VALIDATION.md` for timestamp format details
- **Tests**: Check `tests/` for usage patterns
- **Issues**: See troubleshooting section above
- **Contact**: support@physicalintelligence.company
