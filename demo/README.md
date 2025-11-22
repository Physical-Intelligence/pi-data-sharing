# Demo: Using the Lerobot Dataset Validator

This demo shows how to use the lerobot dataset validator library with a sample dataset.

## Directory Structure

```
demo/
├── README.md
├── sample_dataset/           # Sample lerobot dataset
│   ├── info.json
│   └── meta/
│       ├── custom_metadata.csv      # Sample metadata
│       └── custom_annotation.json   # Sample annotation
└── run_demo.py              # Demo script
```

## Running the Demo

### Option 1: Using the CLI (after installation)

```bash
# Install the package first
uv pip install -e ..

# Run validation (files are in sample_dataset/meta/)
python validate.py \
  --dataset-path demo/sample_dataset \
  --dataset-name "demo-dataset" \
  --bucket-name "demo-bucket" \
  --data-type "teleop"
```

### Option 2: Using Python (without installation)

```bash
# From the repo root
python demo/run_demo.py
```

## What This Demo Tests

1. ✅ Files in correct location (dataset/meta/ folder)
2. ✅ Valid metadata CSV with all required columns
3. ✅ Valid GCS URIs for policy checkpoints
4. ✅ Valid annotation JSON with proper structure
5. ✅ Lerobot dataset with task information
6. ✅ Human intervention only on eval episodes
7. ✅ Time boundaries within episode duration
8. ✅ is_eval_data consistency check
9. ✅ GCP path computation

## Expected Output

If everything is correct, you should see:

```
================================================================================
Lerobot Dataset Validator
================================================================================

Dataset path:    demo/sample_dataset
Metadata CSV:    demo/sample_dataset/meta/custom_metadata.csv
Annotation JSON: demo/sample_dataset/meta/custom_annotation.json
Data type:       Teleop

Running validation...

✓ All validations passed!

================================================================================

Upload Instructions
==================

Your dataset is ready to upload!

GCP Destination Path:
gs://pi-lerobot-datasetsdemopartner/demo-dataset/[version]/

To upload your dataset, run:

    gsutil -m cp -r demo/sample_dataset/* gs://...

================================================================================
```

## Modifying the Demo

Try making these changes to see validation errors:

1. **Move files out of meta folder** - Validation will fail (files not found)
2. **Remove a required column** from `meta/custom_metadata.csv`
3. **Change GCS URI** to invalid format (e.g., remove `gs://`)
4. **Add human intervention** to a non-eval episode
5. **Use time values exceeding episode duration** in annotations
6. **Mix is_eval_episode values** (some True, some False) - Will fail consistency check

