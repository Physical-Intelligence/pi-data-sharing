"""Tests for cloud-storage path construction and provider-neutral readers."""

from pathlib import Path

import pandas as pd
import pytest
from cloudpathlib import S3Path
from cloudpathlib.local import LocalS3Client

from lerobot_validator.storage import (
    dataset_path,
    media_probe_target,
    path_kind,
    read_csv,
    read_parquet,
)


def test_custom_s3_endpoint_uses_region_and_presigned_https(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")

    path = dataset_path(
        "s3://dataset-bucket/run-1",
        s3_endpoint_url="https://namespace.compat.objectstorage.us-ashburn-1.oraclecloud.com",
        s3_region="us-ashburn-1",
    )

    assert isinstance(path, S3Path)
    assert path_kind(path) == "S3-compatible URI"
    assert path.client.client.meta.region_name == "us-ashburn-1"

    with media_probe_target(path / "videos/sample.mp4") as target:
        assert target.startswith(
            "https://namespace.compat.objectstorage.us-ashburn-1.oraclecloud.com/"
            "dataset-bucket/run-1/videos/sample.mp4?"
        )
        assert "X-Amz-Algorithm=AWS4-HMAC-SHA256" in target


def test_s3_options_rejected_for_non_s3_path(tmp_path):
    with pytest.raises(ValueError, match="require an s3:// dataset path"):
        dataset_path(str(tmp_path), s3_region="us-ashburn-1")


def test_cloud_readers_use_bound_client(tmp_path):
    client = LocalS3Client(local_storage_dir=tmp_path / "object-storage")
    csv_path = client.CloudPath("s3://bucket/meta/custom_metadata.csv")
    parquet_path = client.CloudPath("s3://bucket/data/chunk-000/file-000.parquet")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("episode_index,episode_id\n0,ep-0\n")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    with parquet_path.open("wb") as stream:
        pd.DataFrame({"episode_index": [0], "timestamp": [0.0]}).to_parquet(stream, index=False)

    assert read_csv(csv_path).to_dict("records") == [{"episode_index": 0, "episode_id": "ep-0"}]
    assert read_parquet(parquet_path, columns=["episode_index"]).to_dict("records") == [
        {"episode_index": 0}
    ]


def test_local_path_is_unchanged(tmp_path):
    path = dataset_path(str(tmp_path))
    assert isinstance(path, Path)
    assert path == tmp_path
    assert path_kind(path) == "Local path"
