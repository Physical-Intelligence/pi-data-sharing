"""Storage helpers for local, GCS, and S3-compatible dataset paths."""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, Union

import pandas as pd
from cloudpathlib import AnyPath, CloudPath

PathLike = Union[Path, CloudPath]


def dataset_path(
    uri: str,
    *,
    s3_endpoint_url: Optional[str] = None,
    s3_region: Optional[str] = None,
) -> PathLike:
    """Create a local or cloud path, configuring custom S3 endpoints explicitly.

    Credentials intentionally come from boto3's normal credential provider chain.
    This keeps secrets out of command-line arguments and supports environment,
    shared-config, workload-identity, and instance-role credentials.
    """
    if not uri.startswith("s3://"):
        if s3_endpoint_url is not None or s3_region is not None:
            raise ValueError("--s3-endpoint-url/--s3-region require an s3:// dataset path")
        return AnyPath(uri)

    try:
        import boto3
        from cloudpathlib import S3Client
    except ImportError as exc:
        raise RuntimeError("S3 support is not installed; install cloudpathlib[s3]") from exc

    session = boto3.Session(region_name=s3_region)
    client = S3Client(
        boto3_session=session,
        endpoint_url=s3_endpoint_url,
        file_cache_mode="close_file",
    )
    return client.CloudPath(uri)


def path_kind(path: PathLike) -> str:
    """Return a user-facing path kind without assuming every cloud is GCP."""
    if not isinstance(path, CloudPath):
        return "Local path"
    if str(path).startswith("s3://"):
        return "S3-compatible URI"
    if str(path).startswith("gs://"):
        return "GCP URI"
    return "Cloud URI"


def read_csv(path: PathLike, **kwargs: Any) -> pd.DataFrame:
    """Read CSV through the path's configured client.

    Passing ``str(CloudPath)`` to pandas bypasses cloudpathlib and requires a
    separate pandas filesystem plugin. Opening through CloudPath preserves the
    custom endpoint and credential configuration.
    """
    if isinstance(path, CloudPath):
        with path.open("rb") as stream:
            return pd.read_csv(stream, **kwargs)
    return pd.read_csv(path, **kwargs)


def read_parquet(
    path: PathLike,
    *,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read a parquet file through the path's configured client."""
    if isinstance(path, CloudPath):
        with path.open("rb") as stream:
            return pd.read_parquet(stream, columns=columns)
    return pd.read_parquet(path, columns=columns)


def read_parquet_dataset(
    path: PathLike,
    *,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read either a parquet file or a directory of parquet fragments."""
    if not path.is_dir():
        return read_parquet(path, columns=columns)

    fragments = sorted(path.glob("**/*.parquet"))
    if not fragments:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return pd.concat(
        [read_parquet(fragment, columns=columns) for fragment in fragments],
        ignore_index=True,
    )


@contextmanager
def media_probe_target(
    path: PathLike,
    *,
    expire_seconds: int = 3600,
) -> Iterator[str]:
    """Return a URL or local path suitable for ffmpeg/ffprobe.

    Cloud providers are asked for a signed HTTPS URL so media tools can issue
    byte-range requests without downloading the object into cloudpathlib's
    local cache. If a provider cannot sign a URL, fall back to its cached local
    path for compatibility with existing GCS usage.
    """
    if not isinstance(path, CloudPath):
        yield str(path)
        return

    try:
        target = path.as_url(presign=True, expire_seconds=expire_seconds)
    except Exception:
        try:
            yield path.fspath
        finally:
            path.clear_cache()
    else:
        yield target
