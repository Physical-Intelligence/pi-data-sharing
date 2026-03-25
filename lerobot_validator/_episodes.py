import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_episodes_df(root: Any) -> Optional[pd.DataFrame]:
    """Load episodes parquet, trying chunked directory first then flat file."""
    episodes_dir = root / "meta" / "episodes"
    if episodes_dir.exists():
        try:
            return pd.read_parquet(str(episodes_dir))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", episodes_dir, exc)
            return None

    episodes_file = root / "meta" / "episodes.parquet"
    if episodes_file.exists():
        try:
            return pd.read_parquet(str(episodes_file))
        except Exception as exc:
            logger.warning("Failed to read %s: %s", episodes_file, exc)
            return None

    return None


def video_indices(
    episodes_df: Optional[pd.DataFrame],
    ep_idx: int,
    vkey: str,
    info: Dict[str, Any],
) -> Tuple[int, int]:
    """Falls back to chunks_size heuristic for datasets missing per-video columns."""
    chunks_size = info.get("chunks_size", 1000)
    fallback = (ep_idx // chunks_size, ep_idx)

    if episodes_df is None or "episode_index" not in episodes_df.columns:
        return fallback

    chunk_col = f"videos/{vkey}/chunk_index"
    file_col = f"videos/{vkey}/file_index"
    if chunk_col not in episodes_df.columns or file_col not in episodes_df.columns:
        return fallback

    row = episodes_df[episodes_df["episode_index"] == ep_idx]
    if row.empty:
        return fallback

    return int(row.iloc[0][chunk_col]), int(row.iloc[0][file_col])
