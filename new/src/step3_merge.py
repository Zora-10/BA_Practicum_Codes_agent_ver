"""
Step 3: Data Merge
Merges all member staging parquet files into master videos/comments tables.
Fidelity: mirrors 03_data_merge.ipynb.
"""

import pandas as pd
from pathlib import Path
from glob import glob

from .config import BASE_DIR


def merge_staging_to_master(
    staging_dir: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all member staging parquet files and merge into master tables.
    Returns: (videos_master, comments_master, runs_master)
    """
    if staging_dir is None:
        staging_dir = BASE_DIR / "data" / "collection_output"
    if output_dir is None:
        output_dir = staging_dir  # same dir

    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(staging_dir.glob("*_videos_*.parquet"))
    comment_files = sorted(staging_dir.glob("*_comments_*.parquet"))
    run_files = sorted(staging_dir.glob("*_runs_*.parquet"))

    def _load_and_concat(paths):
        dfs = []
        for p in paths:
            try:
                df = pd.read_parquet(p)
                dfs.append(df)
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        # deduplicate by primary key
        id_col = "video_id" if "video_id" in combined.columns else "comment_id"
        if id_col in combined.columns:
            combined = combined.drop_duplicates(subset=[id_col], keep="last")
        return combined.reset_index(drop=True)

    videos_master = _load_and_concat(video_files)
    comments_master = _load_and_concat(comment_files)
    runs_master = _load_and_concat(run_files)

    videos_master.to_parquet(output_dir / "videos_master.parquet", index=False)
    comments_master.to_parquet(output_dir / "comments_master.parquet", index=False)
    runs_master.to_parquet(output_dir / "runs_master.parquet", index=False)

    return videos_master, comments_master, runs_master


def load_latest_master(output_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the most recent master parquet files."""
    if output_dir is None:
        output_dir = BASE_DIR / "data" / "collection_output"
    output_dir = Path(output_dir)

    vpath = output_dir / "videos_master.parquet"
    cpath = output_dir / "comments_master.parquet"

    videos = pd.read_parquet(vpath) if vpath.exists() else pd.DataFrame()
    comments = pd.read_parquet(cpath) if cpath.exists() else pd.DataFrame()

    return videos, comments
