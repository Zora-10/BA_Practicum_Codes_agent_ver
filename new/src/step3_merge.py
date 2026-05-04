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

    # Exclude any pre-existing master files from the staging glob so they are not
    # re-ingested (important when output_dir == staging_dir).
    def _staging_only(paths, master_names):
        return [p for p in paths if p.name not in master_names]

    master_names = {"videos_master.parquet", "comments_master.parquet", "runs_master.parquet"}
    video_files = _staging_only(video_files, master_names)
    comment_files = _staging_only(comment_files, master_names)
    run_files = _staging_only(run_files, master_names)

    def _load_and_concat(paths, id_col: str):
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
        if id_col in combined.columns:
            combined = combined.drop_duplicates(subset=[id_col], keep="last")
        return combined.reset_index(drop=True)

    videos_master = _load_and_concat(video_files, id_col="video_id")
    comments_master = _load_and_concat(comment_files, id_col="comment_id")
    runs_master = _load_and_concat(run_files, id_col="run_id")

    videos_master.to_parquet(output_dir / "videos_master.parquet", index=False)
    comments_master.to_parquet(output_dir / "comments_master.parquet", index=False)
    runs_master.to_parquet(output_dir / "runs_master.parquet", index=False)

    return videos_master, comments_master, runs_master


def load_latest_master(output_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load master parquet files if they exist; otherwise fall back to the latest staging files."""
    if output_dir is None:
        output_dir = BASE_DIR / "data" / "collection_output"
    output_dir = Path(output_dir)

    vpath = output_dir / "videos_master.parquet"
    cpath = output_dir / "comments_master.parquet"

    if vpath.exists() and cpath.exists():
        return pd.read_parquet(vpath), pd.read_parquet(cpath)

    # No master files yet — collect from the latest staging files
    video_files = sorted(output_dir.glob("*_videos_*.parquet"))
    comment_files = sorted(output_dir.glob("*_comments_*.parquet"))

    if not video_files and not comment_files:
        return pd.DataFrame(), pd.DataFrame()

    def _load_and_concat(paths, id_col: str):
        if not paths:
            return pd.DataFrame()
        dfs = []
        for p in paths:
            try:
                dfs.append(pd.read_parquet(p))
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        if id_col in combined.columns:
            combined = combined.drop_duplicates(subset=[id_col], keep="last")
        return combined.reset_index(drop=True)

    return _load_and_concat(video_files, id_col="video_id"), _load_and_concat(comment_files, id_col="comment_id")
