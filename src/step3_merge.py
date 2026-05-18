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
    
    If master files already exist and have data, returns them directly
    instead of re-merging (which can cause issues).
    """
    if staging_dir is None:
        staging_dir = BASE_DIR / "data" / "collection_output"
    if output_dir is None:
        output_dir = staging_dir  # same dir

    staging_dir = Path(staging_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if master files already exist with data - if so, return them directly
    videos_master_path = output_dir / "videos_master.parquet"
    comments_master_path = output_dir / "comments_master.parquet"
    runs_master_path = output_dir / "runs_master.parquet"

    if videos_master_path.exists() and comments_master_path.exists():
        try:
            videos_master = pd.read_parquet(videos_master_path)
            comments_master = pd.read_parquet(comments_master_path)
            runs_master = pd.read_parquet(runs_master_path) if runs_master_path.exists() else pd.DataFrame()
            if not videos_master.empty and not comments_master.empty:
                print(f"[MERGE] Loaded existing master files: {len(videos_master)} videos, {len(comments_master)} comments")
                return videos_master, comments_master, runs_master
            else:
                print(f"[MERGE] Existing master files are empty, will re-merge from staging files")
        except Exception as e:
            print(f"[MERGE] Could not read existing master files: {e}, will re-merge from staging files")

    video_files = sorted(staging_dir.glob("videos_*.parquet"))
    comment_files = sorted(staging_dir.glob("comments_*.parquet"))
    run_files = sorted(staging_dir.glob("runs_*.parquet"))

    # Exclude master files from the list
    video_files = [f for f in video_files if "master" not in f.name]
    comment_files = [f for f in comment_files if "master" not in f.name]
    run_files = [f for f in run_files if "master" not in f.name]

    print(f"[MERGE] Found video files: {[p.name for p in video_files]}")
    print(f"[MERGE] Found comment files: {[p.name for p in comment_files]}")
    print(f"[MERGE] Found run files: {[p.name for p in run_files]}")

    # Exclude any pre-existing master files from the staging glob so they are not
    # re-ingested (important when output_dir == staging_dir).
    def _staging_only(paths, master_names):
        return [p for p in paths if p.name not in master_names]

    master_names = {"videos_master.parquet", "comments_master.parquet", "runs_master.parquet"}
    video_files = _staging_only(video_files, master_names)
    comment_files = _staging_only(comment_files, master_names)
    run_files = _staging_only(run_files, master_names)

    print(f"[MERGE] After filtering master files:")
    print(f"[MERGE]   video files to merge: {[p.name for p in video_files]}")
    print(f"[MERGE]   comment files to merge: {[p.name for p in comment_files]}")

    def _load_and_concat(paths, id_col: str):
        dfs = []
        for p in paths:
            try:
                df = pd.read_parquet(p)
                print(f"[MERGE] Loaded {p.name}: {len(df)} rows")
                dfs.append(df)
            except Exception as e:
                print(f"[WARN] Could not read {p}: {e}")
                import traceback
                traceback.print_exc()
        if not dfs:
            print("[MERGE] WARNING: No files loaded, returning empty DataFrame")
            return pd.DataFrame()
        combined = pd.concat(dfs, ignore_index=True)
        total_rows = len(combined)
        # deduplicate by primary key
        if id_col in combined.columns:
            combined = combined.drop_duplicates(subset=[id_col], keep="last")
        dupes_dropped = total_rows - len(combined)
        print(f"[MERGE] {id_col} deduplication: {total_rows} raw → {len(combined)} unique "
              f"({dupes_dropped} duplicates removed)")
        return combined.reset_index(drop=True)

    videos_master = _load_and_concat(video_files, id_col="video_id")
    comments_master = _load_and_concat(comment_files, id_col="comment_id")
    runs_master = _load_and_concat(run_files, id_col="run_id")

    # Only write master files if we actually have data
    if not videos_master.empty:
        videos_master.to_parquet(output_dir / "videos_master.parquet", index=False)
        print(f"[MERGE] Saved videos_master.parquet with {len(videos_master)} rows")
    else:
        print(f"[MERGE] WARNING: videos_master is empty, NOT overwriting existing file")
    
    if not comments_master.empty:
        comments_master.to_parquet(output_dir / "comments_master.parquet", index=False)
        print(f"[MERGE] Saved comments_master.parquet with {len(comments_master)} rows")
    else:
        print(f"[MERGE] WARNING: comments_master is empty, NOT overwriting existing file")
    
    if not runs_master.empty:
        runs_master.to_parquet(output_dir / "runs_master.parquet", index=False)
        print(f"[MERGE] Saved runs_master.parquet with {len(runs_master)} rows")
    else:
        print(f"[MERGE] WARNING: runs_master is empty, NOT overwriting existing file")

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
