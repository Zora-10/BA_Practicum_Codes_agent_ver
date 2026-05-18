"""
Step 4: Comment-Video Linking
Links comments with video metadata and classifies product categories.
Fidelity: mirrors 04_comment_video_link.ipynb exactly.
"""

import pandas as pd
from pathlib import Path

from .config import LINKED_DIR, LINK_CATEGORY_KEYWORDS


def classify_product_category(row) -> str:
    """Classify product category based on video title and description."""
    text = f"{row.get('title', '')} {row.get('description', '')}".lower()
    matched = []
    for category, keywords in LINK_CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                matched.append(category)
                break
    return "|".join(matched) if matched else "unknown"


def extract_video_context(row) -> str:
    """Extract content type / context from video title."""
    title = str(row.get("title", "")).lower()
    context = []
    if any(x in title for x in ["unboxing", "first look", "unbox"]):
        context.append("unboxing")
    if any(x in title for x in ["review", "honest review", "tested"]):
        context.append("review")
    if any(x in title for x in ["comparison", " vs ", "versus", "better than"]):
        context.append("comparison")
    if any(x in title for x in ["tutorial", "how to", "guide", "setup"]):
        context.append("tutorial")
    if any(x in title for x in ["haul", "what i bought", "shopping"]):
        context.append("haul")
    if any(x in title for x in ["storage", "organization", "organize", "pack"]):
        context.append("organization")
    return "|".join(context) if context else "general"


def link_comments_to_videos(
    comments: pd.DataFrame,
    videos: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Main linking pipeline: merge comments + video metadata,
    classify product categories, extract video context, calculate engagement.
    Returns the linked DataFrame.
    """
    if output_dir is None:
        output_dir = LINKED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate input data
    if videos.empty:
        raise ValueError("Videos DataFrame is empty. Please check your collection data.")
    if comments.empty:
        raise ValueError("Comments DataFrame is empty. Please check your collection data.")
    if "video_id" not in videos.columns:
        raise ValueError(f"'video_id' not found in videos columns: {videos.columns.tolist()}")
    if "video_id" not in comments.columns:
        raise ValueError(f"'video_id' not found in comments columns: {comments.columns.tolist()}")
    print(f"[LINK] Starting with {len(videos)} videos, {len(comments)} comments")

    # Select relevant video columns (only those that exist)
    video_cols = [
        "video_id", "channel_title", "title", "description", "tags",
        "category_id", "default_language", "video_published_at",
        "view_count", "like_count", "comment_count",
    ]
    # Only select columns that exist in the DataFrame
    existing_cols = [c for c in video_cols if c in videos.columns]
    if not existing_cols:
        raise ValueError(f"No expected video columns found. Available columns: {videos.columns.tolist()}")
    videos_subset = videos[existing_cols].drop_duplicates(subset=["video_id"])
    print(f"[LINK] Using video columns: {existing_cols}")

    # Merge
    linked = comments.merge(
        videos_subset,
        on="video_id",
        how="left",
        suffixes=("_comment", "_video"),
    )
    
    # Safely rename columns if they exist
    if "like_count_comment" in linked.columns:
        linked = linked.rename(columns={"like_count_comment": "comment_like_count"})
    if "like_count_video" in linked.columns:
        linked = linked.rename(columns={"like_count_video": "video_like_count"})
    if "comment_count" in linked.columns:
        linked = linked.rename(columns={"comment_count": "video_comment_count"})

    # Product category classification
    linked["product_categories"] = linked.apply(classify_product_category, axis=1)

    # Video context tags
    linked["video_context"] = linked.apply(extract_video_context, axis=1)

    # Engagement rate
    linked["engagement_rate"] = (
        (linked["video_like_count"].fillna(0) + linked["video_comment_count"].fillna(0))
        / linked["view_count"].replace(0, 1)
        * 1000
    ).round(2)

    # YouTube video URL
    linked["video_url"] = "https://www.youtube.com/watch?v=" + linked["video_id"].astype(str)

    # Save
    linked.to_parquet(output_dir / "comments_video_linked.parquet", index=False)
    linked.to_csv(output_dir / "comments_video_linked.csv", index=False)

    # Video summary
    _save_video_summary(linked, output_dir)

    return linked


def _save_video_summary(linked: pd.DataFrame, output_dir: Path):
    video_summary = (
        linked.groupby("video_id")
        .agg({
            "comment_id": "count",
            "title": "first",
            "channel_title": "first",
            "product_categories": "first",
            "video_context": "first",
            "view_count": "first",
            "video_like_count": "first",
            "video_comment_count": "first",
            "engagement_rate": "mean",
            "published_at": "min",
            "video_url": "first",
        })
        .reset_index()
        .rename(columns={
            "comment_id": "comment_count",
            "published_at": "first_comment_date",
        })
        .sort_values("comment_count", ascending=False)
    )
    video_summary.to_parquet(output_dir / "video_summary.parquet", index=False)
    return video_summary


def load_linked_data() -> pd.DataFrame:
    path = LINKED_DIR / "comments_video_linked.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # Guard against stale parquet files missing columns required by downstream steps
    required_cols = {"product_categories", "video_context", "engagement_rate"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[WARN] Linked data is missing columns {missing} — treating as empty.")
        return pd.DataFrame()
    return df


def load_video_summary() -> pd.DataFrame:
    path = LINKED_DIR / "video_summary.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
