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

    # Select relevant video columns
    video_cols = [
        "video_id", "channel_title", "title", "description", "tags",
        "category_id", "default_language", "video_published_at",
        "view_count", "like_count", "comment_count",
    ]
    videos_subset = videos[video_cols].drop_duplicates(subset=["video_id"])

    # Merge
    linked = comments.merge(
        videos_subset,
        on="video_id",
        how="left",
        suffixes=("_comment", "_video"),
    )
    linked = linked.rename(columns={
        "like_count_comment": "comment_like_count",
        "like_count_video": "video_like_count",
        "comment_count": "video_comment_count",
    })

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
    return pd.read_parquet(path)


def load_video_summary() -> pd.DataFrame:
    path = LINKED_DIR / "video_summary.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
