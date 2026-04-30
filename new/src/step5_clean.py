"""
Step 5: Data Cleaning (Linked Data)
Cleans, filters, and enriches comment-video linked data.
Fidelity: mirrors 05_data_cleaning_linked.ipynb exactly.
"""

import re
import pandas as pd
from pathlib import Path

from .config import (
    CLEANED_DIR,
    CLEAN_EXCLUDED_TITLE_PATTERNS,
    CLEAN_RELEVANT_CATEGORIES,
    CLEAN_LOW_VALUE_PATTERNS,
    CLEAN_PRODUCT_KEYWORDS,
    DEMAND_STRONG_PATTERNS,
    DEMAND_EXCLUDE_PATTERNS,
)


# ── Text cleaning ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_excluded_title(title) -> bool:
    title_lower = str(title).lower()
    return any(re.search(pat, title_lower) for pat in CLEAN_EXCLUDED_TITLE_PATTERNS)


def has_relevant_category(cats) -> bool:
    if pd.isna(cats) or cats == "unknown":
        return False
    return any(c in cats.split("|") for c in CLEAN_RELEVANT_CATEGORIES)


def is_low_value(text: str) -> bool:
    for pat in CLEAN_LOW_VALUE_PATTERNS:
        if re.match(pat, text.lower()):
            return True
    if len(text.split()) < 2:
        return True
    return False


def contains_product_keywords(text: str) -> bool:
    return any(kw in text for kw in CLEAN_PRODUCT_KEYWORDS)


# ── Demand signal pre-detection (rule-based) ───────────────────────────────────

def detect_demand_signals(text: str) -> list[str]:
    text_lower = text.lower()
    signals = []
    for signal_type, patterns in DEMAND_STRONG_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                signals.append(signal_type)
                break
    return signals if signals else ["general"]


def should_exclude_comment(text: str) -> bool:
    text_lower = text.lower()
    for patterns in DEMAND_EXCLUDE_PATTERNS.values():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
    return False


def assign_priority(signals: list[str]) -> str:
    if "purchase_intent" in signals or "problem_complaint" in signals:
        return "high"
    if "storage_travel" in signals:
        return "medium"
    return "general"


# ── Main cleaning pipeline ───────────────────────────────────────────────────────

MIN_COMMENTS_PER_VIDEO = 5
MIN_ENGAGEMENT_RATE = 1.0


def clean_linked_data(
    linked_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Full cleaning pipeline applied to linked DataFrame.
    Returns the cleaned DataFrame.
    """
    if output_dir is None:
        output_dir = CLEANED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = linked_df.copy()
    original_len = len(df)

    # ── Step 1: Video-level pre-filtering ────────────────────────────────────

    # 1a. Remove videos with very few comments
    video_comment_counts = df.groupby("video_id").size()
    valid_videos = video_comment_counts[
        video_comment_counts >= MIN_COMMENTS_PER_VIDEO
    ].index
    df = df[df["video_id"].isin(valid_videos)]

    # 1b. Remove videos with extremely low engagement
    df = df[df["engagement_rate"] >= MIN_ENGAGEMENT_RATE]

    # 1c. Remove excluded video titles
    df = df[~df["title"].apply(is_excluded_title)]

    # 1d. Keep only relevant product categories
    df = df[df["product_categories"].apply(has_relevant_category)]

    # ── Step 2: Comment-level cleaning ──────────────────────────────────────

    # 2a. Remove duplicates
    df = df.drop_duplicates(subset=["text_original"])

    # 2b. Remove null and very short comments
    df = df[df["text_original"].notna()]
    df = df[df["text_original"].str.len() > 5]

    # 2c. Text cleaning
    df["clean_text"] = df["text_original"].apply(clean_text)

    # 2d. Remove low-value comments
    df = df[~df["clean_text"].apply(is_low_value)]

    # 2e. Filter by product-related keywords
    df = df[df["clean_text"].apply(contains_product_keywords)]

    # ── Step 3: Demand signal pre-detection ──────────────────────────────────

    # 3a. Exclude nonsense / irrelevant
    df["is_excluded"] = df["clean_text"].apply(should_exclude_comment)
    df = df[~df["is_excluded"]].drop(columns=["is_excluded"])

    # 3b. Detect demand signals
    df["demand_signals_raw"] = df["clean_text"].apply(detect_demand_signals)
    df["demand_signals"] = df["demand_signals_raw"].apply(lambda x: "|".join(x))

    # 3c. Assign priority
    df["priority_level"] = df["demand_signals_raw"].apply(assign_priority)

    # ── Step 4: Select and save output columns ──────────────────────────────

    keep_cols = [
        "comment_id", "video_id", "thread_id", "parent_comment_id", "is_reply",
        "author_display_name", "text_original", "clean_text",
        "comment_like_count", "published_at",
        "product_categories", "video_context", "demand_signals", "priority_level",
        "title", "channel_title", "description",
        "view_count", "engagement_rate", "video_url",
        "search_keyword", "fetched_by",
    ]
    output_cols = [c for c in keep_cols if c in df.columns]
    df_clean = df[output_cols].copy()

    df_clean.to_parquet(output_dir / "cleaned_comments_linked.parquet", index=False)
    df_clean.to_csv(output_dir / "cleaned_comments_linked.csv", index=False)

    summary = {
        "original_rows": original_len,
        "cleaned_rows": len(df_clean),
        "unique_videos": df_clean["video_id"].nunique() if not df_clean.empty else 0,
        "priority_dist": df_clean["priority_level"].value_counts().to_dict()
        if "priority_level" in df_clean.columns
        else {},
        "signal_dist": (
            df_clean["demand_signals"]
            .str.split("|")
            .explode()
            .value_counts()
            .to_dict()
            if "demand_signals" in df_clean.columns
            else {}
        ),
    }

    return df_clean, summary


def load_cleaned_data() -> pd.DataFrame:
    path = CLEANED_DIR / "cleaned_comments_linked.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
