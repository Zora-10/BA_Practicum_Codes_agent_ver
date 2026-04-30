"""
Pipeline module: exposes the full end-to-end run function.
"""

from pathlib import Path
from typing import Optional

from .step1_keywords import get_all_queries_flat
from .step2_collect import run_collection, QuotaExceededException
from .step3_merge import merge_staging_to_master
from .step4_link import link_comments_to_videos, load_linked_data
from .step5_clean import clean_linked_data, load_cleaned_data
from .step6_demand_signal import run_demand_signal_detection, load_latest_demand_signals


def run_full_pipeline(
    api_key: str,
    youtube_api_key: str,
    *,
    # Step 2 config
    days_back: int = 30,
    search_max_pages: int = 2,
    comment_max_pages: int = 5,
    search_results_per_page: int = 50,
    comments_per_page: int = 100,
    region_code: str = "US",
    relevance_language: str = "en",
    fetch_comments: bool = True,
    # Step 6 config
    llm_batch_size: int = 20,
    llm_rate_delay: float = 1.5,
    llm_model: str = "deepseek-chat",
    # Output
    output_dir: Optional[Path] = None,
    progress_callback=None,
) -> dict:
    """
    Run the entire pipeline from keyword generation → collection → linking → cleaning → LLM classification.

    Returns a dict with pipeline results and statistics.
    """
    if output_dir is None:
        from .config import BASE_DIR
        output_dir = BASE_DIR / "data"

    collection_dir = output_dir / "collection_output"
    linked_dir = output_dir / "linked_data"
    cleaned_dir = output_dir / "cleaned_data"

    # ── Step 1: Keywords ────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("step", 1, "Generating keywords...")
    keywords = get_all_queries_flat()

    # ── Step 2: Collection ────────────────────────────────────────────────
    if progress_callback:
        progress_callback("step", 2, "Collecting YouTube data...")
    videos_df, comments_df, runs_df, quota_exceeded = run_collection(
        api_key=youtube_api_key,
        keyword_shard=keywords,
        days_back=days_back,
        search_max_pages=search_max_pages,
        comment_max_pages=comment_max_pages,
        search_results_per_page=search_results_per_page,
        comments_per_page=comments_per_page,
        region_code=region_code,
        relevance_language=relevance_language,
        fetch_comments=fetch_comments,
        output_dir=collection_dir,
    )

    # ── Step 3: Merge ──────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("step", 3, "Merging data...")
    merge_staging_to_master(
        staging_dir=collection_dir,
        output_dir=collection_dir,
    )
    videos_master, comments_master = (
        (videos_df, comments_df) if quota_exceeded
        else (None, None)
    )

    # ── Step 4: Link ───────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("step", 4, "Linking comments to videos...")
    if videos_master is None or comments_master is None:
        from .step3_merge import load_latest_master
        videos_master, comments_master = load_latest_master(collection_dir)

    linked_df = link_comments_to_videos(
        comments=comments_master,
        videos=videos_master,
        output_dir=linked_dir,
    )

    # ── Step 5: Clean ──────────────────────────────────────────────────────
    if progress_callback:
        progress_callback("step", 5, "Cleaning data...")
    cleaned_df, clean_summary = clean_linked_data(
        linked_df=linked_df,
        output_dir=cleaned_dir,
    )

    # ── Step 6: LLM Classification ─────────────────────────────────────────
    if progress_callback:
        progress_callback("step", 6, "Running LLM demand signal detection...")
    full_df, signal_df = run_demand_signal_detection(
        api_key=api_key,
        input_df=cleaned_df,
        batch_size=llm_batch_size,
        rate_limit_delay=llm_rate_delay,
        model=llm_model,
    )

    return {
        "keywords_generated": len(keywords),
        "videos_collected": len(videos_df),
        "comments_collected": len(comments_df),
        "quota_exceeded": quota_exceeded,
        "linked_rows": len(linked_df),
        "cleaned_rows": len(cleaned_df),
        "demand_signals_found": len(signal_df),
        "signal_breakdown": signal_df["signal"].value_counts().to_dict()
        if not signal_df.empty and "signal" in signal_df.columns
        else {},
        "output_dir": str(output_dir),
    }
