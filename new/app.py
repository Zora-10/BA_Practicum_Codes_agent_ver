"""
Market Signal AI Agent Platform
================================
Streamlit dashboard for the YouTube demand signal detection pipeline.

User flow:
  Step 1 → Step 2 → Step 3 → Step 4 → Step 5 → Step 6
Each step is a page in the sidebar navigation.
"""

from __future__ import annotations

import sys
import time
import datetime
import io
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Page must be the first Streamlit command
st.set_page_config(
    page_title="Market Signal Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Add src to path ────────────────────────────────────────────────────────────
from pathlib import Path

NEW_DIR = Path(__file__).resolve().parent
SRC_DIR = NEW_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NEW_DIR))

from src.config import BASE_DIR
from src.step1_keywords import get_all_queries_flat, generate_all, generate_member_queries, generate_leader_queries
from src.step2_collect import run_collection, load_latest_collected, QuotaExceededException
from src.step3_merge import merge_staging_to_master, load_latest_master
from src.step4_link import link_comments_to_videos, load_linked_data, load_video_summary
from src.step5_clean import clean_linked_data, load_cleaned_data
from src.step6_demand_signal import run_demand_signal_detection, load_latest_demand_signals, QuotaExceededError, APIError, LLMCallError

# ── Session state defaults ─────────────────────────────────────────────────────
defaults = {
    "step1_keywords": [],
    "step1_done": False,
    "step2_videos": pd.DataFrame(),
    "step2_comments": pd.DataFrame(),
    "step2_runs": pd.DataFrame(),
    "step2_done": False,
    "step3_videos": pd.DataFrame(),
    "step3_comments": pd.DataFrame(),
    "step3_done": False,
    "step4_linked": pd.DataFrame(),
    "step4_done": False,
    "step5_cleaned": pd.DataFrame(),
    "step5_summary": {},
    "step5_done": False,
    "step6_full": pd.DataFrame(),
    "step6_signals": pd.DataFrame(),
    "step6_done": False,
    "running": False,
    "step2_quota_exceeded": False,
    "collection_started": False,
    "llm_started": False,
    "llm_error": None,
    "llm_error_kind": "error",
    "_flash": None,
}

for key, val in defaults.items():
    st.session_state.setdefault(key, val)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #f8f9fa; border-right: 1px solid #dee2e6; }

    /* Step headers */
    .step-header { font-size: 1.4rem; font-weight: 700; color: #1a73e8; margin-bottom: 0.5rem; }
    .step-subheader { font-size: 0.95rem; color: #6c757d; margin-bottom: 1rem; }

    /* Status boxes */
    .status-box { padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem; font-family: monospace; }
    .status-done    { background: #d4edda; border: 1px solid #28a745; color: #155724; }
    .status-running { background: #fff3cd; border: 1px solid #ffc107; color: #856404; }
    .status-warn    { background: #f8d7da; border: 1px solid #dc3545; color: #721c24; }
    .status-empty   { background: #f8f9fa; border: 1px solid #dee2e6; color: #6c757d; }

    /* Metric cards */
    div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #dee2e6; border-radius: 8px; padding: 1rem; }

    /* Expander styling */
    details { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 0.5rem; }

    /* Dataframe */
    .dataframe { font-size: 0.8rem; }

    /* Download button */
    .stDownloadButton>button { background: #1a73e8; color: white; border: none; font-weight: 600; }
    .stDownloadButton>button:hover { background: #1557b0; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    return buf.getvalue()


def download_button(label: str, df: pd.DataFrame, default_name: str, key: str):
    """Download button with CSV/Parquet format selector; CSV is generated on demand."""
    col_fmt, col_dl = st.columns([1, 3])
    with col_fmt:
        fmt = st.selectbox(
            "Format", ["CSV", "Parquet"], key=f"dl_fmt_{key}",
            label_visibility="collapsed",
        )
    with col_dl:
        base_name = default_name.rsplit(".", 1)[0]
        if fmt == "CSV":
            st.download_button(
                f"{label} ({fmt})",
                data=df_to_csv_bytes(df),
                file_name=f"{base_name}.csv",
                mime="text/csv",
                key=f"dl_btn_{key}",
            )
        else:
            st.download_button(
                f"{label} ({fmt})",
                data=df_to_parquet_bytes(df),
                file_name=f"{base_name}.parquet",
                mime="application/octet-stream",
                key=f"dl_btn_{key}",
            )


def status_box(text: str, status: str = "empty"):
    icons = {"done": "✅", "running": "⏳", "warn": "⚠️", "empty": "○"}
    st.markdown(
        f'<div class="status-box status-{status}">{icons.get(status, "○")} {text}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value, delta: str | None = None):
    st.metric(label=label, value=value, delta=delta)


def section_divider():
    st.markdown("<hr style='border-color:#dee2e6; margin: 1rem 0;'>", unsafe_allow_html=True)


def flash(msg: str, kind: str = "success"):
    """Store a one-time flash message in session state (shown on next page load)."""
    st.session_state._flash = (kind, msg)


def show_flash():
    """Display and clear any pending flash message."""
    if st.session_state._flash:
        kind, msg = st.session_state._flash
        st.session_state._flash = None
        if kind == "success":
            st.success(msg)
        elif kind == "warning":
            st.warning(msg)
        elif kind == "error":
            st.error(msg)
        else:
            st.info(msg)


def page_nav():
    """Render sidebar navigation and return selected page."""
    PAGES = [
        "🏠 Overview",
        "🔑 Step 1: Keywords",
        "📡 Step 2: Collection",
        "🔗 Step 3: Merge",
        "📊 Step 4: Link",
        "🧹 Step 5: Clean",
        "🤖 Step 6: LLM Signals",
        "📈 Results Dashboard",
    ]

    st.sidebar.markdown("## 📋 Navigation")
    selected = st.sidebar.radio("Navigation", PAGES, index=0, label_visibility="collapsed")
    st.sidebar.markdown("---")
    return selected


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════

def render_overview():
    st.markdown("## 🏠 Platform Overview")
    st.markdown(
        "Welcome to the **Market Signal AI Agent** — an end-to-end YouTube demand signal "
        "detection platform for **hard case / protective case** products.\n\n"
        "This platform transforms raw YouTube data into actionable demand insights using "
        "6 automated pipeline stages."
    )

    section_divider()

    # Pipeline flow diagram
    st.markdown("### Pipeline Architecture")
    steps = [
        ("🔑", "Keywords", "Generate search queries\nacross 18 product categories"),
        ("📡", "Collection", "Fetch videos + comments\nfrom YouTube Data API"),
        ("🔗", "Merge", "Deduplicate & merge\nall team members' data"),
        ("📊", "Link", "Attach video metadata\nto every comment"),
        ("🧹", "Clean", "Filter noise, classify\npriority & signal types"),
        ("🤖", "LLM Signals", "DeepSeek AI classifies\ndemand signals"),
        ("📈", "Dashboard", "Interactive results\nvisualization & export"),
    ]

    cols = st.columns(len(steps))
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"**{icon} {title}**")
            for line in desc.split("\n"):
                st.caption(line)
            if i < len(steps) - 1:
                st.markdown("→", unsafe_allow_html=False)

    section_divider()

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    # Check available data
    linked = load_linked_data()
    cleaned = load_cleaned_data()
    _, signals = load_latest_demand_signals()

    with col1:
        metric_card("Keywords Generated", len(get_all_queries_flat()))
    with col2:
        metric_card("Linked Comments", len(linked), f"{linked['video_id'].nunique() if not linked.empty else 0} videos")
    with col3:
        metric_card("Cleaned Comments", len(cleaned), f"{cleaned['video_id'].nunique() if not cleaned.empty else 0} videos")
    with col4:
        metric_card("Demand Signals", len(signals), f"{signals['signal'].nunique() if not signals.empty else 0} types")

    section_divider()

    st.markdown("### Getting Started")
    st.info(
        "1. Navigate through each **Step** using the sidebar\n"
        "2. Each step processes the output from the previous step\n"
        "3. View final results in the **Results Dashboard**"
    )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 1 — KEYWORDS
# ════════════════════════════════════════════════════════════════════════════════

def render_step1_keywords():
    st.markdown("## 🔑 Step 1: Keyword Generation")
    st.markdown(
        "Generate YouTube search queries for **hard case / protective case** demand signals. "
        "Queries cover **18 product categories** across protection, storage, usage, and problem scenarios.\n\n"
    )

    if st.button("▶ Generate Keywords", type="primary"):
        with st.spinner("Generating queries..."):
            all_df, summary = generate_all()
            all_q = all_df["query"].tolist()
            st.session_state.step1_keywords = all_q
            st.session_state.step1_done = True
        st.success(f"Generated {len(all_q):,} keywords across {len(summary['by_category'])} categories!")

    keywords = st.session_state.step1_keywords

    # Always read from session state if done — don't re-run
    if not keywords and st.session_state.step1_done:
        st.success("Keywords already generated.")
        st.stop()

    if not keywords:
        try:
            all_df, summary = generate_all()
            keywords = all_df["query"].tolist()
            st.session_state.step1_keywords = keywords
        except Exception as e:
            st.warning(f"Run 'Generate Keywords' to produce the query list. ({e})")
            return

    st.markdown(f"**Total keywords: {len(keywords):,}**")

    tab1, tab2, tab3 = st.tabs(["📋 All Keywords", "📂 By Category", "🏷️ By Type"])

    with tab1:
        all_df = pd.DataFrame({"query": keywords})
        st.dataframe(all_df, width='stretch', height=400)
        download_button("⬇ Download All Keywords", all_df, "all_keywords", "s1_all")

    with tab2:
        member_df = generate_member_queries()
        leader_df = generate_leader_queries()
        leader_df = leader_df.rename(columns={"type": "type_orig"}).assign(type="discovery", category="discovery", object="discovery")
        combined = pd.concat([member_df, leader_df], ignore_index=True)

        cat_counts = combined.groupby("category").size().sort_values(ascending=False)
        _y_col = cat_counts.reset_index().columns[1]
        fig = px.bar(
            cat_counts.reset_index(),
            x="category",
            y=_y_col,
            title="Keywords per Category",
            labels={_y_col: "Count", "category": "Category"},
            color=_y_col,
            color_continuous_scale="Blues",
        )
        fig.update_layout(template="plotly_white", height=400, xaxis_tickangle=-45)
        st.plotly_chart(fig, width='stretch')

        cat_sel = st.selectbox("Select category to view queries:", cat_counts.index.tolist())
        subset = combined[combined["category"] == cat_sel][["query", "type"]]
        st.dataframe(subset, width='stretch', height=300)
        download_button(f"⬇ Download {cat_sel} Keywords", subset, f"keywords_{cat_sel}", "s1_cat")

    with tab3:
        member_df = generate_member_queries()
        leader_df = generate_leader_queries().assign(category="discovery")
        combined = pd.concat([member_df, leader_df], ignore_index=True)
        type_counts = combined["type"].value_counts()
        fig2 = px.pie(
            type_counts.reset_index(),
            values=type_counts.reset_index().columns[1],
            names="type",
            title="Keywords by Query Type",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig2, width='stretch')

        for qtype, count in type_counts.items():
            st.caption(f"**{qtype}**: {count:,} keywords")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 2 — COLLECTION
# ════════════════════════════════════════════════════════════════════════════════

def render_step2_collection():
    st.markdown("## 📡 Step 2: YouTube Data Collection")
    st.markdown(
        "Search YouTube for videos matching the generated keywords and scrape comments. "
        "**Quota-aware** — will stop gracefully when API quota is exceeded and resume on next run."
    )

    # Status
    if st.session_state.step2_quota_exceeded:
        status_box("Quota exceeded — collection paused. Will resume from checkpoint on next run.", "warn")
    elif st.session_state.collection_started and not st.session_state.running:
        status_box("Collection complete — data ready for Step 3.", "done")
    elif st.session_state.running:
        status_box("Collection in progress...", "running")
    else:
        status_box("Ready to collect — configure settings below and click Start.", "empty")

    # ── Configuration ────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Collection Settings")
    with st.expander("Configure YouTube API & Search Parameters", expanded=True):
        youtube_key = st.text_input(
            "🔑 YouTube Data API Key",
            type="password",
            help="Get a key from https://console.cloud.google.com/apis/credentials",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            days_back = st.number_input("Days to search back", min_value=1, max_value=365, value=30)
            search_max_pages = st.number_input("Search result pages per keyword", min_value=1, max_value=10, value=2)
            comment_max_pages = st.number_input("Comment pages per video", min_value=1, max_value=20, value=5)
        with col_b:
            search_results_per_page = st.slider("Search results per page", 10, 50, 50)
            comments_per_page = st.slider("Comments per page", 10, 100, 100)
            region_code = st.text_input("Region code (e.g. US)", value="US").strip() or None
            relevance_language = st.text_input("Relevance language (e.g. en)", value="en").strip() or None

        fetch_comments = st.checkbox("Fetch comments (recommended)", value=True)

    # ── Start / Stop ─────────────────────────────────────────────────────────
    col_start, col_clear = st.columns([1, 1])
    with col_start:
        if not youtube_key:
            st.warning("Enter your YouTube API key above to begin collection.")
        else:
            if st.button("🚀 Start Collection", type="primary", disabled=st.session_state.running):
                keywords = st.session_state.step1_keywords or get_all_queries_flat()
                st.session_state.running = True
                st.session_state.collection_started = True

                output_dir = BASE_DIR / "data" / "collection_output"
                safe_mkdir(output_dir)

                progress_bar = st.progress(0)
                status_text = st.empty()
                video_count = st.empty()
                comment_count = st.empty()

                def progress_cb(idx, total, kw):
                    pct = int(100 * (idx + 1) / total)
                    progress_bar.progress(pct)
                    status_text.text(f"[{idx+1}/{total}] {kw}")

                with st.spinner("Collecting YouTube data (this may take a while)..."):
                    videos_df, comments_df, runs_df, quota_exceeded = run_collection(
                        api_key=youtube_key,
                        keyword_shard=keywords,
                        days_back=days_back,
                        search_max_pages=search_max_pages,
                        comment_max_pages=comment_max_pages,
                        search_results_per_page=search_results_per_page,
                        comments_per_page=comments_per_page,
                        region_code=region_code,
                        relevance_language=relevance_language,
                        fetch_comments=fetch_comments,
                        output_dir=output_dir,
                        progress_callback=progress_cb,
                    )

                st.session_state.running = False
                st.session_state.step2_videos = videos_df
                st.session_state.step2_comments = comments_df
                st.session_state.step2_runs = runs_df
                st.session_state.step2_quota_exceeded = quota_exceeded
                st.session_state.step2_done = True

                progress_bar.empty()
                status_text.empty()

                if quota_exceeded:
                    st.warning("⚠️ API quota exceeded. Progress saved — resume after quota resets.")
                else:
                    st.success("✅ Collection complete!")

    with col_clear:
        if st.button("🗑 Clear Collection Data"):
            for k in ["step2_videos", "step2_comments", "step2_runs", "collection_started", "step2_quota_exceeded", "step2_done"]:
                st.session_state[k] = defaults[k]
            st.info("Collection data cleared.")

    section_divider()

    # ── Results preview ──────────────────────────────────────────────────────
    st.markdown("### 📋 Collection Results")

    videos = st.session_state.step2_videos
    comments = st.session_state.step2_comments
    runs = st.session_state.step2_runs

    # Always try disk if session state is empty
    if videos.empty:
        videos_disk, comments_disk, runs_disk = load_latest_collected(BASE_DIR / "data" / "collection_output")
        if not videos_disk.empty:
            videos = videos_disk
            comments = comments_disk
            runs = runs_disk

    if not videos.empty:
        videos["video_url"] = "https://www.youtube.com/watch?v=" + videos["video_id"].astype(str)
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Videos Collected", len(videos))
        with c2: metric_card("Comments Collected", len(comments))
        with c3: metric_card("Unique Videos", videos["video_id"].nunique())
        with c4: metric_card("Channels", videos["channel_title"].nunique())

        tab_v, tab_c, tab_r = st.tabs(["🎬 Videos", "💬 Comments", "📜 Run Log"])
        with tab_v:
            show_cols = ["video_id", "title", "video_url", "channel_title", "view_count", "like_count", "comment_count", "video_published_at"]
            show = videos[[c for c in show_cols if c in videos.columns]].head(50)
            st.dataframe(show, width='stretch', height=400)
            download_button("⬇ Download Videos", videos, "collected_videos", "s2_vid")
        with tab_c:
            show_cols = ["comment_id", "video_id", "author_display_name", "text_original", "like_count", "published_at"]
            show = comments[[c for c in show_cols if c in comments.columns]].head(50)
            st.dataframe(show, width='stretch', height=400)
            download_button("⬇ Download Comments", comments, "collected_comments", "s2_cmt")
        with tab_r:
            if not runs.empty:
                st.dataframe(runs, width='stretch', height=400)
    else:
        st.info("No collection data yet. Enter your YouTube API key and click **Start Collection**.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 3 — MERGE
# ════════════════════════════════════════════════════════════════════════════════

def render_step3_merge():
    st.markdown("## 🔗 Step 3: Data Merge")
    st.markdown(
        "Merge all collected staging parquet files into unified **master tables** "
        "(videos + comments). Deduplication is applied automatically.\n\n"
    )

    collection_dir = BASE_DIR / "data" / "collection_output"
    safe_mkdir(collection_dir)

    # Always try to load master data from disk first
    master_videos, master_comments = load_latest_master(collection_dir)
    display_videos = master_videos
    display_comments = master_comments

    # "Re-run" path: use session-state data (from Step 2) as input if available
    session_videos = st.session_state.step3_videos if not st.session_state.step3_videos.empty else pd.DataFrame()
    session_comments = st.session_state.step3_comments if not st.session_state.step3_comments.empty else pd.DataFrame()

    has_disk_data = not display_videos.empty

    # ── Run / Re-run button ─────────────────────────────────────────────────────
    btn_label = "🔄 Re-run Merge" if has_disk_data else "🚀 Run Merge"
    if st.button(btn_label, type="primary"):
        # ── FIX: always scan ALL staging files (from every team member run) ──
        # `merge_staging_to_master` globs `*_videos_*.parquet` / `*_comments_*.parquet`
        # inside collection_dir and concatenates + dedupes them into master tables.
        with st.spinner("Scanning staging files from all members..."):
            run_videos, run_comments, _ = merge_staging_to_master(
                staging_dir=collection_dir,
                output_dir=collection_dir,
            )
        with st.spinner("Merging and deduplicating..."):
            if not run_videos.empty:
                run_videos = run_videos.drop_duplicates(subset=["video_id"], keep="last")
            if not run_comments.empty:
                run_comments = run_comments.drop_duplicates(subset=["comment_id"], keep="last")
            run_videos.to_parquet(collection_dir / "videos_master.parquet", index=False)
            run_comments.to_parquet(collection_dir / "comments_master.parquet", index=False)
            st.session_state.step3_videos = run_videos
            st.session_state.step3_comments = run_comments
            st.session_state.step3_done = True
        flash(f"✅ Merge complete! {len(run_videos):,} videos, {len(run_comments):,} comments saved.", "success")
        st.rerun()

    section_divider()

    # ── Results preview ──────────────────────────────────────────────────────────
    st.markdown("### 📋 Merged Master Data")

    if has_disk_data:
        display_videos["video_url"] = "https://www.youtube.com/watch?v=" + display_videos["video_id"].astype(str)
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Master Videos", len(display_videos))
        with c2: metric_card("Master Comments", len(display_comments))
        with c3: metric_card("Channels", display_videos["channel_title"].nunique())

        t1, t2 = st.tabs(["🎬 Videos", "💬 Comments"])
        with t1:
            cols = ["video_id", "title", "video_url", "channel_title", "view_count", "like_count", "comment_count"]
            st.dataframe(display_videos[[c for c in cols if c in display_videos.columns]].head(20), width='stretch', height=400)
            download_button("⬇ Download Master Videos", display_videos, "videos_master", "s3_vid")
        with t2:
            st.dataframe(display_comments.head(20), width='stretch', height=400)
            download_button("⬇ Download Master Comments", display_comments, "comments_master", "s3_cmt")
    else:
        st.info(
            "No master data found on disk. "
            "Collect data in **Step 2**, then click **Run Merge** above."
        )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 4 — LINK
# ════════════════════════════════════════════════════════════════════════════════

def render_step4_link():
    st.markdown("## 📊 Step 4: Comment-Video Linking")
    st.markdown(
        "Attach YouTube video metadata (title, description, channel, stats) to every comment. "
        "Automatically classifies **product categories** and extracts **video context tags**.\n\n"
    )

    linked_dir = BASE_DIR / "data" / "linked_data"
    collection_dir = BASE_DIR / "data" / "collection_output"
    safe_mkdir(linked_dir)

    # Always try to load linked data from disk first
    linked = load_linked_data()
    has_disk_data = not linked.empty

    # "Re-run" path: source data from session state or load from disk
    session_videos = st.session_state.step3_videos if not st.session_state.step3_videos.empty else pd.DataFrame()
    session_comments = st.session_state.step3_comments if not st.session_state.step3_comments.empty else pd.DataFrame()

    if not has_disk_data:
        # No disk data — check if we have any source to link from
        src_videos = session_videos if not session_videos.empty else pd.DataFrame()
        src_comments = session_comments if not session_comments.empty else pd.DataFrame()
        if src_videos.empty or src_comments.empty:
            src_videos, src_comments = load_latest_master(collection_dir)
        if not src_videos.empty:
            st.caption(f"Source ready: {len(src_videos):,} videos | {len(src_comments):,} comments")

    # ── Run / Re-run button ─────────────────────────────────────────────────────
    btn_label = "🔄 Re-link Comments" if has_disk_data else "🔗 Link Comments to Videos"
    if st.button(btn_label, type="primary"):
        # Initialize from session state, fall back to disk
        src_videos = session_videos if not session_videos.empty else pd.DataFrame()
        src_comments = session_comments if not session_comments.empty else pd.DataFrame()
        if src_videos.empty or src_comments.empty:
            src_videos, src_comments = load_latest_master(collection_dir)

        if src_videos.empty or src_comments.empty:
            flash("No source data found. Collect data in Steps 2–3 first.", "warning")
        else:
            with st.spinner("Linking comments with video metadata..."):
                linked_df = link_comments_to_videos(
                    comments=src_comments,
                    videos=src_videos,
                    output_dir=linked_dir,
                )
                st.session_state.step4_linked = linked_df
                st.session_state.step4_done = True
            flash(f"✅ Linked {len(linked_df):,} comments!", "success")
            st.rerun()

    section_divider()

    # ── Results preview ──────────────────────────────────────────────────────────
    if not linked.empty:
        st.caption(f"Showing {len(linked):,} linked comments from disk.")

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Linked Comments", len(linked))
        with c2: metric_card("Unique Videos", linked["video_id"].nunique())
        with c3: metric_card("Channels", linked["channel_title"].nunique())
        with c4: metric_card("Avg Engagement", f"{linked['engagement_rate'].mean():.2f}")

        # Category distribution
        st.markdown("#### Product Category Distribution")
        if "product_categories" in linked.columns:
            cat_exploded = linked["product_categories"].str.split("|").explode()
            cat_counts = cat_exploded[cat_exploded != "unknown"].value_counts()
            _y_col = cat_counts.reset_index().columns[1]
            fig = px.bar(
                cat_counts.reset_index(),
                x="product_categories",
                y=_y_col,
                title="Comments per Product Category",
                labels={_y_col: "Count", "product_categories": "Category"},
                color=_y_col,
                color_continuous_scale="Viridis",
            )
            fig.update_layout(template="plotly_white", height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, width='stretch')

        # Video context
        st.markdown("#### Video Context Distribution")
        if "video_context" in linked.columns:
            ctx_exploded = linked["video_context"].str.split("|").explode()
            ctx_counts = ctx_exploded[ctx_exploded != "general"].value_counts()
            fig2 = px.pie(
                ctx_counts.reset_index(),
                values=ctx_counts.reset_index().columns[1],
                names="video_context",
                title="Video Context Tags",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig2.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig2, width='stretch')

        section_divider()

        st.markdown("#### Sample Linked Data")
        cols = ["comment_id", "video_id", "title", "video_url", "channel_title",
                "view_count", "video_like_count", "video_comment_count",
                "product_categories", "video_context", "engagement_rate"]
        st.dataframe(linked[[c for c in cols if c in linked.columns]].head(20), width='stretch', height=400)
        download_button("⬇ Download Linked Data", linked, "comments_video_linked", "s4_link")

        # Video summary
        st.markdown("#### Top Videos by Comment Count")
        video_summary = load_video_summary()
        if not video_summary.empty:
            st.dataframe(
                video_summary[["title", "channel_title", "comment_count", "view_count", "engagement_rate"]].head(20),
                width='stretch',
                height=400,
            )
    else:
        st.info(
            "No linked data found on disk. "
            "Collect data in **Steps 2–3**, then click **Link Comments to Videos** above."
        )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 5 — CLEAN
# ════════════════════════════════════════════════════════════════════════════════

def render_step5_clean():
    st.markdown("## 🧹 Step 5: Data Cleaning")
    st.markdown(
        "Apply multi-stage filtering:\n"
        "- Remove low-engagement videos, excluded categories (phone cases, fashion, etc.)\n"
        "- Deduplicate & clean text (URLs, non-ASCII, special characters)\n"
        "- Filter low-value comments (single words, spam patterns)\n"
        "- Rule-based demand signal pre-detection & priority assignment\n\n"
    )

    linked_dir = BASE_DIR / "data" / "linked_data"
    cleaned_dir = BASE_DIR / "data" / "cleaned_data"
    safe_mkdir(cleaned_dir)

    # Always try to load cleaned data from disk first
    cleaned = load_cleaned_data()
    has_disk_data = not cleaned.empty

    # Source linked data for display when we have disk data
    linked = st.session_state.step4_linked if not st.session_state.step4_linked.empty else load_linked_data()

    # ── Run / Re-run button ─────────────────────────────────────────────────────
    if not has_disk_data and linked.empty:
        master_videos, master_comments = load_latest_master(BASE_DIR / "data" / "collection_output")
        if not master_videos.empty:
            st.caption(f"Source ready: {len(master_comments):,} comments from master data.")

    btn_label = "🔄 Re-run Cleaning" if has_disk_data else "🧹 Run Cleaning"
    if st.button(btn_label, type="primary"):
        if linked.empty:
            flash("No linked data found. Complete Steps 2–4 first.", "warning")
        else:
            with st.spinner("Cleaning and filtering..."):
                try:
                    cleaned_df, summary = clean_linked_data(linked_df=linked, output_dir=cleaned_dir)
                except ValueError as e:
                    flash(str(e), "warning")
                    cleaned_df, summary = pd.DataFrame(), {}
                else:
                    st.session_state.step5_cleaned = cleaned_df
                    st.session_state.step5_summary = summary
                    st.session_state.step5_done = True
                    flash(f"✅ Cleaned: {len(cleaned_df):,} / {len(linked):,} comments retained ({100*len(cleaned_df)/max(1,len(linked)):.1f}%)", "success")
            st.rerun()

    section_divider()

    # ── Results preview ──────────────────────────────────────────────────────────
    if not cleaned.empty:
        st.caption(f"Showing {len(cleaned):,} cleaned comments from disk.")

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Retained Comments", len(cleaned))
        with c2: metric_card("Unique Videos", cleaned["video_id"].nunique())
        with c3: metric_card("Avg Text Length", f"{cleaned['clean_text'].str.len().mean():.0f} chars")
        with c4: metric_card("Channels", cleaned["channel_title"].nunique())

        # Priority distribution
        st.markdown("#### Priority Level Distribution")
        if "priority_level" in cleaned.columns:
            priority_order = ["high", "medium", "general"]
            priority_counts = cleaned["priority_level"].value_counts().reindex(priority_order, fill_value=0)
            fig = px.bar(
                x=priority_order,
                y=priority_counts.values,
                title="Priority Level Distribution",
                labels={"x": "Priority", "y": "Comment Count"},
                color=priority_order,
                color_discrete_map={"high": "#f85149", "medium": "#f0c000", "general": "#8b949e"},
            )
            fig.update_layout(template="plotly_white", height=350)
            st.plotly_chart(fig, width='stretch')

        # Rule-based demand signal pre-detection
        st.markdown("#### Rule-Based Demand Signal Pre-Detection")
        if "demand_signals" in cleaned.columns:
            sig_exploded = cleaned["demand_signals"].str.split("|").explode()
            sig_counts = sig_exploded[sig_exploded != "general"].value_counts()
            _y_col = sig_counts.reset_index().columns[1]
            fig2 = px.bar(
                sig_counts.reset_index(),
                x="demand_signals",
                y=_y_col,
                title="Rule-Based Demand Signal Types",
                labels={_y_col: "Count", "demand_signals": "Signal Type"},
                color=_y_col,
                color_continuous_scale="Mint",
            )
            fig2.update_layout(template="plotly_white", height=400, xaxis_tickangle=-30)
            st.plotly_chart(fig2, width='stretch')

        section_divider()
        st.markdown("#### Sample Cleaned Data")
        st.dataframe(cleaned.head(20), width='stretch', height=400)
        download_button("⬇ Download Cleaned Data", cleaned, "cleaned_comments_linked", "s5_clean")
    else:
        st.info(
            "No cleaned data found on disk. "
            "Collect data in **Steps 2–4**, then click **Run Cleaning** above."
        )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 6 — LLM SIGNALS
# ════════════════════════════════════════════════════════════════════════════════

def render_step6_llm():
    st.markdown("## 🤖 Step 6: LLM Demand Signal Detection")
    st.markdown(
        "Classify each cleaned comment into demand signal types using AI.\n"
        "Supports **DeepSeek** and **Google Gemini** models.\n"
        "• `purchase_intent` · `problem_complaint` · `comparison_research` · "
        "`usage_scenario` · `wishful_thinking` · `supply_recommendation`\n\n"
    )

    # Always try to load results from disk first — shown regardless of API key
    full_df, signal_df = pd.DataFrame(), pd.DataFrame()
    try:
        full_df, signal_df = load_latest_demand_signals()
    except Exception:
        pass

    has_disk_data = not signal_df.empty

    # Model selection
    st.markdown("### ⚙️ LLM Settings")
    model_provider = st.selectbox(
        "Model Provider",
        ["deepseek", "gemini"],
        format_func=lambda x: {"deepseek": "🔵 DeepSeek", "gemini": "🟠 Google Gemini"}[x],
        help="Select the AI model provider",
    )

    # Provider-specific API key
    if model_provider == "deepseek":
        api_placeholder = "Enter your DeepSeek API key"
        api_help = "Get from https://platform.deepseek.com/"
        model_options = ["deepseek-chat", "deepseek-coder"]
        default_model = "deepseek-chat"
    else:
        api_placeholder = "Enter your Gemini API key"
        api_help = "Get from https://aistudio.google.com/app/apikey or https://ai.google.dev/"
        model_options = [
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]
        default_model = "gemini-3-flash-preview"

    api_key = st.text_input(api_placeholder, type="password", help=api_help)

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        batch_size = st.number_input("Batch size (comments/call)", min_value=5, max_value=50, value=20)
    with col_b:
        rate_delay = st.slider("Rate limit delay (s)", 0.5, 5.0, 1.5, 0.5)
    with col_c:
        save_every = st.number_input("Checkpoint every N batches", min_value=1, max_value=20, value=5)
    with col_d:
        model_name = st.selectbox("Model", model_options, index=0)

    # Config summary
    with st.expander("🔍 LLM Classification Labels", expanded=False):
        st.markdown("""
        | Label | Description |
        |-------|-------------|
        | **purchase_intent** | Explicit desire or plan to buy a protective case |
        | **problem_complaint** | Frustration about damage or lack of protection |
        | **comparison_research** | Comparing or researching different cases |
        | **usage_scenario** | Describes a specific use context requiring protection |
        | **wishful_thinking** | Wishes they had bought or owned protection |
        | **supply_recommendation** | Recommends or praises a specific case product |
        | **no_signal** | No demand signal for protective cases |
        """)

    # ── Run / Re-run button ─────────────────────────────────────────────────────
    cleaned = st.session_state.step5_cleaned if not st.session_state.step5_cleaned.empty else load_cleaned_data()
    total_comments = len(cleaned) if not cleaned.empty else 0

    if total_comments == 0:
        n_to_analyze = 0
        st.warning("No cleaned data found. Complete Step 5 first.")
    else:
        n_to_analyze = st.number_input(
            "Comments to analyze", 1, total_comments, total_comments,
            help=f"Enter how many of the {total_comments:,} cleaned comments to analyze."
        )

    btn_label = "🚀 Re-run LLM Classification" if has_disk_data else "🚀 Run LLM Classification"

    # Show error from a previous run (outside button block so it persists after rerun)
    if st.session_state.llm_error:
        st.error(st.session_state.llm_error)
        st.session_state.llm_error = None

    if st.button(btn_label, type="primary", disabled=st.session_state.llm_started):
        if total_comments == 0:
            st.warning("No cleaned data available. Complete Step 5 first.")
        else:
            st.session_state.llm_started = True

            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_cb(batch_idx, n_batches, processed, total, elapsed=0, eta=0, results_count=0):
                pct = int(100 * processed / total)
                progress_bar.progress(pct)
                speed = processed / max(elapsed, 0.1)
                status_text.text(
                    f"Batch {batch_idx+1}/{n_batches} | "
                    f"{processed:,}/{total:,} comments ({pct}%) | "
                    f"~{speed:.1f} comments/s | "
                    f"ETA ~{eta:.0f}s | "
                    f"{results_count:,} results"
                )

            spinner_text = f"{model_provider.capitalize()} AI is classifying comments (this may take a while)..."
            try:
                with st.spinner(spinner_text):
                    full_df, signal_df = run_demand_signal_detection(
                        api_key=api_key,
                        input_df=cleaned.head(n_to_analyze),
                        model_provider=model_provider,
                        model_name=model_name,
                        batch_size=batch_size,
                        save_every=save_every,
                        rate_limit_delay=rate_delay,
                        progress_callback=progress_cb,
                        n_to_analyze=n_to_analyze,
                    )
            except (APIError, QuotaExceededError, LLMCallError) as e:
                progress_bar.empty()
                status_text.empty()
                st.session_state.llm_started = False
                if isinstance(e, QuotaExceededError):
                    provider_name = e.provider
                    partial = e.partial_results
                    partial_count = len(partial) if partial else 0
                    detail = e.message or f"{provider_name} quota exceeded"
                    st.session_state.llm_error = (
                        f"⚠️ **Quota exceeded** ({provider_name}) — {detail}\n\n"
                        f"Progress saved: {partial_count:,} of {n_to_analyze:,} comments classified. "
                        f"Resume later or switch to a different provider."
                    )
                    st.session_state.llm_error_kind = "error"
                elif isinstance(e, LLMCallError):
                    provider_name = e.provider
                    reason = e.reason
                    batch_info = f" batch {e.batch_idx+1}" if e.batch_idx is not None else ""
                    partial = e.partial_results
                    partial_count = len(partial) if partial else 0
                    st.session_state.llm_error = (
                        f"⚠️ **LLM Error** ({provider_name}{batch_info}) — {reason}\n\n"
                        f"Progress saved: {partial_count:,} of {n_to_analyze:,} comments classified. "
                        f"Check your API key, network, or model availability."
                    )
                    st.session_state.llm_error_kind = "error"
                else:
                    batches = ", ".join(str(b["batch"] + 1) for b in e.batch_errors)
                    partial = e.partial_results
                    partial_count = len(partial) if partial else 0
                    st.session_state.llm_error = (
                        f"⚠️ **{e.provider} API call failed** (batch {batches}) — "
                        f"Please check your API key or network connection."
                        + (f" Progress saved: {partial_count:,} comments classified." if partial_count else "")
                    )
                    st.session_state.llm_error_kind = "error"
                full_df = pd.DataFrame()
                signal_df = pd.DataFrame()
                st.rerun()
            else:
                total_input = len(cleaned)
                total_done = len(full_df)
                if total_done < total_input:
                    st.warning(
                        f"⚠️ **Partial results** — {total_done:,} of {total_input:,} comments classified "
                        f"(quota may have been exceeded). Progress saved; resume to continue."
                    )
                st.session_state.step6_full = full_df
                st.session_state.step6_signals = signal_df
                st.session_state.step6_done = True
                st.success(f"✅ LLM classification complete! Found {len(signal_df):,} demand signals.")

            progress_bar.empty()
            status_text.empty()

    section_divider()

    # ── Results preview ──────────────────────────────────────────────────────────
    if not signal_df.empty:
        if has_disk_data:
            st.success(f"Showing {len(signal_df):,} demand signals from disk (saved from a previous run).")

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Total Classified", len(full_df))
        with c2: metric_card("Demand Signals", len(signal_df),
                              f"{100*len(signal_df)/max(1,len(full_df)):.1f}%")
        with c3: metric_card("Unique Videos", signal_df["video_id"].nunique())
        with c4: metric_card("Avg Confidence", f"{signal_df['confidence'].mean():.2f}")

        # Signal distribution
        st.markdown("#### Demand Signal Distribution")
        sig_counts = signal_df["signal"].value_counts()
        fig = px.bar(
            x=sig_counts.index,
            y=sig_counts.values,
            title="Demand Signal Types (DeepSeek Classified)",
            labels={"x": "Signal Type", "y": "Comment Count"},
            color=sig_counts.values,
            color_continuous_scale="Burg",
        )
        fig.update_layout(template="plotly_white", height=400, xaxis_tickangle=-30)
        st.plotly_chart(fig, width='stretch')

        # Confidence distribution
        st.markdown("#### Confidence Score Distribution")
        fig2 = px.histogram(
            signal_df,
            x="confidence",
            nbins=20,
            title="LLM Confidence Score Distribution",
            color_discrete_sequence=["#58a6ff"],
        )
        fig2.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig2, width='stretch')

        section_divider()
        st.markdown("#### Top Demand Signals (Highest Confidence)")
        display_cols = ["signal", "confidence", "comment_text", "video_title", "video_url", "channel_title"]
        display = signal_df[[c for c in display_cols if c in signal_df.columns]].head(30)
        st.dataframe(display, width='stretch', height=500)

        dl1, dl2 = st.columns(2)
        with dl1:
            download_button("⬇ Download Full Results", full_df, "demand_signals_full", "s6_full")
        with dl2:
            download_button("⬇ Download Signals Only", signal_df, "demand_signals_only", "s6_sig")
    else:
        st.info("No demand signal results found on disk. Configure settings above and click **Run LLM Classification**.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════

def render_dashboard():
    st.markdown("## 📈 Results Dashboard")
    st.markdown("Final demand signal results with interactive visualizations and data export.")

    # Load latest data
    full_df, signal_df = load_latest_demand_signals()
    if signal_df.empty:
        signal_df = st.session_state.step6_signals
    if full_df.empty:
        full_df = st.session_state.step6_full

    if signal_df.empty:
        st.warning("No demand signal results yet. Complete Step 6 first.")
        return

    # KPIs
    st.markdown("### Key Performance Indicators")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Total Signals", len(signal_df))
    with c2: metric_card("Avg Confidence", f"{signal_df['confidence'].mean():.2f}")
    with c3: metric_card("High Confidence (≥0.8)", len(signal_df[signal_df["confidence"] >= 0.8]))
    with c4: metric_card("Videos Covered", signal_df["video_id"].nunique())
    with c5: metric_card("Channels", signal_df["channel_title"].nunique())

    section_divider()

    # Signal breakdown
    st.markdown("### Demand Signal Breakdown")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        sig_counts = signal_df["signal"].value_counts()
        fig_bar = px.bar(
            x=sig_counts.index,
            y=sig_counts.values,
            title="Demand Signals by Type",
            labels={"x": "Signal Type", "y": "Count"},
            color=sig_counts.values,
            color_continuous_scale="RdPu",
        )
        fig_bar.update_layout(template="plotly_white", height=400, xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, width='stretch')

    with col_right:
        st.markdown("**Signal Type Legend**")
        for sig, count in sig_counts.items():
            pct = 100 * count / len(signal_df)
            st.markdown(f"- **{sig}**: {count:,} ({pct:.1f}%)")

    # Confidence by signal type
    st.markdown("### Confidence by Signal Type")
    fig_box = px.box(
        signal_df,
        x="signal",
        y="confidence",
        color="signal",
        title="Confidence Score Distribution by Signal Type",
        points="outliers",
    )
    fig_box.update_layout(template="plotly_white", height=400, xaxis_tickangle=-30, showlegend=False)
    st.plotly_chart(fig_box, width='stretch')

    # Product categories (from cleaned data)
    st.markdown("### Signals by Product Category")
    cleaned = load_cleaned_data()
    if not cleaned.empty and "product_categories" in cleaned.columns:
        merged = signal_df.merge(
            cleaned[["comment_id", "product_categories", "priority_level"]],
            on="comment_id",
            how="left",
        )
        if "product_categories" in merged.columns:
            cat_exploded = merged["product_categories"].str.split("|").explode()
            cat_sig = cat_exploded[cat_exploded != "unknown"].value_counts()
            _y_col = cat_sig.reset_index().columns[1]
            fig_cat = px.bar(
                cat_sig.reset_index(),
                x="product_categories",
                y=_y_col,
                title="Demand Signals by Product Category",
                labels={_y_col: "Count", "product_categories": "Category"},
                color=_y_col,
                color_continuous_scale="Emrld",
            )
            fig_cat.update_layout(template="plotly_white", height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig_cat, width='stretch')

    # Timeline (if published_at available) — prefer full_df which has richer columns
    st.markdown("### Comment Timeline")
    timeline_df = full_df if "published_at" in full_df.columns else signal_df
    if "published_at" in timeline_df.columns:
        timeline_df = timeline_df.copy()
        timeline_df["published_at"] = pd.to_datetime(timeline_df["published_at"], errors="coerce")
        timeline_df = timeline_df.dropna(subset=["published_at"])
        if not timeline_df.empty and "signal" in timeline_df.columns:
            timeline_df["date"] = timeline_df["published_at"].dt.date
            timeline = timeline_df.groupby(["date", "signal"]).size().reset_index(name="count")
            fig_timeline = px.line(
                timeline,
                x="date",
                y="count",
                color="signal",
                title="Demand Signals Over Time",
                labels={"date": "Date", "count": "Signal Count"},
            )
            fig_timeline.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_timeline, width='stretch')

    # Top videos by signal count
    st.markdown("### Top Videos by Signal Count")
    # Use full_df if it has video_title, otherwise signal_df
    video_df = full_df if "video_title" in full_df.columns else signal_df
    group_cols = [c for c in ["video_id", "video_title", "channel_title"] if c in video_df.columns]
    video_signals = video_df.groupby(group_cols).size().reset_index(name="signal_count")
    video_signals = video_signals.sort_values("signal_count", ascending=False).head(20)
    video_signals["video_url"] = "https://www.youtube.com/watch?v=" + video_signals["video_id"].astype(str)
    disp_cols = [c for c in ["video_title", "channel_title", "signal_count", "video_url"] if c in video_signals.columns]
    st.dataframe(
        video_signals[disp_cols],
        width='stretch',
        height=500,
    )

    section_divider()

    # Full results table (enriched with original comment text)
    st.markdown("### 📋 Complete Demand Signal Table")
    table_df = signal_df.copy()
    if "clean_text" in full_df.columns:
        text_map = full_df[["comment_id", "clean_text"]].drop_duplicates("comment_id").set_index("comment_id")["clean_text"]
        table_df["comment_text"] = table_df["comment_id"].map(text_map)
    disp_cols = ["comment_text"] + [c for c in table_df.columns if c != "comment_text"]
    st.dataframe(table_df[disp_cols], width='stretch', height=600)

    # Export
    st.markdown("### 📥 Export")
    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        download_button("⬇ Export Signals (CSV)", table_df, "demand_signals_export", "dash_sig_csv")
    with dl2:
        download_button("⬇ Export Signals (Parquet)", table_df, "demand_signals_export", "dash_sig_pq")
    with dl3:
        download_button("⬇ Export Full Results (CSV)", full_df, "demand_signals_full_export", "dash_full")


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

def main():
    selected_page = page_nav()
    show_flash()

    if selected_page.startswith("🏠"):
        render_overview()
    elif selected_page.startswith("🔑"):
        render_step1_keywords()
    elif selected_page.startswith("📡"):
        render_step2_collection()
    elif selected_page.startswith("🔗"):
        render_step3_merge()
    elif selected_page.startswith("📊"):
        render_step4_link()
    elif selected_page.startswith("🧹"):
        render_step5_clean()
    elif selected_page.startswith("🤖"):
        render_step6_llm()
    elif selected_page.startswith("📈"):
        render_dashboard()


if __name__ == "__main__":
    main()
