"""
Market Signal AI Agent Platform
================================
Streamlit dashboard for the YouTube demand signal detection pipeline.

Simplified user flow:
  Step 1 → Step 2 (Collect + Full Pipeline) → Step 3 (LLM) → Dashboard
"""

from __future__ import annotations

import io
import random
import sys
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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

from src.config import BASE_DIR, CHECKPOINT_DIR, OUTPUT_DIR
from src.step1_keywords import get_all_queries_flat, generate_all, generate_member_queries, generate_leader_queries, generate_keywords_from_components
from src.step2_collect import run_collection, load_latest_collected, clear_checkpoint
from src.step3_merge import merge_staging_to_master, load_latest_master
from src.step4_link import link_comments_to_videos, load_linked_data, load_video_summary
from src.step5_clean import clean_linked_data, load_cleaned_data
from src.step6_demand_signal import (
    run_demand_signal_detection,
    run_phase1_classification,
    run_phase2_scoring,
    load_latest_demand_signals,
    list_signal_versions,
    load_signal_version,
    load_phase1_checkpoint,
    QuotaExceededError,
    APIError,
    LLMCallError,
    clear_llm_checkpoint,
    _merge_and_build_output,
)

# ── Session state defaults ─────────────────────────────────────────────────────
defaults = {
    "step1_keywords": [],
    "step1_done": False,
    "step2_videos": pd.DataFrame(),
    "step2_comments": pd.DataFrame(),
    "step2_runs": pd.DataFrame(),
    "step2_done": False,
    "step3_full_cleaned": pd.DataFrame(),
    "step3_done": False,
    "step3_summary": {},
    "step4_full": pd.DataFrame(),
    "step4_signals": pd.DataFrame(),
    "step4_done": False,
    "running": False,
    "pipeline_running": False,
    "step2_quota_exceeded": False,
    "collection_started": False,
    "llm_started": False,
    "llm_error": None,
    "llm_phase1_done": False,
    "llm_phase2_done": False,
    "llm_phase1_started": False,
    "llm_phase2_started": False,
    "llm_phase1_error": None,
    "llm_phase2_error": None,
    "step3_master_videos": 0,
    "step3_master_comments": 0,
    "pipeline_days_back": 30,
    "_flash": None,
    # ── Persisted user inputs (survive page navigation within a session) ──────────
    # Step 2: Collection
    "s2_youtube_key": "",
    "s2_days_back": 30,
    "s2_search_max_pages": 2,
    "s2_comment_max_pages": 5,
    "s2_search_results_per_page": 50,
    "s2_comments_per_page": 100,
    "s2_region_code": "US",
    "s2_relevance_language": "en",
    "s2_fetch_comments": True,
    "s2_n_keywords": 1,
    # Step 3: LLM
    "s3_model_provider": "deepseek",
    "s3_api_key": "",
    "s3_batch_size": 20,
    "s3_rate_delay": 1.5,
    "s3_model_name": "deepseek-chat",
    "s3_use_phase2": True,
    "s3_n_to_analyze": 0,
}

for key, val in defaults.items():
    st.session_state.setdefault(key, val)

# Custom CSS
st.markdown("""
<style>
/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #475569;
}
/* ── Sidebar radio text ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #475569;
}
[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] {
    color: #2563eb;
    font-weight: 600;
}
/* ── Main app ─────────────────────────────────────────────────────────────── */
.stApp { background-color: #f1f5f9; }
.main .block-container {
    background-color: #f1f5f9;
    padding-top: 5.5rem;   /* room for sticky nav */
    padding-bottom: 5rem;
}

/* ── Sticky section-nav bar ──────────────────────────────────────────────── */
.section-nav {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999;
    background: #1e293b;
    border-bottom: 2px solid #3b82f6;
    display: flex;
    align-items: center;
    gap: 0.1rem;
    padding: 0 1rem;
    height: 48px;
    overflow-x: auto;
    scrollbar-width: none;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25);
}
.section-nav::-webkit-scrollbar { display: none; }

.section-nav-brand {
    font-size: 0.78rem;
    font-weight: 700;
    color: #93c5fd;
    white-space: nowrap;
    margin-right: 0.75rem;
    letter-spacing: 0.04em;
    flex-shrink: 0;
}

.nav-btn {
    flex-shrink: 0;
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.15s, color 0.15s, transform 0.1s;
    white-space: nowrap;
    line-height: 1;
}
.nav-btn:hover {
    background: rgba(59,130,246,0.2);
    color: #e0e7ff;
    transform: translateY(-1px);
}
.nav-btn.active {
    background: #2563eb;
    color: #ffffff;
    font-weight: 600;
    box-shadow: 0 1px 6px rgba(37,99,235,0.45);
}
.nav-sep {
    width: 1px;
    height: 20px;
    background: rgba(255,255,255,0.12);
    flex-shrink: 0;
    margin: 0 0.1rem;
}

/* ── Floating Back-to-top ────────────────────────────────────────────────── */
.btt-btn {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    z-index: 998;
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 50%;
    width: 46px;
    height: 46px;
    font-size: 1.25rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 16px rgba(37,99,235,0.45);
    transition: background 0.2s, transform 0.2s, box-shadow 0.2s, opacity 0.3s;
    opacity: 0;
    pointer-events: none;
    text-decoration: none;
}
.btt-btn.visible {
    opacity: 1;
    pointer-events: auto;
}
.btt-btn:hover {
    background: #1d4ed8;
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(37,99,235,0.55);
}
.btt-btn:active {
    transform: translateY(-1px);
}

/* ── Section headers ─────────────────────────────────────────────────────── */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding-top: 0.25rem;
}

/* ── Cards / metrics ────────────────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.75rem;
}
div[data-testid="stMetricLabel"] { color: #64748b; font-size: 0.78rem; }
div[data-testid="stMetricValue"] { color: #1e293b; font-size: 1.3rem; }
div[data-testid="stMetricDelta"] { font-size: 0.75rem; }

/* ── Status boxes ─────────────────────────────────────────────────────────── */
.status-box { padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem; font-family: monospace; }
.status-done    { background: #dcfce7; border: 1px solid #16a34a; color: #15803d; }
.status-running { background: #fef9c3; border: 1px solid #ca8a04; color: #854d0e; }
.status-warn    { background: #fee2e2; border: 1px solid #dc2626; color: #991b1b; }
.status-empty   { background: #f8fafc; border: 1px solid #e2e8f0; color: #64748b; }

/* ── Dividers ─────────────────────────────────────────────────────────────── */
.dashboard-divider { border-color: #e2e8f0; margin: 1.25rem 0; }

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-testid="stTabBar"] { border-bottom: 1px solid #e2e8f0; }
.stTabs [data-testid="stTab"] { color: #64748b; }
.stTabs [data-testid="stTab"]:hover { color: #1e293b; }
.stTabs [data-testid="stTab"][aria-selected="true"] {
    color: #2563eb;
    border-bottom: 2px solid #2563eb;
}

/* ── KPI cards ───────────────────────────────────────────────────────────── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #1e293b; line-height: 1.2; }
.kpi-label { font-size: 0.75rem; color: #64748b; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi-sub { font-size: 0.72rem; color: #94a3b8; margin-top: 0.1rem; }

/* ── Section badges ────────────────────────────────────────────────────────── */
.badge-0  { background: #e0e7ff; color: #3730a3; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-1  { background: #dbeafe; color: #1d4ed8; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-2  { background: #ede9fe; color: #6d28d9; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-3  { background: #ccfbf1; color: #0f766e; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-4  { background: #ffedd5; color: #9a3412; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-5  { background: #fce7f3; color: #9d174d; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-6  { background: #e0f2fe; color: #0369a1; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-7  { background: #eef2ff; color: #4338ca; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-8  { background: #fef2f2; color: #991b1b; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-9  { background: #ecfdf5; color: #065f46; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-10 { background: #f3e8ff; color: #6b21a8; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-11 { background: #fff7ed; color: #9a3412; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-12 { background: #eff6ff; color: #1e40af; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-13 { background: #fef3c7; color: #92400e; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-14 { background: #dbeafe; color: #1e40af; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

/* ── Chart containers ──────────────────────────────────────────────────────── */
.chart-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.chart-title { font-size: 0.82rem; color: #64748b; font-weight: 600; margin-bottom: 0.5rem; }

/* ── Pain point highlight ──────────────────────────────────────────────────── */
.pain-point { background: #fff1f2; border: 1px solid #fda4af; border-radius: 6px; padding: 0.5rem 0.75rem; color: #be123c; font-size: 0.82rem; }
.pain-point-title { color: #9f1239; font-weight: 600; margin-bottom: 0.25rem; }

/* ── Download buttons ─────────────────────────────────────────────────────── */
.stDownloadButton>button { background: #2563eb; color: white; border: none; border-radius: 6px; font-weight: 600; }
.stDownloadButton>button:hover { background: #1d4ed8; }

/* ── Dataframe ─────────────────────────────────────────────────────────────── */
.dataframe { font-size: 0.78rem; }

/* ── HR override ──────────────────────────────────────────────────────────── */
hr { border-color: #e2e8f0; }

/* ── Branding hide ────────────────────────────────────────────────────────── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── Annotation text ──────────────────────────────────────────────────────── */
.ann-text { font-size: 0.75rem; color: #94a3b8; }

/* ── Guide callout ────────────────────────────────────────────────────────── */
.guide-callout { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 1rem; }

/* ── Step tabs ────────────────────────────────────────────────────────────── */
.stTabs [data-testid="stTabBar"] { border-bottom: 2px solid #e2e8f0; }
.stTabs [data-testid="stTab"] { color: #64748b; font-weight: 500; }
.stTabs [data-testid="stTab"]:hover { color: #1e293b; }
.stTabs [data-testid="stTab"][aria-selected="true"] {
    color: #2563eb;
    border-bottom: 2px solid #2563eb;
    font-weight: 600;
}

/* ── Scrollbar (sidebar & main) ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
</style>

<!-- Sticky section navigation bar -->
<div class="section-nav" id="main-nav">
  <span class="section-nav-brand">🔍 MARKET SIGNAL</span>
  <div class="nav-sep"></div>
  <button class="nav-btn active" onclick="switchPage(0)" id="nav-btn-0">🏠 Overview</button>
  <button class="nav-btn" onclick="switchPage(1)" id="nav-btn-1">🔑 Keywords</button>
  <button class="nav-btn" onclick="switchPage(2)" id="nav-btn-2">📡 Collection</button>
  <button class="nav-btn" onclick="switchPage(3)" id="nav-btn-3">🤖 LLM Signals</button>
  <button class="nav-btn" onclick="switchPage(4)" id="nav-btn-4">📈 Dashboard</button>
</div>

<!-- Floating Back-to-top button -->
<button class="btt-btn" id="bttBtn" onclick="window.scrollTo({top:0,behavior:'smooth'})" title="Back to top">↑</button>

<script>
// ── Highlight active nav button on scroll ──────────────────────────────────────
const sections = [
  { id: "overview-page",        btn: 0 },
  { id: "step-1-keywords",      btn: 1 },
  { id: "step-2-collection",    btn: 2 },
  { id: "step-3-llm",          btn: 3 },
  { id: "step-4-dashboard",    btn: 4 },
];

function setActiveNav(idx) {
  sections.forEach((_, i) => {
    const el = document.getElementById("nav-btn-" + i);
    if (el) el.classList.toggle("active", i === idx);
  });
}

function switchPage(idx) {
  const map = [
    document.getElementById("overview-page"),
    document.getElementById("step-1-keywords"),
    document.getElementById("step-2-collection"),
    document.getElementById("step-3-llm"),
    document.getElementById("step-4-dashboard"),
  ];
  const target = map[idx];
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    setActiveNav(idx);
  }
}

const observer = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        const match = sections.find(s => entry.target.id === s.id);
        if (match) setActiveNav(match.btn);
        break;
      }
    }
  },
  { rootMargin: "-40% 0px -55% 0px", threshold: 0 }
);

const mainEl = document.querySelector(".main") || document.body;
sections.forEach(s => {
  const el = document.getElementById(s.id);
  if (el) observer.observe(el);
});

// ── Back-to-top button visibility ────────────────────────────────────────────
const bttBtn = document.getElementById("bttBtn");
if (bttBtn) {
  window.addEventListener("scroll", () => {
    bttBtn.classList.toggle("visible", window.scrollY > 400);
  }, { passive: true });
}
</script>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clear_all_collection_data():
    """Delete all staging, master, linked, cleaned, and demand-signal data files."""
    dirs_to_clear = [
        BASE_DIR / "data" / "collection_output",
        BASE_DIR / "data" / "linked_data",
        BASE_DIR / "data" / "cleaned_data",
    ]
    for d in dirs_to_clear:
        for f in d.glob("*"):
            if f.is_file():
                f.unlink()
    cp = BASE_DIR / "data" / "collection_output" / "collection_checkpoint.json"
    if cp.exists():
        cp.unlink()

    # Also clear demand signal outputs and checkpoints
    for pattern in ["demand_signals_full_*.parquet", "demand_signals_only_*.parquet",
                   "phase1_results_*.parquet"]:
        for f in OUTPUT_DIR.glob(pattern):
            f.unlink()
    for f in CHECKPOINT_DIR.glob("llm_*_phase1_*.json"):
        f.unlink()
    for f in CHECKPOINT_DIR.glob("llm_*_phase2_*.json"):
        f.unlink()

    print("[APP] All collection data cleared.")


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    return buf.getvalue()


def download_button(label: str, df: pd.DataFrame, default_name: str, key: str):
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
    st.markdown("<hr class='dashboard-divider'>", unsafe_allow_html=True)


def flash(msg: str, kind: str = "success"):
    st.session_state._flash = (kind, msg)


def show_flash():
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


def run_full_pipeline(progress_bar=None, status_text=None, days_back: int | None = 30):
    collection_dir = BASE_DIR / "data" / "collection_output"
    linked_dir = BASE_DIR / "data" / "linked_data"
    cleaned_dir = BASE_DIR / "data" / "cleaned_data"
    safe_mkdir(linked_dir)
    safe_mkdir(cleaned_dir)

    print(f"\n{'='*50}")
    print(f"[PIPELINE] Starting pipeline...")

    if progress_bar:
        progress_bar.progress(5)
    if status_text:
        status_text.text("Step 1/3: Merging staging files...")

    src_videos, src_comments, _ = merge_staging_to_master(
        staging_dir=collection_dir,
        output_dir=collection_dir,
    )

    if src_videos.empty:
        raise ValueError("Videos DataFrame is empty after merge. Check merge logs above.")

    if not src_videos.empty:
        src_videos = src_videos.drop_duplicates(subset=["video_id"], keep="last")
    if not src_comments.empty:
        src_comments = src_comments.drop_duplicates(subset=["comment_id"], keep="last")

    src_videos.to_parquet(collection_dir / "videos_master.parquet", index=False)
    src_comments.to_parquet(collection_dir / "comments_master.parquet", index=False)

    if progress_bar:
        progress_bar.progress(35)
    if status_text:
        status_text.text("Step 2/3: Linking comments to videos...")

    linked_df = link_comments_to_videos(
        comments=src_comments,
        videos=src_videos,
        output_dir=linked_dir,
    )

    if progress_bar:
        progress_bar.progress(65)
    if status_text:
        status_text.text("Step 3/3: Cleaning and filtering...")

    cleaned_df, summary = clean_linked_data(linked_df=linked_df, output_dir=cleaned_dir, days_back=days_back)

    if progress_bar:
        progress_bar.progress(100)
    if status_text:
        status_text.text("✅ Pipeline complete!")

    return src_videos, src_comments, cleaned_df, summary


def _scroll_to_top():
    st.session_state._pending_scroll_top = True


def page_nav():
    PAGES = [
        "🏠 Overview",
        "🔑 Step 1: Keywords",
        "📡 Step 2: Collection + Pipeline",
        "🤖 Step 3: LLM Signals",
        "📈 Results Dashboard",
    ]
    st.sidebar.markdown("## 📋 Navigation")
    selected = st.sidebar.radio(
        "Navigation", PAGES, index=0, label_visibility="collapsed",
        on_change=_scroll_to_top,
    )

    if st.session_state.get("_pending_scroll_top"):
        # Script runs in the sidebar iframe context — window.top reaches the root page
        components.html(
            "<script>window.top.scrollTo({top:0,behavior:'instant'});</script>",
            height=0,
        )
        st.session_state._pending_scroll_top = False

    # Sub-nav for page-level sections (rendered below the divider)
    _render_sidebar_subnav(selected)

    return selected


def _render_sidebar_subnav(current_page: str):
    """Renders a sub-navigation area below the main nav divider for the active page."""
    pass


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════

def _render_guide_section(icon, title, body_md):
    st.markdown(f"### {icon} {title}")
    st.markdown(body_md)
    st.divider()


def render_overview():
    # ── Section anchor for sticky nav ─────────────────────────────────────────
    st.markdown('<div id="overview-page"></div>', unsafe_allow_html=True)

    st.markdown("## 🏠 Platform Overview")
    st.markdown(
        "**Market Signal AI Agent** — an end-to-end YouTube demand signal detection platform "
        "for **hard case / protective case** products. Transforms raw YouTube data into "
        "actionable demand insights in **4 simple steps**.\n\n"
        "Use the **sidebar** to navigate between pages at any time. "
        "Your inputs (API keys, settings, selections) are **remembered within the same session**."
    )
    st.divider()

    # ── Pipeline architecture ─────────────────────────────────────────────────
    st.markdown("### 🔀 Pipeline Architecture")
    steps = [
        ("🔑", "Step 1: Keywords", "Generate YouTube search queries across 18 product categories"),
        ("📡", "Step 2: Collection", "Fetch videos + comments from YouTube Data API"),
        ("🔄", "Step 2+: Full Pipeline", "Merge → Link → Clean in one click"),
        ("🤖", "Step 3: LLM Signals", "DeepSeek / Gemini AI classifies demand signals"),
        ("📈", "Step 4: Dashboard", "Interactive visualization & export"),
    ]
    cols = st.columns(len(steps))
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"**{icon} {title}**")
            st.caption(desc)
            if i < len(steps) - 1:
                st.markdown("→", unsafe_allow_html=False)

    st.divider()

    # ── Step-by-step guide ───────────────────────────────────────────────────
    st.markdown("### 📖 Step-by-Step Guide")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Before You Start",
        "🔑 Step 1: Keywords",
        "📡 Step 2: Collection",
        "🤖 Step 3: LLM Signals",
        "📈 Step 4: Dashboard",
    ])

    with tab1:
        st.markdown("#### Prerequisites")
        st.markdown("""
        Before using this platform, make sure you have:
        """)
        prereq_cols = st.columns(2)
        with prereq_cols[0]:
            st.markdown("""
            **1. YouTube Data API Key** *(required for Step 2)*
            - Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
            - Create a project → Enable **YouTube Data API v3**
            - Create an API Key (restricted to YouTube API)
            - Note: Free quota is **10,000 units/day**
            """)
        with prereq_cols[1]:
            st.markdown("""
            **2. LLM API Key** *(required for Step 3)*
            - **DeepSeek**: Sign up at [platform.deepseek.com](https://platform.deepseek.com/)
            - **Google Gemini**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
            - Both have free tiers sufficient for testing
            """)
        st.divider()
        st.markdown("#### How to Use This Platform")
        st.markdown("""
        1. **Navigate** using the sidebar on the left
        2. **Your inputs persist** — API keys, settings, selections are saved within your session (until you refresh)
        3. **Checkpoints** — collection and LLM runs save progress automatically, so you can safely close and resume
        4. **Any order** — you can jump between steps; already-completed steps show your saved data
        5. **Navigation** — the top bar lets you jump between pages at any time
        """)

    with tab2:
        st.markdown("#### Step 1: Keyword Generation")
        st.markdown("""
        This step generates YouTube search queries that will be used to find relevant videos.
        You have **two options**:
        """)
        opt1, opt2 = st.columns(2)
        with opt1:
            st.markdown("""
            **Option A: Template-Based (Recommended)**
            - Uses 18 pre-defined product categories (cameras, guitars, drones, etc.)
            - Automatically combines objects + types + negative words
            - Just click **▶ Generate Keywords from Templates**
            - Generates hundreds of keywords instantly
            """)
        with opt2:
            st.markdown("""
            **Option B: Custom Keywords**
            - Define your own objects, types, and negative words
            - Great for specific niches not covered by templates
            - Open the **Custom Keyword Builder** expander below
            """)
        st.markdown("""
        **What happens next?**
        - Keywords are saved in your session and passed to Step 2
        - You can view/download the full keyword list
        - Proceed to **Step 2** when ready
        """)

    with tab3:
        st.markdown("#### Step 2: YouTube Data Collection + Full Pipeline")
        st.markdown("""
        This step searches YouTube for videos and scrapes comments.
        """)
        col_flow, col_settings = st.columns([1, 1])
        with col_flow:
            st.markdown("""
            **What runs automatically (Full Pipeline):**
            1. **Search** — Find videos matching your keywords
            2. **Scrape** — Fetch comments from each video
            3. **Merge** — Combine multiple collection runs
            4. **Link** — Match comments to their parent videos
            5. **Clean** — Remove duplicates and filter by recency
            """)
        with col_settings:
            st.markdown("""
            **Key Settings:**
            - `Days to search back` — How recent videos must be
            - `Search result pages` — More = more videos (uses more quota)
            - `Comment pages` — More = more comments per video
            - `Region code` — e.g. `US`, `GB` (leave blank for global)
            """)
        st.markdown("""
        **Quota Safety:**
        - The collection stops automatically when YouTube API quota is exhausted
        - Checkpoints save progress — click **Start** again to resume
        - Status box in Step 2 shows your current progress
        """)

    with tab4:
        st.markdown("#### Step 3: LLM Demand Signal Detection")
        st.markdown("""
        This step uses AI to classify each comment as containing a **demand signal** or not,
        and scores positive signals across **14 dimensions**.
        """)
        phase_col1, phase_col2 = st.columns(2)
        with phase_col1:
            st.markdown("""
            **Phase 1 — Classification**
            - Classifies ALL comments as `demand_signal` or `no_signal`
            - Fast: uses batch processing
            - Checkpoint saves progress every few minutes
            - Can be resumed if interrupted
            """)
        with phase_col2:
            st.markdown("""
            **Phase 2 — Scoring (optional)**
            - Only processes comments flagged as `demand_signal`
            - Scores 14 dimensions: urgency, purchase intent, price sensitivity, etc.
            - Takes longer but provides richer insights
            """)
        st.markdown("""
        **Signal Types Detected:**
        | Label | Meaning |
        |-------|---------|
        | `purchase_intent` | Explicit desire to buy a protective case |
        | `problem_complaint` | Frustration about damage / lack of protection |
        | `comparison_research` | Comparing different cases |
        | `usage_scenario` | Describing a use context needing protection |
        | `wishful_thinking` | Wishes they had bought protection |
        | `supply_recommendation` | Recommending a specific case product |
        """)

    with tab5:
        st.markdown("#### Step 4: Results Dashboard")
        st.markdown("""
        View, filter, and export your demand signal results.
        """)
        dash_col1, dash_col2 = st.columns(2)
        with dash_col1:
            st.markdown("""
            **Available Views:**
            - **KPIs** — Total signals, category breakdown, score distributions
            - **By Signal Type** — Filter and drill into each signal category
            - **Top Videos** — Which videos generate the most signals
            - **Signal Map** — Word-cloud style visualization
            - **Raw Data** — Full comment table with all scores
            """)
        with dash_col2:
            st.markdown("""
            **Export Options:**
            - Download as **CSV** or **Parquet**
            - Save as a **Named Version** for comparison
            - Version history lets you track results over time
            """)

    st.divider()

    # ── Quick Stats ───────────────────────────────────────────────────────────
    st.markdown("### 📊 Current Data Status")
    col1, col2, col3, col4 = st.columns(4)
    try:
        cleaned = load_cleaned_data()
    except Exception:
        cleaned = pd.DataFrame()
    try:
        _, signals = load_latest_demand_signals()
    except Exception:
        signals = pd.DataFrame()
    with col1:
        metric_card("Keywords Available", len(get_all_queries_flat()), "from templates")
    with col2:
        metric_card("Cleaned Comments", len(cleaned), f"{cleaned['video_id'].nunique() if not cleaned.empty else 0} videos")
    with col3:
        metric_card("Demand Signals", len(signals), f"{signals['signal'].nunique() if not signals.empty else 0} types")
    with col4:
        st.markdown("""
        <div class="kpi-card">
          <div class="kpi-value">4</div>
          <div class="kpi-label">Steps Total</div>
          <div class="kpi-sub">Overview → Dashboard</div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 1 — KEYWORDS
# ════════════════════════════════════════════════════════════════════════════════

def _parse_word_list(raw: str) -> list[str]:
    lines = raw.replace(",", "\n").splitlines()
    return [w.strip() for w in lines if w.strip()]


def _preview_combinations(object_words, type_words, negative_words, separator):
    preview = generate_keywords_from_components(object_words, type_words, negative_words, separator)
    st.info(f"Will generate **{len(preview):,} unique keywords** after filtering.")
    if preview:
        st.caption("Sample (first 10): " + " | ".join(preview[:10]))


def render_step1_keywords():
    # ── Section anchor for sticky nav ─────────────────────────────────────────
    st.markdown('<div id="step-1-keywords"></div>', unsafe_allow_html=True)
    st.markdown("## 🔑 Step 1: Keyword Generation")
    st.markdown(
        "Generate YouTube search queries for **hard case / protective case** demand signals. "
        "Queries cover **18 product categories** across protection, storage, usage, and problem scenarios.\n\n"
    )

    st.markdown("### ✏️ Custom Keyword Builder")
    st.caption("Define keywords by entering words across multiple dimensions — they will be auto-combined.")

    with st.expander("🔧 Open Custom Keyword Builder", expanded=False):
        col_obj, col_type, col_neg = st.columns([1, 1, 1])
        with col_obj:
            st.markdown("**Objects** (one per line)")
            obj_raw = st.text_area(
                "Objects", placeholder="camera\nlaptop\nheadphones\njewelry",
                height=160, label_visibility="collapsed", key="dyn_objects",
            )
            separator = st.text_input("Separator", value=" ", max_chars=3, key="dyn_sep")
        with col_type:
            st.markdown("**Type Words** (one per line, use {obj} as placeholder)")
            type_raw = st.text_area(
                "Type Words", placeholder="protective case\nhard case\nbest {obj} case\ndamaged",
                height=160, label_visibility="collapsed", key="dyn_type",
            )
        with col_neg:
            st.markdown("**Negative / Exclude** (one per line)")
            neg_raw = st.text_area(
                "Negative Words", placeholder="iphone case\nphone case\nipad case",
                height=160, label_visibility="collapsed", key="dyn_neg",
            )
        object_words   = _parse_word_list(obj_raw)
        type_words     = _parse_word_list(type_raw)
        negative_words = _parse_word_list(neg_raw)
        if object_words and type_words:
            _preview_combinations(object_words, type_words, negative_words, separator)
        else:
            st.info("Enter at least **Objects** and **Type Words** to see a preview.")
        if st.button("▶ Generate Custom Keywords", type="primary", key="btn_custom_kw"):
            if not object_words:
                st.error("Please enter at least one object word.")
            elif not type_words:
                st.error("Please enter at least one type word.")
            else:
                keywords = generate_keywords_from_components(object_words, type_words, negative_words, separator)
                st.session_state.step1_keywords = keywords
                st.session_state.step1_done = True
                st.session_state.step1_custom = True
                st.success(f"✅ Generated **{len(keywords):,} custom keywords**!")
                st.rerun()

    st.divider()
    st.markdown("### ⚙️ Template-Based Generator")
    st.caption("Use pre-defined object categories and query templates to generate keywords in bulk.")
    col_gen, col_reload = st.columns([1, 1])
    with col_gen:
        if st.button("▶ Generate Keywords from Templates", type="primary", key="btn_template_kw"):
            with st.spinner("Generating queries..."):
                all_df, summary = generate_all()
                all_q = all_df["query"].tolist()
                st.session_state.step1_keywords = all_q
                st.session_state.step1_done = True
                st.session_state.step1_custom = False
            st.success(f"Generated {len(all_q):,} keywords across {len(summary['by_category'])} categories!")
    with col_reload:
        if st.button("🔄 Reload Saved Keywords", key="btn_reload_kw"):
            try:
                all_df, summary = generate_all()
                keywords = all_df["query"].tolist()
                st.session_state.step1_keywords = keywords
                st.session_state.step1_done = True
                st.session_state.step1_custom = False
                st.success(f"Loaded {len(keywords):,} saved keywords.")
            except Exception as e:
                st.warning(f"Could not reload: {e}")

    keywords = st.session_state.get("step1_keywords", [])
    if st.session_state.get("step1_custom"):
        st.success("✅ Custom keywords (user-defined dimensions)")
    elif keywords:
        st.info("ℹ️ Template-based keywords (pre-defined categories)")
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
        fig = px.bar(cat_counts.reset_index(), x="category", y=_y_col, title="Keywords per Category",
                      labels={_y_col: "Count", "category": "Category"}, color=_y_col, color_continuous_scale="Blues")
        fig.update_layout(template="plotly_white", height=400, xaxis_tickangle=-45,
                          hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"))
        fig.update_traces(hovertemplate="%{x}<br>Count: %{y}<extra></extra>")
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
        fig2 = px.pie(type_counts.reset_index(), values=type_counts.reset_index().columns[1],
                       names="type", title="Keywords by Query Type",
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig2, width='stretch')
        for qtype, count in type_counts.items():
            st.caption(f"**{qtype}**: {count:,} keywords")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 2 — COLLECTION + FULL PIPELINE
# ════════════════════════════════════════════════════════════════════════════════

def render_step2_collection():
    # ── Section anchor for sticky nav ─────────────────────────────────────────
    st.markdown('<div id="step-2-collection"></div>', unsafe_allow_html=True)

    st.markdown("## 📡 Step 2: YouTube Data Collection")
    st.markdown(
        "Search YouTube for videos matching the generated keywords and scrape comments. "
        "**Quota-aware** — will stop gracefully when API quota is exceeded and resume on next run.\n\n"
        "After collection, run the **Full Pipeline** to automatically Merge → Link → Clean."
    )

    checkpoint_path = BASE_DIR / "data" / "collection_output" / "collection_checkpoint.json"
    checkpoint_index = 0
    total_keywords = st.session_state.step1_keywords or get_all_queries_flat()
    total_kw = len(total_keywords)
    if checkpoint_path.exists():
        try:
            import json as _json
            with open(checkpoint_path) as f:
                cp_data = _json.load(f)
            checkpoint_index = cp_data.get("keyword_index", 0)
        except Exception:
            checkpoint_index = 0

    if st.session_state.step2_quota_exceeded:
        status_box("Quota exceeded — collection paused. Progress saved.", "warn")
    elif checkpoint_index > 0:
        status_box(f"Resuming — {checkpoint_index}/{total_kw} keywords already processed. Click Start to continue.", "running")
    elif st.session_state.collection_started and not st.session_state.running:
        status_box("Collection complete — ready for Full Pipeline.", "done")
    elif st.session_state.running:
        status_box("Collection in progress...", "running")
    else:
        status_box("Ready to collect — configure settings below and click Start.", "empty")

    st.markdown("### ⚙️ Collection Settings")
    with st.expander("Configure YouTube API & Search Parameters", expanded=True):
        st.markdown("**🔑 YouTube Data API Key** <span style='color:red'>*(required)*</span>", unsafe_allow_html=True)
        youtube_key = st.text_input(
            "Enter your YouTube Data API Key", type="password",
            help="Get a key from https://console.cloud.google.com/apis/credentials",
            label_visibility="collapsed",
            key="s2_youtube_key",
        )
        if not youtube_key:
            st.markdown(":red[**⚠️ YouTube API key is required to start collection.**]", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            days_back = st.number_input(
                "Days to search back", min_value=1, max_value=365,
                value=st.session_state.get("s2_days_back", 30), key="s2_days_back",
            )
            search_max_pages = st.number_input(
                "Search result pages per keyword", min_value=1, max_value=10,
                value=st.session_state.get("s2_search_max_pages", 2), key="s2_search_max_pages",
            )
            comment_max_pages = st.number_input(
                "Comment pages per video", min_value=1, max_value=20,
                value=st.session_state.get("s2_comment_max_pages", 5), key="s2_comment_max_pages",
            )
        with col_b:
            search_results_per_page = st.slider(
                "Search results per page", 10, 50,
                value=st.session_state.get("s2_search_results_per_page", 50), key="s2_search_results_per_page",
            )
            comments_per_page = st.slider(
                "Comments per page", 10, 100,
                value=st.session_state.get("s2_comments_per_page", 100), key="s2_comments_per_page",
            )
            region_code = st.text_input(
                "Region code (e.g. US)",
                value=st.session_state.get("s2_region_code", "US"), key="s2_region_code",
            ).strip() or None
            relevance_language = st.text_input(
                "Relevance language (e.g. en)",
                value=st.session_state.get("s2_relevance_language", "en"), key="s2_relevance_language",
            ).strip() or None
        fetch_comments = st.checkbox(
            "Fetch comments (recommended)",
            value=st.session_state.get("s2_fetch_comments", True), key="s2_fetch_comments",
        )

    all_keywords = st.session_state.step1_keywords or get_all_queries_flat()
    total_kw = len(all_keywords)
    remaining_kw = max(1, total_kw - checkpoint_index)
    st.markdown(f"**Total available keywords: {total_kw:,}**"
                + (f" | **Checkpoint: {checkpoint_index:,} done — {remaining_kw:,} remaining**" if checkpoint_index > 0 else ""))
    n_keywords = st.number_input(
        "Number of keywords to collect in this run",
        min_value=1, max_value=total_kw,
        value=st.session_state.get("s2_n_keywords", remaining_kw),
        key="s2_n_keywords",
        help=f"Total available: {total_kw:,} | Already done: {checkpoint_index:,}".strip("| "),
    )

    col_start, col_clear, col_reset = st.columns([1, 1, 1])
    with col_start:
        if not youtube_key:
            st.markdown(":red[**⚠️ API key required**]", unsafe_allow_html=True)
            st.button("🚀 Start Collection", type="primary", disabled=True)
        elif st.button("🚀 Start Collection", type="primary", disabled=st.session_state.running):
                keywords = st.session_state.step1_keywords or get_all_queries_flat()
                limit = st.session_state.get("s2_n_keywords", len(keywords))
                start_from = checkpoint_index
                keywords = keywords[start_from:start_from + limit]
                random.shuffle(keywords)
                st.session_state.running = True
                st.session_state.collection_started = True
                output_dir = BASE_DIR / "data" / "collection_output"
                safe_mkdir(output_dir)

                progress_bar = st.progress(0)
                progress_label = st.empty()
                status_text = st.empty()
                log_placeholder = st.empty()
                log_lines = []

                def progress_cb(idx, total, kw, videos_found=0, comments_found=0, phase="searching"):
                    pct = int(100 * (idx + 1) / total)
                    progress_bar.progress(pct)
                    progress_label.caption(f"**{idx+1} / {total}** keywords processed ({pct}%)")
                    if phase == "searching":
                        status_text.text(f"🔍 [{idx+1}/{total}] Searching: {kw}")
                    elif phase == "fetching_videos":
                        status_text.text(f"📹 [{idx+1}/{total}] Fetching video details: {kw} ({videos_found} found)")
                    elif phase == "fetching_comments":
                        status_text.text(f"💬 [{idx+1}/{total}] Fetching comments: {kw} ({comments_found:,} comments)")
                    elif phase == "done":
                        status_text.text(f"✅ [{idx+1}/{total}] Complete: {kw}")
                    if phase == "searching":
                        log_lines.append(f"🔍 [{idx+1}/{total}] Searching: {kw}")
                    elif phase == "fetching_videos":
                        log_lines.append(f"📹 [{idx+1}/{total}] Videos fetched: {kw} ({videos_found} videos)")
                    elif phase == "fetching_comments":
                        log_lines.append(f"💬 [{idx+1}/{total}] Comments: {kw} ({comments_found:,} comments)")
                    elif phase == "done":
                        log_lines.append(f"✅ [{idx+1}/{total}] Done: {kw} → {videos_found} videos, {comments_found} comments")
                    if len(log_lines) > 15:
                        log_lines.pop(0)
                    log_html = (
                        f"<div style='width:100%;box-sizing:border-box;height:120px;overflow-y:auto;"
                        f"background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:5px;"
                        f"font-family:Courier New,monospace;font-size:12px;white-space:pre-wrap;'>"
                        f"{chr(10).join(log_lines)}</div>"
                    )
                    log_placeholder.markdown(log_html, unsafe_allow_html=True)

                with st.spinner("Collecting YouTube data (this may take a while)..."):
                    videos_df, comments_df, runs_df, quota_exceeded = run_collection(
                        api_key=youtube_key, keyword_shard=keywords, days_back=days_back,
                        search_max_pages=search_max_pages, comment_max_pages=comment_max_pages,
                        search_results_per_page=search_results_per_page, comments_per_page=comments_per_page,
                        region_code=region_code, relevance_language=relevance_language,
                        fetch_comments=fetch_comments, output_dir=output_dir,
                        progress_callback=progress_cb,
                    )

                st.session_state.running = False
                st.session_state.step2_videos = videos_df
                st.session_state.step2_comments = comments_df
                st.session_state.step2_runs = runs_df
                st.session_state.step2_quota_exceeded = quota_exceeded
                st.session_state.step2_done = True
                progress_bar.empty()
                progress_label.empty()
                status_text.empty()
                log_placeholder.empty()

                collection_dir = BASE_DIR / "data" / "collection_output"
                for master_file in [
                    collection_dir / "videos_master.parquet",
                    collection_dir / "comments_master.parquet",
                    collection_dir / "runs_master.parquet",
                ]:
                    if master_file.exists():
                        master_file.unlink()

                st.session_state._rerun_after_collection = True
                st.rerun()

    # Clear modal
    def _render_clear_modal():
        with st.container():
            st.markdown("### ⚠️ Clear All Data?")
            st.markdown("This will delete all staging, linked, and cleaned data files, plus reset session state. **This action cannot be undone.**")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Yes, clear everything", key="confirm_clear_yes", type="primary"):
                    clear_all_collection_data()
                    for k in ["step2_videos", "step2_comments", "step2_runs",
                              "collection_started", "step2_quota_exceeded", "step2_done",
                              "step3_full_cleaned", "step3_done", "step3_summary",
                              "step3_master_videos", "step3_master_comments"]:
                        st.session_state[k] = defaults.get(k, None)
                    st.session_state._show_clear_modal = False
                    st.success("All data cleared. Starting fresh.")
                    st.rerun()
            with col_no:
                if st.button("❌ Cancel", key="confirm_clear_no"):
                    st.session_state._show_clear_modal = False
                    st.rerun()

    # Reset modal
    def _render_reset_modal():
        with st.container():
            st.markdown("### 🔄 Reset Checkpoint")
            st.markdown("Resetting the checkpoint means the next collection run will **start from keyword 0**. Choose below whether to also delete all collected staging data.")
            choice = st.radio(
                "Also clear all collection data (staging + master files)?",
                ["no", "yes"],
                format_func=lambda x: {"no": "❌ No — keep collection data (just reset checkpoint)",
                                       "yes": "🗑 Yes — clear everything including collection data"}[x],
                horizontal=True, key="reset_choice",
            )
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Confirm Reset", key="confirm_reset_yes", type="primary"):
                    clear_checkpoint(checkpoint_path)
                    if choice == "yes":
                        clear_all_collection_data()
                        for k in ["step2_videos", "step2_comments", "step2_runs",
                                  "collection_started", "step2_quota_exceeded", "step2_done",
                                  "step3_full_cleaned", "step3_done", "step3_summary",
                                  "step3_master_videos", "step3_master_comments"]:
                            st.session_state[k] = defaults.get(k, None)
                        st.success("Checkpoint reset and all collection data cleared. Starting from keyword 0.")
                    else:
                        st.success("Checkpoint cleared — next run will start from keyword 0.")
                    st.session_state._show_reset_modal = False
                    st.rerun()
            with col_no:
                if st.button("❌ Cancel", key="confirm_reset_no"):
                    st.session_state._show_reset_modal = False
                    st.rerun()

    if st.session_state.get("_show_clear_modal", False):
        _render_clear_modal()
        st.stop()
    if st.session_state.get("_show_reset_modal", False):
        _render_reset_modal()
        st.stop()

    with col_clear:
        if st.button("🗑 Clear Collection Data", key="btn_clear_data"):
            st.session_state._show_clear_modal = True
            st.rerun()
    with col_reset:
        if checkpoint_path.exists():
            try:
                import json as _json
                with open(checkpoint_path) as f:
                    cp_data = _json.load(f)
                checkpoint_index_display = cp_data.get("keyword_index", 0)
            except Exception:
                checkpoint_index_display = 0
            st.caption(f"📌 Checkpoint exists: resume at keyword {checkpoint_index_display}")
            if st.button("🔄 Reset & Start from Beginning", key="btn_reset_checkpoint"):
                st.session_state._show_reset_modal = True
                st.rerun()
        else:
            st.caption("✅ No checkpoint — already starting from beginning")

    section_divider()
    st.markdown("### 🔄 Full Pipeline (Merge → Link → Clean)")

    st.markdown("**Clean date filter:**")
    col_days1, col_days2 = st.columns([1, 3])
    with col_days1:
        days_back = st.selectbox(
            "Keep comments from last",
            [7, 14, 30, None],
            index=[7, 14, 30, None].index(st.session_state.get("pipeline_days_back", 30)),
            format_func=lambda x: {7: "7 days", 14: "14 days", 30: "30 days", None: "All time"}[x],
            key="pipeline_days_back",
        )
    with col_days2:
        if days_back is not None:
            st.caption(f"Only comments published within the **last {days_back} days** will be kept during cleaning.")
        else:
            st.caption("Keeping **all comments** regardless of publication date.")

    section_divider()

    disk_videos, disk_comments, _ = load_latest_collected(BASE_DIR / "data" / "collection_output")
    if disk_videos.empty:
        st.warning("⬆️ No collection data found. Run **Start Collection** above first.")
        return

    disk_cleaned = load_cleaned_data()
    session_cleaned = st.session_state.step3_full_cleaned
    pipeline_done = (not disk_cleaned.empty or not session_cleaned.empty)
    if pipeline_done:
        status_box("Pipeline complete — data is ready for Step 3 (LLM Signals).", "done")

    btn_label = "🔄 Re-run Pipeline" if pipeline_done else "🚀 Run Full Pipeline"
    if st.button(btn_label, type="primary", disabled=st.session_state.pipeline_running):
        st.session_state.pipeline_running = True
        pbar = st.progress(0)
        pstatus = st.empty()
        try:
            src_videos, src_comments, cleaned_df, summary = run_full_pipeline(
                progress_bar=pbar, status_text=pstatus, days_back=days_back
            )
        except Exception as e:
            pbar.empty()
            pstatus.empty()
            st.session_state.pipeline_running = False
            st.error(f"Pipeline failed: {e}")
            st.stop()
        pbar.empty()
        pstatus.empty()
        st.session_state.pipeline_running = False
        st.session_state.step3_full_cleaned = cleaned_df
        st.session_state.step3_summary = summary
        st.session_state.step3_master_videos = len(src_videos)
        st.session_state.step3_master_comments = len(src_comments)
        st.session_state.step3_done = True
        flash("✅ Full pipeline complete! Head to **Step 3** for LLM analysis.", "success")

    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # FULL PIPELINE RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🔄 Full Pipeline Results")

    cleaned = st.session_state.step3_full_cleaned
    if cleaned.empty:
        cleaned = disk_cleaned

    if not cleaned.empty:
        master_videos_n = st.session_state.step3_master_videos
        master_comments_n = st.session_state.step3_master_comments
        linked_check = load_linked_data()

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        with col_m1: metric_card("Master Videos", master_videos_n)
        with col_m2: metric_card("Master Comments", master_comments_n)
        with col_m3: metric_card("Linked Comments", len(linked_check))
        with col_m4: metric_card("Retained", len(cleaned))
        with col_m5:
            rate = 100 * len(cleaned) / max(1, master_comments_n)
            metric_card("Retention Rate", f"{rate:.1f}%")

        section_divider()

        tab_stat, tab_cat, tab_signal = st.tabs(
            ["📊 Clean Stats", "📂 Categories", "🔍 Pre-Detected Signals"]
        )

        with tab_stat:
            st.markdown("##### Cleaning Statistics")
            if "priority_level" in cleaned.columns:
                priority_order = ["high", "medium", "general"]
                priority_counts = cleaned["priority_level"].value_counts().reindex(priority_order, fill_value=0)
                col_p1, col_p2 = st.columns([1, 1])
                with col_p1:
                    fig_p = px.bar(x=priority_order, y=priority_counts.values, title="Priority Level Distribution",
                                    labels={"x": "Priority", "y": "Comment Count"}, color=priority_order,
                                    color_discrete_map={"high": "#f85149", "medium": "#f0c000", "general": "#8b949e"})
                    fig_p.update_layout(template="plotly_white", height=300)
                    st.plotly_chart(fig_p, width='stretch')
                with col_p2:
                    for lvl in priority_order:
                        cnt = int(priority_counts.get(lvl, 0))
                        pct = 100 * cnt / max(1, len(cleaned))
                        bar_len = int(pct / 2)
                        bar = "█" * bar_len + "░" * (50 - bar_len)
                        st.markdown(f"**{lvl.upper()}** `{bar}` {cnt:,} ({pct:.1f}%)")
                section_divider()
            if "clean_text" in cleaned.columns:
                fig_len = px.histogram(cleaned, x=cleaned["clean_text"].str.len(), nbins=30,
                                       title="Clean Text Length Distribution",
                                       labels={"x": "Character Count", "y": "Comment Count"},
                                       color_discrete_sequence=["#79c0ff"])
                fig_len.update_layout(template="plotly_white", height=300)
                st.plotly_chart(fig_len, width='stretch')

        with tab_cat:
            st.markdown("##### Product Category Distribution")
            if "product_categories" in cleaned.columns:
                cat_exploded = cleaned["product_categories"].str.split("|").explode()
                cat_counts = cat_exploded[cat_exploded != "unknown"].value_counts()
                col_c1, col_c2 = st.columns([2, 1])
                with col_c1:
                    fig_cat = px.bar(cat_counts.reset_index(), x="product_categories",
                                      y=cat_counts.reset_index().columns[1],
                                      title="Comments per Product Category",
                                      labels={"product_categories": "Category",
                                              cat_counts.reset_index().columns[1]: "Count"},
                                      color=cat_counts.reset_index().columns[1],
                                      color_continuous_scale="Viridis")
                    fig_cat.update_layout(template="plotly_white", height=400, xaxis_tickangle=-45,
                                         hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"))
                    fig_cat.update_traces(hovertemplate="%{x}<br>Count: %{y}<extra></extra>")
                    st.plotly_chart(fig_cat, width='stretch')
                with col_c2:
                    st.markdown("**Category Breakdown**")
                    for cat, cnt in cat_counts.items():
                        pct = 100 * cnt / max(1, cat_counts.sum())
                        st.markdown(f"- **{cat}**: {cnt:,} ({pct:.1f}%)")
            else:
                st.info("No product_categories column found in cleaned data.")

        with tab_signal:
            st.markdown("##### Rule-Based Demand Signal Pre-Detection")
            if "demand_signals" in cleaned.columns:
                sig_exploded = cleaned["demand_signals"].str.split("|").explode()
                sig_counts = sig_exploded[sig_exploded != "general"].value_counts()
                if not sig_counts.empty:
                    col_s1, col_s2 = st.columns([2, 1])
                    with col_s1:
                        fig_sig = px.bar(sig_counts.reset_index(), x="demand_signals",
                                          y=sig_counts.reset_index().columns[1],
                                          title="Rule-Based Demand Signal Types",
                                          labels={"demand_signals": "Signal Type",
                                                  sig_counts.reset_index().columns[1]: "Count"},
                                          color=sig_counts.reset_index().columns[1],
                                          color_continuous_scale="Mint")
                        fig_sig.update_layout(template="plotly_white", height=400, xaxis_tickangle=-30,
                                            hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"))
                        fig_sig.update_traces(hovertemplate="%{x}<br>Count: %{y}<extra></extra>")
                        st.plotly_chart(fig_sig, width='stretch')
                    with col_s2:
                        st.markdown("**Signal Breakdown**")
                        for sig, cnt in sig_counts.items():
                            pct = 100 * cnt / max(1, sig_counts.sum())
                            st.markdown(f"- **{sig}**: {cnt:,} ({pct:.1f}%)")
                else:
                    st.info("No non-general signals detected in cleaned data.")
            else:
                st.info("No demand_signals column found in cleaned data.")


    section_divider()

    # ── Standalone Full and Cleaned tables ──────────────────────────────────────
    linked = load_linked_data()
    if not linked.empty and not cleaned.empty:
        st.markdown("### 📋 Data Tables")
        tab_full, tab_clean = st.tabs(["💬 Full Comments (Linked)", "🧹 Cleaned Comments"])
        with tab_full:
            st.markdown("##### Full Comments — Original YouTube Text")
            full_show_cols = ["comment_id", "video_id", "title", "channel_title", "author_display_name",
                             "text_original", "like_count"]
            full_show = linked[[c for c in full_show_cols if c in linked.columns]].head(50)
            st.dataframe(full_show, width='stretch', height=500)
            download_button("⬇ Download Full Comments", linked, "full_comments", "s2_full_cmt")
        with tab_clean:
            st.markdown("##### Cleaned Comments — After Filtering")
            clean_show_cols = ["comment_id", "video_id", "title", "channel_title",
                               "priority_level", "demand_signals", "product_categories",
                               "text_original", "clean_text"]
            clean_show = cleaned[[c for c in clean_show_cols if c in cleaned.columns]].head(50)
            st.dataframe(clean_show, width='stretch', height=500)
            download_button("⬇ Download Cleaned Comments", cleaned, "cleaned_comments", "s2_clean_cmt")
    elif not linked.empty:
        st.markdown("### 📋 Data Tables")
        st.markdown("##### Full Comments")
        full_show_cols = ["comment_id", "video_id", "title", "channel_title",
                         "author_display_name", "text_original", "like_count"]
        full_show = linked[[c for c in full_show_cols if c in linked.columns]].head(50)
        st.dataframe(full_show, width='stretch', height=500)
        download_button("⬇ Download Full Comments", linked, "full_comments", "s2_full_cmt")
    else:
        st.info("No linked data found. Run the Full Pipeline first.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: STEP 3 — LLM SIGNALS
# ════════════════════════════════════════════════════════════════════════════════

def render_step3_llm():
    # ── Section anchor for sticky nav ─────────────────────────────────────────
    st.markdown('<div id="step-3-llm"></div>', unsafe_allow_html=True)
    st.markdown("## 🤖 Step 3: LLM Demand Signal Detection")
    st.markdown(
        "Two-phase AI classification pipeline:\n"
        "1. **Phase 1 — Classification**: Classify ALL comments as demand_signal or no_signal (simple, fast)\n"
        "2. **Phase 2 — Scoring**: Score only non-no_signal comments on 14 dimensions\n\n"
        "Each phase can be run independently and checkpoints automatically. "
        "Supports **DeepSeek** and **Google Gemini**."
    )

    full_df, signal_df = pd.DataFrame(), pd.DataFrame()
    try:
        full_df, signal_df = load_latest_demand_signals()
    except Exception:
        pass

    # ── Common settings ─────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Settings")
    model_provider = st.selectbox(
        "Model Provider",
        ["deepseek", "gemini"],
        index=["deepseek", "gemini"].index(st.session_state.get("s3_model_provider", "deepseek")),
        format_func=lambda x: {"deepseek": "🔵 DeepSeek", "gemini": "🟠 Google Gemini"}[x],
        key="s3_model_provider",
    )
    if model_provider == "deepseek":
        api_placeholder = "Enter your DeepSeek API key"
        api_help = "Get from https://platform.deepseek.com/"
        model_options = ["deepseek-chat", "deepseek-coder"]
    else:
        api_placeholder = "Enter your Gemini API key"
        api_help = "Get from https://aistudio.google.com/app/apikey or https://ai.google.dev/"
        model_options = [
            "gemini-3.1-pro-preview", "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview", "gemini-2.5-flash", "gemini-2.5-flash-lite",
        ]
    st.markdown(f"**{api_placeholder}** <span style='color:red'>*(required)*</span>", unsafe_allow_html=True)
    api_key = st.text_input(
        "API Key", type="password",
        help=api_help, key="s3_api_key",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        batch_size = st.number_input(
            "Batch size (comments/call)", min_value=5, max_value=50,
            value=st.session_state.get("s3_batch_size", 20), key="s3_batch_size",
        )
    with col_b:
        rate_delay = st.slider(
            "Rate limit delay (s)", 0.5, 5.0,
            value=st.session_state.get("s3_rate_delay", 1.5), step=0.5, key="s3_rate_delay",
        )
    model_name = st.selectbox(
        "Model", model_options, index=0, key="s3_model_name",
    )

    if not api_key:
        st.markdown(":red[**⚠️ API key is required to run LLM analysis.**]", unsafe_allow_html=True)

    with st.expander("🔍 Signal Labels Reference", expanded=False):
        st.markdown("""
        | Label | Description |
        |-------|-------------|
        | **purchase_intent** | Explicit desire or plan to buy a protective case |
        | **problem_complaint** | Frustration about equipment damage or lack of protection |
        | **comparison_research** | Comparing or researching different cases |
        | **usage_scenario** | Describes a specific use context requiring protection |
        | **wishful_thinking** | Wishes they had bought or owned protection |
        | **supply_recommendation** | Recommends or praises a specific case product |
        | **no_signal** | No demand signal for protective cases |
        """)

    # ── Load cleaned data ────────────────────────────────────────────────────────
    cleaned = st.session_state.step3_full_cleaned if not st.session_state.step3_full_cleaned.empty else load_cleaned_data()
    total_comments = len(cleaned) if not cleaned.empty else 0
    if total_comments == 0:
        st.warning("No cleaned data found. Run the Full Pipeline in Step 2 first.")
        return

    # Read Phase 1 checkpoint BEFORE n_to_analyze input so default/max can reflect remaining
    import json as _json
    cp1_files = sorted(CHECKPOINT_DIR.glob(f"llm_{model_provider}_phase1_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    cp2_files = sorted(CHECKPOINT_DIR.glob(f"llm_{model_provider}_phase2_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    p1_done = 0
    raw_p1_count = 0
    if cp1_files:
        try:
            with open(cp1_files[0]) as f:
                p1_data = _json.load(f)
            raw_p1_count = len(p1_data.get("results", p1_data)) if isinstance(p1_data, dict) else len(p1_data)
            p1_done = raw_p1_count
        except Exception:
            p1_done = 0

    remaining_p1 = max(0, total_comments - p1_done)
    st.markdown(f"**Total cleaned comments: {total_comments:,}**"
                + (f" | **Phase 1 checkpoint: {p1_done:,} done — {remaining_p1:,} remaining**" if p1_done > 0 else ""))
    # default to remaining, but user can still type the original total (e.g. 300 even if 100 done)
    n_to_analyze = st.number_input(
        "Comments to analyze",
        min_value=p1_done + 1, max_value=total_comments,
        value=min(p1_done + remaining_p1, total_comments),
        key="s3_n_to_analyze",
        help=f"Total available: {total_comments:,} | Already done: {p1_done:,}".strip("| "),
    )
    if n_to_analyze < 1:
        st.markdown(":red[**⚠️ At least 1 comment must be analyzed.**]", unsafe_allow_html=True)

    # ── Checkpoint status ───────────────────────────────────────────────────────
    p1_checkpoint_mismatch = (p1_done > 0 and raw_p1_count > n_to_analyze)

    p2_done = 0
    if cp2_files:
        try:
            with open(cp2_files[0]) as f:
                p2_data = _json.load(f)
            p2_done = min(
                len(p2_data.get("results", p2_data)) if isinstance(p2_data, dict) else len(p2_data),
                n_to_analyze,
            )
        except Exception:
            p2_done = 0

    # ── Phase 1 results path ─────────────────────────────────────────────────────
    phase1_parquet_path = OUTPUT_DIR / f"phase1_results_{model_provider}_latest.parquet"

    section_divider()

    # ════════════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Classification
    # ════════════════════════════════════════════════════════════════════════════════
    st.markdown("### 🔍 Phase 1: Classification")
    if p1_checkpoint_mismatch:
        st.warning(f"⚠️ Checkpoint mismatch: found {raw_p1_count:,} results from a previous run (total={total_comments:,}). "
                   "Click **Clear Phase 1 Checkpoint** to start fresh.")
    elif p1_done >= n_to_analyze:
        st.success(f"✅ Phase 1 complete — {p1_done:,}/{total_comments:,} comments classified.")
    elif p1_done > 0:
        st.info(f"📌 Phase 1 checkpoint — {p1_done:,}/{total_comments:,} classified ({100*p1_done/total_comments:.1f}%). Click to resume from comment {p1_done + 1}.")
    elif phase1_parquet_path.exists():
        st.success(f"✅ Phase 1 complete — {n_to_analyze:,}/{n_to_analyze:,} comments classified.")
    else:
        st.info("Phase 1 not started — click **Run Phase 1** to begin classification.")

    p1col1, p1col2 = st.columns([1, 3])
    with p1col1:
        p1_btn = st.button(
            "🚀 Run Phase 1",
            type="primary",
            disabled=st.session_state.llm_phase1_started
            or not api_key
            or total_comments == 0
            or (phase1_parquet_path.exists() and p1_done >= n_to_analyze),
        )
    with p1col2:
        clear1_disabled = not cp1_files and not phase1_parquet_path.exists()
        if st.button(
            "🗑 Clear Phase 1 Checkpoint",
            help="Delete Phase 1 checkpoint and results to restart from comment 0",
            disabled=clear1_disabled or p1_checkpoint_mismatch,
        ):
            for f in cp1_files:
                f.unlink()
            if phase1_parquet_path.exists():
                phase1_parquet_path.unlink()
            st.success("Phase 1 checkpoint and results cleared. Will restart from comment 0.")
            st.rerun()

    if p1_btn:
        st.session_state.llm_phase1_started = True
        progress_bar = st.progress(0)
        status_text = st.empty()

        def p1_progress_cb(batch_idx, n_batches, processed, total, elapsed=0, eta=0, results_count=0, phase=None):
            pct = int(100 * processed / total)
            progress_bar.progress(pct)
            speed = processed / max(elapsed, 0.1)
            status_text.text(
                f"[Phase 1] Batch {batch_idx+1}/{n_batches} | {processed:,}/{total:,} ({pct}%) | "
                f"~{speed:.1f} cmt/s | ETA ~{eta:.0f}s"
            )

        try:
            with st.spinner("Phase 1: Classifying comments as demand signal or no_signal..."):
                p1_df, p1_signals = run_phase1_classification(
                    api_key=api_key,
                    input_df=cleaned.head(n_to_analyze),
                    model_provider=model_provider,
                    model_name=model_name,
                    batch_size=batch_size,
                    rate_limit_delay=rate_delay,
                    progress_callback=p1_progress_cb,
                    n_to_analyze=n_to_analyze,
                )
            st.session_state.llm_phase1_started = False
            st.session_state.llm_phase1_done = True
            sig_count = len(p1_signals)
            no_sig_count = len(p1_df) - sig_count
            progress_bar.empty()
            status_text.empty()
            st.success(
                f"✅ Phase 1 done! **{sig_count:,}** demand signals / **{no_sig_count:,}** no_signal. "
                f"Run Phase 2 to score the {sig_count:,} signal comments."
            )
        except (QuotaExceededError, LLMCallError) as e:
            progress_bar.empty()
            status_text.empty()
            st.session_state.llm_phase1_started = False
            msg = f"⚠️ **{e.provider} error**: {e.message if hasattr(e,'message') else str(e)}"
            st.error(msg)
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.session_state.llm_phase1_started = False
            st.error(f"⚠️ Unexpected error: {e}")

    # Show Phase 1 preview if checkpoint exists
    # Phase 1 real-time preview (from parquet if complete, otherwise from checkpoint)
    if phase1_parquet_path.exists():
        try:
            preview_df = pd.read_parquet(phase1_parquet_path)
            preview_df["comment_id"] = preview_df["comment_id"].astype(str)
            sig_dist = preview_df["signal"].value_counts()
            total_classified = len(preview_df)
            signal_total = int((preview_df["signal"] != "no_signal").sum())
            with st.expander(f"📋 Phase 1 results — {total_classified:,} classified, {signal_total:,} demand signals ({100*signal_total/max(1,total_classified):.1f}%)", expanded=False):
                st.bar_chart(sig_dist, horizontal=True)
                st.dataframe(preview_df[["comment_id", "signal", "confidence", "reason"]].head(50),
                             width='stretch', hide_index=True)
        except Exception:
            pass
    elif p1_done > 0:
        try:
            p1_data = _json.load(open(cp1_files[0])) if cp1_files else {}
            p1_results = p1_data.get("results", p1_data) if isinstance(p1_data, dict) else p1_data
            p1_preview = pd.DataFrame(p1_results)
            if "signal" in p1_preview.columns:
                sig_dist = p1_preview["signal"].value_counts()
                total_classified = len(p1_preview)
                signal_total = int((p1_preview["signal"] != "no_signal").sum())
                with st.expander(f"📋 Phase 1 checkpoint preview — {total_classified:,} classified, {signal_total:,} signals ({100*signal_total/max(1,total_classified):.1f}%)", expanded=False):
                    st.bar_chart(sig_dist, horizontal=True)
                    reason_cols = [c for c in ["comment_id", "signal", "confidence", "reason"] if c in p1_preview.columns]
                    st.dataframe(p1_preview[reason_cols].head(50), width='stretch', hide_index=True)
        except Exception:
            pass

    section_divider()

    # ════════════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Scoring
    # ════════════════════════════════════════════════════════════════════════════════
    st.markdown("### 📊 Phase 2: Scoring")

    # Try to load Phase 1 results (parquet first, then checkpoint)
    phase1_loaded = False
    phase1_df = pd.DataFrame()
    if phase1_parquet_path.exists():
        try:
            phase1_df = pd.read_parquet(phase1_parquet_path)
            phase1_df["comment_id"] = phase1_df["comment_id"].astype(str)
            phase1_loaded = True
        except Exception:
            pass

    p1_signal_count = int(phase1_df[phase1_df["signal"] != "no_signal"].shape[0]) if phase1_loaded and not phase1_df.empty else 0

    if p2_done > 0 and p1_signal_count > 0:
        st.success(f"📌 Phase 2 checkpoint — {p2_done:,}/{p1_signal_count:,} comments scored. "
                   f"Run Phase 2 to resume.")
    elif phase1_loaded and not phase1_df.empty:
        st.success(f"✅ Phase 1 complete ({p1_signal_count:,} signals found). "
                   f"Click **Run Phase 2** to score on 14 dimensions.")
    elif p1_done > 0:
        st.info(f"Phase 1 checkpoint exists ({p1_done:,} done) but parquet not yet saved. "
                f"Wait for Phase 1 to finish or run it to completion.")
    else:
        st.info("Complete Phase 1 first — Phase 2 scores only non-no_signal comments from Phase 1.")

    p2col1, p2col2 = st.columns([1, 3])
    with p2col1:
        p2_btn = st.button(
            "📊 Run Phase 2",
            type="primary",
            disabled=(
                st.session_state.llm_phase2_started
                or not api_key
                or (not phase1_loaded and p1_done == 0)
                or total_comments == 0
            ),
        )
    with p2col2:
        clear2_disabled = not cp2_files
        if st.button(
            "🗑 Clear Phase 2 Checkpoint",
            help="Delete Phase 2 checkpoint to restart scoring from scratch",
            disabled=clear2_disabled,
        ):
            for f in cp2_files:
                f.unlink()
            st.success("Phase 2 checkpoint cleared.")
            st.rerun()

    if p2_btn:
        st.session_state.llm_phase2_started = True
        progress_bar = st.progress(0)
        status_text = st.empty()

        def p2_progress_cb(batch_idx, n_batches, processed, total, elapsed=0, eta=0, results_count=0, phase=None):
            pct = int(100 * processed / total)
            progress_bar.progress(pct)
            speed = processed / max(elapsed, 0.1)
            status_text.text(
                f"[Phase 2] Batch {batch_idx+1}/{n_batches} | {processed:,}/{total:,} ({pct}%) | "
                f"~{speed:.1f} cmt/s | ETA ~{eta:.0f}s"
            )

        # Load Phase 1 parquet (the durable file saved after Phase 1 completes)
        if not phase1_loaded:
            progress_bar.empty()
            status_text.empty()
            st.session_state.llm_phase2_started = False
            st.error("⚠️ Phase 1 parquet not found. Please run Phase 1 to completion first.")
            return

        try:
            with st.spinner("Phase 2: Scoring demand signals on 14 dimensions..."):
                p2_df = run_phase2_scoring(
                    api_key=api_key,
                    phase1_df=phase1_df,
                    input_df=cleaned.head(n_to_analyze),
                    model_provider=model_provider,
                    model_name=model_name,
                    batch_size=batch_size,
                    rate_limit_delay=rate_delay,
                    progress_callback=p2_progress_cb,
                )
            st.session_state.llm_phase2_started = False
            st.session_state.llm_phase2_done = True
            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Phase 2 done! {len(p2_df):,} comments scored on 14 dimensions.")

            # Refresh full results from disk
            full_df, signal_df = load_latest_demand_signals()
            st.session_state.step4_full = full_df
            st.session_state.step4_signals = signal_df
            st.session_state.step4_done = True

        except (QuotaExceededError, LLMCallError) as e:
            progress_bar.empty()
            status_text.empty()
            st.session_state.llm_phase2_started = False
            msg = f"⚠️ **{e.provider} error**: {e.message if hasattr(e,'message') else str(e)}"
            st.error(msg)
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.session_state.llm_phase2_started = False
            st.error(f"⚠️ Unexpected error: {e}")

    section_divider()

    # ── Show results ────────────────────────────────────────────────────────────
    if not signal_df.empty or not full_df.empty:
        df = signal_df if not signal_df.empty else full_df
        if not df.empty:
            c1, c2, c3, c4 = st.columns(4)
            with c1: metric_card("Total Classified", len(full_df) if not full_df.empty else len(df))
            with c2: metric_card("Demand Signals", len(df),
                                  f"{100*len(df)/max(1,len(full_df) if not full_df.empty else len(df)):.1f}%")
            with c3: metric_card("Unique Videos", df["video_id"].nunique() if "video_id" in df.columns else 0)
            with c4: metric_card("Avg Confidence", f"{df['confidence'].mean():.2f}" if "confidence" in df.columns else "N/A")

            st.markdown("#### Signal Distribution")
            sig_counts = df["signal"].value_counts().reset_index()
            sig_counts.columns = ["signal", "count"]
            fig = px.bar(sig_counts, x="signal", y="count", title="Demand Signal Types",
                         labels={"signal": "Signal Type", "count": "Comment Count"},
                         color="count", color_continuous_scale="Burg")
            fig.update_layout(template="plotly_white", height=400, xaxis_tickangle=-30,
                             hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"))
            fig.update_traces(hovertemplate="%{x}<br>Count: %{y}<extra></extra>")
            st.plotly_chart(fig, width='stretch')

            st.markdown("#### Top Demand Signals (Highest Confidence)")
            display_cols = ["signal", "confidence", "reason", "comment_text", "video_title", "video_url", "channel_title"]
            display = df[[c for c in display_cols if c in df.columns]].head(30)
            st.dataframe(display, width='stretch', height=500)

            dl1, dl2 = st.columns(2)
            with dl1: download_button("⬇ Download Full Results", full_df if not full_df.empty else df, "demand_signals_full", "s3_full")
            with dl2: download_button("⬇ Download Signals Only", df, "demand_signals_only", "s3_sig")
    else:
        st.info("No results yet. Complete both phases above to see demand signal results.")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════

DIM_COLS = [
    "fit_score", "protection_score", "texture_score",
    "yellowing_concern", "installation_ease", "compatibility_score",
    "value_perception", "overall_sentiment_score",
        "sentiment_intensity", "urgency_score", "purchase_intent_score",
    "sarcasm_flag", "expertise_level", "specificity",
]

DIM_LABELS = {
    "fit_score": "Fit Score",
    "protection_score": "Protection Score",
    "texture_score": "Texture Score",
    "yellowing_concern": "Yellowing Concern",
    "installation_ease": "Installation Ease",
    "compatibility_score": "Compatibility Score",
    "value_perception": "Value Perception",
    "overall_sentiment_score": "Overall Sentiment",
    "sentiment_intensity": "Sentiment Intensity",
    "urgency_score": "Urgency Score",
    "purchase_intent_score": "Purchase Intent Score",
    "sarcasm_flag": "Sarcasm Flag",
    "expertise_level": "Expertise Level",
    "specificity": "Specificity",
}

# Detailed explanations for each scoring dimension (shown in an expander in the dashboard)
DIM_EXPLANATIONS: dict[str, dict] = {
    "fit_score": {
        "name": "Fit Score",
        "range": "-1 (loose/bad) → +1 (perfect fit)",
        "meaning": (
            "How well the protective case fits the product it is meant to protect. "
            "A high score indicates the reviewer describes a snug, form-fitting case; "
            "a low score means gaps, wobbling, or incompatibility were reported."
        ),
    },
    "protection_score": {
        "name": "Protection Score",
        "range": "-1 (no protection) → +1 (maximum protection)",
        "meaning": (
            "Perceived level of protection the case provides. "
            "Reviewers mentioning shockproof, drop-tested, rugged build get high scores; "
            "those complaining about fragility or vulnerability get low scores."
        ),
    },
    "texture_score": {
        "name": "Texture Score",
        "range": "-1 (bad texture/grip) → +1 (great texture/grip)",
        "meaning": (
            "Assessment of the case's surface feel — grip, roughness, premium touch. "
            "Positive scores for matte, non-slip, soft-touch, premium textures; "
            "negative for sticky, cheap, slippery, or fingerprint-magnet surfaces."
        ),
    },
    "yellowing_concern": {
        "name": "Yellowing Concern",
        "range": "-1 (no yellowing) → +1 (severe yellowing reported)",
        "meaning": (
            "Degree to which the reviewer complains about the case turning yellow over time "
            "(common in clear/TPU cases). Low scores indicate the case stayed clear; "
            "high scores mean yellowing was explicitly mentioned as a problem."
        ),
    },
    "installation_ease": {
        "name": "Installation Ease",
        "range": "-1 (very difficult to install) → +1 (extremely easy)",
        "meaning": (
            "How easy or difficult the reviewer found putting the case on the device. "
            "High scores for snap-on, easy, fits-perfectly; "
            "low scores for tight fit, hard to clip on, risk of breaking."
        ),
    },
    "compatibility_score": {
        "name": "Compatibility Score",
        "range": "-1 (compatibility issues) → +1 (fully compatible)",
        "meaning": (
            "Whether the case works well with the device's features — "
            "ports, cameras, buttons, wireless charging, screen protectors, etc. "
            "Positive = all features accessible; negative = obstructed ports or blocked buttons."
        ),
    },
    "value_perception": {
        "name": "Value Perception",
        "range": "-1 (overpriced/poor value) → +1 (great value/affordable)",
        "meaning": (
            "Reviewer's sense of whether the case is worth its price. "
            "High scores for budget-friendly, great-value-for-money mentions; "
            "low scores for overpriced, not durable enough for the price."
        ),
    },
    "overall_sentiment_score": {
        "name": "Overall Sentiment",
        "range": "-1 (very negative) → +1 (very positive)",
        "meaning": (
            "Aggregate emotional tone of the comment toward the case/product. "
            "Combines all positive and negative language cues to produce a net sentiment score."
        ),
    },
    "sentiment_intensity": {
        "name": "Sentiment Intensity",
        "range": "0 (neutral/flat) → +1 (highly emotional/emphatic)",
        "meaning": (
            "How strongly the reviewer expresses their opinion — "
            "not whether the sentiment is positive or negative, but how intense it is. "
            "Reviewers using strong language ('absolutely love', 'worst ever') score high."
        ),
    },
    "urgency_score": {
        "name": "Urgency Score",
        "range": "0 (no urgency) → +1 (high urgency to act)",
        "meaning": (
            "Whether the comment expresses urgency — the reviewer needs a solution now, "
            "is actively looking, or has an immediate problem to solve. "
            "High scores for phrases like 'need this ASAP', 'looking for', 'urgent'."
        ),
    },
    "purchase_intent_score": {
        "name": "Purchase Intent Score",
        "range": "0 (no intent) → +1 (definite intent to buy)",
        "meaning": (
            "Likelihood that the commenter will purchase (or has already purchased) the case. "
            "Includes phrases like 'going to buy', 'just ordered', 'already bought'. "
            "High scores = strong buying signal; low scores = just browsing or researching."
        ),
    },
    "sarcasm_flag": {
        "name": "Sarcasm Flag",
        "range": "0 (sincere) → +1 (sarcastic/ironic)",
        "meaning": (
            "Indicator of sarcasm or irony in the comment. "
            "Sarcastic reviews can appear positive on the surface but be negative underneath — "
            "a high score means the LLM detected ironic tone that should be interpreted cautiously."
        ),
    },
    "expertise_level": {
        "name": "Expertise Level",
        "range": "0 (casual user) → +1 (expert/technical)",
        "meaning": (
            "How knowledgeable or technically sophisticated the reviewer appears. "
            "High scores for detailed specs, professional use cases, comparison with alternatives; "
            "low scores for casual, non-technical comments."
        ),
    },
    "specificity": {
        "name": "Specificity",
        "range": "0 (vague/generic) → +1 (highly specific)",
        "meaning": (
            "How concrete and detailed the comment is. "
            "Specific mentions of exact product names, model numbers, use scenarios, "
            "or precise problems score high; generic comments like 'it's good' score low."
        ),
    },
}

SCORE_COLS = [c for c in DIM_COLS if c not in ("sarcasm_flag", "review_quality", "key_phrases_used")]

NEGATIVE_THRESHOLDS = {
    "fit_score": -0.5, "protection_score": -0.5, "texture_score": -0.5,
    "yellowing_concern": -0.5, "installation_ease": -0.5, "compatibility_score": -0.5,
    "value_perception": -0.5, "overall_sentiment_score": -0.5,
    "sentiment_intensity": -0.5, "urgency_score": 0.0,
    "purchase_intent_score": -0.5, "sarcasm_flag": 0.5,
    "expertise_level": 0.0, "specificity": 0.2,
}


def _render_score_heatmap(df, cols, title, height=320):
    sub = df[["comment_id"] + [c for c in cols if c in df.columns]].drop_duplicates("comment_id").head(40)
    sub = sub.dropna(subset=cols, how="all")
    if sub.empty or len([c for c in cols if c in sub.columns]) == 0:
        st.info(f"No score data available for: {title}")
        return
    score_data = sub[cols].apply(pd.to_numeric, errors="coerce")
    score_data.index = sub["comment_id"].astype(str).values

    fig = px.imshow(
        score_data.T,
        labels=dict(x="Comment", y="Dimension", color="Score"),
        title=title,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        zmin=-1, zmax=1,
        text_auto=False,
    )

    if fig.data:
        fig.data[0].hovertemplate = (
            "<b>%{x}</b><br>Dimension: %{y}<br>Score: %{z:.2f}<extra></extra>"
        )

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=height,
        paper_bgcolor=CHART_BG, font=dict(color="#334155", size=10),
        xaxis_tickangle=45,
        hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
    )

    st.plotly_chart(fig, width="stretch")


# ── Diverging bar chart helper ─────────────────────────────────────────────────
def _render_diverging_bar(neg_ratios, threshold=20.0):
    """Render diverging bars: negative = right side, positive = left side."""
    sorted_items = sorted(neg_ratios.items(), key=lambda x: x[1], reverse=True)
    labels = [DIM_LABELS.get(k, k) for k, _ in sorted_items]
    vals = [v * 100 for _, v in sorted_items]
    x_max = max(vals) * 1.1 if vals else 50
    fig = px.bar(
        x=vals, y=labels,
        orientation="h",
        title="Negative Ratio by Dimension",
        labels={"x": "% Negative (higher = more pain)", "y": "Dimension"},
        color=vals,
        color_continuous_scale="Reds_r",
        range_color=[0, max(50, x_max)],
    )
    fig.update_traces(
        marker_line=dict(width=0),
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="#fbbf24", annotation_text=f"⚠ Threshold ({threshold}%)")
    fig.update_layout(
        height=max(250, len(labels) * 32),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, max(50, x_max)], title=""),
        paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", size=10),
        hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS DASHBOARD
# ════════════════════════════════════════════════════════════════════════════════


@st.dialog("🗑️ Delete Signal Version")
def _delete_version_dialog(versions: list[dict]):
    st.warning("This action cannot be undone. The selected version and all its data files will be permanently deleted.")
    delete_map = {v["label"]: v["filename"] for v in versions}
    selected = st.selectbox("Select version to delete", list(delete_map.keys()))
    st.write(f"**Provider:** {next(v['provider'] for v in versions if v['label'] == selected)}")

    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.session_state["_show_delete_dialog"] = False
            st.rerun()
    with col_confirm:
        if st.button("🗑️ Yes, Delete", type="primary", use_container_width=True):
            deleted = _delete_version(delete_map[selected])
            st.session_state["_show_delete_dialog"] = False
            st.success(f"Deleted: {', '.join(deleted)}")
            st.rerun()


def render_dashboard():
    # ── Section anchor for sticky nav ─────────────────────────────────────────
    st.markdown('<div id="step-4-dashboard"></div>', unsafe_allow_html=True)
    st.markdown("## 📈 Results Dashboard")
    st.caption(
        "Comprehensive demand signal and product-review scoring results. "
        "Each comment is classified into a demand signal type and scored on **14 quantitative dimensions**."
    )

    # ── Version selector + Delete (same row) ──────────────────────────────────
    versions = list_signal_versions()
    if versions:
        version_options = {v["label"]: v for v in versions}
        default_label = versions[0]["label"]
        if "dashboard_version" not in st.session_state:
            st.session_state.dashboard_version = default_label

        sel_col, del_col = st.columns([3, 1])
        with sel_col:
            selected_label = st.selectbox(
                "📂 Signal Version",
                list(version_options.keys()),
                index=list(version_options.keys()).index(st.session_state.dashboard_version),
                key="dashboard_version",
            )
        with del_col:
            st.markdown("")  # vertical spacing align
            if st.button("🗑️ Delete Version", key="open_delete_dialog"):
                st.session_state["_show_delete_dialog"] = True

        if st.session_state.get("_show_delete_dialog"):
            _delete_version_dialog(versions)

        chosen = version_options[selected_label]
        full_df, signal_df = load_signal_version(chosen["filename"])
    else:
        st.info("No saved signal versions found. Run Step 3 to generate the first version.")
        full_df, signal_df = pd.DataFrame(), pd.DataFrame()
        signal_df = st.session_state.get("step4_signals", pd.DataFrame())
        full_df = st.session_state.get("step4_full", pd.DataFrame())

    if signal_df.empty:
        st.warning("No demand signal results yet. Complete Step 3 first.")
        return

    for col in DIM_COLS:
        for df_ in (full_df, signal_df):
            if col in df_.columns:
                df_[col] = pd.to_numeric(df_[col], errors="coerce")

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 0 — KPI CARDS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(
        "<div style='border-top: 3px solid #3b82f6; margin: 0.75rem 0 1.5rem; "
        "padding-top: 0;'></div>",
        unsafe_allow_html=True,
    )
    _section_header(0, "Key Performance Indicators", "Snapshot of demand signal detection results")
    high_conf = len(signal_df[signal_df["confidence"] >= 0.8])
    avg_conf = signal_df["confidence"].mean()
    _kpi_row([
        {"label": "Total Signals", "value": f"{len(signal_df):,}"},
        {"label": "Avg Confidence", "value": f"{avg_conf:.2f}", "sub": f"{high_conf:,} ≥0.8"},
        {"label": "Videos Covered", "value": f"{signal_df['video_id'].nunique():,}"},
        {"label": "Channels", "value": f"{signal_df['channel_title'].nunique():,}"},
        {"label": "Signal Types", "value": f"{signal_df['signal'].nunique()}"},
    ])
    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — SIGNAL CLASSIFICATION
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(1, "Demand Signal Classification", "What types of demand signals are present")
    sig_counts = signal_df["signal"].value_counts()
    top_signal = sig_counts.idxmax() if not sig_counts.empty else "N/A"

    sig_col, conf_col, sent_col = st.columns([1, 1, 1])

    # Donut + legend
    with sig_col:
        fig_donut = px.pie(
            sig_counts.reset_index(), values=sig_counts.reset_index().columns[1],
            names="signal", title="Signal Type Distribution",
            hole=0.55,
            color="signal",
            color_discrete_sequence=SIGNAL_COLORS,
        )
        fig_donut.update_layout(
            template=PLOTLY_TEMPLATE, height=300,
            paper_bgcolor=CHART_BG, font=dict(color="#334155", size=10),
            title=dict(font=dict(size=12, color="#1e293b")),
            legend=dict(font=dict(color="#334155", size=10)),
            showlegend=True,
        )
        fig_donut.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(fig_donut, width="stretch")

    # Confidence violin
    with conf_col:
        if "confidence" in signal_df.columns:
            fig_violin = px.violin(
                signal_df, x="signal", y="confidence",
                title="Confidence Distribution",
                color="signal", box=True, points="outliers",
                color_discrete_sequence=SIGNAL_COLORS,
            )
            fig_violin.update_layout(
                template=PLOTLY_TEMPLATE, height=300,
                paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#334155", size=10),
                title=dict(font=dict(size=12, color="#1e293b")),
                xaxis_tickangle=-30,
                showlegend=False,
                hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
            )
            st.plotly_chart(fig_violin, width="stretch")

    # Overall sentiment box
    with sent_col:
        if "overall_sentiment_score" in signal_df.columns:
            fig_sb = px.box(
                signal_df, x="signal", y="overall_sentiment_score",
                title="Sentiment by Signal Type",
                color="signal", color_discrete_sequence=SIGNAL_COLORS,
                points="outliers",
            )
            fig_sb.update_layout(
                template=PLOTLY_TEMPLATE, height=300,
                paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#334155", size=10),
                title=dict(font=dict(size=12, color="#1e293b")),
                xaxis_tickangle=-30, showlegend=False,
                hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
            )
            st.plotly_chart(fig_sb, width="stretch")

    # Summary stats row
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.markdown(f"<div class='chart-card'><div class='chart-title'>Top Signal Type</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:#818cf8'>{top_signal}</div></div>",
                    unsafe_allow_html=True)
    with col_stat2:
        st.markdown(f"<div class='chart-card'><div class='chart-title'>Dominant %</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:#818cf8'>"
                    f"{100*sig_counts.iloc[0]/len(signal_df):.1f}%</div></div>",
                    unsafe_allow_html=True)
    with col_stat3:
        neg_signals = (signal_df["overall_sentiment_score"] < 0).sum() if "overall_sentiment_score" in signal_df.columns else 0
        st.markdown(f"<div class='chart-card'><div class='chart-title'>Negative Sentiment</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:#f87171'>{neg_signals:,}</div></div>",
                    unsafe_allow_html=True)
    with col_stat4:
        high_risk_signals = len(signal_df[signal_df["confidence"] < 0.4]) if "confidence" in signal_df.columns else 0
        st.markdown(f"<div class='chart-card'><div class='chart-title'>Low Confidence</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:#fbbf24'>{high_risk_signals:,}</div></div>",
                    unsafe_allow_html=True)
    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — SCORING DIMENSION OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(2, "Scoring Dimension Overview", "Average score per dimension — radar + ranked bar")

    with st.expander("📖 What do these scores mean? Click to expand dimension explanations", expanded=False):
        st.markdown("Each dimension is scored by the LLM on a **-1 to +1 scale** (unless noted). "
                    "Click any row below to see the full meaning and scoring logic.")
        for col_key, info in DIM_EXPLANATIONS.items():
            with st.expander(f"**{info['name']}** `({col_key})` — {info['range']}"):
                st.markdown(f"**Meaning:** {info['meaning']}")

    avail_score_cols = [c for c in SCORE_COLS if c in signal_df.columns]
    if avail_score_cols:
        mean_scores = {c: signal_df[c].mean() for c in avail_score_cols}
        sorted_dims = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
        dim_names = [DIM_LABELS.get(k, k) for k, _ in sorted_dims]
        dim_vals = [v for _, v in sorted_dims]

        col_rad, col_bar = st.columns([1, 1])
        with col_rad:
            radar_labels = [DIM_LABELS.get(c, c) for c in avail_score_cols]
            radar_values = [float(mean_scores.get(c, 0)) for c in avail_score_cols]
            fig_radar = px.line_polar(
                r=radar_values + [radar_values[0]],
                theta=radar_labels + [radar_labels[0]],
                title="Average Scores — Radar",
                line_close=True,
            )
            fig_radar.update_traces(
                fill="toself", fillcolor="rgba(99,179,237,0.15)",
                line_color="#63b3ed", line_width=2,
                hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
            )
            fig_radar.update_layout(
                template=PLOTLY_TEMPLATE, height=380,
                paper_bgcolor=CHART_BG,
                font=dict(color="#334155", size=10),
                polar=dict(
                    bgcolor=CHART_BG,
                    radialaxis=dict(gridcolor="#e2e8f0", tickfont=dict(color="#64748b")),
                    angularaxis=dict(linecolor="#e2e8f0"),
                ),
                hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
            )
            st.plotly_chart(fig_radar, width="stretch")

        with col_bar:
            colors_bar = [
                "#f87171" if v < -0.3 else "#fbbf24" if v < 0 else "#86efac" if v > 0.3 else "#facc15"
                for v in dim_vals
            ]
            fig_bar2 = px.bar(
                x=dim_names, y=dim_vals,
                title="Ranked Average Scores",
                labels={"x": "Dimension", "y": "Average Score"},
                color=dim_vals,
                color_continuous_scale="RdYlGn",
                range_color=[-1, 1],
                text=[f"{v:.2f}" for v in dim_vals],
            )
            fig_bar2.update_traces(
                textposition="outside", textfont_size=9, textfont_color="#334155",
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            )
            fig_bar2.update_layout(
                template=PLOTLY_TEMPLATE, height=380,
                paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#334155", size=10),
                title=dict(font=dict(size=12, color="#1e293b")),
                xaxis_tickangle=-45, yaxis=dict(range=[-1.1, 1.1]),
                hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
            )
            st.plotly_chart(fig_bar2, width="stretch")
    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — SCORE DISTRIBUTIONS
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(3, "Score Distributions", "Histogram of each dimension score")
    score_visual_cols = [c for c in SCORE_COLS if c in signal_df.columns and c != "overall_sentiment_score"]
    n_charts = len(score_visual_cols)
    cols_per_row = 3
    rows = (n_charts + cols_per_row - 1) // cols_per_row
    for row_i in range(rows):
        row_cols = st.columns(cols_per_row)
        for col_j in range(cols_per_row):
            idx = row_i * cols_per_row + col_j
            if idx >= n_charts:
                with row_cols[col_j]:
                    st.empty()
                continue
            col_name = score_visual_cols[idx]
            with row_cols[col_j]:
                fig_h = px.histogram(
                    signal_df, x=col_name, nbins=20,
                    title=DIM_LABELS.get(col_name, col_name),
                    labels={"x": "Score", "y": "Count"},
                    color_discrete_sequence=["#63b3ed"],
                    marginal="box",
                )
                fig_h.update_layout(
                    template=PLOTLY_TEMPLATE, height=260, showlegend=False,
                    paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#334155", size=10),
                    title=dict(font=dict(size=11, color="#e2e8f0")),
                    hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
                )
                st.plotly_chart(fig_h, width="stretch")
    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — NEGATIVE RATIO ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(4, "Negative Ratio Analysis", "Dimensions where users express the most pain")
    neg_ratios = {}
    for col in SCORE_COLS:
        if col in signal_df.columns:
            threshold = NEGATIVE_THRESHOLDS.get(col, -0.5)
            total = signal_df[col].notna().sum()
            if total > 0:
                if col == "sarcasm_flag":
                    neg_ratios[col] = signal_df[col].sum() / total
                else:
                    neg_ratios[col] = (signal_df[col] < threshold).sum() / total
    if neg_ratios:
        neg_sorted = sorted(neg_ratios.items(), key=lambda x: x[1], reverse=True)
        pain_points = [(k, v) for k, v in neg_sorted if v >= 0.2]

        col_neg_chart, col_neg_info = st.columns([2, 1])
        with col_neg_chart:
            fig_neg = _render_diverging_bar(dict(neg_sorted))
            fig_neg.update_layout(
                paper_bgcolor=CHART_BG, font=dict(color="#334155", size=10),
            )
            st.plotly_chart(fig_neg, width="stretch")
        with col_neg_info:
            inner = "<div class='chart-card'>"
            inner += "<div class='chart-title'>⚠️ Pain Points (≥20% negative)</div>"
            if pain_points:
                for k, v in pain_points:
                    pct = v * 100
                    bar_len = int(min(pct / 2, 25))
                    bar = "█" * bar_len + "░" * (25 - bar_len)
                    inner += (
                        f"<div class='pain-point'>"
                        f"<div class='pain-point-title'>{DIM_LABELS.get(k, k)}</div>"
                        f"<div style='font-size:0.78rem;margin-top:2px'>{bar} {pct:.1f}%</div>"
                        f"</div>"
                    )
            else:
                inner += "<div style='color:#16a34a;font-size:0.82rem'>No major pain points detected — user sentiment is generally positive.</div>"
            inner += "</div>"
            st.html(inner)
    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — HIGH-RISK COMMENT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(5, "High-Risk Comment Analysis", "Comments with multiple negative dimensions")
    risk_score = pd.Series(0.0, index=signal_df.index)
    for col in SCORE_COLS:
        if col in signal_df.columns and col != "sarcasm_flag":
            threshold = NEGATIVE_THRESHOLDS.get(col, -0.5)
            risk_score = risk_score + (signal_df[col] < threshold).astype(float)
    high_risk_mask = risk_score >= 2
    if high_risk_mask.sum() > 0:
        high_risk_df = signal_df[high_risk_mask].copy()
        high_risk_df["risk_score"] = risk_score[high_risk_mask].values

        _kpi_row([
            {"label": "High-Risk Comments", "value": f"{len(high_risk_df):,}"},
            {"label": "High-Risk %", "value": f"{100*len(high_risk_df)/max(1,len(signal_df)):.1f}%"},
            {"label": "Avg Risk Score", "value": f"{high_risk_df['risk_score'].mean():.1f}"},
            {"label": "Max Risk Score", "value": f"{int(high_risk_df['risk_score'].max())}"},
        ])

        col_scatter, col_hist = st.columns([1, 1])
        with col_scatter:
            scatter_df = high_risk_df.copy()
            if "overall_sentiment_score" in scatter_df.columns:
                fig_scatter = px.scatter(
                    scatter_df,
                    x="overall_sentiment_score",
                    y="confidence",
                    size="risk_score",
                    color="risk_score",
                    color_continuous_scale="Reds",
                    title="Risk Score vs Sentiment & Confidence",
                    labels={
                        "overall_sentiment_score": "Sentiment",
                        "confidence": "Confidence",
                        "risk_score": "Risk Score",
                    },
                    hover_data=["signal", "comment_text"],
                    size_max=14,
                )
                fig_scatter.update_layout(
                    template=PLOTLY_TEMPLATE, height=300,
                    paper_bgcolor=CHART_BG, font=dict(color="#334155", size=10),
                    title=dict(font=dict(size=12, color="#1e293b")),
                    hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
                )
                scatter_text = scatter_df["comment_text"].fillna("").str[:80].str.replace(r"\n", " ", regex=True) + scatter_df["comment_text"].fillna("").str[80:].apply(lambda x: "..." if len(str(x)) > 80 else "")
                fig_scatter.update_traces(
                    customdata=scatter_df["comment_id"].values,
                    hovertemplate=(
                        "<b>Sentiment:</b> %{x:.3f}<br>"
                        "<b>Confidence:</b> %{y:.3f}<br>"
                        "<b>Risk:</b> %{marker.size:.0f}<br>"
                        "<b>Comment:</b> %{text}<extra></extra>"
                    ),
                    text=scatter_text.values,
                )
                st.plotly_chart(fig_scatter, width="stretch")
        with col_hist:
            fig_risk = px.histogram(
                high_risk_df, x="risk_score", nbins=15,
                title="Risk Score Distribution",
                color_discrete_sequence=["#f97316"],
                labels={"x": "Risk Score", "y": "Comment Count"},
            )
            fig_risk.update_layout(
                template=PLOTLY_TEMPLATE, height=300,
                paper_bgcolor=CHART_BG, font=dict(color="#334155", size=10),
                title=dict(font=dict(size=12, color="#1e293b")),
                showlegend=False, hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
            )
            st.plotly_chart(fig_risk, width="stretch")

        st.markdown("**Top 20 High-Risk Comments**")
        display_cols = ["risk_score", "signal", "confidence",
                        "fit_score", "protection_score", "value_perception", "comment_text"]
        disp = high_risk_df[[c for c in display_cols if c in high_risk_df.columns]].sort_values("risk_score", ascending=False).head(20)
        st.dataframe(disp, width="stretch", height=380)
    else:
        st.success("No high-risk comments detected. Overall product sentiment is positive.")
    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — SENTIMENT & BEHAVIORAL SIGNALS
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(6, "Sentiment & Behavioral Signals", "How users feel and how urgently they express demand")
    col_sent, col_urg, col_intent = st.columns(3)

    with col_sent:
        if "sentiment_intensity" in signal_df.columns:
            bins = [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]
            labels_s = ["Strong Neg", "Moderate Neg", "Neutral", "Moderate Pos", "Strong Pos"]
            sentiment_cats = pd.cut(signal_df["sentiment_intensity"], bins=bins, labels=labels_s)
            sent_counts = sentiment_cats.value_counts().reindex(labels_s, fill_value=0)
            fig_sent2 = px.bar_polar(
                r=sent_counts.values.tolist(),
                theta=labels_s,
                title="Sentiment Intensity",
                color=labels_s,
                color_discrete_map={
                    "Strong Neg":     "#ef9a9a",
                    "Moderate Neg":   "#ffcc80",
                    "Neutral":        "#cfd8dc",
                    "Moderate Pos":   "#80cbc4",
                    "Strong Pos":     "#a5d6a7",
                },
                template="plotly_white",
            )
            fig_sent2.update_traces(
                type="barpolar",
                marker_line_color="#ffffff", marker_line_width=1.5,
                hovertemplate="%{theta}: %{r}<extra></extra>",
            )
            fig_sent2.update_layout(
                polar=dict(
                    bgcolor=CHART_BG,
                    radialaxis=dict(gridcolor="#e2e8f0", tickfont=dict(color="#64748b")),
                    angularaxis=dict(tickfont=dict(color="#64748b", size=10)),
                ),
                paper_bgcolor=CHART_BG, font=dict(color="#334155", size=10),
                title=dict(font=dict(size=12, color="#1e293b")),
                showlegend=False, hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
                height=340,
            )
            st.plotly_chart(fig_sent2, width="stretch")
            st.caption(f"Positive: {sent_counts.iloc[-2:].sum():,} | Negative: {sent_counts.iloc[:2].sum():,}")

    with col_urg:
        if "urgency_score" in signal_df.columns:
            urgency_bins = [0, 0.2, 0.4, 0.7, 1.0]
            urgency_labels = ["None", "Low", "Medium", "High"]
            urg_cats = pd.cut(
                signal_df["urgency_score"], bins=urgency_bins,
                labels=urgency_labels, include_lowest=True,
            )
            urg_counts = urg_cats.value_counts().reindex(urgency_labels, fill_value=0)
            fig_urg = px.bar(
                x=urgency_labels, y=urg_counts.values,
                title="Urgency Level",
                labels={"x": "Urgency", "y": "Count"},
                color=urgency_labels,
                color_discrete_map={
                    "None":   "#eceff1",
                    "Low":    "#b3e5fc",
                    "Medium": "#ffe0b2",
                    "High":   "#ffab91",
                },
                text=urg_counts.values,
            )
            fig_urg.update_traces(
                textposition="auto",
                insidetextanchor="middle",
                textfont_color="#374151",
                marker_line_color="#ffffff", marker_line_width=1,
                hovertemplate="%{x}: %{y}<extra></extra>",
            )
            fig_urg.update_layout(
                paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#334155", size=10),
                title=dict(font=dict(size=12, color="#1e293b")),
                xaxis_tickangle=0, showlegend=False,
                hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
                height=360,
                yaxis=dict(range=[0, None], constrain="range"),
            )
            st.plotly_chart(fig_urg, width="stretch")
            st.caption(f"High urgency: {urg_counts.get('High', 0):,} comments")

    with col_intent:
        if "purchase_intent_score" in signal_df.columns:
            intent_bins = [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]
            intent_labels = ["Strong Detract", "Moderate Detract", "Neutral", "Moderate Pos", "Strong Pos"]
            intent_cats = pd.cut(signal_df["purchase_intent_score"], bins=intent_bins, labels=intent_labels)
            intent_counts = intent_cats.value_counts().reindex(intent_labels, fill_value=0)
            fig_int = px.bar(
                x=intent_labels, y=intent_counts.values,
                title="Purchase Intent",
                labels={"x": "Purchase Intent", "y": "Count"},
                color=intent_labels,
                color_discrete_map={
                    "Strong Detract":  "#ef9a9a",
                    "Moderate Detract": "#ffcc80",
                    "Neutral":         "#cfd8dc",
                    "Moderate Pos":    "#80cbc4",
                    "Strong Pos":      "#a5d6a7",
                },
                text=intent_counts.values,
            )
            fig_int.update_traces(
                textposition="auto",
                insidetextanchor="middle",
                textfont_color="#374151",
                marker_line_color="#ffffff", marker_line_width=1,
                hovertemplate="%{x}: %{y}<extra></extra>",
            )
            fig_int.update_layout(
                paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#334155", size=10),
                title=dict(font=dict(size=12, color="#1e293b")),
                xaxis_tickangle=-30, showlegend=False,
                hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
                height=360,
                yaxis=dict(range=[0, None], constrain="range"),
            )
            st.plotly_chart(fig_int, width="stretch")
    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — REVIEW QUALITY & EXPERTISE
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(7, "Review Quality & Expertise Level", "How detailed and credible are the signals")
    col_rq1, col_rq2, col_rq3 = st.columns(3)

    with col_rq1:
        if "review_quality" in signal_df.columns:
            rq_counts = signal_df["review_quality"].value_counts()
            fig_rq = px.pie(
                rq_counts.reset_index(), values=rq_counts.reset_index().columns[1],
                names="review_quality", title="Review Quality",
                hole=0.5,
                color_discrete_sequence=["#f97316", "#fbbf24", "#4ade80"],
            )
            fig_rq.update_layout(
                template=PLOTLY_TEMPLATE, height=300,
                paper_bgcolor=CHART_BG, font=dict(color="#334155", size=10),
                title=dict(font=dict(size=12, color="#1e293b")),
                showlegend=True,
            )
            fig_rq.update_traces(textposition="outside", textinfo="percent+label")
            st.plotly_chart(fig_rq, width="stretch")

    with col_rq2:
        if "expertise_level" in signal_df.columns:
            fig_exp = px.histogram(
                signal_df, x="expertise_level", nbins=10,
                title="Expertise Level",
                color_discrete_sequence=["#a78bfa"],
                labels={"x": "Expertise Level", "y": "Count"},
                marginal="box",
            )
            fig_exp.update_layout(
                template=PLOTLY_TEMPLATE, height=300,
                paper_bgcolor=CHART_BG, font=dict(color="#334155", size=10),
                title=dict(font=dict(size=12, color="#1e293b")),
                showlegend=False, hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
            )
            st.plotly_chart(fig_exp, width="stretch")

    with col_rq3:
        if "specificity" in signal_df.columns:
            fig_spec = px.histogram(
                signal_df, x="specificity", nbins=10,
                title="Comment Specificity",
                color_discrete_sequence=["#38bdf8"],
                labels={"x": "Specificity Score", "y": "Count"},
                marginal="box",
            )
            fig_spec.update_layout(
                template=PLOTLY_TEMPLATE, height=300,
                paper_bgcolor=CHART_BG, font=dict(color="#334155", size=10),
                title=dict(font=dict(size=12, color="#1e293b")),
                showlegend=False, hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
            )
            st.plotly_chart(fig_spec, width="stretch")

    if "review_quality" in signal_df.columns and "signal" in signal_df.columns:
        rq_crosstab = pd.crosstab(signal_df["signal"], signal_df["review_quality"], normalize="index") * 100
        fig_rq_cross = px.bar(
            rq_crosstab.reset_index(), x="signal", y=[c for c in rq_crosstab.columns],
            title="Review Quality % by Signal Type",
            labels={"value": "%", "signal": "Signal Type", "variable": "Quality"},
            barmode="group",
            color_discrete_sequence=["#f97316", "#fbbf24", "#4ade80"],
            text_auto=True,
        )
        fig_rq_cross.update_traces(
            hovertemplate="Signal: %{x}<br>Quality: %{customdata}<br>%: %{y:.1f}%<extra></extra>",
        )
        fig_rq_cross.update_layout(
            template=PLOTLY_TEMPLATE, height=320,
            paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#334155", size=10),
            title=dict(font=dict(size=12, color="#1e293b")),
            xaxis_tickangle=-30, hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
        )
        for i, col in enumerate([c for c in rq_crosstab.columns]):
            fig_rq_cross.data[i].customdata = [col] * len(fig_rq_cross.data[i].x)
        st.plotly_chart(fig_rq_cross, width="stretch")
    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8 — SARCASM DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    if "sarcasm_flag" in signal_df.columns:
        sarc_df = signal_df[signal_df["sarcasm_flag"] == 1]
        _section_header(8, "Sarcasm Detection", f"{len(sarc_df)} sarcastic comments flagged")
        sarc_pct = 100 * len(sarc_df) / max(1, len(signal_df))
        col_sarc1, col_sarc2 = st.columns([1, 1])
        with col_sarc1:
            fig_sarc = px.histogram(
                signal_df, x="sarcasm_flag", nbins=2,
                title="Sarcasm Flag Distribution",
                color_discrete_sequence=["#4ade80", "#f87171"],
                labels={"x": "Sarcasm", "y": "Count"},
            )
            fig_sarc.update_layout(
                template=PLOTLY_TEMPLATE, height=260,
                paper_bgcolor=CHART_BG, font=dict(color="#334155", size=10),
                title=dict(font=dict(size=12, color="#1e293b")),
                showlegend=False, hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
                xaxis_tickvals=[0, 1], xaxis_ticktext=["Non-Sarcasm", "Sarcasm"],
            )
            st.plotly_chart(fig_sarc, width="stretch")
        with col_sarc2:
            inner = "<div class='chart-card'>"
            inner += "<div class='chart-title'>Sarcasm Summary</div>"
            inner += f"<div style='font-size:0.82rem;line-height:1.7'>"
            inner += f"- <strong>Sarcastic</strong>: {len(sarc_df):,} ({sarc_pct:.1f}%)<br>"
            inner += f"- <strong>Non-sarcastic</strong>: {len(signal_df) - len(sarc_df):,} ({100-sarc_pct:.1f}%)"
            if not sarc_df.empty and "signal" in sarc_df.columns:
                inner += f"<br>- <strong>Top signal in sarcastic</strong>: {sarc_df['signal'].mode().iloc[0]}"
            inner += "</div></div>"
            st.html(inner)
        if not sarc_df.empty:
            sarc_cols = ["signal", "confidence", "sentiment_intensity", "purchase_intent_score", "comment_text"]
            st.markdown("**Sample Sarcastic Comments**")
            st.dataframe(sarc_df[[c for c in sarc_cols if c in sarc_df.columns]].head(15),
                         width="stretch", height=340)
        section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9 — SCORE HEATMAP
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(9, "Per-Comment Score Heatmap", "Green = positive, Red = negative (top 40 signals)")
    _render_score_heatmap(signal_df, avail_score_cols, "", height=450)
    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 10 — KEY PHRASES
    # ══════════════════════════════════════════════════════════════════════════
    if "key_phrases_used" in signal_df.columns:
        all_phrases = signal_df["key_phrases_used"].dropna().tolist()
        flat = []
        for entry in all_phrases:
            if isinstance(entry, list):
                flat.extend(entry)
            elif isinstance(entry, str):
                try:
                    flat.extend(eval(entry))
                except Exception:
                    flat.append(entry)
        if flat:
            phrase_counts = pd.Series(flat).str.lower().str.strip().value_counts().head(40)
            _section_header(10, "Key Phrases", "Most cited phrases from LLM analysis")
            col_phr1, col_phr2 = st.columns([3, 1])
            with col_phr1:
                fig_phr = px.bar(
                    phrase_counts.reset_index(), x="key_phrases_used",
                    y=phrase_counts.reset_index().columns[1],
                    title="Top 40 Key Phrases",
                    labels={"key_phrases_used": "Phrase",
                            phrase_counts.reset_index().columns[1]: "Frequency"},
                    color=phrase_counts.reset_index().columns[1],
                    color_continuous_scale="Blues",
                    text=phrase_counts.values,
                )
                fig_phr.update_traces(
                    textposition="outside", textfont_size=8, textfont_color="#334155",
                    hovertemplate="Phrase: %{x}<br>Count: %{y}<extra></extra>",
                )
                fig_phr.update_layout(
                    template=PLOTLY_TEMPLATE, height=380,
                    paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#334155", size=10),
                    title=dict(font=dict(size=12, color="#1e293b")),
                    xaxis_tickangle=-60, showlegend=False,
                    hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
                )
                st.plotly_chart(fig_phr, width="stretch")
            with col_phr2:
                inner = "<div class='chart-card'>"
                inner += "<div class='chart-title'>Top 5 Phrases</div>"
                for i, (phrase, cnt) in enumerate(phrase_counts.head(5).items(), 1):
                    pct = 100 * cnt / len(flat)
                    inner += f"<div style='font-size:0.82rem;margin-bottom:4px'>{i}. <strong>{phrase}</strong> — {cnt} ({pct:.1f}%)</div>"
                inner += "</div>"
                st.html(inner)
        else:
            st.info("No key phrases found in results.")
        section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 11 — CATEGORIES & TIMELINE
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(11, "Product Categories & Timeline", "Where signals come from and when")
    df_for_cat = signal_df
    if "product_categories" not in df_for_cat.columns:
        cleaned = load_cleaned_data()
        df_for_cat = signal_df.merge(
            cleaned[["comment_id", "product_categories", "priority_level"]],
            on="comment_id", how="left",
        ) if not cleaned.empty else signal_df

    col_cat, col_time = st.columns([1, 1])
    with col_cat:
        if not df_for_cat.empty and "product_categories" in df_for_cat.columns:
            cat_exploded = df_for_cat["product_categories"].str.split("|").explode()
            cat_sig = cat_exploded[cat_exploded != "unknown"].value_counts()
            if not cat_sig.empty:
                fig_cat = px.bar(
                    cat_sig.reset_index(), x="product_categories", y=cat_sig.reset_index().columns[1],
                    title="Signals by Product Category",
                    labels={"product_categories": "Category", cat_sig.reset_index().columns[1]: "Count"},
                    color=cat_sig.reset_index().columns[1],
                    color_continuous_scale="Teal",
                    text=cat_sig.values,
                )
                fig_cat.update_traces(
                    textposition="outside", textfont_color="#334155",
                    hovertemplate="Category: %{x}<br>Count: %{y}<extra></extra>",
                )
                fig_cat.update_layout(
                    template=PLOTLY_TEMPLATE, height=360,
                    paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#334155", size=10),
                    title=dict(font=dict(size=12, color="#1e293b")),
                    xaxis_tickangle=-45, showlegend=False,
                    hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
                )
                st.plotly_chart(fig_cat, width="stretch")

    with col_time:
        timeline_df = full_df if "published_at" in full_df.columns else signal_df
        if "published_at" in timeline_df.columns:
            timeline_df = timeline_df.copy()
            timeline_df["published_at"] = pd.to_datetime(timeline_df["published_at"], errors="coerce")
            timeline_df = timeline_df.dropna(subset=["published_at"])
            if not timeline_df.empty and "signal" in timeline_df.columns:
                timeline_df["date"] = timeline_df["published_at"].dt.date
                timeline = timeline_df.groupby(["date", "signal"]).size().reset_index(name="count")
                fig_timeline = px.line(
                    timeline, x="date", y="count", color="signal",
                    title="Demand Signals Over Time",
                    labels={"date": "Date", "count": "Signal Count", "signal": "Type"},
                    color_discrete_sequence=SIGNAL_COLORS,
                    markers=True,
                )
                fig_timeline.update_layout(
                    template=PLOTLY_TEMPLATE, height=360,
                    paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#334155", size=10),
                    title=dict(font=dict(size=12, color="#1e293b")),
                    hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
                )
                st.plotly_chart(fig_timeline, width="stretch")
    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 12 — TOP VIDEOS
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(12, "Top Videos by Signal Count", "Videos generating the most demand signals")
    video_df = full_df if "video_title" in full_df.columns else signal_df
    group_cols = [c for c in ["video_id", "video_title", "channel_title"] if c in video_df.columns]
    video_signals = video_df.groupby(group_cols).size().reset_index(name="signal_count")
    video_signals = video_signals.sort_values("signal_count", ascending=False).head(20)
    video_signals["video_url"] = "https://www.youtube.com/watch?v=" + video_signals["video_id"].astype(str)
    disp_cols = [c for c in ["video_title", "channel_title", "signal_count", "video_url"] if c in video_signals.columns]
    st.dataframe(video_signals[disp_cols], width="stretch", height=400)
    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 13 — COMPLETE TABLE & EXPORT
    # ══════════════════════════════════════════════════════════════════════════
    _section_header(13, "Complete Results Table", "All demand signals with scores — sorted by confidence")
    table_df = signal_df.copy()
    if "clean_text" in full_df.columns:
        text_map = full_df[["comment_id", "clean_text"]].drop_duplicates("comment_id").set_index("comment_id")["clean_text"]
        table_df["comment_text"] = table_df["comment_id"].map(text_map)
    if "video_url" not in table_df.columns and "video_id" in table_df.columns:
        table_df["video_url"] = "https://www.youtube.com/watch?v=" + table_df["video_id"].astype(str)
    if "risk_score" in table_df.columns:
        table_df = table_df.sort_values(["risk_score", "confidence"], ascending=[False, False])
    else:
        table_df = table_df.sort_values("confidence", ascending=False)
    score_display_cols = [
        "comment_id", "video_url",
        "risk_score", "signal", "confidence", "reason",
        "overall_sentiment_score", "sentiment_intensity",
        "fit_score", "protection_score", "texture_score",
        "value_perception", "purchase_intent_score",
        "review_quality", "sarcasm_flag", "expertise_level",
        "comment_text",
    ]
    disp_cols2 = [c for c in score_display_cols if c in table_df.columns]
    st.dataframe(table_df[disp_cols2], width="stretch", height=580)

    section_divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 13-B — EXECUTIVE SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    _section_header("★", "Executive Summary", "One-glance verdict: which demand signals are winning")
    if signal_df.empty:
        st.info("No signal data available for summary.")
    else:
        # ── Composite "Signal Quality Score" ─────────────────────────────────
        NUMERIC_SCORES = [
            "fit_score", "protection_score", "texture_score",
            "value_perception", "overall_sentiment_score",
            "sentiment_intensity", "urgency_score", "purchase_intent_score",
        ]
        avail_num = [c for c in NUMERIC_SCORES if c in signal_df.columns]

        rank_df = signal_df.copy()
        rank_df = rank_df.dropna(subset=["confidence"])

        if avail_num:
            for c in avail_num:
                rank_df[c] = pd.to_numeric(rank_df[c], errors="coerce")
            # Normalise each dimension to [0,1] then average → composite score
            comp_scores = pd.DataFrame()
            for c in avail_num:
                mn, mx = rank_df[c].min(), rank_df[c].max()
                if mx > mn:
                    comp_scores[c] = (rank_df[c] - mn) / (mx - mn)
                else:
                    comp_scores[c] = 0.5
            rank_df["_composite"] = comp_scores.mean(axis=1)

        rank_df["_quality"] = (
            rank_df["_composite"].fillna(0) * 0.5
            + rank_df["confidence"].fillna(0) * 0.5
        )
        rank_df["_product_score"] = (
            rank_df.get("fit_score", pd.Series(0, index=rank_df.index)).fillna(0)
            + rank_df.get("protection_score", pd.Series(0, index=rank_df.index)).fillna(0)
            + rank_df.get("texture_score", pd.Series(0, index=rank_df.index)).fillna(0)
            + rank_df.get("value_perception", pd.Series(0, index=rank_df.index)).fillna(0)
        ) / 4

        top_signals = rank_df.sort_values("_quality", ascending=False).head(5)

        # ── Top-level verdict cards ───────────────────────────────────────────
        total = len(signal_df)
        avg_conf = signal_df["confidence"].mean() if "confidence" in signal_df.columns else 0
        best_signal = top_signals.iloc[0] if not top_signals.empty else None
        best_text = str(best_signal["signal"]) if best_signal is not None and "signal" in best_signal.index else "N/A"
        best_score = best_signal["_quality"] if best_signal is not None else 0
        best_score_pct = best_score * 100

        col_v1, col_v2, col_v3, col_v4 = st.columns(4)
        with col_v1:
            st.metric("Total Signals", f"{total:,}")
        with col_v2:
            st.metric("Avg Confidence", f"{avg_conf:.2f}")
        with col_v3:
            high_quality = int((rank_df["_quality"] > 0.6).sum())
            st.metric("High Quality", f"{high_quality}")
        with col_v4:
            if "product_categories" in signal_df.columns:
                top_cat = signal_df["product_categories"].value_counts().idxmax()
                st.markdown(
                    f"<div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:0.75rem;text-align:center;'>"
                    f"<div style='color:#64748b;font-size:0.78rem;'>Top Category</div>"
                    f"<div style='color:#1e293b;font-size:1.1rem;font-weight:700;margin-top:4px;word-break:break-word;'>{top_cat}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                top_videos = full_df["video_title"].value_counts().idxmax() if "video_title" in full_df.columns else "N/A"
                st.markdown(
                    f"<div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:10px;padding:0.75rem;text-align:center;'>"
                    f"<div style='color:#64748b;font-size:0.78rem;'>Top Video</div>"
                    f"<div style='color:#1e293b;font-size:1.1rem;font-weight:700;margin-top:4px;word-break:break-word;'>{top_videos}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # ── Top signal verdict banner ─────────────────────────────────────────
        verdict_cols = st.columns([1, 3])
        with verdict_cols[0]:
            st.markdown(
                f"<div style='background: linear-gradient(135deg,#16a34a,#15803d);color:white;padding:28px 20px;"
                f"border-radius:14px;text-align:center;'>"
                f"<div style='font-size:12px;opacity:0.85;margin-bottom:6px;'>🏆 TOP SIGNAL</div>"
                f"<div style='font-size:20px;font-weight:700;line-height:1.2;'>{best_text}</div>"
                f"<div style='font-size:28px;font-weight:800;margin-top:8px;'>{best_score_pct:.0f}<span style='font-size:14px;'> pts</span></div>"
                f"<div style='font-size:11px;opacity:0.75;margin-top:6px;'>Composite Quality Score</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with verdict_cols[1]:
            reasons_html = ""
            if best_signal is not None:
                reasons = []
                if "reason" in best_signal.index and pd.notna(best_signal["reason"]):
                    reasons.append(f"<b>Reason:</b> {str(best_signal['reason'])[:120]}")
                if "purchase_intent_score" in best_signal.index and pd.notna(best_signal["purchase_intent_score"]):
                    reasons.append(f"<b>Purchase Intent:</b> {float(best_signal['purchase_intent_score']):.2f}")
                if "expertise_level" in best_signal.index and pd.notna(best_signal["expertise_level"]):
                    reasons.append(f"<b>Expertise Level:</b> {float(best_signal['expertise_level']):.2f}")
                if "sentiment_intensity" in best_signal.index and pd.notna(best_signal["sentiment_intensity"]):
                    reasons.append(f"<b>Sentiment Intensity:</b> {float(best_signal['sentiment_intensity']):.2f}")
                if "fit_score" in best_signal.index and pd.notna(best_signal["fit_score"]):
                    reasons.append(f"<b>Fit Score:</b> {float(best_signal['fit_score']):.2f}")
                if not reasons:
                    reasons = ["Signal leads on composite quality + confidence score."]
                for r in reasons[:4]:
                    reasons_html += f"<div style='margin-bottom:6px;font-size:12px;color:#14532d;line-height:1.5;'>{r}</div>"
            else:
                reasons_html = "<div style='font-size:12px;color:#94a3b8;'>No data available</div>"

            st.markdown(
                f"<div style='background:#f0fdf4;border:1.5px solid #bbf7d0;border-radius:14px;padding:20px 24px;height:100%;box-sizing:border-box;'>"
                f"<div style='font-size:12px;color:#166534;font-weight:700;margin-bottom:10px;'>📋 WHY THIS SIGNAL WINS</div>"
                f"{reasons_html}"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── Top 5 ranked signals table ────────────────────────────────────────
        st.markdown("##### 🏅 Top 5 Demand Signals")
        top5_cols = ["signal", "confidence", "_quality", "_product_score"]
        top5_cols = [c for c in top5_cols if c in top_signals.columns]
        top5 = top_signals[top5_cols].rename(columns={
            "signal": "Signal", "confidence": "Confidence",
            "_quality": "Quality Score", "_product_score": "Product Score",
        }).reset_index(drop=True)
        top5.index = top5.index + 1
        top5.index.name = "Rank"

        # Colour-code each row by Quality Score using HTML backgrounds via st.markdown table
        def quality_color(v):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return "#f0fdf4"
            if v > 0.7:
                return "#16a34a"  # dark green
            elif v > 0.55:
                return "#22c55e"  # green
            elif v > 0.4:
                return "#86efac"  # light green
            return "#f0fdf4"     # very light

        rows_html = ""
        for _, row in top5.iterrows():
            q = float(row["Quality Score"]) if "Quality Score" in row and pd.notna(row["Quality Score"]) else 0
            bg = quality_color(q)
            text_c = "white" if q > 0.55 else "#14532d"
            rank = f"<td style='background:{bg};color:{text_c};font-weight:700;border:none;padding:6px 10px;text-align:center;'>{row.name}</td>"
            cells = ""
            for col in top5.columns:
                v = row[col]
                try:
                    v = float(v)
                    cell_v = f"{v:.3f}" if "Score" in col or "Confidence" in col else str(v)
                except (TypeError, ValueError):
                    cell_v = str(v) if pd.notna(v) else "—"
                cells += f"<td style='border:none;padding:6px 10px;font-size:12px;'>{cell_v}</td>"
            rows_html += f"<tr>{rank}{cells}</tr>"

        tbl_html = (
            "<table style='width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;'>"
            f"<thead><tr>"
            f"<th style='background:#16a34a;color:white;padding:8px 10px;text-align:center;font-size:12px;'>Rank</th>"
            + "".join(f"<th style='background:#16a34a;color:white;padding:8px 10px;font-size:12px;'>{c}</th>" for c in top5.columns)
            + "</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            "</table>"
        )
        st.markdown(tbl_html, unsafe_allow_html=True)

        # ── Top 5 sample comments with original text & video link ─────────────────
        st.markdown("")
        st.markdown("##### 🔍 Top 5 Sample Comments — Verify Signal Quality")
        st.caption("Click the video link to watch the original clip and confirm the signal.")

        top5_comments = rank_df.sort_values("_quality", ascending=False).head(5)

        # Build a clean comment text map from full_df
        if "clean_text" in full_df.columns:
            text_map = (
                full_df[["comment_id", "clean_text", "video_id"]]
                .drop_duplicates("comment_id")
                .set_index("comment_id")
            )
        else:
            text_map = pd.DataFrame()

        # Build video link map
        if "video_id" in full_df.columns:
            vid_map = (
                full_df[["comment_id", "video_id"]]
                .drop_duplicates("comment_id")
                .set_index("comment_id")
            )
        else:
            vid_map = pd.DataFrame()

        for rank_i, (idx, row) in enumerate(top5_comments.iterrows(), 1):
            cid = row.get("comment_id", "")
            comment_text = ""
            if cid in text_map.index:
                comment_text = str(text_map.loc[cid, "clean_text"])
            elif "comment_text" in row.index and pd.notna(row["comment_text"]):
                comment_text = str(row["comment_text"])
            else:
                comment_text = "—"

            # Truncate long comments for the preview card
            preview = comment_text[:300] + ("..." if len(comment_text) > 300 else "")

            vid_link = ""
            if cid in vid_map.index:
                vid = str(vid_map.loc[cid, "video_id"])
                vid_link = f"https://www.youtube.com/watch?v={vid}"
            elif "video_id" in row.index and pd.notna(row["video_id"]):
                vid_link = f"https://www.youtube.com/watch?v={row['video_id']}"

            signal_name = str(row.get("signal", "—"))
            conf = row.get("confidence", 0)
            qual = row.get("_quality", 0)

            badge_color = "#16a34a" if qual > 0.6 else "#22c55e" if qual > 0.45 else "#86efac"
            link_html = f"<a href='{vid_link}' target='_blank' style='color:#16a34a;font-weight:600;'>▶ Watch Video</a>" if vid_link else ""

            st.markdown(
                f"<div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;"
                f"padding:1rem 1.25rem;margin-bottom:0.75rem;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.5rem;'>"
                f"<div>"
                f"<span style='background:{badge_color};color:white;font-size:0.72rem;font-weight:700;"
                f"padding:2px 8px;border-radius:20px;margin-right:6px;'>#{rank_i}</span>"
                f"<span style='font-weight:700;color:#1e293b;'>{signal_name}</span>"
                f"</div>"
                f"<div style='text-align:right;'>"
                f"<div style='font-size:0.78rem;color:#64748b;'>Conf <b style='color:#1e293b;'>{conf:.3f}</b> &nbsp; Quality <b style='color:#1e293b;'>{qual:.3f}</b></div>"
                f"{link_html}"
                f"</div>"
                f"</div>"
                f"<div style='font-size:0.85rem;color:#334155;line-height:1.6;'>{preview}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Show more — items ranked 6+ in a scrollable container (max ~3 items visible)
        rest = rank_df.sort_values("_quality", ascending=False).iloc[5:]
        if len(rest) > 0:
            st.markdown("")
            expand_key = "exec_show_more_comments"
            if expand_key not in st.session_state:
                st.session_state[expand_key] = False
            if st.button("▼ Show more" if not st.session_state[expand_key] else "▲ Show less",
                         key="btn_exec_show_more"):
                st.session_state[expand_key] = not st.session_state[expand_key]

            if st.session_state[expand_key]:
                scroll_id = "exec-comments-scroll"
                st.markdown(
                    f"<div id='{scroll_id}' style='max-height:540px;overflow-y:auto;"
                    f"scrollbar-width:thin;scrollbar-color:#16a34a #e2e8f0;'>",
                    unsafe_allow_html=True,
                )
                for rank_i, (idx, row) in enumerate(rest.iterrows(), 6):
                    cid = row.get("comment_id", "")
                    comment_text = ""
                    if cid in text_map.index:
                        comment_text = str(text_map.loc[cid, "clean_text"])
                    elif "comment_text" in row.index and pd.notna(row["comment_text"]):
                        comment_text = str(row["comment_text"])
                    else:
                        comment_text = "—"

                    preview = comment_text[:300] + ("..." if len(comment_text) > 300 else "")
                    vid_link = ""
                    if cid in vid_map.index:
                        vid = str(vid_map.loc[cid, "video_id"])
                        vid_link = f"https://www.youtube.com/watch?v={vid}"
                    elif "video_id" in row.index and pd.notna(row["video_id"]):
                        vid_link = f"https://www.youtube.com/watch?v={row['video_id']}"

                    signal_name = str(row.get("signal", "—"))
                    conf = row.get("confidence", 0)
                    qual = row.get("_quality", 0)
                    badge_color = "#16a34a" if qual > 0.6 else "#22c55e" if qual > 0.45 else "#86efac"
                    link_html = f"<a href='{vid_link}' target='_blank' style='color:#16a34a;font-weight:600;'>▶ Watch Video</a>" if vid_link else ""

                    st.markdown(
                        f"<div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;"
                        f"padding:1rem 1.25rem;margin-bottom:0.75rem;'>"
                        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.5rem;'>"
                        f"<div>"
                        f"<span style='background:{badge_color};color:white;font-size:0.72rem;font-weight:700;"
                        f"padding:2px 8px;border-radius:20px;margin-right:6px;'>#{rank_i}</span>"
                        f"<span style='font-weight:700;color:#1e293b;'>{signal_name}</span>"
                        f"</div>"
                        f"<div style='text-align:right;'>"
                        f"<div style='font-size:0.78rem;color:#64748b;'>Conf <b style='color:#1e293b;'>{conf:.3f}</b> &nbsp; Quality <b style='color:#1e293b;'>{qual:.3f}</b></div>"
                        f"{link_html}"
                        f"</div>"
                        f"</div>"
                        f"<div style='font-size:0.85rem;color:#334155;line-height:1.6;'>{preview}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("")

        # ── Two side-by-side: confidence distribution + category breakdown ────
        sum_left, sum_right = st.columns(2)

        with sum_left:
            st.markdown("##### 📊 Confidence Distribution")
            if "confidence" in signal_df.columns:
                conf_bins = [0, 0.3, 0.5, 0.7, 0.85, 1.01]
                conf_labels = ["< 0.3", "0.3–0.5", "0.5–0.7", "0.7–0.85", "> 0.85"]
                conf_cats = pd.cut(signal_df["confidence"], bins=conf_bins, labels=conf_labels, right=False)
                conf_dist = conf_cats.value_counts().reindex(conf_labels).fillna(0).sort_index()
                conf_colors = ["#fecaca", "#fed7aa", "#fef08a", "#bbf7d0", "#86efac"]
                fig_conf = px.bar(
                    x=conf_dist.index.astype(str), y=conf_dist.values,
                    title="Confidence Band",
                    labels={"x": "Confidence Band", "y": "Count"},
                    color=conf_dist.values, color_continuous_scale="Greens",
                    text=conf_dist.values,
                )
                fig_conf.update_traces(textposition="outside", textfont_size=11)
                fig_conf.update_layout(
                    template=PLOTLY_TEMPLATE, height=280,
                    paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#334155", size=10),
                    showlegend=False, hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_conf, width="stretch")

        with sum_right:
            st.markdown("##### 🏷️ Top Product Categories")
            if "product_categories" in signal_df.columns:
                cat_counts = signal_df["product_categories"].value_counts().head(8)
                cat_colors = ["#16a34a", "#22c55e", "#4ade80", "#86efac", "#bbf7d0",
                              "#86efac", "#4ade80", "#22c55e"][:len(cat_counts)]
                fig_cat = px.bar(
                    x=cat_counts.values[::-1], y=cat_counts.index[::-1].astype(str),
                    orientation="h", title="Signal Count by Category",
                    labels={"x": "Signal Count", "y": ""},
                    color=cat_counts.values[::-1], color_continuous_scale="Greens",
                    text=cat_counts.values[::-1],
                )
                fig_cat.update_traces(textposition="outside", textfont_size=10)
                fig_cat.update_layout(
                    template=PLOTLY_TEMPLATE, height=280,
                    paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#334155", size=10),
                    showlegend=False, hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_cat, width="stretch")
            elif "video_title" in full_df.columns:
                vid_counts = full_df["video_title"].value_counts().head(8)
                fig_cat = px.bar(
                    x=vid_counts.values[::-1], y=vid_counts.index[::-1].astype(str),
                    orientation="h", title="Signal Count by Video",
                    labels={"x": "Signal Count", "y": ""},
                    color=vid_counts.values[::-1], color_continuous_scale="Greens",
                    text=vid_counts.values[::-1],
                )
                fig_cat.update_traces(textposition="outside", textfont_size=10)
                fig_cat.update_layout(
                    template=PLOTLY_TEMPLATE, height=280,
                    paper_bgcolor=CHART_BG, plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#334155", size=10),
                    showlegend=False, hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_cat, width="stretch")

        st.markdown("")

        # ── Dimension radar chart: average scores across top signals ──────────
        RADAR_DIMS = ["fit_score", "protection_score", "texture_score",
                      "value_perception", "purchase_intent_score", "overall_sentiment_score"]
        avail_radar = [c for c in RADAR_DIMS if c in signal_df.columns]

        if len(avail_radar) >= 3:
            st.markdown("##### 🎯 Dimension Profile — Top Signal vs Average")
            r_left, r_right = st.columns(2)
            top1_row = rank_df.sort_values("_quality", ascending=False).iloc[0]
            top1_vals = [float(top1_row[c]) for c in avail_radar]
            avg_vals = [float(signal_df[c].mean()) for c in avail_radar]
            labels_radar = [DIM_LABELS.get(c, c) for c in avail_radar]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=top1_vals + [top1_vals[0]],
                theta=labels_radar + [labels_radar[0]],
                fill="toself", fillcolor="rgba(22,163,74,0.25)",
                line=dict(color="#16a34a", width=2),
                name="Top Signal",
                hovertemplate=" %{theta}: %{r:.2f}<extra></extra>",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_vals + [avg_vals[0]],
                theta=labels_radar + [labels_radar[0]],
                fill="toself", fillcolor="rgba(100,116,139,0.15)",
                line=dict(color="#64748b", width=2, dash="dot"),
                name="Dataset Average",
                hovertemplate=" %{theta}: %{r:.2f}<extra></extra>",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
                template=PLOTLY_TEMPLATE, height=340,
                paper_bgcolor=CHART_BG,
                font=dict(color="#334155", size=10),
                legend=dict(bgcolor="rgba(255,255,255,0.8)", font_color="#334155"),
                hoverlabel=dict(bgcolor="#ffffff", font_color="#334155"),
            )
            with r_left:
                st.plotly_chart(fig_radar, width="stretch")
            with r_right:
                delta_vals = [t - a for t, a in zip(top1_vals, avg_vals)]
                delta_df = pd.DataFrame({
                    "Dimension": labels_radar,
                    "Δ vs Avg": delta_vals,
                }).sort_values("Δ vs Avg", ascending=False)
                delta_df["Direction"] = delta_df["Δ vs Avg"].apply(
                    lambda x: "▲" if x > 0.05 else ("▼" if x < -0.05 else "➜")
                )
                delta_df["Color"] = delta_df["Δ vs Avg"].apply(
                    lambda x: "#16a34a" if x > 0.05 else ("#ef4444" if x < -0.05 else "#94a3b8")
                )
                for _, row in delta_df.iterrows():
                    bar_w = min(abs(float(row["Δ vs Avg"])) * 40, 100)
                    sign = "+" if float(row["Δ vs Avg"]) >= 0 else ""
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>"
                        f"<div style='width:110px;font-size:11px;color:#334155;'>{row['Dimension']}</div>"
                        f"<div style='flex:1;background:#f1f5f9;border-radius:4px;height:14px;'>"
                        f"<div style='width:{bar_w:.0f}%;background:{row['Color']};border-radius:4px;height:14px;opacity:0.85;'></div>"
                        f"</div>"
                        f"<div style='width:50px;text-align:right;font-size:11px;color:{row['Color']};font-weight:600;'>"
                        f"{row['Direction']} {sign}{float(row['Δ vs Avg']):.2f}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("")
        # "All-clear" recommendation banner
        high_quality_n = int((rank_df["_quality"] > 0.6).sum())
        medium_quality_n = int(((rank_df["_quality"] > 0.4) & (rank_df["_quality"] <= 0.6)).sum())
        st.markdown(
            f"<div style='background:#f0fdf4;border:2px solid #16a34a;border-radius:12px;padding:16px 24px;'>"
            f"<div style='font-size:14px;font-weight:700;color:#14532d;margin-bottom:6px;'>💡 Bottom Line</div>"
            f"<div style='font-size:13px;color:#166534;line-height:1.6;'>"
            f"Out of <b>{total:,}</b> demand signals analysed, "
            f"<b>{high_quality_n}</b> score as <b style='color:#16a34a;'>high quality</b> "
            f"and <b>{medium_quality_n}</b> as <b style='color:#ca8a04;'>medium quality</b>. "
            f"The <b>top signal \"{best_text}\"</b> leads with a composite quality score of "
            f"<b>{best_score_pct:.0f}/100</b>, driven by strong purchase intent and confidence. "
            f"</div></div>",
            unsafe_allow_html=True,
        )

    section_divider()

    _section_header(14, "Export Results", "Download full signal dataset")
    dl1, dl2 = st.columns(2)
    with dl1:
        download_button("⬇ Export Signals", table_df, "demand_signals_export", "dash_sig")
    with dl2:
        download_button("⬇ Export Full Results", full_df, "demand_signals_full_export", "dash_full")


# Light-mode friendly chart color palettes
CHART_BG = "#f8fafc"
PLOTLY_TEMPLATE = "plotly_white"

SIGNAL_COLORS = [
    "#818cf8", "#34d399", "#fb923c", "#f472b6",
    "#60a5fa", "#a3e635", "#fbbf24", "#f87171",
]
SENTIMENT_COLORS = ["#f87171", "#fbbf24", "#94a3b8", "#86efac", "#4ade80"]
RISK_COLORS = ["#4ade80", "#facc15", "#f97316", "#ef4444"]

# ── Version management ─────────────────────────────────────────────────────────

def _delete_version(filename: str):
    """Delete both _full_ and _only_ parquet/csv files for a given version."""
    from pathlib import Path
    from src.config import OUTPUT_DIR
    output_dir = Path(OUTPUT_DIR)
    deleted = []
    for suffix in ["_full_", "_only_"]:
        for ext in [".parquet", ".csv"]:
            fpath = output_dir / filename.replace("_full_", suffix).replace("_only_", suffix)
            fpath = fpath.with_suffix(ext)
            if fpath.exists():
                fpath.unlink()
                deleted.append(fpath.name)
    return deleted


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _base_layout(fig, title=None, height=350):
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=height,
        paper_bgcolor=CHART_BG,
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#334155", size=11),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=13, color="#1e293b")))
    return fig


def _chart_card(inner):
    """Wrap chart output in a styled card container."""
    with st.container():
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        inner()
        st.markdown('</div>', unsafe_allow_html=True)


def _kpi_row(metrics: list[dict]):
    """Render a row of KPI cards."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            sub_html = f"<div class='kpi-sub'>{m['sub']}</div>" if 'sub' in m else ''
            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-value'>{m['value']}</div>"
                f"<div class='kpi-label'>{m['label']}</div>"
                f"{sub_html}"
                f"</div>",
                unsafe_allow_html=True,
            )


def _section_header(num: int, title: str, subtitle: str = ""):
    badge = f"<span class='badge-{num}'>{num}</span>"
    slug = f"section-{num}"
    st.markdown(
        f"<div class='section-header' id='{slug}'>{badge} {title}"
        + (f" &nbsp;<span style='font-weight:400;font-size:0.82rem;color:#64748b'>{subtitle}</span>" if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )


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
        if st.session_state.get("_rerun_after_collection"):
            st.session_state._rerun_after_collection = False
            st.rerun()
        render_step2_collection()
    elif selected_page.startswith("🤖"):
        render_step3_llm()
    elif selected_page.startswith("📈"):
        render_dashboard()


if __name__ == "__main__":
    main()
