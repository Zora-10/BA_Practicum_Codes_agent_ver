"""
Step 6: Demand Signal Detection via LLM
Classifies cleaned comments using DeepSeek or Google Gemini.
Supports both DeepSeek-chat and Gemini-2.0-flash models.
"""

import json
import time
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Custom exception ────────────────────────────────────────────────────────────

class QuotaExceededError(Exception):
    """Raised when the LLM API returns a 429 rate-limit / quota exceeded error."""
    def __init__(self, provider: str, message: str = "", partial_results: list = None):
        self.provider = provider
        self.message = message or f"{provider} quota exceeded"
        self.partial_results = partial_results or []
        super().__init__(self.message)


class APIError(Exception):
    """Raised when the LLM API call fails after all retries."""
    def __init__(self, provider: str, batch_errors: list[dict], partial_results: list = None):
        self.provider = provider
        self.batch_errors = batch_errors
        self.partial_results = partial_results or []
        super().__init__(f"{provider} API error in {len(batch_errors)} batch(es)")


class LLMCallError(Exception):
    """Raised when any LLM call (API error, JSON parse, network, etc.) fails — stops immediately."""
    def __init__(self, provider: str, reason: str, batch_idx: int = None, partial_results: list = None):
        self.provider = provider
        self.reason = reason
        self.batch_idx = batch_idx
        self.partial_results = partial_results or []
        batch_info = f" batch {batch_idx+1}" if batch_idx is not None else ""
        super().__init__(f"{provider}{batch_info}: {reason}")


def _save_partial_and_raise(exc: Exception, model_provider: str, n_to_analyze: int | None, input_df: pd.DataFrame):
    """Save partial results to parquet and re-raise the exception."""
    partial = getattr(exc, "partial_results", None) or []
    print(f"[ERROR] Stopping at batch. Saved {len(partial)} results.")
    partial_df = pd.DataFrame(partial)
    if not partial_df.empty:
        partial_df["comment_id"] = partial_df["comment_id"].astype(str)
        df_out = input_df.copy()
        df_out["comment_id"] = df_out["comment_id"].astype(str)
        df_out = df_out.merge(
            partial_df[["comment_id", "signal", "confidence", "reason"]],
            on="comment_id", how="left",
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_out.to_parquet(OUTPUT_DIR / f"demand_signals_full_{model_provider}_{ts}.parquet", index=False)
    raise exc

from .config import (
    CLEANED_DIR,
    OUTPUT_DIR,
    CHECKPOINT_DIR,
    LLM_SYSTEM_PROMPT,
)


# ── Checkpoint ─────────────────────────────────────────────────────────────────

class LLMCheckpoint:
    def __init__(self, checkpoint_file: Path):
        self.file = checkpoint_file

    def save(self, results: list[dict]):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

    def load(self) -> tuple[list[dict], int]:
        if not self.file.exists():
            return [], 0
        try:
            with open(self.file, "r", encoding="utf-8") as f:
                results = json.load(f)
            return results, len(results)
        except (json.JSONDecodeError, ValueError):
            return [], 0

    def clear(self):
        if self.file.exists():
            self.file.unlink()


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_user_prompt(batch_df: pd.DataFrame) -> str:
    lines = []
    for _, row in batch_df.iterrows():
        comment_id = str(row["comment_id"])
        text = str(row["text_original"]).strip()
        video_title = str(row.get("title", "N/A"))[:80]
        lines.append(
            f"[Comment ID: {comment_id}] [Video: {video_title}]\n{text}"
        )
    return "\n\n".join(lines)


def _extract_json(raw: str) -> list:
    """Extract JSON array from model response. Handles both plain array
    and {"results": [...]} wrapper formats."""
    if not raw or not raw.strip():
        raise json.JSONDecodeError("empty response", raw, 0)

    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
    else:
        parsed = json.loads(raw)

    # Handle {"results": [...]} wrapper
    if isinstance(parsed, dict) and "results" in parsed:
        return parsed["results"]
    return parsed


# ── DeepSeek caller ────────────────────────────────────────────────────────────

def call_deepseek(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str = "deepseek-chat",
    max_retries: int = 3,
) -> Optional[str]:
    import openai
    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    def _is_429(e: Exception) -> bool:
        code = getattr(e, "status_code", None)
        if code == 429:
            return True
        err_str = str(e).lower()
        return "429" in err_str or "rate limit" in err_str or "quota" in err_str

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if _is_429(e):
                print(f"  [Attempt {attempt+1}] DeepSeek quota/rate-limit error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    raise QuotaExceededError("DeepSeek", str(e))
            else:
                print(f"  [Attempt {attempt+1}] DeepSeek error ({type(e).__name__}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
    return None


# ── Gemini caller ─────────────────────────────────────────────────────────────

def call_gemini(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model: str = "gemini-2.0-flash",
    max_retries: int = 3,
) -> Optional[str]:
    import urllib.request
    import urllib.error

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": f"{system_prompt}\n\n---\n\n{user_prompt}"}]}],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 4096,
        },
    }

    for attempt in range(max_retries):
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            candidates = result.get("candidates", [])
            if candidates:
                return candidates[0]["content"]["parts"][0]["text"].strip()
            return None
        except urllib.error.HTTPError as e:
            err_body_raw = e.read()
            try:
                err_body_str = err_body_raw.decode("utf-8", errors="replace")
            except Exception:
                err_body_str = repr(err_body_raw)
            print(f"  [Attempt {attempt+1}] Gemini HTTP {e.code}: {err_body_str[:200]}")
            if e.code == 429:
                raise QuotaExceededError("Gemini", err_body_str[:300])
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
        except Exception as e:
            print(f"  [Attempt {attempt+1}] Gemini error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    return None


# ── Unified LLM batch caller ──────────────────────────────────────────────────

def call_llm_batch(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    model_provider: str,   # "deepseek" or "gemini"
    model_name: str,
    max_retries: int = 3,
) -> Optional[list[dict]]:
    """
    Call the selected LLM and return parsed JSON results.
    model_provider: "deepseek" | "gemini"
    model_name: e.g. "deepseek-chat" or "gemini-2.0-flash"
    """
    if model_provider == "deepseek":
        raw = call_deepseek(api_key, system_prompt, user_prompt, model=model_name, max_retries=max_retries)
    elif model_provider == "gemini":
        raw = call_gemini(api_key, system_prompt, user_prompt, model=model_name, max_retries=max_retries)
    else:
        raise ValueError(f"Unknown model_provider: {model_provider}")

    if raw is None:
        raise LLMCallError(model_provider, "LLM returned empty response", partial_results=None)

    try:
        return _extract_json(raw)
    except json.JSONDecodeError as e:
        raise LLMCallError(
            model_provider,
            f"JSON parse error: {e}. Raw response: {raw[:500] if raw else '(empty)'}",
            partial_results=None,
        )


# ── Demand signal labels ────────────────────────────────────────────────────────

DEMAND_SIGNAL_LIST = [
    "purchase_intent",
    "problem_complaint",
    "comparison_research",
    "usage_scenario",
    "wishful_thinking",
    "supply_recommendation",
]


# ── Main pipeline ──────────────────────────────────────────────────────────────

ANALYSIS_COLS = [
    "comment_id", "video_id", "text_original", "clean_text",
    "title", "channel_title",
    "comment_like_count",
    "priority_level",
]

FINAL_COLS = [
    "comment_id", "video_id", "video_url", "comment_text",
    "signal", "confidence", "reason",
    "video_title", "channel_title",
    "comment_like_count", "priority_level",
]


def run_demand_signal_detection(
    api_key: str,
    input_df: pd.DataFrame | None = None,
    *,
    model_provider: str = "deepseek",
    model_name: str = "deepseek-chat",
    batch_size: int = 20,
    save_every: int = 5,
    rate_limit_delay: float = 1.5,
    checkpoint_file: Path | None = None,
    progress_callback=None,
    n_to_analyze: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run LLM classification on cleaned comments.
    model_provider: "deepseek" | "gemini"
    model_name: "deepseek-chat" / "gemini-2.0-flash"
    Returns: (full_df, signal_df)
    """
    if input_df is None:
        input_df = pd.read_parquet(CLEANED_DIR / "cleaned_comments_linked.parquet")

    if input_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Build working DataFrame — select columns needed for LLM analysis
    available = [c for c in ANALYSIS_COLS if c in input_df.columns]
    df_work = input_df[available].copy()
    df_work["video_url"] = (
        "https://www.youtube.com/watch?v=" + df_work["video_id"].astype(str)
    )

    N = len(df_work)
    n_batches = (N + batch_size - 1) // batch_size

    if checkpoint_file is None:
        cp_name = f"llm_{model_provider}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        checkpoint_file = CHECKPOINT_DIR / cp_name
    checkpoint = LLMCheckpoint(checkpoint_file)
    existing_results, resume_from = checkpoint.load()

    classified_ids = {r["comment_id"] for r in existing_results}
    results_count = len(existing_results)

    # Filter out already-classified comments first — must happen BEFORE n_batches
    df_work = df_work[
        ~df_work["comment_id"].astype(str).isin(classified_ids)
    ].reset_index(drop=True)

    N = len(df_work)
    n_batches = (N + batch_size - 1) // batch_size

    print(f"[LLM] Starting demand signal detection: {N} comments, {n_batches} batches, batch_size={batch_size}, provider={model_provider}")
    if resume_from > 0:
        print(f"[LLM] Resuming — {resume_from} already-classified results loaded, {N} remaining to process")

    if df_work.empty:
        print("[INFO] All comments already classified.")
        results_df = pd.DataFrame(existing_results)
        if results_df.empty:
            results_df = pd.DataFrame(columns=["comment_id", "signal", "confidence", "reason"])
        results_df["comment_id"] = results_df["comment_id"].astype(str)
        df_output = input_df.copy()
        df_output["comment_id"] = df_output["comment_id"].astype(str)
        df_output = df_output.merge(
            results_df[["comment_id", "signal", "confidence", "reason"]],
            on="comment_id",
            how="left",
        )
        df_signals = df_output[df_output["signal"].isin(DEMAND_SIGNAL_LIST)].copy()
        print(f"[LLM] Done (checkpoint only). {len(df_signals):,} demand signals found")
        return df_output, df_signals
    else:
        start_time = time.time()

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, N)
            batch_df = df_work.iloc[start:end].reset_index(drop=True)

            processed = start + len(batch_df)
            elapsed = time.time() - start_time
            if progress_callback:
                eta = (elapsed / max(1, processed)) * (N - processed)
                progress_callback(batch_idx, n_batches, processed, N, elapsed, eta, results_count)

            user_prompt = build_user_prompt(batch_df)
            try:
                parsed = call_llm_batch(
                    api_key=api_key,
                    system_prompt=LLM_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    model_provider=model_provider,
                    model_name=model_name,
                )
            except QuotaExceededError as exc:
                exc.partial_results = list(existing_results)
                _save_partial_and_raise(exc, model_provider, n_to_analyze=n_to_analyze, input_df=input_df)

            except LLMCallError as exc:
                exc.partial_results = list(existing_results)
                _save_partial_and_raise(exc, model_provider, n_to_analyze=n_to_analyze, input_df=input_df)

            except Exception as exc:
                # Catch-all for any unexpected error (network, timeout, schema change, etc.)
                checkpoint.save(existing_results)
                raise LLMCallError(
                    model_provider,
                    f"Unexpected error ({type(exc).__name__}): {exc}",
                    batch_idx=batch_idx,
                    partial_results=list(existing_results),
                )

            existing_results.extend(parsed)
            results_count = len(existing_results)
            print(f"[LLM] Batch {batch_idx+1}/{n_batches} done | {results_count}/{N} classified | {elapsed:.0f}s elapsed")

            time.sleep(rate_limit_delay)

        checkpoint.save(existing_results)
        checkpoint.clear()

    # Merge back with full input (common path for both empty and non-empty branches)
    if results_df.empty:
        results_df = pd.DataFrame(columns=["comment_id", "signal", "confidence", "reason"])
    results_df["comment_id"] = results_df["comment_id"].astype(str)
    df_output = input_df.copy()
    df_output["comment_id"] = df_output["comment_id"].astype(str)

    df_output = df_output.merge(
        results_df[["comment_id", "signal", "confidence", "reason"]],
        on="comment_id",
        how="left",
    )

    # Filter to demand signal comments only
    df_signals = df_output[df_output["signal"].isin(DEMAND_SIGNAL_LIST)].copy()

    # Rename & reorder to match FINAL_COLS from notebook
    df_signals = df_signals.rename(columns={
        "text_original": "comment_text",
        "title": "video_title",
    })
    if "video_url" not in df_signals.columns:
        df_signals["video_url"] = (
            "https://www.youtube.com/watch?v=" + df_signals["video_id"].astype(str)
        )
    # Keep only columns that exist
    final_cols = [c for c in FINAL_COLS if c in df_signals.columns]
    df_signals = df_signals[final_cols]
    df_signals = df_signals.sort_values("confidence", ascending=False).reset_index(drop=True)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    provider_tag = model_provider
    df_output.to_parquet(OUTPUT_DIR / f"demand_signals_full_{provider_tag}_{timestamp}.parquet", index=False)
    df_output.to_csv(OUTPUT_DIR / f"demand_signals_full_{provider_tag}_{timestamp}.csv", index=False)
    df_signals.to_parquet(OUTPUT_DIR / f"demand_signals_only_{provider_tag}_{timestamp}.parquet", index=False)
    df_signals.to_csv(OUTPUT_DIR / f"demand_signals_only_{provider_tag}_{timestamp}.csv", index=False)

    print(f"[LLM] Done. Classified {len(df_output):,} comments → {len(df_signals):,} demand signals found")
    print(f"[LLM] Signal breakdown: {df_signals['signal'].value_counts().to_dict()}")

    return df_output, df_signals


def load_latest_demand_signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the most recent demand signal output files (any provider), by file mtime."""
    output_dir = Path(OUTPUT_DIR)
    full_files = sorted(output_dir.glob("demand_signals_full_*.parquet"), key=lambda p: p.stat().st_mtime)
    signal_files = sorted(output_dir.glob("demand_signals_only_*.parquet"), key=lambda p: p.stat().st_mtime)

    full_df = pd.read_parquet(full_files[-1]) if full_files else pd.DataFrame()
    signal_df = pd.read_parquet(signal_files[-1]) if signal_files else pd.DataFrame()

    # Normalize column names for backward compatibility (old files may use "title" / "text_original")
    for df in (full_df, signal_df):
        if "title" in df.columns:
            df.rename(columns={"title": "video_title"}, inplace=True)
        if "text_original" in df.columns:
            df.rename(columns={"text_original": "comment_text"}, inplace=True)

    return full_df, signal_df
