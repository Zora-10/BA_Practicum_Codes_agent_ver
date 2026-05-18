"""
Step 6: Demand Signal Detection via LLM
Classifies cleaned comments using DeepSeek or Google Gemini.
Supports both DeepSeek-chat and Gemini-2.0-flash models.
"""

import json
import random
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
    LLM_PHASE1_PROMPT,
    LLM_PHASE2_PROMPT,
)


# ── Checkpoint ─────────────────────────────────────────────────────────────────

class LLMCheckpoint:
    def __init__(self, checkpoint_file: Path):
        self.file = checkpoint_file

    def save(self, results: list[dict], shuffled_order: list[str] | None = None):
        self.file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"results": results}
        if shuffled_order is not None:
            payload["shuffled_order"] = shuffled_order
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    def load(self) -> tuple[list[dict], int, list[str] | None]:
        if not self.file.exists():
            return [], 0, None
        try:
            with open(self.file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            results = payload.get("results", [])
            shuffled_order = payload.get("shuffled_order", None)
            return results, len(results), shuffled_order
        except (json.JSONDecodeError, ValueError):
            return [], 0, None

    def clear(self):
        if self.file.exists():
            self.file.unlink()


def clear_llm_checkpoint(provider: str = "deepseek"):
    """Delete the most recent LLM checkpoint file for the given provider, or the latest one if not specified."""
    from .config import CHECKPOINT_DIR
    pattern = f"llm_{provider}_*.json"
    checkpoints = sorted(CHECKPOINT_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for cp in checkpoints:
        cp.unlink()
    return len(checkpoints)


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_user_prompt(batch_df: pd.DataFrame, phase1_context: dict | None = None) -> str:
    lines = []
    for _, row in batch_df.iterrows():
        comment_id = str(row["comment_id"])
        text = str(row["text_original"]).strip()
        video_title = str(row.get("title", "N/A"))[:80]
        entry = f"[Comment ID: {comment_id}] [Video: {video_title}]\n{text}"
        if phase1_context and comment_id in phase1_context:
            ctx = phase1_context[comment_id]
            entry += (
                f"\n[Phase 1 Classification: {ctx.get('signal', 'N/A')} | "
                f"Confidence: {ctx.get('confidence', 'N/A')} | "
                f"Reason: {ctx.get('reason', 'N/A')}]"
            )
        lines.append(entry)
    return "\n\n".join(lines)


def _extract_json(raw: str) -> list:
    """Extract JSON array from model response. Handles both plain array
    and {"results": [...]} wrapper formats. Attempts to repair truncated
    JSON (e.g. missing closing brackets/braces/trailing commas)."""
    if not raw or not raw.strip():
        raise json.JSONDecodeError("empty response", raw, 0)

    # ── Attempt 1: direct parse ──────────────────────────────────────────────
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "results" in parsed:
            return parsed["results"]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # ── Attempt 2: extract first JSON array ────────────────────────────────────
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # ── Attempt 3: extract first JSON object with "results" ───────────────────
    obj_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            if isinstance(parsed, dict) and "results" in parsed:
                return parsed["results"]
        except json.JSONDecodeError:
            pass

    # ── Attempt 4: repair trailing / missing elements in truncated JSON ───────
    # Fix common truncation patterns:
    #   - trailing comma before ] or }
    #   - unclosed strings (heuristic: count quotes parity)
    repaired = raw.strip()
    # Remove trailing comma before ] or }
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    # Remove trailing comma at end of string
    repaired = re.sub(r",\s*$", "", repaired)
    # Attempt parse again
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, dict) and "results" in parsed:
            return parsed["results"]
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # ── Attempt 5: try extracting just the array from the repaired string ─────
    json_match2 = re.search(r"\[.*\]", repaired, re.DOTALL)
    if json_match2:
        try:
            return json.loads(json_match2.group())
        except json.JSONDecodeError:
            pass

    # All attempts failed — surface the original error
    json.loads(raw)  # raises the original JSONDecodeError
    return []   # unreachable; keeps mypy happy


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
                max_tokens=8192,
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
    checkpoint: "LLMCheckpoint | None" = None,
    existing_results: list | None = None,
    shuffled_order: list[str] | None = None,
) -> Optional[list[dict]]:
    """
    Call the selected LLM and return parsed JSON results.
    model_provider: "deepseek" | "gemini"
    model_name: e.g. "deepseek-chat" or "gemini-2.0-flash"

    On JSON parse error, saves checkpoint and retries up to max_retries times
    before raising LLMCallError.
    """
    for attempt in range(max_retries):
        if model_provider == "deepseek":
            raw = call_deepseek(api_key, system_prompt, user_prompt, model=model_name, max_retries=max_retries)
        elif model_provider == "gemini":
            raw = call_gemini(api_key, system_prompt, user_prompt, model=model_name, max_retries=max_retries)
        else:
            raise ValueError(f"Unknown model_provider: {model_provider}")

        if raw is None:
            raise LLMCallError(model_provider, "LLM returned empty response", partial_results=existing_results)

        try:
            return _extract_json(raw)
        except json.JSONDecodeError as e:
            print(f"  [Attempt {attempt+1}/{max_retries}] JSON parse error: {e}")
            print(f"  Raw response snippet (first 300 chars): {raw[:300]}")
            # Save checkpoint before retry so progress is not lost
            if checkpoint is not None and existing_results is not None:
                checkpoint.save(list(existing_results), shuffled_order=shuffled_order)
                print(f"  [Checkpoint] Saved {len(existing_results)} results before retry.")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # back off before retry
            else:
                raise LLMCallError(
                    model_provider,
                    f"JSON parse error: {e}. Raw response: {raw[:500] if raw else '(empty)'}",
                    partial_results=existing_results,
                )


# ── Demand signal labels ────────────────────────────────────────────────────────

DEMAND_SIGNAL_LIST = [
    "purchase_intent",
    "problem_complaint",
    "comparison_research",
    "usage_scenario",
    "wishful_thinking",
    "supply_recommendation",
    "no_signal",
]

LLM_OUTPUT_COLS = [
    "comment_id", "signal", "confidence", "reason",
    "fit_score", "protection_score", "texture_score",
    "yellowing_concern", "installation_ease", "compatibility_score",
    "value_perception", "overall_sentiment_score", "review_quality",
    "sentiment_intensity", "urgency_score", "purchase_intent_score",
    "sarcasm_flag", "expertise_level", "specificity", "key_phrases_used",
]

NUMERIC_SCORING_COLS = [
    "fit_score", "protection_score", "texture_score",
    "yellowing_concern", "installation_ease", "compatibility_score",
    "value_perception", "overall_sentiment_score",
    "sentiment_intensity", "urgency_score", "purchase_intent_score",
    "sarcasm_flag", "expertise_level", "specificity",
]

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
    "product_categories", "video_context",
    "fit_score", "protection_score", "texture_score",
    "yellowing_concern", "installation_ease", "compatibility_score",
    "value_perception", "overall_sentiment_score", "review_quality",
    "sentiment_intensity", "urgency_score", "purchase_intent_score",
    "sarcasm_flag", "expertise_level", "specificity",
    "key_phrases_used",
]

# ── Standalone Phase 1: Classification-only ─────────────────────────────────────

def _save_phase1_output(phase1_df: pd.DataFrame, input_df: pd.DataFrame, model_provider: str) -> Path:
    """Save Phase 1 results to a durable parquet file for Phase 2 to reload on restart."""
    analyzed = phase1_df.dropna(subset=["signal"]).copy()
    analyzed["comment_id"] = analyzed["comment_id"].astype(str)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"phase1_results_{model_provider}_{ts}.parquet"
    analyzed.to_parquet(path, index=False)
    # Also overwrite the "latest" file so Phase 2 can find it easily
    latest = OUTPUT_DIR / f"phase1_results_{model_provider}_latest.parquet"
    analyzed.to_parquet(latest, index=False)
    print(f"[LLM Phase1] Saved {len(analyzed):,} analyzed rows to {latest}")
    return latest


def load_phase1_checkpoint(model_provider: str) -> tuple[pd.DataFrame, bool]:
    """
    Load the latest Phase 1 results parquet.
    Returns (phase1_df, found). If not found, returns (empty DataFrame, False).
    """
    latest = OUTPUT_DIR / f"phase1_results_{model_provider}_latest.parquet"
    if not latest.exists():
        return pd.DataFrame(), False
    df = pd.read_parquet(latest)
    df["comment_id"] = df["comment_id"].astype(str)
    return df, True


def _save_phase2_output(phase1_df: pd.DataFrame, phase2_df: pd.DataFrame,
                        input_df: pd.DataFrame, model_provider: str) -> None:
    """Save Phase 2 results and final merged output."""
    p2 = phase2_df.copy()
    p2["comment_id"] = p2["comment_id"].astype(str)
    p1 = phase1_df[["comment_id", "signal", "confidence", "reason"]].copy()
    p1["comment_id"] = p1["comment_id"].astype(str)
    results = p1.merge(p2, on="comment_id", how="left", suffixes=("", "_p2"))
    results = results.loc[:, ~results.columns.str.endswith("_p2")]
    no_sig = results["signal"] == "no_signal"
    for col in NUMERIC_SCORING_COLS:
        if col in results.columns:
            results.loc[no_sig, col] = results.loc[no_sig, col].fillna(0.0)
    for col in ["key_phrases_used", "review_quality", "overall_sentiment_score"]:
        if col not in results.columns:
            results[col] = None if col == "key_phrases_used" else ("low" if col == "review_quality" else 0.0)
    results["key_phrases_used"] = results["key_phrases_used"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    for col in NUMERIC_SCORING_COLS:
        if col in results.columns:
            results[col] = pd.to_numeric(results[col], errors="coerce")
    cols = [c for c in LLM_OUTPUT_COLS if c in results.columns]
    df_out = input_df.copy()
    df_out["comment_id"] = df_out["comment_id"].astype(str)
    df_out = df_out.merge(results[cols], on="comment_id", how="left")
    for col in NUMERIC_SCORING_COLS:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_out.to_parquet(OUTPUT_DIR / f"demand_signals_full_{model_provider}_{ts}.parquet", index=False)
    df_out.to_csv(OUTPUT_DIR / f"demand_signals_full_{model_provider}_{ts}.csv", index=False)
    signals = df_out[df_out["signal"].isin(DEMAND_SIGNAL_LIST)].copy()
    signals = signals.rename(columns={"text_original": "comment_text", "title": "video_title"})
    if "video_url" not in signals.columns:
        signals["video_url"] = "https://www.youtube.com/watch?v=" + signals["video_id"].astype(str)
    final_cols = [c for c in FINAL_COLS if c in signals.columns]
    signals = signals[final_cols].sort_values("confidence", ascending=False).reset_index(drop=True)
    signals.to_parquet(OUTPUT_DIR / f"demand_signals_only_{model_provider}_{ts}.parquet", index=False)
    signals.to_csv(OUTPUT_DIR / f"demand_signals_only_{model_provider}_{ts}.csv", index=False)
    print(f"[LLM Phase2] Done. {len(signals):,} demand signals → output saved.")


def run_phase1_classification(
    api_key: str,
    input_df: pd.DataFrame,
    *,
    model_provider: str = "deepseek",
    model_name: str = "deepseek-chat",
    batch_size: int = 20,
    rate_limit_delay: float = 1.5,
    max_retries: int = 3,
    progress_callback=None,
    n_to_analyze: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 1: Classify ALL comments as demand_signal or no_signal.
    Returns (phase1_results_df, signal_df) where signal_df contains only non-no_signal rows.
    Checkpoint: llm_{provider}_phase1_{ts}.json
    """
    available = [c for c in ANALYSIS_COLS if c in input_df.columns]
    df_work = input_df[available].copy()
    df_work["video_url"] = "https://www.youtube.com/watch?v=" + df_work["video_id"].astype(str)

    if n_to_analyze is not None and n_to_analyze < len(df_work):
        df_work = df_work.head(n_to_analyze).reset_index(drop=True)

    N = len(df_work)
    n_batches = (N + batch_size - 1) // batch_size
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cp1_file = CHECKPOINT_DIR / f"llm_{model_provider}_phase1_{ts}.json"
    cp1 = LLMCheckpoint(cp1_file)

    phase1_results, p1_resume, saved_order = cp1.load()
    classified_ids = {r["comment_id"] for r in phase1_results}

    if p1_resume == 0:
        shuffled_ids = df_work["comment_id"].astype(str).tolist()
        random.shuffle(shuffled_ids)
    else:
        shuffled_ids = saved_order if saved_order else df_work["comment_id"].astype(str).tolist()
        shuffled_ids = [cid for cid in shuffled_ids
                       if cid not in classified_ids and cid in df_work["comment_id"].astype(str).values]

    order_map = {cid: i for i, cid in enumerate(shuffled_ids)}
    df_work = df_work.copy()
    df_work["_order"] = df_work["comment_id"].astype(str).map(order_map)
    df_work = df_work.dropna(subset=["_order"]).sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    df_active = df_work.iloc[p1_resume:].reset_index(drop=True)
    n_active = len(df_active)
    n_batches_active = (n_active + batch_size - 1) // batch_size if n_active > 0 else 0

    print(f"[LLM Phase1] {p1_resume}/{N} done. Processing {n_active} remaining in {n_batches_active} batches.")
    start_time = time.time()

    for batch_idx in range(n_batches_active):
        start = batch_idx * batch_size
        end   = min(start + batch_size, n_active)
        batch_df = df_active.iloc[start:end].reset_index(drop=True)

        processed = p1_resume + start + len(batch_df)
        elapsed   = time.time() - start_time
        if progress_callback:
            eta = (elapsed / max(1, processed)) * (N - processed)
            progress_callback(batch_idx, n_batches_active, processed, N, elapsed, eta, len(phase1_results), phase=1)

        user_prompt = build_user_prompt(batch_df)
        try:
            parsed = call_llm_batch(
                api_key=api_key,
                system_prompt=LLM_PHASE1_PROMPT,
                user_prompt=user_prompt,
                model_provider=model_provider,
                model_name=model_name,
                max_retries=max_retries,
                checkpoint=cp1,
                existing_results=list(phase1_results),
                shuffled_order=shuffled_ids,
            )
        except QuotaExceededError as exc:
            exc.partial_results = list(phase1_results)
            _save_partial_and_raise(exc, model_provider, n_to_analyze=n_to_analyze, input_df=input_df)
        except LLMCallError as exc:
            exc.partial_results = list(phase1_results)
            _save_partial_and_raise(exc, model_provider, n_to_analyze=n_to_analyze, input_df=input_df)
        except Exception as exc:
            cp1.save(list(phase1_results), shuffled_order=shuffled_ids)
            raise LLMCallError(model_provider, f"Unexpected error ({type(exc).__name__}): {exc}",
                               batch_idx=batch_idx, partial_results=list(phase1_results))

        phase1_results.extend(parsed)
        elapsed = time.time() - start_time
        print(f"[LLM P1] Batch {batch_idx+1}/{n_batches_active} | {len(phase1_results)}/{N} classified | {elapsed:.0f}s")
        time.sleep(rate_limit_delay)

    cp1.save(list(phase1_results), shuffled_order=shuffled_ids)

    p1_df = pd.DataFrame(phase1_results)
    p1_df["comment_id"] = p1_df["comment_id"].astype(str)
    signal_ids  = set(p1_df[p1_df["signal"] != "no_signal"]["comment_id"])
    no_sig_ids  = set(p1_df[p1_df["signal"] == "no_signal"]["comment_id"])
    signal_df   = p1_df[p1_df["signal"] != "no_signal"].copy()

    print(f"[LLM Phase1] Done. {len(signal_ids):,} signal / {len(no_sig_ids):,} no_signal")
    _save_phase1_output(p1_df, input_df, model_provider)
    return p1_df, signal_df


def run_phase2_scoring(
    api_key: str,
    phase1_df: pd.DataFrame,
    input_df: pd.DataFrame,
    *,
    model_provider: str = "deepseek",
    model_name: str = "deepseek-chat",
    batch_size: int = 20,
    rate_limit_delay: float = 1.5,
    max_retries: int = 3,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Phase 2: Score ONLY non-no_signal comments from Phase 1 on 13 dimensions.
    Returns phase2_df with scoring columns merged.
    Checkpoint: llm_{provider}_phase2_{ts}.json
    """
    signal_ids = set(phase1_df[phase1_df["signal"] != "no_signal"]["comment_id"])
    if not signal_ids:
        print("[LLM Phase2] No signal comments found — skipping Phase 2.")
        return pd.DataFrame()

    available = [c for c in ANALYSIS_COLS if c in input_df.columns]
    df_work = input_df[available].copy()
    df_work["video_url"] = "https://www.youtube.com/watch?v=" + df_work["video_id"].astype(str)
    df_work = df_work[df_work["comment_id"].astype(str).isin(signal_ids)].reset_index(drop=True)

    # Build Phase 1 context dict keyed by comment_id for use in prompt
    phase1_context = {}
    if not phase1_df.empty and "signal" in phase1_df.columns:
        for _, row in phase1_df.iterrows():
            cid = str(row["comment_id"])
            phase1_context[cid] = {
                "signal": str(row.get("signal", "")),
                "confidence": row.get("confidence", ""),
                "reason": str(row.get("reason", "")),
            }

    n_p2 = len(df_work)
    n_batches_p2 = (n_p2 + batch_size - 1) // batch_size
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cp2_file = CHECKPOINT_DIR / f"llm_{model_provider}_phase2_{ts}.json"
    cp2 = LLMCheckpoint(cp2_file)

    phase2_results, p2_resume, _ = cp2.load()
    scored_ids = {r["comment_id"] for r in phase2_results}

    df_active = df_work[~df_work["comment_id"].astype(str).isin(scored_ids)].reset_index(drop=True)
    n_active = len(df_active)
    n_batches_active = (n_active + batch_size - 1) // batch_size if n_active > 0 else 0

    print(f"[LLM Phase2] {p2_resume}/{n_p2} done. Processing {n_active} remaining in {n_batches_active} batches.")
    start_time = time.time()

    for batch_idx in range(n_batches_active):
        start = batch_idx * batch_size
        end   = min(start + batch_size, n_active)
        batch_df = df_active.iloc[start:end].reset_index(drop=True)

        processed = p2_resume + start + len(batch_df)
        elapsed  = time.time() - start_time
        if progress_callback:
            eta = (elapsed / max(1, processed)) * (n_p2 - processed)
            progress_callback(batch_idx, n_batches_active, processed, n_p2, elapsed, eta, len(phase2_results), phase=2)

        user_prompt = build_user_prompt(batch_df, phase1_context=phase1_context)
        try:
            parsed = call_llm_batch(
                api_key=api_key,
                system_prompt=LLM_PHASE2_PROMPT,
                user_prompt=user_prompt,
                model_provider=model_provider,
                model_name=model_name,
                max_retries=max_retries,
                checkpoint=cp2,
                existing_results=list(phase2_results),
                shuffled_order=None,
            )
        except QuotaExceededError as exc:
            exc.partial_results = list(phase2_results)
            _save_partial_and_raise(exc, model_provider, n_to_analyze=None, input_df=input_df)
        except LLMCallError as exc:
            exc.partial_results = list(phase2_results)
            _save_partial_and_raise(exc, model_provider, n_to_analyze=None, input_df=input_df)
        except Exception as exc:
            cp2.save(list(phase2_results), shuffled_order=None)
            raise LLMCallError(model_provider, f"Unexpected error ({type(exc).__name__}): {exc}",
                              batch_idx=batch_idx, partial_results=list(phase2_results))

        phase2_results.extend(parsed)
        elapsed = time.time() - start_time
        print(f"[LLM P2] Batch {batch_idx+1}/{n_batches_active} | {len(phase2_results)}/{n_p2} scored | {elapsed:.0f}s")
        time.sleep(rate_limit_delay)

    cp2.save(list(phase2_results), shuffled_order=None)
    p2_df_result = pd.DataFrame(phase2_results)
    p2_df_result["comment_id"] = p2_df_result["comment_id"].astype(str)
    print(f"[LLM Phase2] Done. {len(p2_df_result):,} comments scored.")
    _save_phase2_output(phase1_df, p2_df_result, input_df, model_provider)
    return p2_df_result


def _merge_and_build_output(
    phase1_df: pd.DataFrame,
    phase2_df: pd.DataFrame,
    input_df: pd.DataFrame,
    model_provider: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge Phase 1 + Phase 2 results and build output DataFrames."""
    phase2_scores = phase2_df.copy()
    phase2_scores["comment_id"] = phase2_scores["comment_id"].astype(str)

    results_df = phase1_df.merge(phase2_scores, on="comment_id", how="left", suffixes=("", "_p2"))
    results_df = results_df.loc[:, ~results_df.columns.str.endswith("_p2")]

    no_sig_mask = results_df["signal"] == "no_signal"
    for col in NUMERIC_SCORING_COLS:
        if col in results_df.columns:
            results_df.loc[no_sig_mask, col] = results_df.loc[no_sig_mask, col].fillna(0.0)
        else:
            results_df[col] = 0.0

    for col in ["key_phrases_used", "review_quality", "overall_sentiment_score"]:
        if col not in results_df.columns:
            results_df[col] = None if col == "key_phrases_used" else ("low" if col == "review_quality" else 0.0)

    results_df["key_phrases_used"] = results_df["key_phrases_used"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    for col in NUMERIC_SCORING_COLS:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    results_cols = [c for c in LLM_OUTPUT_COLS if c in results_df.columns]
    df_output = input_df.copy()
    df_output["comment_id"] = df_output["comment_id"].astype(str)
    df_output = df_output.merge(results_df[results_cols], on="comment_id", how="left")

    for col in NUMERIC_SCORING_COLS:
        if col in df_output.columns:
            df_output[col] = pd.to_numeric(df_output[col], errors="coerce")

    df_signals = df_output[df_output["signal"].isin(DEMAND_SIGNAL_LIST)].copy()
    df_signals = df_signals.rename(columns={"text_original": "comment_text", "title": "video_title"})
    if "video_url" not in df_signals.columns:
        df_signals["video_url"] = "https://www.youtube.com/watch?v=" + df_signals["video_id"].astype(str)
    final_cols = [c for c in FINAL_COLS if c in df_signals.columns]
    df_signals = df_signals[final_cols]
    df_signals = df_signals.sort_values("confidence", ascending=False).reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    provider_tag = model_provider
    df_output.to_parquet(OUTPUT_DIR / f"demand_signals_full_{provider_tag}_{timestamp}.parquet", index=False)
    df_output.to_csv(OUTPUT_DIR / f"demand_signals_full_{provider_tag}_{timestamp}.csv", index=False)
    df_signals.to_parquet(OUTPUT_DIR / f"demand_signals_only_{provider_tag}_{timestamp}.parquet", index=False)
    df_signals.to_csv(OUTPUT_DIR / f"demand_signals_only_{provider_tag}_{timestamp}.csv", index=False)

    return df_output, df_signals


def run_demand_signal_detection(
    api_key: str,
    input_df: pd.DataFrame | None = None,
    *,
    model_provider: str = "deepseek",
    model_name: str = "deepseek-chat",
    batch_size: int = 20,
    save_every: int = 5,
    rate_limit_delay: float = 1.5,
    max_retries: int = 3,
    checkpoint_file: Path | None = None,
    progress_callback=None,
    n_to_analyze: int | None = None,
    use_two_phase: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run LLM classification on cleaned comments.
    Delegates to run_phase1_classification + run_phase2_scoring.
    Returns: (full_df, signal_df)
    """
    if input_df is None:
        input_df = pd.read_parquet(CLEANED_DIR / "cleaned_comments_linked.parquet")
    if input_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if n_to_analyze is not None and n_to_analyze < len(input_df):
        input_df = input_df.head(n_to_analyze).reset_index(drop=True)

    # Phase 1
    p1_df, signal_df = run_phase1_classification(
        api_key=api_key,
        input_df=input_df,
        model_provider=model_provider,
        model_name=model_name,
        batch_size=batch_size,
        rate_limit_delay=rate_limit_delay,
        max_retries=max_retries,
        progress_callback=progress_callback,
    )

    # Phase 2
    p2_df = pd.DataFrame()
    if use_two_phase and not signal_df.empty:
        p2_df = run_phase2_scoring(
            api_key=api_key,
            phase1_df=p1_df,
            input_df=input_df,
            model_provider=model_provider,
            model_name=model_name,
            batch_size=batch_size,
            rate_limit_delay=rate_limit_delay,
            max_retries=max_retries,
            progress_callback=progress_callback,
        )

    # Merge & output
    df_output, df_signals = _merge_and_build_output(
        phase1_df=p1_df,
        phase2_df=p2_df,
        input_df=input_df,
        model_provider=model_provider,
    )
    return df_output, df_signals


def load_latest_demand_signals() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the most recent demand signal output files (any provider), by file mtime."""
    output_dir = Path(OUTPUT_DIR)
    full_files = sorted(output_dir.glob("demand_signals_full_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    signal_files = sorted(output_dir.glob("demand_signals_only_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)

    full_df = pd.read_parquet(full_files[0]) if full_files else pd.DataFrame()
    signal_df = pd.read_parquet(signal_files[0]) if signal_files else pd.DataFrame()

    # Normalize column names for backward compatibility (old files may use "title" / "text_original")
    for df in (full_df, signal_df):
        if "title" in df.columns:
            df.rename(columns={"title": "video_title"}, inplace=True)
        if "text_original" in df.columns:
            df.rename(columns={"text_original": "comment_text"}, inplace=True)
        # Cast numeric scoring columns to float
        for col in NUMERIC_SCORING_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    # Safety net: exclude no_signal from signal_df regardless of how it was saved
    if not signal_df.empty and "signal" in signal_df.columns:
        signal_df = signal_df[signal_df["signal"] != "no_signal"].copy()

    return full_df, signal_df


def list_signal_versions() -> list[dict]:
    """
    Return all saved signal versions sorted newest-first.
    Each entry: {filename, mtime, provider, rows_full, rows_signals}
    """
    output_dir = Path(OUTPUT_DIR)
    full_files = sorted(
        output_dir.glob("demand_signals_full_*.parquet"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    versions = []
    for f in full_files:
        # Extract timestamp from filename: demand_signals_full_{provider}_{ts}.parquet
        ts_str = f.stem.replace("demand_signals_full_", "")
        try:
            dt = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            label = dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            label = ts_str
        try:
            df = pd.read_parquet(f)
            rows = len(df)
        except Exception:
            rows = "?"
        provider = ts_str.split("_")[0] if "_" in ts_str else "?"
        versions.append({
            "filename": f.name,
            "label": label,
            "provider": provider,
            "rows": rows,
        })
    return versions


def load_signal_version(filename: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load a specific signal version by its full filename."""
    output_dir = Path(OUTPUT_DIR)
    full_path = output_dir / filename
    signal_path = output_dir / filename.replace("_full_", "_only_")

    full_df = pd.read_parquet(full_path) if full_path.exists() else pd.DataFrame()
    signal_df = pd.read_parquet(signal_path) if signal_path.exists() else pd.DataFrame()

    for df in (full_df, signal_df):
        if not df.empty:
            if "title" in df.columns:
                df.rename(columns={"title": "video_title"}, inplace=True)
            if "text_original" in df.columns:
                df.rename(columns={"text_original": "comment_text"}, inplace=True)
            for col in NUMERIC_SCORING_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

    if not signal_df.empty and "signal" in signal_df.columns:
        signal_df = signal_df[signal_df["signal"] != "no_signal"].copy()

    return full_df, signal_df
