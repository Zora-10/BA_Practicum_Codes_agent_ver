"""
Step 2: YouTube Data Collection
Collects videos and comments via YouTube Data API v3.
Fidelity: mirrors 02_data_collection.ipynb exactly.
"""

import json
import random
import socket
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import httplib2
import ssl

from .config import BASE_DIR


# ── HTTP / Network Settings ────────────────────────────────────────────────────

DEFAULT_TIMEOUT = 180   # seconds — generous to avoid premature socket timeouts
CONNECT_TIMEOUT = 30    # seconds — httplib2 connect timeout
READ_TIMEOUT    = 150   # seconds — httplib2 read timeout


class QuotaExceededException(Exception):
    pass


class CollectionStoppedException(Exception):
    """Raised when the user requests to stop collection."""
    pass


def _make_http_with_timeout(timeout: int = DEFAULT_TIMEOUT) -> httplib2.Http:
    """Create an httplib2 Http object with explicit read timeout.

    Setting socket.setdefaulttimeout alone is NOT sufficient because httplib2
    manages its own timeout internally. We pass timeout= to the Http constructor
    and also patch socket.getaddrinfo so DNS hangs on macOS are mitigated.
    """
    socket.setdefaulttimeout(timeout)
    http = httplib2.Http(timeout=timeout)

    _original_getaddrinfo = socket.getaddrinfo

    def _patched_getaddrinfo(*args, **kwargs):
        try:
            return _original_getaddrinfo(*args, **kwargs)
        except socket.gaierror:
            raise
        except OSError:
            raise socket.gaierror("DNS resolution failed for: %s" % (args[0],))

    socket.getaddrinfo = _patched_getaddrinfo
    return http


# ── Load Latest Collected ─────────────────────────────────────────────────────

def load_latest_collected(
    collection_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the most recent collected parquet files from disk.
    
    Note: Excludes *_master.parquet files to avoid conflicts.
    """
    if collection_dir is None:
        collection_dir = BASE_DIR / "data" / "collection_output"

    # Load staging files (exclude master files)
    video_files = sorted(collection_dir.glob("videos_*.parquet"))
    comment_files = sorted(collection_dir.glob("comments_*.parquet"))
    run_files = sorted(collection_dir.glob("runs_*.parquet"))
    
    # Exclude master files from the list
    video_files = [f for f in video_files if "master" not in f.name]
    comment_files = [f for f in comment_files if "master" not in f.name]
    run_files = [f for f in run_files if "master" not in f.name]

    if not video_files:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"[LOAD] Loading {len(video_files)} video files, {len(comment_files)} comment files")
    
    videos_list = [pd.read_parquet(f) for f in video_files]
    comments_list = [pd.read_parquet(f) for f in comment_files]
    runs_list = [pd.read_parquet(f) for f in run_files] if run_files else [pd.DataFrame()]

    videos = pd.concat(videos_list, ignore_index=True).drop_duplicates(subset=["video_id"], keep="last")
    comments = pd.concat(comments_list, ignore_index=True).drop_duplicates(subset=["comment_id"], keep="last")
    runs = pd.concat(runs_list, ignore_index=True)

    print(f"[LOAD] Loaded: {len(videos)} videos, {len(comments)} comments, {len(runs)} runs")
    return videos, comments, runs


# ── Utilities ─────────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_rfc3339(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def safe_sleep(seconds: float = 0.2):
    time.sleep(seconds)


def deduplicate_videos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    x = df.copy()
    if "video_published_at" in x.columns:
        x["video_published_at"] = pd.to_datetime(x["video_published_at"], utc=True, errors="coerce")
    x = x.sort_values(
        ["video_published_at", "fetched_at_utc"], ascending=[False, False], na_position="last"
    )
    x = x.drop_duplicates(subset=["video_id"], keep="first")
    return x.reset_index(drop=True)


def deduplicate_comments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    x = df.copy()
    if "published_at" in x.columns:
        x["published_at"] = pd.to_datetime(x["published_at"], utc=True, errors="coerce")
    if "updated_at" in x.columns:
        x["updated_at"] = pd.to_datetime(x["updated_at"], utc=True, errors="coerce")
    x = x.sort_values(
        ["updated_at", "published_at", "fetched_at_utc"],
        ascending=[False, False, False],
        na_position="last",
    )
    x = x.drop_duplicates(subset=["comment_id"], keep="first")
    return x.reset_index(drop=True)


# ── Checkpoint ─────────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, checkpoint_file: Path):
        self.file = checkpoint_file

    def save(self, keyword_index: int, last_keyword: str):
        data = {
            "keyword_index": keyword_index,
            "last_keyword": last_keyword,
            "saved_at_utc": to_rfc3339(utc_now()),
        }
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[CHECKPOINT] Saved: index={keyword_index}, keyword='{last_keyword}'")

    def load(self) -> int:
        if not self.file.exists():
            return 0
        try:
            with open(self.file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[CHECKPOINT] Found checkpoint: index={data['keyword_index']}, "
                  f"last_keyword='{data['last_keyword']}'")
            print(f"[CHECKPOINT] Saved at: {data['saved_at_utc']}")
            return data.get("keyword_index", 0)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[WARN] Failed to load checkpoint: {e}. Starting from beginning.")
            return 0

    def clear(self):
        if self.file.exists():
            self.file.unlink()
            print("[CHECKPOINT] Cleared — all keywords processed!")


def clear_checkpoint(checkpoint_file: Path | None = None) -> bool:
    """Delete the collection checkpoint so the next run starts from keyword 0.

    Returns True if a checkpoint was deleted, False if none existed.
    """
    if checkpoint_file is None:
        checkpoint_file = BASE_DIR / "data" / "collection_output" / "collection_checkpoint.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        print(f"[CHECKPOINT] Cleared: {checkpoint_file}")
        return True
    return False


# ── YouTube API Functions ──────────────────────────────────────────────────────

def search_recent_video_ids(
    youtube_client,
    query: str,
    published_after: datetime,
    max_pages: int = 2,
    results_per_page: int = 50,
    region_code: Optional[str] = None,
    relevance_language: Optional[str] = None,
) -> list[str]:
    video_ids = []
    next_page_token = None

    for page_num in range(max_pages):
        try:
            request = youtube_client.search().list(
                part="snippet",
                q=query,
                type="video",
                order="date",
                maxResults=min(results_per_page, 50),
                publishedAfter=to_rfc3339(published_after),
                pageToken=next_page_token,
                regionCode=region_code,
                relevanceLanguage=relevance_language,
            )
            response = _execute_with_retries(request)
        except HttpError as e:
            if "quotaExceeded" in str(e):
                print(f"[QUOTA EXCEEDED] Stopping all further requests.")
                raise QuotaExceededException(f"Quota exceeded for query={query}")
            print(f"[ERROR] search failed for query={query}: {e}")
            break

        items = response.get("items", [])
        page_ids = [
            item["id"]["videoId"]
            for item in items
            if item.get("id", {}).get("videoId")
        ]
        video_ids.extend(page_ids)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

        safe_sleep(0.2)

    return list(dict.fromkeys(video_ids))


def fetch_video_details(youtube_client, video_ids: list[str], keyword_used: str) -> pd.DataFrame:
    rows = []
    if not video_ids:
        return pd.DataFrame()

    for batch in chunked(video_ids, 50):
        try:
            request = youtube_client.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(batch),
            )
            response = _execute_with_retries(request)
        except HttpError as e:
            if "quotaExceeded" in str(e):
                print(f"[QUOTA EXCEEDED] Stopping video details fetch.")
                raise QuotaExceededException("Quota exceeded in fetch_video_details")
            print(f"[ERROR] fetch_video_details failed: {e}")
            continue

        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            content = item.get("contentDetails", {})

            rows.append({
                "video_id": item.get("id"),
                "channel_id": snippet.get("channelId"),
                "channel_title": snippet.get("channelTitle"),
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "tags": " | ".join(snippet.get("tags", []))
                if isinstance(snippet.get("tags", []), list)
                else None,
                "category_id": snippet.get("categoryId"),
                "default_language": snippet.get("defaultLanguage"),
                "default_audio_language": snippet.get("defaultAudioLanguage"),
                "video_published_at": snippet.get("publishedAt"),
                "duration": content.get("duration"),
                "view_count": int(stats["viewCount"]) if "viewCount" in stats else None,
                "like_count": int(stats["likeCount"]) if "likeCount" in stats else None,
                "comment_count": int(stats["commentCount"])
                if "commentCount" in stats
                else None,
                "search_keyword": keyword_used,
                "fetched_at_utc": to_rfc3339(utc_now()),
                "fetched_by": "dashboard_user",
            })

        safe_sleep(0.2)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["video_published_at"] = pd.to_datetime(
            df["video_published_at"], utc=True, errors="coerce"
        )
    return df


def fetch_comments_for_video(
    youtube_client,
    video_id: str,
    keyword_used: str,
    max_pages: int = 5,
    page_size: int = 100,
) -> pd.DataFrame:
    rows = []
    next_page_token = None

    for _ in range(max_pages):
        try:
            request = youtube_client.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=min(page_size, 100),
                pageToken=next_page_token,
                textFormat="plainText",
                order="time",
            )
            response = _execute_with_retries(request)
        except HttpError as e:
            if "quotaExceeded" in str(e):
                print(f"[QUOTA EXCEEDED] Stopping comment fetch for video {video_id}.")
                raise QuotaExceededException(f"Quota exceeded in fetch_comments_for_video")
            print(f"[WARN] comments skipped for video {video_id}: {e}")
            break

        for item in response.get("items", []):
            thread_id = item.get("id")
            top = item.get("snippet", {}).get("topLevelComment", {})
            top_snippet = top.get("snippet", {})

            rows.append({
                "comment_id": top.get("id"),
                "video_id": video_id,
                "thread_id": thread_id,
                "parent_comment_id": None,
                "is_reply": False,
                "author_channel_id": (
                    top_snippet.get("authorChannelId", {}).get("value")
                    if isinstance(top_snippet.get("authorChannelId"), dict)
                    else None
                ),
                "author_display_name": top_snippet.get("authorDisplayName"),
                "text_original": top_snippet.get("textOriginal"),
                "text_display": top_snippet.get("textDisplay"),
                "like_count": top_snippet.get("likeCount"),
                "published_at": top_snippet.get("publishedAt"),
                "updated_at": top_snippet.get("updatedAt"),
                "total_reply_count": item.get("snippet", {}).get("totalReplyCount"),
                "search_keyword": keyword_used,
                "fetched_at_utc": to_rfc3339(utc_now()),
                "fetched_by": "dashboard_user",
            })

            for reply in item.get("replies", {}).get("comments", []):
                rs = reply.get("snippet", {})
                rows.append({
                    "comment_id": reply.get("id"),
                    "video_id": video_id,
                    "thread_id": thread_id,
                    "parent_comment_id": rs.get("parentId"),
                    "is_reply": True,
                    "author_channel_id": (
                        rs.get("authorChannelId", {}).get("value")
                        if isinstance(rs.get("authorChannelId"), dict)
                        else None
                    ),
                    "author_display_name": rs.get("authorDisplayName"),
                    "text_original": rs.get("textOriginal"),
                    "text_display": rs.get("textDisplay"),
                    "like_count": rs.get("likeCount"),
                    "published_at": rs.get("publishedAt"),
                    "updated_at": rs.get("updatedAt"),
                    "total_reply_count": None,
                    "search_keyword": keyword_used,
                    "fetched_at_utc": to_rfc3339(utc_now()),
                    "fetched_by": "dashboard_user",
                })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
        safe_sleep(0.2)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True, errors="coerce")
    return df


# ── Retry wrapper ─────────────────────────────────────────────────────────────

def _execute_with_retries(request, max_retries: int = 5) -> httplib2.Response:
    """Execute a googleapiclient request with retry-on-timeout logic.

    Uses exponential backoff with jitter. Retries on socket.timeout, socket.gaierror,
    OSError (ECONNRESET, ETIMEDOUT, etc.), TimeoutError and ConnectionError.
    Also catches ssl.SSLError for certificate/handshake hangs on macOS.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return request.execute(num_retries=0)
        except HttpError:
            raise
        except (
            socket.timeout,
            socket.gaierror,
            ConnectionError,
            OSError,
            httplib2.ServerNotFoundError,
            TimeoutError,
            ssl.SSLError,
        ) as exc:
            last_exc = exc
            if attempt < max_retries:
                # Exponential backoff: ~4s, ~8s, ~16s, ~32s, ~64s
                wait = (random.random() + 0.5) * (2 ** (attempt + 1))
                print(f"[RETRY] attempt {attempt + 1}/{max_retries} failed: {exc}. "
                      f"Waiting {wait:.1f}s before retry.")
                time.sleep(wait)
            else:
                print(f"[FATAL] all {max_retries + 1} attempts failed: {exc}")
    raise last_exc


# ── Main Collection Pipeline ─────────────────────────────────────────────────────

def run_collection(
    api_key: str,
    keyword_shard: list[str],
    *,
    days_back: int = 30,
    search_max_pages: int = 2,
    comment_max_pages: int = 5,
    search_results_per_page: int = 50,
    comments_per_page: int = 100,
    region_code: Optional[str] = "US",
    relevance_language: Optional[str] = "en",
    fetch_comments: bool = True,
    output_dir: Path | None = None,
    checkpoint_file: Path | None = None,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Run YouTube data collection.

    Returns: (videos_df, comments_df, runs_log_df, quota_exceeded)
    """
    if output_dir is None:
        output_dir = BASE_DIR / "data" / "collection_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_file is None:
        checkpoint_file = output_dir / "collection_checkpoint.json"

    checkpoint = CheckpointManager(checkpoint_file)
    start_from_index = checkpoint.load()

    published_after = utc_now() - timedelta(days=days_back)
    run_tag = utc_now().strftime("%Y%m%d_%H%M%S")

    print(f"\n===== COLLECTION STARTED =====")
    print(f"Collecting videos published after: {published_after}")
    print(f"[INFO] Total keywords: {len(keyword_shard)}")
    if start_from_index > 0:
        print(f"[INFO] Resuming: {start_from_index}/{len(keyword_shard)} already processed")
    else:
        print(f"[INFO] Starting from keyword index: 0")
    print(f"YouTube client ready (timeout={DEFAULT_TIMEOUT}s, max_retries=5).")

    http = _make_http_with_timeout(DEFAULT_TIMEOUT)
    youtube = build("youtube", "v3", developerKey=api_key, http=http)

    all_video_frames = []
    all_comment_frames = []
    run_log_rows = []
    quota_exceeded = False
    collection_error = None

    for idx, kw in enumerate(keyword_shard):
        if idx < start_from_index:
            print(f"[SKIP] {idx+1}/{len(keyword_shard)}: '{kw}' (before checkpoint)")
            continue

        candidate_video_ids = []
        video_df = pd.DataFrame()
        comment_count_this_kw = 0

        if progress_callback:
            progress_callback(idx, len(keyword_shard), kw, 0, 0, "searching")

        print(f"\n{'='*50}")
        print(f"[INFO] Processing keyword {idx+1}/{len(keyword_shard)}: {kw}")

        print(f"[SEARCH] Fetching video IDs for query: '{kw}' ...")
        try:
            candidate_video_ids = search_recent_video_ids(
                youtube_client=youtube,
                query=kw,
                published_after=published_after,
                max_pages=search_max_pages,
                results_per_page=search_results_per_page,
                region_code=region_code,
                relevance_language=relevance_language,
            )
        except Exception as e:
            print(f"[ERROR] Failed to search for videos: {e}")
            collection_error = str(e)
            break

        print(f"[SEARCH] Candidate videos found: {len(candidate_video_ids)}")

        if progress_callback:
            progress_callback(idx, len(keyword_shard), kw, len(candidate_video_ids), 0, "fetching_videos")

        print(f"[VIDEO DETAILS] Fetching details for {len(candidate_video_ids)} videos ...")
        try:
            video_df = fetch_video_details(
                youtube_client=youtube,
                video_ids=candidate_video_ids,
                keyword_used=kw,
            )
        except Exception as e:
            print(f"[ERROR] Failed to fetch video details: {e}")
            collection_error = str(e)
            break

        video_df = deduplicate_videos(video_df)
        print(f"[VIDEO DETAILS] Videos fetched: {len(video_df)}")

        if not video_df.empty:
            all_video_frames.append(video_df)

        comment_count_this_kw = 0
        if fetch_comments and not video_df.empty:
            if progress_callback:
                progress_callback(idx, len(keyword_shard), kw, len(video_df), 0, "fetching_comments")
            print(f"[COMMENTS] Fetching comments for {len(video_df)} videos ...")
            for vid_idx, vid in enumerate(video_df["video_id"].tolist()):
                try:
                    cdf = fetch_comments_for_video(
                        youtube_client=youtube,
                        video_id=vid,
                        keyword_used=kw,
                        max_pages=comment_max_pages,
                        page_size=comments_per_page,
                    )
                    if not cdf.empty:
                        all_comment_frames.append(cdf)
                        comment_count_this_kw += len(cdf)
                    if (vid_idx + 1) % 10 == 0 or vid_idx == len(video_df) - 1:
                        print(f"[COMMENTS]   Video {vid_idx+1}/{len(video_df)}: "
                              f"{vid} → {len(cdf)} comments")
                        if progress_callback:
                            progress_callback(idx, len(keyword_shard), kw, len(video_df), comment_count_this_kw, "fetching_comments")
                except QuotaExceededException:
                    print(f"[QUOTA EXCEEDED] Stopping comment fetching.")
                    quota_exceeded = True
                    break
            print(f"[COMMENTS] Comments fetched for this keyword: {comment_count_this_kw}")

        if quota_exceeded:
            print(f"\n[QUOTA EXCEEDED] Stopping collection at keyword {idx+1}: {kw}")
            checkpoint.save(idx, kw)
            run_log_rows.append({
                "run_tag": run_tag,
                "team_member": "dashboard_user",
                "search_keyword": kw,
                "published_after": to_rfc3339(published_after),
                "candidate_video_ids": len(candidate_video_ids),
                "videos_fetched": len(video_df),
                "comments_fetched": comment_count_this_kw,
                "run_started_at_utc": to_rfc3339(utc_now()),
                "logged_at_utc": to_rfc3339(utc_now()),
                "quota_exceeded": True,
            })
            break

        print(f"[SUMMARY] Keyword {idx+1} done — "
              f"videos: {len(video_df)}, comments: {comment_count_this_kw}")

        if progress_callback:
            progress_callback(idx, len(keyword_shard), kw, len(video_df), comment_count_this_kw, "done")

        run_log_rows.append({
            "run_tag": run_tag,
            "team_member": "dashboard_user",
            "search_keyword": kw,
            "published_after": to_rfc3339(published_after),
            "candidate_video_ids": len(candidate_video_ids),
            "videos_fetched": len(video_df),
            "comments_fetched": comment_count_this_kw,
            "run_started_at_utc": to_rfc3339(utc_now()),
            "logged_at_utc": to_rfc3339(utc_now()),
            "quota_exceeded": False,
        })

        checkpoint.save(idx + 1, kw if idx + 1 < len(keyword_shard) else keyword_shard[-1])

    # Always save collected data, regardless of how the loop ended
    videos_all = (
        pd.concat(all_video_frames, ignore_index=True) if all_video_frames else pd.DataFrame()
    )
    comments_all = (
        pd.concat(all_comment_frames, ignore_index=True)
        if all_comment_frames
        else pd.DataFrame()
    )
    runs_log = pd.DataFrame(run_log_rows)

    videos_all = deduplicate_videos(videos_all)
    comments_all = deduplicate_comments(comments_all)

    if not quota_exceeded and not collection_error:
        checkpoint.clear()

    try:
        if not videos_all.empty:
            out_video_file = output_dir / f"videos_{run_tag}.parquet"
            videos_all.to_parquet(out_video_file, index=False)
            print(f"[Saved] {out_video_file}")

        if not comments_all.empty:
            out_comment_file = output_dir / f"comments_{run_tag}.parquet"
            comments_all.to_parquet(out_comment_file, index=False)
            print(f"[Saved] {out_comment_file}")

        if not runs_log.empty:
            out_runs_file = output_dir / f"runs_{run_tag}.parquet"
            runs_log.to_parquet(out_runs_file, index=False)
            print(f"[Saved] {out_runs_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save data: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n===== COLLECTION SUMMARY =====")
    print(f"Videos collected : {0 if videos_all.empty else len(videos_all)}")
    print(f"Comments collected: {0 if comments_all.empty else len(comments_all)}")
    print(f"Run log entries  : {0 if runs_log.empty else len(runs_log)}")
    if quota_exceeded:
        print(f"\n[WARNING] Collection complete (quota exceeded). "
              f"Data saved with what was collected so far.")
    elif collection_error:
        print(f"\n[ERROR] Collection stopped due to error: {collection_error}")
    else:
        print(f"\n[SUCCESS] All keywords processed successfully!")

    return videos_all, comments_all, runs_log, quota_exceeded
