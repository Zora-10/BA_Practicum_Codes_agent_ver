"""
Step 2: YouTube Data Collection
Collects videos and comments via YouTube Data API v3.
Fidelity: mirrors 02_data_collection.ipynb exactly.
"""

import json
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from .config import BASE_DIR


class QuotaExceededException(Exception):
    pass


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

    def load(self) -> int:
        if not self.file.exists():
            return 0
        try:
            with open(self.file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("keyword_index", 0)
        except (json.JSONDecodeError, KeyError):
            return 0

    def clear(self):
        if self.file.exists():
            self.file.unlink()


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

    for _ in range(max_pages):
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
            response = request.execute()
        except HttpError as e:
            if "quotaExceeded" in str(e):
                raise QuotaExceededException(f"Quota exceeded for query={query}")
            print(f"[ERROR] search failed for query={query}: {e}")
            break

        items = response.get("items", [])
        page_ids = [
            item["id"]["videoId"] for item in items if item.get("id", {}).get("videoId")
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
            response = request.execute()
        except HttpError as e:
            if "quotaExceeded" in str(e):
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
            response = request.execute()
        except HttpError as e:
            if "quotaExceeded" in str(e):
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
    """
    Run YouTube data collection.

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

    youtube = build("youtube", "v3", developerKey=api_key)

    all_video_frames = []
    all_comment_frames = []
    run_log_rows = []
    quota_exceeded = False

    for idx, kw in enumerate(keyword_shard):
        if idx < start_from_index:
            continue

        if progress_callback:
            progress_callback(idx, len(keyword_shard), kw)

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

            video_df = fetch_video_details(
                youtube_client=youtube,
                video_ids=candidate_video_ids,
                keyword_used=kw,
            )
            video_df = deduplicate_videos(video_df)

            all_video_frames.append(video_df)

            comment_count_this_kw = 0
            if fetch_comments and not video_df.empty:
                for vid in video_df["video_id"].tolist():
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
                    except QuotaExceededException:
                        quota_exceeded = True
                        raise

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

        except QuotaExceededException:
            quota_exceeded = True
            checkpoint.save(idx, kw)
            run_log_rows.append({
                "run_tag": run_tag,
                "team_member": "dashboard_user",
                "search_keyword": kw,
                "published_after": to_rfc3339(published_after),
                "candidate_video_ids": 0,
                "videos_fetched": 0,
                "comments_fetched": 0,
                "run_started_at_utc": to_rfc3339(utc_now()),
                "logged_at_utc": to_rfc3339(utc_now()),
                "quota_exceeded": True,
            })
            break

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

    if not quota_exceeded:
        checkpoint.clear()

    # Save outputs
    if not videos_all.empty:
        videos_all.to_parquet(output_dir / f"videos_{run_tag}.parquet", index=False)
    if not comments_all.empty:
        comments_all.to_parquet(output_dir / f"comments_{run_tag}.parquet", index=False)
    if not runs_log.empty:
        runs_log.to_parquet(output_dir / f"runs_{run_tag}.parquet", index=False)

    return videos_all, comments_all, runs_log, quota_exceeded
