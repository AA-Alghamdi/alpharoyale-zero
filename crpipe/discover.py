"""Discover candidate Clash Royale gameplay videos via yt-dlp search.

Discovery uses yt-dlp's ``ytsearch`` (and optional channel crawls), which only
reads public metadata and does not download media. Results are filtered by
upload date (default: last 2 years), de-duplicated, and written to a JSONL queue
that the download/extraction stages consume.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import yt_dlp

DEFAULT_QUERIES = [
    "clash royale ladder gameplay",
    "clash royale path of legends gameplay",
    "clash royale top ladder",
    "clash royale pro gameplay",
    "clash royale 1v1 match",
    "clash royale grand challenge gameplay",
    "clash royale tournament match",
    "clash royale meta deck gameplay",
]


@dataclass
class VideoCandidate:
    video_id: str
    url: str
    title: str | None = None
    channel: str | None = None
    channel_id: str | None = None
    duration: float | None = None
    upload_date: str | None = None  # YYYYMMDD
    view_count: int | None = None


def _within_window(upload_date: str | None, since: dt.date | None) -> bool:
    if since is None or not upload_date:
        return True
    try:
        d = dt.datetime.strptime(upload_date, "%Y%m%d").date()
    except ValueError:
        return True  # keep if unparseable; filter later
    return d >= since


def search(
    queries: Iterable[str] = DEFAULT_QUERIES,
    per_query: int = 50,
    since: dt.date | None = None,
    min_duration_s: float = 120.0,
    max_duration_s: float = 7200.0,
) -> list[VideoCandidate]:
    """Return de-duplicated candidates across all queries."""
    opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "ignoreerrors": True,
    }
    seen: dict[str, VideoCandidate] = {}
    with yt_dlp.YoutubeDL(opts) as ydl:
        for q in queries:
            info = ydl.extract_info(f"ytsearch{per_query}:{q}", download=False)
            for entry in (info or {}).get("entries", []) or []:
                if not entry:
                    continue
                vid = entry.get("id")
                if not vid or vid in seen:
                    continue
                dur = entry.get("duration")
                if dur is not None and not (min_duration_s <= dur <= max_duration_s):
                    continue
                cand = VideoCandidate(
                    video_id=vid,
                    url=entry.get("url") or f"https://www.youtube.com/watch?v={vid}",
                    title=entry.get("title"),
                    channel=entry.get("channel") or entry.get("uploader"),
                    channel_id=entry.get("channel_id"),
                    duration=dur,
                    upload_date=entry.get("upload_date"),
                    view_count=entry.get("view_count"),
                )
                seen[vid] = cand
    return [c for c in seen.values() if _within_window(c.upload_date, since)]


def write_queue(candidates: list[VideoCandidate], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for c in candidates:
            fh.write(json.dumps(asdict(c)) + "\n")
    return len(candidates)


def two_years_ago(today: dt.date | None = None) -> dt.date:
    today = today or dt.date.today()
    try:
        return today.replace(year=today.year - 2)
    except ValueError:  # Feb 29
        return today.replace(year=today.year - 2, day=28)
