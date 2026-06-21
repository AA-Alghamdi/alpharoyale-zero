#!/usr/bin/env python3
"""Phase A: Download videos and extract frames (no model calls).

This can run unlimited — only constrained by YouTube rate limits and disk space.
Produces frames ready for annotation by batch_annotate.py when model quota is available.

Usage:
    python3 tools/dataset/batch_download.py [--max-videos N]
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crpipe.discover import VideoCandidate
from crpipe.download import DownloadConfig, download_video
from crpipe.frames import sample_uniform

QUEUE_PATH = Path("data/queue.jsonl")
PROGRESS_PATH = Path("data/progress.json")
VIDEOS_DIR = Path("data/raw_videos")
FRAMES_DIR = Path("data/frames")


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {
        "processed_video_ids": [],
        "failed_video_ids": [],
        "downloaded_video_ids": [],
        "total_frames_annotated": 0,
        "total_games_extracted": 0,
        "session_log": [],
    }


def save_progress(prog: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(prog, indent=2))


def load_queue() -> list[VideoCandidate]:
    candidates = []
    with QUEUE_PATH.open() as f:
        for line in f:
            candidates.append(VideoCandidate(**json.loads(line)))
    return candidates


def pick_next(queue: list[VideoCandidate], progress: dict, max_n: int) -> list[VideoCandidate]:
    done = set(progress.get("downloaded_video_ids", []))
    done |= set(progress["processed_video_ids"])
    done |= set(progress["failed_video_ids"])
    remaining = [c for c in queue if c.video_id not in done]
    remaining.sort(key=lambda c: c.duration or 9999)
    return remaining[:max_n]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--fps", type=float, default=1.0)
    args = parser.parse_args()

    print("=== CR Dataset Batch Downloader ===")
    print(f"Started: {datetime.now().isoformat()}")

    progress = load_progress()
    if "downloaded_video_ids" not in progress:
        progress["downloaded_video_ids"] = []

    queue = load_queue()
    batch = pick_next(queue, progress, args.max_videos)
    print(f"Queue: {len(queue)} | Already downloaded: {len(progress['downloaded_video_ids'])} | Batch: {len(batch)}")

    cfg = DownloadConfig(
        out_dir=str(VIDEOS_DIR),
        cookies_from_browser=f"chrome:{Path.home() / '.browser_data_dir'}",
    )

    ok = 0
    fail = 0
    for i, cand in enumerate(batch):
        print(f"\n[{i+1}/{len(batch)}] {cand.title[:60]}  (dur={cand.duration}s)")

        # Check if frames already exist
        frame_dir = FRAMES_DIR / cand.video_id
        existing_frames = list(frame_dir.glob("frame_*.jpg")) if frame_dir.exists() else []
        if len(existing_frames) > 10:
            print(f"  [skip] Already has {len(existing_frames)} frames")
            progress["downloaded_video_ids"].append(cand.video_id)
            save_progress(progress)
            ok += 1
            continue

        # Download
        try:
            path = download_video(cand.url, cand.video_id, cfg)
            if not path or not path.exists() or path.stat().st_size < 100_000:
                print("  [dl] FAILED (too small or missing)")
                progress["failed_video_ids"].append(cand.video_id)
                save_progress(progress)
                fail += 1
                continue
            print(f"  [dl] OK: {path.name} ({path.stat().st_size // 1024}KB)")
        except Exception as e:
            print(f"  [dl] ERROR: {e}")
            progress["failed_video_ids"].append(cand.video_id)
            save_progress(progress)
            fail += 1
            time.sleep(3)
            continue

        # Extract frames
        frame_dir.mkdir(parents=True, exist_ok=True)
        try:
            frames = sample_uniform(path, frame_dir, fps=args.fps)
            print(f"  [frames] {len(frames)} frames at {args.fps} fps")
        except Exception as e:
            print(f"  [frames] ERROR: {e}")
            progress["failed_video_ids"].append(cand.video_id)
            save_progress(progress)
            fail += 1
            continue

        # Clean up video (keep frames)
        try:
            path.unlink()
        except Exception:
            pass

        progress["downloaded_video_ids"].append(cand.video_id)
        save_progress(progress)
        ok += 1

        # Brief pause between downloads
        time.sleep(2)

    print(f"\n{'='*60}")
    print(f"Done: {ok} downloaded, {fail} failed")
    print(f"Total ready for annotation: {len(progress['downloaded_video_ids'])}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
