#!/usr/bin/env python3
"""Batch process CR gameplay videos: download -> frames -> annotate -> GameRecord.

Designed for the free Gemini tier (~1500 calls/day, 15 calls/min).
Saves progress so future sessions resume where we left off.

Usage:
    python3 tools/dataset/batch_process.py [--max-videos N] [--max-frames-total N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crpipe.annotators import GeminiFrameAnnotator
from crpipe.discover import VideoCandidate
from crpipe.download import DownloadConfig, download_video
from crpipe.extract import _failed_timestep, build_game, segment_games, to_timestep
from crpipe.frames import Frame, sample_uniform

# --------------- Config ---------------
QUEUE_PATH = Path("data/queue.jsonl")
PROGRESS_PATH = Path("data/progress.json")
GAMES_DIR = Path("data/games")
VIDEOS_DIR = Path("data/raw_videos")
FRAMES_DIR = Path("data/frames")

# Default rate limits for the remote frame-annotation tier.
MAX_RPM = 14  # stay below 15 RPM limit
RPM_DELAY = 60.0 / MAX_RPM  # ~4.3s between calls
DAILY_BUDGET = 1400  # stay below 1500/day for safety margin


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {
        "processed_video_ids": [],
        "failed_video_ids": [],
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
            d = json.loads(line)
            candidates.append(VideoCandidate(**d))
    return candidates


def pick_next_videos(queue: list[VideoCandidate], progress: dict, max_n: int) -> list[VideoCandidate]:
    done = set(progress["processed_video_ids"]) | set(progress["failed_video_ids"])
    remaining = [c for c in queue if c.video_id not in done]
    # Prefer shorter videos first (more games per model dollar)
    remaining.sort(key=lambda c: c.duration or 9999)
    return remaining[:max_n]


def download_one(candidate: VideoCandidate) -> Path | None:
    cfg = DownloadConfig(
        out_dir=str(VIDEOS_DIR),
        cookies_from_browser=f"chrome:{Path.home() / '.browser_data_dir'}",
    )
    try:
        path = download_video(candidate.url, candidate.video_id, cfg)
        if path and path.exists() and path.stat().st_size > 100_000:
            print(f"  [dl] OK: {path.name} ({path.stat().st_size // 1024}KB)")
            return path
        else:
            print("  [dl] FAILED: file too small or missing")
            return None
    except Exception as e:
        print(f"  [dl] ERROR: {e}")
        return None


def extract_one(video_path: Path, video_id: str, fps: float = 1.0) -> list[Frame]:
    frame_dir = FRAMES_DIR / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = sample_uniform(video_path, frame_dir, fps=fps)
    print(f"  [frames] Extracted {len(frames)} frames at {fps} fps")
    return frames


def annotate_frames_ratelimited(
    frames: list[Frame], provider: GeminiFrameAnnotator, budget_remaining: int
) -> tuple[list, int]:
    """Annotate frames while respecting rate limits.
    Returns (timesteps, calls_used). Stops on quota exhaustion."""
    if len(frames) > budget_remaining:
        print(f"  [annotate] Capping frames from {len(frames)} to {budget_remaining} (daily budget)")
        frames = frames[:budget_remaining]

    calls_used = 0
    timesteps = []
    context = ""

    for i, fr in enumerate(frames):
        start = time.time()
        try:
            raw = provider.extract_state(fr.path, context=context)
            calls_used += 1
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                print(f"  [annotate] QUOTA HIT at frame {i}/{len(frames)}: {e}")
                break
            print(f"  [annotate] Frame {i} failed: {e}")
            timesteps.append(_failed_timestep(fr))
            calls_used += 1
            continue

        ts = to_timestep(raw, fr)
        timesteps.append(ts)

        hand = raw.get("visible_cards_in_hand")
        if hand:
            context = f"prev hand={hand} prev elixir={ts.elixir_player}"

        # Rate-limit sleep
        elapsed = time.time() - start
        sleep_time = RPM_DELAY - elapsed
        if sleep_time > 0 and i < len(frames) - 1:
            time.sleep(sleep_time)

        if (i + 1) % 20 == 0:
            print(f"  [annotate] {i+1}/{len(frames)} frames done...")

    return timesteps, calls_used


def process_video(
    candidate: VideoCandidate, provider: GeminiFrameAnnotator, budget_remaining: int
) -> tuple[list, int]:
    """Full pipeline for one video. Returns (game_records_saved, calls_used)."""
    print(f"\n{'='*60}")
    print(f"Processing: {candidate.title}")
    print(f"  ID={candidate.video_id} dur={candidate.duration}s channel={candidate.channel}")

    # 1. Download
    video_path = download_one(candidate)
    if not video_path:
        return [], 0

    # 2. Extract frames at 1fps
    frames = extract_one(video_path, candidate.video_id, fps=1.0)
    if not frames:
        print("  [frames] No frames extracted, skipping")
        return [], 0

    # 3. Annotate with model (rate-limited)
    print(f"  [annotate] Annotating {len(frames)} frames (budget: {budget_remaining})...")
    timesteps, calls_used = annotate_frames_ratelimited(frames, provider, budget_remaining)
    print(f"  [annotate] Got {len(timesteps)} timesteps, used {calls_used} API calls")

    if len(timesteps) < 3:
        print("  [skip] Too few timesteps for a game")
        # Clean up video to save disk
        try:
            video_path.unlink()
        except Exception:
            pass
        return [], calls_used

    # 4. Segment into games & build records
    games = segment_games(timesteps)
    print(f"  [segment] Found {len(games)} game(s)")

    records_saved = []
    for i, g in enumerate(games):
        record = build_game(
            g,
            video_id=candidate.video_id,
            video_url=candidate.url,
            game_index=i,
            channel=candidate.channel,
            annotation_model=provider.model,
            sample_fps=1.0,
        )
        out_path = GAMES_DIR / f"{candidate.video_id}_game{i:02d}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(record.model_dump_json(indent=2))
        records_saved.append(str(out_path))
        print(f"  [save] {out_path.name} ({len(g)} timesteps, conf={record.mean_confidence})")

    # 5. Clean up video file to save disk space (keep frames)
    try:
        video_path.unlink()
        print("  [cleanup] Removed video file")
    except Exception:
        pass

    return records_saved, calls_used


def main():
    parser = argparse.ArgumentParser(description="Batch process CR gameplay videos")
    parser.add_argument("--max-videos", type=int, default=50, help="Max videos to attempt")
    parser.add_argument("--max-frames-total", type=int, default=DAILY_BUDGET, help="Max annotation calls")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model")
    args = parser.parse_args()

    print("=== CR Dataset Batch Processor ===")
    print(f"Model: {args.model}")
    print(f"Budget: {args.max_frames_total} annotation calls, {args.max_videos} videos")
    print(f"Started: {datetime.now().isoformat()}")

    progress = load_progress()
    queue = load_queue()
    print(f"Queue: {len(queue)} candidates")
    print(f"Already processed: {len(progress['processed_video_ids'])} videos")
    print(f"Already failed: {len(progress['failed_video_ids'])} videos")
    print(f"Total games so far: {progress['total_games_extracted']}")

    batch = pick_next_videos(queue, progress, args.max_videos)
    if not batch:
        print("No videos left to process!")
        return

    print(f"Selected {len(batch)} videos for this run (shortest first)")

    provider = GeminiFrameAnnotator(model=args.model, max_retries=4, timeout=90.0)

    budget_remaining = args.max_frames_total
    session_games = 0
    session_calls = 0
    session_videos_ok = 0

    for candidate in batch:
        if budget_remaining <= 30:
            print(f"\n[BUDGET] Only {budget_remaining} calls remaining, stopping.")
            break

        records, calls_used = process_video(candidate, provider, budget_remaining)
        budget_remaining -= calls_used
        session_calls += calls_used

        if records:
            progress["processed_video_ids"].append(candidate.video_id)
            progress["total_games_extracted"] += len(records)
            session_games += len(records)
            session_videos_ok += 1
        else:
            if calls_used == 0:
                # Download failed, not an annotation issue
                progress["failed_video_ids"].append(candidate.video_id)
            else:
                # Annotation ran but got too few frames; still mark processed.
                progress["processed_video_ids"].append(candidate.video_id)

        progress["total_frames_annotated"] += calls_used
        save_progress(progress)

        time.sleep(2)  # brief pause between videos

    # Final report
    print(f"\n{'='*60}")
    print("=== Session Complete ===")
    print(f"Videos successful: {session_videos_ok}")
    print(f"Games extracted this session: {session_games}")
    print(f"Annotation calls this session: {session_calls}")
    print(f"Total games (all sessions): {progress['total_games_extracted']}")
    print(f"Total frames (all sessions): {progress['total_frames_annotated']}")
    print(f"Budget remaining: {budget_remaining}")
    print(f"Finished: {datetime.now().isoformat()}")

    progress["session_log"].append({
        "timestamp": datetime.now().isoformat(),
        "videos_ok": session_videos_ok,
        "games": session_games,
        "calls_used": session_calls,
        "budget_remaining": budget_remaining,
    })
    save_progress(progress)


if __name__ == "__main__":
    main()
