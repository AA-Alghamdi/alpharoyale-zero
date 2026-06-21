#!/usr/bin/env python3
"""Phase B: Annotate pre-extracted frames (no download).

Reads frames from data/frames/<video_id>/ that were prepared by batch_download.py.
Respects configured request limits. Run when provider quota is available.

Usage:
    python3 tools/dataset/batch_annotate.py [--max-frames N] [--model MODEL]
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
from crpipe.extract import _failed_timestep, build_game, segment_games, to_timestep
from crpipe.frames import Frame

PROGRESS_PATH = Path("data/progress.json")
GAMES_DIR = Path("data/games")
FRAMES_DIR = Path("data/frames")
QUEUE_PATH = Path("data/queue.jsonl")

MAX_RPM = 14
RPM_DELAY = 60.0 / MAX_RPM
DAILY_BUDGET = 1400

# Models to rotate through when one hits a provider quota limit.
MODEL_ROTATION = [
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-2.0-flash",
]


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


def load_queue_metadata() -> dict:
    """Load video metadata from queue for channel/url info."""
    meta = {}
    if QUEUE_PATH.exists():
        with QUEUE_PATH.open() as f:
            for line in f:
                d = json.loads(line)
                meta[d["video_id"]] = d
    return meta


def get_frames_for_video(video_id: str) -> list[Frame]:
    """Load pre-extracted frames from disk."""
    frame_dir = FRAMES_DIR / video_id
    if not frame_dir.exists():
        return []
    jpgs = sorted(frame_dir.glob("frame_*.jpg"))
    frames = []
    for i, p in enumerate(jpgs):
        frames.append(Frame(index=i, video_time_s=float(i), path=str(p)))
    return frames


def videos_ready_for_annotation(progress: dict) -> list[str]:
    """Find videos that have been downloaded but not yet annotated."""
    downloaded = set(progress.get("downloaded_video_ids", []))
    processed = set(progress["processed_video_ids"])
    return sorted(downloaded - processed)


def annotate_video(
    video_id: str, providers: list[GeminiFrameAnnotator], budget: int, meta: dict,
    patient: bool = False, batch_size: int = 1,
) -> tuple[int, int, int]:
    """Annotate one video's frames with model rotation.
    Returns (games_found, calls_used, provider_index_to_resume).
    If patient=True, waits and retries on 429 instead of giving up.
    If batch_size>1, sends multiple frames per API call (saves RPD quota)."""
    frames = get_frames_for_video(video_id)
    if not frames:
        print(f"  [skip] No frames found for {video_id}")
        return 0, 0, 0

    # In batch mode, budget counts API calls not frames
    effective_budget = budget * batch_size if batch_size > 1 else budget
    if len(frames) > effective_budget:
        print(f"  [cap] {len(frames)} -> {effective_budget} frames (budget limit)")
        frames = frames[:effective_budget]

    mode_str = f" batch={batch_size}" if batch_size > 1 else ""
    mode_str += " patient" if patient else ""
    print(f"  [annotate] Annotating {len(frames)} frames ({mode_str.strip()})...")
    timesteps = []
    context = ""
    calls_used = 0
    pi = 0  # current provider index
    consecutive_429 = 0
    max_patient_wait = 600

    i = 0
    while i < len(frames):
        # Determine chunk size
        chunk_end = min(i + batch_size, len(frames))
        chunk_frames = frames[i:chunk_end]
        start = time.time()

        try:
            if batch_size > 1 and len(chunk_frames) > 1:
                raws = providers[pi].extract_states_batch(
                    [f.path for f in chunk_frames], context=context
                )
            else:
                raws = [providers[pi].extract_state(chunk_frames[0].path, context=context)]
            calls_used += 1
            consecutive_429 = 0
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                if patient:
                    consecutive_429 += 1
                    wait = min(30 * consecutive_429, max_patient_wait)
                    if consecutive_429 <= 20:
                        print(f"  [annotate] 429 at frame {i}/{len(frames)}, waiting {wait}s (attempt {consecutive_429})...")
                        time.sleep(wait)
                        pi = (pi + 1) % len(providers)
                        continue
                    else:
                        print(f"  [annotate] Exhausted patience after {consecutive_429} retries")
                        break
                else:
                    pi += 1
                    if pi >= len(providers):
                        print(f"  [annotate] ALL MODELS EXHAUSTED at frame {i}/{len(frames)}")
                        break
                    print(f"  [annotate] Model exhausted, switching to {providers[pi].model}")
                    continue
            print(f"  [annotate] Frame {i} error: {str(e)[:60]}")
            for fr in chunk_frames:
                timesteps.append(_failed_timestep(fr))
            calls_used += 1
            i += len(chunk_frames)
            continue

        # Process results
        for j, (raw, fr) in enumerate(zip(raws, chunk_frames)):
            ts = to_timestep(raw, fr)
            timesteps.append(ts)
            hand = raw.get("visible_cards_in_hand")
            if hand:
                context = f"prev hand={hand} prev elixir={ts.elixir_player}"

        elapsed = time.time() - start
        sleep_needed = RPM_DELAY - elapsed
        if sleep_needed > 0 and i + len(chunk_frames) < len(frames):
            time.sleep(sleep_needed)

        i += len(chunk_frames)
        if i % 25 < batch_size or i >= len(frames):
            print(f"  [annotate] {i}/{len(frames)} done (model={providers[pi].model}, calls={calls_used})...")

    if len(timesteps) < 3:
        return 0, calls_used, pi

    # Segment and save
    games = segment_games(timesteps)
    print(f"  [segment] {len(games)} game(s) from {len(timesteps)} timesteps")

    info = meta.get(video_id, {})
    model_name = providers[min(pi, len(providers)-1)].model
    games_saved = 0
    for gi, g in enumerate(games):
        record = build_game(
            g,
            video_id=video_id,
            video_url=info.get("url", f"https://www.youtube.com/watch?v={video_id}"),
            game_index=gi,
            channel=info.get("channel"),
            annotation_model=model_name,
            sample_fps=1.0,
        )
        out_path = GAMES_DIR / f"{video_id}_game{gi:02d}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(record.model_dump_json(indent=2))
        games_saved += 1
        print(f"  [save] {out_path.name} (conf={record.mean_confidence})")

    return games_saved, calls_used, pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=DAILY_BUDGET)
    parser.add_argument("--max-videos", type=int, default=50)
    parser.add_argument("--models", default=",".join(MODEL_ROTATION),
                        help="Comma-separated models to rotate through on 429")
    parser.add_argument("--patient", action="store_true",
                        help="Wait and retry on 429 instead of stopping (for quota trickle)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Frames per API call (>1 saves RPD quota, e.g. 5)")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    print("=== CR Dataset Batch Annotator ===")
    print(f"Models: {model_names}")
    print(f"Budget: {args.max_frames} calls")
    print(f"Started: {datetime.now().isoformat()}")

    progress = load_progress()
    meta = load_queue_metadata()

    ready = videos_ready_for_annotation(progress)
    print(f"Videos ready for annotation: {len(ready)}")

    if not ready:
        print("Nothing to annotate. Run batch_download.py first.")
        return

    providers = [GeminiFrameAnnotator(model=m, max_retries=3, timeout=90.0) for m in model_names]
    budget = args.max_frames
    total_games = 0
    total_calls = 0
    current_pi = 0  # track which provider to start with

    for vid in ready[:args.max_videos]:
        if budget <= 30:
            print(f"\n[BUDGET EXHAUSTED] Only {budget} calls left.")
            break

        print(f"\n{'='*50}")
        info = meta.get(vid, {})
        print(f"Annotating: {info.get('title', vid)[:60]}")

        # Start from the last-working provider
        active_providers = providers[current_pi:]
        if not active_providers:
            print("  [stop] All models exhausted.")
            break

        games, calls, pi_offset = annotate_video(vid, active_providers, budget, meta,
                                                     patient=args.patient,
                                                     batch_size=args.batch_size)
        current_pi += pi_offset
        budget -= calls
        total_calls += calls
        total_games += games

        if games > 0:
            progress["processed_video_ids"].append(vid)
            progress["total_games_extracted"] += games
        elif calls > 0:
            progress["processed_video_ids"].append(vid)
        progress["total_frames_annotated"] += calls
        save_progress(progress)

        if current_pi >= len(providers):
            print("  [stop] All models exhausted across rotation.")
            break

    print(f"\n{'='*50}")
    print("=== Annotation Session Complete ===")
    print(f"Games extracted: {total_games}")
    print(f"Annotation calls used: {total_calls}")
    print(f"Budget remaining: {budget}")
    print(f"Total games (all time): {progress['total_games_extracted']}")
    print(f"Finished: {datetime.now().isoformat()}")

    progress["session_log"].append({
        "timestamp": datetime.now().isoformat(),
        "phase": "annotate",
        "models": model_names,
        "games": total_games,
        "calls": total_calls,
    })
    save_progress(progress)


if __name__ == "__main__":
    main()
