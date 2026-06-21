#!/usr/bin/env python3
"""CLI for the Clash Royale dataset pipeline.

Examples
--------
# 1) Discover candidate videos (metadata only) from the last 2 years -> queue.jsonl
python tools/dataset/run.py discover --per-query 50 --out data/queue.jsonl

# 2) Process a queue. Needs working YouTube access and a configured frame annotator.
python tools/dataset/run.py run --queue data/queue.jsonl --provider openai --model gpt-4o-mini \
    --fps 1 --cookies-from-browser chrome:/home/ubuntu/.browser_data_dir

# 3) Process a single already-downloaded video file (no YouTube needed).
python tools/dataset/run.py process-local --file game.mp4 --video-id local1 --provider mock

# 4) Dump the dataset JSON Schema.
python tools/dataset/run.py schema
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crpipe.discover import search, two_years_ago, write_queue  # noqa: E402
from crpipe.download import DownloadConfig  # noqa: E402
from crpipe.pipeline import PipelineConfig, process_local_video, process_video  # noqa: E402
from crpipe.schema import GameRecord  # noqa: E402


def cmd_discover(args: argparse.Namespace) -> None:
    since = None if args.all_time else two_years_ago()
    cands = search(per_query=args.per_query, since=since)
    n = write_queue(cands, args.out)
    print(f"discovered {n} candidate videos (since={since}) -> {args.out}")


def _pipeline_cfg(args: argparse.Namespace) -> PipelineConfig:
    dl = DownloadConfig(
        max_height=args.max_height,
        cookiefile=args.cookiefile,
        cookies_from_browser=args.cookies_from_browser,
        proxy=args.proxy,
    )
    return PipelineConfig(
        sample_fps=args.fps, annotator_provider=args.provider,
        annotation_model=args.model, download=dl,
    )


def cmd_run(args: argparse.Namespace) -> None:
    cfg = _pipeline_cfg(args)
    total = 0
    with open(args.queue) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            recs = process_video(
                c["video_id"], c["url"], cfg,
                channel=c.get("channel"), upload_date=c.get("upload_date"),
            )
            total += len(recs)
    print(f"extracted {total} games total")


def cmd_process_local(args: argparse.Namespace) -> None:
    cfg = _pipeline_cfg(args)
    recs = process_local_video(
        args.file, args.video_id,
        args.url or f"local://{args.video_id}", cfg,
    )
    print(f"extracted {len(recs)} games from {args.file}")


def cmd_schema(args: argparse.Namespace) -> None:
    print(json.dumps(GameRecord.model_json_schema(), indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Clash Royale dataset pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="find candidate videos (metadata only)")
    d.add_argument("--per-query", type=int, default=50)
    d.add_argument("--all-time", action="store_true", help="disable 2-year window")
    d.add_argument("--out", default="data/queue.jsonl")
    d.set_defaults(func=cmd_discover)

    def add_common(sp):
        sp.add_argument("--provider", default="mock",
                        choices=["mock", "openai", "anthropic", "gemini"])
        sp.add_argument("--model", default=None)
        sp.add_argument("--fps", type=float, default=1.0)
        sp.add_argument("--max-height", type=int, default=480)
        sp.add_argument("--cookiefile", default=None)
        sp.add_argument("--cookies-from-browser", default=None)
        sp.add_argument("--proxy", default=None)

    r = sub.add_parser("run", help="download + extract a queue of videos")
    r.add_argument("--queue", required=True)
    add_common(r)
    r.set_defaults(func=cmd_run)

    pl = sub.add_parser("process-local", help="extract from a local video file")
    pl.add_argument("--file", required=True)
    pl.add_argument("--video-id", required=True)
    pl.add_argument("--url", default=None)
    add_common(pl)
    pl.set_defaults(func=cmd_process_local)

    s = sub.add_parser("schema", help="print the dataset JSON Schema")
    s.set_defaults(func=cmd_schema)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
