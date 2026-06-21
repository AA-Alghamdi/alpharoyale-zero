#!/usr/bin/env python3
"""End-to-end smoke test with NO network and NO API key.

Generates a short synthetic video with ffmpeg, runs the full
frames -> mock frame annotation -> segment -> GameRecord path, and validates
the output against the schema. This exercises every stage except YouTube
download and a remote annotation provider.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from crpipe.pipeline import PipelineConfig, process_local_video  # noqa: E402
from crpipe.schema import GameRecord  # noqa: E402


def make_synthetic_video(path: Path, seconds: int = 12) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", f"testsrc=size=640x360:rate=30:duration={seconds}",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        video = tmp / "synthetic.mp4"
        make_synthetic_video(video, seconds=12)

        cfg = PipelineConfig(
            sample_fps=1.0,
            frames_dir=str(tmp / "frames"),
            games_dir=str(tmp / "games"),
            annotator_provider="mock",
        )
        records = process_local_video(video, "smoke", "local://smoke", cfg)

        assert records, "no games produced"
        for rec in records:
            # round-trips through the schema validator
            GameRecord.model_validate(rec.model_dump())
            assert rec.timeline, "empty timeline"
            assert rec.schema_version

        total_steps = sum(len(r.timeline) for r in records)
        total_actions = sum(len(t.actions) for r in records for t in r.timeline)
        print(f"OK: {len(records)} game(s), {total_steps} timesteps, "
              f"{total_actions} actions extracted")
        print("sample game (truncated):")
        sample = records[0].model_dump()
        sample["timeline"] = sample["timeline"][:2]
        import json
        print(json.dumps(sample, indent=2)[:1800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
