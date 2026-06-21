"""Frame sampling from a local video using ffmpeg.

Two modes:
  - uniform: sample at a fixed fps (e.g. 1 frame/sec)
  - scene:   sample on scene-change cuts (good for catching card placements)
The two can be combined; frames are de-duplicated by timestamp.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Frame:
    index: int
    video_time_s: float
    path: str


def video_duration_s(video_path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def sample_uniform(
    video_path: str | Path,
    out_dir: str | Path,
    fps: float = 1.0,
    scale_width: int = 640,
    jpeg_quality: int = 3,
) -> list[Frame]:
    """Extract frames at ``fps`` frames per second. Returns frame metadata."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
         "-vf", f"fps={fps},scale={scale_width}:-2",
         "-q:v", str(jpeg_quality), pattern],
        check=True,
    )
    frames = []
    for i, p in enumerate(sorted(out_dir.glob("frame_*.jpg"))):
        frames.append(Frame(index=i, video_time_s=i / fps, path=str(p)))
    return frames


def sample_scene_cuts(
    video_path: str | Path,
    out_dir: str | Path,
    threshold: float = 0.3,
    scale_width: int = 640,
    jpeg_quality: int = 3,
) -> list[Frame]:
    """Extract frames at scene changes; timestamps come from showinfo metadata."""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "scene_%06d.jpg")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "info", "-i", str(video_path),
         "-vf", f"select='gt(scene,{threshold})',scale={scale_width}:-2,showinfo",
         "-vsync", "vfr", "-q:v", str(jpeg_quality), pattern],
        capture_output=True, text=True,
    )
    times: list[float] = []
    for line in proc.stderr.splitlines():
        if "pts_time:" in line:
            try:
                times.append(float(line.split("pts_time:")[1].split()[0]))
            except (IndexError, ValueError):
                pass
    frames = []
    for i, p in enumerate(sorted(out_dir.glob("scene_*.jpg"))):
        t = times[i] if i < len(times) else float(i)
        frames.append(Frame(index=i, video_time_s=t, path=str(p)))
    return frames
