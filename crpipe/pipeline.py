"""End-to-end pipeline for a single video: download -> sample frames ->
annotate -> segment into games -> write GameRecords as JSON."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .annotators import FrameAnnotator, get_provider
from .download import DownloadConfig, download_video
from .extract import build_game, extract_timesteps, segment_games
from .frames import Frame, sample_uniform
from .schema import GameRecord


@dataclass
class PipelineConfig:
    sample_fps: float = 1.0
    frames_dir: str = "data/frames"
    games_dir: str = "data/games"
    scale_width: int = 640
    annotator_provider: str = "mock"
    annotation_model: str | None = None
    download: DownloadConfig = None  # type: ignore

    def __post_init__(self):
        if self.download is None:
            self.download = DownloadConfig()


def _provider(cfg: PipelineConfig) -> FrameAnnotator:
    kwargs = {"model": cfg.annotation_model} if cfg.annotation_model else {}
    return get_provider(cfg.annotator_provider, **kwargs)


def process_local_video(
    video_path: str | Path, video_id: str, video_url: str, cfg: PipelineConfig,
    channel: str | None = None, upload_date: str | None = None,
) -> list[GameRecord]:
    """Run frames -> annotations -> games on an already-downloaded video."""
    frame_dir = Path(cfg.frames_dir) / video_id
    frames: list[Frame] = sample_uniform(
        video_path, frame_dir, fps=cfg.sample_fps, scale_width=cfg.scale_width
    )
    provider = _provider(cfg)
    steps = extract_timesteps(frames, provider)
    games = segment_games(steps)
    records = [
        build_game(
            g, video_id=video_id, video_url=video_url, game_index=i,
            channel=channel, upload_date=upload_date,
            annotation_model=cfg.annotation_model or provider.name, sample_fps=cfg.sample_fps,
        )
        for i, g in enumerate(games)
    ]
    _write_records(records, video_id, cfg)
    return records


def process_video(
    video_id: str, video_url: str, cfg: PipelineConfig,
    channel: str | None = None, upload_date: str | None = None,
) -> list[GameRecord]:
    """Full pipeline including download. Returns [] if download is blocked."""
    path = download_video(video_url, video_id, cfg.download)
    if path is None:
        print(f"[pipeline] download blocked/failed for {video_id}")
        return []
    return process_local_video(path, video_id, video_url, cfg, channel, upload_date)


def _write_records(records: list[GameRecord], video_id: str, cfg: PipelineConfig) -> None:
    out_dir = Path(cfg.games_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.jsonl"
    with out_path.open("w") as fh:
        for rec in records:
            fh.write(rec.model_dump_json() + "\n")
    print(f"[pipeline] wrote {len(records)} game(s) -> {out_path}")
