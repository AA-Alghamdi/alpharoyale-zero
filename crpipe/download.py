"""Download gameplay video (or sections) via yt-dlp with cookie/proxy support.

NOTE ON ACCESS: YouTube actively bot-walls automated downloads from datacenter
IPs ("Sign in to confirm you're not a bot"). Reliable bulk download requires
authenticated cookies (``cookiefile`` / ``cookies_from_browser``) and almost
always a residential proxy pool. Both are configurable here and via config.yaml.
Respect YouTube's Terms of Service and applicable law when operating this stage.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yt_dlp


def _ensure_js_runtime_on_path() -> None:
    """Make a locally-installed Deno visible to yt-dlp's JS challenge solver.

    YouTube requires solving an ``n`` signature challenge to obtain real video
    formats; yt-dlp delegates this to a JS runtime (Deno recommended). A Deno
    installed at ~/.deno/bin won't be on PATH for a non-login shell, so add it.
    """
    deno_bin = Path.home() / ".deno" / "bin"
    if deno_bin.is_dir() and str(deno_bin) not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = str(deno_bin) + os.pathsep + os.environ.get("PATH", "")


@dataclass
class DownloadConfig:
    out_dir: str = "data/raw_videos"
    max_height: int = 480
    cookiefile: str | None = None
    cookies_from_browser: str | None = None  # e.g. "chrome:/path/to/profile"
    proxy: str | None = None
    # Empty => let yt-dlp pick its default player clients (most robust).
    player_clients: tuple[str, ...] = ()
    # EJS challenge-solver scripts needed to solve the YouTube `n` signature.
    remote_components: tuple[str, ...] = ("ejs:github",)
    section: str | None = None  # e.g. "*00:00-05:00"; None => full video
    sleep_interval: float = 1.0
    max_sleep_interval: float = 5.0
    retries: int = 5


def _ydl_opts(cfg: DownloadConfig, outtmpl: str) -> dict:
    opts: dict = {
        "quiet": True,
        "noprogress": True,
        "outtmpl": outtmpl,
        "format": f"bv*[height<={cfg.max_height}]+ba/b[height<={cfg.max_height}]/b",
        "retries": cfg.retries,
        "fragment_retries": cfg.retries,
        "sleep_interval": cfg.sleep_interval,
        "max_sleep_interval": cfg.max_sleep_interval,
    }
    if cfg.player_clients:
        opts["extractor_args"] = {
            "youtube": {"player_client": list(cfg.player_clients)}
        }
    if cfg.remote_components:
        opts["remote_components"] = list(cfg.remote_components)
    if cfg.cookiefile:
        opts["cookiefile"] = cfg.cookiefile
    if cfg.cookies_from_browser:
        browser, _, profile = cfg.cookies_from_browser.partition(":")
        opts["cookiesfrombrowser"] = (browser, profile or None, None, None)
    if cfg.proxy:
        opts["proxy"] = cfg.proxy
    if cfg.section:
        from yt_dlp.utils import download_range_func

        # parse "*HH:MM-HH:MM" style ranges
        opts["download_ranges"] = download_range_func(
            None, _parse_sections(cfg.section)
        )
        opts["force_keyframes_at_cuts"] = True
    return opts


def _parse_sections(section: str) -> list[tuple[float, float]]:
    section = section.lstrip("*")
    start_s, _, end_s = section.partition("-")

    def to_sec(t: str) -> float:
        parts = [float(p) for p in t.split(":")]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec

    return [(to_sec(start_s), to_sec(end_s))]


def download_video(url: str, video_id: str, cfg: DownloadConfig) -> Path | None:
    """Download one video. Returns the local path or None on failure."""
    _ensure_js_runtime_on_path()
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    outtmpl = str(Path(cfg.out_dir) / f"{video_id}.%(ext)s")
    with yt_dlp.YoutubeDL(_ydl_opts(cfg, outtmpl)) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            print(f"[download] failed {video_id}: {exc}")
            return None
    path = Path(ydl.prepare_filename(info))
    if path.exists():
        return path
    # extension may differ after merge; find by id prefix
    matches = sorted(Path(cfg.out_dir).glob(f"{video_id}.*"))
    return matches[0] if matches else None
