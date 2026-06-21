# Emulator Bot

This app runs Clash Royale through BlueStacks and exposes a live-control bridge
for policies in the monorepo.

## Components

| Path | Purpose |
|---|---|
| `perception/` | Frame-to-state adapter with pluggable detector backends. |
| `strategy_bot.py` | Standalone priority-tiered policy for live matches. |
| `proagent_play.py` | Bridge from live detections to the simulator-compatible baseline policy. |
| `play_bot.py` | Earlier elixir-bar runner kept as a reference. |
| `analyze.py` | Match and decision-log summaries. |
| `brain/` | Curated card and strategy knowledge used by the scripted policy work. |

## Run

```bash
adb connect 127.0.0.1:5555
python -m perception.cli frame.png
python strategy_bot.py 5
python proagent_play.py 1
```

Decision logs are written to `~/clash-royale-bot/runs/`.
