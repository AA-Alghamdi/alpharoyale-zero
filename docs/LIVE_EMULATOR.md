# Live Emulator Bridge

The live bridge controls Clash Royale in BlueStacks through ADB. It converts a
screen frame into structured state, maps that state to the simulator policy
interface, selects an action, and sends calibrated taps to the emulator.

## Flow

```text
BlueStacks screenshot -> perception backend -> state shim -> policy -> tile tap
```

## Setup

1. Install BlueStacks Air on macOS / Apple Silicon.
2. Install Clash Royale and enable Android Debug Bridge in BlueStacks settings.
3. Connect ADB:

   ```bash
   adb connect 127.0.0.1:5555
   ```

4. Install the perception backend expected by the app:

   ```bash
   cd apps/emulator_bot
   git clone https://github.com/Pbatch/ClashRoyaleBuildABot.git
   pip install -e "../../[emulator]"
   ```

5. Validate a frame:

   ```bash
   python -m perception.cli frame.png
   ```

## Running

Run the standalone tiered strategy:

```bash
cd apps/emulator_bot
python strategy_bot.py 5
```

Run the simulator-compatible bridge policy:

```bash
cd apps/emulator_bot
python proagent_play.py 1
```

## Engineering Details

- Screenshots use `adb exec-out screencap -p` and retry corrupt frames.
- Taps are emitted through short `input swipe` gestures because instantaneous
  taps can be ignored by BlueStacks.
- Coordinates are computed in the 720x1280 detector space and scaled to the
  1080x1920 emulator surface.
- Spell actions are guarded so the bridge does not cast into an empty board.
- Decisions and match summaries are logged under `~/clash-royale-bot/runs/`.

## Current Capability

The live bridge is operational for low-arena matches with scripted policies and
is suitable for collecting decision logs, validating perception, and testing
policy handoff. Strong learned-policy deployment should wait for a checkpoint
that is stable against the simulator baselines.
