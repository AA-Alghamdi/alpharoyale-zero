# Running ClashRoyaleBuildABot's detectors on macOS / Apple Silicon

The perception backend (`perception/backends/buildabot.py`) imports the detector
stack from a local clone of
[ClashRoyaleBuildABot](https://github.com/Pbatch/ClashRoyaleBuildABot). Four
changes are needed to make it run headless on macOS + drive BlueStacks reliably:

1. **Press-style taps.** BlueStacks ignores instantaneous `input tap`. In
   `clashroyalebuildabot/emulator/emulator.py`, send taps as a zero-length swipe
   with a short duration:
   ```python
   def click(self, x, y):
       sx = int(round(x * self.width / DISPLAY_WIDTH))
       sy = int(round(y * self.height / DISPLAY_HEIGHT))
       self._run_command(["shell", "input", "swipe", str(sx), str(sy), str(sx), str(sy), "90"])
   ```
   (This also fixes the coordinate scale: BuildABot computes in a 720×1280
   `DISPLAY` space; scale to the real device, e.g. ×1.5 for 1080×1920.)

2. **Don't import `keyboard`.** BuildABot's `__init__.py` eagerly imports `.bot`,
   which imports the `keyboard` library; on macOS that installs CoreGraphics event
   taps that corrupt Pillow's JPEG decoder (a `CFData` assertion crash when the
   card-art `.jpg`s load). Slim `clashroyalebuildabot/__init__.py` to import only
   `constants` + `namespaces`, and import detectors directly
   (`from clashroyalebuildabot.detectors.detector import Detector`).

3. **Install onnxruntime.** The YOLO unit detector needs it:
   `pip install onnxruntime`.

4. **Pillow JPEG.** With fix #2 in place, the bundled card `.jpg`s decode fine.
   (Without it, any JPEG decode after importing the package segfaults.)

These are documented rather than vendored to respect the upstream MIT project.
