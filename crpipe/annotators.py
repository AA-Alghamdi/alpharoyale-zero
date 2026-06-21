"""Pluggable annotation providers for per-frame Clash Royale state extraction.

Each provider takes a single frame (PNG/JPEG path) plus light textual context and
returns a JSON dict that maps onto :class:`crpipe.schema.Timestep` fields. The
``mock`` provider lets the entire pipeline run end-to-end with no API key, which
is what CI and local smoke tests use.

Real providers read their key from the environment:
  - openai     -> OPENAI_API_KEY
  - anthropic  -> ANTHROPIC_API_KEY
  - gemini     -> GEMINI_API_KEY
"""

from __future__ import annotations

import base64
import json
import os
import random
import re
import time

import requests

PROMPT_VERSION = "cr-state-v1"


class _RetryableHTTPError(Exception):
    """Raised for HTTP statuses that should trigger a backoff + retry."""

    def __init__(self, status: int, body: str = ""):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status


SYSTEM_PROMPT = """You are a precise Clash Royale match annotator. You are shown a \
single frame from a 1v1 Clash Royale gameplay video. The arena is an 18-wide by \
32-tall tile grid; x increases to the right, y increases from the bottom (the \
player whose elixir bar is shown) toward the top (opponent). The river is near y=16.

Return ONLY a JSON object (no prose, no markdown fences) with this shape:
{
  "is_gameplay": bool,            // false for menus, intros, replays, end screens
  "game_time_s": number|null,     // in-game clock read from the timer if visible
  "phase": "single|double|triple|overtime|unknown",
  "elixir_player": number|null,   // 0-10, the visible bottom elixir bar
  "elixir_opponent": number|null, // usually null (not shown)
  "crowns_player": int|null,
  "crowns_opponent": int|null,
  "player_towers":   {"king_hp": int|null, "left_princess_hp": int|null, "right_princess_hp": int|null, "king_activated": bool|null},
  "opponent_towers": {"king_hp": int|null, "left_princess_hp": int|null, "right_princess_hp": int|null, "king_activated": bool|null},
  "units": [ {"owner":"player|opponent", "card": str, "position": {"x": number, "y": number}, "count": int, "confidence": number} ],
  "actions": [ {"owner":"player|opponent", "card": str, "position": {"x": number, "y": number}, "elixir_cost": int|null, "confidence": number} ],
  "visible_cards_in_hand": [str],  // the 4 cards in the player's hand if readable
  "confidence": number             // overall 0-1 confidence for this frame
}
Use exact official card names (e.g. "Hog Rider", "Fireball", "Mega Knight"). If a \
field is unreadable use null. Do not invent units that are not visible."""

USER_INSTRUCTION = (
    "Annotate this frame. Compare with the previous frame's hand/elixir context to "
    "infer which card was just placed (put it in 'actions'). Context: {context}"
)


class FrameAnnotator:
    name = "base"

    def extract_state(self, frame_path: str, context: str = "") -> dict:
        raise NotImplementedError


def _b64_image(frame_path: str) -> str:
    with open(frame_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _strip_json(text: str) -> dict:
    """Best-effort parse of a JSON object from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


class MockFrameAnnotator(FrameAnnotator):
    """Deterministic-ish plausible output; no network, no key required."""

    name = "mock"

    _CARDS = [
        "Hog Rider", "Fireball", "Musketeer", "Ice Spirit", "Skeletons",
        "Cannon", "The Log", "Ice Golem", "Valkyrie", "Mega Knight",
    ]

    def extract_state(self, frame_path: str, context: str = "") -> dict:
        seed = abs(hash((os.path.basename(frame_path), context))) % (2**32)
        rng = random.Random(seed)
        units = [
            {
                "owner": rng.choice(["player", "opponent"]),
                "card": rng.choice(self._CARDS),
                "position": {"x": round(rng.uniform(0, 18), 1), "y": round(rng.uniform(0, 32), 1)},
                "count": 1,
                "confidence": round(rng.uniform(0.5, 0.95), 2),
            }
            for _ in range(rng.randint(0, 4))
        ]
        actions = []
        if rng.random() < 0.4:
            actions.append({
                "owner": rng.choice(["player", "opponent"]),
                "card": rng.choice(self._CARDS),
                "position": {"x": round(rng.uniform(0, 18), 1), "y": round(rng.uniform(0, 32), 1)},
                "elixir_cost": rng.randint(1, 7),
                "confidence": round(rng.uniform(0.4, 0.9), 2),
            })
        return {
            "is_gameplay": True,
            "game_time_s": None,
            "phase": rng.choice(["single", "double", "overtime"]),
            "elixir_player": rng.randint(0, 10),
            "elixir_opponent": None,
            "crowns_player": rng.randint(0, 2),
            "crowns_opponent": rng.randint(0, 2),
            "player_towers": {"king_hp": rng.randint(2000, 4824), "left_princess_hp": rng.randint(0, 2534),
                               "right_princess_hp": rng.randint(0, 2534), "king_activated": False},
            "opponent_towers": {"king_hp": rng.randint(2000, 4824), "left_princess_hp": rng.randint(0, 2534),
                                 "right_princess_hp": rng.randint(0, 2534), "king_activated": False},
            "units": units,
            "actions": actions,
            "visible_cards_in_hand": rng.sample(self._CARDS, 4),
            "confidence": round(rng.uniform(0.4, 0.85), 2),
        }


class OpenAIFrameAnnotator(FrameAnnotator):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def extract_state(self, frame_path: str, context: str = "") -> dict:
        b64 = _b64_image(frame_path)
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": USER_INSTRUCTION.format(context=context or "none")},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
        }
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload, timeout=120,
        )
        resp.raise_for_status()
        return _strip_json(resp.json()["choices"][0]["message"]["content"])


class AnthropicFrameAnnotator(FrameAnnotator):
    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("ANTHROPIC_MODEL")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        if not self.model:
            raise RuntimeError("Anthropic model not set")

    def extract_state(self, frame_path: str, context: str = "") -> dict:
        b64 = _b64_image(frame_path)
        payload = {
            "model": self.model,
            "max_tokens": 1500,
            "temperature": 0,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": USER_INSTRUCTION.format(context=context or "none")},
                ]},
            ],
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            json=payload, timeout=120,
        )
        resp.raise_for_status()
        return _strip_json(resp.json()["content"][0]["text"])


class GeminiFrameAnnotator(FrameAnnotator):
    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: str | None = None,
        max_retries: int = 5,
        timeout: float = 90.0,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.max_retries = max_retries
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

    def _gen_config(self) -> dict:
        cfg: dict = {
            "temperature": 0,
            "response_mime_type": "application/json",
        }
        if "2.5-flash" in self.model:
            cfg["thinkingConfig"] = {"thinkingBudget": 0}
        return cfg

    def _call_api(self, payload: dict) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise _RetryableHTTPError(resp.status_code, resp.text[:200])
                resp.raise_for_status()
                parts = resp.json()["candidates"][0]["content"]["parts"]
                return parts[0]["text"]
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    _RetryableHTTPError) as exc:
                last_exc = exc
                if attempt == self.max_retries - 1:
                    break
                delay = min(2.0 ** attempt, 30.0) + random.uniform(0, 1.0)
                time.sleep(delay)
        raise RuntimeError(
            f"Gemini call failed after {self.max_retries} attempts: {last_exc}"
        )

    def extract_state(self, frame_path: str, context: str = "") -> dict:
        b64 = _b64_image(frame_path)
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [
                {"text": USER_INSTRUCTION.format(context=context or "none")},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            ]}],
            "generationConfig": self._gen_config(),
        }
        return _strip_json(self._call_api(payload))

    def extract_states_batch(self, frame_paths: list[str], context: str = "") -> list[dict]:
        """Annotate multiple frames in a single API call (saves RPD quota).
        Returns a list of annotation dicts, one per frame."""
        parts: list[dict] = []
        for idx, fp in enumerate(frame_paths):
            b64 = _b64_image(fp)
            parts.append({"text": f"Frame {idx} (video second {idx}):"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})

        batch_instruction = (
            f"Annotate each of the {len(frame_paths)} frames below. "
            f"Return a JSON array of {len(frame_paths)} objects, one per frame, "
            f"in order. Each object has the same schema as described in the system "
            f"instructions. Context from previous frames: {context or 'none'}"
        )
        parts.insert(0, {"text": batch_instruction})

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": parts}],
            "generationConfig": self._gen_config(),
        }
        raw_text = self._call_api(payload)
        # Parse: expect a JSON array
        raw_text = raw_text.strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```[a-zA-Z]*", "", raw_text).strip().rstrip("`").strip()
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return [_strip_json(raw_text)]


_PROVIDERS = {
    "mock": MockFrameAnnotator,
    "openai": OpenAIFrameAnnotator,
    "anthropic": AnthropicFrameAnnotator,
    "gemini": GeminiFrameAnnotator,
}


def get_provider(name: str, **kwargs) -> FrameAnnotator:
    name = name.lower()
    if name not in _PROVIDERS:
        raise ValueError(f"unknown annotation provider {name!r}; choices: {sorted(_PROVIDERS)}")
    return _PROVIDERS[name](**kwargs)
