"""Turn per-frame annotation output into validated Timesteps, segment them into games,
and aggregate match-level fields (deck, result, style)."""

from __future__ import annotations

from collections import Counter

from .annotators import PROMPT_VERSION, FrameAnnotator
from .frames import Frame
from .schema import (
    Action,
    ElixirPhase,
    GameMode,
    GameRecord,
    Owner,
    PlayerProfile,
    Result,
    StyleTag,
    TilePosition,
    Timestep,
    TowerState,
    Unit,
)


def _pos(d: dict | None) -> TilePosition | None:
    if not d or d.get("x") is None or d.get("y") is None:
        return None
    try:
        return TilePosition(x=float(d["x"]), y=float(d["y"]))
    except Exception:
        return None


def _owner(v) -> Owner:
    try:
        return Owner(v)
    except ValueError:
        return Owner.UNKNOWN


def _tower(d: dict | None) -> TowerState:
    d = d or {}
    return TowerState(
        king_hp=d.get("king_hp"),
        left_princess_hp=d.get("left_princess_hp"),
        right_princess_hp=d.get("right_princess_hp"),
        king_activated=d.get("king_activated"),
    )


def to_timestep(raw: dict, frame: Frame) -> Timestep:
    try:
        phase = ElixirPhase(raw.get("phase", "unknown"))
    except ValueError:
        phase = ElixirPhase.UNKNOWN
    units = [
        Unit(owner=_owner(u.get("owner", "unknown")), card=u.get("card", "unknown"),
             position=_pos(u.get("position")), count=int(u.get("count", 1) or 1),
             confidence=float(u.get("confidence", 0.0) or 0.0))
        for u in raw.get("units", []) or []
    ]
    actions = [
        Action(owner=_owner(a.get("owner", "unknown")), card=a.get("card", "unknown"),
               position=_pos(a.get("position")), elixir_cost=a.get("elixir_cost"),
               confidence=float(a.get("confidence", 0.0) or 0.0))
        for a in raw.get("actions", []) or []
    ]
    return Timestep(
        game_time_s=raw.get("game_time_s"),
        video_time_s=frame.video_time_s,
        frame_path=frame.path,
        elixir_player=raw.get("elixir_player"),
        elixir_opponent=raw.get("elixir_opponent"),
        phase=phase,
        player_towers=_tower(raw.get("player_towers")),
        opponent_towers=_tower(raw.get("opponent_towers")),
        crowns_player=raw.get("crowns_player"),
        crowns_opponent=raw.get("crowns_opponent"),
        units=units,
        actions=actions,
        is_gameplay=bool(raw.get("is_gameplay", True)),
        confidence=float(raw.get("confidence", 0.0) or 0.0),
        raw_annotation=raw,
    )


def extract_timesteps(frames: list[Frame], provider: FrameAnnotator) -> list[Timestep]:
    steps: list[Timestep] = []
    context = ""
    for fr in frames:
        try:
            raw = provider.extract_state(fr.path, context=context)
        except Exception as exc:
            # One bad/slow frame must not abort an entire video. Emit a
            # low-confidence non-gameplay placeholder and keep going.
            print(f"[extract] frame failed ({fr.path}): {exc}")
            steps.append(_failed_timestep(fr))
            continue
        ts = to_timestep(raw, fr)
        steps.append(ts)
        hand = raw.get("visible_cards_in_hand")
        if hand:
            context = f"prev hand={hand} prev elixir={ts.elixir_player}"
    return steps


def _failed_timestep(frame: Frame) -> Timestep:
    """Placeholder for a frame whose annotation failed."""
    return Timestep(
        game_time_s=None,
        video_time_s=frame.video_time_s,
        frame_path=frame.path,
        phase=ElixirPhase.UNKNOWN,
        player_towers=_tower(None),
        opponent_towers=_tower(None),
        units=[],
        actions=[],
        is_gameplay=False,
        confidence=0.0,
        raw_annotation={"error": "annotation_failed"},
    )


def segment_games(steps: list[Timestep]) -> list[list[Timestep]]:
    """Split a timeline into individual matches.

    Boundaries are detected when gameplay pauses (a run of non-gameplay frames)
    or the in-game clock / crown counts reset to an earlier value.
    """
    games: list[list[Timestep]] = []
    current: list[Timestep] = []
    gap = 0
    for ts in steps:
        if not ts.is_gameplay:
            gap += 1
            if current and gap >= 2:
                games.append(current)
                current = []
            continue
        gap = 0
        if current:
            prev = current[-1]
            reset = (
                (ts.game_time_s is not None and prev.game_time_s is not None
                 and ts.game_time_s + 30 < prev.game_time_s)
                or (ts.crowns_player is not None and prev.crowns_player is not None
                    and ts.crowns_player < prev.crowns_player)
            )
            if reset:
                games.append(current)
                current = []
        current.append(ts)
    if current:
        games.append(current)
    return [g for g in games if len(g) >= 3]


def _infer_deck(steps: list[Timestep], owner: Owner) -> list[str]:
    counts: Counter[str] = Counter()
    for ts in steps:
        for a in ts.actions:
            if a.owner == owner and a.card and a.card != "unknown":
                counts[a.card] += 1
        for u in ts.units:
            if u.owner == owner and u.card and u.card != "unknown":
                counts[u.card] += 1
        if ts.raw_annotation and owner == Owner.PLAYER:
            for c in ts.raw_annotation.get("visible_cards_in_hand", []) or []:
                counts[c] += 1
    return [c for c, _ in counts.most_common(8)]


def _infer_result(steps: list[Timestep]) -> tuple[Result, int | None, int | None]:
    # Primary: use crown counts from frame annotation.
    cp = co = None
    for ts in steps:
        if ts.crowns_player is not None:
            cp = max(cp or 0, ts.crowns_player)
        if ts.crowns_opponent is not None:
            co = max(co or 0, ts.crowns_opponent)

    # Fallback: infer crowns from destroyed towers (HP=0 or explicitly missing)
    if (cp is None or cp == 0) or (co is None or co == 0):
        p_destroyed = 0
        o_destroyed = 0
        for ts in steps:
            pt = ts.player_towers
            ot = ts.opponent_towers
            if pt:
                if pt.left_princess_hp == 0:
                    o_destroyed = max(o_destroyed, 1)
                if pt.right_princess_hp == 0:
                    o_destroyed = max(o_destroyed, 1)
                if pt.king_hp is not None and pt.king_hp == 0:
                    o_destroyed = 3
            if ot:
                if ot.left_princess_hp == 0:
                    p_destroyed = max(p_destroyed, 1)
                if ot.right_princess_hp == 0:
                    p_destroyed = max(p_destroyed, 1)
                if ot.king_hp is not None and ot.king_hp == 0:
                    p_destroyed = 3
        # Note: tower destruction gives opponent crowns, not player
        if p_destroyed > (cp or 0):
            cp = p_destroyed
        if o_destroyed > (co or 0):
            co = o_destroyed

    if cp is None or co is None:
        return Result.UNKNOWN, cp, co
    if cp > co:
        return Result.PLAYER_WIN, cp, co
    if co > cp:
        return Result.OPPONENT_WIN, cp, co
    return Result.DRAW, cp, co


_BEATDOWN = {"Golem", "Lava Hound", "Electro Giant", "Giant", "Elixir Golem"}
_SIEGE = {"X-Bow", "Mortar"}
_CYCLE = {"Hog Rider", "Skeletons", "Ice Spirit", "The Log", "Electro Spirit"}
_BRIDGE_SPAM = {"Bandit", "Battle Ram", "Royal Ghost", "Ram Rider"}
_LAVALOON = {"Lava Hound", "Balloon"}
_LOGBAIT = {"Goblin Barrel", "Princess", "Goblin Gang", "Rocket"}
_MINER_POISON = {"Miner", "Poison"}
_SPELL_CYCLE = {"Rocket", "Mirror", "Earthquake", "Tornado"}


def _infer_style(deck: list[str]) -> list[StyleTag]:
    tags: list[StyleTag] = []
    dset = set(deck)
    if dset & _BEATDOWN:
        tags.append(StyleTag.BEATDOWN)
    if dset & _SIEGE:
        tags.append(StyleTag.SIEGE)
    if len(dset & _CYCLE) >= 2:
        tags.append(StyleTag.CYCLE)
    if "Graveyard" in dset:
        tags.append(StyleTag.GRAVEYARD)
    if len(dset & _BRIDGE_SPAM) >= 2:
        tags.append(StyleTag.BRIDGE_SPAM)
    if len(dset & _LAVALOON) == 2:
        tags.append(StyleTag.BEATDOWN)
    if len(dset & _LOGBAIT) >= 3:
        tags.append(StyleTag.CYCLE)
    # Deduplicate
    seen = set()
    tags = [t for t in tags if t not in seen and not seen.add(t)]
    return tags or [StyleTag.UNKNOWN]


def build_game(
    steps: list[Timestep], *, video_id: str, video_url: str, game_index: int,
    channel: str | None = None, upload_date: str | None = None,
    annotation_model: str | None = None, sample_fps: float | None = None,
) -> GameRecord:
    result, cp, co = _infer_result(steps)
    player_deck = _infer_deck(steps, Owner.PLAYER)
    opp_deck = _infer_deck(steps, Owner.OPPONENT)
    confs = [s.confidence for s in steps if s.confidence]
    return GameRecord(
        video_id=video_id, video_url=video_url, channel=channel,
        upload_date=upload_date, game_index_in_video=game_index,
        segment_start_s=steps[0].video_time_s, segment_end_s=steps[-1].video_time_s,
        mode=GameMode.UNKNOWN,
        player=PlayerProfile(deck=player_deck, style_tags=_infer_style(player_deck)),
        opponent=PlayerProfile(deck=opp_deck, style_tags=_infer_style(opp_deck)),
        timeline=steps, result=result,
        final_crowns_player=cp, final_crowns_opponent=co,
        annotation_model=annotation_model, sample_fps=sample_fps, prompt_version=PROMPT_VERSION,
        mean_confidence=round(sum(confs) / len(confs), 3) if confs else None,
    )
