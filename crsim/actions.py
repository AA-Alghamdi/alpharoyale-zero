"""Canonical RL action codec: flat action-id ↔ game :class:`Action`.

The policy/value network and every MCTS variant operate over a flat action
space of size :data:`ACTION_SPACE_SIZE`. This module is the **single source of
truth** for translating between a flat id and a game :class:`Action`, so the
simulator, search, training and evaluation paths can never drift apart.

Layout (see :mod:`crsim.constants`)::

    [0, NUM_HAND_SLOTS * ARENA_W * ARENA_H)   placements, id = slot*W*H + x*H + y
    ABILITY_ACTION                            activate the deployed champion ability
    WAIT_ACTION                               do nothing

Historically each search module carried its own ``_action_id_to_action`` copy;
two of them silently dropped the ABILITY case and decoded it as an out-of-range
hand slot. Routing every caller through here removes that hazard and gives us
the inverse (:func:`action_to_action_id`), which imitation learning needs to map
real games onto the action space.
"""

from __future__ import annotations

from crsim.constants import (
    ABILITY_ACTION,
    ACTION_SPACE_SIZE,
    ARENA_H,
    ARENA_W,
    NUM_HAND_SLOTS,
    WAIT_ACTION,
)
from crsim.game import Action

_TILES_PER_SLOT = ARENA_W * ARENA_H


def action_id_to_action(action_id: int, player: int) -> Action:
    """Decode a flat action id into a game :class:`Action` for ``player``.

    Raises
    ------
    ValueError
        If ``action_id`` is outside ``[0, ACTION_SPACE_SIZE)``.
    """
    if not 0 <= action_id < ACTION_SPACE_SIZE:
        raise ValueError(
            f"action_id {action_id} out of range [0, {ACTION_SPACE_SIZE})"
        )
    if action_id == WAIT_ACTION:
        return Action(player=player, hand_slot=-1)
    if action_id == ABILITY_ACTION:
        return Action(player=player, hand_slot=-1, ability=True)
    slot, remainder = divmod(action_id, _TILES_PER_SLOT)
    x, y = divmod(remainder, ARENA_H)
    return Action(player=player, hand_slot=slot, x=float(x), y=float(y))


def action_to_action_id(action: Action) -> int:
    """Encode a game :class:`Action` back into its flat action id (inverse of
    :func:`action_id_to_action`).

    Raises
    ------
    ValueError
        If the action carries an out-of-range hand slot or placement.
    """
    if action.ability:
        return ABILITY_ACTION
    if action.is_wait:
        return WAIT_ACTION
    slot = action.hand_slot
    if not 0 <= slot < NUM_HAND_SLOTS:
        raise ValueError(f"hand_slot {slot} out of range [0, {NUM_HAND_SLOTS})")
    x, y = int(action.x), int(action.y)
    if not (0 <= x < ARENA_W and 0 <= y < ARENA_H):
        raise ValueError(f"placement ({x}, {y}) out of arena bounds")
    return slot * _TILES_PER_SLOT + x * ARENA_H + y
