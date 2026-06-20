"""Core game engine — single-game Clash Royale simulator."""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass

import numpy as np

from crsim.cards import CARD_DEFS, CardDef, CardType, EntityKind, TargetMode
from crsim.constants import (
    ARENA_H,
    ARENA_W,
    BRIDGE_LEFT_COLS,
    BRIDGE_RIGHT_COLS,
    DOUBLE_ELIXIR_START_TICKS,
    ELIXIR_REGEN_DOUBLE,
    ELIXIR_REGEN_NORMAL,
    ELIXIR_REGEN_OVERTIME,
    ELIXIR_REGEN_SUDDEN,
    KING_TOWER_DAMAGE,
    KING_TOWER_HP,
    KING_TOWER_RANGE,
    MAX_ELIXIR,
    MAX_ENTITIES,
    NUM_HAND_SLOTS,
    OVERTIME_TICKS,
    P0_KING_POS,
    P0_PRINCESS_L_POS,
    P0_PRINCESS_R_POS,
    P1_KING_POS,
    P1_PRINCESS_L_POS,
    P1_PRINCESS_R_POS,
    PRINCESS_TOWER_DAMAGE,
    PRINCESS_TOWER_HP,
    PRINCESS_TOWER_RANGE,
    REGULAR_TIME_TICKS,
    RIVER_ROW_HI,
    RIVER_ROW_LO,
    SPEED_MEDIUM,
    STARTING_ELIXIR,
    SUDDEN_DEATH_TICKS,
    TICK_DURATION,
    TOWER_ATTACK_INTERVAL,
)
from crsim.entities import Entity, entity_from_card, make_tower
from crsim.pathfinding import FlowFieldCache, direction_to_target


class GamePhase(enum.IntEnum):
    REGULAR = 0
    OVERTIME = 1
    SUDDEN_DEATH = 2
    ENDED = 3


class GameResult(enum.IntEnum):
    IN_PROGRESS = 0
    P0_WIN = 1
    P1_WIN = 2
    DRAW = 3


@dataclass
class PlayerState:
    """Mutable per-player state."""

    deck: list[CardType]  # full 8-card deck
    hand: list[int]  # indices into deck (size 4)
    next_card_idx: int  # index of the next card to draw from deck
    elixir: float = STARTING_ELIXIR

    def draw_card(self, used_hand_slot: int) -> None:
        """Replace the used hand slot with the next card from the deck cycle."""
        self.hand[used_hand_slot] = self.next_card_idx
        self.next_card_idx = (self.next_card_idx + 1) % len(self.deck)


@dataclass
class Action:
    """A player action: place a card, activate a champion ability, or wait."""

    player: int
    hand_slot: int = -1  # -1 = wait
    x: float = 0.0
    y: float = 0.0
    ability: bool = False  # True = activate the player's deployed champion ability

    @property
    def is_wait(self) -> bool:
        return self.hand_slot < 0 and not self.ability


class CRGame:
    """Single Clash Royale game instance."""

    def __init__(
        self,
        deck_p0: list[CardType] | None = None,
        deck_p1: list[CardType] | None = None,
        seed: int | None = None,
    ) -> None:
        self.rng = random.Random(seed)

        # Default starter decks
        if deck_p0 is None:
            deck_p0 = [
                CardType.KNIGHT, CardType.ARCHERS, CardType.FIREBALL,
                CardType.GIANT, CardType.MUSKETEER, CardType.VALKYRIE,
                CardType.ARROWS, CardType.MINI_PEKKA,
            ]
        if deck_p1 is None:
            deck_p1 = list(deck_p0)

        # Shuffle decks
        self.rng.shuffle(deck_p0)
        self.rng.shuffle(deck_p1)

        self.players: list[PlayerState] = [
            PlayerState(
                deck=deck_p0,
                hand=list(range(4)),
                next_card_idx=4,
            ),
            PlayerState(
                deck=deck_p1,
                hand=list(range(4)),
                next_card_idx=4,
            ),
        ]

        # Entity management
        self.entities: list[Entity] = []
        self._next_eid: int = 0
        self._spawn_towers()

        # Projectile system: list of in-flight projectiles
        # Each: (source_owner, target_eid, x, y, speed, damage, is_splash, splash_radius, stuns, stun_duration)
        self.projectiles: list[dict] = []

        # Delayed area explosions (e.g. Mighty Miner's bomb)
        self.pending_bombs: list[dict] = []

        # Game state
        self.tick_count: int = 0
        self.phase: GamePhase = GamePhase.REGULAR
        self.result: GameResult = GameResult.IN_PROGRESS

        # Pathfinding
        self.flow_cache = FlowFieldCache()

        # Tower tracking for quick access
        self.king_towers: list[Entity | None] = [None, None]
        self.princess_towers: list[list[Entity]] = [[], []]
        self._index_towers()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _alloc_eid(self) -> int:
        eid = self._next_eid
        self._next_eid += 1
        return eid

    def _spawn_towers(self) -> None:
        """Place all 6 towers."""
        positions = [
            (0, P0_KING_POS, True),
            (0, P0_PRINCESS_L_POS, False),
            (0, P0_PRINCESS_R_POS, False),
            (1, P1_KING_POS, True),
            (1, P1_PRINCESS_L_POS, False),
            (1, P1_PRINCESS_R_POS, False),
        ]
        for owner, (px, py), is_king in positions:
            hp = KING_TOWER_HP if is_king else PRINCESS_TOWER_HP
            dmg = KING_TOWER_DAMAGE if is_king else PRINCESS_TOWER_DAMAGE
            rng = KING_TOWER_RANGE if is_king else PRINCESS_TOWER_RANGE
            tower = make_tower(
                eid=self._alloc_eid(),
                owner=owner,
                x=px, y=py,
                hp=hp, damage=dmg,
                attack_range=rng,
                attack_interval=TOWER_ATTACK_INTERVAL,
                is_king=is_king,
            )
            self.entities.append(tower)

    def _index_towers(self) -> None:
        for e in self.entities:
            if not e.is_tower:
                continue
            if e.is_king_tower:
                self.king_towers[e.owner] = e
            else:
                self.princess_towers[e.owner].append(e)

    # ------------------------------------------------------------------
    # Elixir
    # ------------------------------------------------------------------

    def _elixir_rate(self) -> float:
        if self.phase == GamePhase.OVERTIME:
            return ELIXIR_REGEN_OVERTIME
        if self.phase == GamePhase.SUDDEN_DEATH:
            return ELIXIR_REGEN_SUDDEN
        # Double Elixir during the final minute of regulation (2:00-3:00).
        if self.tick_count >= DOUBLE_ELIXIR_START_TICKS:
            return ELIXIR_REGEN_DOUBLE
        return ELIXIR_REGEN_NORMAL

    def _regen_elixir(self) -> None:
        rate = self._elixir_rate()
        for ps in self.players:
            ps.elixir = min(ps.elixir + rate, MAX_ELIXIR)

        # Elixir collectors
        for e in self.entities:
            if (
                e.alive
                and e.card_type == CardType.ELIXIR_COLLECTOR
                and e.spawner_interval > 0
            ):
                e.spawner_timer -= TICK_DURATION
                if e.spawner_timer <= 0:
                    self.players[e.owner].elixir = min(
                        self.players[e.owner].elixir + 1.0, MAX_ELIXIR
                    )
                    e.spawner_timer = e.spawner_interval

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def get_valid_actions_mask(self, player: int) -> np.ndarray:
        """Return a boolean mask of shape (ACTION_SPACE_SIZE,)."""
        from crsim.constants import ABILITY_ACTION, ACTION_SPACE_SIZE, WAIT_ACTION

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[WAIT_ACTION] = True  # wait is always valid

        ps = self.players[player]

        # Champion ability: valid when a deployed champion is off cooldown and
        # the player can afford its activation cost.
        champ = self._ready_champion(player)
        if champ is not None and ps.elixir >= champ.ability_cost:
            mask[ABILITY_ACTION] = True

        for slot_idx in range(NUM_HAND_SLOTS):
            card_idx = ps.hand[slot_idx]
            card_type = ps.deck[card_idx]
            card_def = CARD_DEFS[card_type]

            if ps.elixir < card_def.cost:
                continue

            # Valid placement positions
            for x in range(ARENA_W):
                for y in range(ARENA_H):
                    if self._is_valid_placement(player, card_def, x, y):
                        action_id = slot_idx * ARENA_W * ARENA_H + x * ARENA_H + y
                        mask[action_id] = True

        return mask

    def _is_valid_placement(
        self, player: int, card_def: CardDef, x: int, y: int
    ) -> bool:
        """Check if a card can be placed at (x, y) by the given player."""
        # Out of bounds
        if x < 0 or x >= ARENA_W or y < 0 or y >= ARENA_H:
            return False

        # Spells can be placed anywhere (including enemy side)
        if card_def.kind == EntityKind.SPELL and card_def.card_type != CardType.GOBLIN_BARREL:
            return True

        # Goblin Barrel can be placed anywhere
        if card_def.card_type == CardType.GOBLIN_BARREL:
            return True

        # Troops and buildings: own half only
        if player == 0:
            if y >= RIVER_ROW_LO:
                return False
        else:
            if y <= RIVER_ROW_HI:
                return False

        # Not on river or tower positions
        if RIVER_ROW_LO <= y <= RIVER_ROW_HI:
            return False

        return True

    def apply_action(self, action: Action) -> None:
        """Apply a single player action."""
        if action.ability:
            self._activate_champion_ability(action.player)
            return
        if action.is_wait:
            return

        ps = self.players[action.player]
        card_idx = ps.hand[action.hand_slot]
        card_type = ps.deck[card_idx]
        card_def = CARD_DEFS[card_type]

        # Check elixir
        if ps.elixir < card_def.cost:
            return  # invalid; silently ignore

        ps.elixir -= card_def.cost

        # Spawn entities
        self._spawn_card(action.player, card_def, action.x, action.y)

        # Draw next card
        ps.draw_card(action.hand_slot)

    # ------------------------------------------------------------------
    # Champion abilities
    # ------------------------------------------------------------------

    def _ready_champion(self, player: int) -> Entity | None:
        """The player's deployed champion whose ability can fire right now.

        CR permits a single champion per deck and one on the field, so we return
        the first alive, deployed champion owned by ``player`` whose ability is
        off cooldown. Elixir is *not* checked here; callers that spend elixir
        check it themselves (the mask uses this plus an elixir test).
        """
        for e in self.entities:
            if (
                e.alive
                and e.is_champion
                and e.owner == player
                and e.is_deployed
                and e.ability_cooldown_timer <= 0
            ):
                return e
        return None

    def _activate_champion_ability(self, player: int) -> None:
        """Activate the player's champion ability if one is ready and affordable."""
        champ = self._ready_champion(player)
        if champ is None:
            return
        ps = self.players[player]
        if ps.elixir < champ.ability_cost:
            return
        ps.elixir -= champ.ability_cost
        champ.ability_cooldown_timer = champ.ability_cooldown
        self._dispatch_champion_ability(champ)

    def _dispatch_champion_ability(self, champ: Entity) -> None:
        """Apply a champion's ability effect, dispatched by card type."""
        if champ.card_type == CardType.SKELETON_KING:
            self._ability_skeleton_king(champ)
        elif champ.card_type == CardType.ARCHER_QUEEN:
            self._ability_archer_queen(champ)
        elif champ.card_type == CardType.GOLDEN_KNIGHT:
            self._ability_golden_knight(champ)
        elif champ.card_type == CardType.MONK:
            self._ability_monk(champ)
        elif champ.card_type == CardType.MIGHTY_MINER:
            self._ability_mighty_miner(champ)
        elif champ.card_type == CardType.LITTLE_PRINCE:
            self._ability_little_prince(champ)

    def _ability_little_prince(self, champ: Entity) -> None:
        """Royal Rescue: summon the Guardienne to fight alongside the Prince."""
        if len(self.entities) >= MAX_ENTITIES:
            return
        forward = 1.0 if champ.owner == 0 else -1.0
        guard = entity_from_card(
            eid=self._alloc_eid(),
            owner=champ.owner,
            card_def=CARD_DEFS[CardType.GUARDIENNE],
            x=champ.x,
            y=max(0.0, min(float(ARENA_H - 1), champ.y + 1.0 * forward)),
        )
        self.entities.append(guard)

    def _ability_monk(self, champ: Entity) -> None:
        """Pensive Protection: heavily reduce incoming damage for a short time."""
        duration = 3.5
        champ.damage_reduction_timer = duration
        champ.damage_reduction_mult = 0.2  # takes 20% damage while protected

    def _ability_mighty_miner(self, champ: Entity) -> None:
        """Drop Bomb: plant a delayed high-damage explosion, then burrow forward."""
        self.pending_bombs.append({
            "x": champ.x,
            "y": champ.y,
            "owner": champ.owner,
            "timer": 2.0,
            "damage": 800.0,
            "radius": 2.5,
        })
        # Burrow a few tiles forward toward the enemy side.
        forward = 1.0 if champ.owner == 0 else -1.0
        champ.y = max(0.0, min(float(ARENA_H - 1), champ.y + 4.0 * forward))

    def _ability_archer_queen(self, champ: Entity) -> None:
        """Cloaking Cape: go invisible and fire much faster for the duration."""
        duration = 3.5
        champ.invisible_timer = duration
        champ.attack_speed_timer = duration
        # Hit speed roughly 1.2s -> ~0.7s while cloaked (~1.7x faster).
        champ.attack_speed_mult = 1.7

    def _ability_golden_knight(self, champ: Entity) -> None:
        """Dashing Dash: dash through nearby enemies, damaging each in the chain.

        Approximated as a single forward dash: every enemy unit within
        ``dash_range`` ahead takes the Golden Knight's hit damage, and he
        repositions to the farthest one he struck.
        """
        dash_range = 5.0
        forward = 1.0 if champ.owner == 0 else -1.0
        hit: list[Entity] = []
        for other in self.entities:
            if not (other.alive and other.owner != champ.owner):
                continue
            if other.is_tower or other.is_building:
                continue
            if (other.y - champ.y) * forward < 0:
                continue  # only enemies ahead in the lane direction
            if champ.distance_to(other) <= dash_range:
                hit.append(other)
        for other in hit:
            other.take_damage(self._crown_adjusted_entity(champ, champ.damage_per_hit, other))
        if hit:
            target = max(hit, key=lambda e: (e.y - champ.y) * forward)
            champ.x, champ.y = target.x, target.y

    def _ability_skeleton_king(self, champ: Entity) -> None:
        """Soul Summoning: raise a swarm of skeletons around the King."""
        sk_def = CARD_DEFS[CardType.SKELETONS]
        count = 6
        ring = [
            (0.8, 0.0), (-0.8, 0.0), (0.0, 0.8), (0.0, -0.8),
            (0.8, 0.8), (-0.8, -0.8),
        ]
        for i in range(count):
            if len(self.entities) >= MAX_ENTITIES:
                break
            ox, oy = ring[i % len(ring)]
            skel = entity_from_card(
                eid=self._alloc_eid(),
                owner=champ.owner,
                card_def=sk_def,
                x=champ.x + ox,
                y=champ.y + oy,
                hp_override=sk_def.spawn_hp,
                dps_override=sk_def.spawn_dps,
            )
            self.entities.append(skel)

    def _spawn_card(
        self, player: int, card_def: CardDef, x: float, y: float
    ) -> None:
        """Spawn entities for a played card."""
        # Spells: route to unit-spawning only when the spell actually drops
        # units with their own HP (e.g. Goblin Barrel). Everything else
        # (Fireball, Zap, Arrows, Rocket, Freeze, ...) is a damage/effect
        # spell. Most damage spells have spawn_count==1 by default, so routing
        # on spawn_count alone misclassified them as unit-spawners.
        if card_def.kind == EntityKind.SPELL:
            if card_def.spawn_hp > 0 and card_def.spawn_count > 0:
                self._apply_spell_spawn(player, card_def, x, y)
            else:
                self._apply_spell(player, card_def, x, y)
            return

        # Multi-spawn troops (Archers, Minions, Skeleton Army)
        count = card_def.spawn_count
        if count > 1:
            hp = card_def.spawn_hp
            dps = card_def.spawn_dps
        else:
            hp = card_def.hp
            dps = card_def.dps

        # Deploy stagger: each subsequent unit in a multi-spawn card appears
        # slightly later (e.g., Goblins have 0.2s stagger, Minion Horde 0.1s)
        deploy_stagger = 0.15 if count > 1 else 0.0  # default 150ms between spawns

        for i in range(count):
            if len(self.entities) >= MAX_ENTITIES:
                break
            # Slight offset for multi-unit spawns
            ox = (i % 3 - 1) * 0.5
            oy = (i // 3) * 0.5
            e = entity_from_card(
                eid=self._alloc_eid(),
                owner=player,
                card_def=card_def,
                x=x + ox,
                y=y + oy,
                hp_override=hp if count > 1 else 0.0,
                dps_override=dps if count > 1 else 0.0,
            )
            # Apply deploy stagger: each subsequent unit has additional delay
            e.deploy_timer += deploy_stagger * i
            e.is_deployed = e.deploy_timer <= 0
            self.entities.append(e)

        # Invalidate flow cache if a building was placed
        if card_def.kind == EntityKind.BUILDING:
            self.flow_cache.invalidate()

    @staticmethod
    def _crown_pct_adjust(target: Entity, damage: float, pct: float) -> float:
        """Apply a signed crown_tower_damage_percent to a crown-tower target.

        ``pct`` is a signed reduction: -70 means a crown tower takes 30% of the
        damage. Non-tower targets are unaffected.
        """
        if pct and target.is_tower:
            return damage * max(0.0, (100.0 + pct) / 100.0)
        return damage

    @classmethod
    def _crown_adjusted(cls, target: Entity, damage: float, card_def: CardDef) -> float:
        """Crown-tower reduction for a spell's damage."""
        return cls._crown_pct_adjust(target, damage, card_def.crown_tower_damage_percent)

    @classmethod
    def _crown_adjusted_entity(cls, attacker: Entity, damage: float, target: Entity) -> float:
        """Crown-tower reduction for a unit's own attacks (e.g. Miner -> 25%)."""
        return cls._crown_pct_adjust(target, damage, attacker.crown_tower_damage_percent)

    def _apply_spell(
        self, player: int, card_def: CardDef, x: float, y: float
    ) -> None:
        """Apply area-of-effect spell damage + secondary effects."""
        # AoE radius lives in splash_radius for spells; attack_range is the
        # cast/aim range (often 0 for point-and-click damage spells).
        radius = card_def.splash_radius if card_def.splash_radius > 0 else card_def.attack_range
        # Spell damage is the per-cast hit damage (damage_per_hit). DoT spells
        # (Poison/Earthquake) additionally carry a per-second value in dps.
        damage = card_def.damage_per_hit if card_def.damage_per_hit > 0 else card_def.dps
        ct = card_def.card_type

        # --- Freeze: no damage, applies freeze timer ---
        if ct == CardType.FREEZE:
            for e in self.entities:
                if e.alive and e.owner != player:
                    if e.distance_to_pos(x, y) <= radius:
                        e.freeze_timer = max(e.freeze_timer, 4.0)
                        # Reset inferno on freeze
                        if e.inferno_dps_max > 0:
                            e.inferno_ramp_time = 0.0
            return

        # --- Rage: buff OWN troops with speed/attack boost ---
        if ct == CardType.RAGE:
            for e in self.entities:
                if e.alive and e.owner == player:
                    if e.distance_to_pos(x, y) <= radius:
                        e.rage_timer = max(e.rage_timer, 7.5)
            return

        # --- Poison: DoT over 8 seconds ---
        if ct == CardType.POISON:
            for e in self.entities:
                if e.alive and e.owner != player:
                    if e.distance_to_pos(x, y) <= radius:
                        e.poison_timer = 8.0
                        # dps holds the per-second poison damage; fall back to
                        # spreading the hit damage across the duration.
                        base_dps = card_def.dps if card_def.dps > 0 else damage / 8.0
                        e.poison_dps = self._crown_adjusted(e, base_dps, card_def)
            return

        # --- Tornado: pull enemies toward center ---
        if ct == CardType.TORNADO:
            for e in self.entities:
                if e.alive and e.owner != player and not e.is_building:
                    dist = e.distance_to_pos(x, y)
                    if dist <= radius and dist > 0.1:
                        # Pull toward center
                        pull_strength = 2.0  # tiles per tick
                        dx = x - e.x
                        dy = y - e.y
                        mag = math.sqrt(dx * dx + dy * dy)
                        e.x += (dx / mag) * min(pull_strength * TICK_DURATION, dist)
                        e.y += (dy / mag) * min(pull_strength * TICK_DURATION, dist)
                        e.x = max(0.0, min(float(ARENA_W - 1), e.x))
                        e.y = max(0.0, min(float(ARENA_H - 1), e.y))
                        # Light damage
                        if damage > 0:
                            dmg = self._crown_adjusted(e, damage, card_def)
                            e.take_damage(dmg * TICK_DURATION / 2.5)
            return

        # --- Heal Spirit / Heal: heal OWN troops ---
        if ct == CardType.HEAL_SPIRIT:
            for e in self.entities:
                if e.alive and e.owner == player:
                    if e.distance_to_pos(x, y) <= radius:
                        heal_amount = damage  # stored as dps for heal
                        e.hp = min(e.max_hp, e.hp + heal_amount)
            return

        # --- Lightning: hit 3 highest HP ---
        if ct == CardType.LIGHTNING:
            enemies = [
                e for e in self.entities
                if e.alive and e.owner != player
                and e.distance_to_pos(x, y) <= radius
            ]
            enemies.sort(key=lambda e: e.hp, reverse=True)
            for e in enemies[:3]:
                e.take_damage(self._crown_adjusted(e, damage, card_def))
        else:
            # Standard damage spell (Fireball, Arrows, Rocket, etc.)
            for e in self.entities:
                if e.alive and e.owner != player:
                    if e.distance_to_pos(x, y) <= radius:
                        e.take_damage(self._crown_adjusted(e, damage, card_def))

        # Apply spell secondary effects (knockback, stun, inferno reset)
        self._apply_spell_effects(player, card_def, x, y)

    def _apply_spell_spawn(
        self, player: int, card_def: CardDef, x: float, y: float
    ) -> None:
        """Spawn units from a spell (e.g., Goblin Barrel)."""
        for i in range(card_def.spawn_count):
            if len(self.entities) >= MAX_ENTITIES:
                break
            angle = 2.0 * math.pi * i / card_def.spawn_count
            ox = math.cos(angle) * 1.0
            oy = math.sin(angle) * 1.0
            e = entity_from_card(
                eid=self._alloc_eid(),
                owner=player,
                card_def=card_def,
                x=x + ox,
                y=y + oy,
                hp_override=card_def.spawn_hp,
                dps_override=card_def.spawn_dps,
            )
            e.kind = EntityKind.TROOP  # spawned goblins are troops
            self.entities.append(e)

    # ------------------------------------------------------------------
    # Targeting
    # ------------------------------------------------------------------

    def _king_tower_activated(self, player: int) -> bool:
        """King tower is activated if a princess tower is destroyed or it was hit."""
        kt = self.king_towers[player]
        if kt is None or not kt.alive:
            return False
        if kt.hp < kt.max_hp:
            return True
        for pt in self.princess_towers[player]:
            if not pt.alive:
                return True
        return False

    def _find_target(self, entity: Entity) -> int:
        """Find the best target eid for an entity. Returns -1 if none.
        
        Uses sight_range for initial target acquisition (aggro).
        Once a target is acquired, uses attack_range for actual combat.
        Targeting priority: closest enemy within sight_range.
        Building-targeting troops only see buildings/towers.
        """
        if not entity.active:
            return -1

        # An un-activated king tower does not attack (it only starts firing
        # once it is activated: a princess tower falls or the king is hit).
        if entity.is_king_tower and not self._king_tower_activated(entity.owner):
            return -1

        best_eid = -1
        best_dist = float("inf")
        # Use sight_range for target acquisition
        max_range = entity.sight_range + entity.collision_radius

        for other in self.entities:
            if not other.alive or other.owner == entity.owner:
                continue

            # Target filtering by target_mode
            if entity.target_mode == TargetMode.BUILDINGS:
                if not other.is_building and not other.is_tower:
                    continue
            elif entity.target_mode == TargetMode.GROUND:
                if other.is_flying:
                    continue

            # King tower: only targetable if activated
            if other.is_king_tower:
                if not self._king_tower_activated(other.owner):
                    continue

            # Deploying units can't be targeted
            if not other.is_deployed:
                continue

            # Invisible units (Archer Queen cloak) can't be targeted
            if other.invisible_timer > 0:
                continue

            # Distance calculation accounts for collision radii
            dist = entity.distance_to(other) - other.collision_radius
            if dist > max_range:
                continue

            if dist < best_dist:
                best_dist = dist
                best_eid = other.eid

        return best_eid

    def _get_entity(self, eid: int) -> Entity | None:
        for e in self.entities:
            if e.eid == eid:
                return e
        return None

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _move_toward(self, entity: Entity, tx: float, ty: float, step: float) -> None:
        """Move entity toward (tx, ty) by step tiles, respecting river/bridges."""
        if entity.is_flying:
            dx, dy = direction_to_target(entity.x, entity.y, tx, ty)
            entity.x += dx * step
            entity.y += dy * step
        else:
            need_cross = (
                (entity.owner == 0 and ty > RIVER_ROW_HI)
                or (entity.owner == 1 and ty < RIVER_ROW_LO)
            )
            # Engage bridge-routing only while still on the near bank of the
            # river (and within a few tiles of it). Once the unit steps onto
            # the far bank it stops crossing and heads straight for its target;
            # otherwise it keeps targeting the bridge row it already stands on
            # and stalls forever.
            if entity.owner == 0:
                crossing = RIVER_ROW_LO - 3 <= entity.y <= RIVER_ROW_HI
            else:
                crossing = RIVER_ROW_LO <= entity.y <= RIVER_ROW_HI + 3
            if need_cross and crossing:
                # Aim for the FAR side of the river so the unit walks across.
                far_row = RIVER_ROW_HI + 1 if entity.owner == 0 else RIVER_ROW_LO - 1
                bridge_targets = [
                    (BRIDGE_LEFT_COLS[0], far_row),
                    (BRIDGE_RIGHT_COLS[0], far_row),
                ]
                nearest_bridge = min(
                    bridge_targets,
                    key=lambda b: abs(entity.x - b[0]),
                )
                dx, dy = direction_to_target(
                    entity.x, entity.y,
                    float(nearest_bridge[0]), float(nearest_bridge[1]),
                )
            else:
                dx, dy = direction_to_target(entity.x, entity.y, tx, ty)
            entity.x += dx * step
            entity.y += dy * step
        entity.x = max(0.0, min(float(ARENA_W - 1), entity.x))
        entity.y = max(0.0, min(float(ARENA_H - 1), entity.y))

    def _move_entity(self, entity: Entity) -> None:
        """Move entity toward its target, with charge support.
        
        If no target is in sight_range, the unit moves toward the closest
        enemy tower (princess tower, then king tower). This is the default
        pathing behavior in real CR.
        """
        if not entity.active or entity.speed <= 0 or entity.is_building:
            return

        target = self._get_entity(entity.target_eid)
        if target is None:
            # No target in sight_range: walk toward closest enemy tower
            entity.is_charging = False
            enemy = 1 - entity.owner
            # Find closest alive enemy tower
            best_tower = None
            best_dist = float("inf")
            for pt in self.princess_towers[enemy]:
                if pt.alive:
                    d = entity.distance_to(pt)
                    if d < best_dist:
                        best_dist = d
                        best_tower = pt
            # If no princess towers, go for king
            if best_tower is None and self.king_towers[enemy] and self.king_towers[enemy].alive:
                best_tower = self.king_towers[enemy]
            if best_tower is None:
                return  # No valid destination
            # Move toward tower
            move_speed = entity.effective_speed
            step = move_speed * TICK_DURATION
            self._move_toward(entity, best_tower.x, best_tower.y, step)
            return

        dist = entity.distance_to(target)
        if dist <= entity.attack_range:
            # In range: if charging, next hit is charge hit
            if entity.is_charging:
                entity.next_hit_is_charge = True
                entity.is_charging = False
                entity.charge_distance = 0.0
            return

        # Charge mechanic: start charging after traveling 2+ tiles
        if entity.charge_speed > 0 and not entity.is_charging:
            entity.charge_distance += entity.effective_speed * TICK_DURATION
            if entity.charge_distance > 2.0:
                entity.is_charging = True

        move_speed = entity.effective_speed
        step = move_speed * TICK_DURATION
        self._move_toward(entity, target.x, target.y, step)

    # ------------------------------------------------------------------
    # Collision Resolution
    # ------------------------------------------------------------------

    def _resolve_collisions(self) -> None:
        """Push overlapping units apart based on mass differential.
        
        In real CR, units with higher mass push lighter units when they collide.
        The push is proportional to the mass ratio. Flying units don't collide
        with ground units.
        """
        active_entities = [
            e for e in self.entities
            if e.alive and e.is_deployed and e.speed > 0 and not e.is_building
        ]
        n = len(active_entities)
        for i in range(n):
            a = active_entities[i]
            for j in range(i + 1, n):
                b = active_entities[j]
                # Skip if one is flying and the other isn't
                if a.is_flying != b.is_flying:
                    continue
                # Check overlap
                min_dist = a.collision_radius + b.collision_radius
                dx = b.x - a.x
                dy = b.y - a.y
                dist_sq = dx * dx + dy * dy
                if dist_sq >= min_dist * min_dist:
                    continue
                if dist_sq < 0.001:
                    # Exact overlap: nudge apart
                    dx, dy = 0.1, 0.1
                    dist_sq = 0.02
                dist = math.sqrt(dist_sq)
                overlap = min_dist - dist
                # Push proportional to mass ratio
                total_mass = a.mass + b.mass
                if total_mass <= 0:
                    continue
                push_a = overlap * (b.mass / total_mass)
                push_b = overlap * (a.mass / total_mass)
                # Normalize direction
                nx, ny = dx / dist, dy / dist
                a.x -= nx * push_a
                a.y -= ny * push_a
                b.x += nx * push_b
                b.y += ny * push_b
                # Clamp to arena
                a.x = max(0.0, min(float(ARENA_W - 1), a.x))
                a.y = max(0.0, min(float(ARENA_H - 1), a.y))
                b.x = max(0.0, min(float(ARENA_W - 1), b.x))
                b.y = max(0.0, min(float(ARENA_H - 1), b.y))

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    def _process_combat(self, entity: Entity) -> None:
        """Handle attack logic for a single entity.
        
        Mechanics:
        - Load Time: troops pre-load their attack while moving toward target.
          When they enter attack_range, the pre-loaded time is subtracted from
          the attack timer (so first hit comes faster).
        - Projectile Travel: ranged units fire projectiles that travel at a 
          specific speed. The damage is delayed until impact.
        - Rage: speeds up attack timer countdown by 1.4x.
        """
        if not entity.active or entity.attack_interval <= 0:
            return

        target = self._get_entity(entity.target_eid)
        if target is None or not target.alive:
            return

        # Use attack_range + collision radii for actual attack distance
        effective_range = entity.attack_range + entity.collision_radius + target.collision_radius
        dist = entity.distance_to(target)

        # Minimum range (Mortar/X-Bow): can't hit targets inside the dead zone.
        if entity.minimum_range > 0 and dist < entity.minimum_range:
            return

        if dist > effective_range:
            # Not in range: accumulate load_time if moving toward target
            if entity.load_time > 0 and entity.load_progress < entity.load_time:
                entity.load_progress += TICK_DURATION
                if entity.load_progress > entity.load_time:
                    entity.load_progress = entity.load_time
            return

        # First time entering range: apply load_time reduction to attack_timer
        if entity.load_progress > 0:
            entity.attack_timer -= entity.load_progress
            entity.load_progress = 0.0
            if entity.attack_timer < 0:
                entity.attack_timer = 0.0

        # Rage speeds up attack interval (1.4x attack speed = timer decreases 1.4x faster)
        tick_dt = TICK_DURATION
        if entity.rage_timer > 0:
            tick_dt *= entity.rage_speed_mult
        # Ability attack-speed boost (Archer Queen cloak)
        if entity.attack_speed_timer > 0:
            tick_dt *= entity.attack_speed_mult

        entity.attack_timer -= tick_dt
        if entity.attack_timer <= 0:
            # Compute damage
            if entity.inferno_dps_max > 0:
                entity.inferno_ramp_time += entity.attack_interval
                t = min(entity.inferno_ramp_time / 5.0, 1.0)
                current_dps = (
                    entity.inferno_dps_min
                    + (entity.inferno_dps_max - entity.inferno_dps_min) * t
                )
                dmg = current_dps * entity.attack_interval
            else:
                dmg = entity.damage_per_hit
                # Charge hit deals 2× damage (Prince, Dark Prince, Ram Rider, etc.)
                if entity.next_hit_is_charge:
                    dmg *= 2.0

            card_def = CARD_DEFS.get(entity.card_type)
            stuns = card_def.stuns if card_def else False
            stun_dur = card_def.stun_duration if card_def else 0.0
            resets = card_def.resets_inferno if card_def else False

            # If unit has projectile speed > 0, fire a projectile instead of instant damage
            if entity.has_projectile and entity.projectile_speed > 0:
                self.projectiles.append({
                    'x': entity.x,
                    'y': entity.y,
                    'target_eid': target.eid,
                    'target_x': target.x,
                    'target_y': target.y,
                    'speed': entity.projectile_speed,
                    'damage': dmg,
                    'is_splash': entity.is_splash,
                    'splash_radius': entity.splash_radius,
                    'owner': entity.owner,
                    'stuns': stuns,
                    'stun_duration': stun_dur,
                    'resets_inferno': resets,
                    'crown_pct': entity.crown_tower_damage_percent,
                })
            else:
                # Melee / instant: deal damage immediately
                if stuns and stun_dur > 0:
                    target.apply_stun(stun_dur)
                if resets and target.inferno_dps_max > 0:
                    target.inferno_ramp_time = 0.0

                if entity.is_splash and entity.splash_radius > 0:
                    for other in self.entities:
                        if (
                            other.alive
                            and other.owner != entity.owner
                            and other.distance_to(target) <= entity.splash_radius
                        ):
                            other.take_damage(self._crown_adjusted_entity(entity, dmg, other))
                else:
                    target.take_damage(self._crown_adjusted_entity(entity, dmg, target))

            # Clear charge flag after hit
            entity.next_hit_is_charge = False
            entity.attack_timer = entity.attack_interval

    def _update_projectiles(self) -> None:
        """Advance all in-flight projectiles. Deal damage on impact."""
        surviving = []
        for proj in self.projectiles:
            target = self._get_entity(proj['target_eid'])
            if target is None or not target.alive:
                # Target died mid-flight: projectile disappears (wasted shot)
                continue

            # Move projectile toward target's current position (homing)
            dx = target.x - proj['x']
            dy = target.y - proj['y']
            dist = math.sqrt(dx * dx + dy * dy)
            travel = proj['speed'] * TICK_DURATION

            if dist <= travel:
                # Impact!
                dmg = proj['damage']
                if proj['stuns'] and proj['stun_duration'] > 0:
                    target.apply_stun(proj['stun_duration'])
                if proj['resets_inferno'] and target.inferno_dps_max > 0:
                    target.inferno_ramp_time = 0.0

                pct = proj.get('crown_pct', 0.0)
                if proj['is_splash'] and proj['splash_radius'] > 0:
                    for other in self.entities:
                        if (
                            other.alive
                            and other.owner != proj['owner']
                            and other.distance_to(target) <= proj['splash_radius']
                        ):
                            other.take_damage(self._crown_pct_adjust(other, dmg, pct))
                else:
                    target.take_damage(self._crown_pct_adjust(target, dmg, pct))
            else:
                # Still in flight
                proj['x'] += (dx / dist) * travel
                proj['y'] += (dy / dist) * travel
                surviving.append(proj)

        self.projectiles = surviving

    def _update_bombs(self) -> None:
        """Count down delayed bombs; on expiry deal AoE damage at their spot."""
        surviving = []
        for bomb in self.pending_bombs:
            bomb["timer"] -= TICK_DURATION
            if bomb["timer"] > 0:
                surviving.append(bomb)
                continue
            for other in self.entities:
                if not other.alive or other.owner == bomb["owner"]:
                    continue
                if other.distance_to_pos(bomb["x"], bomb["y"]) <= bomb["radius"]:
                    dmg = bomb["damage"]
                    if other.is_tower:
                        dmg *= 0.35  # spell-style crown-tower reduction
                    other.take_damage(dmg)
        self.pending_bombs = surviving

    # ------------------------------------------------------------------
    # Building decay & spawners
    # ------------------------------------------------------------------

    def _update_buildings(self) -> None:
        for e in self.entities:
            if not e.alive or not e.is_building or e.is_tower:
                continue

            e.building_timer -= TICK_DURATION
            if e.building_timer <= 0:
                e.hp = 0  # building expired
                continue

            # Spawners (Tombstone)
            if e.spawner_interval > 0 and e.card_type == CardType.TOMBSTONE:
                e.spawner_timer -= TICK_DURATION
                if e.spawner_timer <= 0:
                    e.spawner_timer = e.spawner_interval
                    if len(self.entities) < MAX_ENTITIES:
                        skeleton = Entity(
                            eid=self._alloc_eid(),
                            owner=e.owner,
                            card_type=CardType.SKELETON_ARMY,
                            kind=EntityKind.TROOP,
                            x=e.x,
                            y=e.y + (1.0 if e.owner == 0 else -1.0),
                            hp=67,
                            max_hp=67,
                            dps=67,
                            attack_interval=1.0,
                            attack_timer=1.0,
                            attack_range=1.0,
                            target_mode=TargetMode.GROUND,
                            speed=2.5,
                        )
                        self.entities.append(skeleton)

    # ------------------------------------------------------------------
    # Win condition
    # ------------------------------------------------------------------

    def _check_win(self) -> None:
        """Check for game-ending conditions."""
        # King tower destroyed → immediate win
        for p in (0, 1):
            kt = self.king_towers[p]
            if kt is not None and not kt.alive:
                self.result = GameResult.P1_WIN if p == 0 else GameResult.P0_WIN
                self.phase = GamePhase.ENDED
                return

        # In overtime/sudden death the first tower to fall wins instantly
        # (no waiting for the timer): whoever has more crowns ends it now.
        if self.phase in (GamePhase.OVERTIME, GamePhase.SUDDEN_DEATH):
            p0_crowns = self._count_crowns(0)
            p1_crowns = self._count_crowns(1)
            if p0_crowns != p1_crowns:
                self.result = (
                    GameResult.P0_WIN if p0_crowns > p1_crowns else GameResult.P1_WIN
                )
                self.phase = GamePhase.ENDED
                return

        # Time-based phase transitions
        if self.phase == GamePhase.REGULAR:
            if self.tick_count >= REGULAR_TIME_TICKS:
                # Check tower counts
                p0_crowns = self._count_crowns(0)
                p1_crowns = self._count_crowns(1)
                if p0_crowns != p1_crowns:
                    self.result = (
                        GameResult.P0_WIN if p0_crowns > p1_crowns
                        else GameResult.P1_WIN
                    )
                    self.phase = GamePhase.ENDED
                    return
                self.phase = GamePhase.OVERTIME

        if self.phase == GamePhase.OVERTIME:
            if self.tick_count >= REGULAR_TIME_TICKS + OVERTIME_TICKS:
                p0_crowns = self._count_crowns(0)
                p1_crowns = self._count_crowns(1)
                if p0_crowns != p1_crowns:
                    self.result = (
                        GameResult.P0_WIN if p0_crowns > p1_crowns
                        else GameResult.P1_WIN
                    )
                    self.phase = GamePhase.ENDED
                    return
                self.phase = GamePhase.SUDDEN_DEATH

        if self.phase == GamePhase.SUDDEN_DEATH:
            total = REGULAR_TIME_TICKS + OVERTIME_TICKS + SUDDEN_DEATH_TICKS
            if self.tick_count >= total:
                # True tiebreaker: compare remaining tower HP
                p0_hp = sum(
                    e.hp for e in self.entities
                    if e.is_tower and e.owner == 0 and e.alive
                )
                p1_hp = sum(
                    e.hp for e in self.entities
                    if e.is_tower and e.owner == 1 and e.alive
                )
                if p0_hp > p1_hp:
                    self.result = GameResult.P0_WIN
                elif p1_hp > p0_hp:
                    self.result = GameResult.P1_WIN
                else:
                    self.result = GameResult.DRAW
                self.phase = GamePhase.ENDED

    def _count_crowns(self, attacker: int) -> int:
        """Count crowns earned by attacker (= destroyed opponent towers)."""
        opponent = 1 - attacker
        crowns = 0
        kt = self.king_towers[opponent]
        if kt is not None and not kt.alive:
            crowns += 3  # king = instant 3-crown
            return crowns
        for pt in self.princess_towers[opponent]:
            if not pt.alive:
                crowns += 1
        return crowns

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def _tick_timers(self) -> None:
        """Decrement per-entity timers: deploy, stun, slow, freeze, rage, poison."""
        for e in self.entities:
            if not e.alive:
                continue
            # Deploy timer
            if not e.is_deployed:
                e.deploy_timer -= TICK_DURATION
                if e.deploy_timer <= 0:
                    e.is_deployed = True
                    e.deploy_timer = 0.0
            # Stun timer
            if e.stun_timer > 0:
                e.stun_timer -= TICK_DURATION
                if e.stun_timer < 0:
                    e.stun_timer = 0.0
            # Slow timer
            if e.slow_timer > 0:
                e.slow_timer -= TICK_DURATION
                if e.slow_timer < 0:
                    e.slow_timer = 0.0
            # Freeze timer (freeze resets load_progress and inferno ramp)
            if e.freeze_timer > 0:
                e.freeze_timer -= TICK_DURATION
                if e.freeze_timer < 0:
                    e.freeze_timer = 0.0
                    # When freeze ends, load_progress is reset (first hit delay restarts)
                    e.load_progress = 0.0
            # Rage timer
            if e.rage_timer > 0:
                e.rage_timer -= TICK_DURATION
                if e.rage_timer < 0:
                    e.rage_timer = 0.0
            # Poison DoT
            if e.poison_timer > 0:
                poison_dmg = e.poison_dps * TICK_DURATION
                e.take_damage(poison_dmg)
                e.poison_timer -= TICK_DURATION
                if e.poison_timer < 0:
                    e.poison_timer = 0.0
                    e.poison_dps = 0.0
            # Champion ability cooldown / active-effect timers
            if e.ability_cooldown_timer > 0:
                e.ability_cooldown_timer -= TICK_DURATION
                if e.ability_cooldown_timer < 0:
                    e.ability_cooldown_timer = 0.0
            if e.ability_active_timer > 0:
                e.ability_active_timer -= TICK_DURATION
                if e.ability_active_timer < 0:
                    e.ability_active_timer = 0.0
            if e.invisible_timer > 0:
                e.invisible_timer -= TICK_DURATION
                if e.invisible_timer < 0:
                    e.invisible_timer = 0.0
            if e.attack_speed_timer > 0:
                e.attack_speed_timer -= TICK_DURATION
                if e.attack_speed_timer <= 0:
                    e.attack_speed_timer = 0.0
                    e.attack_speed_mult = 1.0
            if e.damage_reduction_timer > 0:
                e.damage_reduction_timer -= TICK_DURATION
                if e.damage_reduction_timer <= 0:
                    e.damage_reduction_timer = 0.0
                    e.damage_reduction_mult = 1.0

    def _process_death_spawns(self) -> None:
        """Handle death effects: death damage + spawn units (Golem→Golemites, etc.)."""
        new_entities: list[Entity] = []
        for e in self.entities:
            if e.alive or e.is_tower:
                continue

            # Death damage (Golem explosion, etc.) — authentic value + radius.
            if e.death_damage > 0:
                death_radius = e.death_damage_radius if e.death_damage_radius > 0 else 2.0
                for other in self.entities:
                    if other.alive and other.owner != e.owner:
                        if other.distance_to(e) <= death_radius:
                            # Death damage is reduced for towers (35%)
                            if other.is_tower:
                                other.take_damage(e.death_damage * 0.35)
                            else:
                                other.take_damage(e.death_damage)

            if e.death_spawn_count > 0 and e.death_spawn_hp > 0:
                for i in range(e.death_spawn_count):
                    if len(self.entities) + len(new_entities) >= MAX_ENTITIES:
                        break
                    angle = 2.0 * math.pi * i / max(e.death_spawn_count, 1)
                    ox = math.cos(angle) * 0.5
                    oy = math.sin(angle) * 0.5
                    spawned = Entity(
                        eid=self._alloc_eid(),
                        owner=e.owner,
                        card_type=e.card_type,
                        kind=EntityKind.TROOP,
                        x=e.x + ox,
                        y=e.y + oy,
                        hp=e.death_spawn_hp,
                        max_hp=e.death_spawn_hp,
                        dps=e.death_spawn_dps,
                        attack_interval=1.5,
                        attack_timer=1.5,
                        attack_range=1.2,
                        target_mode=e.target_mode,
                        speed=e.speed if e.speed > 0 else SPEED_MEDIUM,
                        base_speed=e.speed if e.speed > 0 else SPEED_MEDIUM,
                        is_flying=e.is_flying,
                    )
                    new_entities.append(spawned)
        self.entities.extend(new_entities)

    def _apply_spell_effects(
        self, player: int, card_def: CardDef, x: float, y: float,
    ) -> None:
        """Apply knockback, stun, and inferno reset from spells."""
        for e in self.entities:
            if not e.alive or e.owner == player:
                continue
            dist = e.distance_to_pos(x, y)
            if dist > card_def.attack_range:
                continue
            # Knockback
            if card_def.has_knockback and card_def.knockback_distance > 0:
                if not e.is_flying and not e.is_building:
                    dx = e.x - x
                    dy = e.y - y
                    mag = math.sqrt(dx * dx + dy * dy)
                    if mag > 0.01:
                        e.x += (dx / mag) * card_def.knockback_distance
                        e.y += (dy / mag) * card_def.knockback_distance
                        e.x = max(0.0, min(float(ARENA_W - 1), e.x))
                        e.y = max(0.0, min(float(ARENA_H - 1), e.y))
            # Stun
            if card_def.stuns and card_def.stun_duration > 0:
                e.apply_stun(card_def.stun_duration)
            # Inferno reset
            if card_def.resets_inferno and e.inferno_dps_max > 0:
                e.inferno_ramp_time = 0.0

    def step(self, actions: list[Action]) -> None:
        """Advance the game by one tick (50ms = 0.05s).

        Parameters
        ----------
        actions : list[Action]
            One action per player (or empty for no actions this tick).
        """
        if self.phase == GamePhase.ENDED:
            return

        # 1. Elixir regen
        self._regen_elixir()

        # 2. Apply player actions
        for a in actions:
            self.apply_action(a)

        # 3. Tick timers (deploy, stun, slow)
        self._tick_timers()

        # 4. Update buildings (decay, spawners)
        self._update_buildings()

        # 5. For each entity: find target, move, attack
        for entity in self.entities:
            if not entity.alive:
                continue
            entity.target_eid = self._find_target(entity)

        for entity in self.entities:
            if not entity.alive:
                continue
            self._move_entity(entity)

        # 5b. Resolve collisions (push mechanics based on mass)
        self._resolve_collisions()

        for entity in self.entities:
            if not entity.alive:
                continue
            self._process_combat(entity)

        # 5c. Update in-flight projectiles
        self._update_projectiles()

        # 5d. Update delayed area explosions (Mighty Miner bomb)
        self._update_bombs()

        # 6. Process death spawns before removing dead entities
        self._process_death_spawns()

        # 7. Remove dead entities (keep towers for crown tracking)
        self.entities = [
            e for e in self.entities if e.alive or e.is_tower
        ]

        # Invalidate flow cache if buildings died
        self.flow_cache.invalidate()

        # 8. Advance tick and check win
        self.tick_count += 1
        self._check_win()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.phase == GamePhase.ENDED

    def get_reward(self, player: int) -> float:
        """Terminal reward for the given player: +1 win, -1 loss, 0 draw/ongoing."""
        if self.result == GameResult.IN_PROGRESS:
            return 0.0
        if self.result == GameResult.DRAW:
            return 0.0
        if (player == 0 and self.result == GameResult.P0_WIN) or (
            player == 1 and self.result == GameResult.P1_WIN
        ):
            return 1.0
        return -1.0

    def clone(self) -> CRGame:
        """Deep copy the game state for MCTS simulation."""
        import copy
        return copy.deepcopy(self)
