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
    """A player action: place a card or wait."""

    player: int
    hand_slot: int = -1  # -1 = wait
    x: float = 0.0
    y: float = 0.0

    @property
    def is_wait(self) -> bool:
        return self.hand_slot < 0


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
        from crsim.constants import ACTION_SPACE_SIZE, WAIT_ACTION

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[WAIT_ACTION] = True  # wait is always valid

        ps = self.players[player]
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

    def _spawn_card(
        self, player: int, card_def: CardDef, x: float, y: float
    ) -> None:
        """Spawn entities for a played card."""
        # Pure damage spells
        if card_def.kind == EntityKind.SPELL and card_def.spawn_count <= 0:
            self._apply_spell(player, card_def, x, y)
            return

        if card_def.kind == EntityKind.SPELL and card_def.spawn_count > 0:
            # Goblin barrel: spell that spawns units at target
            self._apply_spell_spawn(player, card_def, x, y)
            return

        # Multi-spawn troops (Archers, Minions, Skeleton Army)
        count = card_def.spawn_count
        if count > 1:
            hp = card_def.spawn_hp
            dps = card_def.spawn_dps
        else:
            hp = card_def.hp
            dps = card_def.dps

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
            self.entities.append(e)

        # Invalidate flow cache if a building was placed
        if card_def.kind == EntityKind.BUILDING:
            self.flow_cache.invalidate()

    def _apply_spell(
        self, player: int, card_def: CardDef, x: float, y: float
    ) -> None:
        """Apply area-of-effect spell damage."""
        radius = card_def.attack_range
        damage = card_def.dps  # total damage stored in dps field for spells

        if card_def.card_type == CardType.LIGHTNING:
            # Lightning hits 3 highest-HP enemies in radius
            enemies = [
                e for e in self.entities
                if e.alive and e.owner != player
                and e.distance_to_pos(x, y) <= radius
            ]
            enemies.sort(key=lambda e: e.hp, reverse=True)
            for e in enemies[:3]:
                e.hp -= damage
        else:
            # Area damage
            for e in self.entities:
                if e.alive and e.owner != player:
                    if e.distance_to_pos(x, y) <= radius:
                        e.hp -= damage

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

    def _find_target(self, entity: Entity) -> int:
        """Find the best target eid for an entity. Returns -1 if none."""
        best_eid = -1
        best_dist = float("inf")

        for other in self.entities:
            if not other.alive or other.owner == entity.owner:
                continue

            # Target filtering
            if entity.target_mode == TargetMode.BUILDINGS:
                if not other.is_building and not other.is_tower:
                    continue
            elif entity.target_mode == TargetMode.GROUND:
                if other.is_flying:
                    continue

            # King tower only activates after a princess tower dies or
            # it is directly attacked (simplified: always targetable)
            dist = entity.distance_to(other)
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

    def _move_entity(self, entity: Entity) -> None:
        """Move entity toward its target."""
        if entity.speed <= 0 or entity.is_building:
            return

        target = self._get_entity(entity.target_eid)
        if target is None:
            return

        dist = entity.distance_to(target)
        if dist <= entity.attack_range:
            return  # in range, don't move

        step = entity.speed * TICK_DURATION

        if entity.is_flying:
            # Straight-line movement
            dx, dy = direction_to_target(entity.x, entity.y, target.x, target.y)
            entity.x += dx * step
            entity.y += dy * step
        else:
            # Ground movement via flow field
            # Determine if we need to cross the river
            need_cross = (
                (entity.owner == 0 and target.y > RIVER_ROW_HI)
                or (entity.owner == 1 and target.y < RIVER_ROW_LO)
            )

            if need_cross and RIVER_ROW_LO <= int(entity.y + 0.5) <= RIVER_ROW_HI + 1:
                # Navigate to nearest bridge
                bridge_targets = [
                    (BRIDGE_LEFT_COLS[0], RIVER_ROW_LO if entity.owner == 0 else RIVER_ROW_HI),
                    (BRIDGE_RIGHT_COLS[0], RIVER_ROW_LO if entity.owner == 0 else RIVER_ROW_HI),
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
                dx, dy = direction_to_target(
                    entity.x, entity.y, target.x, target.y,
                )

            entity.x += dx * step
            entity.y += dy * step

        # Clamp to arena
        entity.x = max(0.0, min(float(ARENA_W - 1), entity.x))
        entity.y = max(0.0, min(float(ARENA_H - 1), entity.y))

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    def _process_combat(self, entity: Entity) -> None:
        """Handle attack logic for a single entity."""
        if entity.attack_interval <= 0:
            return

        target = self._get_entity(entity.target_eid)
        if target is None or not target.alive:
            return

        dist = entity.distance_to(target)
        if dist > entity.attack_range + 0.5:  # small tolerance
            return

        entity.attack_timer -= TICK_DURATION
        if entity.attack_timer <= 0:
            # Compute damage
            if entity.inferno_dps_max > 0:
                # Inferno tower ramp-up
                entity.inferno_ramp_time += entity.attack_interval
                t = min(entity.inferno_ramp_time / 5.0, 1.0)  # 5s to full ramp
                current_dps = (
                    entity.inferno_dps_min
                    + (entity.inferno_dps_max - entity.inferno_dps_min) * t
                )
                dmg = current_dps * entity.attack_interval
            else:
                dmg = entity.damage_per_hit

            if entity.is_splash and entity.splash_radius > 0:
                # Splash damage to all enemies near target
                for other in self.entities:
                    if (
                        other.alive
                        and other.owner != entity.owner
                        and other.distance_to(target) <= entity.splash_radius
                    ):
                        other.hp -= dmg
            else:
                target.hp -= dmg

            entity.attack_timer = entity.attack_interval

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

    def step(self, actions: list[Action]) -> None:
        """Advance the game by one tick (0.5 s).

        Parameters
        ----------
        actions : list[Action]
            One action per player (len == 2).
        """
        if self.phase == GamePhase.ENDED:
            return

        # 1. Elixir regen
        self._regen_elixir()

        # 2. Apply player actions
        for a in actions:
            self.apply_action(a)

        # 3. Update buildings (decay, spawners)
        self._update_buildings()

        # 4. For each entity: find target, move, attack
        for entity in self.entities:
            if not entity.alive:
                continue
            entity.target_eid = self._find_target(entity)

        for entity in self.entities:
            if not entity.alive:
                continue
            self._move_entity(entity)

        for entity in self.entities:
            if not entity.alive:
                continue
            self._process_combat(entity)

        # 5. Remove dead entities (keep towers for crown tracking)
        self.entities = [
            e for e in self.entities if e.alive or e.is_tower
        ]

        # Invalidate flow cache if buildings died
        self.flow_cache.invalidate()

        # 6. Advance tick and check win
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
