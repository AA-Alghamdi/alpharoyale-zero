"""Adapter: wraps the Rust CREngine to match crsim.game.CRGame interface.

Drop-in replacement for CRGame in the RL pipeline. Converts between
Rust millitile coordinates and Python tile coordinates, and exposes
the same entity/player/game interface that features.py, self_play_v2.py,
and gumbel_search.py expect.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass

import numpy as np

from crsim.cards import CardType, EntityKind, TargetMode, CARD_DEFS
from crsim.constants import (
    ARENA_H, ARENA_W, NUM_HAND_SLOTS, ACTION_SPACE_SIZE, WAIT_ACTION,
    KING_TOWER_HP, PRINCESS_TOWER_HP, TOTAL_MAX_TICKS,
    RIVER_ROW_LO, RIVER_ROW_HI,
)
from crsim.entities import Entity
from crsim.game import GamePhase, GameResult, PlayerState, Action

# Mapping from Python CardType enum to Rust engine card names.
# Maps to spells_characters.csv Name column for troops/buildings,
# or spells_other.csv for instant spells.
_CARD_TO_RUST: dict[CardType, str] = {
    # --- Troops ---
    CardType.KNIGHT: "Knight",
    CardType.ARCHERS: "Archers",
    CardType.GOBLINS: "Goblins",
    CardType.GIANT: "Giant",
    CardType.PEKKA: "Pekka",
    CardType.MINIONS: "Minions",
    CardType.BALLOON: "Balloon",
    CardType.WITCH: "Witch",
    CardType.BARBARIANS: "Barbarians",
    CardType.GOLEM: "Golem",
    CardType.SKELETONS: "Skeletons",
    CardType.VALKYRIE: "Valkyrie",
    CardType.SKELETON_ARMY: "SkeletonArmy",
    CardType.BOMBER: "Bomber",
    CardType.MUSKETEER: "Musketeer",
    CardType.BABY_DRAGON: "BabyDragon",
    CardType.PRINCE: "Prince",
    CardType.WIZARD: "Wizard",
    CardType.MINI_PEKKA: "MiniPekka",
    CardType.SPEAR_GOBLINS: "SpearGoblins",
    CardType.GIANT_SKELETON: "GiantSkeleton",
    CardType.HOG_RIDER: "HogRider",
    CardType.MINION_HORDE: "MinionHorde",
    CardType.ICE_WIZARD: "IceWizard",
    CardType.ROYAL_GIANT: "RoyalGiant",
    CardType.GUARDS: "Guards",
    CardType.PRINCESS: "Princess",
    CardType.DARK_PRINCE: "DarkPrince",
    CardType.THREE_MUSKETEERS: "ThreeMusketeers",
    CardType.LAVA_HOUND: "LavaHound",
    CardType.ICE_SPIRIT: "IceSpirit",
    CardType.FIRE_SPIRIT: "FireSpirit",
    CardType.MINER: "Miner",
    CardType.SPARKY: "Sparky",
    CardType.BOWLER: "Bowler",
    CardType.LUMBERJACK: "Lumberjack",
    CardType.BATTLE_RAM: "BattleRam",
    CardType.INFERNO_DRAGON: "InfernoDragon",
    CardType.ICE_GOLEM: "IceGolem",
    CardType.MEGA_MINION: "MegaMinion",
    CardType.DART_GOBLIN: "DartGoblin",
    CardType.GOBLIN_GANG: "GoblinGang",
    CardType.ELECTRO_WIZARD: "ElectroWizard",
    CardType.ELITE_BARBARIANS: "EliteBarbarians",
    CardType.HUNTER: "Hunter",
    CardType.EXECUTIONER: "Executioner",
    CardType.BANDIT: "Bandit",
    CardType.ROYAL_RECRUITS: "RoyalRecruits",
    CardType.NIGHT_WITCH: "NightWitch",
    CardType.BATS: "Bats",
    CardType.ROYAL_GHOST: "RoyalGhost",
    CardType.RAM_RIDER: "RamRider",
    CardType.ZAPPIES: "Zappies",
    CardType.RASCALS: "Rascals",
    CardType.CANNON_CART: "CannonCart",
    CardType.MEGA_KNIGHT: "MegaKnight",
    CardType.SKELETON_BARREL: "SkeletonBarrel",
    CardType.FLYING_MACHINE: "FlyingMachine",
    CardType.WALL_BREAKERS: "WallBreakers",
    CardType.ROYAL_HOGS: "RoyalHogs",
    CardType.GOBLIN_GIANT: "GoblinGiant",
    CardType.FISHERMAN: "Fisherman",
    CardType.MAGIC_ARCHER: "MagicArcher",
    CardType.ELECTRO_DRAGON: "ElectroDragon",
    CardType.FIRECRACKER: "Firecracker",
    CardType.MIGHTY_MINER: "MightyMiner",
    CardType.ELIXIR_GOLEM: "ElixirGolem",
    CardType.BATTLE_HEALER: "BattleHealer",
    CardType.SKELETON_KING: "SkeletonKing",
    CardType.ARCHER_QUEEN: "ArcherQueen",
    CardType.GOLDEN_KNIGHT: "GoldenKnight",
    CardType.MONK: "Monk",
    CardType.SKELETON_DRAGONS: "SkeletonDragons",
    CardType.MOTHER_WITCH: "MotherWitch",
    CardType.ELECTRO_SPIRIT: "ElectroSpirit",
    CardType.ELECTRO_GIANT: "ElectroGiant",
    CardType.PHOENIX: "Phoenix",
    CardType.LITTLE_PRINCE: "LittlePrince",
    CardType.GOBLIN_DEMOLISHER: "GoblinDemolisher",
    CardType.GOBLIN_MACHINE: "GoblinMachine",
    CardType.SUSPICIOUS_BUSH: "SuspiciousBush",
    CardType.GOBLINSTEIN: "Goblinstein",
    CardType.RUNE_GIANT: "RuneGiant",
    CardType.BERSERKER: "Berserker",
    CardType.BOSS_BANDIT: "BossBandit",
    # --- Buildings ---
    CardType.CANNON: "Cannon",
    CardType.GOBLIN_HUT: "Goblin Hut",
    CardType.MORTAR: "Mortar",
    CardType.INFERNO_TOWER: "Inferno Tower",
    CardType.BOMB_TOWER: "Bomb Tower",
    CardType.BARBARIAN_HUT: "Barbarian Hut",
    CardType.TESLA: "Tesla",
    CardType.ELIXIR_COLLECTOR: "Elixir Collector",
    CardType.X_BOW: "Xbow",
    CardType.TOMBSTONE: "Tombstone",
    CardType.FURNACE: "Furnace",
    CardType.GOBLIN_CAGE: "Goblin Cage",
    CardType.GOBLIN_DRILL: "Goblin Drill",
    # --- Spells ---
    CardType.FIREBALL: "Fireball",
    CardType.ARROWS: "Arrows",
    CardType.RAGE: "Rage",
    CardType.ROCKET: "Rocket",
    CardType.GOBLIN_BARREL: "GoblinBarrel",
    CardType.FREEZE: "Freeze",
    CardType.MIRROR: "Mirror",
    CardType.LIGHTNING: "Lightning",
    CardType.ZAP: "Zap",
    CardType.POISON: "Poison",
    CardType.GRAVEYARD: "Graveyard",
    CardType.THE_LOG: "Log",
    CardType.TORNADO: "Tornado",
    CardType.CLONE: "Clone",
    CardType.EARTHQUAKE: "Earthquake",
    CardType.BARBARIAN_BARREL: "BarbarianBarrel",
    CardType.HEAL_SPIRIT: "HealSpirit",
    CardType.GIANT_SNOWBALL: "GiantSnowball",
    CardType.ROYAL_DELIVERY: "RoyalDelivery",
    CardType.VOID: "Void",
    CardType.GOBLIN_CURSE: "GoblinCurse",
    CardType.SPIRIT_EMPRESS: "SpiritEmpress",
    CardType.VINES: "Vines",
}

# Reverse mapping: Rust name → CardType
_RUST_TO_CARD: dict[str, CardType] = {v: k for k, v in _CARD_TO_RUST.items()}
# Add tower names
_RUST_TO_CARD["KingTower"] = CardType.KNIGHT  # placeholder, towers use is_tower flag
_RUST_TO_CARD["PrincessTower"] = CardType.KNIGHT

# Millitiles per tile
_MT = 1000


def _mt_to_tile(mt: int) -> float:
    """Convert millitiles to tiles."""
    return mt / _MT


def _tile_to_mt(tile: float) -> int:
    """Convert tiles to millitiles."""
    return int(tile * _MT)


class CRGameRust:
    """Drop-in replacement for CRGame backed by the Rust cr_engine.

    Matches the CRGame interface used by:
      - model/features.py (encode_state, extract_entity_features, extract_auxiliary_targets)
      - mcts/gumbel_search.py (clone, step, get_valid_actions_mask, get_reward)
      - training/self_play_v2.py (done, phase, tick_count, players)
    """

    def __init__(
        self,
        deck_p0: list[CardType] | None = None,
        deck_p1: list[CardType] | None = None,
        seed: int | None = None,
        data_dir: str = "cr_engine/gamedata",
        *,
        _engine=None,
        _decks=None,
    ) -> None:
        from cr_engine import CREngine

        rng = random.Random(seed)

        if deck_p0 is None:
            deck_p0 = [
                CardType.KNIGHT, CardType.ARCHERS, CardType.FIREBALL,
                CardType.GIANT, CardType.MUSKETEER, CardType.VALKYRIE,
                CardType.ARROWS, CardType.MINI_PEKKA,
            ]
        if deck_p1 is None:
            deck_p1 = list(deck_p0)

        rng.shuffle(deck_p0)
        rng.shuffle(deck_p1)

        self._deck_p0 = deck_p0
        self._deck_p1 = deck_p1
        self._data_dir = data_dir

        if _engine is not None:
            self._engine = _engine
        else:
            rust_deck0 = [_CARD_TO_RUST[c] for c in deck_p0]
            rust_deck1 = [_CARD_TO_RUST[c] for c in deck_p1]
            self._engine = CREngine(data_dir, rust_deck0, rust_deck1)

        # Build Python player state objects
        self.players: list[PlayerState] = [
            PlayerState(deck=deck_p0, hand=list(range(4)), next_card_idx=4),
            PlayerState(deck=deck_p1, hand=list(range(4)), next_card_idx=4),
        ]

        # Tower tracking (populated on demand)
        self.king_towers: list[Entity | None] = [None, None]
        self.princess_towers: list[list[Entity]] = [[], []]
        self._entities_dirty = True
        self._entities_cache: list[Entity] = []
        self._sync_elixir()

    def _sync_elixir(self) -> None:
        """Sync only elixir (fast, called every tick)."""
        for pid in range(2):
            self.players[pid].elixir = self._engine.get_elixir(pid)

    def _sync_hand(self) -> None:
        """Sync hand from Rust engine."""
        for pid in range(2):
            hand = self._engine.get_hand(pid)
            deck = self._deck_p0 if pid == 0 else self._deck_p1
            rust_names = {_CARD_TO_RUST.get(c, ""): i for i, c in enumerate(deck)}
            for slot_idx, card_name in enumerate(hand):
                if card_name in rust_names:
                    self.players[pid].hand[slot_idx] = rust_names[card_name]

    @property
    def entities(self) -> list[Entity]:
        if self._entities_dirty:
            self._rebuild_entities()
            self._entities_dirty = False
        return self._entities_cache

    def _rebuild_entities(self) -> None:
        """Build Python Entity objects from Rust engine state."""
        self._entities_cache = []
        self.king_towers = [None, None]
        self.princess_towers = [[], []]

        # Get tower HP
        for pid in range(2):
            hp_list = self._engine.get_tower_hp(pid)
            king_hp, lp_hp, rp_hp = hp_list[0], hp_list[1], hp_list[2]

            king_x = 9.0
            king_y = 3.0 if pid == 0 else 29.0
            kt = Entity(
                eid=pid * 3,
                owner=pid,
                card_type=CardType.KNIGHT,
                kind=EntityKind.TROOP,
                x=king_x, y=king_y,
                hp=float(king_hp), max_hp=float(2400),
                dps=109.0, attack_interval=1.0, attack_timer=0.0,
                attack_range=7.0, target_mode=TargetMode.AIR_GROUND,
                speed=0.0,
                is_tower=True, is_king_tower=True,
                is_building=True,
            )
            self.king_towers[pid] = kt
            self._entities_cache.append(kt)

            lp_x, lp_y = (3.5, 6.5) if pid == 0 else (3.5, 25.5)
            lp = Entity(
                eid=pid * 3 + 1,
                owner=pid,
                card_type=CardType.KNIGHT,
                kind=EntityKind.TROOP,
                x=lp_x, y=lp_y,
                hp=float(lp_hp), max_hp=float(1400),
                dps=109.0, attack_interval=1.0, attack_timer=0.0,
                attack_range=7.5, target_mode=TargetMode.AIR_GROUND,
                speed=0.0,
                is_tower=True, is_king_tower=False,
                is_building=True,
            )
            if lp_hp > 0:
                self.princess_towers[pid].append(lp)
            self._entities_cache.append(lp)

            rp_x, rp_y = (14.5, 6.5) if pid == 0 else (14.5, 25.5)
            rp = Entity(
                eid=pid * 3 + 2,
                owner=pid,
                card_type=CardType.KNIGHT,
                kind=EntityKind.TROOP,
                x=rp_x, y=rp_y,
                hp=float(rp_hp), max_hp=float(1400),
                dps=109.0, attack_interval=1.0, attack_timer=0.0,
                attack_range=7.5, target_mode=TargetMode.AIR_GROUND,
                speed=0.0,
                is_tower=True, is_king_tower=False,
                is_building=True,
            )
            if rp_hp > 0:
                self.princess_towers[pid].append(rp)
            self._entities_cache.append(rp)

        # Get troops from engine: (id, player_id, x, y, hp, max_hp, name, is_air)
        for eid, player_id, x, y, hp, max_hp, name, is_air in self._engine.get_entities():
            card_type = _RUST_TO_CARD.get(name, CardType.KNIGHT)
            card_def = CARD_DEFS.get(card_type)

            e = Entity(
                eid=eid,
                owner=player_id,
                card_type=card_type,
                kind=EntityKind.TROOP,
                x=_mt_to_tile(x), y=_mt_to_tile(y),
                hp=float(hp), max_hp=float(max_hp),
                dps=card_def.dps if card_def else 0.0,
                attack_interval=card_def.attack_interval if card_def else 1.0,
                attack_timer=0.0,
                attack_range=card_def.attack_range if card_def else 1.0,
                target_mode=card_def.target_mode if card_def else TargetMode.GROUND,
                speed=card_def.speed if card_def else 0.0,
                is_flying=is_air,
                is_splash=card_def.is_splash if card_def else False,
            )
            self._entities_cache.append(e)

    @property
    def tick_count(self) -> int:
        # Rust runs at 20 ticks/sec (50ms), Python at 2 ticks/sec (500ms)
        # Scale so feature encoder gets correct time fraction
        return self._engine.tick // 10

    @property
    def phase(self) -> GamePhase:
        tick = self._engine.tick
        if self._engine.game_over:
            return GamePhase.ENDED
        if tick >= 4800:  # > 4 min
            return GamePhase.SUDDEN_DEATH
        if tick >= 3600:  # > 3 min
            return GamePhase.OVERTIME
        return GamePhase.REGULAR

    @property
    def done(self) -> bool:
        return self._engine.game_over

    @property
    def result(self) -> GameResult:
        if not self._engine.game_over:
            return GameResult.IN_PROGRESS
        w = self._engine.winner
        if w is None:
            return GameResult.DRAW
        if w == 0:
            return GameResult.P0_WIN
        return GameResult.P1_WIN

    def get_reward(self, player: int) -> float:
        r = self.result
        if r == GameResult.IN_PROGRESS or r == GameResult.DRAW:
            return 0.0
        if (player == 0 and r == GameResult.P0_WIN) or (player == 1 and r == GameResult.P1_WIN):
            return 1.0
        return -1.0

    def get_valid_actions_mask(self, player: int) -> np.ndarray:
        """Return boolean mask of shape (ACTION_SPACE_SIZE,)."""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[WAIT_ACTION] = True

        self._sync_hand()
        ps = self.players[player]
        elixir = self._engine.get_elixir(player)

        for slot_idx in range(NUM_HAND_SLOTS):
            card_idx = ps.hand[slot_idx]
            card_type = ps.deck[card_idx]
            card_def = CARD_DEFS.get(card_type)
            if card_def is None:
                continue
            if elixir < card_def.cost:
                continue

            for x in range(ARENA_W):
                for y in range(ARENA_H):
                    if self._is_valid_placement(player, card_def, x, y):
                        action_id = slot_idx * ARENA_W * ARENA_H + x * ARENA_H + y
                        mask[action_id] = True

        return mask

    def _is_valid_placement(self, player, card_def, x, y):
        if x < 0 or x >= ARENA_W or y < 0 or y >= ARENA_H:
            return False
        if card_def.kind == EntityKind.SPELL and card_def.card_type != CardType.GOBLIN_BARREL:
            return True
        if card_def.card_type == CardType.GOBLIN_BARREL:
            return True
        if player == 0:
            if y >= RIVER_ROW_LO:
                return False
        else:
            if y <= RIVER_ROW_HI:
                return False
        if RIVER_ROW_LO <= y <= RIVER_ROW_HI:
            return False
        return True

    def apply_action(self, action: Action) -> None:
        if action.is_wait:
            return
        self._engine.play_card(
            action.player,
            action.hand_slot,
            _tile_to_mt(action.x),
            _tile_to_mt(action.y),
        )

    def step(self, actions: list[Action] | None = None) -> None:
        """Advance one game tick (maps to 10 Rust ticks to match Python timing)."""
        if actions:
            for a in actions:
                self.apply_action(a)
        self._engine.step_n(10)
        self._entities_dirty = True
        self._sync_elixir()

    def clone(self) -> CRGameRust:
        """Deep copy for MCTS simulation — uses Rust's fast clone."""
        cloned = CRGameRust.__new__(CRGameRust)
        cloned._engine = self._engine.clone_state()
        cloned._deck_p0 = list(self._deck_p0)
        cloned._deck_p1 = list(self._deck_p1)
        cloned._data_dir = self._data_dir
        cloned.players = [
            PlayerState(
                deck=list(self.players[0].deck),
                hand=list(self.players[0].hand),
                next_card_idx=self.players[0].next_card_idx,
                elixir=self.players[0].elixir,
            ),
            PlayerState(
                deck=list(self.players[1].deck),
                hand=list(self.players[1].hand),
                next_card_idx=self.players[1].next_card_idx,
                elixir=self.players[1].elixir,
            ),
        ]
        cloned.king_towers = [None, None]
        cloned.princess_towers = [[], []]
        cloned._entities_dirty = True
        cloned._entities_cache = []
        cloned._sync_elixir()
        return cloned

    def _count_crowns(self, player: int) -> int:
        """Count destroyed towers for a player."""
        opponent = 1 - player
        hp = self._engine.get_tower_hp(opponent)
        crowns = 0
        if hp[0] <= 0:
            crowns = 3  # king tower destroyed = 3 crowns
        else:
            if hp[1] <= 0:
                crowns += 1
            if hp[2] <= 0:
                crowns += 1
        return crowns

    def _get_entity(self, eid: int):
        """Find an entity by eid."""
        for e in self.entities:
            if e.eid == eid:
                return e
        return None
