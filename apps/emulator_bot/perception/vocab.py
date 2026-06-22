"""Canonical Clash Royale card vocabulary (the 125 cards from the dataset).

Every perception backend normalizes its own label set to THIS vocabulary, so
downstream consumers (RL agent, analytics, real-device play) get one stable
card namespace regardless of which detection model produced the labels.
"""

# The 125 canonical cards (order = card_id), mirrored from AlphaRoyale-Zero's
# crsim.cards.CardType so the id matches the simulator's CardType value.
CANONICAL = [
    'KNIGHT', 'ARCHERS', 'GOBLINS', 'GIANT', 'PEKKA', 'MINIONS', 'BALLOON',
    'WITCH', 'BARBARIANS', 'GOLEM', 'SKELETONS', 'VALKYRIE', 'SKELETON_ARMY',
    'BOMBER', 'MUSKETEER', 'BABY_DRAGON', 'PRINCE', 'WIZARD', 'MINI_PEKKA',
    'SPEAR_GOBLINS', 'GIANT_SKELETON', 'HOG_RIDER', 'MINION_HORDE', 'ICE_WIZARD',
    'ROYAL_GIANT', 'GUARDS', 'PRINCESS', 'DARK_PRINCE', 'THREE_MUSKETEERS',
    'LAVA_HOUND', 'ICE_SPIRIT', 'FIRE_SPIRIT', 'MINER', 'SPARKY', 'BOWLER',
    'LUMBERJACK', 'BATTLE_RAM', 'INFERNO_DRAGON', 'ICE_GOLEM', 'MEGA_MINION',
    'DART_GOBLIN', 'GOBLIN_GANG', 'ELECTRO_WIZARD', 'ELITE_BARBARIANS', 'HUNTER',
    'EXECUTIONER', 'BANDIT', 'ROYAL_RECRUITS', 'NIGHT_WITCH', 'BATS',
    'ROYAL_GHOST', 'RAM_RIDER', 'ZAPPIES', 'RASCALS', 'CANNON_CART',
    'MEGA_KNIGHT', 'SKELETON_BARREL', 'FLYING_MACHINE', 'WALL_BREAKERS',
    'ROYAL_HOGS', 'GOBLIN_GIANT', 'FISHERMAN', 'MAGIC_ARCHER', 'ELECTRO_DRAGON',
    'FIRECRACKER', 'MIGHTY_MINER', 'ELIXIR_GOLEM', 'BATTLE_HEALER',
    'SKELETON_KING', 'ARCHER_QUEEN', 'GOLDEN_KNIGHT', 'MONK', 'SKELETON_DRAGONS',
    'MOTHER_WITCH', 'ELECTRO_SPIRIT', 'ELECTRO_GIANT', 'PHOENIX', 'LITTLE_PRINCE',
    'GOBLIN_DEMOLISHER', 'GOBLIN_MACHINE', 'SUSPICIOUS_BUSH', 'GOBLINSTEIN',
    'RUNE_GIANT', 'BERSERKER', 'BOSS_BANDIT', 'CANNON', 'GOBLIN_HUT', 'MORTAR',
    'INFERNO_TOWER', 'BOMB_TOWER', 'BARBARIAN_HUT', 'TESLA', 'ELIXIR_COLLECTOR',
    'X_BOW', 'TOMBSTONE', 'FURNACE', 'GOBLIN_CAGE', 'GOBLIN_DRILL', 'FIREBALL',
    'ARROWS', 'RAGE', 'ROCKET', 'GOBLIN_BARREL', 'FREEZE', 'MIRROR', 'LIGHTNING',
    'ZAP', 'POISON', 'GRAVEYARD', 'THE_LOG', 'TORNADO', 'CLONE', 'EARTHQUAKE',
    'BARBARIAN_BARREL', 'HEAL_SPIRIT', 'GIANT_SNOWBALL', 'ROYAL_DELIVERY',
    'VOID', 'GOBLIN_CURSE', 'SPIRIT_EMPRESS', 'VINES', 'GOBLIN_BRAWLER',
    'BUSH_GOBLINS', 'CURSED_HOG', 'GUARDIENNE',
]
assert len(CANONICAL) == 125, len(CANONICAL)
_CANON_SET = set(CANONICAL)
_ID = {n: i for i, n in enumerate(CANONICAL)}


def _key(s: str) -> str:
    """Collapse a label to alphanumerics, dropping evolution/skin suffixes."""
    k = ''.join(ch for ch in str(s).lower() if ch.isalnum())
    for suf in ('ev1', 'ev2', 'evolution', 'evolved', 'evo'):
        k = k.replace(suf, '')
    return k


_NORM = {_key(n): n for n in CANONICAL}
# A few label variants seen across detectors (BuildABot / KataCR / wiki).
_ALIASES = {
    'log': 'THE_LOG', 'minipekka': 'MINI_PEKKA', 'pekka': 'PEKKA',
    'xbow': 'X_BOW', 'pekkas': 'PEKKA', 'iceswizard': 'ICE_WIZARD',
    'skeletondragon': 'SKELETON_DRAGONS', 'minionhorde': 'MINION_HORDE',
}
for k, v in _ALIASES.items():
    _NORM.setdefault(k, v)


def normalize(name) -> str | None:
    """Map any detector label to a canonical card name, or None if unknown."""
    if not name:
        return None
    return _NORM.get(_key(name))


def card_id(canonical_name) -> int:
    """Stable id for a canonical name (matches the simulator's CardType), or -1."""
    return _ID.get(canonical_name, -1)
