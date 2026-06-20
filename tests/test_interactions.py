"""Golden card-interaction tests (the verification harness).

These tests encode *known* Clash Royale outcomes (spell kill thresholds, 1v1
matchups, tower mechanics) and assert the simulator reproduces them. They are
the ground-truth gate the project rested on but never had: an RL agent trained
on a sim that fails these will learn to exploit sim bugs, not play Clash Royale.

Expected outcomes are sourced from live-game balance at tournament standard
(Level 11) card stats, which is what ``crsim/cards.py`` encodes.
"""

from __future__ import annotations

import pytest

from crsim.cards import CARD_DEFS, CardType
from crsim.entities import Entity, entity_from_card
from crsim.game import Action, CRGame

# ---------------------------------------------------------------------------
# Harness helpers
# ---------------------------------------------------------------------------


def fresh_game() -> CRGame:
    """A game with the standard 6 towers and no other entities."""
    return CRGame(seed=0)


def spawn(
    game: CRGame,
    card_type: CardType,
    owner: int,
    x: float,
    y: float,
    *,
    active: bool = True,
) -> Entity:
    """Place a single live entity, bypassing elixir/hand/deploy delay.

    With ``active=True`` the unit is immediately deployed (no 1s deploy timer)
    so combat tests are deterministic.
    """
    card_def = CARD_DEFS[card_type]
    e = entity_from_card(game._alloc_eid(), owner, card_def, x, y)
    if active:
        e.deploy_timer = 0.0
        e.is_deployed = True
    game.entities.append(e)
    return e


def cast_spell(game: CRGame, card_type: CardType, owner: int, x: float, y: float) -> None:
    """Cast a spell at a location through the real game spawn routing."""
    game._spawn_card(owner, CARD_DEFS[card_type], x, y)


def step_n(game: CRGame, n: int) -> None:
    wait = [Action(0, -1), Action(1, -1)]
    for _ in range(n):
        game.step(wait)


def secs(t: float) -> int:
    """Convert seconds to ticks (20 ticks/sec)."""
    return int(round(t / 0.05))


def living(game: CRGame, owner: int, card_type: CardType) -> list[Entity]:
    return [
        e
        for e in game.entities
        if e.alive and e.owner == owner and e.card_type == card_type and not e.is_tower
    ]


# ---------------------------------------------------------------------------
# Spell kill-threshold tests (no positioning involved)
# ---------------------------------------------------------------------------


def test_zap_kills_skeletons():
    g = fresh_game()
    for dx in (-0.3, 0.0, 0.3):
        spawn(g, CardType.SKELETONS, 1, 9.0 + dx, 16.0)
    n_before = len(living(g, 1, CardType.SKELETONS))
    assert n_before >= 1
    cast_spell(g, CardType.ZAP, 0, 9.0, 16.0)
    step_n(g, 2)
    assert living(g, 1, CardType.SKELETONS) == [], "Zap (192) must kill skeletons (81 hp)"


def test_zap_does_not_kill_archers():
    g = fresh_game()
    a = spawn(g, CardType.ARCHERS, 1, 9.0, 16.0)
    cast_spell(g, CardType.ZAP, 0, 9.0, 16.0)
    step_n(g, 2)
    assert a.alive, "Zap (192) must NOT kill an Archer (304 hp)"


def test_arrows_kill_archers():
    g = fresh_game()
    spawn(g, CardType.ARCHERS, 1, 9.0, 16.0)
    spawn(g, CardType.ARCHERS, 1, 9.3, 16.0)
    cast_spell(g, CardType.ARROWS, 0, 9.15, 16.0)
    step_n(g, 2)
    assert living(g, 1, CardType.ARCHERS) == [], "Arrows (322) must kill Archers (304 hp)"


def test_fireball_does_not_kill_musketeer():
    g = fresh_game()
    m = spawn(g, CardType.MUSKETEER, 1, 9.0, 16.0)
    cast_spell(g, CardType.FIREBALL, 0, 9.0, 16.0)
    step_n(g, 2)
    assert m.alive, "Fireball (689) must NOT kill a Musketeer (721 hp)"
    assert m.hp < m.max_hp, "Fireball must still damage the Musketeer"


def test_fireball_plus_zap_kills_musketeer():
    g = fresh_game()
    m = spawn(g, CardType.MUSKETEER, 1, 9.0, 16.0)
    cast_spell(g, CardType.FIREBALL, 0, 9.0, 16.0)
    cast_spell(g, CardType.ZAP, 0, 9.0, 16.0)
    step_n(g, 2)
    assert not m.alive, "Fireball (689) + Zap (192) = 881 must kill a Musketeer (721 hp)"


def test_fireball_kills_archers():
    g = fresh_game()
    spawn(g, CardType.ARCHERS, 1, 9.0, 16.0)
    spawn(g, CardType.ARCHERS, 1, 9.3, 16.0)
    cast_spell(g, CardType.FIREBALL, 0, 9.15, 16.0)
    step_n(g, 2)
    assert living(g, 1, CardType.ARCHERS) == [], "Fireball (689) must kill Archers (304 hp)"


def test_fireball_does_not_kill_wizard():
    g = fresh_game()
    w = spawn(g, CardType.WIZARD, 1, 9.0, 16.0)
    cast_spell(g, CardType.FIREBALL, 0, 9.0, 16.0)
    step_n(g, 2)
    assert w.alive, "Fireball (689) must NOT kill a Wizard (755 hp)"


def test_rocket_kills_musketeer():
    g = fresh_game()
    m = spawn(g, CardType.MUSKETEER, 1, 9.0, 16.0)
    cast_spell(g, CardType.ROCKET, 0, 9.0, 16.0)
    step_n(g, 2)
    assert not m.alive, "Rocket (1485) must kill a Musketeer (721 hp)"


def test_arrows_dont_kill_knight():
    g = fresh_game()
    k = spawn(g, CardType.KNIGHT, 1, 9.0, 16.0)
    cast_spell(g, CardType.ARROWS, 0, 9.0, 16.0)
    step_n(g, 2)
    assert k.alive, "Arrows (322) must NOT kill a Knight (1766 hp)"


# ---------------------------------------------------------------------------
# 1v1 troop combat (clear outcomes regardless of micro)
# ---------------------------------------------------------------------------


def test_knight_kills_single_skeleton():
    g = fresh_game()
    k = spawn(g, CardType.KNIGHT, 0, 9.0, 8.0)
    s = spawn(g, CardType.SKELETONS, 1, 9.0, 8.6)
    step_n(g, secs(4))
    assert k.alive and not s.alive, "Knight must kill a lone Skeleton and survive"


def test_mini_pekka_beats_knight():
    g = fresh_game()
    # Duel deep in player 0's half so the winner does not immediately reach
    # (and get shot down by) the enemy tower.
    mp = spawn(g, CardType.MINI_PEKKA, 0, 9.0, 8.0)
    k = spawn(g, CardType.KNIGHT, 1, 9.0, 8.8)
    step_n(g, secs(7))
    assert mp.alive and not k.alive, "Mini Pekka (640/hit) must beat a Knight 1v1"


def test_pekka_beats_mini_pekka():
    g = fresh_game()
    p = spawn(g, CardType.PEKKA, 0, 9.0, 8.0)
    mp = spawn(g, CardType.MINI_PEKKA, 1, 9.0, 8.8)
    step_n(g, secs(9))
    assert p.alive and not mp.alive, "P.E.K.K.A must beat a Mini Pekka 1v1"


def test_valkyrie_clears_skeleton_army():
    g = fresh_game()
    v = spawn(g, CardType.VALKYRIE, 0, 9.0, 8.0)
    for i in range(15):
        dx = (i % 5 - 2) * 0.35
        dy = (i // 5) * 0.35
        spawn(g, CardType.SKELETON_ARMY, 1, 9.0 + dx, 8.6 + dy)
    step_n(g, secs(8))
    assert v.alive, "Valkyrie (splash) must survive a Skeleton Army"
    assert living(g, 1, CardType.SKELETON_ARMY) == [], "Valkyrie must clear the Skeleton Army"


# ---------------------------------------------------------------------------
# Tower mechanics
# ---------------------------------------------------------------------------


def test_king_tower_starts_inactive():
    g = fresh_game()
    # An enemy unit next to the king tower should not be shot while it is
    # inactive (no princess tower destroyed, king undamaged).
    assert not g._king_tower_activated(0)


def test_giant_damages_princess_tower():
    g = fresh_game()
    pt = g.princess_towers[1][0]
    start_hp = pt.hp
    # Giant placed right in front of the left princess tower of player 1.
    spawn(g, CardType.GIANT, 0, pt.x, pt.y - 1.5)
    step_n(g, secs(6))
    assert pt.hp < start_hp, "A Giant in range must damage the princess tower"


def test_hog_rider_reaches_and_hits_tower():
    g = fresh_game()
    pt = g.princess_towers[1][0]
    start_hp = pt.hp
    # Hog rider placed at the bridge area, building-targeting, should run to the tower.
    spawn(g, CardType.HOG_RIDER, 0, pt.x, 18.0)
    step_n(g, secs(12))
    assert pt.hp < start_hp, "Hog Rider must reach and damage the princess tower"


def test_ground_troop_crosses_river_to_enemy_tower():
    g = fresh_game()
    pt = g.princess_towers[1][1]  # right princess tower at (13, 27)
    start_hp = pt.hp
    # Knight placed on its own side near the right bridge; must cross and attack.
    k = spawn(g, CardType.KNIGHT, 0, 13.0, 13.0)
    step_n(g, secs(25))
    assert k.y > 16.0, "Knight must cross the river (y past the bridge)"
    assert pt.hp < start_hp, "Knight must cross and damage the enemy princess tower"


# ---------------------------------------------------------------------------
# More spell / status mechanics
# ---------------------------------------------------------------------------


def test_lightning_kills_musketeer():
    g = fresh_game()
    m = spawn(g, CardType.MUSKETEER, 1, 9.0, 16.0)
    cast_spell(g, CardType.LIGHTNING, 0, 9.0, 16.0)
    step_n(g, 2)
    assert not m.alive, "Lightning (1057) must kill a Musketeer (721 hp)"


def test_poison_kills_musketeer_over_time():
    g = fresh_game()
    m = spawn(g, CardType.MUSKETEER, 1, 9.0, 16.0)
    cast_spell(g, CardType.POISON, 0, 9.0, 16.0)
    step_n(g, 2)
    assert m.alive, "Poison must NOT instantly kill (it is damage-over-time)"
    step_n(g, secs(8.5))
    assert not m.alive, "Poison (~92/s over 8s) must kill a Musketeer over its duration"


def test_log_kills_goblins():
    g = fresh_game()
    for dx in (-0.4, 0.0, 0.4):
        spawn(g, CardType.GOBLINS, 1, 9.0 + dx, 16.0)
    cast_spell(g, CardType.THE_LOG, 0, 9.0, 16.0)
    step_n(g, 2)
    assert living(g, 1, CardType.GOBLINS) == [], "The Log (269) must kill Goblins (202 hp)"


def test_freeze_stops_tower_from_firing():
    g = fresh_game()
    pt = g.princess_towers[0][0]
    enemy = spawn(g, CardType.KNIGHT, 1, pt.x, pt.y + 1.0)
    cast_spell(g, CardType.FREEZE, 1, pt.x, pt.y)
    assert pt.frozen, "Freeze must apply a freeze timer to the tower"
    hp_at_freeze = enemy.hp
    step_n(g, secs(1.0))
    assert enemy.hp == hp_at_freeze, "A frozen tower must not deal damage"


# ---------------------------------------------------------------------------
# Targeting rules
# ---------------------------------------------------------------------------


def test_giant_only_targets_buildings():
    g = fresh_game()
    pt = g.princess_towers[1][0]
    giant = spawn(g, CardType.GIANT, 0, pt.x, pt.y - 3.0)
    # A defender right next to the Giant must be ignored for targeting.
    spawn(g, CardType.KNIGHT, 1, pt.x + 0.5, pt.y - 3.0)
    step_n(g, secs(1.0))
    tgt = g._get_entity(giant.target_eid)
    assert tgt is not None and (tgt.is_building or tgt.is_tower), (
        "Giant (building-targeting) must target the tower, not the Knight"
    )


def test_cannon_does_not_target_air():
    g = fresh_game()
    spawn(g, CardType.CANNON, 0, 9.0, 10.0)
    minion = spawn(g, CardType.MINIONS, 1, 9.0, 10.5)  # flying
    step_n(g, secs(2.0))
    assert minion.alive, "A ground-only Cannon must not be able to kill flying Minions"


def test_princess_tower_kills_lone_skeleton():
    g = fresh_game()
    pt = g.princess_towers[0][0]
    s = spawn(g, CardType.SKELETONS, 1, pt.x, pt.y + 1.5)
    step_n(g, secs(3.0))
    assert not s.alive, "Princess tower must kill a lone Skeleton in range"


# ---------------------------------------------------------------------------
# Minimum range (Mortar dead zone)
# ---------------------------------------------------------------------------


def test_mortar_cannot_hit_point_blank_attacker():
    g = fresh_game()
    mortar = spawn(g, CardType.MORTAR, 0, 9.0, 10.0)
    assert mortar.minimum_range >= 3.0, "Mortar must have a minimum range"
    # Enemy hugging the Mortar, well inside the dead zone.
    knight = spawn(g, CardType.KNIGHT, 1, 9.0, 10.3)
    knight.target_eid = mortar.eid
    mortar.target_eid = knight.eid
    step_n(g, secs(3.0))
    assert knight.alive, "Mortar must not damage a point-blank attacker (inside min range)"


def test_mortar_hits_target_beyond_minimum_range():
    g = fresh_game()
    mortar = spawn(g, CardType.MORTAR, 0, 9.0, 10.0)
    # Enemy past the dead zone but within the Mortar's long range.
    target = spawn(g, CardType.MUSKETEER, 1, 9.0, 10.0 + mortar.minimum_range + 1.5)
    target.target_eid = mortar.eid
    mortar.target_eid = target.eid
    hp0 = target.hp
    step_n(g, secs(6.0))
    assert target.hp < hp0, "Mortar must damage a target beyond its minimum range"


# ---------------------------------------------------------------------------
# Death damage (Golem explosion)
# ---------------------------------------------------------------------------


def test_golem_death_damage_hits_nearby_enemy():
    g = fresh_game()
    golem = spawn(g, CardType.GOLEM, 0, 9.0, 16.0)
    victim = spawn(g, CardType.GOBLINS, 1, 9.0, 16.5)  # within death radius (2.0)
    assert golem.death_damage > 0
    hp0 = victim.hp
    golem.hp = 0.0  # kill the Golem so its death explosion fires
    step_n(g, 2)
    assert victim.hp < hp0, "Golem death explosion must damage a nearby enemy"


def test_death_damage_spares_distant_enemy():
    g = fresh_game()
    golem = spawn(g, CardType.GOLEM, 0, 9.0, 16.0)
    far = spawn(g, CardType.GOBLINS, 1, 9.0, 16.0 + golem.death_damage_radius + 2.0)
    hp0 = far.hp
    golem.hp = 0.0
    step_n(g, 2)
    assert far.hp == hp0, "Death explosion must not reach an enemy outside its radius"


# ---------------------------------------------------------------------------
# Crown-tower damage reduction (spells / Miner deal less to towers)
# ---------------------------------------------------------------------------


def _enemy_princess(game: CRGame) -> object:
    return next(e for e in game.entities if e.owner == 1 and e.is_tower and not e.is_king_tower)


def test_fireball_deals_reduced_damage_to_crown_tower():
    g = fresh_game()
    fb = CARD_DEFS[CardType.FIREBALL]
    assert fb.crown_tower_damage_percent == -70.0
    tower = _enemy_princess(g)
    hp0 = tower.hp
    cast_spell(g, CardType.FIREBALL, 0, tower.x, tower.y)
    step_n(g, 2)
    # -70% => the crown tower takes 30% of the full 689 damage.
    assert (hp0 - tower.hp) == pytest.approx(fb.damage_per_hit * 0.30, rel=1e-3)


def test_fireball_full_damage_to_troops():
    g = fresh_game()
    fb = CARD_DEFS[CardType.FIREBALL]
    musk = spawn(g, CardType.MUSKETEER, 1, 9.0, 16.0)
    hp0 = musk.hp
    cast_spell(g, CardType.FIREBALL, 0, 9.0, 16.0)
    step_n(g, 2)
    # A troop takes the full hit, not the crown-tower-reduced amount.
    assert (hp0 - musk.hp) == pytest.approx(fb.damage_per_hit, rel=1e-3)


def test_rocket_and_log_have_authentic_crown_reduction():
    assert CARD_DEFS[CardType.ROCKET].crown_tower_damage_percent == -75.0
    assert CARD_DEFS[CardType.THE_LOG].crown_tower_damage_percent == -80.0
    assert CARD_DEFS[CardType.MINER].crown_tower_damage_percent == -75.0


# ---------------------------------------------------------------------------
# Death-spawned units carry authentic stats (scaled by their OWN row)
# ---------------------------------------------------------------------------


def test_golem_spawns_authentic_golemites():
    golem = CARD_DEFS[CardType.GOLEM]
    # Golem spawns 2 Golemites; each is an Epic unit (~1040 HP, ~20 DPS at L11),
    # scaled from the Golemite row -- not the Golem's own stats.
    assert golem.death_spawn_count == 2
    assert golem.death_spawn_hp == pytest.approx(1040.0, rel=0.02)
    assert golem.death_spawn_dps == pytest.approx(19.8, rel=0.05)


# ---------------------------------------------------------------------------
# Champion abilities (manually activated via the ABILITY action)
# ---------------------------------------------------------------------------


def test_ability_action_only_valid_with_ready_champion():
    from crsim.constants import ABILITY_ACTION

    g = fresh_game()
    # No champion on the field: the ability action is masked off.
    assert not g.get_valid_actions_mask(0)[ABILITY_ACTION]

    champ = spawn(g, CardType.SKELETON_KING, 0, 9.0, 6.0)
    g.players[0].elixir = 10.0
    assert g.get_valid_actions_mask(0)[ABILITY_ACTION]

    # An enemy champion does not enable player 0's ability.
    assert not g.get_valid_actions_mask(1)[ABILITY_ACTION]
    assert champ.is_champion


def test_skeleton_king_ability_summons_skeletons_and_costs_elixir():
    g = fresh_game()
    spawn(g, CardType.SKELETON_KING, 0, 9.0, 6.0)
    g.players[0].elixir = 8.0

    before = len(living(g, 0, CardType.SKELETONS))
    g.apply_action(Action(player=0, ability=True))
    after = len(living(g, 0, CardType.SKELETONS))

    assert after - before == 6  # Soul Summoning raises 6 skeletons
    assert g.players[0].elixir == pytest.approx(6.0)  # 8 - 2 ability cost


def test_archer_queen_cloak_makes_her_untargetable_and_faster():
    g = fresh_game()
    aq = spawn(g, CardType.ARCHER_QUEEN, 0, 9.0, 16.0)
    enemy = spawn(g, CardType.MUSKETEER, 1, 9.5, 16.0)
    g.players[0].elixir = 8.0

    # Before cloak the enemy can see and target her.
    assert g._find_target(enemy) == aq.eid

    g.apply_action(Action(player=0, ability=True))
    assert aq.invisible_timer == pytest.approx(3.5)
    assert aq.attack_speed_mult == pytest.approx(1.7)
    assert g.players[0].elixir == pytest.approx(7.0)  # 8 - 1 ability cost

    # While cloaked she cannot be targeted.
    assert g._find_target(enemy) != aq.eid


def test_golden_knight_dash_damages_a_line_and_repositions():
    g = fresh_game()
    gk = spawn(g, CardType.GOLDEN_KNIGHT, 0, 9.0, 10.0)
    g.players[0].elixir = 8.0
    # Three enemies ahead of the Golden Knight within dash range.
    targets = [
        spawn(g, CardType.MUSKETEER, 1, 9.0, 11.0),
        spawn(g, CardType.MUSKETEER, 1, 9.0, 12.0),
        spawn(g, CardType.MUSKETEER, 1, 9.0, 14.0),
    ]
    hp0 = [t.hp for t in targets]
    g.apply_action(Action(player=0, ability=True))

    dmg = CARD_DEFS[CardType.GOLDEN_KNIGHT].damage_per_hit
    for t, h0 in zip(targets, hp0):
        assert (h0 - t.hp) == pytest.approx(dmg, rel=1e-3)
    # He repositions onto the farthest enemy he struck.
    assert gk.y == pytest.approx(14.0)


def test_monk_protection_reduces_incoming_damage():
    g = fresh_game()
    monk = spawn(g, CardType.MONK, 0, 9.0, 16.0)
    g.players[0].elixir = 8.0

    hp0 = monk.hp
    monk.take_damage(100.0)
    full = hp0 - monk.hp
    assert full == pytest.approx(100.0)

    monk.hp = hp0
    g.apply_action(Action(player=0, ability=True))
    assert monk.damage_reduction_timer > 0
    monk.take_damage(100.0)
    # Protected: takes 20% of the hit.
    assert (hp0 - monk.hp) == pytest.approx(20.0)


def test_mighty_miner_bomb_explodes_after_delay():
    g = fresh_game()
    mm = spawn(g, CardType.MIGHTY_MINER, 0, 9.0, 12.0)
    enemy = spawn(g, CardType.MUSKETEER, 1, 9.0, 12.0)
    g.players[0].elixir = 8.0
    g.apply_action(Action(player=0, ability=True))

    assert len(g.pending_bombs) == 1
    # The Miner burrowed forward from his planting spot.
    assert mm.y > 12.0
    # Drive only the bomb update so the timing check isn't perturbed by movement.
    hp0 = enemy.hp
    for _ in range(secs(1.0)):
        g._update_bombs()
    assert enemy.hp == pytest.approx(hp0)  # no damage before the fuse runs out
    assert len(g.pending_bombs) == 1
    for _ in range(secs(1.2)):
        g._update_bombs()
    assert enemy.hp < hp0  # bomb detonated at the planting spot
    assert len(g.pending_bombs) == 0


def test_champion_ability_respects_cooldown_and_cost():
    from crsim.constants import ABILITY_ACTION

    g = fresh_game()
    spawn(g, CardType.SKELETON_KING, 0, 9.0, 6.0)
    g.players[0].elixir = 8.0
    g.apply_action(Action(player=0, ability=True))

    # On cooldown immediately after use -> masked off even with elixir to spare.
    assert not g.get_valid_actions_mask(0)[ABILITY_ACTION]
    # A second activation while on cooldown spawns nothing and costs nothing.
    elx = g.players[0].elixir
    n = len(living(g, 0, CardType.SKELETONS))
    g.apply_action(Action(player=0, ability=True))
    assert len(living(g, 0, CardType.SKELETONS)) == n
    assert g.players[0].elixir == pytest.approx(elx)
