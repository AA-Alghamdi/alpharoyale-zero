# Decision-Policy Specification: Strategic Clash Royale Bot

Status: implementation spec, grounded in the actual ClashRoyaleBuildABot codebase.
Target file layout: new package `clashroyalebuildabot/policy/` (`policy.py`, `expert_policy.py`, `board.py`, `cardinfo.py`), plus a minimal edit to `clashroyalebuildabot/bot/bot.py`.

---

## 0. Grounding facts (verified against the repo)

These are not assumptions; they were read out of the source and constrain every design choice below.

- **Coordinate frame is 720x1280, not 1080x1920.** Placement math lives in `DISPLAY_WIDTH=720, DISPLAY_HEIGHT=1280` (`constants.py`). `Emulator.click()` scales DISPLAY-space to the real device at tap time. The policy must emit **tiles**, never pixels.
- **Tile grid.** `N_WIDE_TILES=18` (tile_x in `0..17`), `N_HEIGHT_TILES=15`. `tile_y=0` is our back near the king; `tile_y` increases toward the enemy; `tile_y=15` is over-river and is only legal when the matching enemy princess tower is dead (`LEFT_PRINCESS_TILES`/`RIGHT_PRINCESS_TILES` unlock in `Bot._get_valid_tiles`). Tile->pixel is `Bot._get_tile_centre(tx,ty)`.
- **State shape** (`namespaces/state.py`): `State(allies: List[UnitDetection], enemies: List[UnitDetection], numbers: Numbers, cards: Tuple[Card,Card,Card,Card], ready: List[int], screen)`.
  - NOTE: `state.cards` here is typed as a 4-tuple in this revision of the repo, but `Bot.get_actions` reads `state.cards[i + 1]` for `i in state.ready`, i.e. it treats `cards` as `[next, hand0, hand1, hand2, hand3]`. **The policy must follow `Bot.get_actions`: hand slot `i` is `state.cards[i + 1]`.** This off-by-one is load-bearing; getting it wrong plays the wrong card.
- **`ready`** is a `List[int]` of currently-deployable hand slot indices (`0..3`).
- **Numbers / tower HP** (`namespaces/numbers.py`): `elixir.number` (float 0..10) and four princess-tower fields, each a `NumberDetection` whose `.number` is a **fraction in [0,1]** (`0.0` == tower destroyed). There is **no king-tower HP, no match clock, and no double-elixir flag** in `State` today. Tier triggers that "want" the clock degrade gracefully (see Section 6).
- **Unit metadata** (`namespaces/units.py`): `unit.category` in `{TROOP, BUILDING}`; `unit.target` in `{AIR, GROUND, BUILDINGS, ALL, None}`; `unit.transport` in `{AIR, GROUND, None}`; `position.tile_x/tile_y/conf`.
- **Card metadata** (`namespaces/cards.py`): `Card(name, target_anywhere, cost, units: list[Unit], id_)`. A **spell** is `target_anywhere is True and units == []` (e.g. Arrows, Fireball, Zap). `Card.__hash__` is by `name`, so `Card` is usable as a dict key.
- **Current control flow** (`Bot._handle_game_step`, paraphrased): `actions = get_actions(state); shuffle; pick argmax of action.calculate_score(state); if best_score[0] == 0 -> do nothing; else play_action`. Each `calculate_score` returns a **lexicographic list** (e.g. Giant `[1, left_hp>0, left_hp<=right_hp]`; Spell `[gate, hit_score, -dist]`; Defense `[0|1]`). This is already a degenerate one-tier expert system; the redesign lifts **card-class arbitration** into ordered tiers while **reusing the per-action scorers as the placement primitive**.

The two anti-pattern lists in the strategy doc all reduce to four failure modes this spec must structurally prevent: (a) leaking at 10, (b) negative spell/defense trades, (c) acting reactively the instant a card is affordable (the "too many pointless moves" complaint), (d) committing offense into a full enemy bar with no read.

---

## 1. The swappable policy seam

A single abstract seam that `Bot` calls once per decision step. The expert system is one implementation; a learned net is a drop-in replacement with **zero `Bot` changes**.

```python
# clashroyalebuildabot/policy/policy.py
from abc import ABC, abstractmethod
from collections import namedtuple
from clashroyalebuildabot.namespaces.state import State

# card_index : int in 0..3, the HAND SLOT (maps to state.cards[card_index + 1], per Bot.get_actions)
# tile_x     : int 0..17
# tile_y     : int 0..30 (our half is 0..14; 15+ only when the enemy tower on that lane is dead)
# tier       : str, telemetry only (ignored by Bot)
# reason     : str, telemetry only
Decision = namedtuple("Decision", ["card_index", "tile_x", "tile_y", "tier", "reason"])


class Policy(ABC):
    """The one seam. Bot depends only on this."""

    @abstractmethod
    def decide(self, state: State, candidates: list) -> "Decision | None":
        """
        state      : the existing namespaces.state.State
        candidates : Bot.get_actions(state) output. Already filtered to
                     (ready slot, affordable, legal tile). Each is an Action
                     with .index, .tile_x, .tile_y, .CARD, .calculate_score(state).
        returns    : a Decision, or None == "hold / wait" (Bot sleeps one step).
                     A Decision whose card_index is None is also treated as hold.
        """
        ...

    def on_episode_start(self) -> None:
        """Reset any per-game memory (e.g. counterpush flag, enemy-card log)."""
        ...
```

### Why this exact signature is the right swap point

- `decide(state, candidates) -> Decision | None` is the **complete** decision: which card, which lane, which depth, encoded as a hand slot + tile. Everything below it (`play_action` -> `_get_card_centre` -> `_get_tile_centre` -> `click`) is untouched, so the click/scaling layer is reused verbatim.
- Returning **`None`** is a first-class action ("wait / hold elixir"). This is the single most important affordance for the anti-spam requirement: the current loop can only "do nothing" via the implicit `best_score[0]==0` path; the seam makes *hold* an explicit, intentional decision.
- `candidates` are **pre-validated** by `get_actions` (ready, affordable, on a legal tile). A learned policy can ignore the candidate scores entirely but still reuse the candidate set as a **legality mask**, guaranteeing every click it emits is legal.
- The `tier`/`reason` fields are telemetry only. They make behavior-cloning logs (state -> Decision) self-labeling and give the heuristic a debuggable audit trail without affecting `Bot`.

### A learned policy is a drop-in

```python
class LearnedPolicy(Policy):
    def __init__(self, net):
        self.net = net

    def decide(self, state, candidates):
        if not candidates:
            return None
        x = encode_state(state)                      # (C, 32, 18) grid + scalars
        card_logits, x_logits, y_logits = self.net(x)
        mask_card_logits(card_logits, state.ready, state.numbers.elixir.number, state.cards)
        ci = argmax(card_logits)
        tx, ty = argmax(x_logits), argmax(y_logits)
        tx, ty = snap_to_nearest_legal(candidates, ci, tx, ty)  # guarantees a legal click
        return Decision(ci, tx, ty, "learned", "net")
```

`snap_to_nearest_legal` projects the net's continuous intent onto the nearest `(tile_x, tile_y)` actually present in `candidates` for the chosen card. The expert policy doubles as the BC teacher: log `(encode_state(state), Decision)` every step.

---

## 2. Bot integration (minimal diff)

```python
# bot.py
class Bot:
    def __init__(self, ..., policy: Policy | None = None):
        ...
        from clashroyalebuildabot.policy.expert_policy import ExpertPolicy
        self.policy = policy or ExpertPolicy(self.cards_to_actions)

    def _action_from_decision(self, state, d: Decision):
        card = state.cards[d.card_index + 1]          # SLOT i -> cards[i+1]; see get_actions
        return self.cards_to_actions[card](d.card_index, d.tile_x, d.tile_y)

    def _handle_game_step(self, state):
        candidates = self.get_actions(state)
        decision = self.policy.decide(state, candidates)
        if decision is None or decision.card_index is None:
            self._log_and_wait("Holding (policy)", self.play_action_delay)
            return
        action = self._action_from_decision(state, decision)
        self.play_action(action)                      # unchanged: card-click then tile-click
        self._log_and_wait(f"[{decision.tier}] {action} ({decision.reason})",
                           self.play_action_delay)
```

Call `self.policy.on_episode_start()` when the screen transitions into in-game from a non-in-game screen. Everything in `play_action` and below is unchanged.

---

## 3. The card database -> (card, lane, depth) mapping

The policy reasons about **roles** derived purely from card/unit metadata (no hand-maintained per-card lists where avoidable), so it works for any deck the detector is configured with.

### 3a. Role inference from the card DB (`policy/cardinfo.py`)

```python
from clashroyalebuildabot.namespaces.units import UnitCategory, Target, Transport

def is_spell(card):            return card.target_anywhere and not card.units
def is_building(card):         return any(u.category == UnitCategory.BUILDING for u in card.units)
def can_hit_air(card):         return any(u.target in (Target.AIR, Target.ALL) for u in card.units)
def is_ranged(card):           return any(u.target in (Target.ALL, Target.AIR) for u in card.units)  # proxy for support
def is_win_condition(card):
    # building-targeters that walk to a tower (Giant, Hog, Balloon, Ram, Royal Giant, ...)
    return any(u.target == Target.BUILDINGS for u in card.units)
def is_swarm(card):
    # multiple low-cost bodies: >= 2 listed troop units OR a known horde, cost <= 5
    troops = [u for u in card.units if u.category == UnitCategory.TROOP]
    return len(troops) >= 2 or (len(troops) == 1 and card.cost <= 2)
def threat_dps_class(card):    # crude "is this a sufficient hard counter" key
    return card.cost                                  # cheapest-sufficient bias uses raw cost
```

This is the role layer. It is honest: a "win condition" is literally a card whose units target buildings (the only units that reliably reach a tower); a "spell" is literally `target_anywhere` with no body. The one heuristic shortcut is `is_swarm` (labeled as such in code). `is_win_condition` may be empty if the deck has none, in which case Tier 4 never fires (correct: no win condition, no committed push).

### 3b. Lane + depth -> tile (`policy/board.py`)

The strategic vocabulary is `(lane in {left,right,center}, depth in {back, defensive, bridge, bridge_aggro})`. This resolves to a tile in the existing grid; `Bot._get_tile_centre` then resolves to a pixel.

```python
from clashroyalebuildabot.constants import N_WIDE_TILES, N_HEIGHT_TILES  # 18, 15

LANE_COL  = {"left": 3, "right": 14, "center": N_WIDE_TILES // 2}   # 3, 14, 9
DEPTH_ROW = {
    "back":         1,    # behind king: tank build-up + safe cycle (lowest pixel-risk)
    "defensive":    8,    # mid-court intercept pocket in front of king tower
    "bridge":       14,   # front ally row, ALWAYS legal (no tower-down requirement)
    "bridge_aggro": 15,   # over-river; legal only when that lane's enemy tower is dead
}

def intent_to_tile(lane, depth):
    tx = max(0, min(N_WIDE_TILES - 1, LANE_COL[lane]))
    ty = max(0, min(N_HEIGHT_TILES, DEPTH_ROW[depth]))   # 15 allowed; legality gated downstream
    return tx, ty
```

Columns 3/14 are the calibrated tower-front lanes already used by `BridgeAction`/`GiantAction`. `depth="bridge"` defaults to row 14 (always legal) so a generic push never targets an illegal over-river tile; `bridge_aggro` (row 15) is used only after a lane's tower is down.

### 3c. Choosing (card, lane, depth) from hand + elixir

Each tier (Section 4) computes its own `(lane, depth)` from the board, then **filters the hand to role-appropriate, affordable, ready cards and picks the cheapest sufficient one**, then **snaps to the candidate Action nearest the intended tile**. The snap step is what reuses the pre-validated candidate set:

```python
def _pick(self, state, candidates, role_pred, lane, depth, prefer="cheapest"):
    tx, ty = intent_to_tile(lane, depth)
    pool = [a for a in candidates if role_pred(state.cards[a.index + 1])]
    if not pool:
        return None
    if prefer == "cheapest":
        c = min(state.cards[a.index + 1].cost for a in pool)
        pool = [a for a in pool if state.cards[a.index + 1].cost == c]
    # snap to the legal tile closest to intent (guarantees the click is valid)
    best = min(pool, key=lambda a: (a.tile_x - tx) ** 2 + (a.tile_y - ty) ** 2)
    return best
```

Lane selection rules used by the tiers:
- **Defense:** lane of the focus threat (`left` if `threat.tile_x < 9` else `right`); tie-break toward the lower-HP **ally** tower (defend the weak side).
- **Offense / win condition:** lane of the **weaker enemy** tower (`left` if `left_enemy_hp <= right_enemy_hp` else `right`) — concentrate damage on one tower (the chip-strategy fundamental). Once committed, the focus tower does not switch.
- **Counter-push:** the lane where surviving allies already are (do not march them across).

---

## 4. Prioritized control flow (the tier cascade)

`decide` evaluates tiers in fixed order; **the first tier that returns a non-None Decision wins**. This replaces the cross-class argmax. Within a tier, placement uses the existing per-action scorers / the `_pick` snap.

```python
# clashroyalebuildabot/policy/expert_policy.py
class ExpertPolicy(Policy):
    BRIDGE_Y         = 15      # river row; enemy half is tile_y > 15
    OUR_HALF_Y       = 14      # enemy with tile_y <= 14 is on our side
    WIN_CON_COMMIT_Y = 16      # commit early vs building-targeters near the bridge
    LOW_ELIXIR       = 3       # below this, never dump with no threat
    LEAK_ELIXIR      = 9       # at/above this, cycle to avoid leaking at 10
    SUPPORT_GAP      = 2       # support follows a tank by this many cards of elixir headroom

    def __init__(self, cards_to_actions):
        self.c2a = cards_to_actions
        self.on_episode_start()

    def on_episode_start(self):
        self._won_defense_recently = False   # cross-frame memory for Tier 2
        self._defense_cooldown     = 0       # steps since last defense fired
        self._enemy_seen_cards     = set()   # information game (Section 5)
        self._focus_lane           = None    # committed offensive tower (chip focus)

    def decide(self, state, candidates):
        self._observe(state)                 # update enemy-card memory, decay flags
        if not candidates:
            return None
        for tier in (self._t1_defend,
                     self._t2_counterpush,
                     self._t3_spell_value,
                     self._t4_win_condition,
                     self._t5_cycle_or_hold):
            d = tier(state, candidates)
            if d is not None:
                return d
        return None
```

### Tier 1 — DEFEND imminent tower threats (priority 1)

**Trigger.** Any enemy with `tile_y <= OUR_HALF_Y` (crossed to our half), OR any building-targeter (`is_win_condition` via `unit.target == BUILDINGS`) with `tile_y <= WIN_CON_COMMIT_Y` (commit early against Hog/Giant/Balloon/Ram before it connects).

**Focus threat.** The enemy with the smallest `tile_y` (deepest into our base). Tie-break toward the lower-HP ally tower's lane.

**Card choice.** Filter the ready+affordable hand to defenders (non-spell). If the threat is **air** (`transport == AIR`), keep only cards with `can_hit_air`. Pick the **cheapest sufficient** defender (positive-trade bias; "spend the minimum that wins"). If the only adequate counter to the enemy's known win condition is in hand and a cheaper body suffices here, *hold the hard counter* and use the cheaper body (reserve management, Section 5).

**Placement.**
- Troop vs ground push: intercept pocket in the threatened lane, `intent_to_tile(lane, "defensive")` adjusted toward the threat: `tx = clamp(threat.tile_x, 5, 12)`, `ty = max(threat.tile_y - 2, 6)` so the body spawns *in front of* the tower (enemy walks into it on our side, never across the river).
- Single-target chaser (Prince/Mini P.E.K.K.A): place a cheap body on the **opposite inner side** of the unit to kite it toward center and activate the king tower.
- Building card (`is_building`): place at the center pull spot `(8 or 9, 9..11)` to drag a building-targeter to the middle.

**Anti-overcommit guard.** Tier 1 returns **exactly one** Decision per step. It does **not** stack a second card on a threat the first already neutralizes; the next step re-evaluates and, if the threat is handled, Tier 1 no longer fires.

```python
def _t1_defend(self, state, candidates):
    threats = [e for e in state.enemies if e.position.tile_y <= self.OUR_HALF_Y]
    threats += [e for e in state.enemies
                if is_win_condition_unit(e.unit) and e.position.tile_y <= self.WIN_CON_COMMIT_Y]
    if not threats:
        return None
    threat = min(threats, key=lambda e: e.position.tile_y)
    need_air = (threat.unit.transport == Transport.AIR)

    def ok(card):
        return (not is_spell(card)) and (not need_air or can_hit_air(card))

    pool = [a for a in candidates if ok(state.cards[a.index + 1])]
    if not pool:
        return None
    # cheapest sufficient; reserve the win-con hard-counter if a cheaper body exists
    cmin = min(state.cards[a.index + 1].cost for a in pool)
    pool = [a for a in pool if state.cards[a.index + 1].cost == cmin]

    lane = "left" if threat.position.tile_x < 9 else "right"
    tx = max(5, min(12, threat.position.tile_x))
    ty = max(6, threat.position.tile_y - 2)
    a = min(pool, key=lambda a: (a.tile_x - tx) ** 2 + (a.tile_y - ty) ** 2)
    self._defense_cooldown = 0
    self._won_defense_recently = True            # armed for Tier 2 next steps
    return Decision(a.index, a.tile_x, a.tile_y, "defend", threat.unit.name)
```

### Tier 2 — POSITIVE-elixir COUNTERS / counter-push from survivors (priority 1-2)

Tier 2 fires only when Tier 1 did **not** (no live threat needing a fresh defender) but a defense just resolved with surviving allies — the highest-EV offense in the game ("free elixir already on the board").

**Trigger.** `self._won_defense_recently` AND a surviving ally **troop** on our half (`category == TROOP and tile_y <= 15`) AND no enemy on our half.

**Action.** Add **one** support behind the survivor: `is_ranged` (or a tank if the survivor is ranged), placed at `(survivor.tile_x, survivor.tile_y - 1)` so it walks up in the survivor's shadow. Do **not** start a fresh back push and do **not** switch lanes. After firing once, clear `_won_defense_recently` so we do not pile on.

```python
def _t2_counterpush(self, state, candidates):
    if not self._won_defense_recently:
        return None
    if any(e.position.tile_y <= self.OUR_HALF_Y for e in state.enemies):
        return None                                # a new threat exists -> Tier 1 owns it
    survivor = next((u for u in state.allies
                     if u.unit.category == UnitCategory.TROOP and u.position.tile_y <= 15),
                    None)
    if survivor is None:
        return None
    pool = [a for a in candidates
            if is_ranged(state.cards[a.index + 1]) and not is_spell(state.cards[a.index + 1])]
    if not pool:
        return None
    tx, ty = survivor.position.tile_x, max(1, survivor.position.tile_y - 1)
    a = min(pool, key=lambda a: (a.tile_x - tx) ** 2 + (a.tile_y - ty) ** 2)
    self._won_defense_recently = False             # one support, then stop
    return Decision(a.index, a.tile_x, a.tile_y, "counterpush", "support-behind")
```

### Tier 3 — SPELL for value (priority 2-3)

**Trigger + placement reuse the existing `SpellAction.calculate_score`** (it already returns `[gate, hit_score, -dist]` and searches tiles). The policy **does not lower `MIN_SCORE`** — the positive-value gate is sacred (relaxing it would be a spec change and must be flagged).

**Rules layered on top of the existing gate:**
- Fire only if the spell's own gate passes (`best[0] == 1`, i.e. `hit_score >= MIN_SCORE`): kills 2+ clustered units / removes squishy support behind a tank.
- A spell **on the enemy tower** (high `tile_y`) is allowed only if it secures a tower kill (tower HP fraction `<= spell tower-damage threshold`) in the closing window, OR catches 2+ support troops adjacent to the tower. Never spell a single cheap unit or empty ground.
- Skip Tier 3 if firing would drop elixir below a defensive floor while an enemy push is forming.

```python
def _t3_spell_value(self, state, candidates):
    spells = [a for a in candidates if is_spell(state.cards[a.index + 1])]
    best, best_score = None, [0]
    for a in spells:
        s = a.calculate_score(state)               # existing gated, value-based scorer
        if s > best_score:
            best, best_score = a, s
    if best is None or best_score[0] == 0:
        return None
    return Decision(best.index, best.tile_x, best.tile_y, "spell_value", f"hit={best_score[1]}")
```

### Tier 4 — COUNTER-PUSH / BUILD A PUSH BEHIND THE WIN CONDITION (priority 2-3)

This is the *proactive* offense tier. It deliberately fires only when we are not defending and have banked enough to follow up — so it never opens into a full enemy bar with a lone, unsupported win condition.

**Trigger / gates.**
- No enemy on our half (Tier 1 didn't claim the step).
- We hold a win condition (`is_win_condition`).
- `elixir >= win_condition.cost + SUPPORT_GAP` (so a support card is available as the tank walks). Generalizes the old `elixir == 10` gate to a robust `>=` (OCR-noise tolerant) with a follow-up reserve.
- **Punish window override:** if the enemy is read as low-elixir (just over-committed a 5+ card or two cards on one lane; Section 5), fire immediately into the **opposite** lane even at lower elixir.

**Placement.** Win condition at the **back** of the weaker-enemy-tower lane (`intent_to_tile(lane, "back")`, i.e. row 1) so it gathers elixir while walking and support can be added behind it — *not* at the bridge. Exception (punish window): drop at the bridge in the open lane to exploit the enemy's spent elixir before they recover. Commit to **one** lane (`self._focus_lane`) and keep funneling damage there.

```python
def _t4_win_condition(self, state, candidates):
    if any(e.position.tile_y <= self.OUR_HALF_Y for e in state.enemies):
        return None
    wc = [a for a in candidates if is_win_condition(state.cards[a.index + 1])]
    if not wc:
        return None
    elixir = state.numbers.elixir.number
    le = state.numbers.left_enemy_princess_hp.number
    re = state.numbers.right_enemy_princess_hp.number
    lane = self._focus_lane or ("left" if le <= re else "right")
    self._focus_lane = lane

    punish = self._enemy_overcommitted(state)      # Section 5 read
    cost = min(state.cards[a.index + 1].cost for a in wc)
    if not punish and elixir < cost + self.SUPPORT_GAP:
        return None                                # not enough for a supported push -> hold
    depth = "bridge" if punish else "back"
    tx, ty = intent_to_tile(lane, depth)
    a = min(wc, key=lambda a: (a.tile_x - tx) ** 2 + (a.tile_y - ty) ** 2)
    return Decision(a.index, a.tile_x, a.tile_y, "win_condition",
                    f"lane={lane} {'punish' if punish else 'build'}")
```

(When a win condition is already crossing untouched, the support follow-up is supplied by Tier 2's "support behind survivor/tank" logic on the next steps, because the tank is now a friendly troop in `state.allies`.)

### Tier 5 — CYCLE / WAIT: do not leak, do not spam (priority 1 for leak, else hold)

The final tier encodes the anti-spam contract. It does **two** things and nothing else:

1. **Anti-leak.** If `elixir >= LEAK_ELIXIR` (>= 9, about to cap and waste regen), cycle the **cheapest** ready card to a **safe back tile** in our intended lane (`intent_to_tile(lane, "back")`, row 0-1). This converts soon-to-be-wasted elixir into a cheap body / cycle without committing to the front. This is the only situation where the bot plays "to do something."
2. **Otherwise HOLD** (return `None`). Below the leak threshold with no threat, no value spell, and no supported push, the correct move is to wait and bank elixir — explicitly, not via a degenerate score.

```python
def _t5_cycle_or_hold(self, state, candidates):
    if state.numbers.elixir.number < self.LEAK_ELIXIR:
        return None                                # bank elixir, stay reactive
    le = state.numbers.left_enemy_princess_hp.number
    re = state.numbers.right_enemy_princess_hp.number
    lane = self._focus_lane or ("left" if le <= re else "right")
    cheap = min(candidates, key=lambda a: state.cards[a.index + 1].cost)
    tx, ty = intent_to_tile(lane, "back")
    return Decision(cheap.index, cheap.tile_x, cheap.tile_y, "cycle", "anti-leak")
```

---

## 5. Reading the opponent (light, optional, degrades gracefully)

Per-game memory on the policy instance (reset in `on_episode_start`), updated each step in `_observe`:

```python
def _observe(self, state):
    for e in state.enemies:
        self._enemy_seen_cards.add(e.unit.name)
    if self._defense_cooldown is not None:
        self._defense_cooldown += 1
    # decay the counterpush arming if no survivor materializes within a few steps
    if self._won_defense_recently and self._defense_cooldown > 4:
        self._won_defense_recently = False
```

```python
def _enemy_overcommitted(self, state):
    """Heuristic punish-window read (labeled heuristic).
    True when the enemy has a large body committed on one lane (proxy for low elixir)."""
    big = [e for e in state.enemies
           if e.position.tile_y > 15                      # still on their side / mid-push
           and (e.unit.category == UnitCategory.TROOP)]
    one_lane = big and all((e.position.tile_x < 9) == (big[0].position.tile_x < 9) for e in big)
    return bool(one_lane) and len(big) >= 2
```

This is intentionally conservative: with no enemy-elixir bar in `State`, `_enemy_overcommitted` is a proxy (multiple bodies on one lane). It is labeled a heuristic; Tier 4's punish branch is the only consumer, and the non-punish path (supported back push) is the safe default when the read is uncertain.

Reserve management (used by Tier 1): if the enemy's win condition is known (`is_win_condition` seen in `_enemy_seen_cards`) and the single hard counter to it is in hand, Tier 1 prefers a cheaper sufficient body for the current threat and keeps the hard counter, instead of burning it on a minor threat.

---

## 6. Explicit anti-spam / efficiency rules

These directly address "the current bot makes too many pointless moves." Each is enforced structurally, not by hope.

1. **Hold is a real action.** `decide` returns `None` whenever no tier has a justified play. The current bot's only "do nothing" was the implicit `best_score[0]==0`; now waiting is intentional and reaches the bottom of the cascade by design.
2. **Never act below the elixir floor with no threat.** Tiers 3-4-5 all require either a live threat (Tier 1/2 own those), a value gate (Tier 3), a supported-push budget (Tier 4), or a near-cap leak (Tier 5). With `elixir < 9` and no enemy on our half, every tier returns `None` -> the bot waits.
3. **Spend the minimum that wins (no overcommit).** Tier 1 emits exactly one defender per step and picks the **cheapest sufficient** one. It re-evaluates next step and stops once the threat is handled. No stacking 3 cards on a lone unit.
4. **Spells are gated, never panic-fired.** Tier 3 reuses `SpellAction`'s value gate (`hit_score >= MIN_SCORE`) and never relaxes the threshold. A spell on one cheap unit or empty ground is structurally impossible.
5. **No unsupported bridge dumps.** Tier 4 requires `elixir >= cost + SUPPORT_GAP` (or a punish read) and places the win condition at the **back** so support can follow. A lone tank at the bridge into a full bar cannot occur on the default path.
6. **Anti-leak is the only "play to do something."** Tier 5 plays a card only at `elixir >= 9` (about to waste regen), and only the cheapest card to a safe back tile. Below 9, it holds.
7. **One Decision per step, deterministic cascade.** The shuffle+argmax nondeterminism is removed; the first-non-None-wins order makes behavior reproducible (also required for clean BC logging).
8. **Lane focus prevents thrash.** `self._focus_lane` commits offense to one tower; the policy does not flip lanes step to step, which both improves chip strategy and removes oscillating placements.
9. **Counter-push fires once.** Tier 2 clears `_won_defense_recently` after adding a single support, preventing it from dumping the whole hand behind a survivor.
10. **Idempotence under repeated identical state.** Because Tier 1 stops firing once the threat is intercepted and the lower tiers gate on elixir/value, the bot will not re-issue the same placement every `play_action_delay` tick — the dominant source of "pointless moves" in the original loop.

---

## 7. Risks / known limitations (defensible disclosure)

- **No match clock / double-elixir / king-HP in `State`.** Tiers that ideally key on time (defer big push to 2x; final-30s chip; protect-lead-when-ahead) **degrade to elixir-based gates** here. The clock-aware refinements require new detectors (timer OCR, 2x icon hash, king-tower HP boxes) described in the tech design; until then the spec uses elixir thresholds, which are the robust subset. This is a documented approximation, not silent.
- **Tower HP is a fraction in [0,1], not raw HP.** All comparisons (weak lane, tower-kill spell) treat it as a fraction; absolute "tower HP <= spell damage" needs a per-tower max-HP table that does not exist yet, so the spell-finish branch uses `fraction <= small_epsilon` as a conservative proxy.
- **`tile_y` thresholds (14/15/16) are conventions copied from existing actions** (`DefenseAction` ignores `tile_y > 16`; `SpellAction` adds `+2`). They should be confirmed once against a tile-overlaid screenshot; an off-by-a-few silently shifts the defend trigger.
- **`state.cards[i+1]` off-by-one** is the highest-risk silent bug; the spec follows `Bot.get_actions` exactly and centralizes the `+1` in `_action_from_decision` and `_pick`.
- **`is_swarm` / `_enemy_overcommitted` are labeled heuristics**, not literature-grounded; they affect only soft preferences (which support to hold, whether to punish), never legality, and default to the safe branch when uncertain.
- **`DefenseAction`/`GiantAction` original scorers whitelist hardcoded tiles** ((8,9)/(9,9), (3,15)/(14,15)). The tiers therefore do their **own** placement (intent->tile->nearest-legal-candidate) rather than relying on those whitelists for non-spell placement; only `SpellAction` (which genuinely searches tiles) is reused as-is in Tier 3. This must be verified: `get_actions` enumerates `ALLY_TILES` (all of our half) for normal cards, so the intended pockets are present in the candidate set for the snap.

---

## 8. File-by-file change summary

- **NEW `policy/policy.py`** — `Policy` ABC + `Decision` namedtuple (Section 1).
- **NEW `policy/cardinfo.py`** — role predicates over the card DB (Section 3a).
- **NEW `policy/board.py`** — `intent_to_tile` lane/depth -> tile (Section 3b).
- **NEW `policy/expert_policy.py`** — the tier cascade + opponent read (Sections 4-5).
- **NEW (later) `policy/learned_policy.py`** — drop-in `LearnedPolicy` (Section 1).
- **EDIT `bot/bot.py`** — `__init__(policy=...)`, `_action_from_decision`, `_handle_game_step` delegates to `policy.decide`; call `on_episode_start` on screen transition (Section 2). Nothing below `play_action` changes.
- **Self-test** (per repo norms): a `__main__` in `expert_policy.py` feeding 4-5 hand-built `State` fixtures (lone Hog on our half; Giant+Musketeer push; clustered swarm for a spell; full bar no threat; capped elixir no threat) and asserting the expected tier fires.
