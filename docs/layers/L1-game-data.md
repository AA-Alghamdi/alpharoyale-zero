# L1 — Game Data & Rules

The bottom layer: the authoritative card/stat data every backend reads. The key fact that
makes everything else possible is that **the engine (and our sims) read stats by *column
name* and ignore unrecognized columns** — so data is additive and modern cards are partially
portable.

## Sources of truth

| Path | What | Used by |
|------|------|---------|
| `cr_engine/gamedata_v2/*.csv` | Extracted Supercell APK tables: `characters.csv`, `projectiles.csv`, `spells_characters.csv`, `spells_buildings.csv`, `spells_other.csv`, `area_effect_objects.csv` | `cr_engine/src/data.rs`, `crsim/gamedata.py` |
| `cr_engine/gamedata_full/*.json` | Modern roster: `cards.json`, `cards_evo.json`, `cards_stats_{characters,building,projectile,spell}.json`, `rarities.json` | champion/evolution features |
| `crsim/gamedata.py` | Python port of `data.rs` — parses the 6 CSVs + applies level scaling | crsim |
| `crsim/cards.py`, `constants.py` | Card roster + arena/timing/action constants | crsim, features, env |

`checklist.txt` (on the feature branches) is the running source of truth for the
"1:1 replica" data effort and documents exactly what's verified vs pending.

## Level scaling (a real, subtle correctness issue)

Stats in the CSVs are stored at **level 1**; cards in battle are at tournament-standard
levels (commonly L11). The data pipeline applies per-rarity multipliers. Two landmines already
hit and fixed:

- **Double-scaling bug**: APK rows are L1 (capital-cased rarity, must be scaled); some
  third-party rows (ClashStrategic) are *already* L11 (lowercase rarity, must NOT be scaled).
  Detection is by rarity casing. Getting this wrong silently doubles or halves damage.
- **Approximate L11 multipliers**: Legendary/Epic multipliers in `data.rs` are approximate
  (e.g. Legendary 1.3781 ≠ 1.1²). Open item: use the exact CR per-level table for <2% damage
  precision.

Validation anchors against the wiki (Knight HP 1766, Giant 4091, Musketeer ~218, Fireball 689)
catch regressions — keep these as data tests.

## Mechanics that live in data vs code

The dividing line for any card is **"can it be expressed as a stat-variant of a mechanic the
engine already implements?"**

- **Stats = data**: HP, damage, DPS, hit speed, load time, speed, range, sight, deploy time,
  cost, target mode, flying, splash, spawn count, collision radius, minimum range, death
  damage/radius, death-spawn. All read from rows; an unknown column is ignored.
- **Mechanics = code**: only triggered when data references a behavior the binary/sim already
  implements (e.g. `Projectile`, `SummonCharacter`, `AttackEffect` must point at existing
  rows). Novel behaviors with no implementation are no-ops or crashes.

Already-wired mechanics worth noting (from `checklist.txt`): minimum range (Mortar dead zone),
death damage/radius (Golem 224/2.0, Ice Golem 85/2.0; replaced an old tower-damage hack that
wrongly made Miner deal death damage), death-spawn stats, crown-tower damage reduction.

## Modern cards: the converter (downstream, non-blocking)

Adding post-2016 cards is feasible but tiered:

- **Tier 1 (~60–70%, hours each):** new troops/buildings that are stat combos of existing
  archetypes. Add rows with modern numbers; point `Projectile`/`SummonCharacter`/`AttackEffect`
  at existing entries. The engine/sim runs them correctly. **For headless RL no art is
  needed** — only logic stats are simulated.
- **Tier 2 (hard):** Champions (ability system) and Evolutions (a whole subsystem) — the 2016
  `libg.so` has neither, so on Scroll they must be stubbed to the base troop or the engine
  must be patched per mechanic (weeks each). **Our own crsim/cr_engine already added**
  `crsim/heroes.py` (champions) and `crsim/evolutions.py` (+ `hidden_stats.py`), so champions
  and evolutions are simulatable on the *Python/Rust* backends even though Scroll can't run
  them.

### Three gotchas for any data addition
1. **Referential closure** — every referenced projectile/character/effect row must exist and
   resolve, or you get a null-lookup crash. A converter must emit the transitive closure.
2. **Art** — none for headless RL (logic-only). A *playable* client must reuse an existing
   card's sprites.
3. **Checksum** — client and server must load byte-identical card data (Scroll verifies a
   checksum to detect desync). Trivial here since we control both sides, but a converter must
   write the same data everywhere.

### Recommended converter shape
A standalone tool: `read modern-card JSON → emit {gamedata_v2 CSV rows for the Python/Rust
sims} + {Scroll-schema rows}`, substituting the nearest existing behavior/art for Tier-2
mechanics, and validating referential closure before writing. A few days of work; strictly
downstream of a running battle — do not let it compete with P0.

## Compound-summon open item
Cards like Rascals (boy+girls), Goblin Gang (gob+spear), 3M currently read only the primary
`SummonCharacter`; heterogeneous spawns need `SummonCharacterSecond` wired. Six cards with no
CSV entry remain hand-coded (GoblinCage, GoblinDrill, GoblinBrawler, BushGoblins, CursedHog,
Guardienne).
