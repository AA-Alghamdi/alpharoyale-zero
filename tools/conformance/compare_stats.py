"""Cross-engine + ground-truth stat conformance for the CR simulator.

This is the GPU-free accuracy gate. It answers "how accurate is the engine?"
in a measurable, reproducible way by comparing our card stats against:

1. **Reality** — a small set of hand-verified Level-11 wiki anchors
   (``wiki_anchors.py``). This is authoritative but narrow.
2. **An independent engine** — samdickson22/clash-simulator, a separately
   authored Python CR engine sourced from its own ``gamedata.json``. Two
   independent engines agreeing is strong evidence of correctness; a
   disagreement localises a bug (in one of them) to a specific card/stat.

Run::

    python -m tools.conformance.compare_stats                 # human report
    python -m tools.conformance.compare_stats --out report.md # write markdown
    CLASHER_DIR=/path/to/clash-simulator python -m tools.conformance.compare_stats

The reference engine is optional: point ``--clasher`` / ``$CLASHER_DIR`` at a
clone of clash-simulator. If it is absent the wiki-anchor check still runs.

Nothing here imports the reference at module load, and the reference's noisy
import-time logging is suppressed, so this is safe to import from tests.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import statistics as st
import sys
from dataclasses import dataclass

from crsim.cards import CARD_DEFS

from .wiki_anchors import WIKI_ANCHORS

# Cards whose name does not normalise cleanly to the reference's english name.
# Maps our normalised CardType name -> reference normalised english name.
_NAME_OVERRIDES = {
    "minionhorde": "minions",  # ref stores the single-minion stat line
    "threemusketeers": "musketeer",
    "skeletonarmy": "skeletons",
    "elitebarbarians": "elitebarbarian",
}

# Outlier threshold: |delta| above this (percent) is flagged for review.
OUTLIER_PCT = 5.0


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


@dataclass
class StatRow:
    card: str
    our_hp: float
    our_dmg: float
    our_hs: float
    ref_hp: float | None = None
    ref_dmg: float | None = None
    ref_hs: float | None = None

    @staticmethod
    def _pct(ours: float, ref: float | None) -> float | None:
        if ref is None or ref == 0:
            return None
        return 100.0 * (ours - ref) / ref

    @property
    def hp_delta(self) -> float | None:
        return self._pct(self.our_hp, self.ref_hp) if self.our_hp else None

    @property
    def dmg_delta(self) -> float | None:
        return self._pct(self.our_dmg, self.ref_dmg) if self.our_dmg else None


def _load_reference(clasher_dir: str) -> dict[str, tuple[float | None, float | None, float | None]]:
    """Load reference card stats keyed by normalised english name.

    Returns ``{norm_name: (scaled_hp, scaled_damage, hit_speed_seconds)}``.
    Returns an empty dict if the reference cannot be loaded.
    """
    gamedata = os.path.join(clasher_dir, "gamedata.json")
    if not os.path.isdir(clasher_dir) or not os.path.isfile(gamedata):
        return {}

    saved_path = list(sys.path)
    saved_cwd = os.getcwd()
    out: dict[str, tuple[float | None, float | None, float | None]] = {}
    try:
        sys.path.insert(0, clasher_dir)
        os.chdir(clasher_dir)  # the loader reads ./gamedata.json by relative path
        # The reference prints a wall of [Detect]/[Lifecycle] lines on load.
        with contextlib.redirect_stdout(io.StringIO()):
            from src.clasher.data import CardDataLoader  # type: ignore

            cards = CardDataLoader("gamedata.json").load_cards()
        for key, c in cards.items():
            name = getattr(c, "english_name", None) or getattr(c, "name", key)
            if not name:
                continue
            try:
                hp = c.scaled_hitpoints
            except Exception:
                hp = None
            try:
                dmg = c.scaled_damage
            except Exception:
                dmg = None
            hs_ms = getattr(c, "hit_speed", None)
            hs = (hs_ms / 1000.0) if hs_ms else None
            out[_norm(name)] = (hp, dmg, hs)
    except Exception as exc:  # pragma: no cover - reference is best-effort
        print(f"[warn] could not load reference engine: {exc}", file=sys.stderr)
        out = {}
    finally:
        sys.path[:] = saved_path
        os.chdir(saved_cwd)
        # Drop any reference modules so repeated runs re-import cleanly.
        for mod in [m for m in sys.modules if m.startswith("src.clasher")]:
            del sys.modules[mod]
    return out


def build_rows(clasher_dir: str | None) -> list[StatRow]:
    ref = _load_reference(clasher_dir) if clasher_dir else {}
    rows: list[StatRow] = []
    for ct, cd in sorted(CARD_DEFS.items(), key=lambda kv: kv[0].name):
        n = _norm(ct.name)
        n = _NAME_OVERRIDES.get(n, n)
        rhp = rdmg = rhs = None
        if n in ref:
            rhp, rdmg, rhs = ref[n]
        rows.append(
            StatRow(
                card=ct.name,
                our_hp=float(getattr(cd, "hp", 0.0) or 0.0),
                our_dmg=float(getattr(cd, "damage_per_hit", 0.0) or 0.0),
                our_hs=float(getattr(cd, "hit_speed", 0.0) or 0.0),
                ref_hp=rhp,
                ref_dmg=rdmg,
                ref_hs=rhs,
            )
        )
    return rows


def anchor_report() -> list[str]:
    """Compare our stats to verified wiki Level-11 values (vs reality)."""
    lines = ["## Ground-truth anchors (our engine vs verified wiki L11)", ""]
    lines.append("| Card | Stat | Ours | Wiki | delta% | status | source |")
    lines.append("|---|---|---:|---:|---:|---|---|")
    n_pass = n_fail = 0
    for name, anc in sorted(WIKI_ANCHORS.items()):
        cd = next((d for ct, d in CARD_DEFS.items() if ct.name == name), None)
        if cd is None:
            continue
        for stat, ours_attr, wiki_val in (
            ("hp", "hp", anc.hp),
            ("damage", "damage_per_hit", anc.damage),
        ):
            if wiki_val is None:
                continue
            ours = float(getattr(cd, ours_attr, 0.0) or 0.0)
            delta = 100.0 * (ours - wiki_val) / wiki_val if wiki_val else 0.0
            ok = abs(delta) <= 1.0  # within rounding
            n_pass += ok
            n_fail += (not ok)
            status = "OK" if ok else "**MISMATCH**"
            lines.append(
                f"| {name} | {stat} | {ours:.0f} | {wiki_val:.0f} | "
                f"{delta:+.1f} | {status} | {anc.source} |"
            )
    lines.append("")
    lines.append(f"**Anchor result: {n_pass} pass, {n_fail} mismatch** "
                 "(tolerance ±1%).")
    lines.append("")
    return lines


def cross_engine_report(rows: list[StatRow]) -> list[str]:
    matched = [r for r in rows if r.ref_hp is not None or r.ref_dmg is not None]
    if not matched:
        return [
            "## Cross-engine comparison (vs samdickson22/clash-simulator)",
            "",
            "_Reference engine not found — set `--clasher` or `$CLASHER_DIR` to a "
            "clone of clash-simulator to enable this section._",
            "",
        ]

    hp_d = [abs(r.hp_delta) for r in matched if r.hp_delta is not None]
    dmg_d = [abs(r.dmg_delta) for r in matched if r.dmg_delta is not None]
    outliers = [
        r
        for r in matched
        if (r.hp_delta is not None and abs(r.hp_delta) > OUTLIER_PCT)
        or (r.dmg_delta is not None and abs(r.dmg_delta) > OUTLIER_PCT)
    ]

    lines = ["## Cross-engine comparison (our engine vs clash-simulator)", ""]
    lines.append(f"- Cards compared: **{len(matched)}**")
    if hp_d:
        lines.append(
            f"- HP   |delta|: mean **{st.mean(hp_d):.2f}%**, "
            f"median **{st.median(hp_d):.2f}%**, max {max(hp_d):.1f}%"
        )
    if dmg_d:
        lines.append(
            f"- Dmg  |delta|: mean **{st.mean(dmg_d):.2f}%**, "
            f"median **{st.median(dmg_d):.2f}%**, max {max(dmg_d):.1f}%"
        )
    # Signed median tells us about systematic offset (scaling-convention drift).
    signed_hp = [r.hp_delta for r in matched if r.hp_delta is not None]
    if signed_hp:
        lines.append(
            f"- HP signed median: **{st.median(signed_hp):+.2f}%** "
            "(systematic offset between the two engines' level-scaling)"
        )
    lines.append("")
    lines.append(f"### Outliers (|delta| > {OUTLIER_PCT:.0f}% — investigate)")
    lines.append("")
    lines.append("| Card | ourHP | refHP | HPd% | ourDmg | refDmg | Dmgd% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in sorted(outliers, key=lambda r: -(abs(r.hp_delta or 0) + abs(r.dmg_delta or 0))):
        lines.append(
            f"| {r.card} | {r.our_hp:.0f} | {(r.ref_hp or 0):.0f} | "
            f"{(r.hp_delta or 0):+.1f} | {r.our_dmg:.0f} | {(r.ref_dmg or 0):.0f} | "
            f"{(r.dmg_delta or 0):+.1f} |"
        )
    lines.append("")
    return lines


def generate_report(clasher_dir: str | None) -> str:
    rows = build_rows(clasher_dir)
    out = ["# CR Simulator — Stat Conformance Report", ""]
    out += anchor_report()
    out += cross_engine_report(rows)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--clasher",
        default=os.environ.get("CLASHER_DIR", os.path.expanduser("~/refs/clash-simulator")),
        help="Path to a clone of samdickson22/clash-simulator (for cross-engine diff).",
    )
    ap.add_argument("--out", default=None, help="Write the markdown report to this file.")
    args = ap.parse_args(argv)

    report = generate_report(args.clasher)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
