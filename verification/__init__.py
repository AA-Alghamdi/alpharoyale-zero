"""Cross-engine verification harness.

Compares the crsim simulator against a second, independently-built Clash Royale
engine (the "oracle", samdickson22/clash-simulator) so that agreement between
two independent implementations of authentic game data raises confidence in
fidelity beyond what golden tests alone provide.

- :mod:`verification.conformance` — kind-aware card-stat comparison.
- :mod:`verification.report` — human-readable conformance report.
- :mod:`verification.behavioral` — scenario-level behavioural comparison
  (requires a local oracle clone; not run in CI).
"""
