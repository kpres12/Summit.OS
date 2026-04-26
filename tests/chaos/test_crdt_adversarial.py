"""
CRDT Adversarial / Property Tests
===================================
Probes the CRDT implementations for correctness under adversarial conditions.
All three fundamental CRDT properties are verified mathematically, plus
real-world failure modes: clock skew, Byzantine flooding, split-brain, and
the tombstone memory growth problem in ORSet.

Properties verified per CRDT:
  - Commutativity:  merge(A, B) == merge(B, A)
  - Associativity:  merge(merge(A,B), C) == merge(A, merge(B,C))
  - Idempotency:    merge(A, A) == A

Adversarial scenarios:
  - LWW: concurrent writes from N nodes converge to highest timestamp
  - LWW: clock-skewed node always wins (known hazard, documented)
  - ORSet: concurrent add+remove resolves deterministically (add wins)
  - ORSet: tombstone growth under add/remove cycling (memory hazard)
  - CRDTStore: split-brain partition then heal converges
  - GCounter/PNCounter: Byzantine flooding doesn't corrupt other nodes
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))

from mesh.crdt import LWWRegister, GCounter, PNCounter, ORSet, CRDTStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node(n: int) -> str:
    return f"node-{n}"


def _store(node_id: str) -> CRDTStore:
    return CRDTStore(node_id=node_id)


# ---------------------------------------------------------------------------
# LWW Register — mathematical properties
# ---------------------------------------------------------------------------

def test_lww_commutativity():
    a = LWWRegister(_node(1), "A", timestamp=1.0)
    b = LWWRegister(_node(2), "B", timestamp=2.0)
    ab = a.merge(b)
    ba = b.merge(a)
    assert ab.value == ba.value, "LWW merge must be commutative"
    assert ab.timestamp == ba.timestamp


def test_lww_associativity():
    a = LWWRegister(_node(1), "A", timestamp=1.0)
    b = LWWRegister(_node(2), "B", timestamp=2.0)
    c = LWWRegister(_node(3), "C", timestamp=3.0)
    left  = a.merge(b).merge(c)
    right = a.merge(b.merge(c))
    assert left.value == right.value, "LWW merge must be associative"


def test_lww_idempotency():
    a = LWWRegister(_node(1), "A", timestamp=1.0)
    once  = a.merge(a)
    twice = a.merge(a).merge(a)
    assert once.value == twice.value == a.value, "LWW merge must be idempotent"


def test_lww_concurrent_writes_converge():
    """30 nodes all write to the same register concurrently. The merge of all
    should converge to the node with the highest timestamp."""
    N = 30
    registers = [
        LWWRegister(_node(i), f"value-{i}", timestamp=float(i))
        for i in range(N)
    ]
    # Merge them in random-ish order (both directions)
    result_fwd = registers[0]
    for r in registers[1:]:
        result_fwd = result_fwd.merge(r)

    result_rev = registers[-1]
    for r in reversed(registers[:-1]):
        result_rev = result_rev.merge(r)

    assert result_fwd.value == result_rev.value == f"value-{N-1}", \
        "All merge orders must converge to highest-timestamp value"


def test_lww_clock_skew_known_hazard():
    """DOCUMENTED HAZARD: a node with a clock 100s ahead always wins LWW,
    regardless of 'real' write order. This test documents the behavior —
    if it ever starts failing, our clock-sync assumptions changed."""
    future_node = LWWRegister(_node(99), "future-value", timestamp=time.time() + 100.0)
    current_node = LWWRegister(_node(1), "current-value", timestamp=time.time())

    merged = current_node.merge(future_node)
    assert merged.value == "future-value", \
        "KNOWN HAZARD: future-clocked node always wins LWW. " \
        "Mitigation: bound clock skew via NTP + AntiReplayFilter timestamp check."


def test_lww_tie_break_deterministic():
    """Tie (same timestamp) must resolve deterministically by node_id, not randomly."""
    ts = 42.0
    a = LWWRegister("node-alpha", "alpha", timestamp=ts)
    b = LWWRegister("node-zeta",  "zeta",  timestamp=ts)

    ab = a.merge(b)
    ba = b.merge(a)
    assert ab.value == ba.value, "Tie-break must be deterministic"
    # Higher node_id wins (lexicographic)
    assert ab.value == "zeta", "node-zeta > node-alpha lexicographically"


# ---------------------------------------------------------------------------
# ORSet — concurrent add+remove, tombstone growth
# ---------------------------------------------------------------------------

def test_orset_commutativity():
    a = ORSet(_node(1))
    b = ORSet(_node(2))
    a.add("x"); b.add("x"); b.remove("x")

    ab = a.merge(b)
    ba = b.merge(a)
    assert ab.values() == ba.values(), "ORSet merge must be commutative"


def test_orset_associativity():
    a, b, c = ORSet(_node(1)), ORSet(_node(2)), ORSet(_node(3))
    a.add("x"); b.add("x"); b.remove("x"); c.add("x")

    left  = a.merge(b).merge(c)
    right = a.merge(b.merge(c))
    assert left.values() == right.values(), "ORSet merge must be associative"


def test_orset_idempotency():
    s = ORSet(_node(1))
    s.add("x"); s.add("y"); s.remove("x")

    once  = s.merge(s)
    twice = s.merge(s).merge(s)
    assert once.values() == twice.values() == s.values(), \
        "ORSet merge must be idempotent"


def test_orset_concurrent_add_remove_add_wins():
    """Concurrent add (node A) + remove (node B) = element present.
    This is the ORSet semantic guarantee: add always wins over concurrent remove."""
    adder  = ORSet(_node(1))
    remover = ORSet(_node(2))

    adder.add("target")
    remover.add("target")
    remover.remove("target")

    # Concurrent: adder has "target" added, remover has "target" removed
    merged = adder.merge(remover)
    assert merged.contains("target"), \
        "ORSet: concurrent add+remove must resolve with element PRESENT (add wins). " \
        "If this fails, the ORSet doesn't meet the CvRDT spec."


def test_orset_remove_then_readd():
    """Remove then re-add must result in element being present."""
    s = ORSet(_node(1))
    s.add("x")
    s.remove("x")
    assert not s.contains("x")
    s.add("x")
    assert s.contains("x"), "Re-added element must be present"


def test_orset_tombstone_growth_hazard():
    """DOCUMENTED HAZARD: ORSet tombstones grow without bound.
    After N add/remove cycles, tombstones set has N entries.
    This test documents the growth rate so we know when it becomes a problem."""
    s = ORSet(_node(1))
    N = 1_000

    for i in range(N):
        s.add(f"item-{i % 10}")  # only 10 distinct items, but N tombstones
        s.remove(f"item-{i % 10}")

    # Document: tombstones grew to ~N
    tombstone_count = len(s.tombstones)
    assert tombstone_count >= N * 0.9, \
        f"Expected ~{N} tombstones from {N} removes, got {tombstone_count}"

    # The hazard: in production, this would grow to millions of entries.
    # Mitigation needed: periodic tombstone compaction or epoch-based reset.
    # This test is a TRIP WIRE — if tombstone_count < 100 after 1000 removes,
    # something changed in the ORSet implementation.
    assert tombstone_count < N * 10, \
        f"Tombstone growth exploded unexpectedly: {tombstone_count}"


def test_orset_values_correct_after_merge():
    """After merging two diverged ORSets, values() must reflect the true union."""
    partition_a = ORSet(_node(1))
    partition_b = ORSet(_node(2))

    partition_a.add("alpha")
    partition_a.add("shared")
    partition_b.add("beta")
    partition_b.add("shared")
    partition_b.remove("shared")

    merged = partition_a.merge(partition_b)
    # "shared" was added by A but removed by B concurrently → add wins
    assert "alpha" in merged.values()
    assert "beta" in merged.values()
    assert "shared" in merged.values(), \
        "concurrent add(A)+remove(B) of 'shared' must leave it present"


# ---------------------------------------------------------------------------
# GCounter / PNCounter
# ---------------------------------------------------------------------------

def test_gcounter_commutativity():
    a = GCounter(_node(1)); a.increment(5)
    b = GCounter(_node(2)); b.increment(3)
    assert a.merge(b).value() == b.merge(a).value() == 8


def test_pncounter_net_value():
    c = PNCounter(_node(1))
    c.increment(10)
    c.decrement(3)
    assert c.value() == 7


def test_pncounter_merge_converges():
    a = PNCounter(_node(1)); a.increment(5)
    b = PNCounter(_node(2)); b.increment(3); b.decrement(1)
    merged_ab = a.merge(b)
    merged_ba = b.merge(a)
    assert merged_ab.value() == merged_ba.value() == 7


def test_gcounter_byzantine_flooding():
    """A Byzantine node floods incrementing its own counter to INT_MAX equivalent.
    Other nodes' counters must be unaffected."""
    legitimate = GCounter(_node(1))
    legitimate.increment(5)

    byzantine = GCounter(_node(99))
    byzantine.increment(10_000_000)

    merged = legitimate.merge(byzantine)
    # The merged value includes the byzantine node's count (that's correct —
    # it's a G-counter, not a trust-filtered system). What must NOT happen:
    # legitimate's own count must be preserved.
    assert merged.counts.get(_node(1), 0) == 5, \
        "Byzantine flooding must not corrupt other nodes' counts"
    assert merged.value() == 10_000_005


# ---------------------------------------------------------------------------
# CRDTStore — split-brain partition and heal
# ---------------------------------------------------------------------------

def test_crdt_store_split_brain_converges():
    """Simulate a network partition: two groups of nodes diverge, then reconnect.
    After merge, all stores must reflect the same state."""
    # Partition A: nodes 1-3
    stores_a = [_store(f"a{i}") for i in range(3)]
    # Partition B: nodes 4-6
    stores_b = [_store(f"b{i}") for i in range(3)]

    # Each partition writes independently
    for i, s in enumerate(stores_a):
        s.get_register("entity.hostile").set(f"hostile-a{i}", t=float(i + 1))
        s.get_counter("engagements").increment(i + 1)

    for i, s in enumerate(stores_b):
        s.get_register("entity.hostile").set(f"hostile-b{i}", t=float(i + 10))
        s.get_counter("engagements").increment(i + 10)

    # Partition A converges internally
    for s in stores_a[1:]:
        stores_a[0].merge(s)

    # Partition B converges internally
    for s in stores_b[1:]:
        stores_b[0].merge(s)

    # Heal: merge the two partition leaders
    stores_a[0].merge(stores_b[0])
    stores_b[0].merge(stores_a[0])

    # After heal: both leaders must have identical state
    reg_a = stores_a[0].get_register("entity.hostile").get()
    reg_b = stores_b[0].get_register("entity.hostile").get()
    assert reg_a == reg_b, \
        f"After partition heal, LWW registers must converge: {reg_a!r} != {reg_b!r}"

    cnt_a = stores_a[0].get_counter("engagements").value()
    cnt_b = stores_b[0].get_counter("engagements").value()
    assert cnt_a == cnt_b, \
        f"After partition heal, PNCounters must converge: {cnt_a} != {cnt_b}"


def test_crdt_store_merge_idempotent_under_replay():
    """Receiving the same delta N times must be identical to receiving it once.
    This is critical for mesh re-sync after reconnection."""
    source = _store("source")
    source.get_register("track.hostile").set("target-001", t=1.0)
    source.get_set("active_missions").add("mission-alpha")

    dest = _store("dest")

    # Apply the same delta 10 times (simulating re-delivery)
    for _ in range(10):
        dest.merge(source)

    single = _store("single")
    single.merge(source)

    assert dest.get_register("track.hostile").get() == \
           single.get_register("track.hostile").get(), \
        "Repeated merge must be idempotent"
    assert dest.get_set("active_missions").values() == \
           single.get_set("active_missions").values(), \
        "ORSet repeated merge must be idempotent"


# ---------------------------------------------------------------------------
# Scale: merge performance under large state
# ---------------------------------------------------------------------------

def test_crdt_store_large_merge_performance():
    """Merge two stores each holding 10,000 registers. Must complete in < 5s.
    Regression guard against O(N²) merge implementations."""
    a = _store("perf-a")
    b = _store("perf-b")

    N = 10_000
    for i in range(N):
        a.get_register(f"entity.{i}").set(f"state-a-{i}", t=float(i) + 1.0)
        b.get_register(f"entity.{i}").set(f"state-b-{i}", t=float(i) + 2.0)

    t0 = time.perf_counter()
    a.merge(b)
    elapsed = time.perf_counter() - t0

    assert elapsed < 5.0, \
        f"Merging two 10k-register stores took {elapsed:.2f}s — check for O(N²) merge"
    # Verify correctness: b's values should win (higher timestamp +2.0 > +1.0)
    assert a.get_register("entity.0").get() == "state-b-0"
