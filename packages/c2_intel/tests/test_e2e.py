"""
End-to-end simulation tests for c2_intel package.
Run: python -m pytest packages/c2_intel/tests/test_e2e.py -v
Or:  cd packages/c2_intel && python tests/test_e2e.py
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Allow running directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from c2_intel.models import (
    C2Observation, C2EventType, SensorSource, ObservationPriority
)
from c2_intel.resolver import C2EntityResolver
from c2_intel.graph import C2EntityGraph, build_graph_from_world_store
from c2_intel.priority import C2PriorityMatrix
from c2_intel.dedup import ObservationDeduplicator, generate_observation_fingerprint
from c2_intel.anomaly import get_anomaly_detector
from c2_intel.timing import get_timing_predictor
from c2_intel.relevance import get_relevance_model
from c2_intel.learning import (
    ObservationFeedbackLearner, ObservationFeedback,
    FeedbackType, LearningMetrics,
)
from c2_intel.chains import get_chain_detector
from c2_intel.timing_engine import get_timing_engine
from c2_intel.ontology import (
    get_ontology, list_domains
)
from c2_intel.evidence import C2EvidenceAggregator
from c2_intel.embeddings import get_embedding_service

PASSED = []
FAILED = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        PASSED.append(name)
        print(f"  PASS  {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}" + (f": {detail}" if detail else ""))


def _obs(event_type, node_id="node-alpha", source=SensorSource.MAVLINK,
         title=None, confidence=0.8, score=75, priority=ObservationPriority.HIGH,
         detected_at=None):
    return C2Observation(
        event_type=event_type,
        node_id=node_id,
        title=title or event_type.value.replace("_", " ").title(),
        source=source,
        confidence=confidence,
        score=score,
        priority=priority,
        detected_at=detected_at or datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# 1. Models
# ---------------------------------------------------------------------------
print("\n=== 1. Models ===")

obs = _obs(C2EventType.COMMS_DEGRADED)
check("C2Observation created", obs.event_type == C2EventType.COMMS_DEGRADED)
check("C2Observation has id", bool(obs.id))
check("C2Observation detected_at auto-set", obs.detected_at is not None)
check("C2EventType.LINK_DEGRADED exists", hasattr(C2EventType, "LINK_DEGRADED"))
check("C2EventType.BATTERY_LOW exists", hasattr(C2EventType, "BATTERY_LOW"))
check("C2EventType enum count >= 40", len(list(C2EventType)) >= 40)

# ---------------------------------------------------------------------------
# 2. Resolver
# ---------------------------------------------------------------------------
print("\n=== 2. Entity Resolver ===")

resolver = C2EntityResolver()
m = resolver.match("node-bravo", "mavlink-node-bravo")
check("Resolver: match found", m is not None)
check("Resolver: confidence > 0", m.confidence > 0 if m else False)

canonical = resolver.resolve("mavlink-heli-sim-01")
check("Resolver: resolve returns canonical", canonical == "heli-sim-01", f"got {canonical}")

# ---------------------------------------------------------------------------
# 3. Graph
# ---------------------------------------------------------------------------
print("\n=== 3. C2EntityGraph ===")

graph = C2EntityGraph()
nid = graph.add_node("node-alpha", sector="alpha")
aid = graph.add_asset("drone-01", domain="aerial")
check("add_node returns id", "node-alpha" in nid)
check("add_asset returns id", "drone-01" in aid)

# Check via NetworkX node count
node_count = len(graph._G.nodes) if graph._G is not None else 0
check("Graph has nodes after add", node_count >= 2)

graph.ingest_observation(
    obs_id="obs-001",
    node_id="node-alpha",
    event_type="comms_degraded",
    source="mavlink",
    score=80,
    metadata={"commanded_by": "command-hq", "peers": ["node-beta"]},
)
count_after = len(graph._G.nodes) if graph._G is not None else 0
check("ingest_observation adds nodes", count_after > node_count)

ws_entities = [
    {"entity_id": "node-charlie", "entity_type": "node",
     "commanded_by": "command-hq", "last_seen": datetime.now(timezone.utc).isoformat()},
    {"entity_id": "drone-02", "entity_type": "asset", "domain": "aerial",
     "last_seen": datetime.now(timezone.utc).isoformat()},
]
ws_graph = build_graph_from_world_store(ws_entities)
ws_count = len(ws_graph._G.nodes) if ws_graph._G is not None else 0
check("build_graph_from_world_store works", ws_count >= 2)

# ---------------------------------------------------------------------------
# 4. Priority Matrix
# ---------------------------------------------------------------------------
print("\n=== 4. Priority Matrix ===")

matrix = C2PriorityMatrix()
obs_comms = _obs(C2EventType.COMMS_DEGRADED, score=80)
priority, actions = matrix.score_observation(obs_comms)
check("score_observation returns priority", isinstance(priority, ObservationPriority))
check("COMMS_DEGRADED >= HIGH", priority in (ObservationPriority.HIGH, ObservationPriority.CRITICAL))
check("score_observation returns actions", isinstance(actions, list))

obs_threat = _obs(C2EventType.THREAT_IDENTIFIED, score=90)
obs_bat    = _obs(C2EventType.BATTERY_CRITICAL, score=85)
result = matrix.score_node_observations("node-alpha", [obs_comms, obs_threat, obs_bat])
check("score_node_observations returns dict", isinstance(result, dict))
check("score_node_observations has composite_priority", "composite_priority" in result)
check("Compound priority promotion to CRITICAL",
      result["composite_priority"] == ObservationPriority.CRITICAL,
      f"got {result.get('composite_priority')}")

# TA1 simulation: hypothetical preview
hypothetical = [obs_comms, _obs(C2EventType.AUTHORITY_DELEGATED, score=95)]
preview = matrix.score_node_observations("node-alpha", hypothetical)
check("TA1 simulate: authority delegation → CRITICAL",
      preview["composite_priority"] == ObservationPriority.CRITICAL,
      f"got {preview.get('composite_priority')}")

# ---------------------------------------------------------------------------
# 5. Deduplication
# ---------------------------------------------------------------------------
print("\n=== 5. Deduplication ===")

deduper = ObservationDeduplicator()
now = datetime.now(timezone.utc)

obs_a = _obs(C2EventType.ENTITY_DETECTED, node_id="node-bravo",
             source=SensorSource.RADAR, title="Entity Detected", detected_at=now)
obs_b = _obs(C2EventType.ENTITY_DETECTED, node_id="node-bravo",
             source=SensorSource.ADS_B, title="Entity Detected", detected_at=now)  # same ts
obs_c = _obs(C2EventType.BATTERY_CRITICAL, node_id="node-bravo",
             source=SensorSource.MAVLINK, title="Battery Critical", detected_at=now)

deduped = deduper.deduplicate([obs_a, obs_b, obs_c])
check("Dedup: same-type same-title → 2 unique", len(deduped) == 2,
      f"got {len(deduped)}")
check("Dedup: picks higher-quality sensor",
      deduped[0].source == SensorSource.RADAR or deduped[1].source == SensorSource.MAVLINK)

fp1 = generate_observation_fingerprint(obs_a)
fp2 = generate_observation_fingerprint(obs_b)
check("Fingerprint: same-title same-bucket matches", fp1 == fp2, f"{fp1} != {fp2}")

# ---------------------------------------------------------------------------
# 6. Anomaly (graceful fallback — no trained model)
# ---------------------------------------------------------------------------
print("\n=== 6. Anomaly Detector ===")

detector = get_anomaly_detector()
boost = detector.get_anomaly_boost("comms_degraded", obs_last_5m=3, obs_last_30m=8, obs_last_90m=12)
check("Anomaly boost returns int", isinstance(boost, int))
check("Anomaly boost in range [0,15]", 0 <= boost <= 15)

# ---------------------------------------------------------------------------
# 7. Timing Predictor (graceful fallback — no trained model)
# ---------------------------------------------------------------------------
print("\n=== 7. Timing Predictor ===")

predictor = get_timing_predictor()
result = predictor.predict_window("comms_degraded", "urban_sar", score=75, n_obs=3)
check("Timing predictor returns None or dict without model", result is None or isinstance(result, dict))

# ---------------------------------------------------------------------------
# 8. Relevance Model (graceful fallback — no trained model)
# ---------------------------------------------------------------------------
print("\n=== 8. Relevance Model ===")

rel = get_relevance_model()
score = rel.predict_base_score("threat_identified", "radar", confidence=0.9,
                               priority="HIGH", is_dismissed=False)
check("Relevance model returns None or int without model", score is None or isinstance(score, int))

# ---------------------------------------------------------------------------
# 9. Learning
# ---------------------------------------------------------------------------
print("\n=== 9. Feedback Learning ===")

learner = ObservationFeedbackLearner(min_samples=3)

# Record 6 feedbacks so quality_score threshold (< 5) is exceeded
for i in range(6):
    learner.record_feedback(ObservationFeedback(
        observation_id=f"obs-{i}",
        operator_id="op-alpha",
        feedback_type=FeedbackType.ACTIONED,
        event_type="THREAT_IDENTIFIED",
        sensor_source="RADAR",
        confidence=0.9,
        score=80,
    ))

adj = learner.get_score_adjustment("THREAT_IDENTIFIED", "RADAR", "op-alpha")
check("Learning: engagement → score boost > 1.0", adj > 1.0, f"adj={adj}")

learner2 = ObservationFeedbackLearner(min_samples=3)
for i in range(6):
    learner2.record_feedback(ObservationFeedback(
        observation_id=f"obs-fp-{i}",
        operator_id="op-beta",
        feedback_type=FeedbackType.FALSE_POSITIVE,
        event_type="ENTITY_DETECTED",
        sensor_source="OPENSKY",
        confidence=0.4,
        score=30,
    ))

adj2 = learner2.get_score_adjustment("ENTITY_DETECTED", "OPENSKY", "op-beta")
check("Learning: false positives → score penalty < 1.0", adj2 < 1.0, f"adj2={adj2}")

metrics = LearningMetrics(total_observations=10, actioned_count=8, resolved_count=7)
check("LearningMetrics quality_score > 0.5 with good data", metrics.quality_score > 0.5)

# ---------------------------------------------------------------------------
# 10. Signal Chains
# ---------------------------------------------------------------------------
print("\n=== 10. Signal Chains ===")

chain_detector = get_chain_detector()
chain_obs = [
    _obs(C2EventType.COMMS_DEGRADED, node_id="node-bravo"),
    _obs(C2EventType.LINK_DEGRADED, node_id="node-bravo", source=SensorSource.RADAR),
]
predictions = chain_detector.predict(chain_obs)
check("Chain: comms+link → predictions", len(predictions) > 0)
check("Chain: predicts node failure",
      any("node_failed" in p.event_type.lower() for p in predictions),
      f"predictions: {[p.event_type for p in predictions]}")

summary = chain_detector.chain_summary(chain_obs)
check("Chain summary is dict with chain_count", isinstance(summary, dict) and "chain_count" in summary)

battery_obs = [_obs(C2EventType.BATTERY_CRITICAL, node_id="drone-01")]
bat_preds = chain_detector.predict(battery_obs)
check("Chain: battery critical → asset offline predicted",
      any("asset_offline" in p.event_type.lower() for p in bat_preds))

# ---------------------------------------------------------------------------
# 11. Timing Engine (doctrine-seeded)
# ---------------------------------------------------------------------------
print("\n=== 11. Timing Engine ===")

timing = get_timing_engine()
insight = timing.get_insight("comms_degraded", context="urban_sar")
check("Timing engine returns insight", isinstance(insight, dict))
check("Timing engine has median_minutes", "median_minutes" in insight)
check("COMMS_DEGRADED median < 10min", insight["median_minutes"] < 10,
      f"got {insight.get('median_minutes')}")

multi = timing.get_multi_event_brief(
    ["comms_degraded", "battery_critical", "threat_identified"], "urban_sar"
)
check("Multi-event brief returns dict", isinstance(multi, dict))
check("Multi-event brief has most_urgent_event", "most_urgent_event" in multi)

timing.record_outcome("comms_degraded", minutes_to_command=2.5, escalated=True)
insight2 = timing.get_insight("comms_degraded", context="urban_sar")
check("Timing engine records outcome without error", isinstance(insight2, dict))

# ---------------------------------------------------------------------------
# 12. Ontology
# ---------------------------------------------------------------------------
print("\n=== 12. Domain Ontology ===")

domains = list_domains()
check("Ontology: 4 domains registered", len(domains) >= 4, f"got {domains}")

wildfire = get_ontology("wildfire")
check("Wildfire ontology loaded", wildfire is not None)
play = wildfire.get_action_play(["comms_degraded"], entity_id="node-alpha", composite_score=80)
check("Wildfire: comms → action play", play is not None)
check("Action play has recommended_action", play is not None and bool(play.recommended_action))

military = get_ontology("military_ace")
check("Military ACE ontology loaded", military is not None)
mil_play = military.get_action_play(["comms_degraded"], entity_id="node-bravo", composite_score=85)
check("Military ACE: comms → PACE plan", mil_play is not None)

headline = wildfire.format_brief_headline(entity_id="sector-7", top_event="sensor_loss", score=75)
check("Wildfire headline is string", isinstance(headline, str) and len(headline) > 0)

# ---------------------------------------------------------------------------
# 13. Evidence Aggregation
# ---------------------------------------------------------------------------
print("\n=== 13. Evidence Aggregation ===")

agg = C2EvidenceAggregator()
now = datetime.now(timezone.utc)
evidence_obs = [
    _obs(C2EventType.COMMS_DEGRADED, node_id="node-bravo",
         source=SensorSource.MAVLINK, detected_at=now),
    _obs(C2EventType.LINK_DEGRADED, node_id="node-bravo",
         source=SensorSource.RADAR, detected_at=now + timedelta(seconds=90)),
    _obs(C2EventType.COMMS_DEGRADED, node_id="node-bravo",
         source=SensorSource.MESH_PEER, detected_at=now + timedelta(seconds=180)),
    _obs(C2EventType.THREAT_IDENTIFIED, node_id="node-bravo",
         source=SensorSource.EO_IR, score=90, detected_at=now + timedelta(seconds=60)),
]

clusters = agg.aggregate(evidence_obs, entity_id="node-bravo")
check("Evidence: clusters returned", len(clusters) > 0)
check("Evidence: comms cluster exists",
      any(c.obs_category == "comms" for c in clusters))
check("Evidence: comms+threat → combination_context",
      any(c.combination_context is not None for c in clusters),
      f"contexts: {[c.combination_context for c in clusters]}")

top = clusters[0]
check("Evidence cluster has hypothesis", bool(top.hypothesis))
check("Evidence cluster has answer", bool(top.answer))
check("Evidence: to_dict works", isinstance(top.to_dict(), dict))

# ---------------------------------------------------------------------------
# 14. Embeddings (graceful fallback — no API key in test)
# ---------------------------------------------------------------------------
print("\n=== 14. Embeddings ===")

import asyncio
svc = get_embedding_service()
check("Embedding service created", svc is not None)
status = svc.get_status()
check("Embedding status has keys", "initialized" in status and "cache_size" in status)

result = asyncio.run(svc.embed_text("Node-Bravo comms degraded on MAVLINK link"))
check("embed_text returns None when no key (graceful)", result is None or isinstance(result, list))

sim = svc.cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
check("cosine_similarity returns 1.0 for identical", abs(sim - 1.0) < 1e-6)

# ---------------------------------------------------------------------------
# 15. Full OODA Loop Simulation
# ---------------------------------------------------------------------------
print("\n=== 15. Full OODA Loop Simulation ===")

"""
Observe → Orient → Decide → Act
Scenario: Node-Bravo is losing comms from two sensor sources simultaneously.
Multiple mesh peers confirm the same event. Battery also degrading.
System must: dedup → prioritize → chain-predict → evidence-cluster → timing.
"""

t0 = datetime.now(timezone.utc)

# OBSERVE: raw observations from multiple sensors
raw_obs = [
    C2Observation(
        event_type=C2EventType.COMMS_DEGRADED,
        node_id="node-bravo",
        title="Comms Degraded",  # identical title → same fingerprint
        source=SensorSource.MAVLINK,
        confidence=0.92,
        score=78,
        priority=ObservationPriority.HIGH,
        detected_at=t0,
    ),
    C2Observation(
        event_type=C2EventType.COMMS_DEGRADED,
        node_id="node-bravo",
        title="Comms Degraded",  # same title, same bucket → dedup as duplicate
        source=SensorSource.MESH_PEER,
        confidence=0.75,
        score=65,
        priority=ObservationPriority.HIGH,
        detected_at=t0,  # same timestamp ensures same 30s bucket
    ),
    C2Observation(
        event_type=C2EventType.LINK_DEGRADED,
        node_id="node-bravo",
        title="Link Degraded",
        source=SensorSource.RADAR,
        confidence=0.87,
        score=82,
        priority=ObservationPriority.HIGH,
        detected_at=t0 + timedelta(seconds=15),
    ),
    C2Observation(
        event_type=C2EventType.BATTERY_CRITICAL,
        node_id="node-bravo",
        title="Battery Critical",
        source=SensorSource.MAVLINK,
        confidence=0.95,
        score=90,
        priority=ObservationPriority.HIGH,
        detected_at=t0 + timedelta(seconds=20),
    ),
]

# ORIENT: dedup
deduper2 = ObservationDeduplicator()
deduped2 = deduper2.deduplicate(raw_obs)
check("OODA: dedup removes MESH_PEER dupe", len(deduped2) == 3,
      f"got {len(deduped2)} (expected 3 after dedup)")

# ORIENT: priority
matrix2 = C2PriorityMatrix()
node_result = matrix2.score_node_observations("node-bravo", deduped2)
check("OODA: composite priority is CRITICAL",
      node_result["composite_priority"] == ObservationPriority.CRITICAL,
      f"got {node_result['composite_priority']}")

# ORIENT: signal chains
preds2 = chain_detector.predict(deduped2)
check("OODA: chain predictions generated", len(preds2) > 0)

# ORIENT: evidence clusters
clusters2 = agg.aggregate(deduped2, entity_id="node-bravo")
check("OODA: evidence clusters generated", len(clusters2) > 0)
has_combo = any(c.combination_context for c in clusters2)
check("OODA: comms+health combination detected", has_combo,
      f"contexts: {[c.combination_context for c in clusters2]}")

# DECIDE: timing
t_insight = timing.get_insight("comms_degraded", context="urban_sar")
action_window = t_insight.get("median_minutes", 5)
check("OODA: timing insight for comms degraded", action_window > 0)

# ACT: TA1 simulate authority delegation
hypothetical_delegation = deduped2 + [
    C2Observation(
        event_type=C2EventType.AUTHORITY_DELEGATED,
        node_id="node-bravo",
        title="Authority Delegated",
        source=SensorSource.WORLD_STORE,
        confidence=1.0,
        score=95,
        priority=ObservationPriority.CRITICAL,
        detected_at=t0 + timedelta(seconds=30),
    )
]
ta1_preview = matrix2.score_node_observations("node-bravo", hypothetical_delegation)
check("OODA TA1: authority delegation simulation works",
      ta1_preview["composite_priority"] == ObservationPriority.CRITICAL)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = len(PASSED) + len(FAILED)
print(f"\n{'='*60}")
print(f"Results: {len(PASSED)}/{total} passed")
if FAILED:
    print(f"\nFailed ({len(FAILED)}):")
    for f in FAILED:
        print(f"  - {f}")
print('='*60)

if FAILED:
    sys.exit(1)
