"""
c2_intel — C2 Intelligence Engine for Heli.OS

Sensors → world model → inference → tasking → human-in-the-loop.

Ported and domain-remapped from Mira Signals' commercial intelligence engine.
All modifications live here; Mira Signals codebase is read-only.

Components:
  models        — C2 domain types (C2EventType, ObservationPriority, SensorSource, C2Observation)
  resolver      — entity deconfliction across sensor sources (7-layer)
  graph         — NetworkX relationship graph (effect radius, path finding, lookalike)
  priority      — composite condition scoring + TA1 simulation layer
  dedup         — observation deduplication across redundant sensor streams
  anomaly       — Isolation Forest anomaly boost (unusual activity spike detection)
  timing        — quantile regression timing predictor (minutes to operator command)
  relevance     — gradient-boosted relevance classifier (base score prediction)
  learning      — operator feedback learning loop (score weight adjustment)
  chains        — C2 signal chain detector (predict cascade events from observations)
  timing_engine — doctrine-seeded event timing engine (OODA loop performance)
  ontology      — domain ontologies (wildfire, urban_sar, military_ace, disaster_response)
  evidence      — evidence aggregation into compound insight clusters
  embeddings    — Google AI semantic embedding service (GEMINI_API_KEY required)
"""

from .models import (
    C2EventType,
    ObservationPriority,
    SensorSource,
    C2ActionType,
    C2Observation,
)
from .resolver import C2EntityResolver, EntityMatch, normalize_entity_id
from .graph import C2EntityGraph, NodeType, EdgeType, get_c2_graph, build_graph_from_world_store
from .priority import C2PriorityMatrix, CONDITION_BASE_PRIORITY
from .dedup import ObservationDeduplicator, generate_observation_fingerprint
from .anomaly import C2AnomalyDetector, get_anomaly_detector
from .timing import C2TimingPredictor, get_timing_predictor
from .relevance import C2RelevanceModel, get_relevance_model
from .learning import (
    ObservationFeedbackLearner, ObservationFeedback,
    FeedbackType, DismissReason, LearningMetrics,
    get_learner, record_observation_feedback,
)
from .chains import C2ChainDetector, C2PredictedEvent, get_chain_detector
from .timing_engine import C2TimingEngine, get_timing_engine
from .ontology import (
    C2DomainOntology, C2ActionPlay,
    WildfireOntology, UrbanSARontology, MilitaryACEOntology, DisasterResponseOntology,
    get_ontology, list_domains, register_ontology,
)
from .evidence import C2EvidenceAggregator, C2EvidenceCluster, ObservationEvidence
from .embeddings import C2EmbeddingService, get_embedding_service

__all__ = [
    # Models
    "C2EventType", "ObservationPriority", "SensorSource", "C2ActionType", "C2Observation",
    # Resolver
    "C2EntityResolver", "EntityMatch", "normalize_entity_id",
    # Graph
    "C2EntityGraph", "NodeType", "EdgeType", "get_c2_graph", "build_graph_from_world_store",
    # Priority
    "C2PriorityMatrix", "CONDITION_BASE_PRIORITY",
    # Dedup
    "ObservationDeduplicator", "generate_observation_fingerprint",
    # Anomaly
    "C2AnomalyDetector", "get_anomaly_detector",
    # Timing predictor (ML)
    "C2TimingPredictor", "get_timing_predictor",
    # Relevance
    "C2RelevanceModel", "get_relevance_model",
    # Learning
    "ObservationFeedbackLearner", "ObservationFeedback",
    "FeedbackType", "DismissReason", "LearningMetrics",
    "get_learner", "record_observation_feedback",
    # Chains
    "C2ChainDetector", "C2PredictedEvent", "get_chain_detector",
    # Timing engine (doctrine-seeded)
    "C2TimingEngine", "get_timing_engine",
    # Ontology
    "C2DomainOntology", "C2ActionPlay",
    "WildfireOntology", "UrbanSARontology", "MilitaryACEOntology", "DisasterResponseOntology",
    "get_ontology", "list_domains", "register_ontology",
    # Evidence
    "C2EvidenceAggregator", "C2EvidenceCluster", "ObservationEvidence",
    # Embeddings
    "C2EmbeddingService", "get_embedding_service",
]
