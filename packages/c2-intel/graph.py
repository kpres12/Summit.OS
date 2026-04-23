"""
C2 Entity Relationship Graph

Ported from Mira Signals' entity_graph.py.
Domain remapped: company/person/signal nodes → node/asset/mission/observation nodes.

Enables queries a flat entity list cannot answer:
  - "Find all nodes with similar degradation patterns to Node-Alpha"
  - "What assets are affected if Node-Bravo loses comms?"
  - "What's the relationship path between this asset and the command node?"
  - "Which entities co-degrade when comms drop in sector 4?"

Built on NetworkX (in-memory). Designed to swap to Neo4j at scale.
Wire to WorldStore via build_graph_from_world_store().
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    logger.warning("networkx not installed — C2EntityGraph disabled. pip install networkx")


# ---------------------------------------------------------------------------
# Node and edge types  (C2 domain)
# ---------------------------------------------------------------------------

class NodeType:
    NODE        = "node"         # A C2 node (physical location / compute)
    ASSET       = "asset"        # A drone, UGV, sensor platform
    OPERATOR    = "operator"     # A human operator
    SENSOR      = "sensor"       # A sensor type/system
    LOCATION    = "location"     # Geographic area / sector
    EVENT       = "event"        # A significant occurrence (mission, degradation event)
    OBSERVATION = "observation"  # A specific sensor observation


class EdgeType:
    CO_OBSERVED   = "co_observed"    # appeared in same observation window
    TEMPORAL      = "temporal"       # observations within configurable time window
    ADVERSARIAL   = "adversarial"    # known adversarial relationship
    COMMANDS      = "commands"       # command/control relationship (node commands asset)
    EQUIPPED_WITH = "equipped_with"  # asset/node uses this sensor
    LOCATED_AT    = "located_at"     # entity located in sector/area
    ASSIGNED_TO   = "assigned_to"    # asset assigned to node/mission
    SUBORDINATE   = "subordinate"    # subordinate node relationship
    PEER          = "peer"           # mesh peer relationship


# ---------------------------------------------------------------------------
# C2 Entity Graph
# ---------------------------------------------------------------------------

class C2EntityGraph:
    """
    In-memory C2 entity relationship graph.

    Build from WorldStore entity stream, query for situation awareness
    and downstream effect analysis (TA1 simulation layer).
    """

    def __init__(self):
        if not NX_AVAILABLE:
            self._G = None
            return
        self._G: nx.DiGraph = nx.DiGraph()
        self._built_at: Optional[datetime] = None
        self._observation_count: int = 0

    # ------------------------------------------------------------------
    # Node builders
    # ------------------------------------------------------------------

    def add_node(self, node_id: str, **attrs) -> str:
        if not NX_AVAILABLE or not node_id:
            return node_id
        nid = f"node:{node_id.lower().strip()}"
        if not self._G.has_node(nid):
            self._G.add_node(nid, type=NodeType.NODE, label=node_id, **attrs)
        else:
            for k, v in attrs.items():
                if v is not None:
                    self._G.nodes[nid][k] = v
        return nid

    def add_asset(self, asset_id: str, domain: str = "aerial", **attrs) -> str:
        if not NX_AVAILABLE or not asset_id:
            return asset_id
        nid = f"asset:{asset_id.lower().strip()}"
        if not self._G.has_node(nid):
            self._G.add_node(nid, type=NodeType.ASSET, label=asset_id, domain=domain, **attrs)
        else:
            for k, v in attrs.items():
                if v is not None:
                    self._G.nodes[nid][k] = v
        return nid

    def add_sensor(self, sensor_type: str) -> str:
        if not NX_AVAILABLE or not sensor_type:
            return sensor_type
        nid = f"sensor:{sensor_type.lower().strip()}"
        if not self._G.has_node(nid):
            self._G.add_node(nid, type=NodeType.SENSOR, label=sensor_type)
        return nid

    def add_location(self, name: str, **attrs) -> str:
        if not NX_AVAILABLE or not name:
            return name
        nid = f"location:{name.lower().strip()}"
        if not self._G.has_node(nid):
            self._G.add_node(nid, type=NodeType.LOCATION, label=name, **attrs)
        return nid

    def ingest_observation(
        self,
        obs_id: str,
        node_id: str,
        event_type: str,
        source: str,
        score: int,
        detected_at: Optional[datetime] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Ingest a C2 observation into the graph.
        Creates/updates nodes and edges from the observation's metadata.
        """
        if not NX_AVAILABLE:
            return

        entity_nid = self.add_asset(node_id) if "asset" in node_id.lower() else self.add_node(node_id)

        meta = metadata or {}

        # Sensor sources
        for sensor in meta.get("source_sensors", []):
            sid = self.add_sensor(sensor)
            self._add_edge(entity_nid, sid, EdgeType.EQUIPPED_WITH, weight=1.0)

        # Location
        if meta.get("location") or meta.get("sector"):
            loc = meta.get("location") or meta.get("sector")
            lid = self.add_location(str(loc))
            self._add_edge(entity_nid, lid, EdgeType.LOCATED_AT, weight=1.0)

        # Commanding node
        if meta.get("commanded_by"):
            cmd_nid = self.add_node(meta["commanded_by"])
            self._add_edge(cmd_nid, entity_nid, EdgeType.COMMANDS, weight=1.0)

        # Peer nodes
        for peer in meta.get("peers", []):
            peer_nid = self.add_node(peer)
            self._add_edge(entity_nid, peer_nid, EdgeType.PEER, weight=1.0)

        self._observation_count += 1

    def add_temporal_edges(self, observations: List[Dict], window_seconds: float = 30.0):
        """
        Connect entities that have observations within window_seconds of each other.
        Reveals co-degradation patterns invisible from flat entity lists.
        """
        if not NX_AVAILABLE or not observations:
            return

        dated = []
        for obs in observations:
            try:
                ts = obs.get("last_seen") or obs.get("detected_at")
                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                elif isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    continue
                dated.append((dt, obs))
            except Exception:
                continue

        dated.sort(key=lambda x: x[0])
        window = timedelta(seconds=window_seconds)

        for i, (dt_i, obs_i) in enumerate(dated):
            nid_i = obs_i.get("entity_id", "")
            if not nid_i:
                continue
            node_i = f"asset:{nid_i.lower()}"

            for j in range(i + 1, len(dated)):
                dt_j, obs_j = dated[j]
                if dt_j - dt_i > window:
                    break
                nid_j = obs_j.get("entity_id", "")
                if not nid_j or nid_j == nid_i:
                    continue
                node_j = f"asset:{nid_j.lower()}"
                if self._G.has_node(node_i) and self._G.has_node(node_j):
                    self._add_edge(node_i, node_j, EdgeType.TEMPORAL, weight=1.0,
                                   seconds_apart=float((dt_j - dt_i).total_seconds()))

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def find_similar_nodes(self, node_id: str, top_n: int = 10) -> List[Dict]:
        """
        Find nodes/assets with similar observation patterns and network neighbors.
        Uses 2-hop Jaccard neighborhood overlap.

        C2 use: "Find nodes with similar degradation signatures to Node-Alpha."
        """
        if not NX_AVAILABLE or self._G is None:
            return []

        candidates = [f"node:{node_id.lower()}", f"asset:{node_id.lower()}"]
        target_id = None
        for c in candidates:
            if self._G.has_node(c):
                target_id = c
                break
        if not target_id:
            return []

        target_nbrs = set(self._G.successors(target_id)) | set(self._G.predecessors(target_id))
        target_2hop: set = set()
        for n in target_nbrs:
            target_2hop.update(self._G.successors(n))
            target_2hop.update(self._G.predecessors(n))
        target_2hop.discard(target_id)

        scores = []
        for other_id, data in self._G.nodes(data=True):
            if data.get("type") not in (NodeType.NODE, NodeType.ASSET) or other_id == target_id:
                continue
            other_nbrs = set(self._G.successors(other_id)) | set(self._G.predecessors(other_id))
            other_2hop: set = set()
            for n in other_nbrs:
                other_2hop.update(self._G.successors(n))
                other_2hop.update(self._G.predecessors(n))
            other_2hop.discard(other_id)
            union = target_2hop | other_2hop
            if not union:
                continue
            jaccard = len(target_2hop & other_2hop) / len(union)
            if jaccard > 0:
                scores.append({
                    "entity": data.get("label", other_id),
                    "type": data.get("type"),
                    "similarity_score": round(jaccard, 3),
                    "shared_connections": len(target_2hop & other_2hop),
                })

        scores.sort(key=lambda x: x["similarity_score"], reverse=True)
        return scores[:top_n]

    def get_effect_radius(self, node_id: str, hops: int = 2) -> Dict[str, Any]:
        """
        For a node/asset with a high-priority observation, find all connected entities
        within N hops.

        C2 use: "If Node-Bravo loses comms, what assets and missions are affected?"
        TA1 use: Simulate before committing — show operator the downstream impact.
        """
        if not NX_AVAILABLE or self._G is None:
            return {"entity": node_id, "affected": []}

        candidates = [f"node:{node_id.lower()}", f"asset:{node_id.lower()}"]
        target_id = None
        for c in candidates:
            if self._G.has_node(c):
                target_id = c
                break
        if not target_id:
            return {"entity": node_id, "affected": []}

        affected = []
        # Use undirected view for reachability
        undirected = self._G.to_undirected()
        lengths = nx.single_source_shortest_path_length(undirected, target_id, cutoff=hops)
        for nid, depth in lengths.items():
            if depth == 0:
                continue
            ndata = self._G.nodes[nid]
            edge_data = (
                self._G.get_edge_data(target_id, nid) or
                self._G.get_edge_data(nid, target_id) or {}
            )
            affected.append({
                "entity": ndata.get("label", nid),
                "type": ndata.get("type", "unknown"),
                "hops": depth,
                "relationship": edge_data.get("type", "connected"),
            })

        return {
            "entity": node_id,
            "total_affected": len(affected),
            "affected": affected,
        }

    def find_relationship_path(self, entity_a: str, entity_b: str) -> Optional[List[Dict]]:
        """
        Find shortest relationship path between two entities.
        C2 use: "How is Asset-7 connected to Command-Node-Alpha?"
        """
        if not NX_AVAILABLE or self._G is None:
            return None

        candidates_a = [f"node:{entity_a.lower()}", f"asset:{entity_a.lower()}"]
        candidates_b = [f"node:{entity_b.lower()}", f"asset:{entity_b.lower()}"]
        node_a = next((c for c in candidates_a if self._G.has_node(c)), None)
        node_b = next((c for c in candidates_b if self._G.has_node(c)), None)
        if not node_a or not node_b:
            return None

        try:
            undirected = self._G.to_undirected()
            path = nx.shortest_path(undirected, node_a, node_b)
            result = []
            for i, nid in enumerate(path):
                ndata = self._G.nodes[nid]
                step = {"entity": ndata.get("label", nid), "type": ndata.get("type", "unknown")}
                if i > 0:
                    edge = (self._G.get_edge_data(path[i - 1], nid) or
                            self._G.get_edge_data(nid, path[i - 1]) or {})
                    step["via"] = edge.get("type", "connected")
                result.append(step)
            return result
        except Exception:
            return None

    def get_high_centrality_nodes(self, top_n: int = 10) -> List[Dict]:
        """Nodes/assets with the most connections — the hubs of the C2 network."""
        if not NX_AVAILABLE or self._G is None:
            return []
        results = []
        for nid, data in self._G.nodes(data=True):
            if data.get("type") in (NodeType.NODE, NodeType.ASSET):
                results.append({
                    "entity": data.get("label", nid),
                    "type": data.get("type"),
                    "connections": self._G.degree(nid),
                })
        results.sort(key=lambda x: x["connections"], reverse=True)
        return results[:top_n]

    def get_co_occurring_event_types(self, node_id: str) -> List[Dict]:
        """What event types co-occur with this node's neighbors? Used to predict what comes next."""
        if not NX_AVAILABLE or self._G is None:
            return []
        candidates = [f"node:{node_id.lower()}", f"asset:{node_id.lower()}"]
        target_id = next((c for c in candidates if self._G.has_node(c)), None)
        if not target_id:
            return []
        counts: Dict[str, int] = defaultdict(int)
        for n in list(self._G.successors(target_id)) + list(self._G.predecessors(target_id)):
            ndata = self._G.nodes[n]
            if ndata.get("type") == NodeType.OBSERVATION:
                counts[ndata.get("event_type", "unknown")] += 1
        return sorted(
            [{"event_type": k, "count": v} for k, v in counts.items()],
            key=lambda x: x["count"], reverse=True,
        )

    def stats(self) -> Dict:
        if not NX_AVAILABLE or self._G is None:
            return {"available": False, "reason": "networkx not installed"}
        type_counts: Dict[str, int] = defaultdict(int)
        for _, data in self._G.nodes(data=True):
            type_counts[data.get("type", "unknown")] += 1
        edge_type_counts: Dict[str, int] = defaultdict(int)
        for _, _, data in self._G.edges(data=True):
            edge_type_counts[data.get("type", "unknown")] += 1
        return {
            "available": True,
            "nodes": self._G.number_of_nodes(),
            "edges": self._G.number_of_edges(),
            "node_types": dict(type_counts),
            "edge_types": dict(edge_type_counts),
            "observations_ingested": self._observation_count,
            "built_at": self._built_at.isoformat() if self._built_at else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _add_edge(self, node_a: str, node_b: str, edge_type: str, weight: float = 1.0, **attrs):
        if self._G.has_edge(node_a, node_b):
            self._G[node_a][node_b]["weight"] = self._G[node_a][node_b].get("weight", 1.0) + weight
        else:
            self._G.add_edge(node_a, node_b, type=edge_type, weight=weight, **attrs)


# ---------------------------------------------------------------------------
# Singleton + WorldStore builder
# ---------------------------------------------------------------------------

_graph: Optional[C2EntityGraph] = None


def get_c2_graph() -> C2EntityGraph:
    global _graph
    if _graph is None:
        _graph = C2EntityGraph()
    return _graph


def build_graph_from_world_store(entities: List[Dict]) -> C2EntityGraph:
    """
    Build (or rebuild) the C2 entity graph from a WorldStore entity snapshot.

    entities: list of entity dicts from WorldStore.get_all_entities() or
              equivalent — each dict has entity_id, entity_type, domain,
              position, source_sensors, properties, last_seen, etc.
    """
    graph = get_c2_graph()
    if not NX_AVAILABLE:
        return graph

    raw_for_temporal = []
    for entity in entities:
        eid = entity.get("entity_id", "")
        domain = entity.get("domain", "aerial")
        props = entity.get("properties") or {}
        sensors = entity.get("source_sensors") or []

        asset_nid = graph.add_asset(eid, domain=domain,
                                    entity_type=entity.get("entity_type"),
                                    callsign=entity.get("callsign"))

        for sensor in sensors:
            sid = graph.add_sensor(sensor)
            graph._add_edge(asset_nid, sid, EdgeType.EQUIPPED_WITH)

        if props.get("commanded_by"):
            cmd = graph.add_node(props["commanded_by"])
            graph._add_edge(cmd, asset_nid, EdgeType.COMMANDS)

        raw_for_temporal.append(entity)

    graph.add_temporal_edges(raw_for_temporal, window_seconds=30.0)
    graph._built_at = datetime.now(timezone.utc)

    logger.info(
        "C2 entity graph built: %d nodes, %d edges",
        graph._G.number_of_nodes() if graph._G else 0,
        graph._G.number_of_edges() if graph._G else 0,
    )
    return graph


__all__ = [
    "C2EntityGraph",
    "NodeType",
    "EdgeType",
    "get_c2_graph",
    "build_graph_from_world_store",
]
