"""
C2 Embedding Service

Generates vector embeddings for semantic search over C2 observations and entities.
Uses Google AI (text-embedding-004, 768 dimensions) with in-memory caching.

Set GEMINI_API_KEY environment variable to enable (already in render.yaml).
Falls back gracefully when unavailable — no hard dependency.

Usage:
    from c2_intel.embeddings import get_embedding_service

    svc = get_embedding_service()
    vec = await svc.embed_text("Node-Bravo comms degraded on MAVLINK link")
    # → 768-float list, or None if service unavailable
"""

import hashlib
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    _HAS_GOOGLE_AI = True
except ImportError:
    _HAS_GOOGLE_AI = False
    logger.info("[C2Embeddings] google-generativeai not installed. Run: pip install google-generativeai")

EMBEDDING_DIMENSION = 768

_cache: Dict[str, List[float]] = {}
_MAX_CACHE_SIZE = 5000


class C2EmbeddingService:
    """
    Embedding service for semantic search over C2 observations and entities.

    Uses Google AI text-embedding-004 (768 dimensions).
    Falls back gracefully if API key is not set or SDK not installed.
    """

    def __init__(self):
        self.api_key    = os.getenv("GEMINI_API_KEY")
        self.model_name = "models/text-embedding-004"
        self.initialized = False

        if _HAS_GOOGLE_AI and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.initialized = True
                logger.info("[C2Embeddings] Google AI initialized")
            except Exception as e:
                logger.error("[C2Embeddings] Failed to initialize Google AI: %s", e)
        elif not _HAS_GOOGLE_AI:
            logger.info("[C2Embeddings] Google AI SDK not available")
        else:
            logger.info("[C2Embeddings] GEMINI_API_KEY not set — embeddings disabled")

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()

    def _get_cached(self, text: str) -> Optional[List[float]]:
        return _cache.get(self._cache_key(text))

    def _set_cached(self, text: str, embedding: List[float]):
        global _cache
        if len(_cache) >= _MAX_CACHE_SIZE:
            keys = list(_cache.keys())[:_MAX_CACHE_SIZE // 10]
            for k in keys:
                del _cache[k]
        _cache[self._cache_key(text)] = embedding

    async def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text string.

        Args:
            text: Observation description, entity ID, or free-form C2 context

        Returns:
            768-dimensional embedding vector, or None if unavailable.
        """
        if not self.initialized or not text or not text.strip():
            return None

        cached = self._get_cached(text)
        if cached:
            return cached

        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text[:30000],
                task_type="retrieval_document",
            )
            embedding = result["embedding"]
            self._set_cached(text, embedding)
            return embedding
        except Exception as e:
            logger.error("[C2Embeddings] embed_text error: %s", e)
            return None

    async def embed_texts(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts (batch where possible)."""
        if not self.initialized:
            return [None] * len(texts)

        results = [None] * len(texts)
        uncached_texts   = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                continue
            cached = self._get_cached(text)
            if cached:
                results[i] = cached
            else:
                uncached_texts.append(text[:30000])
                uncached_indices.append(i)

        if uncached_texts:
            try:
                batch_result = genai.embed_content(
                    model=self.model_name,
                    content=uncached_texts,
                    task_type="retrieval_document",
                )
                for idx, embedding in zip(uncached_indices, batch_result["embedding"]):
                    results[idx] = embedding
                    self._set_cached(texts[idx], embedding)
            except Exception as e:
                logger.error("[C2Embeddings] embed_texts batch error: %s", e)

        return results

    async def embed_for_query(self, query: str) -> Optional[List[float]]:
        """Generate embedding optimized for semantic search queries."""
        if not self.initialized or not query or not query.strip():
            return None
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=query[:30000],
                task_type="retrieval_query",
            )
            return result["embedding"]
        except Exception as e:
            logger.error("[C2Embeddings] embed_for_query error: %s", e)
            return None

    async def embed_observation(self, observation_dict: Dict) -> Optional[List[float]]:
        """
        Embed a C2 observation for semantic indexing.

        Combines event_type, entity_id, sensor_source, and any detail field.
        """
        parts = [
            observation_dict.get("event_type", ""),
            f"entity: {observation_dict.get('entity_id', '')}",
            f"source: {observation_dict.get('sensor_source', '')}",
        ]
        detail = observation_dict.get("detail", "")
        if detail:
            parts.append(detail)

        text = " | ".join(p for p in parts if p)
        return await self.embed_text(text)

    async def embed_entity(self, entity_dict: Dict) -> Optional[List[float]]:
        """
        Embed a WorldStore entity for semantic search.

        Combines entity_id, type, callsign, and capabilities.
        """
        parts = [
            entity_dict.get("entity_id", entity_dict.get("id", "")),
            entity_dict.get("entity_type", ""),
            entity_dict.get("callsign", ""),
        ]
        caps = entity_dict.get("capabilities", [])
        if isinstance(caps, list) and caps:
            parts.append("capabilities: " + ", ".join(str(c) for c in caps[:5]))

        text = " | ".join(p for p in parts if p)
        return await self.embed_text(text)

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        try:
            import numpy as np
            va = np.array(a)
            vb = np.array(b)
            dot  = float(np.dot(va, vb))
            norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
            return dot / norm if norm > 0 else 0.0
        except ImportError:
            # Pure Python fallback
            dot  = sum(x * y for x, y in zip(a, b))
            na   = sum(x * x for x in a) ** 0.5
            nb   = sum(x * x for x in b) ** 0.5
            return dot / (na * nb) if na * nb > 0 else 0.0

    def get_status(self) -> Dict:
        return {
            "initialized": self.initialized,
            "model":       self.model_name,
            "dimension":   EMBEDDING_DIMENSION,
            "cache_size":  len(_cache),
            "api_key_set": bool(self.api_key),
        }


_service: Optional[C2EmbeddingService] = None


def get_embedding_service() -> C2EmbeddingService:
    global _service
    if _service is None:
        _service = C2EmbeddingService()
    return _service
