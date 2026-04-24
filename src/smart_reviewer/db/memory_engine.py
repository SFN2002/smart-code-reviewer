from __future__ import annotations
import aiosqlite
import numpy as np
import asyncio
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import json
import pickle
import base64
from datetime import datetime, timezone
import os


@dataclass
class VectorRecord:
    vector_id: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    reliability_score: float
    timestamp: str


class BayesianReliabilityScorer:
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self._alpha = prior_alpha
        self._beta = prior_beta

    def update(self, success: bool) -> None:
        if success:
            self._alpha += 1.0
        else:
            self._beta += 1.0

    def score(self) -> float:
        return self._alpha / (self._alpha + self._beta)

    def variance(self) -> float:
        return (self._alpha * self._beta) / (
            (self._alpha + self._beta) ** 2 * (self._alpha + self._beta + 1.0)
        )

    def credible_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        from scipy import stats

        lower = stats.beta.ppf((1.0 - confidence) / 2.0, self._alpha, self._beta)
        upper = stats.beta.ppf((1.0 + confidence) / 2.0, self._alpha, self._beta)
        return float(lower), float(upper)


class AsyncMemoryEngine:
    def __init__(self, db_path: str = "smart_reviewer.db"):
        self.db_path = db_path
        self._pool: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()
        self._scorer = BayesianReliabilityScorer()

    async def initialize(self) -> None:
        self._pool = await aiosqlite.connect(
            self.db_path, timeout=30.0, check_same_thread=False
        )
        await self._pool.execute("PRAGMA journal_mode = WAL")
        await self._pool.execute("PRAGMA busy_timeout = 5000")
        await self._pool.execute(
            "CREATE TABLE IF NOT EXISTS vectors ("
            "vector_id TEXT PRIMARY KEY,"
            "embedding BLOB,"
            "metadata TEXT,"
            "reliability REAL,"
            "timestamp TEXT"
            ")"
        )
        await self._pool.execute(
            "CREATE TABLE IF NOT EXISTS reliability_log ("
            "log_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "vector_id TEXT,"
            "old_score REAL,"
            "new_score REAL,"
            "timestamp TEXT"
            ")"
        )
        await self._pool.commit()

    async def store_vector(
        self, vector_id: str, embedding: np.ndarray, metadata: Dict[str, Any]
    ) -> None:
        async with self._lock:
            if self._pool is None:
                raise RuntimeError("Engine not initialized")
            serialized = base64.b64encode(pickle.dumps(embedding)).decode()
            reliability = self._scorer.score()
            timestamp = datetime.now(timezone.utc).isoformat()
            await self._pool.execute(
                "INSERT OR REPLACE INTO vectors (vector_id, embedding, metadata, reliability, timestamp) VALUES (?, ?, ?, ?, ?)",
                (vector_id, serialized, json.dumps(metadata), reliability, timestamp),
            )
            await self._pool.commit()

    async def retrieve_vector(self, vector_id: str) -> Optional[VectorRecord]:
        async with self._lock:
            if self._pool is None:
                return None
            cursor = await self._pool.execute(
                "SELECT * FROM vectors WHERE vector_id = ?", (vector_id,)
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            embedding = pickle.loads(base64.b64decode(row[1]))
            return VectorRecord(
                vector_id=row[0],
                embedding=embedding,
                metadata=json.loads(row[2]),
                reliability_score=row[3],
                timestamp=row[4],
            )

    async def similarity_search(
        self, query: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        async with self._lock:
            if self._pool is None:
                return []
            cursor = await self._pool.execute(
                "SELECT vector_id, embedding FROM vectors"
            )
            rows = await cursor.fetchall()
            results = []
            query_flat = query.flatten()
            query_norm = np.linalg.norm(query_flat)
            for row in rows:
                vec = pickle.loads(base64.b64decode(row[1]))
                vec_flat = vec.flatten()
                dot = np.dot(query_flat, vec_flat)
                norm = query_norm * np.linalg.norm(vec_flat)
                sim = float(dot / norm) if norm > 0 else 0.0
                results.append((row[0], sim))
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    async def update_reliability(self, vector_id: str, success: bool) -> None:
        async with self._lock:
            if self._pool is None:
                return
            record = await self.retrieve_vector(vector_id)
            if record is None:
                return
            old_score = record.reliability_score
            self._scorer.update(success)
            new_score = self._scorer.score()
            timestamp = datetime.utcnow().isoformat()
            await self._pool.execute(
                "UPDATE vectors SET reliability = ? WHERE vector_id = ?",
                (new_score, vector_id),
            )
            await self._pool.execute(
                "INSERT INTO reliability_log (vector_id, old_score, new_score, timestamp) VALUES (?, ?, ?, ?)",
                (vector_id, old_score, new_score, timestamp),
            )
            await self._pool.commit()

    async def get_reliability_distribution(self) -> Dict[str, float]:
        async with self._lock:
            if self._pool is None:
                return {}
            cursor = await self._pool.execute("SELECT reliability FROM vectors")
            rows = await cursor.fetchall()
            scores = [row[0] for row in rows if row[0] is not None]
            if not scores:
                return {}
            arr = np.array(scores)
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

    async def get_record_count(self) -> int:
        """Get the total number of records in the vectors table."""
        async with self._lock:
            if self._pool is None:
                return 0
            cursor = await self._pool.execute("SELECT COUNT(*) FROM vectors")
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def close(self) -> None:
        try:
            if self._pool:
                await self._pool.close()
                self._pool = None
        except asyncio.CancelledError:
            pass
