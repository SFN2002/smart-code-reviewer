from __future__ import annotations
import asyncio
import sys
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from smart_reviewer.core.analyzer import (
    DeepSignalExtractor,
    AnalyzerOrchestrator,
    MultiDimensionalTensor,
)
from smart_reviewer.core.validator import UnifiedValidator, LogicalVarianceWarning
from smart_reviewer.db.memory_engine import AsyncMemoryEngine, VectorRecord
from smart_reviewer.utils.ast_toolkit import ASTToolkit
from smart_reviewer.ui.dashboard import DashboardRenderer


@dataclass
class ReviewConfig:
    syntax_dim: int = 512
    dataflow_dim: int = 512
    intent_dim: int = 256
    mahalanobis_threshold: float = 3.0
    entropy_threshold: float = 4.0
    z_threshold: float = 2.5
    db_path: str = "omni_reviewer.db"
    live_dashboard: bool = True


class AsyncLifecycleManager:
    def __init__(self, config: ReviewConfig):
        self.config = config
        self.extractor = DeepSignalExtractor(
            syntax_dim=config.syntax_dim,
            dataflow_dim=config.dataflow_dim,
            intent_dim=config.intent_dim,
        )
        self.analyzer = AnalyzerOrchestrator(self.extractor)
        self.memory = AsyncMemoryEngine(config.db_path)
        self.validator = UnifiedValidator(
            mahalanobis_threshold=config.mahalanobis_threshold,
            entropy_threshold=config.entropy_threshold,
            z_threshold=config.z_threshold,
            memory_engine=self.memory,
        )
        self.ast_toolkit = ASTToolkit()
        self.dashboard = DashboardRenderer()
        self._running = False
        self._review_count: int = 0
        self._anomaly_count: int = 0
        self._dashboard_task: Optional[asyncio.Task] = None

    async def bootstrap(self) -> None:
        await self.memory.initialize()
        self.dashboard.setup_logging()
        self.dashboard.add_log("INFO", "Omni-Reviewer bootstrap sequence initiated")
        if self.config.live_dashboard:
            self._dashboard_task = asyncio.create_task(self.dashboard.live_update())
        self._running = True

    async def review_source(self, source_id: str, source: str) -> Dict[str, Any]:
        self.dashboard.start_task(f"Reviewing {source_id}", total=6.0)

        self.dashboard.advance_task(1.0)
        tensor = self.analyzer.analyze(source)

        self.dashboard.advance_task(1.0)
        sym_vec = self.ast_toolkit.extract_symbol_vector(source)

        self.dashboard.advance_task(1.0)
        flat_tensor = tensor.flatten()
        combined = np.concatenate([flat_tensor, sym_vec])
        logits = np.random.randn(10) * 0.5

        self.dashboard.advance_task(1.0)
        validation = await self.validator.validate_tensor(combined, logits)

        self.dashboard.advance_task(1.0)
        await self.memory.store_vector(
            source_id,
            combined,
            {
                "mahalanobis": validation["mahalanobis_distance"],
                "entropy": validation["entropy"],
                "z_score": validation["z_score"],
                "confidence": validation["confidence"],
            },
        )

        self.dashboard.advance_task(1.0)
        self.dashboard.update_metric("mahalanobis", validation["mahalanobis_distance"])
        self.dashboard.update_metric("entropy", validation["entropy"])
        self.dashboard.update_metric("z_score", validation["z_score"])
        self.dashboard.update_metric("confidence", validation["confidence"])
        self.dashboard.update_metric(
            "divergence", self.analyzer.compute_divergence(tensor)
        )
        self.dashboard.update_metric("frobenius", tensor.frobenius_norm())

        self._review_count += 1
        if validation["is_anomalous"]:
            self._anomaly_count += 1

        self.dashboard.add_log(
            "WARNING" if validation["is_anomalous"] else "INFO",
            f"Review {source_id} completed. Anomalous: {validation['is_anomalous']}. Confidence: {validation['confidence']:.4f}",
            validation,
        )

        return validation

    async def warm_up(self) -> None:
        self.dashboard.add_log(
            "INFO", "Warm-up phase: initializing with 5 dummy analyses"
        )
        warmup_samples = [
            ("warmup_0", "def _(): pass"),
            ("warmup_1", "x = 10\ny = 20\nprint(x + y)"),
            ("warmup_2", "def foo(a, b):\n    return a * b"),
            ("warmup_3", "class Test:\n    def method(self): return True"),
            ("warmup_4", "import math\nmath.sqrt(16)"),
        ]
        for source_id, source in warmup_samples:
            tensor = self.analyzer.analyze(source)
            sym_vec = self.ast_toolkit.extract_symbol_vector(source)
            flat_tensor = tensor.flatten()
            combined = np.concatenate([flat_tensor, sym_vec])
            logits = np.random.randn(10) * 0.5
            validation = await self.validator.validate_tensor(combined, logits)
            await self.memory.store_vector(
                source_id,
                combined,
                {
                    "mahalanobis": validation["mahalanobis_distance"],
                    "entropy": validation["entropy"],
                    "z_score": validation["z_score"],
                    "confidence": validation["confidence"],
                },
            )
        self.dashboard.add_log("INFO", "Warm-up phase complete")

    async def shutdown(self) -> None:
        self.dashboard.request_shutdown()
        if self._dashboard_task:
            self._dashboard_task.cancel()
            try:
                await self._dashboard_task
            except asyncio.CancelledError:
                pass
        await self.memory.close()
        self._running = False
        self.dashboard.add_log(
            "INFO",
            f"Omni-Reviewer shutdown. Total: {self._review_count}, Anomalies: {self._anomaly_count}",
        )

    async def run_batch(self, sources: List[tuple]) -> List[Dict[str, Any]]:
        results = []
        for source_id, source in sources:
            result = await self.review_source(source_id, source)
            results.append(result)
        return results

    async def run_continuous(self, source_stream: asyncio.Queue) -> None:
        if self.config.live_dashboard:
            asyncio.create_task(self.dashboard.live_update())
        while self._running:
            source_id, source = await source_stream.get()
            await self.review_source(source_id, source)
            source_stream.task_done()


async def main():
    config = ReviewConfig()
    lifecycle = AsyncLifecycleManager(config)
    await lifecycle.bootstrap()

    await lifecycle.warm_up()

    sample_sources = [
        ("module_a", "def foo(x): return x + 1"),
        ("module_b", "class Bar:\n    def __init__(self):\n        self.x = 0"),
        (
            "module_c",
            "def complex(n):\n    for i in range(n):\n        if i % 2 == 0: yield i",
        ),
    ]

    results = await lifecycle.run_batch(sample_sources)
    await lifecycle.shutdown()

    print("\n" + "=" * 60)
    print("OMNI-REVIEWER RESULTS")
    print("=" * 60)
    for idx, res in enumerate(results, 1):
        print(f"\n[Review {idx}]")
        for k, v in res.items():
            if isinstance(v, float):
                print(f"  {k:20s}: {v:.6f}")
            else:
                print(f"  {k:20s}: {v}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    asyncio.run(main())
