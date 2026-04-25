from __future__ import annotations
import asyncio
import sys
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

from smart_code_reviewer.core.analyzer import (
    DeepSignalExtractor,
    AnalyzerOrchestrator,
    MultiDimensionalTensor,
)
from smart_code_reviewer.core.validator import UnifiedValidator, LogicalVarianceWarning
from smart_code_reviewer.db.memory_engine import AsyncMemoryEngine, VectorRecord
from smart_code_reviewer.utils.ast_toolkit import ASTToolkit
from smart_code_reviewer.ui.dashboard import DashboardRenderer
from smart_code_reviewer.config import ReviewConfig
from smart_code_reviewer.io.export_engine import ExportEngine
from smart_code_reviewer.core.git_diff import GitDiffAnalyzer
from smart_code_reviewer.ui.html_reporter import HTMLReporter
from smart_code_reviewer.security.pattern_matcher import SecurityPatternMatcher
from smart_code_reviewer.core.auto_fix import AutoFixEngine
import argparse
import os
import glob
from datetime import datetime





class AsyncLifecycleManager:
    def __init__(self, config: ReviewConfig):
        self.config = config
        self.extractor = DeepSignalExtractor(
            syntax_dim=config.syntax_dim,
            dataflow_dim=config.dataflow_dim,
            intent_dim=config.intent_dim,
            use_neural=config.use_neural,
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
        self.exporter = ExportEngine()
        self.security = SecurityPatternMatcher()
        self.autofix = AutoFixEngine()
        self._running = False
        self._review_count: int = 0
        self._anomaly_count: int = 0
        self._dashboard_task: Optional[asyncio.Task] = None
        self.is_hydrated = False

    async def bootstrap(self) -> None:
        await self.memory.initialize()
        historical = await self.memory.get_all_vectors()
        for rec in historical:
            self.validator.mahalanobis.ingest(rec.embedding)
            self.validator.zscore.ingest(rec.embedding)
        self.is_hydrated = len(historical) > 0
        self.dashboard.setup_logging()
        self.dashboard.add_log("INFO", f"Smart-Code-Reviewer bootstrap sequence initiated. Hydrated: {self.is_hydrated}")
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
        security_flags = self.security.analyze(source)
        validation["security_flags"] = security_flags
        validation["source_id"] = source_id

        self.dashboard.advance_task(1.0)
        await self.memory.store_vector(
            source_id,
            combined,
            {
                "mahalanobis": validation["mahalanobis_distance"],
                "entropy": validation["entropy"],
                "z_score": validation["z_score"],
                "confidence": validation["confidence"],
                "security_flags": security_flags,
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
            fix = self.autofix.suggest_fix("anomaly", source)
            if fix:
                validation["suggested_fix"] = fix

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
            f"Smart-Code-Reviewer shutdown. Total: {self._review_count}, Anomalies: {self._anomaly_count}",
        )

    async def run_batch(self, sources: List[tuple]) -> List[Dict[str, Any]]:
        results = []
        for source_id, source in sources:
            result = await self.review_source(source_id, source)
            results.append(result)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exporter.to_json(results, f"smart_code_review_{ts}.json")
        self.exporter.to_csv(results, f"smart_code_review_{ts}.csv")
        return results

    async def run_continuous(self, source_stream: asyncio.Queue) -> None:
        if self.config.live_dashboard:
            asyncio.create_task(self.dashboard.live_update())
        while self._running:
            source_id, source = await source_stream.get()
            await self.review_source(source_id, source)
            source_stream.task_done()


async def main():
    parser = argparse.ArgumentParser(description="Smart-Code-Reviewer Engine")
    parser.add_argument("path", nargs="?", help="Path to file or directory for analysis")
    parser.add_argument("--git-diff", action="store_true", help="Analyze git diff between HEAD~1 and HEAD")
    parser.add_argument("--html-report", action="store_true", help="Generate HTML report")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI REST server")
    args = parser.parse_args()

    if args.serve:
        from smart_code_reviewer.api.server import start_server_async
        await start_server_async()
        return

    config = ReviewConfig.load()
    lifecycle = AsyncLifecycleManager(config)
    await lifecycle.bootstrap()

    if not lifecycle.is_hydrated:
        await lifecycle.warm_up()

    sources = []
    if args.git_diff:
        diff_analyzer = GitDiffAnalyzer()
        diff_data = diff_analyzer.get_diff_content()
        for path, diff, new_content in diff_data:
            lifecycle.dashboard.add_log("INFO", f"Analyzing drift for {path}")
            base_result = await lifecycle.review_source(f"{path}_base", "def _(): pass")
            new_result = await lifecycle.review_source(path, new_content)
            if new_result["mahalanobis_distance"] > base_result["mahalanobis_distance"] * 1.5:
                new_result["drift_detected"] = True
                lifecycle.dashboard.add_log("WARNING", f"Drift detected in {path}")
            sources.append((path, new_content))
    elif args.path:
        if os.path.isfile(args.path):
            with open(args.path, "r") as f:
                sources.append((args.path, f.read()))
        elif os.path.isdir(args.path):
            for file_path in glob.glob(os.path.join(args.path, "**/*.py"), recursive=True):
                with open(file_path, "r") as f:
                    sources.append((file_path, f.read()))
    else:
        sources = [
            ("module_a", "def foo(x): return x + 1"),
            ("module_b", "class Bar:\n    def __init__(self):\n        self.x = 0"),
            ("module_c", "def complex(n):\n    for i in range(n):\n        if i % 2 == 0: yield i"),
        ]

    results = await lifecycle.run_batch(sources)
    
    if args.html_report:
        reporter = HTMLReporter()
        report_path = reporter.generate(results)
        lifecycle.dashboard.add_log("INFO", f"HTML report generated at {report_path}")

    await lifecycle.shutdown()
    return results


if __name__ == "__main__":
    asyncio.run(main())
