from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.logging import RichHandler
import logging
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class LogEntry:
    timestamp: str
    level: str
    message: str
    metadata: Dict[str, Any]


class DashboardRenderer:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5),
        )
        self.layout["main"].split_row(Layout(name="metrics"), Layout(name="logs"))
        self._logs: List[LogEntry] = []
        self._metrics: Dict[str, float] = {}
        self._progress = Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        self._task_id: Optional[int] = None
        self._alert_threshold: float = 0.8
        self._shutdown_requested: bool = False

    def request_shutdown(self) -> None:
        self._shutdown_requested = True

    def setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(console=self.console, rich_tracebacks=True)],
        )

    def update_metric(self, key: str, value: float) -> None:
        self._metrics[key] = value
        if key == "confidence" and value < self._alert_threshold:
            self.add_log(
                "ERROR",
                f"Confidence dropped to {value:.4f}",
                {"threshold": self._alert_threshold},
            )

    def add_log(
        self, level: str, message: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            metadata=metadata or {},
        )
        self._logs.append(entry)
        if len(self._logs) > 1000:
            self._logs = self._logs[-500:]

    def _render_header(self) -> Panel:
        text = Text("Smart-Code-Reviewer Engine", style="bold cyan")
        return Panel(text, style="blue")

    def _render_metrics(self) -> Panel:
        table = Table(title="Signal Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Status", style="green")
        for k, v in self._metrics.items():
            status = "OK" if v < 2.0 else "WARN" if v < 3.0 else "CRIT"
            color = (
                "red" if status == "CRIT" else "yellow" if status == "WARN" else "green"
            )
            table.add_row(k, f"{v:.6f}", f"[{color}]{status}[/{color}]")
        return Panel(table, title="Metrics")

    def _render_logs(self) -> Panel:
        table = Table(title="Event Stream")
        table.add_column("Time", style="dim")
        table.add_column("Level", style="bold")
        table.add_column("Message")
        for entry in self._logs[-25:]:
            style = (
                "red"
                if entry.level == "ERROR"
                else "yellow" if entry.level == "WARNING" else "green"
            )
            table.add_row(
                entry.timestamp[11:19],
                f"[{style}]{entry.level}[/{style}]",
                entry.message,
            )
        return Panel(table, title="Logs")

    def _render_footer(self) -> Panel:
        return Panel(self._progress, title="Progress")

    def render(self) -> Layout:
        self.layout["header"].update(self._render_header())
        self.layout["metrics"].update(self._render_metrics())
        self.layout["logs"].update(self._render_logs())
        self.layout["footer"].update(self._render_footer())
        return self.layout

    async def live_update(self, interval: float = 0.5) -> None:
        with Live(self.layout, console=self.console, refresh_per_second=4) as live:
            while not self._shutdown_requested:
                live.update(self.render())
                await asyncio.sleep(interval)

    def start_task(self, description: str, total: float = 100.0) -> None:
        self._task_id = self._progress.add_task(description, total=total)

    def advance_task(self, amount: float = 1.0) -> None:
        if self._task_id is not None:
            self._progress.advance(self._task_id, amount)

    def compute_metric_statistics(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for k, v in self._metrics.items():
            stats[k] = {"current": v, "rolling_mean": v, "variance": 0.0}
        return stats
