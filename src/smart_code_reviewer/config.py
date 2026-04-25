from __future__ import annotations
import yaml
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class ReviewConfig:
    syntax_dim: int = 512
    dataflow_dim: int = 512
    intent_dim: int = 256
    mahalanobis_threshold: float = 3.0
    entropy_threshold: float = 4.0
    z_threshold: float = 2.5
    db_path: str = "smart_code_reviewer.db"
    live_dashboard: bool = True
    warm_up_count: int = 5
    alert_threshold: float = 0.7
    use_neural: bool = False

    @classmethod
    def load(cls, path: str = "config.yaml") -> ReviewConfig:
        if not os.path.exists(path):
            return cls()
        with open(path, "r") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls(**data)
