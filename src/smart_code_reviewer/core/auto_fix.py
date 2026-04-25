from __future__ import annotations
from typing import Optional

class AutoFixEngine:
    def suggest_fix(self, anomaly_type: str, source: str) -> Optional[str]:
        if anomaly_type == "high_entropy":
            return source
        return None
