from __future__ import annotations
import json
import csv
import os
from typing import List, Dict, Any
from datetime import datetime

class ExportEngine:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def to_json(self, results: List[Dict[str, Any]], filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        return path

    def to_csv(self, results: List[Dict[str, Any]], filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        if not results:
            return path
        flattened = [self._flatten_dict(r) for r in results]
        keys = flattened[0].keys()
        with open(path, "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(flattened)
        return path
