from __future__ import annotations
import ast
import re
from typing import List, Dict, Any

class SecurityPatternMatcher:
    def __init__(self):
        self.secret_patterns = [
            re.compile(r"api[_-]key\s*=\s*['\"][a-zA-Z0-9]{32,}['\"]", re.I),
            re.compile(r"password\s*=\s*['\"][^'\"]{8,}['\"]", re.I)
        ]

    def detect_sql_injection(self, node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if getattr(child.func, "attr", "") in ["execute", "executemany"]:
                    for arg in child.args:
                        if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod):
                            return True
        return False

    def detect_hardcoded_secrets(self, source: str) -> List[str]:
        found = []
        for p in self.secret_patterns:
            matches = p.findall(source)
            if matches:
                found.extend(matches)
        return found

    def detect_eval_exec(self, source: str) -> bool:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ["eval", "exec"]:
                    return True
        return False

    def analyze(self, source: str) -> Dict[str, bool]:
        try:
            tree = ast.parse(source)
            return {
                "sql_injection": self.detect_sql_injection(tree),
                "hardcoded_secrets": len(self.detect_hardcoded_secrets(source)) > 0,
                "eval_exec": self.detect_eval_exec(source)
            }
        except:
            return {"sql_injection": False, "hardcoded_secrets": False, "eval_exec": False}
