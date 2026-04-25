from __future__ import annotations
import git
import os
from typing import List, Dict, Any, Tuple

class GitDiffAnalyzer:
    def __init__(self, repo_path: str = "."):
        self.repo = git.Repo(repo_path)

    def get_diff_files(self, base_commit: str = "HEAD~1", target_commit: str = "HEAD") -> List[str]:
        diff = self.repo.commit(base_commit).diff(target_commit)
        return [item.a_path for item in diff if item.a_path.endswith(".py")]

    def get_diff_content(self, base_commit: str = "HEAD~1", target_commit: str = "HEAD") -> List[Tuple[str, str, str]]:
        diffs = self.repo.commit(base_commit).diff(target_commit, create_patch=True)
        results = []
        for d in diffs:
            if d.a_path.endswith(".py"):
                results.append((d.a_path, d.diff.decode("utf-8"), self.repo.git.show(f"{target_commit}:{d.a_path}")))
        return results
