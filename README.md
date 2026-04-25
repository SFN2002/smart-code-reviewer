# Smart-Code-Reviewer

A high-fidelity, modular AI code-analysis engine that mirrors the internal reasoning of LLMs to detect structural and semantic code anomalies through multi-dimensional signal extraction, statistical validation, and asynchronous persistence.

## Architecture Overview

Smart-Code-Reviewer is not a script — it is a code-intelligence engine built around a strictly modular tree architecture. Each component is deep, mathematically rigorous, and handles multiple edge cases.

```text
smart-code-reviewer/
├── pyproject.toml
├── src/
│   └── smart_code_reviewer/
│       ├── __init__.py
│       ├── main.py                 # Async lifecycle orchestrator
│       ├── core/
│       │   ├── analyzer.py         # Multi-layered signal extraction
│       │   └── validator.py        # Statistical OOD detection
│       ├── db/
│       │   └── memory_engine.py    # SQLite vector persistence + Bayesian scoring
│       ├── utils/
│       │   └── ast_toolkit.py      # Deep AST visitor + CFG builder
│       └── ui/
│           └── dashboard.py        # Rich-based real-time terminal UI
```

## Core Capabilities

1. **Multi-Dimensional Tensor Decoupling**
The DeepSignalExtractor breaks down any Python source into a 3-plane tensor:
- **Syntax Plane**: AST node-type embeddings via sinusoidal hashing
- **Data-Flow Plane**: Assignment depth, target cardinality, log-depth features
- **Intent Plane**: Cyclomatic complexity latent representations
Each plane undergoes SVD decoupling and covariance trace analysis.

2. **Out-of-Distribution (OOD) Detection — Mahalanobis Distance**
The MahalanobisOODDetector maintains a running mean and precision matrix over trusted code vectors. New vectors are scored by their distance from the established manifold.

3. **Bayesian Reliability Scoring**
Every analysis result is cross-referenced with historical data in SQLite. Reliability scores are updated using a Beta distribution prior, allowing the engine to "learn" what constitutes a stable code signal over time.

## Installation

```bash
pip install .
```

## Usage

```bash
# Standard analysis
python -m smart_code_reviewer.main path/to/code

# Git diff analysis
python -m smart_code_reviewer.main --git-diff

# Start REST API
python -m smart_code_reviewer.main --serve

# Generate HTML report
python -m smart_code_reviewer.main path/to/code --html-report
```
