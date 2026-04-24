from __future__ import annotations
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Protocol, runtime_checkable, Dict, List, Tuple, Any, Union, Callable, Optional
import ast
import inspect
from dataclasses import dataclass, field

@runtime_checkable
class SignalProtocol(Protocol):
    def emit(self, tensor: np.ndarray) -> np.ndarray: ...

class RegistryMeta(ABCMeta):
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if not hasattr(cls, '_registry'):
            cls._registry = {}
        if 'register' in namespace:
            cls._registry[name] = cls
        return cls

class TensorDimensions:
    SYNTAX = 0
    DATAFLOW = 1
    INTENT = 2

@dataclass
class MultiDimensionalTensor:
    syntax_plane: np.ndarray
    dataflow_plane: np.ndarray
    intent_plane: np.ndarray
    covariance_trace: float = field(init=False)
    eigenvalue_spectrum: np.ndarray = field(init=False)
    
    def __post_init__(self):
        stacked = np.stack([
            self.syntax_plane.flatten()[:64],
            self.dataflow_plane.flatten()[:64],
            self.intent_plane.flatten()[:64]
        ])
        cov = np.cov(stacked)
        self.covariance_trace = float(np.trace(cov))
        self.eigenvalue_spectrum = np.linalg.eigvalsh(cov)
    
    def flatten(self) -> np.ndarray:
        return np.concatenate([
            self.syntax_plane.flatten(),
            self.dataflow_plane.flatten(),
            self.intent_plane.flatten()
        ])
    
    def decouple(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_s, s_s, vh_s = np.linalg.svd(self.syntax_plane, full_matrices=False)
        u_d, s_d, vh_d = np.linalg.svd(self.dataflow_plane, full_matrices=False)
        u_i, s_i, vh_i = np.linalg.svd(self.intent_plane, full_matrices=False)
        return u_s @ np.diag(s_s), u_d @ np.diag(s_d), u_i @ np.diag(s_i)
    
    def frobenius_norm(self) -> float:
        return float(
            np.linalg.norm(self.syntax_plane, 'fro') +
            np.linalg.norm(self.dataflow_plane, 'fro') +
            np.linalg.norm(self.intent_plane, 'fro')
        )

class AbstractSignalExtractor(metaclass=ABCMeta):
    @abstractmethod
    def extract_syntax(self, source: str) -> np.ndarray: ...
    
    @abstractmethod
    def extract_dataflow(self, source: str) -> np.ndarray: ...
    
    @abstractmethod
    def extract_intent(self, source: str) -> np.ndarray: ...

class DeepSignalExtractor(AbstractSignalExtractor, metaclass=RegistryMeta):
    def __init__(self, syntax_dim: int = 512, dataflow_dim: int = 512, intent_dim: int = 256):
        self.syntax_dim = syntax_dim
        self.dataflow_dim = dataflow_dim
        self.intent_dim = intent_dim
        self._syntax_kernel = np.random.randn(syntax_dim, syntax_dim) * 0.01
        self._dataflow_kernel = np.random.randn(dataflow_dim, dataflow_dim) * 0.01
        self._intent_kernel = np.random.randn(intent_dim, intent_dim) * 0.01
        self._syntax_bias = np.random.randn(syntax_dim) * 0.001
        self._dataflow_bias = np.random.randn(dataflow_dim) * 0.001
        self._intent_bias = np.random.randn(intent_dim) * 0.001
    
    def extract_syntax(self, source: str) -> np.ndarray:
        tree = ast.parse(source)
        node_types = [type(n).__name__ for n in ast.walk(tree)]
        encoded = np.zeros(self.syntax_dim)
        for i, node_type in enumerate(node_types[:self.syntax_dim]):
            hash_val = hash(node_type) % self.syntax_dim
            encoded[i] = np.sin(hash_val * 0.1) * np.cos(i * 0.05)
        activated = np.tanh(encoded @ self._syntax_kernel + self._syntax_bias)
        return activated
    
    def extract_dataflow(self, source: str) -> np.ndarray:
        tree = ast.parse(source)
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
        flows = np.zeros(self.dataflow_dim)
        for idx, assign in enumerate(assignments[:self.dataflow_dim // 2]):
            targets = len(assign.targets)
            value_depth = self._ast_depth(assign.value)
            flows[idx * 2] = targets / (value_depth + 1.0)
            flows[idx * 2 + 1] = np.log1p(value_depth)
        activated = np.tanh(flows @ self._dataflow_kernel + self._dataflow_bias)
        return activated
    
    def extract_intent(self, source: str) -> np.ndarray:
        tree = ast.parse(source)
        func_defs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        intent_vec = np.zeros(self.intent_dim)
        for i, func in enumerate(func_defs[:self.intent_dim]):
            complexity = self._cyclomatic_complexity(func)
            intent_vec[i] = np.exp(-complexity / 10.0) * np.sin(i * 0.1)
        activated = np.tanh(intent_vec @ self._intent_kernel + self._intent_bias)
        return activated
    
    def _ast_depth(self, node: ast.AST) -> int:
        if not isinstance(node, ast.AST):
            return 0
        return 1 + max((self._ast_depth(child) for child in ast.iter_child_nodes(node)), default=0)
    
    def _cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        return complexity

class AnalyzerOrchestrator(metaclass=RegistryMeta):
    def __init__(self, extractor: AbstractSignalExtractor):
        self.extractor = extractor
        self._history: List[MultiDimensionalTensor] = []
        self._divergence_matrix: Optional[np.ndarray] = None
    
    def analyze(self, source: str) -> MultiDimensionalTensor:
        syntax = self.extractor.extract_syntax(source)
        dataflow = self.extractor.extract_dataflow(source)
        intent = self.extractor.extract_intent(source)
        syntax_plane = self._reshape_or_pad(syntax, 16, 32)
        dataflow_plane = self._reshape_or_pad(dataflow, 16, 32)
        intent_plane = self._reshape_or_pad(intent, 16, 16)
        tensor = MultiDimensionalTensor(
            syntax_plane=syntax_plane,
            dataflow_plane=dataflow_plane,
            intent_plane=intent_plane
        )
        self._history.append(tensor)
        self._update_divergence_matrix()
        return tensor
    
    def _reshape_or_pad(self, vec: np.ndarray, rows: int, cols: int) -> np.ndarray:
        target = rows * cols
        if vec.size < target:
            padded = np.pad(vec, (0, target - vec.size))
        else:
            padded = vec[:target]
        return padded.reshape(rows, cols)
    
    def _update_divergence_matrix(self) -> None:
        n = len(self._history)
        if n < 2:
            return
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                flat_i = self._history[i].flatten()
                flat_j = self._history[j].flatten()
                min_len = min(len(flat_i), len(flat_j))
                dist = np.linalg.norm(flat_i[:min_len] - flat_j[:min_len])
                mat[i, j] = dist
                mat[j, i] = dist
        self._divergence_matrix = mat
    
    def compute_divergence(self, tensor: MultiDimensionalTensor) -> float:
        if not self._history:
            return 0.0
        prev = self._history[-2] if len(self._history) > 1 else self._history[0]
        flat_curr = tensor.flatten()
        flat_prev = prev.flatten()
        min_len = min(len(flat_curr), len(flat_prev))
        return float(np.linalg.norm(flat_curr[:min_len] - flat_prev[:min_len]))
    
    def get_history_mean_tensor(self) -> Optional[np.ndarray]:
        if not self._history:
            return None
        flats = [t.flatten() for t in self._history]
        max_len = max(len(f) for f in flats)
        padded = [np.pad(f, (0, max_len - len(f))) for f in flats]
        return np.mean(padded, axis=0)