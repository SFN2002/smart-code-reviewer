from __future__ import annotations
import ast
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Any,
    Protocol,
    runtime_checkable,
    Union,
)
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import numpy as np


@runtime_checkable
class ASTVisitorProtocol(Protocol):
    def visit(self, node: ast.AST) -> Any: ...


@dataclass
class SymbolNode:
    name: str
    node_type: str
    scope_depth: int
    references: List[str] = field(default_factory=list)
    definition_line: int = 0
    usage_count: int = 0


class SymbolTable:
    def __init__(self):
        self._symbols: Dict[str, List[SymbolNode]] = {}
        self._scope_stack: List[str] = []
        self._scope_depth: int = 0

    def register(self, name: str, node_type: str, line: int) -> None:
        depth = self._scope_depth
        node = SymbolNode(
            name=name, node_type=node_type, scope_depth=depth, definition_line=line
        )
        if name not in self._symbols:
            self._symbols[name] = []
        self._symbols[name].append(node)

    def resolve(self, name: str) -> Optional[SymbolNode]:
        if name in self._symbols and self._symbols[name]:
            return self._symbols[name][-1]
        return None

    def increment_usage(self, name: str) -> None:
        resolved = self.resolve(name)
        if resolved:
            resolved.usage_count += 1

    def all_symbols(self) -> Dict[str, List[SymbolNode]]:
        return self._symbols

    def compute_scope_entropy(self) -> float:
        depths = [node.scope_depth for syms in self._symbols.values() for node in syms]
        if not depths:
            return 0.0
        unique, counts = np.unique(depths, return_counts=True)
        probs = counts / np.sum(counts)
        return float(-np.sum(probs * np.log2(probs + 1e-12)))


@dataclass
class ControlFlowEdge:
    source: int
    target: int
    edge_type: str
    condition: Optional[str] = None


class ControlFlowGraph:
    def __init__(self):
        self._nodes: Set[int] = set()
        self._edges: List[ControlFlowEdge] = []
        self._adjacency: Dict[int, List[int]] = {}
        self._in_degree: Dict[int, int] = {}
        self._out_degree: Dict[int, int] = {}

    def add_node(self, line: int) -> None:
        self._nodes.add(line)
        if line not in self._adjacency:
            self._adjacency[line] = []
        if line not in self._in_degree:
            self._in_degree[line] = 0
        if line not in self._out_degree:
            self._out_degree[line] = 0

    def add_edge(
        self,
        source: int,
        target: int,
        edge_type: str = "sequential",
        condition: Optional[str] = None,
    ) -> None:
        self._edges.append(
            ControlFlowEdge(
                source=source, target=target, edge_type=edge_type, condition=condition
            )
        )
        if source in self._adjacency:
            self._adjacency[source].append(target)
        self._out_degree[source] = self._out_degree.get(source, 0) + 1
        self._in_degree[target] = self._in_degree.get(target, 0) + 1

    def to_adjacency_matrix(self) -> np.ndarray:
        sorted_nodes = sorted(self._nodes)
        n = len(sorted_nodes)
        mat = np.zeros((n, n))
        idx_map = {node: i for i, node in enumerate(sorted_nodes)}
        for edge in self._edges:
            if edge.source in idx_map and edge.target in idx_map:
                mat[idx_map[edge.source], idx_map[edge.target]] = 1.0
        return mat

    def compute_pagerank(
        self, damping: float = 0.85, iterations: int = 100
    ) -> Dict[int, float]:
        nodes = sorted(self._nodes)
        n = len(nodes)
        if n == 0:
            return {}
        mat = self.to_adjacency_matrix()
        col_sums = mat.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        transition = mat / col_sums
        rank = np.ones(n) / n
        for _ in range(iterations):
            rank = (1.0 - damping) / n + damping * (transition @ rank)
        return {node: float(rank[i]) for i, node in enumerate(nodes)}


class DeepASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.symbol_table = SymbolTable()
        self.cfg = ControlFlowGraph()
        self._current_scope: List[str] = []
        self._last_line: int = 0
        self._loop_depth: int = 0
        self._branch_depth: int = 0

    def visit_Module(self, node: ast.Module) -> None:
        self._current_scope.append("module")
        for stmt in node.body:
            self.visit(stmt)
        self._current_scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.symbol_table.register(node.name, "function", node.lineno)
        self._current_scope.append(node.name)
        self.cfg.add_node(node.lineno)
        if self._last_line > 0 and self._last_line != node.lineno:
            self.cfg.add_edge(self._last_line, node.lineno, "call")
        self._last_line = node.lineno
        for arg in node.args.args:
            self.symbol_table.register(arg.arg, "parameter", node.lineno)
        for stmt in node.body:
            self.visit(stmt)
        self._current_scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.symbol_table.register(node.name, "async_function", node.lineno)
        self._current_scope.append(node.name)
        self.cfg.add_node(node.lineno)
        if self._last_line > 0:
            self.cfg.add_edge(self._last_line, node.lineno, "async_call")
        self._last_line = node.lineno
        for arg in node.args.args:
            self.symbol_table.register(arg.arg, "async_parameter", node.lineno)
        for stmt in node.body:
            self.visit(stmt)
        self._current_scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbol_table.register(node.name, "class", node.lineno)
        self._current_scope.append(node.name)
        self.cfg.add_node(node.lineno)
        for base in node.bases:
            self.visit(base)
        for stmt in node.body:
            self.visit(stmt)
        self._current_scope.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        self.cfg.add_node(node.lineno)
        if self._last_line > 0 and self._last_line != node.lineno:
            self.cfg.add_edge(self._last_line, node.lineno, "assign")
        self._last_line = node.lineno
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.symbol_table.register(target.id, "variable", node.lineno)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        self.symbol_table.register(elt.id, "variable", node.lineno)
        self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.cfg.add_node(node.lineno)
        if self._last_line > 0:
            self.cfg.add_edge(self._last_line, node.lineno, "aug_assign")
        self._last_line = node.lineno
        if isinstance(node.target, ast.Name):
            self.symbol_table.increment_usage(node.target.id)
        self.visit(node.value)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.symbol_table.increment_usage(node.id)

    def visit_If(self, node: ast.If) -> None:
        self.cfg.add_node(node.lineno)
        if self._last_line > 0:
            self.cfg.add_edge(self._last_line, node.lineno, "branch", "if")
        self._last_line = node.lineno
        self._branch_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        if node.orelse:
            self.cfg.add_edge(
                node.lineno,
                (
                    node.orelse[0].lineno
                    if hasattr(node.orelse[0], "lineno")
                    else node.lineno + 1
                ),
                "branch",
                "else",
            )
            for stmt in node.orelse:
                self.visit(stmt)
        self._branch_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self.cfg.add_node(node.lineno)
        if self._last_line > 0:
            self.cfg.add_edge(self._last_line, node.lineno, "loop", "for")
        self._last_line = node.lineno
        self._loop_depth += 1
        if isinstance(node.target, ast.Name):
            self.symbol_table.register(node.target.id, "loop_var", node.lineno)
        for stmt in node.body:
            self.visit(stmt)
        self.cfg.add_edge(
            (
                node.body[-1].lineno
                if node.body and hasattr(node.body[-1], "lineno")
                else node.lineno
            ),
            node.lineno,
            "loop_back",
            "for",
        )
        self._loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self.cfg.add_node(node.lineno)
        if self._last_line > 0:
            self.cfg.add_edge(self._last_line, node.lineno, "loop", "while")
        self._last_line = node.lineno
        self._loop_depth += 1
        for stmt in node.body:
            self.visit(stmt)
        if node.body and hasattr(node.body[-1], "lineno"):
            self.cfg.add_edge(node.body[-1].lineno, node.lineno, "loop_back", "while")
        self._loop_depth -= 1

    def visit_Try(self, node: ast.Try) -> None:
        self.cfg.add_node(node.lineno)
        if self._last_line > 0:
            self.cfg.add_edge(self._last_line, node.lineno, "try")
        self._last_line = node.lineno
        for stmt in node.body:
            self.visit(stmt)
        for handler in node.handlers:
            self.visit(handler)
        if node.finalbody:
            for stmt in node.finalbody:
                self.visit(stmt)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        self.cfg.add_node(node.lineno)
        if self._last_line > 0:
            self.cfg.add_edge(self._last_line, node.lineno, "exception")
        self._last_line = node.lineno
        if node.type and isinstance(node.type, ast.Name):
            self.symbol_table.register(node.type.id, "exception_type", node.lineno)
        for stmt in node.body:
            self.visit(stmt)

    def compute_symbol_density(self) -> float:
        total_refs = sum(
            len(s.references)
            for syms in self.symbol_table.all_symbols().values()
            for s in syms
        )
        total_usage = sum(
            s.usage_count
            for syms in self.symbol_table.all_symbols().values()
            for s in syms
        )
        total_syms = sum(len(syms) for syms in self.symbol_table.all_symbols().values())
        if total_syms == 0:
            return 0.0
        return (total_refs + total_usage) / total_syms

    def compute_complexity_vector(self, dim: int = 32) -> np.ndarray:
        vec = np.zeros(dim)
        symbols = self.symbol_table.all_symbols()
        func_count = sum(
            1 for syms in symbols.values() for s in syms if s.node_type == "function"
        )
        class_count = sum(
            1 for syms in symbols.values() for s in syms if s.node_type == "class"
        )
        var_count = sum(
            1 for syms in symbols.values() for s in syms if s.node_type == "variable"
        )
        loop_count = sum(
            1 for syms in symbols.values() for s in syms if s.node_type == "loop_var"
        )
        vec[0] = func_count
        vec[1] = class_count
        vec[2] = var_count
        vec[3] = loop_count
        vec[4] = self._loop_depth
        vec[5] = self._branch_depth
        vec[6] = len(self.cfg._nodes)
        vec[7] = len(self.cfg._edges)
        pagerank = self.cfg.compute_pagerank()
        if pagerank:
            vec[8] = np.mean(list(pagerank.values()))
            vec[9] = np.std(list(pagerank.values()))
        vec[10] = self.symbol_table.compute_scope_entropy()
        return vec


class ASTToolkit:
    def __init__(self):
        self._visitor_class = DeepASTVisitor

    def parse_and_map(self, source: str) -> Tuple[SymbolTable, ControlFlowGraph]:
        tree = ast.parse(source)
        visitor = self._visitor_class()
        visitor.visit(tree)
        return visitor.symbol_table, visitor.cfg

    def extract_symbol_vector(self, source: str, dim: int = 128) -> np.ndarray:
        sym_table, cfg = self.parse_and_map(source)
        vec = np.zeros(dim)
        all_syms = sym_table.all_symbols()
        for i, (name, nodes) in enumerate(list(all_syms.items())[:dim]):
            vec[i] = len(nodes) * np.sin(hash(name) % 100) * np.cos(i * 0.1)
        complexity = self._compute_complexity_from_source(source)
        if dim > 64:
            vec[64 : min(dim, 64 + len(complexity))] = complexity[: dim - 64]
        return vec

    def _compute_complexity_from_source(self, source: str) -> np.ndarray:
        tree = ast.parse(source)
        visitor = DeepASTVisitor()
        visitor.visit(tree)
        return visitor.compute_complexity_vector()

    def compute_cfg_laplacian(self, source: str) -> np.ndarray:
        _, cfg = self.parse_and_map(source)
        mat = cfg.to_adjacency_matrix()
        n = mat.shape[0]
        if n == 0:
            return np.array([[0.0]])
        degree = np.diag(mat.sum(axis=1))
        return degree - mat
