from __future__ import annotations

import math
import time
from dataclasses import dataclass

import chess
import numpy as np

from .node import Node


@dataclass
class SearchResult:
    move: chess.Move
    visits: dict[chess.Move, int]
    wdl: np.ndarray
    simulations: int
    elapsed_s: float
    pv: list[chess.Move]


class PUCTSearch:
    def __init__(self, evaluator, simulations=800, leaf_batch_size=64, cpuct=1.5):
        self.evaluator, self.simulations, self.leaf_batch_size, self.cpuct = evaluator, simulations, leaf_batch_size, cpuct

    def _select(self, root):
        node, path = root, [root]
        while node.expanded and node.children:
            scale = math.sqrt(max(1, node.visits))
            node = max(node.children.values(), key=lambda child: -child.q + self.cpuct * child.prior * scale / (1 + child.visits))
            path.append(node)
        return node, path

    @staticmethod
    def _backup(path, leaf_value):
        value = leaf_value
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    @staticmethod
    def _reserve(path):
        # A reversible virtual win from each child's perspective makes that
        # child temporarily unattractive to its parent (-child.q in selection).
        for node in path:
            node.visits += 1
            node.value_sum += 1.0

    @staticmethod
    def _release(path):
        for node in path:
            node.visits -= 1
            node.value_sum -= 1.0

    @staticmethod
    def _expand(node, priors):
        if node.expanded:
            return
        for move, prior in priors.items():
            board = node.board.copy(stack=True); board.push(move)
            node.children[move] = Node(board, prior, move, node)
        node.expanded = True

    def search(self, board: chess.Board):
        if board.is_game_over(claim_draw=True):
            raise ValueError("cannot search a terminal position")
        started = time.perf_counter()
        root = Node(board.copy(stack=True))
        priors, value, wdl = self.evaluator.evaluate([root.board])[0]
        self._expand(root, priors); self._backup([root], value)
        completed = 0
        while completed < self.simulations:
            target = min(self.leaf_batch_size, self.simulations - completed)
            pending, terminal = [], []
            reserved = set()
            attempts = 0
            while len(pending) + len(terminal) < target and attempts < target * 8:
                attempts += 1
                leaf, path = self._select(root)
                key = id(leaf)
                terminal_value = leaf.terminal_value()
                if terminal_value is not None:
                    terminal.append((path, terminal_value)); continue
                if key in reserved:
                    continue
                reserved.add(key); self._reserve(path); pending.append((leaf, path))
            for path, terminal_value in terminal:
                self._backup(path, terminal_value)
            if pending:
                evaluations = self.evaluator.evaluate([leaf.board for leaf, _ in pending])
                for (leaf, path), (priors, leaf_value, _) in zip(pending, evaluations):
                    self._release(path)
                    self._expand(leaf, priors); self._backup(path, leaf_value)
            done = len(pending) + len(terminal)
            if done == 0:
                break
            completed += done
        best = max(root.children.values(), key=lambda child: child.visits)
        pv, cursor = [], root
        while cursor.children:
            cursor = max(cursor.children.values(), key=lambda child: child.visits)
            pv.append(cursor.move)
            if len(pv) >= 16:
                break
        root_q = root.q
        root_wdl = np.array([max(root_q, 0), 1 - abs(root_q), max(-root_q, 0)], dtype=np.float32)
        return SearchResult(best.move, {move: child.visits for move, child in root.children.items()}, root_wdl, completed, time.perf_counter() - started, pv)
