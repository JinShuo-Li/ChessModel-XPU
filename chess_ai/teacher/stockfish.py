from __future__ import annotations

import math
from pathlib import Path

import chess
import chess.engine
import numpy as np

from chess_ai.board.moves import move_to_index


class StockfishTeacher:
    def __init__(self, path: str, *, threads=1, hash_mb=128, multipv=8, nodes=10000, temperature=0.15):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": threads, "Hash": hash_mb})
        self.multipv, self.nodes, self.temperature = multipv, nodes, temperature
        self.version = self.engine.id.get("name", "unknown")

    def close(self):
        self.engine.quit()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def _wdl(info, board):
        pov = info["score"].pov(board.turn)
        try:
            wdl = pov.wdl(model="sf16", ply=board.ply())
            values = np.array([wdl.wins, wdl.draws, wdl.losses], dtype=np.float64)
        except Exception:
            score = pov.score(mate_score=100000) or 0
            q = math.tanh(score / 600.0)
            values = np.array([max(q, 0), 1 - abs(q), max(-q, 0)], dtype=np.float64)
        return values / values.sum()

    def analyse(self, board: chess.Board):
        infos = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), multipv=min(self.multipv, board.legal_moves.count()))
        if isinstance(infos, dict):
            infos = [infos]
        moves, qualities, wdls = [], [], []
        for info in infos:
            move = info["pv"][0]
            wdl = self._wdl(info, board)
            moves.append(move_to_index(board, move))
            wdls.append(wdl)
            qualities.append(wdl[0] - wdl[2])
        logits = (np.asarray(qualities) - max(qualities)) / self.temperature
        probs = np.exp(logits); probs /= probs.sum()
        best = infos[0]["score"].pov(board.turn).score(mate_score=100000) or 0
        return np.asarray(moves), probs.astype(np.float32), np.asarray(wdls[0], dtype=np.float32), float(best)

