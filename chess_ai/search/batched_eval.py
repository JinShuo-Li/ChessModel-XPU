from __future__ import annotations

import time

import numpy as np
import torch

from chess_ai.board.encoding import encode_board
from chess_ai.board.moves import legal_moves_and_indices
from chess_ai.runtime import autocast_context


class NeuralEvaluator:
    def __init__(self, model, device, precision="bf16"):
        self.model, self.device, self.precision = model, device, precision
        self.model.eval()
        self.timings = {"calls": 0, "positions": 0, "inference_s": 0.0}

    @torch.inference_mode()
    def evaluate(self, boards):
        start = time.perf_counter()
        inputs = torch.from_numpy(np.stack([encode_board(board) for board in boards])).to(self.device)
        with autocast_context(self.device, self.precision):
            outputs = self.model(inputs)
        policies = outputs["policy"].float().cpu().numpy()
        wdls = torch.softmax(outputs["wdl"].float(), dim=1).cpu().numpy()
        results = []
        for board, logits, wdl in zip(boards, policies, wdls):
            moves, ids = legal_moves_and_indices(board)
            order = np.argsort(ids, kind="stable")
            ids = ids[order]
            moves = [moves[i] for i in order]
            legal_logits = logits[ids]
            legal_logits -= legal_logits.max(initial=0.0)
            probs = np.exp(legal_logits); probs /= probs.sum()
            priors = {move: float(prob) for move, prob in zip(moves, probs)}
            results.append((priors, float(wdl[0] - wdl[2]), wdl))
        elapsed = time.perf_counter() - start
        self.timings["calls"] += 1; self.timings["positions"] += len(boards); self.timings["inference_s"] += elapsed
        return results
