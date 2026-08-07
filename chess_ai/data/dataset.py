from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import chess

from chess_ai.board.moves import POLICY_SIZE, legal_move_mask
from .format import TeacherRecord, read_shard


class TeacherDataset(Dataset):
    def __init__(self, root: str | Path):
        root = Path(root)
        paths = sorted(root.glob("*.npz")) if root.is_dir() else [root]
        if not paths:
            raise FileNotFoundError(f"no .npz shards found at {root}")
        self.records: list[TeacherRecord] = []
        self.profile = {"total_read_s": 0.0, "bitplane_unpack_s": 0.0}
        for path in paths:
            records, manifest = read_shard(path)
            self.records.extend(records)
            for key in self.profile: self.profile[key] += manifest["profile"][key]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        policy[record.move_ids] = record.move_probs
        fen = record.metadata.get("fen")
        if not fen:
            raise ValueError("record metadata must contain FEN for legal policy masking")
        return {"position": torch.from_numpy(record.encoded), "policy": torch.from_numpy(policy), "legal_mask": torch.from_numpy(legal_move_mask(chess.Board(fen))), "wdl": torch.from_numpy(record.wdl), "moves_left": torch.tensor(float(record.metadata.get("moves_left", 0.0)))}
