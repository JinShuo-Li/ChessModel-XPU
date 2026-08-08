from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import chess

from chess_ai.board.moves import POLICY_SIZE, legal_move_mask
from .format import PackedShard, TeacherRecord, read_packed_shard


class _RecordSequence(Sequence):
    """Compatibility view that materializes TeacherRecord objects on demand."""

    def __init__(self, dataset: "TeacherDataset"):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.dataset._record(i) for i in range(*index.indices(len(self)))]
        return self.dataset._record(index)


class TeacherDataset(Dataset):
    def __init__(self, root: str | Path):
        root = Path(root)
        paths = sorted(root.glob("*.npz")) if root.is_dir() else [root]
        if not paths:
            raise FileNotFoundError(f"no .npz shards found at {root}")
        self.shards: list[PackedShard] = []
        self._ends: list[int] = []
        self.profile = {"total_read_s": 0.0, "bitplane_unpack_s": 0.0}
        total = 0
        for path in paths:
            shard = read_packed_shard(path)
            self.shards.append(shard)
            total += len(shard)
            self._ends.append(total)
            manifest = shard.manifest
            for key in self.profile: self.profile[key] += manifest["profile"][key]
        self.records = _RecordSequence(self)

    @property
    def storage_nbytes(self) -> int:
        return sum(shard.storage_nbytes for shard in self.shards)

    def __len__(self):
        return self._ends[-1]

    def _location(self, index: int) -> tuple[PackedShard, int]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._ends, index)
        start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        return self.shards[shard_index], index - start

    def _record(self, index: int) -> TeacherRecord:
        shard, local_index = self._location(index)
        return shard.record(local_index)

    def metadata(self, index: int) -> dict:
        shard, local_index = self._location(index)
        return shard.manifest["records"][local_index]

    def __getitem__(self, index):
        record = self._record(index)
        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        policy[record.move_ids] = record.move_probs
        fen = record.metadata.get("fen")
        if not fen:
            raise ValueError("record metadata must contain FEN for legal policy masking")
        return {"position": torch.from_numpy(record.encoded), "policy": torch.from_numpy(policy), "legal_mask": torch.from_numpy(legal_move_mask(chess.Board(fen))), "wdl": torch.from_numpy(record.wdl), "moves_left": torch.tensor(float(record.metadata.get("moves_left", 0.0)))}
