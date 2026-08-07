from __future__ import annotations

import time
from pathlib import Path

from chess_ai.board.encoding import encode_board
from chess_ai.data.format import TeacherRecord, write_shard


def generate(positions, teacher, output, *, shard_size=4096, metadata=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records, total, label_s, shard = [], 0, 0.0, 0
    for board in positions:
        start = time.perf_counter()
        ids, probs, wdl, cp = teacher.analyse(board)
        label_s += time.perf_counter() - start
        records.append(TeacherRecord(encode_board(board), ids, probs, wdl, cp, {"fen": board.fen(), "stockfish": teacher.version, "nodes": teacher.nodes}))
        total += 1
        if len(records) >= shard_size:
            write_shard(output / f"shard-{shard:05d}.npz", records, metadata); records = []; shard += 1
    if records:
        write_shard(output / f"shard-{shard:05d}.npz", records, metadata)
    return {"positions": total, "label_seconds": label_s, "positions_per_second": total / max(label_s, 1e-9)}
