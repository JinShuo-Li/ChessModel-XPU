import chess
import numpy as np

from chess_ai.board import encode_board, move_to_index
from chess_ai.data import TeacherDataset, TeacherRecord, read_shard, write_shard


def test_shard_and_dataset_roundtrip(tmp_path):
    board = chess.Board(); move = chess.Move.from_uci("e2e4")
    record = TeacherRecord(encode_board(board), np.array([move_to_index(board, move)]), np.array([1.0]), np.array([0.25, 0.5, 0.25]), 22.0, {"fen": board.fen()})
    path = write_shard(tmp_path / "shard-00000.npz", [record], {"stockfish": "test"})
    loaded, manifest = read_shard(path)
    assert manifest["format_version"] == 1
    np.testing.assert_allclose(loaded[0].encoded, record.encoded, atol=3e-4)
    dataset = TeacherDataset(tmp_path); item = dataset[0]
    assert item["position"].shape == (112, 8, 8)
    assert item["policy"].sum() == 1
    assert item["legal_mask"].sum() == 20
    assert item["wdl"].shape == (3,)
