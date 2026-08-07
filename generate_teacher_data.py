from __future__ import annotations

import argparse

from chess_ai.config import load_config
from chess_ai.data.sampler import fen_positions, pgn_positions, random_positions
from chess_ai.teacher.generation import generate
from chess_ai.teacher.stockfish import StockfishTeacher


def main():
    parser = argparse.ArgumentParser(description="Generate compact Stockfish teacher shards")
    parser.add_argument("--config", required=True); parser.add_argument("--stockfish", required=True); parser.add_argument("--output", required=True); parser.add_argument("--positions", type=int, required=True); parser.add_argument("--nodes", type=int); parser.add_argument("--multipv", type=int); parser.add_argument("--fen-file"); parser.add_argument("--pgn"); parser.add_argument("--shard-size", type=int, default=4096); parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(); cfg = load_config(args.config); sf = cfg["stockfish"]
    positions = pgn_positions(args.pgn, args.positions) if args.pgn else fen_positions(args.fen_file, args.positions) if args.fen_file else random_positions(args.positions, args.seed)
    with StockfishTeacher(args.stockfish, threads=sf["threads"], hash_mb=sf["hash_mb"], multipv=args.multipv or sf["multipv"], nodes=args.nodes or sf["nodes"], temperature=sf["temperature"]) as teacher:
        stats = generate(positions, teacher, args.output, shard_size=args.shard_size, metadata={"source": "pgn" if args.pgn else "fen" if args.fen_file else "random"})
        print({"stockfish": teacher.version, **stats})


if __name__ == "__main__": main()

