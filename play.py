from __future__ import annotations

import argparse

import chess

from chess_ai.uci.engine import build_engine


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--device", choices=("cpu", "cuda"), required=True); parser.add_argument("--simulations", type=int, default=400); parser.add_argument("--leaf-batch-size", type=int, default=64)
    args = parser.parse_args(); search = build_engine(args.checkpoint, args.device, args.simulations, args.leaf_batch_size); board = chess.Board()
    while not board.is_game_over(claim_draw=True):
        print(board, "\n")
        if board.turn == chess.WHITE:
            raw = input("Your move (UCI, or quit): ").strip()
            if raw == "quit": return
            try: move = board.parse_uci(raw)
            except ValueError as error: print(error); continue
        else:
            result = search.search(board); move = result.move
            print(f"Engine: {move.uci()} WDL={result.wdl.tolist()} simulations={result.simulations} time={result.elapsed_s:.3f}s PV={' '.join(m.uci() for m in result.pv)}")
        board.push(move)
    print(board, board.outcome(claim_draw=True))


if __name__ == "__main__": main()

