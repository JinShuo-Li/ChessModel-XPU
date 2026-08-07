from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from chess_ai.evaluation.matches import elo_from_score, play_game
from chess_ai.uci.engine import build_engine


def neural_player(search):
    return lambda board: search.search(board)


def main():
    parser = argparse.ArgumentParser(description="Paired-color neural engine matches")
    parser.add_argument("--checkpoint", required=True); group = parser.add_mutually_exclusive_group(required=True); group.add_argument("--opponent-checkpoint"); group.add_argument("--stockfish")
    parser.add_argument("--device", choices=("cpu", "xpu"), required=True); parser.add_argument("--games", type=int, default=20); parser.add_argument("--simulations", type=int, default=400); parser.add_argument("--leaf-batch-size", type=int, default=64); parser.add_argument("--stockfish-nodes", type=int, default=10000); parser.add_argument("--pgn", default="matches.pgn")
    args = parser.parse_args()
    if args.games % 2: parser.error("--games must be even for paired colors")
    Path(args.pgn).parent.mkdir(parents=True, exist_ok=True)
    ours = neural_player(build_engine(args.checkpoint, args.device, args.simulations, args.leaf_batch_size))
    sf = None
    if args.stockfish:
        sf = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        opponent = lambda board: sf.play(board, chess.engine.Limit(nodes=args.stockfish_nodes)).move
        opponent_name = sf.id.get("name", "Stockfish")
    else:
        opponent = neural_player(build_engine(args.opponent_checkpoint, args.device, args.simulations, args.leaf_batch_size)); opponent_name = Path(args.opponent_checkpoint).name
    wins = draws = losses = 0; move_times = []
    try:
        with open(args.pgn, "w", encoding="utf-8") as handle:
            for game_id in range(args.games):
                ours_white = game_id % 2 == 0
                board, outcome, average_time = play_game(ours if ours_white else opponent, opponent if ours_white else ours); move_times.append(average_time)
                if outcome is None or outcome.winner is None: draws += 1
                elif outcome.winner == ours_white: wins += 1
                else: losses += 1
                game = chess.pgn.Game.from_board(board); game.headers["White"] = "ChessModel-XPU" if ours_white else opponent_name; game.headers["Black"] = opponent_name if ours_white else "ChessModel-XPU"; print(game, file=handle, end="\n\n")
    finally:
        if sf: sf.quit()
    score = (wins + 0.5 * draws) / args.games
    print(json.dumps({"wins": wins, "draws": draws, "losses": losses, "score": score, "estimated_elo_difference": elo_from_score(score), "average_move_time_s": sum(move_times) / len(move_times), "simulations": args.simulations, "pgn": args.pgn}, indent=2))


if __name__ == "__main__": main()
