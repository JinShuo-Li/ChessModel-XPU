from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess.pgn


VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}


def sampled_positions(plies: int, every: int = 8) -> int:
    return sum(1 for ply in range(plies) if ply >= 5 and ply % every == 0)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter, deduplicate, and game-split a PGN for formal training")
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--validation-output", required=True)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-plies", type=int, default=16)
    parser.add_argument("--required-train-positions", type=int, default=1_000_000)
    parser.add_argument("--required-validation-positions", type=int, default=50_000)
    parser.add_argument("--metadata-output")
    args = parser.parse_args()
    if not 1 <= args.validation_percent <= 50:
        parser.error("--validation-percent must be between 1 and 50")

    source = Path(args.input).resolve()
    train_output = Path(args.train_output).resolve()
    validation_output = Path(args.validation_output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if train_output == validation_output:
        raise ValueError("training and validation outputs must differ")
    for output in (train_output, validation_output):
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)

    train_tmp = train_output.with_suffix(train_output.suffix + ".tmp")
    validation_tmp = validation_output.with_suffix(validation_output.suffix + ".tmp")
    for temporary in (train_tmp, validation_tmp):
        if temporary.exists():
            raise FileExistsError(temporary)

    seen: set[bytes] = set()
    stats = {
        "parsed_games": 0,
        "accepted_games": 0,
        "duplicate_games": 0,
        "parse_error_games": 0,
        "nonstandard_games": 0,
        "invalid_result_games": 0,
        "short_games": 0,
        "train_games": 0,
        "validation_games": 0,
        "train_available_positions": 0,
        "validation_available_positions": 0,
    }
    seed_bytes = str(args.seed).encode("ascii")

    try:
        with source.open(encoding="utf-8", errors="replace") as handle, train_tmp.open("w", encoding="utf-8", newline="\n") as train_handle, validation_tmp.open("w", encoding="utf-8", newline="\n") as validation_handle:
            while (game := chess.pgn.read_game(handle)) is not None:
                stats["parsed_games"] += 1
                if game.errors:
                    stats["parse_error_games"] += 1
                    continue
                if game.headers.get("Variant", "Standard") != "Standard" or "FEN" in game.headers or game.headers.get("SetUp") == "1":
                    stats["nonstandard_games"] += 1
                    continue
                if game.headers.get("Result") not in VALID_RESULTS:
                    stats["invalid_result_games"] += 1
                    continue
                moves = list(game.mainline_moves())
                if len(moves) < args.min_plies:
                    stats["short_games"] += 1
                    continue
                fingerprint = hashlib.sha256(" ".join(move.uci() for move in moves).encode("ascii")).digest()
                if fingerprint in seen:
                    stats["duplicate_games"] += 1
                    continue
                seen.add(fingerprint)
                positions = sampled_positions(len(moves))
                split = int.from_bytes(hashlib.sha256(seed_bytes + fingerprint).digest()[:8], "big") % 100
                if split < args.validation_percent:
                    output_handle = validation_handle
                    stats["validation_games"] += 1
                    stats["validation_available_positions"] += positions
                else:
                    output_handle = train_handle
                    stats["train_games"] += 1
                    stats["train_available_positions"] += positions
                exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
                print(game.accept(exporter), file=output_handle, end="\n\n")
                stats["accepted_games"] += 1
                if stats["parsed_games"] % 10_000 == 0:
                    print(json.dumps({"progress": stats["parsed_games"], "accepted": stats["accepted_games"]}), flush=True)

        if stats["train_available_positions"] < args.required_train_positions:
            raise ValueError(f"training split has only {stats['train_available_positions']} sampled positions; need {args.required_train_positions}")
        if stats["validation_available_positions"] < args.required_validation_positions:
            raise ValueError(f"validation split has only {stats['validation_available_positions']} sampled positions; need {args.required_validation_positions}")
        train_tmp.replace(train_output)
        validation_tmp.replace(validation_output)
    except BaseException:
        train_tmp.unlink(missing_ok=True)
        validation_tmp.unlink(missing_ok=True)
        raise

    metadata_output = Path(args.metadata_output).resolve() if args.metadata_output else train_output.with_suffix(".metadata.json")
    metadata = {
        "source": str(source),
        "source_sha256": file_sha256(source),
        "seed": args.seed,
        "validation_percent": args.validation_percent,
        "min_plies": args.min_plies,
        **stats,
        "train_output": str(train_output),
        "validation_output": str(validation_output),
    }
    metadata_output.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
