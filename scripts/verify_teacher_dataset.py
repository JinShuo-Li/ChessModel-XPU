from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_ai.data import read_packed_shard


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify teacher shard integrity and exact record count")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected-positions", type=int, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    args = parser.parse_args()

    paths = sorted(Path(args.dataset).glob("*.npz"))
    if len(paths) != args.expected_shards:
        raise ValueError(f"expected {args.expected_shards} shards, found {len(paths)}")
    positions = storage_bytes = 0
    for path in paths:
        shard = read_packed_shard(path)
        positions += len(shard)
        storage_bytes += shard.storage_nbytes
    if positions != args.expected_positions:
        raise ValueError(f"expected {args.expected_positions} positions, found {positions}")
    print(json.dumps({"dataset": args.dataset, "shards": len(paths), "positions": positions, "packed_storage_bytes": storage_bytes}))


if __name__ == "__main__":
    main()
