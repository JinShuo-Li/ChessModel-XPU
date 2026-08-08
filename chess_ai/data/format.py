from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chess_ai.board.encoding import pack_encoded, unpack_encoded

FORMAT_VERSION = 1
MAX_CANDIDATES = 32


@dataclass
class TeacherRecord:
    encoded: np.ndarray
    move_ids: np.ndarray
    move_probs: np.ndarray
    wdl: np.ndarray
    cp: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class PackedShard:
    packed: np.ndarray
    scalars: np.ndarray
    move_ids: np.ndarray
    move_probs: np.ndarray
    wdl: np.ndarray
    cp: np.ndarray
    manifest: dict

    def __len__(self) -> int:
        return self.manifest["count"]

    @property
    def storage_nbytes(self) -> int:
        return sum(array.nbytes for array in (self.packed, self.scalars, self.move_ids, self.move_probs, self.wdl, self.cp))

    def record(self, index: int) -> TeacherRecord:
        valid = self.move_ids[index] >= 0
        return TeacherRecord(
            unpack_encoded(self.packed[index], self.scalars[index]),
            self.move_ids[index][valid].astype(np.int64),
            self.move_probs[index][valid].astype(np.float32),
            self.wdl[index].astype(np.float32),
            float(self.cp[index]),
            self.manifest["records"][index],
        )


def write_shard(path: str | Path, records: list[TeacherRecord], metadata: dict | None = None) -> Path:
    if not records:
        raise ValueError("cannot write an empty shard")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(records)
    packed = np.empty((count, 110, 8), dtype=np.uint8)
    scalars = np.empty((count, 2), dtype=np.float16)
    move_ids = np.full((count, MAX_CANDIDATES), -1, dtype=np.int16)
    move_probs = np.zeros((count, MAX_CANDIDATES), dtype=np.float16)
    wdl = np.empty((count, 3), dtype=np.float16)
    cp = np.empty(count, dtype=np.float32)
    record_meta = []
    for i, record in enumerate(records):
        binary, scalar = pack_encoded(record.encoded)
        packed[i], scalars[i] = binary, scalar
        n = min(len(record.move_ids), MAX_CANDIDATES)
        move_ids[i, :n] = record.move_ids[:n]
        probs = np.asarray(record.move_probs[:n], dtype=np.float32)
        move_probs[i, :n] = probs / max(float(probs.sum()), 1e-12)
        wdl[i] = np.asarray(record.wdl, dtype=np.float32) / max(float(np.asarray(record.wdl).sum()), 1e-12)
        cp[i] = record.cp
        record_meta.append(record.metadata)
    manifest = {"format_version": FORMAT_VERSION, "count": count, "metadata": metadata or {}, "records": record_meta}
    payload = json.dumps(manifest, sort_keys=True)
    checksum = hashlib.sha256(packed.tobytes() + move_ids.tobytes() + payload.encode()).hexdigest()
    np.savez_compressed(path, packed=packed, scalars=scalars, move_ids=move_ids, move_probs=move_probs, wdl=wdl, cp=cp, manifest=payload, checksum=checksum)
    return path


def read_packed_shard(path: str | Path) -> PackedShard:
    started = time.perf_counter()
    with np.load(path, allow_pickle=False) as data:
        # NpzFile lazily decompresses an archive member on every __getitem__.
        # Materialize each member once before iterating over records; indexing
        # data["packed"] inside the loop otherwise re-reads the whole member for
        # every position and amplifies a small shard into gigabytes of I/O.
        manifest_payload = str(data["manifest"])
        expected_checksum = str(data["checksum"])
        packed = data["packed"]
        scalars = data["scalars"]
        move_ids = data["move_ids"]
        move_probs = data["move_probs"]
        wdl = data["wdl"]
        cp = data["cp"]

        manifest = json.loads(manifest_payload)
        if manifest["format_version"] != FORMAT_VERSION:
            raise ValueError(f"unsupported format version {manifest['format_version']}")
        checksum = hashlib.sha256(packed.tobytes() + move_ids.tobytes() + manifest_payload.encode()).hexdigest()
        if checksum != expected_checksum:
            raise ValueError("shard integrity check failed")
    manifest["profile"] = {"total_read_s": time.perf_counter() - started, "bitplane_unpack_s": 0.0}
    return PackedShard(packed, scalars, move_ids, move_probs, wdl, cp, manifest)


def read_shard(path: str | Path) -> tuple[list[TeacherRecord], dict]:
    started = time.perf_counter()
    shard = read_packed_shard(path)
    unpack_started = time.perf_counter()
    records = [shard.record(i) for i in range(len(shard))]
    shard.manifest["profile"] = {"total_read_s": time.perf_counter() - started, "bitplane_unpack_s": time.perf_counter() - unpack_started}
    return records, shard.manifest
