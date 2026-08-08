# ChessModel-XPU

ChessModel-XPU is a Linux-native neural chess engine optimized for NVIDIA CUDA GPUs. It distills Stockfish 18 MultiPV/WDL analysis into a compact policy/WDL residual network and uses that network with chess rules and batched PUCT at inference. Stockfish is a teacher and evaluation opponent only; the playable/UCI engine never invokes it.

The research objective is strength per unit compute, not parameter count:

```text
Stockfish teacher → broad distillation → compact policy/WDL network
→ fast batched PUCT → hard-position discovery → deeper relabeling
→ continued distillation → selective self-play → paired evaluation
```

No pretrained strong model or dataset is included. Smoke checkpoints are disposable and have no expected chess strength.

## Platform

This repository targets Linux (native or WSL2 with NVIDIA driver passthrough), bash, Conda Python 3.11, PyTorch CUDA, and NVIDIA GPUs. CPU fallback disguised as CUDA execution is unsupported. `--device cuda` fails if CUDA is unavailable; CPU is used only when explicitly requested.

Requirements: NVIDIA driver with CUDA 12 support (check `nvidia-smi`); BF16 autocast requires an Ampere (RTX 30 series) or newer GPU — on older GPUs use `--precision fp16` or `fp32`.

## Architecture

The input is `112×8×8`, canonically oriented to the side to move:

- Planes 0–103: up to eight history frames, each with friendly then enemy `P,N,B,R,Q,K` planes and a twofold-repetition indicator. Missing older frames are zero.
- Planes 104–107: our/opp king-side and queen-side castling rights.
- Plane 108: canonical en-passant target square.
- Plane 109: absolute side to move (white is one).
- Plane 110: clipped halfmove clock divided by 100.
- Plane 111: clipped fullmove number divided by 200.

Binary planes are bit-packed on disk (eight bytes per plane); policy targets are sparse move-ID/probability arrays. Shards include a format version, SHA-256 integrity check, Stockfish/node metadata, and FEN for legal masking and audit. Split PGN-derived data at game level; never split adjacent positions randomly.

Policy uses the AlphaZero `8×8×73 = 4672` mapping: 56 queen-like rays, 8 knight moves, and 9 underpromotions. Queen promotions share the forward ray. Every legal move—including castling, en passant, and all promotions—round-trips through the mapping. Illegal logits are masked before policy normalization in training, validation, and search.

The network is a BatchNorm residual tower with a spatial 73-plane policy head, three-logit Win/Draw/Loss head, and optional moves-left head. Presets:

| Preset | Blocks × channels | SE | Purpose |
|---|---:|---:|---|
| smoke | 2 × 32 | no | pipeline only (599,340 parameters) |
| tiny | 10 × 128 | no | development/ablation |
| main_cuda | 12 × 192 | hidden 32 | intended formal model (8,932,076 parameters) |
| large | 16 × 256 | hidden 64 | experimental; benchmark first |

PUCT performs selection, expansion, neural evaluation, and alternating-perspective backup. Leaves are reserved and evaluated in configurable batches. Checkmate, stalemate, claimed repetition/fifty-move draws, and insufficient material are terminal before inference.

## Repository layout

```text
configs/                 smoke, tiny, main_cuda, large presets
chess_ai/board/          state planes and 4672 move mapping
chess_ai/model/          residual tower, SE, policy/WDL heads
chess_ai/data/           compact versioned shards and loaders
chess_ai/teacher/        Stockfish UCI teacher and generation
chess_ai/training/       losses, trainer, checkpoint/resume
chess_ai/search/         nodes, batched evaluator, PUCT
chess_ai/evaluation/     neural metrics and match utilities
chess_ai/uci/            Stockfish-free UCI engine
scripts/                 reproducible Linux setup and smoke test
tests/                   CPU correctness and real-CUDA smoke coverage
```

## Linux setup

The safe re-runnable setup script creates `chessmodel`, installs torch only from the official CUDA index, installs the remaining dependencies, and executes real CUDA compute:

```bash
cd /path/to/chess
./scripts/setup_linux.sh
```

If the target machine needs a different wheel set, adjust the `--index-url` in the script (e.g. a newer `cu12x` index matching the installed driver).

Equivalent manual setup:

```bash
conda create -n chessmodel python=3.11 -y
conda activate chessmodel
python -m pip install --upgrade pip
python -m pip install torch \
    --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Torch is deliberately absent from `requirements.txt` to prevent an accidental CPU wheel. Verify more than the availability flag:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -m chess_ai.diagnostics --device cuda
```

The diagnostic performs a CUDA matrix operation and neural forward/backward. It fails loudly if CUDA cannot execute.

## Stockfish 18

Download only from the [official Stockfish download page](https://stockfishchess.org/download/). The AVX2 Linux build is appropriate for most recent x64 systems. Extract it outside Git or under ignored `tools/stockfish/`; never commit the executable or NNUE files. All teacher commands accept an explicit path and use reproducible node budgets.

Example local path used during development:

```text
tools/stockfish/unpacked/stockfish/stockfish-ubuntu-x86-64-avx2
```

## Tests and required smoke test

```bash
conda activate chessmodel
pytest -q
```

Run the complete tiny Stockfish → shard → loader → CUDA forward/backward → optimizer → checkpoint/reload → batched PUCT → legal move → UCI path:

```bash
./scripts/smoke_test.sh \
    tools/stockfish/unpacked/stockfish/stockfish-ubuntu-x86-64-avx2
```

It labels only 16 random positions at 100 nodes/MultiPV 2, executes five optimizer steps, searches four simulations, and removes `tmp/smoke` in `finally`.

## Benchmark before choosing batch size

Do not assume BF16, channels-last, or `torch.compile` wins. Establish a short eager BF16 baseline, then vary one dimension:

```bash
python benchmark_cuda.py --config configs/main_cuda.yaml --steps 5
python benchmark_cuda.py --config configs/main_cuda.yaml --batches 128 256 512 --steps 5 --precision fp32
python benchmark_cuda.py --config configs/main_cuda.yaml --batches 128 256 512 --steps 5 --precision bf16 --layout channels_last
python benchmark_cuda.py --config configs/main_cuda.yaml --batches 128 256 --steps 5 --precision bf16 --compile
```

OOM is reported and stops larger batches gracefully. Output includes forward/training positions per second, step latency, dtype, layout, eager/compile mode, parameter count, and approximate allocated CUDA memory when exposed by PyTorch.

## Teacher datasets

Stockfish labeling and neural-network training are separate, sequential stages. The
generator first writes complete, checksummed `.npz` shards; `train.py` subsequently
loads those immutable shards. It does not train one position as soon as Stockfish
produces it, and the formal workflow does not require the generator and trainer to
run at the same time.

Formal broad randomized generation:

```bash
python generate_teacher_data.py \
    --config configs/main_cuda.yaml \
    --stockfish tools/stockfish/unpacked/stockfish/stockfish-ubuntu-x86-64-avx2 \
    --output data/teacher_main \
    --positions 1000000 \
    --nodes 10000 \
    --multipv 8
```

PGN-driven, correlation-thinned sampling (every eighth eligible ply internally):

```bash
python generate_teacher_data.py \
    --config configs/main_cuda.yaml \
    --stockfish tools/stockfish/unpacked/stockfish/stockfish-ubuntu-x86-64-avx2 \
    --output data/teacher_pgn \
    --positions 1000000 \
    --nodes 10000 \
    --multipv 8 \
    --pgn datasets/games.pgn
```

PGNs supply realistic positions and game histories; their game results are not used
as the policy/WDL labels. Stockfish 18 analyzes each sampled position and supplies
the actual MultiPV policy and WDL targets. The sampler skips the first five plies and
then takes every eighth eligible ply to reduce adjacent-position correlation.

For rigorous validation, divide PGNs into train/validation files by game before
generation. Never randomly split positions from the same game across datasets. FEN
sources use `--fen-file positions.fen`. MultiPV expected WDL quality
`P(win)-P(loss)` is converted to a soft target with configured temperature (0.15 in
the formal config); mate scores are mapped explicitly through a large mate score
when retaining centipawns.

## One-million-position formal workflow

Run these stages in order. Dataset production is CPU/Stockfish and disk-I/O work;
model training begins only after both datasets pass verification.
For copy-ready, single-line bash commands covering generation through human
play, use [`INSTRUCTIONS.md`](INSTRUCTIONS.md).

### 1. Prepare game-level-disjoint PGNs

Download a reasonably sized, high-quality standard-chess PGN into the ignored
`datasets/source` directory. The preparation script removes parse failures,
non-standard starting positions, games without a decisive/draw result, short games,
and duplicate move sequences. It then makes a deterministic game-level 90/10 split
and refuses to publish outputs unless both sides contain enough sampleable positions:

```bash
conda run -n chessmodel python scripts/prepare_formal_pgn.py \
    --input datasets/source/high_quality_games.pgn \
    --train-output datasets/formal_train.pgn \
    --validation-output datasets/formal_validation.pgn \
    --validation-percent 10 \
    --seed 7 \
    --required-train-positions 1000000 \
    --required-validation-positions 50000 \
    --metadata-output datasets/formal_pgn_metadata.json
```

The script writes through temporary files and refuses to overwrite an existing
split. Preserve the generated metadata JSON and source hash locally for provenance.
PGNs and metadata under `datasets` are intentionally not versioned.

### 2. Generate and verify Stockfish labels

Confirm the Stockfish path, then run:

```bash
./scripts/generate_formal_1m.sh
```

Defaults are 1,000,000 training positions, 50,000 validation positions, 10,000
Stockfish nodes, MultiPV 8, and 4,096 records per shard. This produces 245 training
shards under `data/formal_1m_train` and 13 validation shards under
`data/formal_50k_validation`. The script refuses existing output directories,
requires distinct train/validation PGNs, and verifies every checksum and the exact
shard and record totals before reporting success. Override any default with an
environment variable, e.g. `STOCKFISH=path/to/stockfish ./scripts/generate_formal_1m.sh`.

To recheck completed data independently:

```bash
conda run -n chessmodel python scripts/verify_teacher_dataset.py \
    --dataset data/formal_1m_train \
    --expected-positions 1000000 \
    --expected-shards 245

conda run -n chessmodel python scripts/verify_teacher_dataset.py \
    --dataset data/formal_50k_validation \
    --expected-positions 50000 \
    --expected-shards 13
```

`TeacherDataset` loads each shard's bit-packed boards and sparse targets once, keeps
that compact representation in memory, and decodes only requested samples. The
million-position dataset therefore avoids expanding all boards into resident
float32 tensors. Opening shards is disk-I/O-heavy; steady-state forward/backward
training is primarily GPU-heavy.

### 3. Benchmark and train

Benchmark first. If batch 512 is unsuitable on the target machine, copy
`configs/formal_1m.yaml` and change only the tested batch size. Start formal training
only after the verification commands above succeed:

```bash
python train.py \
    --config configs/formal_1m.yaml \
    --dataset data/formal_1m_train \
    --device cuda \
    --output checkpoints/formal_1m_latest.pt \
    --logdir runs/formal_1m
```

Loss is soft-target policy cross entropy plus WDL cross entropy and optional Huber moves-left loss, with configurable weights. AdamW uses warmup plus cosine decay and `1e-4` default weight decay. BF16 CUDA autocast is the default main preset.

Monitor:

```bash
tensorboard --logdir runs
```

### 4. Validate and resume safely

Evaluate against the game-disjoint validation set:

```bash
python evaluate.py \
    --checkpoint checkpoints/formal_1m_latest.pt \
    --dataset data/formal_50k_validation \
    --device cuda \
    --batch-size 512
```

Resume model, optimizer, scheduler, step/epoch, CPU RNG states, config, architecture
metadata, and recorded Git commit:

```bash
python train.py \
    --config configs/formal_1m.yaml \
    --dataset data/formal_1m_train \
    --device cuda \
    --resume checkpoints/formal_1m_latest.pt \
    --output checkpoints/formal_1m_latest.pt \
    --logdir runs/formal_1m
```

## Validation and engine matches

Neural validation metrics include policy cross entropy/KL, teacher top-1/top-3 agreement, WDL cross entropy/accuracy, and WDL Brier calibration:

```bash
python evaluate.py \
    --checkpoint checkpoints/main_latest.pt \
    --dataset data/teacher_validation \
    --device cuda \
    --batch-size 512
```

Paired-color model versus Stockfish (formal evaluation; not a smoke command):

```bash
python engine_match.py \
    --checkpoint checkpoints/main_latest.pt \
    --stockfish tools/stockfish/unpacked/stockfish/stockfish-ubuntu-x86-64-avx2 \
    --device cuda \
    --games 100 \
    --simulations 800 \
    --leaf-batch-size 64 \
    --stockfish-nodes 10000 \
    --pgn artifacts/main-vs-sf.pgn
```

For a paired new/previous checkpoint match, replace `--stockfish ... --stockfish-nodes ...` with `--opponent-checkpoint checkpoints/previous.pt`. Reports include W/D/L, score, logistic Elo estimate, average move time, simulations, and PGN. Tiny matches do not justify an Elo claim.

## Hard-example mining and deeper relabeling

Rank disagreement, policy KL, WDL error, entropy, and uncertainty; this writes both audit JSON and a matching FEN list:

```bash
python mine_hard_positions.py \
    --checkpoint checkpoints/main_latest.pt \
    --dataset data/teacher_candidates \
    --output hard_positions/ranked.json \
    --device cuda \
    --top-k 100000
```

Relabel those exact positions more deeply (future formal job):

```bash
python generate_teacher_data.py \
    --config configs/main_cuda.yaml \
    --stockfish tools/stockfish/unpacked/stockfish/stockfish-ubuntu-x86-64-avx2 \
    --output data/teacher_hard_deep \
    --positions 100000 \
    --nodes 100000 \
    --multipv 12 \
    --fen-file hard_positions/ranked.fen
```

Continue distillation by placing compatible broad and hard shards under one dataset directory (or constructing a desired sampled mixture directory) and resume training. Sampling ratios are data/config decisions, not hard-coded.

## Selective self-play

Self-play stores MCTS visit targets and final game WDL. Run only after supervised policy quality is useful:

```bash
python selfplay.py \
    --checkpoint checkpoints/main_latest.pt \
    --output selfplay_data/main \
    --device cuda \
    --games 10000 \
    --simulations 800 \
    --leaf-batch-size 64 \
    --temperature 1.0
```

Train on a deliberate teacher/self-play mixture by collecting selected shards into a dataset directory. A possible 70/30 ratio is an experiment, not a constant in the code.

## Human play and UCI

Play as White in the console:

```bash
python play.py \
    --checkpoint checkpoints/main_latest.pt \
    --device cuda \
    --simulations 800 \
    --leaf-batch-size 64
```

Run the Stockfish-free UCI process for a GUI:

```bash
python -m chess_ai.uci.engine \
    --checkpoint checkpoints/main_latest.pt \
    --device cuda \
    --simulations 800 \
    --leaf-batch-size 64
```

Serve a local browser game using an explicit checkpoint and TCP port:

```bash
python serve.py \
    --checkpoint checkpoints/formal_tiny_latest.pt \
    --device cuda \
    --host 127.0.0.1 \
    --port 8765 \
    --simulations 400 \
    --leaf-batch-size 64
```

Open `http://127.0.0.1:8765` and choose a color. The same server exposes JSON endpoints `GET /api/state`, `POST /api/new` with `{"human_color":"white|black"}`, and `POST /api/move` with `{"move":"e2e4"}`. It binds to localhost by default; do not expose it to an untrusted network because it has no authentication.

Implemented commands are `uci`, `isready`, `ucinewgame`, `position startpos|fen ... [moves ...]`, `go`, `stop`, and `quit`. Search info includes nodes, time, and principal variation.

## Profiling

Teacher generation prints labeling throughput. Trainer step records CPU→GPU transfer, forward, backward, and optimizer time. `NeuralEvaluator.timings` records batched leaf inference calls/positions/time; `SearchResult.elapsed_s` covers tree plus inference. The CUDA benchmark separates forward and full training throughput. These lightweight timings identify whether labeling, shard loading/unpacking, transfers, model kernels, or tree logic is the next bottleneck.

## Git workflow and artifacts

Use small reviewed commits and preserve reproducibility:

```bash
git status
git add chess_ai tests configs README.md
git commit -m "feat: describe the verified change"
git push origin linux-cuda
```

Datasets, PGNs, Stockfish binaries, checkpoints, runs, logs, artifacts, executables, credentials, and temporary smoke files are ignored. Inspect `git status` and tracked file sizes before every push.

## Troubleshooting

- `CUDA was explicitly requested...`: confirm `torch.version.cuda` is set and `torch.cuda.is_available()` is true; reinstall torch from the CUDA index, then run diagnostics. Never mask this with CPU fallback.
- CUDA unavailable in WSL2: install the NVIDIA Windows driver with WSL support; verify with `nvidia-smi` inside WSL2.
- Conda channel Terms of Service: accept the prompted official channel terms, then rerun `scripts/setup_linux.sh`.
- CUDA OOM: benchmark lower batches first; compare BF16 and layouts; do not assume compile helps.
- BF16 kernel error: reproduce with `--precision fp32` in the benchmark. Keep formal training stopped until the failing CUDA path is understood. BF16 requires an Ampere (RTX 30 series) or newer GPU; use fp16/fp32 on older GPUs.
- Stockfish not found: pass a quoted absolute path; do not hard-code it into source or commit it.
- Slow first command: native CUDA/driver initialization in a fresh Python process can dominate tiny jobs.
- UCI GUI appears idle: ensure the GUI launches the Python module with the same Conda environment and uses an existing compatible checkpoint.
- Shard integrity/version error: do not bypass it; regenerate or migrate the shard explicitly.

## License

Project code is MIT-licensed. Stockfish is a separate GPLv3 program and is not distributed by this repository.
