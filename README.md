# ChessModel-XPU

ChessModel-XPU is a Windows-native neural chess engine optimized for Intel Arc XPU. It distills Stockfish 18 MultiPV/WDL analysis into a compact policy/WDL residual network and uses that network with chess rules and batched PUCT at inference. Stockfish is a teacher and evaluation opponent only; the playable/UCI engine never invokes it.

The research objective is strength per unit compute, not parameter count:

```text
Stockfish teacher → broad distillation → compact policy/WDL network
→ fast batched PUCT → hard-position discovery → deeper relabeling
→ continued distillation → selective self-play → paired evaluation
```

No pretrained strong model or dataset is included. Smoke checkpoints are disposable and have no expected chess strength.

## Platform

This repository targets native Windows 11, native PowerShell, native Conda Python 3.11, PyTorch XPU, and Intel Arc. WSL, CUDA wheels, and CPU fallback disguised as XPU execution are unsupported. `--device xpu` fails if XPU is unavailable; CPU is used only when explicitly requested.

Tested development machine: Intel Core Ultra 9 285H, Intel Arc 140T (16 GB shared memory), PyTorch 2.13.0+xpu.

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
| main_xpu | 12 × 192 | hidden 32 | intended formal model (8,932,076 parameters) |
| large | 16 × 256 | hidden 64 | experimental; benchmark first |

PUCT performs selection, expansion, neural evaluation, and alternating-perspective backup. Leaves are reserved and evaluated in configurable batches. Checkmate, stalemate, claimed repetition/fifty-move draws, and insufficient material are terminal before inference.

## Repository layout

```text
configs/                 smoke, tiny, main_xpu, large presets
chess_ai/board/          state planes and 4672 move mapping
chess_ai/model/          residual tower, SE, policy/WDL heads
chess_ai/data/           compact versioned shards and loaders
chess_ai/teacher/        Stockfish UCI teacher and generation
chess_ai/training/       losses, trainer, checkpoint/resume
chess_ai/search/         nodes, batched evaluator, PUCT
chess_ai/evaluation/     neural metrics and match utilities
chess_ai/uci/            Stockfish-free UCI engine
scripts/                 reproducible Windows setup and smoke test
tests/                   CPU correctness and real-XPU smoke coverage
```

## Native Windows setup

The safe re-runnable setup script creates `chessmodel`, installs torch only from the official XPU index, installs the remaining dependencies, and executes real XPU compute:

```powershell
Set-Location C:\Users\lijs\Desktop\work\chess
.\scripts\setup_windows.ps1
```

Equivalent manual setup:

```powershell
conda create -n chessmodel python=3.11 -y
conda activate chessmodel
python -m pip install --upgrade pip
python -m pip install torch `
    --index-url https://download.pytorch.org/whl/xpu
python -m pip install -r requirements.txt
```

Torch is deliberately absent from `requirements.txt` to prevent an accidental CPU or CUDA wheel. Verify more than the availability flag:

```powershell
python -c "import torch; print(torch.__version__); print(torch.xpu.is_available()); print(torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'NO XPU')"
python -m chess_ai.diagnostics --device xpu
```

The diagnostic performs an XPU matrix operation and neural forward/backward. It fails loudly if XPU cannot execute.

## Stockfish 18

Download only from the [official Stockfish download page](https://stockfishchess.org/download/). The AVX2 Windows build is appropriate for most recent x64 systems. Extract it outside Git or under ignored `tools\stockfish\`; never commit the executable or NNUE files. All teacher commands accept an explicit path and use reproducible node budgets.

Example local path used during development:

```text
tools\stockfish\unpacked\stockfish\stockfish-windows-x86-64-avx2.exe
```

## Tests and required smoke test

```powershell
conda activate chessmodel
pytest -q
```

Run the complete tiny Stockfish → shard → loader → XPU forward/backward → optimizer → checkpoint/reload → batched PUCT → legal move → UCI path:

```powershell
.\scripts\smoke_test.ps1 `
    -Stockfish ".\tools\stockfish\unpacked\stockfish\stockfish-windows-x86-64-avx2.exe"
```

It labels only 16 random positions at 100 nodes/MultiPV 2, executes five optimizer steps, searches four simulations, and removes `tmp\smoke` in `finally`.

## Benchmark before choosing batch size

Do not assume BF16, channels-last, or `torch.compile` wins. Establish a short eager BF16 baseline, then vary one dimension:

```powershell
python benchmark_xpu.py --config configs\main_xpu.yaml --steps 5
python benchmark_xpu.py --config configs\main_xpu.yaml --batches 128 256 512 --steps 5 --precision fp32
python benchmark_xpu.py --config configs\main_xpu.yaml --batches 128 256 512 --steps 5 --precision bf16 --layout channels_last
python benchmark_xpu.py --config configs\main_xpu.yaml --batches 128 256 --steps 5 --precision bf16 --compile
```

OOM is reported and stops larger batches gracefully. Output includes forward/training positions per second, step latency, dtype, layout, eager/compile mode, parameter count, and approximate allocated XPU memory when exposed by PyTorch.

## Teacher datasets

Formal broad randomized generation (documented only—do not launch until ready):

```powershell
python generate_teacher_data.py `
    --config configs\main_xpu.yaml `
    --stockfish "C:\Tools\Stockfish\stockfish-windows-x86-64-avx2.exe" `
    --output data\teacher_main `
    --positions 1000000 `
    --nodes 10000 `
    --multipv 8
```

PGN-driven, correlation-thinned sampling (every eighth eligible ply internally):

```powershell
python generate_teacher_data.py `
    --config configs\main_xpu.yaml `
    --stockfish "C:\Tools\Stockfish\stockfish-windows-x86-64-avx2.exe" `
    --output data\teacher_pgn `
    --positions 1000000 `
    --nodes 10000 `
    --multipv 8 `
    --pgn datasets\games.pgn
```

For rigorous validation, divide PGNs into train/validation files by game before generation. FEN sources use `--fen-file positions.fen`. MultiPV expected WDL quality `P(win)-P(loss)` is converted to a soft target with configured temperature (initially 0.15); mate scores are mapped explicitly through a large mate score when retaining centipawns.

## Formal supervised training

First benchmark, edit `training.batch_size` in a copied config if necessary, then:

For the one-million-position formal run, place game-level-disjoint PGNs at `datasets\formal_train.pgn` and `datasets\formal_validation.pgn`, then run `scripts\generate_formal_1m.ps1`. It refuses existing output directories, produces 245 training shards plus 13 validation shards using `configs\formal_1m.yaml`, and verifies every checksum plus the exact record totals. `TeacherDataset` retains bit-packed boards and sparse targets in memory and decodes only requested samples, so the million-position dataset does not expand into tens of gigabytes of resident float32 boards.

```powershell
python train.py `
    --config configs\main_xpu.yaml `
    --dataset data\teacher_main `
    --device xpu `
    --output checkpoints\main_latest.pt `
    --logdir runs\main
```

Loss is soft-target policy cross entropy plus WDL cross entropy and optional Huber moves-left loss, with configurable weights. AdamW uses warmup plus cosine decay and `1e-4` default weight decay. BF16 XPU autocast is the default main preset.

Monitor:

```powershell
tensorboard --logdir runs
```

Resume model, optimizer, scheduler, step/epoch, CPU RNG states, config, architecture metadata, and recorded Git commit:

```powershell
python train.py `
    --config configs\main_xpu.yaml `
    --dataset data\teacher_main `
    --device xpu `
    --resume checkpoints\main_latest.pt `
    --output checkpoints\main_latest.pt `
    --logdir runs\main
```

## Validation and engine matches

Neural validation metrics include policy cross entropy/KL, teacher top-1/top-3 agreement, WDL cross entropy/accuracy, and WDL Brier calibration:

```powershell
python evaluate.py `
    --checkpoint checkpoints\main_latest.pt `
    --dataset data\teacher_validation `
    --device xpu `
    --batch-size 512
```

Paired-color model versus Stockfish (formal evaluation; not a smoke command):

```powershell
python engine_match.py `
    --checkpoint checkpoints\main_latest.pt `
    --stockfish "C:\Tools\Stockfish\stockfish-windows-x86-64-avx2.exe" `
    --device xpu `
    --games 100 `
    --simulations 800 `
    --leaf-batch-size 64 `
    --stockfish-nodes 10000 `
    --pgn artifacts\main-vs-sf.pgn
```

For a paired new/previous checkpoint match, replace `--stockfish ... --stockfish-nodes ...` with `--opponent-checkpoint checkpoints\previous.pt`. Reports include W/D/L, score, logistic Elo estimate, average move time, simulations, and PGN. Tiny matches do not justify an Elo claim.

## Hard-example mining and deeper relabeling

Rank disagreement, policy KL, WDL error, entropy, and uncertainty; this writes both audit JSON and a matching FEN list:

```powershell
python mine_hard_positions.py `
    --checkpoint checkpoints\main_latest.pt `
    --dataset data\teacher_candidates `
    --output hard_positions\ranked.json `
    --device xpu `
    --top-k 100000
```

Relabel those exact positions more deeply (future formal job):

```powershell
python generate_teacher_data.py `
    --config configs\main_xpu.yaml `
    --stockfish "C:\Tools\Stockfish\stockfish-windows-x86-64-avx2.exe" `
    --output data\teacher_hard_deep `
    --positions 100000 `
    --nodes 100000 `
    --multipv 12 `
    --fen-file hard_positions\ranked.fen
```

Continue distillation by placing compatible broad and hard shards under one dataset directory (or constructing a desired sampled mixture directory) and resume training. Sampling ratios are data/config decisions, not hard-coded.

## Selective self-play

Self-play stores MCTS visit targets and final game WDL. Run only after supervised policy quality is useful:

```powershell
python selfplay.py `
    --checkpoint checkpoints\main_latest.pt `
    --output selfplay_data\main `
    --device xpu `
    --games 10000 `
    --simulations 800 `
    --leaf-batch-size 64 `
    --temperature 1.0
```

Train on a deliberate teacher/self-play mixture by collecting selected shards into a dataset directory. A possible 70/30 ratio is an experiment, not a constant in the code.

## Human play and UCI

Play as White in the console:

```powershell
python play.py `
    --checkpoint checkpoints\main_latest.pt `
    --device xpu `
    --simulations 800 `
    --leaf-batch-size 64
```

Run the Stockfish-free UCI process for a GUI:

```powershell
python -m chess_ai.uci.engine `
    --checkpoint checkpoints\main_latest.pt `
    --device xpu `
    --simulations 800 `
    --leaf-batch-size 64
```

Serve a local browser game using an explicit checkpoint and TCP port:

```powershell
python serve.py `
    --checkpoint checkpoints\formal_tiny_latest.pt `
    --device xpu `
    --host 127.0.0.1 `
    --port 8765 `
    --simulations 400 `
    --leaf-batch-size 64
```

Open `http://127.0.0.1:8765` and choose a color. The same server exposes JSON endpoints `GET /api/state`, `POST /api/new` with `{"human_color":"white|black"}`, and `POST /api/move` with `{"move":"e2e4"}`. It binds to localhost by default; do not expose it to an untrusted network because it has no authentication.

Implemented commands are `uci`, `isready`, `ucinewgame`, `position startpos|fen ... [moves ...]`, `go`, `stop`, and `quit`. Search info includes nodes, time, and principal variation.

## Profiling

Teacher generation prints labeling throughput. Trainer step records CPU→XPU transfer, forward, backward, and optimizer time. `NeuralEvaluator.timings` records batched leaf inference calls/positions/time; `SearchResult.elapsed_s` covers tree plus inference. The XPU benchmark separates forward and full training throughput. These lightweight timings identify whether labeling, shard loading/unpacking, transfers, model kernels, or tree logic is the next bottleneck.

## Git workflow and artifacts

Use small reviewed commits and preserve reproducibility:

```powershell
git status
git add chess_ai tests configs README.md
git commit -m "feat: describe the verified change"
git push origin main
```

Datasets, PGNs, Stockfish binaries, checkpoints, runs, logs, artifacts, executables, credentials, and temporary smoke files are ignored. Inspect `git status` and tracked file sizes before every push.

## Troubleshooting

- `XPU was explicitly requested...`: confirm `torch.__version__` ends in `+xpu`; reinstall torch from the XPU index, then run diagnostics. Never mask this with CPU fallback.
- Conda channel Terms of Service: accept the prompted official channel terms, then rerun `scripts\setup_windows.ps1`.
- XPU OOM: benchmark lower batches first; compare BF16 and layouts; do not assume compile helps.
- BF16 kernel error: reproduce with `--precision fp32` in the benchmark. Keep formal training stopped until the failing XPU path is understood.
- Stockfish not found: pass a quoted absolute `.exe` path; do not hard-code it into source or commit it.
- Slow first command: native XPU/oneAPI initialization in a fresh Windows Python process can dominate tiny jobs.
- UCI GUI appears idle: ensure the GUI launches the Python module with the same Conda environment and uses an existing compatible checkpoint.
- Shard integrity/version error: do not bypass it; regenerate or migrate the shard explicitly.

## License

Project code is MIT-licensed. Stockfish is a separate GPLv3 program and is not distributed by this repository.
