# 正式训练操作指令

本文面向 Linux bash（原生 Linux 或 WSL2）。所有命令均为单行，可逐条复制执行；除特别说明外，均在项目根目录运行。路径示例按本机 WSL2 环境书写，其他机器请替换为实际路径。

## 0. 进入项目并检查 CUDA

```bash
cd /mnt/c/Users/lijs/Desktop/work/chess
```

```bash
conda run --no-capture-output -n chessmodel python -m chess_ai.diagnostics --device cuda
```

诊断必须成功完成真实 CUDA 前向和反向计算，不能用 CPU fallback 继续正式训练。BF16 需要 NVIDIA Ampere（RTX 30 系）及以上显卡，老卡请改用 fp16/fp32。

## 1. 首次准备 PGN（当前机器已经完成，不要重复执行）

解压已下载的 Lichess Elite PGN：

```bash
unzip -o datasets/source/lichess_elite_2023-12.zip -d datasets/source
```

过滤、去重并按整局棋确定性地切分训练集和验证集：

```bash
conda run --no-capture-output -n chessmodel python scripts/prepare_formal_pgn.py --input datasets/source/lichess_elite_2023-12.pgn --train-output datasets/formal_train.pgn --validation-output datasets/formal_validation.pgn --validation-percent 10 --seed 7 --required-train-positions 1000000 --required-validation-positions 50000 --metadata-output datasets/formal_pgn_metadata.json
```

准备脚本拒绝覆盖已有输出。PGN 只负责提供高质量真实局面和历史；监督 policy/WDL 标签由 Stockfish 18 重新分析产生，而不是直接使用 PGN 的胜负结果。

## 2. 生产 1M 训练数据（当前任务正在运行，不要再次启动）

仅在 `data/formal_1m_train` 和 `data/formal_50k_validation` 均不存在时执行：

```bash
./scripts/generate_formal_1m.sh
```

默认使用 `tools/stockfish/unpacked/stockfish/stockfish-ubuntu-x86-64-avx2`（需自行下载 Linux 版 Stockfish 放到该位置，此目录不入库）；Stockfish 在其他位置时用 `STOCKFISH=路径 ./scripts/generate_formal_1m.sh` 覆盖。

默认生产 1,000,000 个训练位置和 50,000 个验证位置，参数为 10,000 nodes、MultiPV 8、每 shard 4,096 条。脚本先完成全部训练数据，再生产验证数据，最后自动校验 checksum、位置总数和 shard 总数；数据生产与神经网络训练不是同步流水线。

只读查看当前进度：

```bash
echo "train=$(ls data/formal_1m_train/*.npz 2>/dev/null | wc -l)/245 validation=$(ls data/formal_50k_validation/*.npz 2>/dev/null | wc -l)/13"
```

## 3. 生产完成后独立校验

训练集必须恰好通过 1,000,000 positions / 245 shards 校验：

```bash
conda run --no-capture-output -n chessmodel python scripts/verify_teacher_dataset.py --dataset data/formal_1m_train --expected-positions 1000000 --expected-shards 245
```

验证集必须恰好通过 50,000 positions / 13 shards 校验：

```bash
conda run --no-capture-output -n chessmodel python scripts/verify_teacher_dataset.py --dataset data/formal_50k_validation --expected-positions 50000 --expected-shards 13
```

任一命令失败都不要启动训练；先保留现场并调查，不要手工跳过 checksum 或修改计数。

## 4. 正式训练前做 CUDA batch 基准

```bash
conda run --no-capture-output -n chessmodel python benchmark_cuda.py --config configs/formal_1m.yaml --batches 256 512 --steps 5 --precision bf16
```

正式配置默认 batch size 为 512。若 512 OOM 或明显慢于 256，应复制配置文件并只修改经过测试的 batch size，不要直接改变数据或模型结构。

## 5. 开始正式训练

```bash
conda run --no-capture-output -n chessmodel python train.py --config configs/formal_1m.yaml --dataset data/formal_1m_train --device cuda --output checkpoints/formal_1m_latest.pt --logdir runs/formal_1m
```

默认模型为 12 blocks × 192 channels、BF16、batch 512、20 epochs。checkpoint 在每个 epoch 结束时保存，包含模型、优化器、scheduler、epoch/step、随机状态、配置和 Git commit。本分支与 main 分支共用同一份 checkpoints 本地目录，checkpoint 格式完全兼容，可在 CUDA 机器上直接加载/继续训练。

在另一个终端窗口启动 TensorBoard：

```bash
conda run --no-capture-output -n chessmodel tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

在浏览器查看训练曲线：

```bash
xdg-open http://127.0.0.1:6006
```

## 6. 中断后恢复训练

只有 `checkpoints/formal_1m_latest.pt` 存在且能正常读取时才执行：

```bash
conda run --no-capture-output -n chessmodel python train.py --config configs/formal_1m.yaml --dataset data/formal_1m_train --device cuda --resume checkpoints/formal_1m_latest.pt --output checkpoints/formal_1m_latest.pt --logdir runs/formal_1m
```

不要同时运行原训练命令和恢复命令，它们会写同一个 checkpoint 与 TensorBoard 目录。

## 7. 在独立验证集上评估

```bash
conda run --no-capture-output -n chessmodel python evaluate.py --checkpoint checkpoints/formal_1m_latest.pt --dataset data/formal_50k_validation --device cuda --batch-size 512
```

保存输出中的 policy cross entropy/KL、top-1/top-3 agreement、WDL cross entropy/accuracy 和 Brier calibration，作为后续 checkpoint 的比较基线。

## 8. 与 Stockfish 做成对颜色对局

先用 20 局做功能和速度检查：

```bash
conda run --no-capture-output -n chessmodel python engine_match.py --checkpoint checkpoints/formal_1m_latest.pt --stockfish tools/stockfish/unpacked/stockfish/stockfish-ubuntu-x86-64-avx2 --device cuda --games 20 --simulations 400 --leaf-batch-size 64 --stockfish-nodes 10000 --pgn artifacts/formal_1m-vs-stockfish-smoke.pgn
```

检查无异常后运行 100 局正式评测：

```bash
conda run --no-capture-output -n chessmodel python engine_match.py --checkpoint checkpoints/formal_1m_latest.pt --stockfish tools/stockfish/unpacked/stockfish/stockfish-ubuntu-x86-64-avx2 --device cuda --games 100 --simulations 800 --leaf-batch-size 64 --stockfish-nodes 10000 --pgn artifacts/formal_1m-vs-stockfish.pgn
```

比赛应使用偶数局以交换执棋颜色。PGN 和汇总结果要保留用于审计；少量对局只能作为回归检查，不能据此宣称稳定 Elo。

## 9. 使用指定 checkpoint 与人类对局

启动仅绑定本机的网页服务：

```bash
conda run --no-capture-output -n chessmodel python serve.py --checkpoint checkpoints/formal_1m_latest.pt --device cuda --host 127.0.0.1 --port 8765 --simulations 800 --leaf-batch-size 64
```

在另一个终端窗口打开网页：

```bash
xdg-open http://127.0.0.1:8765
```

也可以直接使用控制台对局（人类固定执白）：

```bash
conda run --no-capture-output -n chessmodel python play.py --checkpoint checkpoints/formal_1m_latest.pt --device cuda --simulations 800 --leaf-batch-size 64
```

## 10. 可选的后续迭代

从候选数据中挖掘模型最不确定或与教师分歧最大的局面：

```bash
conda run --no-capture-output -n chessmodel python mine_hard_positions.py --checkpoint checkpoints/formal_1m_latest.pt --dataset data/formal_1m_train --output hard_positions/formal_1m_ranked.json --device cuda --top-k 10000 --batch-size 512
```

仅在监督模型已经具备可用棋力并完成基准评测后，再考虑自我对弈：

```bash
conda run --no-capture-output -n chessmodel python selfplay.py --checkpoint checkpoints/formal_1m_latest.pt --output selfplay_data/formal_1m --device cuda --games 1000 --simulations 800 --leaf-batch-size 64 --temperature 1.0
```

数据集、PGN、checkpoint、日志、比赛 PGN 和自我对弈产物均为本地运行产物，不应提交到 Git。
