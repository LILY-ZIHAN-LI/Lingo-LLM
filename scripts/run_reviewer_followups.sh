#!/usr/bin/env bash
# proposal 审稿意见里还需要跑的两件事，顺序跑（只有 8 个核，并行会互相抢）。
#   1. 审稿意见 #2：更丰富的 Pareto archive 状态表示（--rich-state）
#   2. 审稿意见 #1：生成式那条臂上的单 expert 基线
set -x
cd "$(dirname "$0")/.."

python scripts/run_experiment.py --editor-kind rdkit \
    --n-train 200 --n-test 100 --repeats 3 --train-repeats 3 \
    --max-steps 20 --workers 8 --seed 0 --rich-state \
    --out results_rich0 > results_rich0_log.txt 2>&1

python scripts/run_fixed_expert.py --results results_gen_router \
    --editor-kind gen --gen-ckpt results_gen/generator_delta.pt \
    --workers 8 --val-repeats 1 > fixed_gen_log.txt 2>&1

echo "ALL DONE"
