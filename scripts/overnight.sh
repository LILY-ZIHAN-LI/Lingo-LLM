#!/bin/bash
# 无人值守的分阶段执行器。
#
# 设计要点（对应"自主运行与故障恢复"那一节）：
#   - 每阶段独立日志 + 完成标记文件（.done）+ 输出文件检查
#   - 已完成的阶段跳过，所以中断后重跑这个脚本即可续上
#   - python 一律 -u，否则管道块缓冲会让日志一直是空的（踩过）
#   - 进程退出码为 0 不等于成功：还要检查该阶段声明的产出文件存在且非空
#   - 单阶段失败不吞异常：标记 .failed、记录原因、继续后面不依赖它的阶段
#
# 用法: bash scripts/overnight.sh            （从第一个未完成的阶段继续）
#       STAGE=rollout bash scripts/overnight.sh   （只跑某一阶段）

set -u
cd "$(dirname "$0")/.." || exit 1
PY="${LINGOLLM_PYTHON:-python}"
ST=results_v2/_stages
mkdir -p "$ST" results_v2/dynamic_router/{logs,models,figures} \
         results_v2/baselines

log() { echo "[$(date '+%F %T')] $*" | tee -a "$ST/overnight.log"; }

# wait_pid <name> —— 等一个已经在跑的 tmux session 结束
wait_session() {
  local s=$1
  while tmux has-session -t "$s" 2>/dev/null; do sleep 20; done
}

# stage <名字> <产出文件...> -- <命令...>
stage() {
  local name=$1; shift
  local outs=()
  while [ "$1" != "--" ]; do outs+=("$1"); shift; done
  shift
  if [ -f "$ST/$name.done" ]; then log "跳过 $name（已完成）"; return 0; fi
  log "开始 $name"
  local lg="$ST/$name.log"
  "$@" > "$lg" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "！$name 退出码 $rc —— 见 $lg"
    echo "exit=$rc" > "$ST/$name.failed"
    return 1
  fi
  for f in "${outs[@]}"; do
    if [ ! -s "$f" ]; then
      log "！$name 退出码 0 但产出缺失或为空: $f"
      echo "missing=$f" > "$ST/$name.failed"
      return 1
    fi
  done
  : > "$ST/$name.done"
  log "完成 $name"
}

# ---------------------------------------------------------------- 阶段

# 0) 等已经在跑的两个任务（basline 套件；它可能已经结束）
if tmux has-session -t bl 2>/dev/null; then
  log "等待已在运行的 bl（基线套件）..."
  wait_session bl
fi
if [ -s results_v2/baselines/baseline_comparison.csv ]; then
  : > "$ST/baselines.done"; log "baselines 产出已存在"
fi

# 1) 修好的可学习空间探测（fold 共用、best_fix 按 fold、Ridge 分组内层 CV）
stage probe_fixed \
  results_v2/baselines/learnable_space.json -- \
  "$PY" -u scripts/probe_learnable_space.py \
      --out results_v2/baselines/learnable_space.json

# 2) Dynamic Router 训练 + 消融（可能已经在 tmux dr 里跑着）
if tmux has-session -t dr 2>/dev/null; then
  log "等待已在运行的 dr（Router 训练）..."
  wait_session dr
fi
if [ -s results_v2/dynamic_router/model_comparison.csv ] \
   && [ -s results_v2/dynamic_router/ablation_results.csv ]; then
  : > "$ST/router_train.done"; log "router_train 产出已存在"
fi
stage router_train \
  results_v2/dynamic_router/model_comparison.csv \
  results_v2/dynamic_router/ablation_results.csv \
  results_v2/dynamic_router/models/router_meta.json -- \
  "$PY" -u scripts/train_dynamic_router.py \
      --models ridge,rf,gb,twostage --groups R1,R2,R3,R4,R5,R6 --seeds 0,1,2

# 3) 部署前的 smoke test：离线/在线特征一致性 + Router 能在 env 里走满 20 步
stage router_smoke \
  "$ST/router_smoke.log" -- \
  "$PY" -u scripts/test_router_smoke.py

# 4) 验证集完整 20 步实验（两套耗尽处理）
stage rollout \
  results_v2/dynamic_router/rollout_comparison.csv \
  results_v2/dynamic_router/budget_comparison.csv -- \
  "$PY" -u scripts/run_rollout_comparison.py --pool sbt_first \
      --seeds data/seeds_val.txt --n-mol 50 --exhaustion both

# 5) 图
stage figures \
  results_v2/dynamic_router/figures/regret_vs_cost.png -- \
  "$PY" -u scripts/plot_router.py

log "全部阶段处理完毕。状态:"
ls -1 "$ST" | sed 's/^/  /' | tee -a "$ST/overnight.log"
