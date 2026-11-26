#!/usr/bin/env bash
# §16c 的预注册检验：在 3.01 有效维度的池子上跑完整路由实验 + 单 expert 基线。
# 池子 = 手写池 + qed_disjoint + 按模板抽样 + 每条模板取排序首个产物。
# 之后把 2.68 / 2.42 两个点也补上，把 §12 的曲线从 2.50 延伸到 3.01。
set -x
cd "$(dirname "$0")/.."

for PICK in first last random; do
  python scripts/run_experiment.py --editor-kind rdkit \
      --n-train 200 --n-test 100 --repeats 3 --train-repeats 3 \
      --max-steps 20 --workers 8 --seed 0 \
      --qed-disjoint --sample-by-template --product-pick $PICK \
      --out results_sbt_$PICK > results_sbt_${PICK}_log.txt 2>&1

  python scripts/run_fixed_expert.py --results results_sbt_$PICK \
      --qed-disjoint --sample-by-template --product-pick $PICK \
      --val-repeats 2 --workers 8 > fixed_sbt_${PICK}_log.txt 2>&1
done

python scripts/run_fixed_expert.py --results results_gen_router \
    --editor-kind gen --gen-ckpt results_gen/generator_delta.pt \
    --workers 8 --val-repeats 1 > fixed_gen_log.txt 2>&1

echo "ALL DONE"
