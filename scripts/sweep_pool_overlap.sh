#!/bin/bash
# Sweep the overlap of the rdkit pools to measure routing gain against effective
# action-space dimension.
#
# Why: the conclusion rested on two points (rdkit 2.20 and generative 1.80) that
# also differ in editor family, yield and validity, so "gain tracks dimension" was
# an argument rather than an identification. Here only the overlap changes, within
# one editor family, which turns it into a curve.
#
# Each variant runs 3 training seeds: measured retraining noise is 0.00355, larger
# than every effect in the sweep, so a single seed per point would only give noise.
#
# alpha=0 is not re-run; results_base0 / results_seed1 / results_seed2 already
# cover it. q007 (1.70) and q003 (1.97) place same-family points either side of
# the generative branch's 1.80, so if the rdkit family also loses the bandit's gain
# near 1.8 dimensions, the claim is no longer a cross-family argument.
#
# The protocol matches the rdkit arm exactly: 200/100 seeds, 20 steps, repeats 3,
# train-repeats 3. One round takes about 53 min and 18 rounds about 16 h. Run
# sequentially, since the evaluation stage already saturates 8 cores.
set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p results_sweep

# Variants are chosen by measured dimension, not by uniform alpha: alpha affects
# dimension very non-linearly. Blending in a single generic deletion
# ([c:1][C;!R;H3]>>[c:1], aromatic demethylation) drops 2.20 to 1.57, because the
# gsk3b and jnk3 pools are all additions and one generic deletion immediately gives
# them sa-like behaviour. Uniform alpha would therefore crowd points into 1.4-1.6
# and leave 1.6-2.2 empty, which is exactly where the generative branch's 1.80
# sits. The low end uses alpha; the 1.7-2.5 range descends from qed_disjoint's 2.50
# instead.
for v in "a100 --pool-overlap 1.0" \
         "a050 --pool-overlap 0.5" \
         "a025 --pool-overlap 0.25" \
         "q007 --pool-overlap 0.07 --qed-disjoint" \
         "q003 --pool-overlap 0.03 --qed-disjoint" \
         "qdis --qed-disjoint"; do
  set -- $v
  tag=$1; shift
  for s in 0 1 2; do
    out="results_sweep/${tag}_s${s}"
    if [ -f "$out/summary.json" ]; then
      echo "=== skip $tag seed=$s (already done)"
      continue
    fi
    echo "=== $tag seed=$s  start $(date +%H:%M) ==="
    python3 -u scripts/run_experiment.py \
        --editor-kind rdkit --n-train 200 --n-test 100 \
        --repeats 3 --train-repeats 3 --max-steps 20 --workers 8 \
        --seed "$s" "$@" --out "$out" \
        > "results_sweep/${tag}_s${s}.log" 2>&1
    echo "    done $(date +%H:%M)  -> $out"
  done
done

echo "=== SWEEP DONE $(date +%H:%M) ==="
