#!/bin/bash
# Re-run every analysis and figure from an existing results/raw.json and
# results/policies.pkl. Does not re-run the experiment itself, which takes tens of
# minutes (see run_experiment.py).
#
#   bash scripts/run_all_analyses.sh [results_dir]
#
# Requires micromamba activate lingollm first.
set -euo pipefail
cd "$(dirname "$0")/.."
RES=${1:-results}

echo "=== 1. DQN diagnostic ==="
python scripts/diagnose_dqn.py --policies "$RES/policies.pkl" \
    --out "$RES/dqn_diagnosis.json"

echo "=== 2. Budget-definition comparison ==="
python scripts/compare_budgets.py --raw "$RES/raw.json" \
    --out "$RES/budget_comparison.json"

echo "=== 3. Absorbing state / dead steps ==="
python scripts/analyze_editor_depletion.py --raw "$RES/raw.json" \
    --out "$RES/editor_depletion.json"

echo "=== 4. Objective correlation and PCA ==="
python scripts/analyze_objectives.py --out "$RES/objective_analysis.json" \
    --npz "$RES/objective_scores.npz"

echo "=== 5. Action-space degeneracy ==="
python scripts/analyze_action_space.py --out "$RES/action_space.json"

echo "=== 6. Failure trajectories and structure figures ==="
python scripts/analyze_failures.py --raw "$RES/raw.json" \
    --policies "$RES/policies.pkl" --out "$RES/failure_cases.json"

echo "=== 7. Retry rule (cost of the absorbing state) ==="
python scripts/eval_masking.py --policies "$RES/policies.pkl" \
    --out "$RES/masking.json" --raw-out "$RES/masking_raw.json"

echo "=== 8. Training-seed variance and escape-bit A/B ==="
# Needs results directories for several seeds; skip rather than fail the script
if [ -f results_seed1/raw.json ] && [ -f results_seed2/raw.json ]; then
    python scripts/compare_seeds.py \
        --runs results_base0 results_seed1 results_seed2 \
        --escape results_escape0:results_base0 results_escape1:results_seed1 \
        --out "$RES/seed_variance.json"
else
    echo "skipped (no multi-seed runs; run run_experiment.py --seed 1/2 first)"
fi

echo "=== 9. Figures ==="
python scripts/plot_results.py --raw "$RES/raw.json" --out figures
python scripts/plot_diagnostics.py --results "$RES" --out figures

echo
echo "Done."
