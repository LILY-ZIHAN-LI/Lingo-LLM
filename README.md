# LingoLLM: Learning to Route Molecular Editing Experts for Multi-Objective Optimization

The experts that propose molecular edits are held fixed. The question is whether a
learned router picks which expert to call next better than a heuristic one.

## Contents

```
src/        environment, editor pools, SMILES generator, routing policies and features
scripts/    training, counterfactual collection, evaluation and diagnostics
data/       scaffold-split seeds, generator corpus, within-scaffold analog pairs
oracle/     GSK3B and JNK3 predictors, SA fingerprint scores
```

`src/` holds the shared edit-and-score environment (`env.py`), the two editor
families (`editors.py` for the RDKit reaction pool, `gen_editors.py` +
`generator.py` for the conditional SMILES transformer), the baseline policies
(`policies.py`), and the paired-evaluation helpers (`analysis.py`).
`dynamic_router.py` scores the four actions at each step using the features in
`router_features.py`. The evaluation scripts also implement StaticSelector,
which picks one expert per molecule, TwoPhase, which switches experts at a
fixed step, and a molecule-level Risk Gate that can fall back to TwoPhase.

## Setup

```bash
conda env create -f environment.yml
conda activate lingollm
python -m pip install --no-deps PyTDC==0.4.1
```

The bundled oracle files require scikit-learn 1.0.2. Install PyTDC without
dependencies so it uses the RDKit and scikit-learn versions in the environment.
Run the commands below from the repository root.

`data/chembl_v29.csv` is not committed (150 MB). The original seed splits and
generator pairs are included. `scripts/make_seeds.py` obtains the source data
through PyTDC. To prepare the separate 200-molecule test split, exclude the
scaffolds from all three existing splits.

```bash
python scripts/make_seeds.py --only test --suffix _v2 --n-test 200 \
    --raw 800000 --scan 120000 --workers 8 --seed 0 \
    --exclude data/seeds_train.txt data/seeds_val.txt data/seeds_test.txt
```

To rebuild the generator data after downloading ChEMBL, run
`scripts/prepare_generator_data.py`, `scripts/build_pairs.py` and
`scripts/rebin_delta.py` in that order. These scripts write to `data/` by default.

## Running

```bash
# RDKit editor pool used by the routing pipeline, including the LinUCB policy
python scripts/run_experiment.py --editor-kind rdkit \
    --qed-disjoint --sample-by-template --product-pick first \
    --n-train 200 --n-test 100 --repeats 3 --train-repeats 3 \
    --max-steps 20 --workers 8 --seed 0 --out results_sbt_first

# Training labels and one-step counterfactuals
python scripts/collect_expert_labels.py --qed-disjoint --sample-by-template \
    --product-pick first --out results_v2/labels/train_sbt_first.json
python scripts/collect_counterfactuals.py --pool sbt_first --n-mol 200 --noise-reps 0

# Dynamic Router (R5 / random forest) and the TwoPhase baseline
python scripts/train_dynamic_router.py --pool sbt_first --models rf --groups R5 --main-group R5
python scripts/run_baselines.py --pool sbt_first --fit-twophase --n-fit 60
python scripts/test_router_smoke.py
```

To evaluate StaticSelector separately, run the fixed experts first to produce
the validation trajectories.

```bash
python scripts/run_fixed_expert.py --results results_sbt_first \
    --qed-disjoint --sample-by-template --product-pick first
python scripts/train_expert_selector.py --pool sbt_first
```

The independent evaluation expects a frozen router checkpoint. On a fresh run,
copy the trained R5 random forest to that path before evaluating the held-out
split. The original frozen artifact is required to reproduce the reported
checkpoint hash. Use the same source data and software versions when
regenerating the test split and rerunning training.

```bash
mkdir -p results_v2/dynamic_router/models/v1
cp results_v2/dynamic_router/models/router_R5_rf.pkl \
    results_v2/dynamic_router/models/v1/DynamicRouter_v1.pkl
python scripts/run_test_v2.py --dry-run
python scripts/run_test_v2.py --seeds data/seeds_test_v2.txt --n-mol 200 --repeats 3
```

Run the rollout and training-set diagnostics after generating the routing
artifacts above. The horizon sweep and Risk Gate use the training split.

```bash
python scripts/run_rollout_comparison.py --pool sbt_first --exhaustion both
python scripts/run_horizon_snr.py --pool sbt_first --n-mol 40 --reps 5
python scripts/run_risk_gate.py --pool sbt_first --folds 5 --repeats 3
```

The generative editor pool requires a separately trained checkpoint.

```bash
python scripts/train_generator.py --corpus data/gen_pairs_delta.jsonl \
    --gen-meta data/gen_pairs_delta_meta.json --out results_gen/generator_delta.pt
python scripts/run_experiment.py --editor-kind gen \
    --gen-ckpt results_gen/generator_delta.pt --out results_gen_router
```

The commands above regenerate the routing labels, model checkpoints and
experiment outputs. The independent split, new training artifacts and results
are kept out of Git. See the header comment of each script in `scripts/` for
its inputs and outputs.

## Scope

A methodology experiment on public benchmark oracles. At a fixed budget of
20 expert calls, Dynamic Router outperformed random routing, the fixed expert
and the earlier LinUCB router. On 200 scaffold-disjoint test molecules, its
mean hypervolume gain was 0.01470, compared with 0.01426 for TwoPhase. The mean
improvement was 3.1% and the paired 95% bootstrap interval for the difference
crossed zero, so the pre-registered significance criterion was not met.

Longer-horizon gain prediction showed no clear advantage. The molecule-level
Risk Gate showed a learnable signal, with no established improvement over
TwoPhase across evaluations. OneStepOracle makes additional probe calls and
is reported as a diagnostic at a higher cost.

All results use computational scores and do not establish biological activity
or clinical value. The files in `oracle/` are third-party predictors
redistributed from Therapeutics Data Commons and RDKit.
