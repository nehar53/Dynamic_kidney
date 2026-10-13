# How to run the pipeline, step by step

Run the steps in order. Each step writes files that the next step reads.

| Step | What | Command | Time |
|---|---|---|---|
| 0 | Install | `pip install -e ".[dev]"` | 1 min |
| 1 | Tests | `python -m pytest` | 15 s |
| 2 | Smoke test | `python -m dkex.cli all --config configs/quick.yaml` | 1 min |
| 3 | Calibrate | `python -m dkex.cli calibrate --config configs/paper2.yaml` | 1 min |
| 4 | Single trace | `python -m dkex.cli trace --config configs/paper2.yaml --baseline dyn_nsga3_ucb1_myopic --tau 12` | 4 min |
| 5 | Pilot (10 runs) | `python -m dkex.cli simulate --config configs/paper2.yaml --runs 10 --jobs 8` | about 1.5 h on 8 cores |
| 6 | Full run (100 runs) | `python -m dkex.cli simulate --config configs/paper2.yaml --jobs 8` | about 13 h on 8 cores |
| 7 | Analyse | `python -m dkex.cli analyse --config configs/paper2.yaml` | 1 min |
| 8 | Figures | `python -m dkex.cli plot --config configs/paper2.yaml` | 1 min |
| 9 | Sensitivity runs | see below | varies |
| 10 | Scalability | `python -m dkex.cli all --config configs/scalability.yaml --jobs 8` | hours |
| 11 | PrefLib out-of-distribution check | `python -m dkex.cli ood --config configs/paper2.yaml --wmd <file>.wmd --runs 10` | hours |

Timing measured on the full budget (N = 100, G = 150): about 10 s per clearing
round, so a 25-round NSGA-III trajectory takes 3–4 min. The full design has
1,800 NSGA-III trajectories (6 baselines × 3 τ × 100 runs), roughly 100
CPU-hours in total. The two ILP baselines are fast.

---

## Step 0: Install (on your own machine or a server)

```bash
git clone https://github.com/nehar53/Dynamic_kidney.git
cd Dynamic_kidney
git checkout claude/module-wise-pipeline-oms5i3
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

You need Python 3.10 or newer. No commercial solver is required: the ILPs use
the HiGHS solver that ships with SciPy.

## Step 1: Tests

```bash
python -m pytest
```

All 24 tests should pass. Don't continue if any test fails.

## Step 2: Smoke test (whole pipeline, tiny budget)

```bash
python -m dkex.cli all --config configs/quick.yaml
```

This checks that every stage works on your machine. Outputs go to
`results/quick/`. The numbers are **not** results; the budget is far too
small to mean anything.

## Step 3: Calibrate (do this before any real run)

```bash
python -m dkex.cli calibrate --config configs/paper2.yaml
```

This prints the simulated pool composition (blood groups, high-cPRA share,
graph density, ILP transplants), the CIT feasibility matrix, and any
**warnings**. It writes the same information to
`results/paper2/calibration.json`.

You will see three warnings with the current concept-note values. Decide how
to handle each one **in `configs/paper2.yaml`** before step 5:

1. **"CIT constraint never binds."** Add realistic door-to-door transport times
   and delays under `logistics:`. For example:
   ```yaml
   logistics:
     inter_centre_delay_hours: 6.0
   ```
   Alternatively, keep 24 h and say in the paper that CIT does not bind for
   living-donor transport within ANZ.
2. **"Batched policy may never clear."** Pick one of:
   ```yaml
   clearing:
     batch_max_interval: 3     # pi_B also clears every 3 months at the latest
   ```
   or raise `pool.arrival_rate`, or lower `clearing.batch_threshold`.
3. **Pool size vs transplant volume.** Tune `pool.arrival_rate` (λ per month)
   and the compatibility settings until the steady pool is 60–90 *and* the
   annual transplant count is close to ANZKX's 50–66. Re-run `calibrate` and
   step 4 after each change. Report the final values in the paper's Table 2.

Also fix these choices in the config and state them in the paper:

```yaml
moea:
  f3_mode: min_gap            # or max_access (the -F3 sign in Eq. 5)
  selection_rule: compromise  # which Pareto solution is implemented each round
```

**After step 5 starts, do not change the config.** If you do change it, also
change `experiment.name`, because the cached runs in `raw/` were made with the
old settings.

## Step 4: Watch one trajectory

```bash
python -m dkex.cli trace --config configs/paper2.yaml --baseline dyn_nsga3_ucb1_myopic --tau 12
python -m dkex.cli trace --config configs/paper2.yaml --baseline dyn_nsga3_ucb1_batched --tau 12
```

This prints pool size, F1 against the ILP ceiling, and PoF_t for every round.
Check that the pool stays near 60–90 patients and that batched clearing
actually fires.

## Step 5: Pilot (10 runs)

```bash
python -m dkex.cli simulate --config configs/paper2.yaml --runs 10 --jobs 8
python -m dkex.cli analyse  --config configs/paper2.yaml --runs 10
python -m dkex.cli plot     --config configs/paper2.yaml --runs 10
```

Set `--jobs` to your number of CPU cores. Look at
`results/paper2/summary.md` and `results/paper2/figures/trajectory_PoF.png`.
If something looks wrong, fix the config, change `experiment.name`, and
redo steps 3–5.

## Step 6: Full run (100 runs per configuration)

```bash
nohup python -m dkex.cli simulate --config configs/paper2.yaml --jobs 8 > paper2.log 2>&1 &
tail -f paper2.log
```

* The pilot's 10 runs are reused, so only runs 10–99 are computed.
* **Resumable.** If the machine stops, run the same command again and it
  continues where it stopped. Each trajectory is cached in
  `results/paper2/raw/`.
* On a cluster you can split the work by τ. Copy `paper2.yaml` three times
  with `taus: [6]`, `[12]`, `[18]`, keeping the **same `name` and `seed`**, and
  run one copy per node on a shared disk.

## Step 7: Analyse

```bash
python -m dkex.cli analyse --config configs/paper2.yaml
```

This writes these files to `results/paper2/`:

| File | Use in the paper |
|---|---|
| `summary.md` | quick overview of all results |
| `trajectory_metrics.csv` | main results table: mean PoF, PoF variance, transplants, high-cPRA transplants, F2–F4, HV, high-cPRA accumulation (mean and SD, per baseline and τ) |
| `wilcoxon.csv` | significance table: Wilcoxon signed-rank tests, Bonferroni-adjusted p-values, Cohen's d |
| `clearing_comparison.csv` | **central research question**: PoF_t under myopic vs batched clearing, per round |
| `trajectories.csv` | per-round means with 95 % CIs (data behind the figures) |
| `arm_distribution.csv` | UCB1 vs LinUCB arm usage per round |
| `rounds.csv`, `runs.csv` | raw data for any extra analysis |

The pre-defined comparisons in `wilcoxon.csv` are: myopic vs batched (for each
AOS method), UCB1 vs LinUCB (for each clearing policy), and static vs dynamic.

## Step 8: Figures

```bash
python -m dkex.cli plot --config configs/paper2.yaml
```

This writes these figures to `results/paper2/figures/`:

* `trajectory_PoF.png`: **main figure**, the PoF trajectory with one panel per τ
* `trajectory_HV.png`, `trajectory_F2.png`, `trajectory_F3.png`, `trajectory_F4.png`
* `trajectory_n_high_cpra.png`: high-cPRA patients building up in the pool
* `arms_<baseline>.png`: which mutation rate the AOS chose in each round

## Step 9: Sensitivity analyses (suggested for reviewers)

For each analysis, copy `configs/paper2.yaml` to a new file, change **only**
`experiment.name` and the one parameter being tested, then run
`simulate`, `analyse` and `plot` on it. Fewer runs, for example `--runs 30`,
are usually enough here.

| Sensitivity | Change |
|---|---|
| τ | already in the main run (6, 12, 18) |
| CIT binding | `configs/cit_binding.yaml` (`logistics.inter_centre_delay_hours: 6`) |
| Competing exit rate μ_d | `dynamics.deceased_exit_rate: 0.015` and `0.06` |
| Implemented-solution rule | `moea.selection_rule: max_f1` / `min_f3` |
| F3 definition | `moea.f3_mode: max_access` |
| Batch threshold | `clearing.batch_threshold: 60` / `120` |

## Step 10: Scalability (n = 200, 500, 1000)

```bash
python -m dkex.cli all --config configs/scalability.yaml --jobs 8
```

Then copy the file twice more with `initial_pool_size: 500` /
`arrival_rate: 50` and `1000` / `100`, using a new `name` each time. At
n ≥ 500 you may need to raise `structures.max_cycles`. Report the
`solve_seconds` column of `rounds.csv` as runtime.

## Step 11: PrefLib out-of-distribution validation

1. Download PrefLib dataset 00036 from https://preflib.org/dataset/00036
   (`.wmd` files).
2. Run:
   ```bash
   python -m dkex.cli ood --config configs/paper2.yaml --wmd path/to/00036-00000001.wmd --runs 10
   ```
3. Results go to `results/paper2/ood/`. PrefLib files do not include blood
   group, cPRA or centre. The paper must state that cPRA is approximated from
   in-degree and centres are assigned at random (see `dkex/data/preflib.py`).

## Reproducibility checklist for the paper

* Report `experiment.seed` (12345) and the git commit hash
  (`git rev-parse HEAD`).
* Archive `configs/paper2.yaml` and `results/paper2/calibration.json`, for
  example on Zenodo.
* State the eight choices listed under "Implementation choices the concept
  note leaves open" in `README.md`.
* All baselines in the same run share random numbers, so the tests are
  **paired** Wilcoxon signed-rank tests.
