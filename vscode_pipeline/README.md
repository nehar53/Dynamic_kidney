# Paper 2 pipeline: modules 1–10 (VS Code version)

Modules 1–5 are your files. Modules 6–10 are new and import your functions
directly. None of modules 1–5 were changed.

| # | File | What it does | Status |
|---|------|--------------|--------|
| 1 | `module1_generator.py` | Patient/pair generator, Pool, initial pool | yours |
| 2 | `module2_pool_evolution.py` | Pool evolution (arrivals, withdrawals, competing exits) | yours |
| 3 | `module3_logistics.py` | CIT logistics + compatibility graph | yours |
| 4 | `module4_objectives.py` | F1–F4, PoF | yours |
| 5 | `module5_nsga3_aos.py` | NSGA-III + UCB1/LinUCB | yours |
| 6 | `module6_ilp_baseline.py` | ILP upper bound (CBC) + Dickerson lexicographic baseline | **new** |
| 7 | `module7_clearing.py` | π_M myopic / π_B batched + one full T-round trajectory | **new** |
| 8 | `module8_stats.py` | Wilcoxon signed-rank, Bonferroni, Cohen's d | **new** |
| 9 | `module9_preflib.py` | PrefLib 00036 out-of-distribution validation | **new** |
| 10 | `module10_experiment.py` | 8 baselines × τ × 100 runs, tables, figures | **new** |

Module 7 imports `module2_pool_evolution`. If your file is named
`module2_evolution.py`, that works too.

## Setup (once)

```bash
pip install -r requirements.txt
```

Use **pulp 2.9.0**. PuLP 3 and later no longer bundle the CBC solver. If CBC
is missing, module 6 falls back to SciPy's HiGHS solver, which returns the same
optimum.

## Run order

Run each module's self-test in order. Each one prints `MODULE n COMPLETE`.

```bash
python module1_generator.py
python module2_pool_evolution.py
python module3_logistics.py
python module4_objectives.py
python module5_nsga3_aos.py
python module6_ilp_baseline.py     # ILP F1, cycles, MOEA <= ILP check        (~5 s)
python module7_clearing.py         # myopic vs batched, paired-design check     (~1 min)
python module8_stats.py            # Wilcoxon/Bonferroni/Cohen's d on toy data  (~1 s)
python module9_preflib.py          # self-test on a synthetic .wmd file         (~1 min)
```

Then run module 10 (experiments of concept note Section 4) in stages:

```bash
# 1. Quick test of the whole pipeline (~1 min on 4 cores)
python module10_experiment.py --quick --jobs 4

# 2. Pilot: 5 runs at the full budget (N=100, G=150) -> real timing
python module10_experiment.py --pilot --jobs 4

# 3. Main design: 8 baselines x tau {6,12,18} x 100 runs x 25 rounds
python module10_experiment.py --jobs 4

# 4. Sensitivity and scalability (reuse the base runs of step 3)
python module10_experiment.py --experiment sensitivity --jobs 4
python module10_experiment.py --experiment scalability --jobs 4

# Tables and figures again, at any time (also while runs are missing)
python module10_experiment.py --analyse-only --experiment all
```

| Experiment | What varies | Runs |
|---|---|---|
| `main` | 8 baselines x tau 6/12/18 | 100 |
| `sens_select` | implemented solution: 90 % floor / 80 % floor / compromise | 30 |
| `sens_cit` | CIT limit 24 / 16 / 12 h | 30 |
| `sens_batch` | batched clearing with a 3-round maximum interval | 30 |
| `sens_mud` | mu_d 0.01 / 0.02 / 0.04, high-cPRA exits at 0.3 mu_d | 30 |
| `scalability` | initial pool 200 (500, 1000 with `--allow-large`) | 10 |

**Runtime.** After the module 5 speed-up a 25-round myopic NSGA-III
trajectory at the full budget takes about 2 min on one core (batched about
0.5 min), so the main design (1,800 MOEA trajectories) needs roughly 50
CPU-hours. Rounds with at most 200 feasible matchings are solved exactly
(column `exact_front`); the AOS is not invoked in those rounds. Use
`--jobs` (about the number of cores), the `--pilot` timing, and if possible
several computers with `--shard 1/3`, `--shard 2/3`, `--shard 3/3` and the
same `--out` folder copied together afterwards.

**Resumable.** Each trajectory is saved to `results/raw/` as soon as it
finishes. Ctrl+C at any time and re-run the same command.
`results/manifest.json` records the budget and package versions; one
folder holds one budget.

PrefLib validation: see module 9 (`python module9_preflib.py --preset test`).

## Outputs (`results/`)

| File | Use in the paper |
|---|---|
| `table_main_results.csv` / `.tex` | Main table: mean ± sd of PoF, Var PoF, transplants, high-cPRA transplants, F2–F4, HV, high-cPRA patients lost |
| `table_friedman.csv` | Friedman omnibus test and mean ranks per metric and tau |
| `table_wilcoxon.csv` | Paired Wilcoxon, Bonferroni and Holm p, Cohen's d, rank-biserial, bootstrap CI |
| `table_pof_trend.csv` | Slope of PoF_t over rounds per run, tested against 0 |
| `table_pof_trajectory.csv` | PoF_t per round, myopic vs batched, UCB1 vs LinUCB |
| `table_arm_distribution.csv` | Share of generations per mutation rate (AOS) |
| `table_sens_*.csv`, `table_scalability.csv` | Sensitivity and scalability |
| `figures/trajectory_*.png` | PoF_t, F2, F3, F4, HV, pool, high-cPRA, cumulative transplants (mean ± 95 % CI) |
| `figures/clearing_myopic_vs_batched.png` | Central question: pi_M vs pi_B |
| `figures/pof_mean_boxplot.png`, `aos_arm_selection.png`, `sens_*.png` | Supporting figures |
| `summary.md`, `rounds.csv`, `runs.csv` | Summary and all raw data |

`pof_vs_myopic_ilp` in `runs.csv` = 1 − transplants / myopic-ILP transplants
on the same pool sequence: a trajectory-level loss that, unlike PoF_t, is
comparable between myopic and batched clearing.

---

## Issues in modules 1–5 you should know about

I did not edit your files. Where possible, modules 6–10 work around the
problem. Severity reflects how much each issue affects the paper's claims.

### Must fix before the full run

**1. F2 is the same for every solution (module 4 / module 5).**
`compute_f2_gini_fast(pool_pairs)` uses the waiting times of the *whole pool*
and never looks at which pairs are matched. So F2 has one value for all 100
solutions in a round; I checked this, and `np.unique(result.F[:,1])` has
size 1. NSGA-III is effectively optimising three objectives. Fix: compute the
Gini over the pairs **left waiting** after the matching:

```python
def compute_f2_gini_fast(pool_pairs, matched_pair_ids=frozenset()):
    waiting = [p for p in pool_pairs
               if p.pair_id not in matched_pair_ids and not p.is_altruistic]
    ...same code on `waiting`...
```

and pass `matched_ids` in module 5's `compute_all_objectives`.

**2. cPRA has no effect on matching (modules 1 / 3).** The graph checks ABO,
CIT and HLA, but there is no crossmatch, so a patient with cPRA 0.95 is as easy
to match as one with cPRA 0.05. "Highly sensitised" then only labels a random
30 % of patients, and F3 and the high-cPRA results mean nothing clinically.
Fix: in `compute_edge_weight`, add a virtual crossmatch that fails with
probability cPRA_j. Make the draw fixed per (donor, patient) couple so it
doesn't change between rounds:

```python
u = np.random.default_rng([patient_i.pair_id, patient_j.pair_id]).random()
if u < patient_j.cpra:
    return 0.0          # positive crossmatch
```

**3. AOS picks one mutation rate per round, not per generation (module 5).**
`run_nsga3_with_aos` asks the bandit once, then runs all 150 generations with
that rate. Over a 25-round run the bandit gets only 25 pulls. The reward is the
HV difference between *two different pools*, which reflects how the pool
changed, not how good the operator was. Concept note §3.5.1 says arm *a* is
chosen "at generation g". Fix: in pymoo, use a `Callback` or run
`algorithm.next()` in a loop, choose the rate each generation, and reward the
HV gain of that generation. As it stands, you can't claim UCB1 vs LinUCB as
AOS results. The current code compares round-level parameter control.

**4. No altruistic arrivals after round 0 (module 1 `generate_arrivals`).**
`int(n_arrivals * 0.08)` is 0 for any n < 13, and arrivals are about 6 per
round, so altruists exist only in the initial pool. Fix: draw each arrival as
altruistic with probability 0.08.

**5. Altruists are treated as patients (modules 1 / 3 / 4).** Altruistic
donors get a patient blood type, cPRA and HLA. In module 3 they can *receive*
edges and sit inside cycles. They count in |S|, |V_t| and the F3/F4 group
sizes. Fix: skip `j` when `pool_pairs[j].is_altruistic` in
`build_compatibility_graph`, and exclude altruists from F2–F4. Chains (L = 4
in Table 2) are not modelled. Modules 6 and 5 both use cycles only, so they
stay consistent, but the paper should say chains are excluded, or chains
should be added.

### Affects interpretation (handled or flagged in modules 6–10)

**A. The empty matching is always Pareto-optimal.** F3 and F4 are absolute
gaps, so matching nobody gives F3 = F4 = 0. With a plain "compromise" pick,
the empty solution was chosen in most rounds, giving PoF = 100 %. Module 7
never implements an empty solution if a non-empty one exists. The default is
`--selection floor`: the fairest solution among those reaching ≥ 90 % of the
front's best F1. Your `extract_best_solution` (max F1) is available as
`--selection max_f1`. With that rule, PoF only measures how far the MOEA is
from the ILP optimum, not a fairness trade-off. **Say which rule you use in
the paper. It largely determines the PoF values.**

**B. CIT never binds.** With 24 h CIT and 6 h preparation, a match is ruled
out only if transport takes more than 18 h, and your longest route is
Perth–Auckland at 8.5 h. Module 3's own test prints "All routes feasible", so
F4 can't come from logistics. Either use door-to-door times plus scheduling
delays, or say in the paper that CIT does not restrict living-donor transport
within ANZ.

**C. Batched clearing barely fires.** λ = 6 per round means the pool can
reach n* = 90 only when τ is large. At τ = 6 it is about 36 pairs at most, so
π_B never clears. Module 10 prints a warning. Use `--batch-max-interval 3`, or
a higher λ, or a lower n*.

**D. Withdrawal cliff.** All initial pairs start with waiting time 0, so 58 of
75 withdrew in the same round (round τ). Module 7 spreads initial waiting
times over 0–5 by default (`--initial-wait-max 6`).

**E. Paired design.** Module 2 draws exits and arrivals from one shared RNG,
and pymoo reseeds the global NumPy RNG. So different baselines would see
different arrivals, and the paired Wilcoxon test would be invalid. Module 7's
`PairedPoolEvolution` ties every draw to (seed, round) or (seed, pair, round).
Module 7's self-test checks this ("Paired design ... True").

**F. PoF when no transplant is possible.** When F1_ILP = 0, PoF is undefined.
Module 7 stores NaN so those rounds don't pull mean PoF towards 0.

**G. Settings that differ from the concept note (module 5).** These are UCB1
window W = 10 (note says 15), Das–Dennis p = 5 (note says 12), and an extra
UCB1 constant c. In `LinUCBAOS.build_context_vector`, feature 2 is mean waiting
time, but the note's s_t uses the high-cPRA share p_PRA. Features are divided
by 10, so round t and competing exits saturate at 1 after round 10.
`pof_prev` is clipped to [0, 1], so module 7 passes PoF/100. Match the code or
the note.
