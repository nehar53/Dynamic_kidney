# Dynamic Kidney Exchange — module-wise pipeline

Code for the Paper 2 concept note *"Dynamic Kidney Exchange Optimisation Under
Realistic Clinical Constraints for Equitable Allocation"*. It is a synthetic
simulator calibrated to ANZKX that measures the **Price of Fairness trajectory
{PoF_t}** across sequential clearing rounds. It compares myopic and batched
clearing, and stateless UCB1 against contextual LinUCB adaptive operator
selection inside NSGA-III.

> Synthetic, ANZKX-*calibrated* simulation only. It is not validated against
> real ANZKX patient-level data. The HLA score and F4 are proxies (concept note
> §3.2, §3.4).

## Pipeline at a glance

```
            ┌────────────── per clearing round t = 0 … T-1 ───────────────┐
 config ──► │ M4 dynamics  V_t ─► M9 clearing policy: clear now?           │
 (Table 2)  │                         │ yes                                │
            │ M2+M3 graph G_t (ABO, crossmatch, HLA, CIT) ─► M5 cycles/chains│
            │      ├─► M7 ILP ceiling F1^ILP(t)                            │
            │      └─► M8 NSGA-III + AOS(UCB1/LinUCB) ─► Pareto front ─► M_t │
            │ M6 objectives F1–F4 ─► M10 PoF_t                             │
            │ M4: V_{t+1} = V_t \ M_t \ W_t \ X_t ∪ A_t                     │
            └──────────────────────────────────────────────────────────────┘
                 ─► M11 runner (8 baselines × τ × runs) ─► M12 metrics / Wilcoxon / plots
                 ─► M13 PrefLib 00036 out-of-distribution run
```

| # | Module | File | Concept note |
|---|--------|------|--------------|
| M0 | Configuration (every Table 2 assumption) | `dkex/config.py` | Table 2, §4.2 |
| M1 | Synthetic pool generator (Saidman, ANZKX-calibrated) | `dkex/simulation/generator.py`, `pool.py` | §4.1 |
| M2 | Compatibility graph, `w_ij = 1 − m_ij/6` | `dkex/simulation/compatibility.py` | §3.2, Eq. 2–3 |
| M3 | Six-node logistics network, CIT feasibility | `dkex/simulation/logistics.py` | §3.3, §4.3, Eq. 4/14 |
| M4 | Dynamic pool: arrivals, withdrawals (τ), deceased-donor exits | `dkex/simulation/dynamics.py` | §3.1, Eq. 1 |
| M5 | Cycle (K = 3) and chain (L = 4) enumeration | `dkex/optimisation/structures.py` | Table 2 |
| M6 | Objectives F1–F4 | `dkex/optimisation/objectives.py` | §3.4, Eq. 5–9 |
| M7 | Myopic ILP and Dickerson lexicographic baselines | `dkex/optimisation/ilp.py` | §4.4 (7, 8) |
| M8 | NSGA-III / NSGA-II with feasibility-preserving operators | `dkex/optimisation/moea.py` | §4.2.1 |
| M8b | AOS: sliding-window UCB1, LinUCB, fixed | `dkex/aos/` | §3.5, Eq. 10–12 |
| M9 | Clearing policies π_M, π_B | `dkex/simulation/clearing.py` | §3.7 |
| M10 | Price of Fairness | `dkex/analysis/pof.py` | §3.6, Eq. 13 |
| M11 | Eight baselines and the trajectory simulator | `dkex/experiments.py` | §4.4 |
| M12 | Per-round/trajectory metrics, Wilcoxon + Bonferroni, Cohen's d, figures | `dkex/analysis/` | §4.5 |
| M13 | PrefLib 00036 loader and dynamic extension | `dkex/data/preflib.py` | §4.6 |
| — | Stage orchestration and CLI | `dkex/pipeline.py`, `dkex/cli.py` | — |

## Running it

```bash
pip install -e .            # or: pip install -r requirements.txt
python -m pytest            # 24 tests, ~15 s
# Step-by-step research workflow: see RUNNING.md

# Stage by stage (each stage reads the previous stage's files)
python -m dkex.cli calibrate --config configs/quick.yaml   # check pool vs Table 2, print warnings
python -m dkex.cli simulate  --config configs/quick.yaml   # resumable; --jobs N for parallel runs
python -m dkex.cli analyse   --config configs/quick.yaml   # metrics + Wilcoxon tables + summary.md
python -m dkex.cli plot      --config configs/quick.yaml   # figures/*.png
python -m dkex.cli all       --config configs/quick.yaml   # all four stages (~1 min on 4 cores)

python -m dkex.cli trace --baseline dyn_nsga3_linucb_myopic --tau 12   # one verbose trajectory
python -m dkex.cli ood   --config configs/quick.yaml --wmd 00036-00000001.wmd
```

Configs:

* `configs/quick.yaml` — smoke test with the same pipeline and a small budget.
* `configs/paper2.yaml` — the full design: N = 100, G = 150, T = 25,
  100 runs × 3 τ × 8 baselines. Plan on many CPU-hours. Use `--jobs`.
  The run can be resumed because each trajectory is cached in `raw/`.
* `configs/scalability.yaml` — n = 200 (edit it for 500 / 1000).
* `configs/cit_binding.yaml` — a scenario where CIT actually removes edges (see below).

Outputs go to `results/<name>/`: `calibration.json`, `rounds.csv` (one row per
round: pool state, F1–F4, F1^ILP, PoF_t, HV_t, AOS arm counts, exits),
`runs.csv` (trajectory metrics), `trajectories.csv`, `wilcoxon.csv`,
`clearing_comparison.csv`, `arm_distribution.csv`, `summary.md`, and `figures/`.

## The eight baselines (§4.4)

| Name | Solver | AOS | Clearing | State across rounds |
|------|--------|-----|----------|--------------------|
| `static_nsga2_ucb1` | NSGA-II | UCB1 | myopic | reset each round (Paper 1) |
| `static_nsga3_fixed` | NSGA-III | fixed p_m = 0.10 | myopic | reset |
| `dyn_nsga3_ucb1_myopic` | NSGA-III | UCB1 | myopic | AOS persists + warm start |
| `dyn_nsga3_linucb_myopic` | NSGA-III | LinUCB | myopic | persists + warm start |
| `dyn_nsga3_ucb1_batched` | NSGA-III | UCB1 | batched (n* = 90) | persists + warm start |
| `dyn_nsga3_linucb_batched` | NSGA-III | LinUCB | batched | persists + warm start |
| `myopic_ilp` | ILP (HiGHS) | — | myopic | — |
| `dickerson_lex` | lexicographic ILP | — | myopic | — |

**Paired design.** Every baseline in run *r* uses the same seed. That gives the
same initial pool, the same arrival stream, the same crossmatch/HLA draws
(hashed on donor–patient ids) and the same deceased-donor exit draws (hashed on
pair id and round). Configurations differ only through their matching
decisions, so the comparisons can use Wilcoxon *signed-rank* tests.

## Implementation choices the concept note leaves open

Each of these is a config switch, so you can change any of them.

1. **Sign of F3.** Eq. 5 minimises `−F3`, which means *maximising* F3. Eq. 8
   defines F3 as an absolute *gap*, which should be *minimised*. The default
   `moea.f3_mode: min_gap` follows Eq. 8 and the "Group Fairness Gap GFG_t"
   metric. `max_access` follows Eq. 5 and maximises `|M∩S|/|S|`. **Please fix
   one reading in the paper.**
2. **F2 population.** Gini over the patients who are *still waiting* after the
   round's matching.
3. **Implemented solution.** One point from the final Pareto front becomes M_t
   and drives the pool. The default is `compromise`: the front point closest
   to the normalised ideal point. `max_f1` and `min_f3` are also available.
   PoF_t depends on this rule, so the paper should state it.
4. **Round length** is 1 month, so τ ∈ {6, 12, 18} means 6, 12 or 18 rounds.
5. **AOS reward** is the per-generation hypervolume gain. HV is estimated by
   Monte Carlo on normalised objectives with a fixed sample per round. The gain
   is divided by the largest gain seen so far in that round, which keeps it in
   [0, 1].
6. **LinUCB context** (Eq. 11) changes only *between* rounds, so within a round
   LinUCB acts like a linear-prior UCB.
7. **Chains** are capped at `structures.max_chains`. Beyond the cap, chains of
   three or more recipients are random-walk sampled. The ILP and the MOEA get
   the *same* candidate set, so PoF_t keeps the identical-feasibility property
   required by §3.6.
8. **High-cPRA share.** 30 % is imposed on the *pool*, not on the arriving
   population.

## Issues found while building (please review)

* **The CIT constraint never binds with the stated numbers.** With CIT_max = 24 h
  and T_prep = 6 h, a match is ruled out only if transport takes more than 18 h.
  No ANZ route comes close: Perth–Auckland is about 13 h door to door. So every
  centre pair is feasible, and F4 cannot come from logistics at all. `calibrate`
  prints a warning. `configs/cit_binding.yaml` shows a binding scenario. The
  paper needs published transport times plus scheduling delays, or a different
  feasibility rule.
* **Batched clearing can stall.** π_B waits for 90 waiting patients. If the
  steady-state pool (roughly λ·τ) stays below 90, π_B never clears. That happens
  at τ = 6 with λ = 8 per month. `clearing.batch_max_interval` adds an optional
  safety valve, and `calibrate` warns when this applies.
* **Pool size and transplant volume cannot both match ANZKX.** Myopic ILP over
  25 monthly rounds, 3 runs each:

  | λ/month | τ | steady waiting patients | transplants / 25 months |
  |---|---|---|---|
  | 8 | 6 / 12 / 18 | ≈ 28 / 43 / 54 | ≈ 176 / 184 / 185 |
  | 16 | 6 / 12 / 18 | ≈ 56 / 80 / 93 | ≈ 315 / 323 / 325 |

  ANZKX reports a pool of 60–90 and 50–66 transplants a year (about 105–140
  per 25 months). The Saidman graph (density about 0.2) is easier to match than
  the real pool. Getting a realistic pool *and* a realistic volume needs a
  harder-to-match population: more highly sensitised patients, or a calibrated
  crossmatch/HLA model. Calibrate λ and compatibility together before the full
  runs.

## Tests

`tests/` covers ABO rules, calibration of the 30 % high-cPRA share,
deterministic hashing, CIT filtering, the Eq. 1 pool identity, withdrawal after
τ, both clearing policies, cycle/chain validity, ILP ≥ lexicographic volume,
feasibility of operator outputs, MOEA F1 ≤ ILP F1, Gini against its
definition, PoF, non-dominated sorting, Das–Dennis reference points (455 for
M = 4, p = 12), HV monotonicity, UCB1 picking the best arm, LinUCB learning a
context-dependent arm, every baseline end to end, the full stage pipeline
(including resume), and PrefLib parsing.
