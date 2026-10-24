# Paper 2 — complete results, modules 1 to 14

**Paper:** Dynamic Kidney Exchange Optimisation Under Realistic Clinical Constraints for Equitable Allocation (IEEE TEVC)
**Code:** `vscode_pipeline/`, branch `claude/module-wise-pipeline-oms5i3`
**Purpose:** one place with every result, in plain words, for the author and for another assistant. All numbers below are copied from the module outputs (module self-tests, `paper_numbers.md` of module 11, and the `.md` reports of modules 12, 13 and 14). Nothing is estimated.

**How to read the numbers**
- All results are from a **simulation** (synthetic pools calibrated to aggregate ANZKX statistics), plus one external dataset (PrefLib 00036). No result is a clinical outcome.
- "Paired runs": every method sees the same initial pool, the same arrivals and the same exits in run *k* (common random numbers). Tests are paired Wilcoxon signed-rank with Bonferroni correction; effect size is Cohen's d_z. "Significant" below always means p_bonf < 0.05.
- "n.s." = not significant after Bonferroni.
- τ = withdrawal threshold in months (6, 12, 18). T = 25 monthly clearing rounds.

---

## 0. The problem in one paragraph

Patient–donor pairs (and altruistic donors) arrive every month. Each month a clearing decides which cycles (2–3 pairs) and chains (altruist + up to 4 transplants) are carried out. Four objectives:

| Objective | Meaning | Better |
|---|---|---|
| F1 | number of patients transplanted in the round | higher |
| F2 | Gini coefficient of waiting times after the round (matched w, unmatched w+1) | lower |
| F3 | gap between the match rate of highly sensitised patients (cPRA ≥ 0.80) and the overall match rate | lower |
| F4 | gap between the match rate of patients at distant nodes (Perth, Auckland) and central nodes | lower |

- **PoF_t** (Price of Fairness in round t) = (F1 of the volume-maximising ILP − F1 chosen) / F1 of the ILP × 100%.
- **Δ** = transplants lost over the whole 25-round trajectory compared with the myopic ILP on the same pool sequence. Δ can be negative (more transplants than the ILP over time).

---

## Part A — Building blocks (modules 1–8): what each module does and what its tests showed

### Module 1 — Pool generator (extended Saidman, ANZKX-calibrated)
- Initial pool 75 vertices (70 pairs + 5 altruists); 7.9% altruists over 5,000 generated.
- 31.4% highly sensitised (cPRA ≥ 0.80).
- Why a pair is in the programme: 63.9% ABO-incompatible, 36.1% positive crossmatch.
- Blood types (patients / donors): O 59.3 / 26.9%, A 28.1 / 49.3%, B 10.8 / 17.8%, AB 1.8 / 6.0% — skewed towards O patients, as in real pools.
- Six regional nodes (a model, not ANZKX's 25+ units): Perth, Adelaide, Melbourne, Sydney, Brisbane, Auckland.
- **Key fact:** a highly sensitised patient has about 6× fewer compatible donors (15.0 vs 87.7 in the generator test).

### Module 2 — Pool evolution
- Pool equation V_{t+1} = (V_t minus matched, minus deceased-donor exits μ_d = 0.02, minus withdrawals at wait ≥ τ) plus Poisson(λ = 10) arrivals.
- Steady pools in the self-test: 57 / 96 / 135 patients at τ = 6 / 12 / 18 without matching.
- Checks passed: pool equation balances; identical arrivals across methods (paired design).

### Module 3 — Logistics and exchange structures
- Door-to-door transport time (flight + ground + waiting), e.g. Perth–Auckland 14.5 h.
- CIT limit 24 h: never binds. 16 h: rules out Auckland–Perth only. 12 h: rules out 73.3% of node pairs.
- 75-vertex pool: 22.9% of possible arcs feasible; high-cPRA patients have 4.6 compatible donors vs 26.2 for others.
- Chains capped at 5,000; when the cap binds, the chains of the exact PICEF optimum are added, so the ILP optimum is exact (20 of 20 test pools match PICEF; without the fix 6 of 20 lost 1–4 transplants).

### Module 4 — Objectives
- Shows that F2 must be measured "post-round": that version rewards serving long waiters (0.346 < 0.380); the "residual" version rewarded the wrong thing.
- The empty matching always has F3 = F4 = 0, so every selection rule must ignore it.

### Module 5 — NSGA-III + adaptive operator selection (AOS)
- NSGA-III, N = 100, G = 150, 455 reference directions; the bandit (UCB1 or LinUCB) chooses the mutation rate {0.01, 0.05, 0.10, 0.25} every generation from the hypervolume gain.
- **Key fact:** unguided NSGA-III reached only 25 of 35 possible transplants; with ILP guidance it reached 35. This is why the framework is ILP-guided — otherwise PoF would measure search failure, not fairness.
- External archive keeps every non-dominated solution; rounds with ≤ 200 feasible matchings are solved exactly by enumeration (no search).

### Module 6 — ILP baselines
- Myopic volume ILP (max transplants), Dickerson lexicographic (max high-cPRA, then total), PICEF (exact chains), and later the two-stage ILP (module 12).
- CBC retries + HiGHS fallback added after solver crashes.

### Module 7 — Clearing policies and one trajectory
- Myopic = clear every round; batched = clear when ≥ 90 patients wait (variant: also every 3 rounds at most).
- Implemented solution (original framework): among front solutions with F1 ≥ 90% of the maximum, the one closest to the ideal in (F2, F3, F4). This caps PoF_t at 10%.

### Module 8 — Statistics
- Friedman, paired Wilcoxon + Bonferroni/Holm, d_z, bootstrap CI, slope test. Self-test: planted effects detected, nulls not.

---

## Part B — Main experiment with the ORIGINAL framework (module 10; numbers from module 11's report)

Original framework = volume-ILP-guided NSGA-III with AOS, warm start, archive, 90% floor + compromise rule. 8 baselines, 100 paired runs, τ = 6 / 12 / 18.

### B1. Calibration (myopic ILP, λ = 10)
- 64.9 / 68.9 / 70.0 transplants per year (rounds 2–25); pool 31.5 / 50.3 / 64.0 patients (rounds 13–25); round 1 clears 43.9 transplants.
- ANZKX: about 50 transplants per year from 60–90 pairs. **No single λ matches both**; λ = 10 is a compromise. (λ = 7 and 13 tested as sensitivity.)

### B2. Framework (D-UCB1 myopic) vs myopic ILP
| | τ = 6 | τ = 12 | τ = 18 |
|---|---|---|---|
| Transplants | equal (n.s.), Δ −0.06% | equal (n.s.), Δ 0.16% | equal (n.s.), Δ 0.27% |
| F3 | −21.4% | −24.4% | −24.1% |
| F4 | −28.1% | −34.0% | −32.3% |
| F2 | −3.9% | −2.9% | −2.7% |
| High-cPRA patients withdrawn | 25.5 vs 26.3 | 13.6 vs 14.4 | 7.1 vs 7.9 |
All fairness differences p_bonf < 0.001; withdrawals p_bonf ≤ 0.032.

- PoF 1.60 / 1.81 / 1.91%. Round 1 (backlog): 8.44%; rounds 2–25: 1.28 / 1.50 / 1.61%, **flat** (slope n.s., p = 0.42–0.54). Do not write "PoF falls over time".
- PoF > 0 only in the 21–24% of rounds with F1_ILP ≥ 10 (there ≈ 7.7%).
- **Deferral:** 7.0–8.0 transplants per run were given up round by round, but the trajectory ended only −0.06 to 0.50 transplants below the ILP: 94–100% of deferred transplants happened later.

### B3. Dickerson lexicographic
- Same volume and PoF as the framework. About 20% more high-cPRA transplants (59.2 / 65.1 / 67.0) and far fewer high-cPRA withdrawals (17.5 / 7.5 / 3.1).
- But the largest F3 (0.087–0.095): it over-corrects (high-cPRA patients matched far more often than others), and no F2/F4 gain.

### B4. Myopic vs batched clearing (central question)
- τ = 6: batched **never clears** (pool never reaches 90): 0 transplants, 65.6 high-cPRA withdrawals vs 25.5.
- τ = 12: −24% transplants, −24% high-cPRA transplants, +33% high-cPRA withdrawals. τ = 18: −12% transplants.
- Batched PoF 8.1–8.4% per clearing.
- Variants (30 runs): 3-round maximum interval −4.4 / −4.5%; n* = 60 −5.5 / −1.5%; λ = 13 −12%; λ = 7 −60%. **No batching variant beats myopic clearing.**

### B5. AOS (honest negative result)
- UCB1 = LinUCB on every outcome; dynamic = static. UCB1's arm choice is ≈ uniform; LinUCB adapts slightly (prefers 0.25 under myopic, 0.01 under batched) but outcomes do not change.
- Likely reason: ILP guidance makes the search near-optimal from the start.

---

## Part C — PrefLib 00036, original framework (module 9)
12 real-graph instances (128–147 vertices), 10 paired runs each (120 units), τ = 12.
- Framework 75.03 vs ILP 75.38 transplants (n.s.); PoF 0.73%.
- F3 0.060 vs 0.062: **not significant** — the fairness gain did not transfer clearly.
- Batched (3-round max): −3.9% transplants, about double the F3.
- Dickerson costs 0.84% transplants on average over 179 instances (up to 2.2% at 294 vertices).
- Reason: sparse graphs and 91% of rounds small enough to solve exactly; few rounds offer a choice between matchings.

---

## Part D — Module 11: sensitivity, scalability, ablations

### D1. Selection rule (τ = 12, 30 runs, original framework)
- 100% floor: PoF = 0, transplants 179.2 vs ILP 178.4 (n.s.), F3 −18% vs ILP — **most of the fairness gain is free**.
- 80% floor: PoF 10.3%, −1.8% transplants, F3 −23% vs base.
- Compromise rule (no floor): PoF 37%, −5.8% transplants, F3 −60% but fewer high-cPRA transplants (levelling down).

### D2. Other sensitivity
- CIT 16 h: no change. CIT 12 h: −13 to −14% transplants for every method; framework still equals the ILP and still lowers F3.
- μ_d (0.01, 0.04, 0.3×μ_d for high-cPRA): no effect on PoF.
- λ = 7 / 13: framework equals the ILP on volume; F3 −22 / −27%; PoF 1.03 / 3.04%.

### D3. Scalability
- 200 initial vertices: framework 37 s per round (ILP 5 s, lex 8 s), keeps the ILP's volume (27.6 vs 27.7 per round), F3 −39%, PoF 7.0%.
- 500 vertices: ILP 38 s, lex 60 s per round; framework not run (memory: exhaustive cycle enumeration). Lex F3 grows to 0.26 (200) and 0.46 (500).

### D4. Hybrid (guidance) ablation — one round, 30 pools of 75 vertices
| Arm | Pools at ILP optimum | Max F1 (ILP = 42.3) | HV | PoF of chosen solution | s/round |
|---|---|---|---|---|---|
| NSGA-III alone | 0/30 | 32.6 | 0.566 | 28.3% | 18.2 |
| ILP seed only | 0/30 | 32.7 | 0.566 | 28.8% | 20.9 |
| Seed + repair priority | 23/30 | 41.8 | 0.696 | 9.2% | 25.1 |
| Full hybrid (+ archive) | 30/30 | 42.3 | 0.703 | 8.23% | 23.6 |
- Seeding alone does nothing (repair destroys the seed); repair priority is what makes guidance work; the archive guarantees the optimum stays on the front.

### D5. AOS ablation
- With guidance, UCB1 beats fixed 0.10 and 0.25, slightly below fixed 0.01; all HV differences < 3%.

### D6. Implementation checks
- 48% of rounds solved exactly; chain cap bound in 9.6% of rounds; PICEF chains added in 6.9%; 0 floor violations; 23 s per framework clearing.

---

## Part E — Module 12: two-stage ILP baseline (supervisor's request)

Two-stage ILP: stage 1 maximises transplants; stage 2 fixes that number and picks the fairest maximum matching, minimising F2/r2 + F3/r3 + F4/r4 ("sum", r = values of the volume-ILP matching, floored at 0.01) or F3 alone. Same pool, graph, structures and constraints as the framework. Exact (linearised) with a time limit.

### E1. Same pool, one round (30 pools of 75 vertices)
- Two-stage sum F3 0.0217 / F4 0.0348 vs the MOEA's 100%-floor pick 0.0548 / 0.1143. The MOEA front held only ≈ 1.1 solutions at maximum volume.
- Time: ILP 3.2 s, two-stage sum 233 s (hit the 300 s limit in 67% of pools), two-stage F3 58 s (17%), MOEA 27 s.

### E2. Same pool in every round of a trajectory (689 rounds, τ = 12)
- Combined score (lower = fairer): ILP 3.0, two-stage 2.322, MOEA pick 2.606 → the MOEA achieved about 58% of the possible improvement.
- MOEA's best F3 at maximum volume equalled the optimum in 95% of rounds; the gap was mainly the selection rule (exact rounds 2.79 vs 2.63) and search (searched rounds 2.49 vs 2.14).

### E3. Over trajectories (30 runs, τ = 6 / 12 / 18)
- Two-stage sum ≈ original framework (90% floor) on fairness (τ = 12 n.s.), better F4 at τ = 18, more transplants at τ = 6, PoF 0. Better than the framework at a 100% floor on F2/F3/F4.

**Conclusion of module 12:** at maximum volume the exact two-stage ILP beats the original framework. The MOEA's possible added value is the trade-off below maximum volume. This motivated module 13.

---

## Part F — Module 13: the new framework, two-stage-ILP-guided NSGA-III (TSG)

### F1. What changed (everything else identical)
| Component | Original framework | TSG |
|---|---|---|
| Guide | volume-ILP matching (seed + repair priority) | two-stage ILP "sum" matching (seed + repair priority); volume-ILP and two-stage F3 matchings also seeded |
| When the two-stage ILP is solved | — | only in searched rounds (> 200 matchings) |
| Final front | archive | archive + the 3 exact matchings (re-filtered, HV recomputed); always contains the fairest maximum matching |
| Selection | F1 ≥ 0.9·max, closest to ideal after min–max scaling | F1 ≥ ρ·max (ρ = 0.9), minimise F2/r2 + F3/r3 + F4/r4 (the two-stage stage-2 objective) |
| Implementation | modules 5, 7 | module 13, run-time extension; modules 1–12 unchanged |

30 paired runs per τ (seeds 42–71, the same runs as module 12). The original framework and baselines are the corresponding runs of module 10 (so their means differ slightly from Part B, which uses 100 runs).

### F2. Main results (myopic, UCB1)
| Metric | τ | Old framework | TSG | Two-stage ILP (sum) | Myopic ILP | Dickerson lex |
|---|---|---|---|---|---|---|
| Transplants | 6 / 12 / 18 | 170.2 / 178.6 / 180.6 | 170.8 / 178.6 / 181.0 | 171.9 / 179.1 / 181.5 | 170.0 / 178.4 / 180.5 | 170.8 / 178.1 / 180.2 |
| High-cPRA tx | 6 / 12 / 18 | 49.9 / 54.9 / 56.5 | 51.1 / 55.4 / 57.1 | 51.5 / 55.7 / 57.5 | 48.9 / 53.9 / 55.5 | 59.7 / 65.5 / 67.9 |
| F2 | 6 / 12 / 18 | 0.355 / 0.378 / 0.388 | 0.354 / 0.374 / 0.385 | 0.355 / 0.375 / 0.385 | 0.370 / 0.389 / 0.400 | 0.368 / 0.388 / 0.398 |
| F3 | 6 / 12 / 18 | 0.0595 / 0.0373 / 0.0337 | 0.0520 / 0.0333 / 0.0303 | 0.0544 / 0.0352 / 0.0319 | 0.0752 / 0.0500 / 0.0450 | 0.0891 / 0.0840 / 0.0839 |
| F4 | 6 / 12 / 18 | 0.101 / 0.061 / 0.058 | 0.089 / 0.053 / 0.047 | 0.091 / 0.056 / 0.048 | 0.140 / 0.093 / 0.081 | 0.142 / 0.092 / 0.080 |
| PoF (%) | 6 / 12 / 18 | 1.46 / 1.68 / 1.83 | 1.06 / 1.23 / 1.21 | 0 | 0 | 1.36 / 1.85 / 2.07 |
| Δ (%) | 6 / 12 / 18 | −0.13 / −0.10 / −0.05 | −0.51 / −0.15 / −0.27 | −1.12 / −0.43 / −0.53 | 0 | −0.48 / 0.19 / 0.19 |
| s/round | | ≈ 20 | ≈ 25–27 | ≈ 4–5 | ≈ 0.5 | ≈ 1 |

### F3. Significance (TSG UCB1 myopic vs each baseline; p_bonf < 0.05)
| vs | τ = 6 | τ = 12 | τ = 18 | Transplants / Δ |
|---|---|---|---|---|
| Old framework | F3 ↓, F4 ↓ | F2 ↓, F4 ↓, PoF ↓ | F2 ↓, F4 ↓, PoF ↓ | n.s. |
| Old framework, LinUCB pair | F3 ↓, F4 ↓ | F3 ↓, F4 ↓, PoF ↓ | F3 ↓, F4 ↓, PoF ↓ | n.s. |
| Static NSGA-II + UCB1 | PoF ↓ | F2, F3, F4, PoF ↓ | F3, F4, PoF ↓ | n.s. |
| Static NSGA-III fixed | F3 ↓, F4 ↓ | F2, F3, F4 ↓ | F2, F3, F4 ↓ | n.s. |
| Two-stage ILP (sum) | none significant (TSG means slightly lower on F2/F3/F4; about 1 fewer transplant at τ = 6, p_bonf = 0.065) | none | none | n.s. |
| Two-stage ILP (F3) | F2 ↓, F4 ↓ | F2 ↓ | F2 ↓ | n.s. |
| Myopic ILP | F2, F3, F4 ↓ (d_z ≈ 2) | F2, F3, F4 ↓; high-cPRA withdrawn ↓ | F2, F3, F4 ↓ | n.s. |
| Dickerson lex | F2, F3, F4 ↓ — but lex has ≈ 9–11 more high-cPRA transplants and fewer high-cPRA withdrawals (significant) | same | same | n.s. |
PoF vs the ILP baselines is always higher for TSG by definition (ILPs have PoF 0).

### F4. Myopic vs batched with TSG
- τ = 6: batched never clears (as in Part B).
- τ = 12: batched −46 transplants (132.3 vs 178.6), Δ 26.5%, PoF 6.4%; lower F3/F4 per round. τ = 18: Δ 10.3%.
- Myopic wins on volume and high-cPRA transplants at every τ (all p_bonf < 0.001). TSG batched improves on old batched (τ = 12: F4 0.014 vs 0.073, Δ 26.5% vs 30.1%; not formally tested).

### F5. Selection rule, floor ρ and guide (τ = 12)
| Configuration | Tx | F2 | F3 | F4 | PoF | Δ |
|---|---|---|---|---|---|---|
| TSG ρ = 0.9 | 178.6 | 0.3743 | 0.0333 | 0.0528 | 1.23 | −0.15 |
| TSG ρ = 1.0 | 179.1 | 0.3747 | 0.0351 | 0.0557 | 0 | −0.41 |
| TSG ρ = 0.8 | 176.6 | 0.3697 | 0.0242 | 0.0426 | 7.66 | 1.01 |
| TSG, original rule | 178.6 | 0.3727 | 0.0352 | 0.0586 | 1.80 | −0.12 |
| Old + score rule | 178.8 | 0.3807 | 0.0356 | 0.0540 | 1.34 | −0.26 |
| Old framework | 178.6 | 0.3778 | 0.0373 | 0.0610 | 1.68 | −0.10 |
| Two-stage ILP (sum) | 179.1 | 0.3752 | 0.0352 | 0.0556 | 0 | −0.43 |
- **ρ = 1.0 ≈ two-stage ILP:** no significant difference on any metric (not bit-identical: stage 2 hits its time limit in ≈ 8% of searched rounds, and ties in exact rounds are broken differently).
- **ρ = 0.8 vs two-stage ILP:** F2, F3, F4 all significantly lower (d_z −1.5 to −1.8), costing 2.5 transplants (Δ 1.0% vs −0.43%, p_bonf 0.002) and 1.5 fewer high-cPRA transplants (p_bonf 0.018). Caution: a smaller F3 gap does NOT mean more high-cPRA transplants.
- **Guide vs rule:** the score rule lowers PoF (1.23 vs 1.80, significant) but alone worsens F2; the two-stage guide improves F2 (0.3743 vs 0.3807, significant). Both contribute.

### F6. Cost
- Two-stage solve ≈ 10–11 s per searched round (myopic; ≈ 50% of rounds are searched); stage 2 hits the 60 s limit in ≈ 8% of searched myopic rounds and 62–80% of batched rounds (larger pools).

---

## Part G — Module 14: TSG on PrefLib 00036
Same 12 instances, 120 paired units, τ = 12, as module 9.

| Configuration | Tx | High-cPRA tx | F2 | F3 | PoF | Δ | High-cPRA withdrawn |
|---|---|---|---|---|---|---|---|
| TSG UCB1 myopic | 75.28 | 11.02 | 0.3113 | 0.0554 | 0.44 | 0.21 | 4.81 |
| TSG ρ = 0.8 | 74.36 | 10.71 | 0.3086 | 0.0543 | 3.04 | 1.43 | 5.07 |
| Old UCB1 myopic | 75.03 | 10.59 | 0.3160 | 0.0598 | 0.73 | 0.50 | 5.18 |
| Two-stage ILP (sum) | 75.79 | 11.40 | 0.3120 | 0.0553 | 0 | −0.50 | 4.58 |
| Two-stage ILP (F3) | 75.36 | 11.45 | 0.3242 | 0.0562 | 0 | 0.06 | 4.60 |
| Myopic ILP | 75.38 | 10.72 | 0.3250 | 0.0623 | 0 | 0 | 5.18 |
| Dickerson lex | 75.09 | 13.57 | 0.3240 | 0.0799 | 1.54 | 0.41 | 2.93 |
| TSG batched (max 3) | 72.48 | 10.62 | 0.3228 | 0.1027 | 1.36 | 3.88 | 4.87 |
| Old batched (max 3) | 72.12 | 10.16 | 0.3255 | 0.1182 | 2.20 | 4.35 | 5.25 |

- **TSG vs old framework:** F2, F3, PoF lower; +0.43 high-cPRA transplants; fewer high-cPRA withdrawals (all significant); transplants/Δ n.s. → **the fairness gain now transfers to PrefLib** (it did not with the old framework).
- **TSG vs myopic ILP:** F2 (d_z −1.4) and F3 (d_z −0.6) significantly lower; same transplants.
- **TSG vs two-stage ILP (sum): the two-stage ILP is better on PrefLib** — 0.5 more transplants, more high-cPRA transplants, better Δ, fewer high-cPRA withdrawals (all significant); F2/F3 equal.
- **ρ = 0.8 does not pay on PrefLib:** −1.4 transplants vs the two-stage ILP, F2 slightly better, F3 equal.
- Reason: 91% of PrefLib rounds are solved exactly and the graphs are sparse; rounds rarely contain a fairer matching below maximum volume, so the 90% floor gives up transplants without gaining fairness.
- Batched worse than myopic (Δ 3.9%, F3 nearly double); TSG batched better than old batched.
- Cost: stage 2 hits the 60 s limit in 34% of searched PrefLib rounds.

---

## Part H — What the evidence supports (and what it does not)

### Supported claims
1. **Myopic beats batched clearing** at every τ, with both frameworks and on PrefLib; no batching variant tested beats myopic clearing (batched never clears at τ = 6 with n* = 90).
2. **Fairness at almost no volume cost:** the ILP-guided MOEA frameworks equal the myopic ILP in transplants over 25 rounds while lowering F2, F3 and F4; most deferred transplants are recovered later (94–100%); PoF is flat after round 1.
3. **TSG is fairer than the original framework** (F2/F3/F4/PoF, most significant) at the same volume, on synthetic data and on PrefLib, and fairer than both static MOEA baselines.
4. **TSG matches the two-stage ILP on synthetic pools at ρ = 0.9** (no significant difference) and reaches **significantly fairer solutions at ρ = 0.8** for ≈ 1% of transplants — something an ILP fixed at maximum volume cannot offer.
5. **On PrefLib the two-stage ILP is the better method** at maximum volume; the value of the MOEA front depends on how often rounds contain several matchings that differ in fairness.
6. **Dickerson lex** is best for high-cPRA access (transplants, withdrawals) but over-corrects (largest F3) and does not improve F2/F4 — a different goal, not a worse or better method overall.
7. **ILP guidance (repair priority + archive) is essential**; seeding alone does nothing.
8. **AOS does not change outcomes** under ILP guidance (honest negative result).

### Do NOT claim
- "TSG outperforms all baselines" (ties the two-stage ILP on synthetic, loses to it on PrefLib, loses to lex on high-cPRA access).
- "Lower F3 = more high-cPRA transplants" (F3 is a gap between rates).
- "PoF falls over time"; "AOS improves performance"; "exact ANZKX calibration"; "six ANZKX transplant centres"; "scalability to 1,000 vertices"; any clinical outcome.

### Evidence, assumptions and synthetic results (supervisor's distinction)
- **Published evidence used as input:** ANZKX/ANZDATA aggregate statistics (pool size 60–90 pairs, ≈ 50 transplants per year, ≈ 30% highly sensitised), blood-group frequencies, Saidman generator structure, PrefLib 00036 graphs.
- **Modelling assumptions:** λ = 10, τ ∈ {6, 12, 18}, μ_d = 0.02, node weights, transport times, CIT 24 h, F4 node grouping, 90% floor, cycle ≤ 3 / chain ≤ 4.
- **Synthetic results:** everything in Parts B, D, E, F (simulated pools); Parts C and G use real compatibility graphs but simulated arrivals, waiting times and node locations.

### Limitations to state
- Synthetic pools; calibration mismatch (≈ 65–70 transplants per year vs ≈ 50; pool ≈ 50 vs 60–90 at τ = 12).
- F4 is a proxy; CIT never binds at 24 h; no post-match crossmatch failures, suspensions or medical urgency.
- PoF level is set by the floor rule.
- Exhaustive cycle enumeration limits scale (framework not run at 500 vertices).
- Two-stage stage 2 is time-limited (60 s); it hits the limit in 8% (synthetic myopic), 62–80% (batched) and 34% (PrefLib) of searched rounds.
- TSG results use 30 paired runs (main experiment used 100); ρ = 0.8 tested at τ = 12 only.
- Batched clearing was evaluated without information about upcoming departures (favours myopic clearing).

---

## Part I — Status
- Done: modules 1–14. Module 15 (figures/tables for TSG) written; to be run.
- Decision pending with the supervisor: replace the original framework by TSG in the paper (option A) or keep the original and add TSG as an extension (option B).
- Optional if option A: module 13 with 100 runs to match the main experiment.
