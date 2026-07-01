# `srt.overfitting.cscv` — CSCV & Probability of Backtest Overfitting (PBO)

Combinatorially Symmetric Cross-Validation (CSCV) and the resulting **Probability of
Backtest Overfitting (PBO)** — the library's headline diagnostic for distinguishing a
real out-of-sample edge from a selection-bias artifact.

> **Note:** This README was generated with the assistance of an AI tool (Claude) and
> reviewed by the author. It documents the *intended* design of the module; the
> implementation is written by hand by the author.

> **Reference:** Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of
> Backtest Overfitting*, Journal of Computational Finance.

---

## The question it answers

Given `N` candidate strategy configurations backtested over the same `T` periods: if
you select the configuration that looks best in-sample, how likely is it that this
selection is an overfit that underperforms out-of-sample? PBO quantifies exactly that
probability.

## Input

A `T×N` matrix of per-period returns/PnL (rows = time, columns = candidate
configurations) — the aligned matrix produced by `srt.loader`.

## Output

A scalar **PBO ∈ [0, 1]**, plus (intended) the full distribution of per-combination
logits `λ_c` for inspection.

---

## Algorithm

Let `S` be an even number of contiguous sub-matrices (e.g. `S = 16`).

1. **Partition.** Split the `T` rows into `S` disjoint, contiguous blocks.
2. **Enumerate.** Form all `C(S, S/2)` combinations of blocks. Each combination `c` is
   an in-sample (training) set `J`; its complement `J̄` is the out-of-sample (testing)
   set. For `S = 16` this is `C(16, 8) = 12,870` combinations.
3. **For each combination `c`:**
   1. Build `J` (the `S/2` chosen blocks stacked) and `J̄` (the complement).
   2. Score every configuration on `J` → performance vector `R` of length `N`.
   3. `n* = argmax(R)` — the index of the in-sample-best configuration.
   4. Score every configuration on `J̄` → performance vector `R̄` of length `N`.
   5. Find the **relative rank** `ω_c ∈ (0, 1)` of `R̄[n*]` within `R̄`.
   6. Compute the **logit** `λ_c = ln( ω_c / (1 − ω_c) )`.
4. **Aggregate.** `PBO = fraction of combinations with λ_c < 0`.

`λ_c < 0` ⟺ `ω_c < 0.5` ⟺ the in-sample-best configuration landed **below the OOS
median** ⟺ overfit for that split.

### Combinatorial symmetry

The set of `C(S, S/2)` combinations is closed under complement: for every training set
`J`, its complement also appears as another combination's training set. IS and OOS
roles swap across the enumeration — this is what makes the cross-validation
"symmetric," and it is the answer to *why not a single train/test split*: a single
split gives one noisy estimate; the symmetric enumeration averages over all balanced
splits.

---

## Key conventions (to be enforced)

- **Relative rank → `(0, 1)`.** `ω_c = rank / (N + 1)` with 1-based ascending rank
  (rank 1 = worst OOS, rank `N` = best OOS). The `N + 1` denominator keeps `ω_c`
  strictly inside `(0, 1)` so the logit never diverges to `±∞`.
- **Direction.** Higher OOS performance → higher rank → higher `ω` → higher `λ`.
  A configuration that is strong IS *and* strong OOS yields large positive `λ`
  (consistent, not overfit); strong IS but weak OOS yields large negative `λ` (overfit).
- **`λ_c = 0` at the OOS median** (`ω_c = 0.5`) — the reference line for "no
  relationship."
- **Per-block performance scalar.** The IS/OOS score is computed on the concatenated
  chosen rows (not a per-block average). Sharpe is the standard choice.
- **Ties.** Use a deterministic rank convention (e.g. average ranks) for reproducibility.

---

## Invariants (tests)

| Invariant | Rationale |
|-----------|-----------|
| **PBO ≈ 0.5** on i.i.d. no-edge returns | With no real edge, the IS-best config is a coin-flip OOS. |
| Deliberately-overfit synthetic → **high PBO** | Catches a ranking **sign flip** that the 0.5 anchor cannot (0.5 is symmetric under `PBO → 1 − PBO`). |
| Combination count `== C(S, S/2)` | Enumeration completeness (cardinality). |
| Each block appears in exactly `C(S−1, S/2−1)` training sets | Enumeration completeness (uniform coverage). |
| Every combination's complement is also present | Combinatorial symmetry holds. |

The **sign-flip test is essential**: the headline `≈ 0.5` invariant is symmetric and
will pass even if the rank direction is reversed, so a separate high-PBO synthetic is
required to pin the orientation.

---

## Design / performance notes

- The expensive quantity is the per-block performance statistic, and there are only
  `S` blocks. Precompute per-block summary statistics once (e.g. `n, Σx, Σx²` per block
  per configuration), then assemble each combination's IS/OOS score from the cached
  blocks rather than re-slicing the full matrix `C(S, S/2)` times.
- Mean and variance of a concatenation of blocks are recoverable from per-block
  `(n, Σx, Σx²)`, so a concatenated-Sharpe can be assembled cheaply without
  recomputing over raw rows.

## Relationship to `sharpe_stats`

PBO measures the **selection process**. It runs on the **full population** of `N`
candidates — do **not** pre-filter candidates by PSR/MinTRL/DSR before CSCV, as that
would be in-sample selection on the same data the test evaluates (leakage). PSR, DSR
(which takes the trial count `N` as input to deflate the benchmark), and MinTRL sit
**downstream** as a verdict on survivors, not as a pre-filter.

## Status

Specification / design document. Implementation in progress, written by hand by the
author.
