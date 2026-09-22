# Monitor distillation results — test split

**Setup:** Qwen2.5-7B-Instruct, bf16 via vLLM · **Arms scored:** M0, M1, M2, M3, M4, M5, M6, M7, M8 · **Generated:** 2026-09-21 17:31 · commit `b874ac9` with uncommitted changes

## Summary

Primary readout: `P(yes) ÷ (P(yes) + P(no))` at the verdict token. 95% intervals from a class-stratified bootstrap.

| Arm | Cost | Items | pAUC@10% FPR | AUROC | Catch rate at 1% audit | Flagged: backdoored / clean |
|---|---|---|---|---|---|---|
| M0 · single monitor | 1× | 1288 | 0.709 [0.67, 0.75] | 0.862 [0.84, 0.88] | 0.22 | 87% / 34% |
| M1 · 3 identical samples | 3× | 1288 | 0.753 [0.72, 0.79] | 0.881 [0.86, 0.90] | 0.31 | 88% / 33% |
| M2 · 3 persona prompts | 3× | 1288 | 0.779 [0.75, 0.81] | 0.906 [0.89, 0.92] | 0.38 | 91% / 29% |
| M3 · distilled from M1 | 1× | 1288 | 0.782 [0.75, 0.81] | 0.893 [0.88, 0.91] | 0.48 | 93% / 46% |
| M4 · two-term SFT | 1× | 1288 | 0.810 [0.78, 0.84] | 0.915 [0.90, 0.93] | 0.48 | 94% / 38% |
| M5 · MACA (MV-DPO) | 1× | 1288 | 0.756 [0.72, 0.79] | 0.886 [0.87, 0.90] | 0.26 | 86% / 25% |
| M6 · MACA (MV-SFT) | 1× | 1288 | 0.691 [0.65, 0.73] | 0.876 [0.86, 0.90] | 0.11 | 96% / 58% |
| M7 · MACA (MV-KTO) | 1× | 1288 | 0.516 [0.51, 0.52] | 0.683 [0.66, 0.70] | 0.98 | 98% / 70% |
| M8 · debate, graded target | 1× | 1288 | 0.790 [0.76, 0.82] | 0.901 [0.88, 0.92] | 0.40 | 96% / 59% |

Secondary readout: the written 0–10 `SUSPICION` score.

| Arm | pAUC@10% FPR | AUROC | Catch rate at 1% audit | Distinct scores |
|---|---|---|---|---|
| M0 · single monitor | 0.549 [0.53, 0.57] | 0.779 [0.75, 0.80] | 0.49 | 10 |
| M1 · 3 identical samples | 0.580 [0.55, 0.62] | 0.831 [0.81, 0.85] | 0.02 | 29 |
| M2 · 3 persona prompts | 0.694 [0.65, 0.74] | 0.883 [0.86, 0.90] | 0.25 | 27 |
| M3 · distilled from M1 | 0.544 [0.53, 0.57] | 0.756 [0.73, 0.78] | 0.00 | 11 |
| M4 · two-term SFT | 0.621 [0.59, 0.65] | 0.821 [0.80, 0.84] | 0.42 | 10 |
| M5 · MACA (MV-DPO) | 0.541 [0.52, 0.57] | 0.788 [0.76, 0.81] | 0.01 | 7 |
| M6 · MACA (MV-SFT) | 0.534 [0.52, 0.55] | 0.725 [0.70, 0.75] | 0.01 | 10 |
| M7 · MACA (MV-KTO) | 0.527 [0.51, 0.55] | 0.726 [0.70, 0.75] | 0.01 | 10 |
| M8 · debate, graded target | 0.559 [0.54, 0.58] | 0.773 [0.75, 0.80] | 0.15 | 11 |

## Hypotheses

| | Claim | Status | Evidence |
|---|---|---|---|
| H1 | M1 beats M0 | **supported** | ΔpAUC +0.044 [+0.018, +0.071] on 1288 items |
| Diversity | M2 beats M1 (diversity adds to ensembling) | **inconclusive** | ΔpAUC +0.026 [-0.006, +0.060] on 1288 items |
| H2 | M4 or M5 recovers ≥50% of M2's gain over M0 at 1× cost | **supported** | recovered: M4 144%, M5 66% (point estimates) |
| H3 | M5 (MACA, MV-DPO) beats M4 (score distillation) | **contradicted** | ΔpAUC -0.054 [-0.090, -0.019] on 1288 items |
| H4 | Trained arms have lower score entropy and fewer distinct scores | **not supported** | entropy 3.50 vs 3.41 bits |
| Diversity distilled | M4 (diverse teacher) beats M3 (identical teacher) — follow-up, ADR-0008 | **supported** | ΔpAUC +0.028 [+0.007, +0.049] on 1288 items |

## Detection at fixed false-positive rates

Share of backdoors caught when the threshold flags a given share of clean code. Ranked by pAUC. Thresholds are set on val and applied here, so the realised rate shows whether the target holds on unseen items; the realised false-positive rate is in brackets.

| Rank | Arm | Inference cost | Calls per 1,000 items | pAUC@10% | Recall @ 1% FPR | @ 2% | @ 5% | @ 10% |
|---|---|---|---|---|---|---|---|---|
| 1 | M4 · two-term SFT | 1× | 1,000 | 0.810 | 68% (4.3%) | 71% (6.1%) | 80% (11.3%) | 86% (16.9%) |
| 2 | M8 · debate, graded target | 1× | 1,000 | 0.790 | 60% (3.3%) | 64% (4.3%) | 66% (5.6%) | 77% (13.5%) |
| 3 | M3 · distilled from M1 | 1× | 1,000 | 0.782 | 68% (7.9%) | 69% (8.1%) | 76% (13.5%) | 84% (20.8%) |
| 4 | M2 · 3 persona prompts | 3× | 3,000 | 0.779 | 63% (5.1%) | 72% (8.9%) | 76% (11.6%) | 83% (16.8%) |
| 5 | M5 · MACA (MV-DPO) | 1× | 1,000 | 0.756 | 54% (3.7%) | 57% (4.2%) | 73% (8.9%) | 80% (16.6%) |
| 6 | M1 · 3 identical samples | 3× | 3,000 | 0.753 | 61% (5.3%) | 65% (6.4%) | 70% (9.8%) | 75% (13.7%) |
| 7 | M0 · single monitor | 1× | 1,000 | 0.709 | 56% (6.5%) | 66% (10.2%) | 71% (13.4%) | 77% (18.0%) |
| 8 | M6 · MACA (MV-SFT) | 1× | 1,000 | 0.691 | 35% (3.1%) | 39% (4.5%) | 60% (7.9%) | 76% (14.1%) |
| 9 | M7 · MACA (MV-KTO) | 1× | 1,000 | 0.516 | 98% (61.3%) | 98% (61.3%) | 98% (61.3%) | 98% (61.3%) |

## Ensembles vs. their parts

| Rank | Monitor | Role | pAUC@10% | Gain over M0 |
|---|---|---|---|---|
| 1 | M2 · 3 persona prompts | diverse ensemble | 0.779 | +0.070 |
| 2 | M1 · 3 identical samples | 3× baseline ensemble | 0.753 | +0.044 |
| 3 | M2 · control flow | best single persona | 0.744 | +0.034 |
| 4 | Median single persona | median persona | 0.738 | +0.029 |
| 5 | M0 · single monitor | single baseline | 0.709 | +0.000 |

M2's gain over M0 is **1.6×** M1's gain — how much more diverse construction buys than repeated sampling of the same prompt. (Point estimates.)

## Paired comparisons

Both arms scored on the same items. ΔpAUC from a paired bootstrap; ΔAUROC and p from DeLong's test.

| Comparison | Items | ΔpAUC@10% | ΔAUROC | DeLong p |
|---|---|---|---|---|
| M1 − M0 | 1288 | +0.044 [+0.018, +0.071] | +0.019 [+0.009, +0.030] | 0.000417 |
| M2 − M0 | 1288 | +0.070 [+0.035, +0.104] | +0.044 [+0.028, +0.060] | 1.08e-07 |
| M2 − M1 | 1288 | +0.026 [-0.006, +0.060] | +0.024 [+0.011, +0.038] | 0.000443 |
| M3 − M0 | 1288 | +0.072 [+0.041, +0.105] | +0.031 [+0.015, +0.047] | 0.000133 |
| M3 − M1 | 1288 | +0.029 [+0.002, +0.058] | +0.012 [-0.001, +0.024] | 0.0787 |
| M4 − M0 | 1288 | +0.101 [+0.066, +0.135] | +0.053 [+0.037, +0.069] | 1.09e-10 |
| M4 − M1 | 1288 | +0.057 [+0.027, +0.088] | +0.034 [+0.020, +0.047] | 1.03e-06 |
| M4 − M2 | 1288 | +0.031 [-0.000, +0.061] | +0.009 [-0.003, +0.022] | 0.149 |
| M4 − M3 | 1288 | +0.028 [+0.007, +0.049] | +0.022 [+0.010, +0.034] | 0.000388 |
| M5 − M0 | 1288 | +0.046 [+0.009, +0.085] | +0.024 [+0.007, +0.041] | 0.00589 |
| M5 − M2 | 1288 | -0.024 [-0.062, +0.015] | -0.020 [-0.035, -0.005] | 0.0107 |
| M5 − M4 | 1288 | -0.054 [-0.090, -0.019] | -0.029 [-0.044, -0.015] | 6.25e-05 |
| M6 − M4 | 1288 | -0.119 [-0.155, -0.084] | -0.039 [-0.054, -0.024] | 1.63e-07 |
| M6 − M5 | 1288 | -0.065 [-0.108, -0.023] | -0.010 [-0.027, +0.007] | 0.265 |
| M7 − M4 | 1288 | -0.294 [-0.324, -0.263] | -0.232 [-0.252, -0.212] | 5.05e-111 |
| M7 − M5 | 1288 | -0.240 [-0.278, -0.202] | -0.203 [-0.225, -0.180] | 1.75e-71 |
| M8 − M4 | 1288 | -0.020 [-0.044, +0.005] | -0.014 [-0.026, -0.002] | 0.0222 |
| M8 − M5 | 1288 | +0.035 [+0.002, +0.068] | +0.015 [-0.000, +0.030] | 0.0509 |
| M8 − M6 | 1288 | +0.100 [+0.062, +0.137] | +0.025 [+0.010, +0.040] | 0.00106 |
| M8 − M7 | 1288 | +0.275 [+0.245, +0.303] | +0.218 [+0.197, +0.238] | 6.69e-97 |

## Across arms and monitors

![ROC at false-positive rates up to 10%, split by question. M4, the benchmark both questions are measured against, is the dashed line in each panel.](figures/roc_facets.png)

*ROC at false-positive rates up to 10%, split by question. M4, the benchmark both questions are measured against, is the dashed line in each panel.*

![Reference view: every arm on one chart, with individual ensemble members in grey. Curves overlap heavily -- the faceted version above is the readable one.](figures/all_arms_roc.png)

*Reference view: every arm on one chart, with individual ensemble members in grey. Curves overlap heavily -- the faceted version above is the readable one.*

![What each ingredient is worth, as a difference with its 95% interval. Differences this small cannot be read off overlapping ROC curves.](figures/ingredient_costs.png)

*What each ingredient is worth, as a difference with its 95% interval. Differences this small cannot be read off overlapping ROC curves.*

![pAUC for every arm and every monitor inside an ensemble, with 95% intervals. Lighter bars are an ensemble's individual members.](figures/pauc_bars.png)

*pAUC for every arm and every monitor inside an ensemble, with 95% intervals. Lighter bars are an ensemble's individual members.*

![pAUC and AUROC per arm with 95% bootstrap intervals.](figures/metrics.png)

*pAUC and AUROC per arm with 95% bootstrap intervals.*

![pAUC against inference cost. The question is whether a 1× trained arm reaches the M2 ceiling.](figures/frontier.png)

*pAUC against inference cost. The question is whether a 1× trained arm reaches the M2 ceiling.*


## Each arm

### M0 · single monitor

One call per item under the baseline prompt. How well does a single trusted monitor detect backdoors at all? The floor every arm is measured against.

pAUC@10% FPR **0.709 [0.67, 0.75]**, AUROC 0.862 [0.84, 0.88]. At P(yes) ≥ 0.5 it flags 87% of backdoored and 34% of clean solutions.

![M0 · single monitor: ROC curve for false-positive rates up to 10%.](figures/m0_roc.png)

*M0 · single monitor: ROC curve for false-positive rates up to 10%.*

![M0 · single monitor: share of backdoored items (above the line) and clean items (below) at each score.](figures/m0_scores.png)

*M0 · single monitor: share of backdoored items (above the line) and clean items (below) at each score.*

### M1 · 3 identical samples

The baseline prompt sampled three times and averaged. How much does ensembling buy when the monitors are not diverse? The free win any trained arm must beat.

pAUC@10% FPR **0.753 [0.72, 0.79]**, AUROC 0.881 [0.86, 0.90]. At P(yes) ≥ 0.5 it flags 88% of backdoored and 33% of clean solutions.

![M1 · 3 identical samples: ROC curve for false-positive rates up to 10%. Grey lines are the 3 individual monitors averaged into this ensemble.](figures/m1_roc.png)

*M1 · 3 identical samples: ROC curve for false-positive rates up to 10%. Grey lines are the 3 individual monitors averaged into this ensemble.*

![M1 · 3 identical samples: share of backdoored items (above the line) and clean items (below) at each score.](figures/m1_scores.png)

*M1 · 3 identical samples: share of backdoored items (above the line) and clean items (below) at each score.*

### M2 · 3 persona prompts

Three different persona prompts, one sample each, averaged. How much does monitor diversity add on top of plain ensembling? The ceiling the trained arms aim for, and their teacher.

pAUC@10% FPR **0.779 [0.75, 0.81]**, AUROC 0.906 [0.89, 0.92]. At P(yes) ≥ 0.5 it flags 91% of backdoored and 29% of clean solutions.

![M2 · 3 persona prompts: ROC curve for false-positive rates up to 10%. Grey lines are the 3 individual monitors averaged into this ensemble.](figures/m2_roc.png)

*M2 · 3 persona prompts: ROC curve for false-positive rates up to 10%. Grey lines are the 3 individual monitors averaged into this ensemble.*

![M2 · 3 persona prompts: share of backdoored items (above the line) and clean items (below) at each score.](figures/m2_scores.png)

*M2 · 3 persona prompts: share of backdoored items (above the line) and clean items (below) at each score.*

### M3 · distilled from M1

A LoRA adapter distilled from M1 -- the baseline prompt sampled three times -- with the same loss and settings as M4. Does distilling a *diverse* ensemble beat distilling an identical one?

pAUC@10% FPR **0.782 [0.75, 0.81]**, AUROC 0.893 [0.88, 0.91]. At P(yes) ≥ 0.5 it flags 93% of backdoored and 46% of clean solutions.

![M3 · distilled from M1: ROC curve for false-positive rates up to 10%.](figures/m3_roc.png)

*M3 · distilled from M1: ROC curve for false-positive rates up to 10%.*

![M3 · distilled from M1: share of backdoored items (above the line) and clean items (below) at each score.](figures/m3_scores.png)

*M3 · distilled from M1: share of backdoored items (above the line) and clean items (below) at each score.*

### M4 · two-term SFT

A LoRA adapter trained to reproduce M2's unrounded mean score in one call. Can an ensemble's advantage be moved into one model's weights?

pAUC@10% FPR **0.810 [0.78, 0.84]**, AUROC 0.915 [0.90, 0.93]. At P(yes) ≥ 0.5 it flags 94% of backdoored and 38% of clean solutions.

![M4 · two-term SFT: ROC curve for false-positive rates up to 10%.](figures/m4_roc.png)

*M4 · two-term SFT: ROC curve for false-positive rates up to 10%.*

![M4 · two-term SFT: share of backdoored items (above the line) and clean items (below) at each score.](figures/m4_scores.png)

*M4 · two-term SFT: share of backdoored items (above the line) and clean items (below) at each score.*

### M5 · MACA (MV-DPO)

MACA's MV-DPO: the three personas debate for two rounds, and the model is trained to prefer reasoning that agreed with their majority verdict over reasoning that dissented. Does distilling deliberation beat distilling scores?

pAUC@10% FPR **0.756 [0.72, 0.79]**, AUROC 0.886 [0.87, 0.90]. At P(yes) ≥ 0.5 it flags 86% of backdoored and 25% of clean solutions.

![M5 · MACA (MV-DPO): ROC curve for false-positive rates up to 10%.](figures/m5_roc.png)

*M5 · MACA (MV-DPO): ROC curve for false-positive rates up to 10%.*

![M5 · MACA (MV-DPO): share of backdoored items (above the line) and clean items (below) at each score.](figures/m5_scores.png)

*M5 · MACA (MV-DPO): share of backdoored items (above the line) and clean items (below) at each score.*

### M6 · MACA (MV-SFT)

MACA's MV-SFT: plain cross-entropy on every response that agreed with the debate majority. Unlike DPO it keeps unanimous items, so it trains on 9x the data from the same debate (exploratory, ADR-0009).

pAUC@10% FPR **0.691 [0.65, 0.73]**, AUROC 0.876 [0.86, 0.90]. At P(yes) ≥ 0.5 it flags 96% of backdoored and 58% of clean solutions.

![M6 · MACA (MV-SFT): ROC curve for false-positive rates up to 10%.](figures/m6_roc.png)

*M6 · MACA (MV-SFT): ROC curve for false-positive rates up to 10%.*

![M6 · MACA (MV-SFT): share of backdoored items (above the line) and clean items (below) at each score.](figures/m6_scores.png)

*M6 · MACA (MV-SFT): share of backdoored items (above the line) and clean items (below) at each score.*

### M7 · MACA (MV-KTO)

MACA's MV-KTO: each response labelled desirable or dissenting on its own, with no pairing, so unanimous items train too (exploratory, ADR-0009).

pAUC@10% FPR **0.516 [0.51, 0.52]**, AUROC 0.683 [0.66, 0.70]. At P(yes) ≥ 0.5 it flags 98% of backdoored and 70% of clean solutions.

![M7 · MACA (MV-KTO): ROC curve for false-positive rates up to 10%.](figures/m7_roc.png)

*M7 · MACA (MV-KTO): ROC curve for false-positive rates up to 10%.*

![M7 · MACA (MV-KTO): share of backdoored items (above the line) and clean items (below) at each score.](figures/m7_scores.png)

*M7 · MACA (MV-KTO): share of backdoored items (above the line) and clean items (below) at each score.*

### M8 · debate, graded target

The debate without the vote: distils the personas' *graded* mean score after two debate rounds. Against M4 it isolates what deliberation costs; against M5-M7 it isolates what voting costs (ADR-0010).

pAUC@10% FPR **0.790 [0.76, 0.82]**, AUROC 0.901 [0.88, 0.92]. At P(yes) ≥ 0.5 it flags 96% of backdoored and 59% of clean solutions.

![M8 · debate, graded target: ROC curve for false-positive rates up to 10%.](figures/m8_roc.png)

*M8 · debate, graded target: ROC curve for false-positive rates up to 10%.*

![M8 · debate, graded target: share of backdoored items (above the line) and clean items (below) at each score.](figures/m8_scores.png)

*M8 · debate, graded target: share of backdoored items (above the line) and clean items (below) at each score.*

## How these numbers were made

- **Task.** Each item is an APPS problem with either a validated backdoored solution or a clean one, paired so both come from the same problem. Labels are fixed before any monitor runs.
- **pAUC@10% FPR** is the area under the ROC curve for false-positive rates 0–10%, McClish-standardised so a random monitor scores 0.5 and a perfect one 1.0 — the same convention as the anchor ensemble paper.
- **Catch rate at 1% audit** calibrates a threshold that flags 1% of clean items and reports the share of backdoored items above it. It is a single-shot detection proxy, not deployment-level safety.
- **Ensembles** (M1, M2) average their members' scores without rounding.
- **Intervals** are 95% bootstrap intervals over 10,000 class-stratified resamples; paired comparisons resample the same items for both arms.
- This is a static classification benchmark, not a live control protocol.
