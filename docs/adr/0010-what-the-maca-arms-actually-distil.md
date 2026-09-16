---
status: accepted
date: 2026-09-16
---

# The MACA arms are capped by their target, and M8 tests the fix

M5, M6 and M7 all lost to M4. ADR-0009 predicted the cause was sparse
preference data; M6 had 8.1x M5's data and did worse, so that is falsified.
Measuring the **teachers themselves** on train explains the whole ordering:

| Teacher signal | pAUC | AUROC | mean score on clean |
|---|---|---|---|
| Round-1 mean, all 3 personas (**M4's**) | **0.800** | 0.918 | 0.306 |
| Mean of round-1 and round-2 | 0.799 | 0.918 | 0.416 |
| Round-2 mean, all 3 (**M8's**) | 0.700 | 0.888 | 0.525 |
| Round-2 mean, consensus only | 0.700 | 0.887 | 0.536 |
| Round-1 mean, consensus only | 0.655 | 0.863 | 0.456 |
| **Binary majority vote (what MACA distils)** | **0.520** | 0.710 | 0.547 |

Each arm lands near the information content of what it distils:

| Arm | What it distils | Teacher pAUC | Arm pAUC |
|---|---|---|---|
| M4 | graded ensemble mean | 0.800 | **0.810** |
| M5 (MV-DPO) | contrast between agreeing and dissenting answers | -- | 0.756 |
| M6 (MV-SFT) | consensus answer text | -- | 0.691 |
| M7 (MV-KTO) | the majority verdict, most directly | 0.520 | **0.516** |

## Two findings

**Voting is the bottleneck.** The graded ensemble mean carries 0.800 pAUC; the
majority verdict computed from the same answers carries 0.520. MACA's
objectives all optimise agreement with the discrete verdict, so they inherit
that ceiling. pAUC is a *ranking* metric and nothing in MV-SFT, MV-DPO, MV-KTO
or MV-GRPO rewards calibration. Score resolution across the trained arms falls
in lockstep with pAUC: 965 distinct scores (M4), 469 (M5), 301 (M6), 70 (M7).

**Deliberation also costs signal, separately from voting.** The graded mean
*after* two debate rounds is 0.700 against 0.800 before it, and mean score on
clean code rises 0.306 -> 0.525. Debate raised persona unanimity from 63% to
84% (ADR-0009); that agreement is systematically over-suspicious, and it
destroys the disagreement that made the ensemble worth distilling.

## M8: debate without voting

M8 distils the **round-2 graded mean across all three personas** -- the debate
is kept, the vote is dropped. With M4 it isolates deliberation (both graded,
one debated); with M5-M7 it isolates voting (all debated, one graded). Its
teacher is 0.700, so **M8 is predicted to land between M5 and M4 and not to
beat M4.** It is run to attribute the loss, not to win.

Reasoning text comes from round 1, since 73.3% of round-2 answers cite peers a
solo monitor lacks (ADR-0009). Everything else matches M4 exactly: same loss,
same lambda, same prompt, same seed, same 1x serving cost.

## The M7 diagnostic, and a disclosed deviation

M7's first run collapsed to answering "yes" almost always (70 distinct scores,
83.6% above 0.99). The table above suggests this was not a bug -- it matched
its 0.520 teacher -- but a collapse and a faithful fit are not the same claim.
Short low-learning-rate runs on a seeded subsample test which it was:

- If scores stay degenerate at a low learning rate, the objective is
  reproducing a near-uninformative target: **not an implementation fault.**
- If scores spread out and pAUC rises materially, the first run was an
  **optimisation failure**, and M7's reported number was mine, not MV-KTO's.

### Diagnostic result: not an implementation fault

Val split, 300 items:

| Config | pAUC | AUROC | distinct | frac > 0.99 | mean on clean |
|---|---|---|---|---|---|
| Original (lr 1e-4, w 4.12, 4,713 ex, 2 ep) | 0.519 | 0.707 | 18 | 0.797 | 0.630 |
| A: low lr **2e-5** (w 4.46, 1,500 ex, 1 ep) | 0.529 | 0.752 | 54 | 0.823 | 0.691 |
| B: **no reweighting** (lr 1e-4, 1,500 ex, 1 ep) | 0.515 | 0.676 | 22 | 0.813 | 0.664 |

A five-fold lower learning rate moves pAUC by +0.010 and leaves the scores just
as saturated; removing the class reweighting is slightly worse. Every
configuration sits at the binary vote's **0.520** ceiling. **MV-KTO reproduced
its target faithfully; the collapse is the objective doing its job on an
uninformative signal, not a defect in this implementation.**

Consequently **M7's reported test pAUC of 0.516 stands unchanged.** No
configuration earned a test re-score, so the post-hoc selection disclosed above
never actually fired -- nothing about M7 in the results is tuned.

### A second implementation fault, found and fixed

The first diagnostic attempt was void: `learning_rate` was computed into the run
metadata but the **default** was passed to the training loop, and `--tag` never
reached the run directory, so the second config overwrote the first. Config A
had therefore run at 1e-4 while its metadata would have claimed 2e-5 -- a run
recording a hyperparameter it never used. Fixed in `f328435`, with a test
asserting overrides reach both the loop and the run name. The numbers above come
from the corrected re-run, whose metadata was checked against the per-step
learning rates in `history.jsonl`.

**Disclosed deviation:** the diagnostic's configuration is chosen on **val**,
and only the chosen one is scored on test. That is post-hoc selection after
seeing M7 fail, so M7's updated number is *not* preregistered and must be
reported as a tuned result. No other arm's hyperparameters were selected this
way; M0-M6 used the shared settings throughout.


## Numbering, and what is not claimed

The debate-graded arm was **renumbered M9 -> M8 on 16 Sep** so the reported arms
run M0-M8 with no gap, exactly as the MACA arm was renumbered M3 -> M5 on
15 Sep. Anything dated before 16 Sep that calls this arm "M9" means today's M8.
The scored artifacts were renamed with it; the adapter weights are unchanged.

**MV-GRPO holds no arm number because it was never run.** Its trainer
(``src/train_grpo.py``) and runner (``infra/vast/run_mv_grpo.sh``) are committed
and unit-tested, and a run needs only GPU time. The write-up must say MACA has
four objectives, that **three were run** (MV-SFT, MV-DPO, MV-KTO) and that
MV-GRPO was implemented but not executed. Closing the numbering gap is a
presentation choice; it must never read as four objectives evaluated.

## Result, recorded after the run

M8 scored **pAUC 0.790 [0.760, 0.820]** on test, between M5 and M4 as predicted.
That separates MACA's two ingredients on identical debate data:

- **voting costs** M8 - M5 +0.034 · M8 - M6 +0.099 · M8 - M7 +0.274
- **deliberation costs** M8 - M4 -0.020 [-0.044, +0.005], DeLong p 0.022

Both graded students exceed their teachers (M4 0.810 from 0.800, M8 0.790 from
0.700); every voted arm lands at or below its 0.520 target.
