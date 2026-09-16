---
status: accepted
date: 2026-09-16
---

# The three remaining MACA objectives run as a secondary, exploratory family

MACA (arXiv:2509.15172) evaluates four objectives on one debate: MV-SFT,
MV-DPO, MV-GRPO and MV-KTO. M5 implemented MV-DPO (ADR-0003) and lost to M4's
score distillation on test: **ΔpAUC -0.054 [-0.090, -0.019], DeLong
p = 6.3e-05**. The other three objectives train on the **same debate
transcripts already generated**, so replicating them costs GPU time for
training and scoring only -- no new generation.

They are added as **M6 (MV-SFT)** and **M7 (MV-KTO)**. MV-GRPO was implemented
and unit-tested but **never run**, so it carries no arm number; see ADR-0010.

## This is registered before the runs, not after

These arms were chosen **after** seeing H3 fail. That is exactly the shape of a
garden-of-forking-paths error, so the rules are fixed here, in advance:

- **M0-M5 remain the preregistered primary analysis.** The headline claim of
  the project and H1-H4 are settled by those six arms and do not change.
- **M6-M8 are a secondary, exploratory family**, reported as such. Their
  pairwise comparisons carry **Holm-corrected p-values across the family**, and
  any uncorrected p-value is labelled uncorrected.
- **The prediction is stated now, before any of them is trained** (below). It is
  a falsifiable mechanistic claim, not a search for an arm that wins.

## The prediction

The debate raised persona unanimity from 63.2% to 84.2%, and only disagreement
yields preference pairs. MV-DPO therefore trained on **480 pairs from 240 of
1,574 items**, while the other objectives keep unanimous items:

| Objective | Signal from the same debate | vs MV-DPO |
|---|---|---|
| MV-DPO (M5) | 480 pairs | -- |
| MV-SFT (M6) | 4,468 agreeing responses | 9.3x |
| MV-KTO (M7) | 4,708 labelled responses (4,468 desirable / 240 undesirable) | 9.8x |
| MV-GRPO (no arm; unrun) | 1,571 prompts with a majority verdict as reward | rollout-bound |

**If M5's loss was caused by sparse preference data, M6 and M7 beat M5.**
(Registered before the runs. It was **falsified**: see ADR-0010.) If
instead the debate signal itself is weak, or the no-context training choice is
what costs, M6 and M7 land at or below M5 despite ~9x the data.

Note this **predicts against the paper at this model size**: MACA reports
MV-DPO best at 4-8B, so a faithful replication expects M6 and M7 to be worse
than M5. Both outcomes are informative -- agreement replicates MACA, and
disagreement shows the ranking depends on how much disagreement the debate
leaves behind.

## The debate text mentions peers; the deployed monitor has none

Measured while building M6: **73.3% of round-2 debate responses refer to the
other reviewers** ("all reviewers agree", "as Reviewer B notes"). These arms are
served **alone** under the baseline prompt, so training on that text teaches the
model to cite reviewers that do not exist at serve time.

So the vote comes from the debate, but the **text** comes from round 1 -- each
persona's independent answer, kept when its own round-1 verdict matches the
post-debate majority, so text and verdict never contradict each other. The
mechanism under test is untouched: **the majority vote still decides whose
reasoning was right.**

| | Round-2 text | Round-1 text (default) |
|---|---|---|
| Responses mentioning peers | 73.3% | **0.5%** |
| Desirable examples | 4,468 | 3,901 |
| MV-KTO class ratio | 18.6:1 | **4.8:1** |

``--text-source round2`` reproduces the other convention; both are recorded in
each run's metadata.

**This also bears on M5.** 63% of MV-DPO's chosen responses mention peers, so
M5 was trained toward text it cannot justify when served alone. Because chosen
and rejected mention peers at nearly the same rate (63.1% vs 63.8%), the
*preference* signal is largely unaffected, but the effect on M5 is not zero and
is now a fourth candidate explanation for its result. An M5 re-run with
``--text-source round1`` is the clean ablation, and is cheap (480 pairs).

## Objective-specific decisions

- **M6 (MV-SFT):** plain cross-entropy on the majority-winning response text
  (round-1 text by default, per the section above),
  with no KD term. The KD term is M4's (ADR-0005); dropping it keeps M6-vs-M4 a
  clean contrast of *target selection* -- vote-chosen text against mean-score
  distillation -- at identical cost, prompt and base model. All agreeing
  responses are used, not one per item, as the paper trains on the winning
  traces.
- **M7 (MV-KTO):** desirable = agrees with the majority, undesirable = dissents.
  The classes are **18.6:1 imbalanced**, far outside KTO's recommended
  desirable:undesirable working range, so the undesirable weight is raised to
  bring the effective ratio into that range. That reweighting is a documented
  deviation and a candidate explanation for any M7 result.
- **MV-GRPO (unrun):** reward would be whether a sampled completion's written verdict
  matches the **frozen** majority verdict from the debate already run, so no
  re-debate is needed; group-normalised advantage with a KL penalty to the
  adapter-disabled reference. It is run **last** because its rollout loop is the
  only part that needs generation and training on the GPU at once.

## Consequences

- The all-arms ROC plot is crowded past six arms; the exploratory arms are
  labelled as such in the report rather than hidden.
- Every arm still scores through the same vLLM path at 1x inference cost, so the
  cost axis of the frontier plot is unchanged.
- GRPO did not fit the schedule. It is reported as **not attempted**, never as a
  null result, and holds no arm number so no figure implies a missing run.
