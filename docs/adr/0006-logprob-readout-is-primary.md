---
status: accepted
date: 2026-09-09
---

# The logprob readout is primary; the text score is the anchor bridge

Every monitor call yields two suspicion scores: the `0–10` integer the model
writes (**text readout**) and `P("yes")` at the verdict token (**logprob
readout**). The v4 plan made text primary and held logprob in reserve "if DPO
collapses the text score." That is inverted, and it is now fixed the other way
round **before any run**: logprob decides H1–H4, text is reported alongside as
the comparability bridge to the anchor paper.

## Considered options

- **Logprob primary, text secondary** (chosen).
- **Text primary, logprob as a rescue** (rejected — the v4 position).
- **Text only** (rejected). Cheapest to implement, but forfeits all resolution.

## Consequences

- **Ties stop dominating the metric.** A 7B monitor concentrates its written
  scores on a handful of values (typically 0/3/5/8), so ranking 1,288 test items
  on ~6 distinct levels produces heavy ties — worst precisely at the low-FPR end
  where pAUC@[0,10%] integrates, and DeLong is unreliable on heavily tied data.
  `P("yes")` is continuous and has no such ceiling.
- **It matches where M4's signal goes.** Under ADR-0005 the teacher's fractional
  mean is carried by the `KD_yes` term into `P("yes")`. Judging M4 primarily by
  its text score would report the one number that signal deliberately bypassed.
- **The choice is unimpeachable only because it is made in advance.** Selecting a
  readout after seeing which one flatters the result is fishing; pre-registering
  it costs nothing.
- **The anchor comparison demotes to a secondary table.** The anchor paper used
  text scores, so "our M0 vs their 0.7226" now sits beside the headline rather
  than in it. Acceptable: the anchor ran GPT-4.1-Mini, so that comparison was
  never apples-to-apples.
- **Both readouts are recorded for every arm on every item**, so nothing is lost
  and either can be reported in full.
