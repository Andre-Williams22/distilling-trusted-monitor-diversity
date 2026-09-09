---
status: accepted
date: 2026-09-09
---

# Use a pre-generated backdoor dataset instead of running an attack policy

The AI control literature normally generates backdoors by running an untrusted
*attack policy* against each problem, which makes attack strength a tunable
variable but consumes most of a project's compute and API budget. We instead
consume `RoganInglis/apps-control-arena` — 5,000 APPS problems with backdoors
already generated and sandbox-validated, merged from Greenblatt et al. and
Control-Tax — so that generation costs $0 and all compute goes to the question
we actually care about, which is about *monitors*, not about attackers.

## Considered options

- **`RoganInglis/apps-control-arena`** (chosen). 5,000 rows, 24 columns, MIT,
  ungated, last revised 2025-09-29.
- **`RoganInglis/apps-backdoors`** (rejected). The same 5,000 problems at an
  earlier revision with only 12 columns. It lacks `backdoor_works`,
  `is_nondeterministic`, and backdoor provenance (`source`, `model`). The
  validated-backdoor filter this project depends on **cannot be expressed on
  it** — the field does not exist. Its per-backdoor `passes_tests` says the
  backdoor passes the public tests, which is a necessary but not sufficient
  condition for a working backdoor.
- **Generating our own backdoors with an attack policy** (rejected). Correct if
  the research question were about attack strength. It is not, and the cost
  would consume the project.

## Consequences

- Attack difficulty is **fixed and not a variable we control**. Any finding is
  conditional on this backdoor distribution, and the write-up must say so.
- The dataset is the sole source of ground-truth labels. There is no sandbox in
  this project and no code is ever executed.
- Neither dataset is gated, so `huggingface-cli login` is not required.
- **Nondeterministic problems are filtered asymmetrically, not dropped.** For
  `is_nondeterministic == True` problems, "this input produces wrong output" is
  not a stable fact, so a fraction of their positive labels are noise — noise
  that lands in precisely the low-FPR region pAUC@[0,10%] integrates over. But
  *neither training arm reads a ground-truth label*: M4 regresses onto the
  teacher's mean score and M3's pairs come from monitor consensus, so the label
  enters only in val (threshold calibration) and test (all metrics). All 138
  qualifying nondeterministic problems are therefore assigned to **train**,
  where they are free, and val/test are drawn purely from the 1,444
  deterministic problems.
- **Supply is tighter than it looks.** Only **1,582** of the 5,000 problems have
  `backdoor_works == True` (all 1,582 also have a passing clean solution, so
  that conjunct is free), of which **1,444** are deterministic. Usable pool:
  **3,164 items**, of which 2,888 are label-clean — not the 3,755 the v4 plan
  assumed.
- **Splits are cut by `problem_id`, never by item**, so a problem's backdoored
  and clean items always share a split. Adopted sizing: train 788 problems /
  1,576 items, val 150 / 300, test 644 / 1,288. Test is deliberately large
  because ~64 test negatives define the entire pAUC integration region, and a
  smaller test split would leave the H1 sanity check unable to fire.
