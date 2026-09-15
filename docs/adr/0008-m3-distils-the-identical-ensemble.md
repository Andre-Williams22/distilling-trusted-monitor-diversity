---
status: accepted
date: 2026-09-15
---

# M3 distils M1's identical ensemble, to test whether diversity is what distils

M4 is distilled from M2, a *diverse* ensemble of three persona prompts, and
beats the single monitor by 0.10 pAUC. That shows distillation works but not
that **diversity** is what gets distilled: fine-tuning on any ensemble's
judgements might do as well. M3 answers this. It is distilled from **M1**, the
baseline prompt sampled three times, with every other setting identical to M4:
the same 1,575 train items, two-term loss, aggregation, reasoning rule, adapter,
hyperparameters, seed, card type, serving and scoring. The only difference is
the teacher.

## Considered options

- **Identical-ensemble teacher (M1)** (chosen). Label-free like M4, so it stays
  in the deployment setting, and it isolates diversity.
- **Label-supervised baseline** (tried and excluded; ADR-0007). A deployed
  trusted monitor has no attack labels.
- **M4 with the KD weight at 0** (not run). Isolates the probability-matching
  term rather than diversity; a possible follow-up.

## Consequences

- **Follow-up, not preregistered.** Designed after M4's results, so M4 vs M3 is
  reported as a labelled follow-up comparison and cannot change H1-H4.
- **Reading the outcome.** M4 > M3 supports the thesis that the teacher's
  diversity is what distils. M4 ≈ M3 means self-distillation from repeated
  sampling does as well. M3 > M4 means repeated sampling is the better teacher.
- **Reuses the M3 slot** retired in ADR-0007. The label baseline's score files
  are archived in `data/generations/excluded_m3_labels/`.
