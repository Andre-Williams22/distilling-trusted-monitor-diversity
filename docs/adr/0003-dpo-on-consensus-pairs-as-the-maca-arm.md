---
status: accepted
date: 2026-09-09
---

# Implement the "MACA" arm as DPO on debate-consensus preference pairs

MACA as published (arXiv:2509.15172) is **MV-SFT plus MV-GRPO** — supervised
fine-tuning on majority-vote-consistent trajectories, followed by reinforcement
learning with a consensus reward. This project implements neither. M3 instead
runs a two-round debate between the three personas, bins the round-2 scores,
takes the majority as consensus, splits traces into agree-with-majority
(chosen) and dissent (rejected), and trains **QLoRA DPO (β = 0.1)** on those
pairs. DPO reaches the same objective — prefer reasoning that survives peer
scrutiny — without a rollout loop, a reward model, or the GRPO
infrastructure that would not fit in a 24-day sprint on one GPU.

## Considered options

- **DPO on (chosen, rejected) consensus pairs** (chosen). One offline training
  run, no rollouts, standard TRL support.
- **Faithful MV-SFT + MV-GRPO** (rejected). The published method, but GRPO needs
  an online rollout loop and substantially more GPU-hours and debugging time
  than the budget allows.
- **MV-SFT alone** (rejected). Cheapest, but reduces to "SFT on a filtered
  subset", which collapses the contrast against M4 — the two arms would differ
  only in which data they see, not in what they optimise.

## Consequences

- **Naming honesty is a hard requirement.** The write-up must call this
  *"DPO on MACA-style consensus-derived preference pairs"*, never "MACA". A
  reader who checks the MACA paper will find no DPO in it, and an unqualified
  claim invites the reviewer's first and most damaging objection.
- H3 ("M3 beats M4") therefore tests *this adaptation*, not MACA itself. A null
  result is evidence about DPO-on-consensus, and is **not** evidence that MACA
  fails on monitoring.
- Binary consensus binning carries **no information about score magnitude**, so
  M3 can improve the ranking metric only through better reasoning, never through
  better calibration. This is a direct mechanism for the sharpening cost that H4
  predicts.
- Preference pairs are generated under persona prompts but trained under the
  baseline prompt (**prompt normalisation**), so chosen traces may carry persona
  phrasing into a model that will never again see a persona framing.
- **The two substituted stages are deferred, not rejected.** MV-SFT and
  MV-GRPO are recorded as future extensions in project-plan.md section 10 and
  are to be named in the write-up, so that readers can see which questions were
  deferred rather than answered. MV-SFT in particular would separate something
  H3 currently conflates: whether the value of debate lies in *filtering* to
  what monitors agreed on, or in the *contrast* with what they abandoned.
