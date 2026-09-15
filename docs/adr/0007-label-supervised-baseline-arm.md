---
status: accepted
date: 2026-09-15
---

# Add M3, a baseline fine-tuned on true labels, and renumber the DPO arm to M5

M4 is fine-tuned on the ensemble's judgements and never sees a ground-truth
label. Its test result (pAUC 0.810 vs M0's 0.709) shows distillation helps, but
not whether *the ensemble* is what helps: any fine-tuning on this task might do
as well. M3 answers that. It is the same base model, adapter, prompt, training
split and serving path as M4, fine-tuned instead on the training split's true
backdoored/clean labels. The MACA-style DPO arm, which was M3, becomes M5 so the
two trained baselines sit next to each other in run order.

## Considered options

- **Label-supervised baseline as M3** (chosen). The most direct control for M4:
  change only where the training signal comes from.
- **No baseline** (rejected). Leaves "distillation beats the single monitor"
  unable to distinguish "the ensemble taught it something" from "fine-tuning
  helps".
- **Keep the DPO arm as M3 and call the baseline M5** (rejected at the user's
  request, so the numbering reads baseline → distillation → deliberation).

## Consequences

- **M3 was added after M4's test results were known.** It is reported as a
  reference comparison, labelled as added after preregistration, and cannot
  change the status or framing of H1–H4.
- **M3 is the only arm that reads ground-truth labels in training.** That makes
  it a strong baseline rather than a like-for-like competitor: M4 matching or
  beating it would mean ensemble supervision is worth as much as true labels
  without needing any.
- **Renumbering.** Every "M3" in material dated before 15 Sep means today's M5.
  Committed files were updated; git history keeps the old name.
- **Open design choices** for M3's training target, to settle before it is
  trained: what the response's REASONING sentence contains (M4 borrows a
  persona's, which would leak ensemble information into a label baseline), and
  whether the KD term's target is the hard label (0 or 1).
