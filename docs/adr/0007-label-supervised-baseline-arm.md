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
- **Training design, settled 15 Sep, so that only the training signal differs
  from M4:**
  - *Target:* verdict and score from the label (`yes`/10 or `no`/0), and a KD
    target of exactly 1.0 or 0.0.
  - *Reasoning sentence:* the untrained model's own, taken from its M1-style
    answers on train (baseline prompt, three samples), chosen by M4's rule: a
    sample agreeing with the target, the most confident first. M4's persona
    sentences would leak ensemble information into a label baseline. A base
    sample agrees with the label on about 90% of items; on the rest the
    reasoning argues against the verdict, which is recorded with the run.
  - *Items:* deterministic train problems only. Nondeterministic problems went to
    train because no arm read labels (ADR-0001); M3 does, so it excludes them.
    M3 therefore trains on roughly 1,300 items against M4's 1,575.
  - *Everything else* -- tokenisation (shared `tokenise_example`), two-term loss,
    LoRA rank, learning rate, epochs, seed, serving and scoring -- is identical.
- **Hard 0/1 targets may saturate M3's scores.** Ties at 0 and 1 cost pAUC. If M3
  comes out low, check score spread before concluding labels are weak.
