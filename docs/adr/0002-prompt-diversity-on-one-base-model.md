---
status: accepted
date: 2026-09-09
---

# Source ensemble diversity from persona prompts, not from different models

The anchor paper obtained its headline +13% pAUC from an ensemble of
*architecturally different* models, and only +5.5% from three identical ones.
This project instead builds its diverse ensemble (M2) from **one base model,
`Qwen2.5-7B-Instruct`, under three hand-written persona prompts**, because a
single base model means one vLLM server, one set of weights on disk, one
download, and no re-provisioning between phases on a single 48GB A6000 — and
because the distilled student must be the *same* model as the ensemble members
for the "1× inference cost" claim to be literally true rather than confounded
by the student simply being a better model.

## Considered options

- **Three persona prompts over one base model** (chosen). Cheap, single-GPU,
  keeps the cost claim clean.
- **Three different model families** (rejected for v1). Closer to the anchor
  paper's actual result and likely a larger diversity gain, but needs three
  model downloads, sequential vLLM re-provisioning, and makes "the student is
  one of the ensemble members" impossible to state cleanly.
- **Three samples of one prompt at temperature** (this is M1, retained as a
  control, not as the diverse ensemble).

## Consequences

- **This is the project's largest scientific risk.** The anchor's +13% is not
  evidence that *prompt* diversity buys the same gain. If M2 fails to beat M1
  meaningfully, the ceiling that M5 and M4 are trying to reach mostly
  disappears, and the distillation question loses its motivation even though it
  remains answerable.
- The persona prompts are frozen before any generation run and are never edited
  after seeing results, so they cannot be tuned into a gain that isn't there.
- A cross-family ensemble remains the obvious follow-up if prompt diversity
  replicates weakly.
- **The assumption is insured, not merely asserted.** In week 2 — after the
  14 Sep spine gate, and before the 9,456 debate generations commit us — val is
  additionally scored by Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3
  under the baseline prompt, so that `mean(Qwen, Llama, Mistral)` can be
  compared against `mean(3 Qwen personas)`. Qwen's val baseline already exists
  from M0, so this costs ~600 generations plus two model downloads.
- **What the check is for.** With 14 days left it is unlikely to cause M2 to be
  redefined even if model diversity wins decisively. Its value is knowing which
  discussion section is being written — *"the ceiling is real"* versus *"the
  ceiling is soft, and here is the measurement"* — rather than having a reviewer
  raise it against no data.
- It is deliberately **not** scheduled in week 1: adding model-swapping plumbing
  before the spine gate would compete with the very deadline that protects M5.
