---
status: proposed
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
  meaningfully, the ceiling that M3 and M4 are trying to reach mostly
  disappears, and the distillation question loses its motivation even though it
  remains answerable.
- The persona prompts are frozen before any generation run and are never edited
  after seeing results, so they cannot be tuned into a gain that isn't there.
- A cross-family ensemble remains the obvious follow-up if prompt diversity
  replicates weakly.
