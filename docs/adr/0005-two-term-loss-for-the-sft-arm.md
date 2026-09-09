---
status: accepted
date: 2026-09-09
---

# Train M4 with a two-term loss, not plain SFT

The ensemble's advantage is largely *resolution*: three monitors scoring 5, 6
and 8 produce a mean of **6.33**, a value no single monitor emitting integers can
express, and finer ranking is exactly what pAUC rewards. Plain text SFT forces
that mean to be rounded into a target sentence (`Score: 6`), deleting the signal
M4 exists to inherit. M4 is therefore trained with
`loss = CE_text + λ · KD_yes` (λ = 1): ordinary cross-entropy on a response
ending in the rounded score, plus a term pushing the student's `P("yes")` at the
verdict token toward `teacher_mean / 10`. Probabilities are continuous, so the
fractional teacher signal survives.

## Considered options

- **`CE_text` + `KD_yes` jointly** (chosen). ~20 lines over a standard QLoRA
  loop. Keeps a readable, anchor-comparable text score *and* the teacher's
  resolution.
- **Plain text SFT on the rounded mean** (rejected as the primary design,
  retained as the fallback). Free from TRL, but see Consequences — it
  pre-determines H4.
- **Soft-label KD across the `0`–`10` score tokens** (rejected). Principled, but
  `10` is two tokens in Qwen's BPE, so a clean 11-way single-position readout
  does not exist.
- **A scalar regression head with MSE** (rejected). The cleanest regression, but
  it changes the serving path and destroys the logprob readout that every arm is
  instrumented for.

## Consequences

- **H4 becomes measurable rather than assumed.** Under plain SFT the model is
  fit to 11 hard tokens, so a finding that training reduces score entropy would
  be an artifact of the loss — the model was *instructed* to be coarse and then
  reported as coarse. The KD term removes that circularity.
- **Every prompt must place a yes/no verdict token and a 0–10 score at
  deterministic positions**, because the loss has to locate the yes token.
  Prompts are frozen before generation, so this constrains §4.2 permanently.
- **Teacher means are persisted unrounded** in the M4 targets file. Rounding at
  write time would silently defeat the whole decision.
- **Fallback tied to a date:** if the joint loss is not converging on the smoke
  test by **17 Sep**, set λ = 0, ship plain text SFT, and name the rounding as a
  stated limitation. The project survives; a clean H4 does not.
