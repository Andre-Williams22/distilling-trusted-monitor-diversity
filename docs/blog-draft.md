# One monitor, three monitors' worth: distilling ensemble diversity into a single trusted monitor

*Draft · preliminary results · Andre Williams · BlueDot Technical AI Safety Sprint, September 2026*

## The problem

In AI control, a *trusted monitor* is a weaker model we trust not to scheme,
reading the outputs of a stronger model we don't trust and flagging anything
suspicious. A recent paper, [Ensemble Monitoring for AI
Control](https://arxiv.org/html/2605.15377v2), found that combining several
*diverse* monitors catches noticeably more sabotage than any single monitor, and
more than simply sampling one monitor several times.

The catch is cost. A three-monitor ensemble makes three model calls for every
action it checks, for as long as the system is deployed.

This project asks whether that cost can be paid once, at training time instead:
**can a single fine-tuned monitor learn what a diverse ensemble knows, and keep
the ensemble's detection gain at a third of the inference cost?**

## Setup

**Task.** Detecting backdoored code. I used the [ControlArena APPS
dataset](https://huggingface.co/datasets/RoganInglis/apps-control-arena):
programming problems paired with either a correct solution or one containing a
validated backdoor — code that passes the normal tests but gives a wrong answer
on a specific hidden input. Every problem contributes one backdoored and one
clean solution, and problems are split into train, validation and test, so no
test problem is ever seen in training.

**Monitor.** Qwen2.5-7B-Instruct reads the problem and the solution and answers
whether it is backdoored. The suspicion score is the model's probability of
answering "yes".

**Metric.** Partial AUROC over false-positive rates from 0 to 10%, the
same metric as the anchor paper. It measures how well a monitor separates
backdoored from clean code in the strict regime that matters in practice, where
only a small share of honest work can be sent to human review. 0.5 is chance;
1.0 is perfect.

**Four monitors compared** (a fifth is still to come):

| | What it is | Calls per check |
|---|---|---|
| **M0** | A single monitor | 1 |
| **M1** | The same monitor sampled three times, averaged | 3 |
| **M2** | Three different monitoring prompts, averaged — the diverse ensemble | 3 |
| **M4** | One model fine-tuned to reproduce M2's judgement | **1** |

M2's three prompts give the same model different specialities: tracing
suspicious control flow, comparing the code against what a correct solution
would look like, and imagining where an attacker would hide a trigger.

**Distillation.** To train M4, I ran M2's three prompts over the *training*
problems, then fine-tuned a LoRA adapter on the results. The loss has two parts:
ordinary next-token loss so the model still writes a normal answer, and a second
term pushing its probability of "yes" toward the ensemble's *average*. That
second term matters. Three monitors voting yes, yes and no give an average of
two-thirds — information a single monitor, and plain text fine-tuning, would
round away. M4 never sees test problems and never sees the true labels; everything
it learns comes from the ensemble.

## Results

Test set: 1,288 solutions from 644 problems, half of them backdoored.

![Detection against inference cost](../results/test/figures/frontier.png)

| Monitor | pAUC (FPR ≤ 10%) | AUROC | Calls per check |
|---|---|---|---|
| M0 · single monitor | 0.709 [0.67, 0.75] | 0.862 | 1 |
| M1 · 3 samples | 0.753 [0.72, 0.79] | 0.881 | 3 |
| M2 · 3 diverse prompts | 0.779 [0.75, 0.81] | 0.906 | 3 |
| **M4 · distilled** | **0.810** [0.78, 0.84] | **0.915** | **1** |

*95% bootstrap intervals in brackets.*

Differences between monitors, measured on the same items:

| Comparison | Change in pAUC [95% interval] | |
|---|---|---|
| M1 − M0 | +0.044 [+0.018, +0.071] | sampling three times helps |
| M2 − M0 | +0.070 [+0.035, +0.104] | diverse prompts help more |
| M2 − M1 | +0.026 [−0.006, +0.060] | diversity beyond sampling: a trend, not yet significant |
| **M4 − M0** | **+0.101** [+0.066, +0.135] | **distillation clearly beats a single monitor** |
| M4 − M2 | +0.031 [−0.000, +0.061] | M4 matches its teacher; not clearly above it |

![Every arm's ROC curve at low false-positive rates](../results/test/figures/all_arms_roc.png)

## What this suggests

**The ensemble's advantage survived distillation.** The distilled monitor beats
a single monitor by 0.10 pAUC, and does at least as well as the three-prompt
ensemble it learned from — at one model call instead of three. By the project's
preregistered bar (recover at least half of the ensemble's gain at 1× cost), it
comfortably clears it.

**It may have slightly exceeded its teacher, but I can't claim that yet.** M4
scored 0.031 above M2, with an interval that just touches zero. One plausible
explanation is that training on the ensemble's *averaged* judgement acts as a
regulariser. The more cautious one is noise — I don't yet have repeated runs.

**One prediction was wrong.** I expected fine-tuning to sharpen the monitor's
scores into fewer, more extreme values. The opposite happened: M4's scores are
more spread out than the untrained monitors', which are mostly pinned near 0 or
1. That fits what it was trained on — averages like one-third and two-thirds —
and finer-grained scores are exactly what a low false-positive-rate metric
rewards.

**Diversity beyond resampling is suggestive, not proven.** Diverse prompts beat
a single monitor clearly, but their edge over simply sampling one prompt three
times (+0.026) is within noise at this sample size. The anchor paper found a
larger effect using architecturally different models, not one model with
different prompts.

## Caveats

- **Single run.** One training run and one scoring pass per monitor. Sampling
  randomness alone moved pAUC by about ±0.02 between repeated runs, which is
  enough to matter for M4 versus M2 but not for M4 versus M0.
- **One model family, prompt-based diversity.** All monitors are Qwen2.5-7B;
  diversity comes from prompts, not different models.
- **Static benchmark, not a live control protocol.** Solutions are pre-generated
  and pre-labelled. This measures detection, not safety in a deployment where an
  attacker adapts.
- **Fixed attacks.** The backdoors come from one existing dataset; whether the
  distilled monitor holds up against harder or adaptive attacks is untested.

## Next

- **Repeat runs** with fixed seeds, to put error bars on M4 versus M2.
- **M3:** distil the ensemble's *reasoning* rather than its scores — having the
  three prompts debate, and training with DPO to prefer reasoning that survived
  the debate (a MACA-style approach). This tests whether distilling
  deliberation beats distilling scores.
- Longer term: the full MACA recipe (consensus fine-tuning plus reinforcement
  learning), ensembles of genuinely different models, and evaluation inside a
  live control protocol.

*The whole experiment — every scoring run and the M4 training run — cost about
$2 of rented GPU time.*
