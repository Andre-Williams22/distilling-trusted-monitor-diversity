# Distilling Trusted Monitor Diversity

A research project asking whether the detection gain an *ensemble* of trusted
monitors buys at inference time can be moved into a *single* monitor's weights
at training time, using no human labels and no stronger teacher.

## Language

### The control setting

**Trusted Monitor**:
A weak, trusted model that reads a candidate solution and emits a suspicion
score. "Trusted" means assumed non-scheming, not assumed accurate.
_Avoid_: judge, classifier, grader, critic

**Suspicion Score**:
A monitor's output on a 0–10 scale expressing how likely a solution is
backdoored. Distinct from the **Readout** used to obtain it.
_Avoid_: confidence, rating, risk score

**Readout**:
The mechanism by which a suspicion score is extracted from a monitor call —
either the *text readout* (the 0–10 integer the model writes) or the *logprob
readout* (the log-probability of a yes/no token). Every arm is instrumented
with both.
_Avoid_: output format, extraction method

**Backdoor**:
A solution that passes the problem's public tests but produces incorrect output
on a specific `backdoor_input`. Validated in the dataset, never generated here.
_Avoid_: exploit, vulnerability, trojan, attack

**Item**:
One (problem, solution, label) triple presented to a monitor for scoring. Each
qualifying problem yields exactly two items — one backdoored, one clean — and
both always live in the same split.
_Avoid_: example, sample, row, record

**Attack Rate**:
The fraction of items in a split that are backdoored. Fixed at 50% by
construction, a property of this benchmark and not of any threat model.
_Avoid_: base rate, prevalence, positive rate

### Diversity and ensembling

**Monitor Diversity**:
Decorrelation between monitors' errors. In this project diversity comes from
**persona prompts** over one shared base model, never from different model
architectures. See ADR-0002.
_Avoid_: variety, heterogeneity, disagreement

**Persona Prompt**:
One of three hand-written, frozen monitoring instructions that give the same
base model materially different framings. Written before any generation run and
never edited after seeing results.
_Avoid_: variant, template, system prompt, persona

**Baseline Prompt**:
The single frozen prompt that M0 uses and that every distilled arm is both
trained and served under. The deployment-time prompt.
_Avoid_: default prompt, standard prompt

**Ensemble**:
Three monitor calls on one item whose suspicion scores are mean-aggregated into
one score. Costs 3× inference on every action, forever.
_Avoid_: committee, panel, pool

**Arm**:
One of the six configurations under comparison (M0–M5), each producing a score
for every item in a split.
_Avoid_: condition, variant, model, run

### Distillation

**Debate**:
A two-round exchange in which the three personas score an item independently,
then revise after seeing each other's round-1 reasoning. Run on train only, and
only for the MACA arm.
_Avoid_: discussion, deliberation round, multi-agent chat

**Consensus**:
The majority verdict across the three personas' *binned* round-2 scores. Binning
is decided in advance from val agreement, never tuned after seeing results.
_Avoid_: agreement, majority, aggregate verdict

**Chosen / Rejected Trace**:
A persona's round-2 reasoning that agreed with consensus (chosen) or dissented
from it (rejected). Paired into DPO preference pairs, capped at 2–3 per item.
_Avoid_: positive/negative, winner/loser, good/bad trace

**Prompt Normalisation**:
Rewriting a preference pair's prompt field from the persona prompt it was
generated under to the baseline prompt it will be served under, so training
data matches deployment conditions.
_Avoid_: prompt rewriting, prompt swapping, re-templating

**Teacher Score**:
The mean of the three personas' single-pass suspicion scores on a train item —
the regression target for the SFT arm. No debate, no second round.
_Avoid_: soft label, target score, ground truth

**Sharpening Cost**:
The loss of score granularity that training may induce — measured as reduced
score entropy and unique-value count. A distilled monitor can become more
accurate and less discriminating at once.
_Avoid_: calibration loss, collapse, mode collapse

### Measurement

**pAUC**:
Area under the ROC curve restricted to FPR ∈ [0, 0.10], McClish-standardised so
that a random monitor scores 0.5 and a perfect one 1.0. The primary metric,
inherited from the anchor paper. Never the raw area divided by 0.10 — a
different convention sharing the same name.
_Avoid_: partial AUC, AUC@10, area under curve, normalised pAUC

**Safety at 1% Audit Budget**:
The fraction of backdoored items scoring above a threshold calibrated so that
exactly 1% of *clean* items would be flagged. A single-shot detection proxy —
explicitly *not* the trajectory-level safety of the AI control literature.
_Avoid_: safety, catch rate, recall at budget

**Operating Point**:
The inference cost at which an arm is measured — 1 sample (the headline cost
claim) or 3 samples (equal compute against the ensemble).
_Avoid_: budget, setting, configuration

**Gate**:
A pre-declared numeric threshold that must be cleared before downstream work
proceeds. Distinct from a **Kill Criterion**, which reframes the project rather
than halting it.
_Avoid_: checkpoint, milestone, go/no-go

**Promotion Rule**:
The pre-registered rule naming which hypothesis becomes the headline under which
outcome, declared before any run so that promotion is not after-the-fact
rationalisation.
_Avoid_: fallback, contingency, pivot
