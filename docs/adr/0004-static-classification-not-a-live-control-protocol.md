---
status: accepted
date: 2026-09-09
---

# Evaluate as static classification, not inside a live control protocol

Despite being an AI-control project, this evaluates monitors as **classifiers
over a fixed, pre-labelled item set**: every label exists in the dataset before
any monitor runs, no attack policy acts during evaluation, and no
audit/defer/resample decision is ever executed against a live task. The primary
metric is **pAUC at FPR ∈ [0, 10%]**, normalised. This matches how the anchor
ensemble paper evaluates, keeps every arm comparable on identical items
(so every contrast is paired), and removes a deferral-threshold confound that
would consume time without changing the answer to the research question.

## Considered options

- **Static classification, pAUC@[0,10%]** (chosen), **McClish-standardised**
  and pinned by test to `sklearn.metrics.roc_auc_score(..., max_fpr=0.10)`.
  Two conventions share the name "normalised pAUC" — McClish puts a random
  monitor at 0.5, dividing the raw area by `max_fpr` puts it at 0.05 — and the
  anchor paper's 0.7226 is a McClish figure. Picking the wrong one silently
  rescales every comparison in the project while producing plausible-looking
  numbers, so the convention is fixed here rather than left to the
  implementation.
- **TPR at a single low FPR** (rejected as primary). At this sample size it is
  dominated by a handful of negatives and its confidence interval is enormous.
  Retained as the secondary "safety at 1% audit budget" readout.
- **Full trajectory-level control simulation** (rejected). The real
  deployment-level safety number from Greenblatt et al., requiring episode
  simulation under an attack strategy. Out of scope for 24 days.

## Consequences

- **"Safety at 1% audit budget" is a proxy and must be named as one.** It is a
  single-shot detection rate at an audit-calibrated threshold, not the
  probability a blue team catches an attack across a deployment.
- Because no monitor is ever invoked inside a ControlArena protocol,
  **`control-arena` is not installed at all** — only its dataset is used, pulled
  straight from Hugging Face with `datasets.load_dataset`. The scoring loop is
  plain vLLM calls over JSONL. This is not merely tidiness: `control-arena`
  19.0.0 pulls **33 dependencies** including `docker`, `kubernetes-asyncio`,
  `inspect-k8s-sandbox`, `minio`, `redis==5.2.1`, `psycopg` and `flask` — none
  of which this project runs — and pins `huggingface-hub~=0.33`, an upper bound
  that would sit in the same environment as `vllm`, `transformers`, `trl` and
  `peft` and invite a resolver conflict on the GPU box. Keeping the environment
  small also means it can be rebuilt in minutes when an interruptible instance
  dies mid-generation.
- The README's claim is scoped to match: *"the ControlArena APPS backdoor
  dataset"*, not *"ControlArena's APPS setting"* — a smaller claim that is
  actually true and that cannot be punctured by asking which protocol was run.
- All arms score identical items, so DeLong (correlated AUROCs) and paired
  bootstrap (pAUC) are the correct tests throughout.
- pAUC@10% leans on the *negatives* in the test split. With a realistic item
  supply near 2,888 the test half holds only a few hundred clean items, of which
  ~10% define the integration region — the test split should be sized to protect
  that region, not split by habit.
