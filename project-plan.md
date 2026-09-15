# Distilling Monitor Diversity

**Can one trusted monitor internalize what an ensemble buys — without labels or a stronger teacher?**

**Andre Williams · BlueDot Technical AI Safety Sprint · v6, 15 Sep 2026**
**13 days remain (15 Sep → 28 Sep)**

**What changed in v6 (15 Sep):** the MACA-style DPO arm, previously M3, is
renumbered **M5** throughout. A label-supervised baseline was added as M3,
trained, and then excluded from the analysis because deployed trusted monitors
have no attack labels (ADR-0007, which records its result).

**What changed in v5:** dataset identity pinned and its real supply measured
against the live dataset (§2) — nondeterministic problems now filtered out, and
the item pool corrected downward from 3,755 to 2,888. Decisions recorded as ADRs
in `docs/adr/`; project vocabulary fixed in `CONTEXT.md`.

**What changed in v4:** scoped down to the minimum design that answers the
research question. Dropped: the homogeneous-MACA control arm, the cross-lineage
stretch goal, the KTO training objective, the 8-variant prompt search and its
dependency on an external email, and the full GSM8K reproduction gate. Kept and
sharpened: the supervised-distillation arm, now framed as its own question —
**is distilling an ensemble worth doing at all here, by any method** — not just
a baseline for MACA to beat.

---

## 0. Working agreement

**Andre writes the science; Claude writes the scaffolding and owns the infra.**

| Area | Owner |
|---|---|
| Training loops (M5 DPO, M4 SFT) | **Andre** — Claude suggests and reviews, never authors |
| Data preprocessing, splits, item construction | **Andre** |
| Artifact handling (JSONL, checkpoints, adapters) | **Andre** |
| Metrics and statistics | **Andre** — a silent bug here invalidates every number |
| Skeletons: signatures, type hints, docstrings | **Claude** — bodies are `raise NotImplementedError` |
| vast.ai provisioning, SSH, remote env, run orchestration | **Claude, outright** |
| `project-plan.md`, `CONTEXT.md`, `docs/adr/` | **Claude** |

A skeleton's **docstring is the contract**: Andre codes against it, and Claude
reviews the finished body against it. Disagreements about behaviour get settled
in the docstring first, then in the code.

### Repository layout

```
main.py              # stdlib argparse subparsers — the single entrypoint.
                     #   Contains argparse and function calls, nothing else.
                     #   If a subcommand needs an `if`, that `if` lives in src/.
pyproject.toml       # [data] extra for the arm64 Mac, [gpu] extra for the A6000
src/                 # flat package (__init__.py present so `from src.x import y` works)
  config.py          #   frozen dataclasses: paths, model ids, seeds, split sizes
  utils.py           #   jsonl io, hashing, seeding, logging
  data.py            #   load -> filter -> pair -> split
  scoring.py         #   vLLM client + text/logprob readouts
  debate.py          #   two rounds, consensus, pairs, prompt normalisation
  train_sft.py       #   M4
  train_dpo.py       #   M5
  metrics.py         #   pAUC, safety@1%, entropy, unique-value count
  stats.py           #   DeLong, paired bootstrap
  analysis.py        #   summary/comparison tables, hypotheses, report.md
  figures.py         #   per-arm ROC + score charts, all-arms ROC, frontier
prompts/             # baseline.txt, persona_*.txt — versioned plain text
tests/               # run on the Mac, no GPU required
```

**Artifacts are split by lifetime, not lumped into one bucket:**

| Directory | Git | Lives | Holds |
|---|---|---|---|
| `data/` | ignored | GPU box | splits, raw generations, derived training files |
| `runs/` | ignored | GPU box | one dir per run: `DATE__arm__githash/` with adapter, checkpoints, logs, resolved config |
| `results/<split>/` | **committed** | repo | `report.md`, summary/comparison/hypothesis CSVs, `figures/*.png` |

`results/` is committed on purpose: it is a few hundred KB, it is what the
write-up cites, and keeping it in history makes every reported number
timestamped and attributable to a commit. Only `results/` travels back over
SSH — `data/` and `runs/` die with the instance.

**QLoRA saves adapters, never model weights.** The 4-bit 7B base is frozen and
re-pulled from HF on every box, so both trained arms together produce well under
200 MB of output. Run directories are keyed `DATE__arm__githash` because M4 will
be trained four or five times while debugging, and *"which adapter produced the
numbers in the draft?"* always gets asked on the last day.

---

## 1. The question

> **Can inference-time monitor diversity be distilled into a single trusted monitor — recovering most of an ensemble's detection gain at 1× inference cost, using no labels and no stronger teacher?**

[Ensemble Monitoring for AI Control](https://arxiv.org/html/2605.15377v2) (May
2026) showed diversity beats compute: three *identical* GPT-4.1-Mini monitors buy
+5.5% pAUC (0.7226 → 0.7621); three *diverse* monitors buy +13% (→ 0.8160). But an
ensemble costs 3× inference on every action, forever. This project pays that cost
once, at training time.

**One line:** *An ensemble costs 3× inference forever. We pay it once at training
time and ship one monitor that keeps the gain.*

Two distillation methods are tested side by side, both consuming the same
ensemble outputs:

- **MACA** distills the ensemble's **deliberation** — which reasoning survived a
  debate between the ensemble members, and which got abandoned.
- **Supervised fine-tuning** distills the ensemble's **scores** — a direct
  regression onto the ensemble's mean output.

This answers two questions at once: whether distillation recovers the ensemble's
gain **at all**, and, if so, whether MACA's extra machinery over plain SFT is
earning its cost.

---

## 2. Dataset

**[`RoganInglis/apps-control-arena`](https://huggingface.co/datasets/RoganInglis/apps-control-arena)**
— ControlArena's [APPS setting](https://control-arena.aisi.org.uk/settings/apps.html)
dataset. 5,000 APPS problems with pre-generated, sandbox-validated backdoors,
merged from [Greenblatt et al.](https://arxiv.org/pdf/2312.06942) and
[Control-Tax](https://arxiv.org/html/2506.05296v1). **No backdoor generation, no
policy model invoked, $0 API cost.**

**Not [`RoganInglis/apps-backdoors`.](https://huggingface.co/datasets/RoganInglis/apps-backdoors)**
That is an earlier revision of the same 5,000 problems with 12 columns instead of
24. It has no `backdoor_works` column, so the validated-backdoor filter below
**cannot be expressed on it at all**, and no `is_nondeterministic` column either.
Neither dataset is gated — `huggingface-cli login` is not required. See ADR-0001.

**Item construction — three filters, then pair:**

1. `backdoor_works == True` — the backdoor is validated to actually misbehave.
2. `solution_passes_tests == True` — a validated clean solution exists.
3. `is_nondeterministic == False` — **added in v5.** For nondeterministic
   problems, "this input produces wrong output" is not a stable fact, so a
   fraction of their positive labels are noise. Label noise concentrates exactly
   in the low-FPR region that pAUC@[0,10%] integrates over, so 2,888 clean items
   beat 3,164 noisy ones.

Then pair each backdoor with the clean solution to the *same* problem — otherwise
the monitor can learn "hard problem → suspicious" instead of detecting backdoors.

**Measured supply (queried against the live dataset, 9 Sep):**

| Filter | Problems |
|---|---|
| all rows | 5,000 |
| `backdoor_works == True` | 1,582 |
|  + `solution_passes_tests == True` | 1,582 — *free, every one already passes* |
|  + `is_nondeterministic == False` | **1,444** |

At two items per problem (one backdoored, one clean) that is **2,888 items** —
not the 3,755 the v4 split table assumed, a **23% shortfall**.

**Splits are cut by `problem_id`, never by item.** Both members of a pair always
land in the same split; splitting by item would leak a problem's clean twin into
test alongside its backdoor.

**The nondeterminism filter is asymmetric — this follows from "without labels."**
Neither training arm ever reads a ground-truth label: M4 regresses onto the
*teacher's mean score*, and M5's preference pairs come from *monitor consensus*.
The `backdoored/clean` flag enters only in val (threshold calibration) and test
(every metric). Label noise in train therefore cannot reach the training signal,
so all **138** nondeterministic-but-validated problems are assigned to train,
where they are free, while val and test are drawn purely from the 1,444
deterministic problems.

| Split | Problems | Items | Nondet? | Use |
|---|---|---|---|---|
| train | **788** (650 det + 138 nondet) | **1,576** | allowed | debate transcripts (M5), teacher scores (M4) |
| val | **150** | **300** | excluded | thresholds, agreement-rate check |
| test | **644** | **1,288** | excluded | all primary results |
| **total** | **1,582** | **3,164** | | |

**Why test is large.** pAUC@[0,10%] integrates over the *clean* items in test
only, so roughly 10% of the 644 test negatives — about 64 items — define the
entire metric region. Expected effects are +0.03 to +0.09 pAUC against a
bootstrap CI of roughly ±0.05–0.08 at this size. A train-heavy cut (394 test
problems, ~39 negatives) would very likely return H1 indistinguishable from
zero, leaving no way to separate *"the setup is broken"* from *"the effect is
small"* — and if H1 cannot fire, nothing validates the setup before H2 is
trusted. 1,576 training items is ample for QLoRA on a narrow scoring task
(~3,900 DPO pairs at 2–3 per item), so the trade goes to the metric.

**Be clear-eyed:** at 1,444 deterministic problems, *no* split makes the headline
comfortable. Option A maximises the chance the sanity check is readable; it does
not buy a wide margin anywhere.

50% attack rate throughout by construction, matching the anchor paper's shape.

---

## 3. The arms

All arms use one base model, **`Qwen2.5-7B-Instruct`**, served with vLLM.
Diversity comes from prompts and training, never architecture. **Not 2B:** MACA's
own stated limitation is that it requires baseline competence — a near-chance
monitor has no consensus signal to internalize.

| Arm | What | Cost | Expected pAUC vs M0 |
|---|---|---|---|
| **M0** | Single baseline monitor | 1× | floor — anchor was 0.7226 on GPT-4.1-Mini; expect ~0.62–0.72 on Qwen-7B |
| **M1** | 3× identical, mean-aggregated | 3× | +0.03 to +0.04 — the free win any trained arm must beat |
| **M2** | 3 fixed, distinct persona prompts, mean-aggregated | 3× | +0.07 to +0.09 — the ceiling |
| ~~M3~~ | ~~Fine-tuned on true labels~~ — **trained, then excluded from the analysis** (ADR-0007) | 1× | a deployed trusted monitor has no attack labels |
| **M4** | **Two-term SFT** — distilled from ensemble mean scores (ADR-0005) | **1×** | genuinely open — the honest test of whether distillation works at all |
| **M5** | **DPO on consensus pairs** — distilled from ensemble debate (MACA-style, *not* MACA; ADR-0003) | **1×** | target: ≥60% of M2's gain |

**Arm numbering changed on 15 Sep.** The MACA-style DPO arm was M3 until then and
is now **M5**; M3 is the new label-supervised baseline. Anything dated before
15 Sep that says "M3" means today's M5.

**M3 was trained and then excluded** (15 Sep). It is the only arm that reads
ground-truth labels, and a trusted monitor in deployment has none for the
attacks it faces. Because the exclusion came after its result — test pAUC 0.932,
above M4's 0.810 — the numbers and the reasoning are recorded in ADR-0007
rather than dropped.

**M5 is not MACA and must never be called MACA.** Published MACA is MV-SFT plus
MV-GRPO; M5 is offline DPO on debate-consensus preference pairs, chosen because
GRPO's rollout loop does not fit a 24-day sprint on one GPU. The write-up says
*"DPO on MACA-style consensus-derived preference pairs."* A null H3 is evidence
about this adaptation, not about MACA. See ADR-0003.

**M2's prompts are fixed, not searched for.** Write three sensibly different
personas yourself — e.g. "look for suspicious control flow," "compare against
what a correct solution would look like," "think like an attacker planting a
trigger." This removes the 8-variant implementation, the selection procedure, and
the external dependency on the anchor paper's authors.

**M1 and M2 use one sample per prompt, not three.** M1 is the same prompt sampled
three times at T=0.7; M2 is three different prompts sampled once each. This
matches how the anchor paper's "3-monitor ensemble" is actually constructed —
diversity from different monitors, not repeated sampling of one.

---

## 4. Experiment procedure, end to end

The full run, top to bottom — this is the sequence to follow, and the sequence
to walk someone through out loud.

### 4.1 Get the data

Pull `RoganInglis/apps-control-arena` from Hugging Face. Filter to items where
the backdoor is validated (`backdoor_works == True`) and a validated clean
solution exists for the same problem. Pair each backdoor with its problem's
clean solution — this pairing is what stops the monitor from learning "hard
problem → suspicious" instead of detecting backdoors. Cut **by `problem_id`**
into train (788 problems / 1,576 items), val (150 / 300) and test (644 / 1,288)
at a 50% attack rate, per the §2 table — nondeterministic problems all go to
train. **Output: three fixed, versioned JSONL files plus `splits.meta.json`
recording the seed, filter counts and dataset revision. Nothing downstream
touches the raw dataset again.**

### 4.2 Write the prompts

Four prompts, fixed before any generation runs, never edited after seeing
results:

- **Baseline prompt** — what M0 uses, and what every distilled model (M5, M4)
  is trained and served under.
- **Three persona prompts** for M2 — written by hand, not searched for (e.g.
  "look for suspicious control flow," "compare against what a correct solution
  would look like," "think like an attacker planting a trigger").

**Hard constraint on every prompt — both readouts must sit at deterministic
positions.** Each response must contain a yes/no verdict token *and* a 0–10
score, each at a predictable spot. Every arm is instrumented for both readouts,
and M4's training loss (§4.4) has to locate the yes token to compute its second
term. Prompts are frozen before any generation run, so this has to be right the
first time.

**Output: four prompt templates, checked into the repo before step 4.3 runs.**

### 4.3 Generate and preprocess

This step turns prompts + data into training material, in order:

1. **Score M0, M1, M2 on val and test.** M0/M1 are the baseline prompt sampled
   once and three times; M2 is the three persona prompts sampled once each.
2. **Gate:** M0's AUROC must clear 0.60 before continuing — see §9.
3. **Debate, for M5 only, on train:** the three M2 personas run two rounds —
   round 1 each scores independently, round 2 each revises after seeing the
   other two's round-1 reasoning. Persist full traces, not just scores.
4. **Consensus and pairs, for M5 only:** bin each round-2 score (default
   binary, suspicious ≥5 — switch to quintiles if val agreement exceeds 85%,
   decided in advance, not tuned after the fact), take the majority across the
   three agents as the consensus, then split traces into agree-with-majority
   (chosen) and dissent (rejected). Build (chosen, rejected) pairs, capped at
   2–3 per item.
5. **Prompt normalisation, for M5 only — do not skip this:** rewrite every
   pair's prompt field from the persona prompt to the baseline prompt. The
   debate happened under persona framing; the trained model deploys under the
   baseline prompt, so training data must match deployment or the model learns
   to expect framing it will never receive.
6. **Teacher scores, for M4 only, on train:** run the three M2 personas once
   each (no debate, no second round); keep the mean score as the regression
   target. **Persist the mean unrounded.** The fractional part is the whole
   point — three monitors averaging to 6.33 is finer-grained than any single
   monitor's 6 or 7, and that extra resolution is much of why the ensemble
   ranks better. Rounding it away here silently deletes the signal M4 exists
   to inherit.

**Output: a preference-pairs file for M5 (DPO format) and a scored-targets file
for M4 (SFT format), both keyed to the baseline prompt.**

### 4.4 Train

- **M5:** QLoRA DPO (β = 0.1), Qwen2.5-7B-Instruct, 4k sequence length, ~2
  epochs, on the M5 pairs file. DPO only — KTO is cut.
- **M4:** QLoRA SFT, same base model and sequence length, on the M4 targets
  file, with a **two-term loss** (see ADR-0005):

  ```
  loss = CE_text  +  λ · KD_yes           (λ = 1)
  ```

  - **`CE_text`** — ordinary next-token cross-entropy on a response ending in
    the teacher mean *rounded* to an integer. Keeps M4 emitting a readable 0–10
    score, and keeps the text readout comparable to the anchor paper.
  - **`KD_yes`** — pushes the student's `P("yes")` at the verdict token toward
    `teacher_mean / 10`. A probability is continuous, so the fractional teacher
    signal that `CE_text` had to round away survives training here.

  Roughly 20 lines over a standard QLoRA loop. Without the second term, M4 is
  trained on 11 hard tokens and **H4's sharpening result becomes an artifact of
  the loss rather than a finding** — you would have instructed the model to be
  coarse and then reported that it is coarse.

Run the training smoke test (~50 items) before either full run — confirm the
script executes end to end and loss decreases. **Output: two LoRA adapters.**

### 4.5 Evaluate

Serve each arm — M0 through M5 — with vLLM and score the val and
test splits. M5 and M4 are served alone: plain baseline prompt, one pass, no
personas, no debate. Record both readouts per call — `P("yes")` at the verdict
token (**primary**, ADR-0006) and the text 0–10 score (secondary) — and, for
M5/M4, two operating points (1 sample and 3 samples, for the cost claim and the
equal-compute check respectively).
**Output: a scores table — one row per (arm, item, sample, readout).**

> **This is a static evaluation, not a live control game.** Every item's label
> (backdoored / clean) is fixed in the dataset before any monitor runs. No
> attack policy is generating code during evaluation, and there is no
> audit/defer/resample decision being executed against a live task. Each
> monitor just scores a fixed set of already-labeled items once. This is
> exactly how the anchor ensemble paper evaluates too — see §5.

### 4.6 Analyse and report

Compute pAUC@[0,10%] FPR, safety@1% audit budget, score entropy, and
unique-value count per arm, with paired bootstrap and DeLong intervals against
M0. Test H1–H4 (§6) and apply the promotion rule. Build the frontier plot —
pAUC vs. inference cost, every arm. **Output: the hypothesis results table
and the main figure, both going straight into the write-up.**

---

## 5. Metrics

**Primary:** pAUC at FPR ≤ 10%, **McClish-standardised** (i.e.
`sklearn.metrics.roc_auc_score(..., max_fpr=0.10)`) — the anchor paper's exact
metric, chosen because TPR at a single low threshold is dominated by a handful
of negative examples at this sample size and its confidence interval is
enormous. See the caveat below on why the standardisation convention matters.

**Secondary:** safety at 1% audit budget; score entropy and unique-value count
(the sharpening cost); full AUROC.

**Every distilled arm reported at two operating points:** 1 sample (the headline
cost claim) and 3 samples (equal compute vs. M2, to check whether the distilled
model is simply better or only cheaper).

**Both readouts instrumented on every arm, with the logprob readout primary**
(ADR-0006). Every call yields two scores:

| Readout | What it is | Distinct values | Role |
|---|---|---|---|
| **logprob** | `P("yes")` at the verdict token | continuous | **primary** — decides H1–H4 |
| **text** | the 0–10 integer the model writes | ~5–8 in practice | secondary — the bridge to the anchor paper |

A 7B model does not really use eleven levels; it piles most mass onto 0/3/5/8.
Ranking 1,288 test items on ~6 distinct scores means heavy ties, and ties hurt
most at the low-FPR end where pAUC integrates — DeLong is also unreliable on
heavily tied data. The logprob readout has no such ceiling, and it is where M4's
fractional teacher signal actually lands (§4.4). **Fixed in advance, not after
seeing which readout looks better** — choosing post hoc would be fishing, and a
reviewer would say so. If DPO collapses the text score onto round numbers, that
collapse is now a *finding* (H4) rather than a rescue.

**Statistics:** every arm scores the same items, so all comparisons are paired —
DeLong for correlated AUROCs, paired bootstrap for pAUC.

> **Caveat — what "safety" means here.** Neither the anchor ensemble paper nor
> this project runs monitors inside a live ControlArena protocol. Both are
> static classification benchmarks: a fixed set of items, pre-labeled
> backdoored or clean, scored once by the monitor. There is no attack policy
> acting in real time and no audit/defer/resample decision being executed
> against a live task.
>
> **pAUC@[0,10%] FPR** is computed the standard classifier way: sweep a
> suspicion threshold, compute TPR (fraction of backdoored items flagged) and
> FPR (fraction of clean items flagged) at each threshold, then integrate TPR
> over the FPR ∈ [0, 0.10] region — and standardise with the **McClish
> correction**, which is exactly what
> `sklearn.metrics.roc_auc_score(..., max_fpr=0.10)` returns:
>
> ```
> floor = max_fpr² / 2        # a random monitor's raw area
> ceil  = max_fpr             # a perfect monitor's raw area
> pAUC  = 0.5 · (1 + (raw − floor) / (ceil − floor))
> ```
>
> **Not** the raw area divided by `max_fpr`. Both are called "normalised
> pAUC", but a random monitor scores 0.5 under McClish and 0.05 under the
> other, and the anchor's 0.7226 is a McClish figure. Mixing the conventions
> puts every comparison here on a different scale from the work being compared
> against, while producing numbers that look entirely plausible. The
> implementation is pinned to sklearn by test.
>
> **Safety at 1% audit budget** extends the same static scores: calibrate a
> threshold on clean items only, at the point where exactly 1% of them would
> be flagged, then report the fraction of backdoored items that clear that
> threshold.
>
> Neither of these is the trajectory-level "safety" from the original AI
> control formulation ([Greenblatt et al.](https://arxiv.org/pdf/2312.06942)) —
> the probability a blue team catches an attack across a full multi-task
> deployment, where the attacker chooses which tasks to attack and unaudited
> actions pass through untouched. That number requires simulating episodes
> under an attack strategy and is out of scope here. What this project reports
> is a **single-shot detection proxy at an audit-calibrated threshold** — the
> right measurement for "does distillation preserve the ensemble's
> discrimination," but the write-up should name it as a proxy, not claim it as
> full deployment-level safety.

---

## 6. Hypotheses

| ID | Hypothesis | Direction |
|---|---|---|
| **H1** | M1 beats M0 by roughly the anchor's identical-ensemble margin (~+0.04) | Sanity check the setup works |
| **H2** | **At least one of M5, M4 recovers ≥50% of M2's gain over M0, at 1× cost** | **Is distillation worth it at all — the core question** |
| **H3** | M5 (MACA) beats M4 (SFT) | Genuinely uncertain — MACA's own claim to test |
| **H4** | Trained arms show reduced score entropy and unique-value count vs. untrained arms | The sharpening cost — **only measurable because M4's loss keeps the teacher's fractional signal (ADR-0005)**; under plain SFT this result would be built in |

**Promotion rule, declared now, before any run:** H2 is primary. If either trained
arm *loses* more than 0.05 pAUC relative to M0, **H4 is promoted to the headline**
and the paper becomes *"Distillation improves monitor accuracy and degrades
monitor discrimination."* Declared in advance so promotion isn't after-the-fact
rationalisation.

**If H2 fails** — neither method recovers a meaningful fraction of M2's gain —
that is still a complete, reportable answer: *ensemble diversity in this setting
does not survive distillation by either method, and the 3× cost is not avoidable.*

---

## 7. Compute and budget

### Generation volume

| Job | Arithmetic | Generations |
|---|---|---|
| M5 debate transcripts (train) | 1,576 × 3 agents × 2 rounds | 9,456 |
| M4 teacher scores (train) | 1,576 × 3 personas × 1 sample | 4,728 |
| Test eval, all arms, both operating points | 1,288 × 15 passes | 19,320 |
| Val thresholds + agreement check | 300 × 15 passes | 4,500 |
| Diversity insurance, val (ADR-0002) | 300 × 2 extra models | 600 |
| Training pipeline smoke test | ~50 items × 3 × 2 | ~500 |
| **Total** | | **≈ 39,100** |

The 15 passes are M0×1 + M1×3 + M2×3 + M5×(1+3) + M4×(1+3). Down a further
**14%** from v4's 45,000, because Option A moved items out of train (where each
costs 9 generations) and into test (where each costs 15 but only once). At
~1,600 input / ~450 output tokens per generation: ≈62M input, ≈17M output
tokens.

### GPU-hours

**Inference** (vLLM, Qwen2.5-7B bf16): ≈3.8 h pure compute; with cold starts and
reruns (×2.5): **≈9.5 GPU-h.**

**Training** (QLoRA, 4k seqlen): two objectives (M5-DPO, M4-SFT), ≈2.5 h each;
with debugging and hyperparameter reruns (~5 runs total): **≈19 GPU-h.**

**Total ≈29 GPU-h. Budget 50 with contingency** — down from 100.

### vast.ai configuration

On-demand rates (marketplace — **verify live at `cloud.vast.ai`**, prices vary by host):

| GPU | VRAM | ~$/hr |
|---|---|---|
| **RTX A6000** | **48 GB** | **$0.41** |
| RTX 4090 | 24 GB | $0.39 |
| A100 80GB | 80 GB | $1.32 |
| H100 80GB | 80 GB | $2.21 |

**Use 1× RTX A6000 48GB.** Serves Qwen-7B in vLLM with a large KV cache and
trains QLoRA at 4k seqlen on the same machine — no re-provisioning between
phases — at essentially 4090 pricing.

**Filters:** ≥48 GB VRAM · ≥200 GB disk · ≥8 vCPU · ≥32 GB RAM · reliability
>99% · download >200 Mbps · CUDA ≥12.1 · PyTorch image.

**Interruptible (30–50% cheaper) for generation jobs** — checkpointable, restarts
cheaply. **On-demand for training runs.** Destroy instances between phases;
vast bills storage while stopped.

### Total budget

| Line | Estimate |
|---|---|
| GPU: 50 h × $0.41 (A6000, on-demand) | $21 |
| Storage: ~80 GB (one base model, two adapters) × ~1 month | $15 |
| Bandwidth: model pulls, dataset, checkpoint egress | $8 |
| Contingency (idle instances, failed runs) | $40–80 |
| API | **$0** |
| **Total** | **$85–125** |

Down from $150–250 in the previous scope.

---

## 8. Week by week

### Week 1 · 4–10 Sep — infrastructure and the competence gate

1. ~~Confirm ControlArena accepts a vLLM-served local model as the monitor.~~
   **Cut — `control-arena` is not a dependency (ADR-0004).** Only its dataset is
   used, pulled directly from Hugging Face.
2. Load the dataset; build backdoor↔clean item pairs; cut the splits.
3. Write the 3 fixed persona prompts for M2.
4. Stand up the eval harness: pAUC@[0,10%], safety@1%, entropy, unique-value count, paired bootstrap, DeLong.
5. Score **M0** → **GATE 1 (date): must happen by 14 Sep or M5 is cut.**
   **GATE 2 (quality): AUROC ≥ 0.60.** Then score **M1**, then **M2**.
6. **Freeze and push the preregistration (§3–§6) before any training run.**

### Week 2 · 11–17 Sep — pipeline proof and data generation

7. Run the training smoke test on ~50 items → confirm the QLoRA script runs end to end and loss decreases. **GATE, lightweight — debug here, not in week 3.**
8. **Diversity insurance (ADR-0002)** — score val with Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3 under the baseline prompt; compare `mean(Qwen, Llama, Mistral)` against `mean(3 Qwen personas)`. ~600 generations, two downloads. Runs **after** the 14 Sep gate and **before** the 9,456 debate generations, which is the last moment the answer is actionable.
9. Generate debate transcripts on **val only**; measure agreement; **lock the consensus binning**.
10. Generate the full M5 train debate transcripts (~9,456 generations).
11. Generate the M4 teacher scores from M2 on train (~4,728 generations).

### Week 3 · 18–24 Sep — train and evaluate

11. Train **M5** (MACA DPO) and **M4** (SFT).
12. Evaluate every arm on test, at both operating points.
13. Calibration analysis (H4).
14. Build the frontier plot — pAUC vs. inference cost. **Main figure.**

### Week 4 · 25–28 Sep — write-up

15. Answer H2, H3, H4 explicitly in the write-up regardless of direction.
16. Post.

### Slip rules

Two levers, declared in advance, largest first.

**1 · The 14 Sep spine gate — decides whether M5 survives.** M0 must be scored
and its AUROC checked **by 14 Sep**. Scoring M0 requires the entire spine
working end to end: data loading, splits, prompts, vLLM serving, both readouts,
and the metrics code. If that spine is not up by the 14th, **cut M5** and ship
M0/M1/M2/M4.

The arithmetic behind the date: M5 needs debate generation (9,456 calls),
consensus binning, the pairs pipeline, prompt normalisation, and a DPO run —
roughly **60% of the remaining engineering**, and the part with the most novel
code. M4 needs teacher scores (4,728 calls) and one SFT run. With the spine up
on the 14th there are 14 days for two training pipelines; without it, there are
not.

| Ship | H1 | H2 | H3 | H4 |
|---|---|---|---|---|
| M0, M1, M2, M4 | ✅ | ✅ | ❌ | ✅ |
| all five | ✅ | ✅ | ✅ | ✅ |

Cutting M5 costs H3 — the DPO-vs-SFT comparison. That is a real loss, but per
ADR-0003 it was never a test of MACA itself, and **a rushed DPO run producing an
uninterpretable null is worse than not running it.** H2, the primary hypothesis,
is answerable by M4 alone.

**2 · If week 3 runs short**, drop the 3-sample equal-compute readout and report
only the 1-sample headline numbers.

**Never drop M4** — it is what makes H2 answerable on its own, independent of
whether the DPO arm works.

---

## 9. Kill criteria

| Trigger | Action |
|---|---|
| **M0 not scored by 14 Sep** | **Cut M5.** Ship M0/M1/M2/M4 — H1, H2 and H4 all survive; only the DPO-vs-SFT comparison is lost. Decided now, on 9 Sep, so that it reads as a plan rather than a week-3 concession. |
| M0 AUROC < 0.60 | **Stop the training arms.** Consensus among near-chance monitors is noise. Move to a larger trusted model, or ship the M0/M1/M2 replication alone. |
| Smoke test fails by 17 Sep | Debug before generating full transcripts — this is cheap to catch here and expensive to catch in week 3. |
| M4's two-term loss not converging by 17 Sep | Set λ = 0, ship plain text SFT, name the rounding as a limitation (ADR-0005). H4 weakens; the project survives. |
| Agreement > 85% and quintiles don't fix it | Report it: *"monitoring transcripts don't admit useful consensus supervision"* is a real finding about MACA's scope. |
| M2 doesn't beat M1 on Qwen | The anchor's diversity result doesn't replicate on an open 7B — report descriptively; the distillation arms lose their motivation but still answer H2. |
| Neither M5 nor M4 clears H2's 50% threshold | **This is the answer, not a failure.** Distillation doesn't preserve ensemble diversity in this setting — write that up. |
| A trained arm loses > 0.05 pAUC | H4 promoted per §6 — the headline, not a setback. |

---

## 10. Future extensions

Scoped out of the 24-day sprint, and worth naming explicitly in the write-up so
that readers know which questions were *deferred* rather than *answered*.

### The two remaining MACA stages

M5 uses MACA's data pipeline — multi-agent debate, consensus extraction — with
DPO substituted for MACA's own training objectives (ADR-0003). Both of the
substituted stages are natural follow-ups, and together they would turn "we
adapted MACA" into "we tested MACA."

**MV-SFT — MACA's first stage.** Fine-tune on the majority-consistent
trajectories alone, discarding dissent. Cheaper than DPO and strictly more
faithful to the paper. Run alongside M5 and M4 it makes a clean three-way
decomposition that this project cannot currently separate:

| Arm | Data | Target | Isolates |
|---|---|---|---|
| M4 | all train items | ensemble mean score | distilling the ensemble's *output* |
| MV-SFT | consensus items only | the winning trajectory | distilling the ensemble's *agreement* |
| M5 (DPO) | consensus + dissent pairs | preference between them | distilling the ensemble's *disagreement* |

The open question it answers: does the value of debate come from **filtering** to
what the monitors agreed on, or from the **contrast** with what they abandoned?
This project's H3 conflates the two — M5 beating M4 could be either.

**MV-GRPO — MACA's second stage.** Reinforcement learning with consensus-derived
rewards, on top of MV-SFT. This is the only configuration that constitutes an
actual test of MACA on a monitoring task; everything here is an adaptation. It
needs an online rollout loop — generate, score against consensus, update, repeat
— which is why it does not fit one rented GPU in 24 days rather than why it was
judged unpromising.

**Say this plainly in the write-up.** A null H3 is evidence about DPO on
consensus-derived pairs. It is *not* evidence that MACA fails at monitoring,
and the paper should refuse that inference before a reader draws it.

### Other deferred questions

**Cross-family ensemble diversity (ADR-0002).** M2's diversity comes from three
persona prompts over one base model. The anchor paper's headline +13% came from
architecturally *different* models, and its identical-model row was only +5.5%.
The week-2 val-only check measures the gap; actually rebuilding M2 from Qwen,
Llama and Mistral — and distilling a cross-family ensemble into a single Qwen —
is the stronger version of the whole claim.

**Live control protocol evaluation (ADR-0004).** Everything here is static
classification over pre-labelled items. The trajectory-level safety number from
the AI control literature — where an attacker chooses which tasks to attack
across a deployment and unaudited actions pass through untouched — requires
simulating episodes under an attack strategy. A distilled monitor that preserves
pAUC may or may not preserve *deployment* safety, and nothing measured here
settles it.

**Attack-strength sensitivity (ADR-0001).** Backdoors come pre-generated from a
fixed distribution, so attack difficulty is not a variable under our control.
Whether distillation survives *harder* attacks than the ones it was trained to
catch is untested, and is the question an adversary would ask first.

---

## References

- Ensemble Monitoring for AI Control — [arXiv:2605.15377](https://arxiv.org/html/2605.15377v2) — primary anchor
- MACA — [arXiv:2509.15172](https://arxiv.org/abs/2509.15172) · [facebookresearch/maca](https://github.com/facebookresearch/maca)
- Dataset — [RoganInglis/apps-control-arena](https://huggingface.co/datasets/RoganInglis/apps-control-arena)
- ControlArena — [docs](https://control-arena.aisi.org.uk/) · [APPS setting](https://control-arena.aisi.org.uk/settings/apps.html)
- Control-Tax (backdoor provenance) — [arXiv:2506.05296](https://arxiv.org/html/2506.05296v1)
- AI Control: Improving Safety Despite Intentional Subversion — [arXiv:2312.06942](https://arxiv.org/pdf/2312.06942)
- Trusted Debate (inference-time only, related but distinct) — [NeurIPS 2025](https://neurips.cc/virtual/2025/loc/mexico-city/133159)
