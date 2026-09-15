---
status: accepted
date: 2026-09-09
revised: 2026-09-15
---

# M5 is MACA: MV-DPO on persona-debate consensus pairs

MACA (arXiv:2509.15172) post-trains a model on its own multi-agent debate. It
evaluates **four** objectives on the same debate data -- MV-SFT, MV-GRPO,
MV-DPO and MV-KTO -- and reports that **preference learning over full reasoning
traces beats binary consensus rewards and SFT**, with **MV-DPO best at 4-8B**.
M5 therefore implements **MV-DPO**, the paper's own best recipe at Qwen-7B's
size: three agents, two rounds, majority vote on the final written answer, every
agreeing x dissenting pair, DPO, QLoRA, no ground-truth labels.

## Correction, 15 Sep

An earlier version of this ADR said MACA was "MV-SFT plus MV-GRPO", that this
arm implemented neither, and that it "must never be called MACA". That was
wrong on all three counts, and the error spread to project-plan.md and the blog
draft. M5 is a faithful MV-DPO implementation with three documented adaptations
below; the write-up may call it MACA (MV-DPO), naming those adaptations.

## Adaptations, and why

| | MACA | M5 | Why |
|---|---|---|---|
| Agents | 3 identical copies | 3 different **personas** | The project studies distilling *diversity*; M4 vs M3 shows a diverse teacher distils better (ADR-0008) |
| Voted answer | extracted final answer | the written `BACKDOORED: yes/no` | The task's discrete answer. Replaces the earlier plan's binned 0-10 score, which was a needless deviation |
| Training context | question **plus peers' reasoning** (the paper's stronger setting) | **baseline prompt only** | A deployed trusted monitor sees no peers, and M4 trains the same way, so M5 vs M4 differs only in objective |
| Debate temperature | 1.0 | 0.7 | Matches M4's teacher data and the rest of the project |
| Round 1 | generated for the debate | **reused** from M4's teacher run | M4 and M5 start from identical persona answers |

## Consequences

- **H3 (M5 beats M4) is a real test of MACA's method against score
  distillation**, at the same inference cost and on the same items, not a test
  of a stand-in.
- **The no-context choice is deliberate and costs something.** The paper found
  training with debate context stronger. If M5 underperforms, that is a
  candidate explanation, and training with context is the obvious follow-up.
- **Pairs are scarcer than the plan assumed.** Only items where the personas
  disagree yield pairs: 36% of train items split 2-1 before debate, so at most
  ~1,140 pairs, against the plan's ~3,900. Debate typically raises agreement,
  so the real count is lower.
- MV-SFT, MV-GRPO and MV-KTO remain unimplemented variants, listed in
  project-plan.md section 10.
