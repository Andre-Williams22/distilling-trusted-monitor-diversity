#!/usr/bin/env bash
# M7 diagnostic (ADR-0010): did MV-KTO collapse, or did my hyperparameters?
# Run ON THE VM after infra/vast/remote_setup.sh, inside tmux:
#
#   cd ~ && tmux new -d -s m7diag 'bash infra/vast/run_m7_diag.sh'
#   tmux switch-client -t m7diag
#
# M7's first run answered "yes" almost always (70 distinct scores over 1,288
# items). Two short runs on a seeded subsample separate the explanations:
#   A  low learning rate, class weights as computed  -> tests optimisation
#   B  original learning rate, no class reweighting  -> tests the weighting
#
# Each is scored on VAL ONLY. Whichever (if either) avoids collapse is the
# candidate for a full re-run; test is never touched here.
set -euo pipefail
cd "$HOME"
mkdir -p logs
exec > >(tee -a "logs/m7diag_$(date +%Y%m%d_%H%M%S).log") 2>&1

MODEL="Qwen/Qwen2.5-7B-Instruct"
EXAMPLES="${EXAMPLES:-1500}"
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

wait_for_vllm() {
  for _ in $(seq 1 120); do
    curl -fs localhost:8000/health >/dev/null 2>&1 && { echo "  vLLM ready"; return; }
    tmux has-session -t "$1" 2>/dev/null || { echo "vLLM exited; see logs/$1.log" >&2; exit 1; }
    sleep 5
  done
  echo "vLLM did not come up" >&2; exit 1
}

stop_vllm() {
  for session in vllm vllm-m3 vllm-m4 vllm-m5 vllm-m6 vllm-m7 vllm-m8 vllm-m9 vllm-diag; do
    tmux kill-session -t "$session" 2>/dev/null || true
  done
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && { echo "  GPU free (${used} MiB in use)"; return; }
    sleep 2
  done
  echo "GPU still holds memory after stopping vLLM" >&2; exit 1
}

for f in data/generations/teacher__train.json data/generations/debate__train.json; do
  [ -s "$f" ] || { echo "$f missing; upload it from the laptop first" >&2; exit 1; }
done

stop_vllm
step "A. low learning rate (2e-5), weights as computed, $EXAMPLES examples, 1 epoch"
.venv/bin/python main.py train-kto --learning-rate 2e-5 --max-examples "$EXAMPLES" \
  --epochs 1 --tag lowlr

step "B. original learning rate (1e-4), no class reweighting, $EXAMPLES examples, 1 epoch"
.venv/bin/python main.py train-kto --undesirable-weight 1.0 --max-examples "$EXAMPLES" \
  --epochs 1 --tag noweight

step "C. Serve both adapters and score each on val only"
lowlr=$(ls -td runs/*__m7-maca-kto-lowlr__*/adapter | head -1)
noweight=$(ls -td runs/*__m7-maca-kto-noweight__*/adapter | head -1)
echo "  A: $lowlr"
echo "  B: $noweight"
stop_vllm
tmux new-session -d -s vllm-diag \
  "VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/vllm serve $MODEL --port 8000 \
   --gpu-memory-utilization 0.85 --max-model-len 8192 --max-logprobs 20 \
   --enable-lora --max-lora-rank 16 \
   --lora-modules m7lowlr=$lowlr m7noweight=$noweight \
   2>&1 | tee logs/vllm-diag.log"
wait_for_vllm vllm-diag
# Score each adapter in turn, moving the file aside so the next run cannot
# resume into it (both write the same default path).
for name in m7lowlr m7noweight; do
  rm -f data/generations/m7__val.json
  .venv/bin/python main.py score --arm m7 --split val --adapter "$name" --no-resume
  mv data/generations/m7__val.json "data/generations/m7__val__${name}.json"
done

step "D. Collapse check"
.venv/bin/python - <<'PY'
import json
import numpy as np
from src.metrics import auroc, partial_auroc

print("%-12s %7s %7s %9s %9s %11s" % (
    "config", "pAUC", "AUROC", "distinct", "frac>.99", "mean|clean"))
for name in ("m7lowlr", "m7noweight"):
    rows = json.load(open(f"data/generations/m7__val__{name}.json"))
    pairs = [(r["logprob_score"], r["label"]) for r in rows
             if r["logprob_score"] is not None]
    s = np.array([p for p, _ in pairs]); y = np.array([l for _, l in pairs])
    print("%-12s %7.3f %7.3f %9d %9.3f %11.3f" % (
        name, partial_auroc(s, y), auroc(s, y),
        len(np.unique(np.round(s, 6))), np.mean(s > 0.99), s[y == 0].mean()))
print()
print("Original M7 on val was: pAUC 0.519 · 70 distinct on test · frac>.99 0.836")
print("Degenerate again -> the objective matched its 0.520 teacher, not a bug.")
print("Spread out and higher -> the first run was an optimisation failure.")
PY

echo
echo "Done. From the laptop: VAST_SSH_KEY=~/.ssh/arena_key bash infra/vast/pull_results.sh"
