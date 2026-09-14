#!/usr/bin/env bash
# M4 smoke test on the GPU box: does the whole pipeline run, and does the loss
# move? Not a result. Run ON THE VM after infra/vast/remote_setup.sh, in tmux:
#
#   cd ~ && tmux new -d -s m4 'bash infra/vast/run_m4_smoke.sh'
#   tmux switch-client -t m4
#
# Stages:
#   1. Teacher scores on all of train (~8 min; the full run needs them too).
#   2. Stop vLLM so training has the GPU.
#   3. Smoke-train M4 on the first 50 train items.
#   4. Serve the adapter with vLLM and score the first 100 val items.
#   5. Check both readouts parse on the adapter's responses.
#
# Every stage resumes or is cheap to repeat. The whole log is kept in logs/.
set -euo pipefail
cd "$HOME"
mkdir -p logs
exec > >(tee -a "logs/m4_smoke_$(date +%Y%m%d_%H%M%S).log") 2>&1

MODEL="Qwen/Qwen2.5-7B-Instruct"
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

wait_for_vllm() {
  local session=$1
  for _ in $(seq 1 120); do
    if curl -fs localhost:8000/health >/dev/null 2>&1; then echo "  vLLM ready"; return; fi
    if ! tmux has-session -t "$session" 2>/dev/null; then
      echo "vLLM exited during startup; see logs/$session.log" >&2
      grep -E "Error|error" "logs/$session.log" | tail -5 >&2
      exit 1
    fi
    sleep 5
  done
  echo "vLLM did not come up in 10 minutes; see logs/$session.log" >&2
  exit 1
}

stop_vllm() {
  for session in vllm vllm-m4; do
    tmux kill-session -t "$session" 2>/dev/null || true
  done
  # Training needs the memory vLLM reserved; wait until the GPU is actually free.
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && { echo "  GPU free (${used} MiB in use)"; return; }
    sleep 2
  done
  echo "GPU still holds memory after stopping vLLM" >&2
  exit 1
}

step "1. Teacher scores on train"
curl -fs localhost:8000/health >/dev/null || {
  echo "base vLLM is not serving on :8000 -- run infra/vast/remote_setup.sh first" >&2
  exit 1
}
.venv/bin/python main.py teacher-scores --split train

step "2. Stop vLLM to free the GPU for training"
stop_vllm

step "3. Smoke-train M4 (first 50 train items)"
.venv/bin/python main.py train-sft --smoke
adapter=$(ls -td runs/*m4-sft-smoke*/adapter | head -1)
echo "  adapter: $adapter"

step "4. Serve the adapter and score the first 100 val items"
tmux new-session -d -s vllm-m4 \
  "VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/vllm serve $MODEL --port 8000 \
   --gpu-memory-utilization 0.85 --max-model-len 8192 --max-logprobs 20 \
   --enable-lora --max-lora-rank 16 --lora-modules m4=$adapter \
   2>&1 | tee logs/vllm-m4.log"
wait_for_vllm vllm-m4
rm -f data/generations/m4__val__limit100.json
.venv/bin/python main.py score --arm m4 --split val --limit 100 --adapter m4

step "5. Did the adapter keep the output format?"
.venv/bin/python - <<'PY'
import json
rows = json.load(open("data/generations/m4__val__limit100.json"))
both = sum(r["logprob_score"] is not None and r["text_score"] is not None for r in rows)
print(f"  {both}/{len(rows)} responses have both readouts · model {rows[0]['model']}")
print("  sample response:\n   ", rows[0]["response"][:300].replace("\n", "\n    "))
assert rows[0]["model"] == "m4", "scored with the base model, not the adapter"
assert both >= 0.95 * len(rows), "the adapter broke the output format"
PY

echo
echo "Smoke test passed. The loss line from training is above (search 'loss decreased')."
echo "From the laptop: VAST_SSH_KEY=~/.ssh/arena_key bash infra/vast/pull_results.sh"
