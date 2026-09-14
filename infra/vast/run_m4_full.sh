#!/usr/bin/env bash
# The full M4 run on the GPU box. Run ON THE VM after run_m4_smoke.sh passed,
# inside tmux:
#
#   cd ~ && tmux new -d -s m4full 'bash infra/vast/run_m4_full.sh'
#   tmux switch-client -t m4full
#
# Stages:
#   1. Train M4 on all of train (teacher scores from the smoke test are reused).
#   2. Serve the adapter with vLLM.
#   3. Score M4 on val, then test.
#
# The comparison against M0-M2 is built on the laptop, where those arms' score
# files live: pull data/generations/, then `python main.py analyse --split test`.
set -euo pipefail
cd "$HOME"
mkdir -p logs
exec > >(tee -a "logs/m4_full_$(date +%Y%m%d_%H%M%S).log") 2>&1

MODEL="Qwen/Qwen2.5-7B-Instruct"
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

test -s data/training/sft_targets__train.jsonl || {
  echo "no teacher targets; run infra/vast/run_m4_smoke.sh first" >&2
  exit 1
}

step "1. Stop vLLM to free the GPU for training"
for session in vllm vllm-m4; do tmux kill-session -t "$session" 2>/dev/null || true; done
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] && { echo "  GPU free (${used} MiB in use)"; break; }
  sleep 2
done

step "2. Train M4 on all of train"
.venv/bin/python main.py train-sft
adapter=$(ls -td runs/*__m4-sft__*/adapter | head -1)
echo "  adapter: $adapter"
grep -E "loss (decreased|DID NOT DECREASE)" "$(ls -t logs/m4_full_*.log | head -1)" | tail -1

step "3. Serve the adapter"
tmux new-session -d -s vllm-m4 \
  "VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/vllm serve $MODEL --port 8000 \
   --gpu-memory-utilization 0.85 --max-model-len 8192 --max-logprobs 20 \
   --enable-lora --max-lora-rank 16 --lora-modules m4=$adapter \
   2>&1 | tee logs/vllm-m4-full.log"
for _ in $(seq 1 120); do
  curl -fs localhost:8000/health >/dev/null 2>&1 && { echo "  vLLM ready"; break; }
  tmux has-session -t vllm-m4 2>/dev/null || { echo "vLLM exited; see logs/vllm-m4-full.log" >&2; exit 1; }
  sleep 5
done
curl -fs localhost:8000/health >/dev/null || { echo "vLLM did not come up" >&2; exit 1; }

step "4. Score M4 on val, then test"
for split in val test; do
  .venv/bin/python main.py score --arm m4 --split "$split" --adapter m4
done

.venv/bin/python - <<'PY'
import json
for split in ("val", "test"):
    rows = json.load(open(f"data/generations/m4__{split}.json"))
    both = sum(r["logprob_score"] is not None and r["text_score"] is not None for r in rows)
    print(f"  {split}: {both}/{len(rows)} readable · model {rows[0]['model']}")
    assert rows[0]["model"] == "m4", "scored with the base model, not the adapter"
PY

echo
echo "Done. From the laptop: VAST_SSH_KEY=~/.ssh/arena_key bash infra/vast/pull_results.sh"
