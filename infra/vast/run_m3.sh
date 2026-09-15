#!/usr/bin/env bash
# M3: the label-supervised baseline (ADR-0007). Run ON THE VM after
# infra/vast/remote_setup.sh, inside tmux:
#
#   cd ~ && tmux new -d -s m3 'bash infra/vast/run_m3.sh'
#   tmux switch-client -t m3
#
# Stages:
#   1. Base monitor answers on train (baseline prompt, 3 samples) -- the source
#      of M3's reasoning sentences.
#   2. Stop vLLM so training has the GPU.
#   3. Train M3 on the true labels of deterministic train items. Same loss,
#      adapter, hyperparameters and seed as M4.
#   4. Serve the adapter and score M3 on val, then test.
#
# Comparisons against the other arms are built on the laptop, where their score
# files live.
set -euo pipefail
cd "$HOME"
mkdir -p logs
exec > >(tee -a "logs/m3_$(date +%Y%m%d_%H%M%S).log") 2>&1

MODEL="Qwen/Qwen2.5-7B-Instruct"
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "1. Base monitor answers on train (reasoning source)"
curl -fs localhost:8000/health >/dev/null || {
  echo "base vLLM is not serving on :8000 -- run infra/vast/remote_setup.sh first" >&2
  exit 1
}
.venv/bin/python main.py score --arm m1 --split train

step "2. Stop vLLM to free the GPU for training"
for session in vllm vllm-m3; do tmux kill-session -t "$session" 2>/dev/null || true; done
for _ in $(seq 1 30); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 2000 ] && { echo "  GPU free (${used} MiB in use)"; break; }
  sleep 2
done

step "3. Train M3 on true labels (deterministic train items)"
.venv/bin/python main.py train-sft --targets labels
adapter=$(ls -td runs/*__m3-sft-labels__*/adapter | head -1)
echo "  adapter: $adapter"

step "4. Serve the adapter and score M3 on val, then test"
tmux new-session -d -s vllm-m3 \
  "VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/vllm serve $MODEL --port 8000 \
   --gpu-memory-utilization 0.85 --max-model-len 8192 --max-logprobs 20 \
   --enable-lora --max-lora-rank 16 --lora-modules m3=$adapter \
   2>&1 | tee logs/vllm-m3.log"
for _ in $(seq 1 120); do
  curl -fs localhost:8000/health >/dev/null 2>&1 && { echo "  vLLM ready"; break; }
  tmux has-session -t vllm-m3 2>/dev/null || { echo "vLLM exited; see logs/vllm-m3.log" >&2; exit 1; }
  sleep 5
done
curl -fs localhost:8000/health >/dev/null || { echo "vLLM did not come up" >&2; exit 1; }

for split in val test; do
  .venv/bin/python main.py score --arm m3 --split "$split" --adapter m3
done

.venv/bin/python - <<'PY'
import json
for split in ("val", "test"):
    rows = json.load(open(f"data/generations/m3__{split}.json"))
    both = sum(r["logprob_score"] is not None and r["text_score"] is not None for r in rows)
    print(f"  {split}: {both}/{len(rows)} readable · model {rows[0]['model']}")
    assert rows[0]["model"] == "m3", "scored with the base model, not the adapter"
PY

echo
echo "Done. From the laptop: VAST_SSH_KEY=~/.ssh/arena_key bash infra/vast/pull_results.sh"
