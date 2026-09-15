#!/usr/bin/env bash
# M3: distilled from M1's identical ensemble (ADR-0008). Run ON THE VM after
# infra/vast/remote_setup.sh, inside tmux:
#
#   cd ~ && tmux new -d -s m3 'bash infra/vast/run_m3_m1.sh'
#   tmux switch-client -t m3
#
# Stages:
#   1. M1 samples on train (the teacher); skipped items already scored.
#   2. Stop vLLM so training has the GPU.
#   3. Train M3 on M1's judgements -- every setting identical to M4.
#   4. Serve the adapter and score M3 on val, then test.
set -euo pipefail
cd "$HOME"
mkdir -p logs
exec > >(tee -a "logs/m3_m1_$(date +%Y%m%d_%H%M%S).log") 2>&1

MODEL="Qwen/Qwen2.5-7B-Instruct"
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
  for session in vllm vllm-m3 vllm-m4; do tmux kill-session -t "$session" 2>/dev/null || true; done
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && { echo "  GPU free (${used} MiB in use)"; return; }
    sleep 2
  done
  echo "GPU still holds memory after stopping vLLM" >&2; exit 1
}

for f in data/generations/m3__val.json data/generations/m3__test.json; do
  [ -e "$f" ] && { echo "$f exists; archive it first so a different model's scores are not resumed" >&2; exit 1; }
done

step "1. M1 samples on train (the teacher)"
if [ ! -s data/generations/m1__train.json ]; then
  stop_vllm
  tmux new-session -d -s vllm "VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/vllm serve $MODEL --port 8000 \
     --gpu-memory-utilization 0.85 --max-model-len 8192 --max-logprobs 20 2>&1 | tee logs/vllm.log"
  wait_for_vllm vllm
  .venv/bin/python main.py score --arm m1 --split train
else
  echo "  data/generations/m1__train.json present; reusing it"
fi

step "2. Stop vLLM to free the GPU for training"
stop_vllm

step "3. Train M3 on M1's judgements"
.venv/bin/python main.py train-sft --targets m1-ensemble
adapter=$(ls -td runs/*__m3-sft-m1-ensemble__*/adapter | head -1)
echo "  adapter: $adapter"

step "4. Serve the adapter and score M3 on val, then test"
tmux new-session -d -s vllm-m3 \
  "VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/vllm serve $MODEL --port 8000 \
   --gpu-memory-utilization 0.85 --max-model-len 8192 --max-logprobs 20 \
   --enable-lora --max-lora-rank 16 --lora-modules m3=$adapter \
   2>&1 | tee logs/vllm-m3.log"
wait_for_vllm vllm-m3
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
