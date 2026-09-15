#!/usr/bin/env bash
# M5: MACA's MV-DPO on persona-debate consensus pairs (ADR-0003). Run ON THE VM
# after infra/vast/remote_setup.sh, inside tmux:
#
#   cd ~ && tmux new -d -s m5 'bash infra/vast/run_m5.sh'
#   tmux switch-client -t m5
#
# Needs data/generations/teacher__train.json (M2's round-1 persona answers,
# uploaded from the laptop) -- M4 and M5 start from identical round-1 answers.
#
# Stages:
#   1. Serve the base model; debate round 2 on train.
#   2. Build preference pairs by majority written verdict.
#   3. Stop vLLM, smoke-train on a few pairs to prove the loop.
#   4. Full DPO training.
#   5. Serve the adapter and score M5 on val, then test.
set -euo pipefail
cd "$HOME"
mkdir -p logs
exec > >(tee -a "logs/m5_$(date +%Y%m%d_%H%M%S).log") 2>&1

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
  for session in vllm vllm-m3 vllm-m4 vllm-m5; do tmux kill-session -t "$session" 2>/dev/null || true; done
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 2000 ] && { echo "  GPU free (${used} MiB in use)"; return; }
    sleep 2
  done
  echo "GPU still holds memory after stopping vLLM" >&2; exit 1
}

[ -s data/generations/teacher__train.json ] || {
  echo "data/generations/teacher__train.json missing; upload it from the laptop first" >&2
  exit 1
}
for f in data/generations/m5__val.json data/generations/m5__test.json; do
  [ -e "$f" ] && { echo "$f exists; archive it first so a different model's scores are not resumed" >&2; exit 1; }
done

step "1. Serve the base model and debate round 2 on train"
stop_vllm
tmux new-session -d -s vllm "VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/vllm serve $MODEL --port 8000 \
   --gpu-memory-utilization 0.85 --max-model-len 8192 --max-logprobs 20 2>&1 | tee logs/vllm.log"
wait_for_vllm vllm
.venv/bin/python main.py debate --split train

step "2. Build preference pairs from the majority verdict"
.venv/bin/python main.py build-pairs --split train
cat data/training/dpo_pairs__train.stats.json

step "3. Stop vLLM, then smoke-train to prove the loop"
stop_vllm
.venv/bin/python main.py train-dpo --smoke

step "4. Full DPO training"
.venv/bin/python main.py train-dpo
adapter=$(ls -td runs/*__m5-maca-dpo__*/adapter | head -1)
echo "  adapter: $adapter"

step "5. Serve the adapter and score M5 on val, then test"
tmux new-session -d -s vllm-m5 \
  "VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/vllm serve $MODEL --port 8000 \
   --gpu-memory-utilization 0.85 --max-model-len 8192 --max-logprobs 20 \
   --enable-lora --max-lora-rank 16 --lora-modules m5=$adapter \
   2>&1 | tee logs/vllm-m5.log"
wait_for_vllm vllm-m5
for split in val test; do
  .venv/bin/python main.py score --arm m5 --split "$split" --adapter m5
done

.venv/bin/python - <<'PY'
import json
for split in ("val", "test"):
    rows = json.load(open(f"data/generations/m5__{split}.json"))
    both = sum(r["logprob_score"] is not None and r["text_score"] is not None for r in rows)
    print(f"  {split}: {both}/{len(rows)} readable · model {rows[0]['model']}")
    assert rows[0]["model"] == "m5", "scored with the base model, not the adapter"
PY

echo
echo "Done. From the laptop: VAST_SSH_KEY=~/.ssh/arena_key bash infra/vast/pull_results.sh"
