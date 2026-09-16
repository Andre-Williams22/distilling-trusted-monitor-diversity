#!/usr/bin/env bash
# M8: MACA's MV-GRPO on individually labelled debate answers (ADR-0009). Run ON
# THE VM after infra/vast/remote_setup.sh, inside tmux:
#
#   cd ~ && tmux new -d -s m8 'bash infra/vast/run_m8.sh'
#   tmux switch-client -t m8
#
# Needs the same two uploaded files as M6 -- no generation happens here:
#   data/generations/teacher__train.json   round-1 persona answers (the text)
#   data/generations/debate__train.json    round-2 transcripts (the vote)
#
# The only online arm: each step samples a group of answers from the CURRENT
# policy and rewards those whose verdict matches the frozen debate majority.
# Rollouts dominate the runtime, so this is much slower than M6 and M7.
#
# Stages:
#   1. Smoke-train to prove the loop.
#   2. Full MV-GRPO training.
#   3. Serve the adapter and score M8 on val, then test.
set -euo pipefail
cd "$HOME"
mkdir -p logs
exec > >(tee -a "logs/m8_$(date +%Y%m%d_%H%M%S).log") 2>&1

MODEL="Qwen/Qwen2.5-7B-Instruct"
TEXT_SOURCE="${TEXT_SOURCE:-round1}"
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
  for session in vllm vllm-m3 vllm-m4 vllm-m5 vllm-m6 vllm-m7 vllm-m8; do
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
for f in data/generations/m8__val.json data/generations/m8__test.json; do
  [ -e "$f" ] && { echo "$f exists; archive it first so a different model's scores are not resumed" >&2; exit 1; }
done

step "1. Smoke-train to prove the loop (text source: $TEXT_SOURCE)"
stop_vllm
.venv/bin/python main.py train-grpo --text-source "$TEXT_SOURCE" --smoke

step "2. Full MV-GRPO training"
.venv/bin/python main.py train-grpo --text-source "$TEXT_SOURCE"
adapter=$(ls -td runs/*__m8-maca-grpo__*/adapter | head -1)
echo "  adapter: $adapter"

step "3. Serve the adapter and score M8 on val, then test"
tmux new-session -d -s vllm-m8 \
  "VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/vllm serve $MODEL --port 8000 \
   --gpu-memory-utilization 0.85 --max-model-len 8192 --max-logprobs 20 \
   --enable-lora --max-lora-rank 16 --lora-modules m8=$adapter \
   2>&1 | tee logs/vllm-m8.log"
wait_for_vllm vllm-m8
for split in val test; do
  .venv/bin/python main.py score --arm m8 --split "$split" --adapter m8
done

.venv/bin/python - <<'PY'
import json
for split in ("val", "test"):
    rows = json.load(open(f"data/generations/m8__{split}.json"))
    both = sum(r["logprob_score"] is not None and r["text_score"] is not None for r in rows)
    print(f"  {split}: {both}/{len(rows)} readable · model {rows[0]['model']}")
    assert rows[0]["model"] == "m8", "scored with the base model, not the adapter"
PY

echo
echo "Done. From the laptop: VAST_SSH_KEY=~/.ssh/arena_key bash infra/vast/pull_results.sh"
