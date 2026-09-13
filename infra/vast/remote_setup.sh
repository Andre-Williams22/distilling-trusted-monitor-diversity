#!/usr/bin/env bash
# One-time setup on a fresh vast.ai GPU instance. Run it ON THE VM:
#
#   bash <(curl -fsSL https://raw.githubusercontent.com/Andre-Williams22/distilling-trusted-monitor-diversity/main/infra/vast/remote_setup.sh)
#
# It installs the environment, rebuilds the splits and checks they match the
# preregistered ones byte for byte, starts vLLM in a tmux session, and scores
# four items as a smoke test. Safe to re-run: every step skips work already done.
set -euo pipefail

REPO_URL="https://github.com/Andre-Williams22/distilling-trusted-monitor-diversity.git"
# The repo lives directly in the home directory, not in a subfolder.
REPO_DIR="$HOME"
MODEL="Qwen/Qwen2.5-7B-Instruct"

# SHA-256 prefixes of the splits built on the laptop. A mismatch means the
# dataset or the split code changed, and nothing downstream can be trusted.
declare -A EXPECTED_SPLIT_HASH=(
  [train]=240dbf5bd6358be4
  [val]=b8d51443bba83137
  [test]=ab858453f1d36088
)

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "System packages"
# build-essential is required, not optional: vLLM compiles GPU kernels with
# torch.compile/Triton on first start, and without gcc the engine dies with
# "Failed to find C compiler" after the model has already downloaded.
if ! command -v tmux >/dev/null || ! command -v git >/dev/null \
   || ! command -v rsync >/dev/null || ! command -v gcc >/dev/null \
   || ! command -v ninja >/dev/null; then
  apt-get update -qq && apt-get install -y -qq tmux git curl rsync build-essential ninja-build >/dev/null
fi
gcc --version | head -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

step "uv (fast installer; the project itself installs from requirements.txt)"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

step "Code"
cd "$REPO_DIR"
if [ -d .git ]; then
  git pull --ff-only
else
  # git clone refuses a non-empty directory, and home always holds dotfiles,
  # so initialise in place and check out main over it.
  git init -q
  git remote add origin "$REPO_URL"
  git fetch -q origin main
  git checkout -q -f -B main origin/main
fi
git log --oneline -n 1

step "Python 3.12 environment from requirements.txt"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -q -r requirements.txt
.venv/bin/python -c "import torch, vllm; print('torch', torch.__version__, '| cuda ok:', torch.cuda.is_available(), '| vllm', vllm.__version__)"

step "Splits (rebuilt here, then verified against the laptop)"
.venv/bin/python main.py build-data
for split in train val test; do
  actual=$(sha256sum "data/splits/$split.jsonl" | cut -c1-16)
  if [ "$actual" != "${EXPECTED_SPLIT_HASH[$split]}" ]; then
    echo "SPLIT MISMATCH: $split is $actual, expected ${EXPECTED_SPLIT_HASH[$split]}" >&2
    exit 1
  fi
  echo "  $split.jsonl matches ($actual)"
done

step "vLLM server (tmux session 'vllm')"
mkdir -p logs
if ! tmux has-session -t vllm 2>/dev/null; then
  # VLLM_USE_FLASHINFER_SAMPLER=0: FlashInfer's sampler JIT-compiles CUDA
  # kernels on first use, which needs ninja and nvcc, and the image has neither.
  # PyTorch's sampler draws from the same distribution, and the P(yes) readout
  # comes from the logprobs, not from the sampler, so scores are unaffected.
  tmux new-session -d -s vllm \
    "VLLM_USE_FLASHINFER_SAMPLER=0 .venv/bin/vllm serve $MODEL --port 8000 \
     --gpu-memory-utilization 0.85 --max-model-len 8192 --max-logprobs 20 \
     2>&1 | tee logs/vllm.log"
fi
printf 'waiting for the server (first start downloads ~15 GB)'
for _ in $(seq 1 180); do
  if curl -fs localhost:8000/health >/dev/null 2>&1; then echo " ready"; break; fi
  if ! tmux has-session -t vllm 2>/dev/null; then
    echo; echo "vLLM exited during startup. Last lines of logs/vllm.log:" >&2
    grep -E "Error|error" logs/vllm.log | tail -5 >&2
    exit 1
  fi
  printf '.'; sleep 5
done
curl -fs localhost:8000/health >/dev/null || { echo "vLLM did not come up; see logs/vllm.log" >&2; exit 1; }

step "Smoke test: score 4 val items through the real backend"
rm -f data/generations/m0__val__limit4.json
.venv/bin/python main.py score --arm m0 --split val --limit 4
.venv/bin/python - <<'PY'
import json
rows = json.load(open("data/generations/m0__val__limit4.json"))
readable = sum(r["logprob_score"] is not None and r["text_score"] is not None for r in rows)
print(f"  {readable}/{len(rows)} rows have both readouts · backend {rows[0]['backend']} · model {rows[0]['model']}")
assert readable == len(rows), "readouts failed to parse -- do not start the real run"
PY
rm -f data/generations/m0__val__limit4.json

step "Ready. Start the real run with: cd ~ && tmux new -d -s run 'bash infra/vast/run_untrained_arms.sh' && tmux switch-client -t run"
