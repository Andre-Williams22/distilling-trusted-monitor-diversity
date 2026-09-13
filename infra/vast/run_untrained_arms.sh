#!/usr/bin/env bash
# The real scoring run for the untrained arms, on the GPU box. Run ON THE VM,
# inside tmux so it survives a dropped SSH connection:
#
#   cd ~/dtmd && tmux new -s run 'bash infra/vast/run_untrained_arms.sh'
#
# Order matters:
#   1. val before test -- test thresholds are set on val.
#   2. M1 before M0 -- M0 is taken from M1's first sample (preregistered).
# Every scoring step resumes, so re-running after an interruption only does
# the remaining items. The whole log is kept in logs/.
set -euo pipefail
cd "$HOME/dtmd"
mkdir -p logs
exec > >(tee -a "logs/run_untrained_$(date +%Y%m%d_%H%M%S).log") 2>&1

curl -fs localhost:8000/health >/dev/null || {
  echo "vLLM is not serving on :8000 -- run infra/vast/remote_setup.sh first" >&2
  exit 1
}

for split in val test; do
  .venv/bin/python main.py score --arm m1 --split "$split"
  .venv/bin/python main.py derive-m0 --split "$split"
  .venv/bin/python main.py score --arm m2 --split "$split"
done

.venv/bin/python main.py analyse --split val
.venv/bin/python main.py analyse --split test

echo
echo "Done. From the laptop: bash infra/vast/pull_results.sh"
