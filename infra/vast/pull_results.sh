#!/usr/bin/env bash
# Copy results and raw scores from the GPU box back to the laptop. Run ON THE
# LAPTOP from the repo root:
#
#   bash infra/vast/pull_results.sh            # uses the first running instance
#   bash infra/vast/pull_results.sh 12345678   # or a specific instance id
#
# If the key registered with vast.ai is not your default SSH key, point at it:
#
#   VAST_SSH_KEY=~/.ssh/arena_key bash infra/vast/pull_results.sh
#
# results/, data/ and the LoRA adapters travel back; the base model weights and
# the environment stay on the box and die with it.
set -euo pipefail

instance_id="${1:-$(uvx vastai show instances --raw | python3 -c \
  'import json,sys; r=[i for i in json.load(sys.stdin) if i.get("actual_status")=="running"]; print(r[0]["id"] if r else "")')}"
[ -n "$instance_id" ] || { echo "no running instance found" >&2; exit 1; }

url=$(uvx vastai ssh-url "$instance_id")        # ssh://root@HOST:PORT
host_port=${url#ssh://}
host=${host_port%:*}
port=${host_port##*:}
ssh_cmd="ssh -p $port -o StrictHostKeyChecking=accept-new${VAST_SSH_KEY:+ -i $VAST_SSH_KEY}"

# Plain -az only: macOS ships openrsync, which rejects GNU-only options such as
# --info. A rejected option aborts before copying anything.
# A directory a given run never wrote (data/training on an M6/M7 box, say) is
# not an error. Without this guard rsync's failure aborts the whole pull under
# `set -e`, and the later directories -- including runs/ -- are never copied.
for dir in results data/generations data/training runs logs; do
  if ! $ssh_cmd "$host" "test -d $dir" 2>/dev/null; then
    echo "  skip     $dir (not on this box)"
    continue
  fi
  mkdir -p "$dir"
  rsync -az -e "$ssh_cmd" "$host:$dir/" "$dir/"
done

# Verify before anyone destroys the instance. Destroying deletes the VM's disk,
# so this check is the only thing standing between a failed copy and lost runs.
missing=0
for f in results/val/report.md results/test/report.md; do
  if [ -s "$f" ]; then echo "  ok       $f"; else echo "  MISSING  $f" >&2; missing=1; fi
done
generations=$(ls data/generations/*.json 2>/dev/null | wc -l | tr -d ' ')
echo "  ok       $generations score files in data/generations/"
adapters=$(ls -d runs/*/adapter 2>/dev/null | wc -l | tr -d ' ')
echo "  ok       $adapters trained adapters in runs/"

if [ "$missing" -ne 0 ] || [ "$generations" -eq 0 ]; then
  echo >&2
  echo "PULL INCOMPLETE. Do NOT destroy instance $instance_id." >&2
  exit 1
fi
echo
echo "Pulled and verified. Safe to destroy: uvx vastai destroy instance $instance_id"
