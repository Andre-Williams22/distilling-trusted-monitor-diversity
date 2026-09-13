#!/usr/bin/env bash
# Copy results and raw scores from the GPU box back to the laptop. Run ON THE
# LAPTOP from the repo root:
#
#   bash infra/vast/pull_results.sh            # uses the first running instance
#   bash infra/vast/pull_results.sh 12345678   # or a specific instance id
#
# Only results/ and data/generations/ travel back; the model weights and the
# environment stay on the box and die with it.
set -euo pipefail

instance_id="${1:-$(uvx vastai show instances --raw | python3 -c \
  'import json,sys; r=[i for i in json.load(sys.stdin) if i.get("actual_status")=="running"]; print(r[0]["id"] if r else "")')}"
[ -n "$instance_id" ] || { echo "no running instance found" >&2; exit 1; }

url=$(uvx vastai ssh-url "$instance_id")        # ssh://root@HOST:PORT
host_port=${url#ssh://}
host=${host_port%:*}
port=${host_port##*:}
ssh_cmd="ssh -p $port -o StrictHostKeyChecking=accept-new"

for dir in results data/generations logs; do
  mkdir -p "$dir"
  rsync -az --info=stats1 -e "$ssh_cmd" "$host:dtmd/$dir/" "$dir/"
done
echo "pulled results/, data/generations/ and logs/ from instance $instance_id"
