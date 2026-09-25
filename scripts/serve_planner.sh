#!/usr/bin/env bash
# The planner model for Duck Studio, served on this machine's Tailscale address only.
#
#   scripts/serve_planner.sh            # E2B (p002: 28/30 in 8.5 s, thinking off)
#   MODEL=~/models/gemma-4-E4B_q4_0-it.gguf scripts/serve_planner.sh
#
# In Duck Studio: Settings → models → add an OpenAI-compatible server,
# address http://<this machine's tailnet IP>:8081/v1. Only devices on your tailnet can reach it;
# llama-server itself has no password, so do not bind it to 0.0.0.0.
set -euo pipefail
MODEL=${MODEL:-$HOME/models/gemma-4-E2B_q4_0-it.gguf}
BIND=${BIND:-$(ip -4 -o addr show tailscale0 | awk '{print $4}' | cut -d/ -f1)}
[ -n "$BIND" ] || { echo "no tailscale0 address yet" >&2; exit 1; }
exec "$HOME/.local/llama/llama-b11188/llama-server" -m "$MODEL" -t 16 -c 4096 \
  --host "$BIND" --port 8081 --reasoning off --alias gemma-4-planner
