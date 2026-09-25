#!/usr/bin/env bash
# Assemble the walker arena Space: arena/ (this repo) + duckbench's runtime, pinned.
#
#   DUCKBENCH=~/ref/duckbench scripts/build_arena.sh [out-dir]
#
# The physics loop, the scene, the duck's meshes and the vendored MuJoCo/onnxruntime are
# duckbench's, copied at the commit recorded in BUILD.txt, so the arena runs the exact files
# duckbench's parity gates cover. Nothing is re-implemented here except the multi-lane view.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${DUCKBENCH:-$HOME/ref/duckbench}"
OUT="${1:-$HERE/build/arena}"
[ -f "$SRC/site/duckloop.mjs" ] || { echo "no duckbench at $SRC (set DUCKBENCH)"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT/vendor"
cp "$HERE"/arena/{index.html,arena.js,arena-render.js,policies.json,README.md} "$OUT/"
cp "$SRC"/site/{duckloop.mjs,stairs.js,duckkit-constants.json,scene.mjb,duck-visual.bin} "$OUT/"
cp -r "$SRC"/site/vendor/{mujoco.js,mujoco.wasm,ort} "$OUT/vendor/"
{
  echo "duckbench $(git -C "$SRC" rev-parse HEAD)"
  echo "duckbatch $(git -C "$HERE" rev-parse HEAD)"
  echo "built $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/BUILD.txt"
echo "arena -> $OUT"; cat "$OUT/BUILD.txt"; du -sh "$OUT"
