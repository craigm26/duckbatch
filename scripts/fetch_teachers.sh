#!/usr/bin/env bash
# Pollen's shipped policies, pinned to a revision and checked by sha256 (Apache-2.0).
set -euo pipefail
cd "$(dirname "$0")/../teachers"
REV=1b56c396825c052a4e26e95cf2b8d8298af9e9b4
BASE="https://huggingface.co/pollen-robotics/microduck-policies/resolve/$REV"
fetch() { [ -f "$1" ] || curl -fsSL -o "$1" "$BASE/$1"; echo "$2  $1" | sha256sum -c -; }
fetch velstand.onnx      1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45
fetch alpha_walking.onnx e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c
fetch alpha_stand.onnx       1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b
fetch alpha_sitstand.onnx    c6c40e35e726eabd803d633e090d112994f469921152448367953fbaf9799bc8
fetch alpha_ground_pick.onnx ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872
fetch ball_kick_left.onnx    d6928284dccd3dd61e08bf2f760effa74309fbefd97b2b31afb2a60f526d196a
fetch ball_kick_right.onnx   147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af
