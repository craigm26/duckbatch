#!/usr/bin/env bash
# Pollen's shipped policies, pinned to a revision and checked by sha256 (Apache-2.0).
set -euo pipefail
cd "$(dirname "$0")/../teachers"
REV=1b56c396825c052a4e26e95cf2b8d8298af9e9b4
BASE="https://huggingface.co/pollen-robotics/microduck-policies/resolve/$REV"
fetch() { [ -f "$1" ] || curl -fsSL -o "$1" "$BASE/$1"; echo "$2  $1" | sha256sum -c -; }
fetch velstand.onnx      1c659be55da94bc5753b707de5c6a3e7c49931e05ca3b6991615cef1a8ba9a45
fetch alpha_walking.onnx e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c
