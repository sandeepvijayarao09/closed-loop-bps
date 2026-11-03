#!/usr/bin/env bash
# Package trained decoders + encoders for a GitHub release, so readers can skip the ~40 min pipeline.
# Usage: scripts/make_release_asset.sh [version]   -> dist/closed-loop-bps-artifacts-<version>.tar.gz + SHA256SUMS
set -euo pipefail
cd "$(dirname "$0")/.."
VER=${1:-v0.2}
mkdir -p dist
NAME=closed-loop-bps-artifacts-$VER.tar.gz
tar -czf "dist/$NAME" artifacts/*.pkl
( cd dist && shasum -a 256 "$NAME" > SHA256SUMS && cat SHA256SUMS )
echo "Upload dist/$NAME as a release asset; unpack at the repo root to restore artifacts/."
