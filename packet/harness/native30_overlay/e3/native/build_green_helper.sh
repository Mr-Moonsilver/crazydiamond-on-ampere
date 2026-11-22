#!/bin/bash
# Owner-only CPU compiler container; new output, no GPU or engine access.
# Same sm_86/shared-cudart recipe qualified by q17a. Container is retained.
# Final overlay manifest and declared FLASH_S5_GREEN_LIB_SHA256 must cover
# the new library BEFORE source declaration or any test-engine start.
set -eu
img=$1; dir=$(cd "$2" && pwd)
[[ "$img" == sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b ]]
[[ ! -e "$dir/flash_s5_green.so" && ! -L "$dir/flash_s5_green.so" ]]
name="e3-green-build-$(date -u +%Y%m%dT%H%M%SZ)-$$"
# Inner timeout terminates compiler before the client bound.
timeout --signal=TERM --kill-after=20s 300s docker run --name "$name" --network none   -v "$dir:/native" --entrypoint timeout "$img" --signal=TERM --kill-after=10s 240s   nvcc -O2 -std=c++17 -shared -Xcompiler=-fPIC   --generate-code=arch=compute_86,code=sm_86 --cudart=shared -lcuda   /native/flash_s5_green.cu -o /native/flash_s5_green.so
sha256sum "$dir/flash_s5_green.so"
echo "retained build container: $name"
