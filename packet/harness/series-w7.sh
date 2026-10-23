#!/bin/bash
# Operator (agentic coding): 8 agents at the native 262k context instead of 12 x 150k. Prefix caching on,
# MTP off, layers 24/24, max-num-seqs 8 (9 row-cache rows instead of 13), larger GPU pool for headroom.
#   w7a: 8 x 254,000 prompt + 4096 out, pool 1.8 GB (baseline capacity 8.01x at 262k)
#   w7b: same with pool 2.1 GB (uses the VRAM freed by the smaller row cache)
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
C=(DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False MAX_SEQS=8 MTP_TOKENS=0 PP_LAYER_PARTITION=24,24
   PREFIX_CACHING=1 QUICK_WINDOW_S=90 "EXTRA_MOUNT=$M" "EXTRA_ENV=QSA_ROWCACHE_SLOTS=12288 QSA_ROWCACHE_REQS=8")
CHUNK=$(cat /data/vllm/gpu-tuning-20261021/flash-quick/BEST_CHUNK 2>/dev/null || echo 512)
./flash-x.sh w7a-262x8-pc 8 254000 8192 "${C[@]}" PREFILL_CHUNK=$CHUNK
# a larger pool also grows the pinned host KV (allocated for every block): raise the startup guard on purpose
./flash-x.sh w7b-262x8-pc-pool21 8 254000 8192 "${C[@]}" PREFILL_CHUNK=$CHUNK GPU_KV_BYTES=2100000000 HOST_KV_GIB_PER_RANK=32
# the edge: 9 agents at 262k (row cache needs one more row)
C9=("${C[@]/QSA_ROWCACHE_REQS=8/QSA_ROWCACHE_REQS=9}"); C9=("${C9[@]/MAX_SEQS=8/MAX_SEQS=9}")
./flash-x.sh w7c-262x9-pc-pool21 9 254000 8192 "${C9[@]}" PREFILL_CHUNK=$CHUNK GPU_KV_BYTES=2100000000 HOST_KV_GIB_PER_RANK=32
