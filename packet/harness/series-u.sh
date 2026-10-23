#!/bin/bash
# 150k finale with the identity-keyed row cache (custom all-reduce on, full GPU pool).
#   u1: MTP-1, 8 x 153,537, 8192 out, 8192 slots      (MTP-1 fits at most 9 streams at 150k)
#   u2: MTP off, 12 x 153,537, 16384 out, layers 24/24, 12288 slots
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
C=(DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False)
./flash-quick.sh u1-150x8mtp1 8 153537 8192 "${C[@]}" MAX_SEQS=8 "EXTRA_MOUNT=$M" \
  "EXTRA_ENV=QSA_ROWCACHE_SLOTS=8192 QSA_ROWCACHE_REQS=8"
./flash-quick.sh u2-150x12mtp0 12 153537 16384 "${C[@]}" MAX_SEQS=12 MTP_TOKENS=0 PP_LAYER_PARTITION=24,24 \
  "EXTRA_MOUNT=$M" "EXTRA_ENV=QSA_ROWCACHE_SLOTS=12288 QSA_ROWCACHE_REQS=12"
