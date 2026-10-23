#!/bin/bash
# q: MTP off, 12 x 153,537 prompt, 16384 output (12 serialized prefills need a longer output for a
# common 12-stream window), layers 24/24, 12,288 cache slots per request; then the churn retrieval test.
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
./flash-quick.sh q-150x12mtp0 12 153537 16384 DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False \
  MAX_SEQS=12 MTP_TOKENS=0 PP_LAYER_PARTITION=24,24 CHURN=24 "EXTRA_MOUNT=$M" \
  "EXTRA_ENV=QSA_ROWCACHE_SLOTS=12288 QSA_ROWCACHE_REQS=12"
