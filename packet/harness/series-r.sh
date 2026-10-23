#!/bin/bash
# Diagnose q's 7% hit rate: identity build at the known-good 32k MTP-1 setup (r1), then q's setup at 32k (r2).
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
C=(DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False)
./flash-quick.sh r1-32k-mtp1 8 32768 2048 "${C[@]}" GPU_KV_BYTES=1250000000 MAX_SEQS=8 "EXTRA_MOUNT=$M" \
  "EXTRA_ENV=QSA_ROWCACHE_SLOTS=8192 QSA_ROWCACHE_REQS=8"
./flash-quick.sh r2-32k-x12mtp0 12 32768 4096 "${C[@]}" MAX_SEQS=12 MTP_TOKENS=0 PP_LAYER_PARTITION=24,24 "EXTRA_MOUNT=$M" \
  "EXTRA_ENV=QSA_ROWCACHE_SLOTS=12288 QSA_ROWCACHE_REQS=12"
