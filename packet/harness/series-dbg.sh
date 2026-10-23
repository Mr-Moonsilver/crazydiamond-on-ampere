#!/bin/bash
# Eager probe: does batch index i refer to the same request across decode steps (PP2, MTP-1)?
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.dbg.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
./flash-quick.sh s-dbg 8 2048 64 EAGER=1 DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False \
  GPU_KV_BYTES=1250000000 MAX_SEQS=8 "EXTRA_MOUNT=$M" \
  "EXTRA_ENV=QSA_ROWCACHE_SLOTS=4096 QSA_ROWCACHE_REQS=8 QSA_ROWCACHE_DEBUG=1"
