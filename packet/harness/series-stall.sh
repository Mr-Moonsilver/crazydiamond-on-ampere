#!/bin/bash
# Decode stall during a fresh 200k prefill, chunk 1024 vs 2048, final configuration (MTP off, prefix
# caching, row cache, 8 seqs). 4 agents at 32k generating; one fresh 200k prompt arrives.
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
C=(DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False MAX_SEQS=8 MTP_TOKENS=0 PP_LAYER_PARTITION=24,24
   PREFIX_CACHING=1 QUICK_WINDOW_S=30 "EXTRA_MOUNT=$M" "EXTRA_ENV=QSA_ROWCACHE_SLOTS=12288 QSA_ROWCACHE_REQS=8")
for chunk in 1024 2048; do
  ./flash-s.sh s-stall-chunk$chunk 4 32768 512 "${C[@]}" PREFILL_CHUNK=$chunk "STALL=4 32768 200000"
done
