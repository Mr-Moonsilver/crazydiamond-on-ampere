#!/bin/bash
# Realistic agentic-coding stall: 8 agents generating at ~100k, a 9th arrives with a fresh 200k prompt.
# Uses the chunk chosen after the stall test (flash-quick/BEST_CHUNK). MAX_SEQS 9 so the 9th is admitted.
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
CHUNK=$(cat /data/vllm/gpu-tuning-20261021/flash-quick/BEST_CHUNK 2>/dev/null || echo 2048)
./flash-s.sh s-agents8-chunk$CHUNK 4 32768 512 DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False MAX_SEQS=9 \
  MTP_TOKENS=0 PP_LAYER_PARTITION=24,24 PREFIX_CACHING=1 QUICK_WINDOW_S=30 PREFILL_CHUNK=$CHUNK "EXTRA_MOUNT=$M" \
  "EXTRA_ENV=QSA_ROWCACHE_SLOTS=12288 QSA_ROWCACHE_REQS=9" "STALL=8 100000 200000"
