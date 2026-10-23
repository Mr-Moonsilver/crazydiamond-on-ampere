#!/bin/bash
# Low concurrency with today's stack (custom all-reduce + identity-keyed row cache): 1 and 4 streams at
# ~150k, MTP-1 and MTP off. One server per MTP setting; quick_c8 runs back to back in the same server.
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
C=(DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False QUICK_WINDOW_S=60 "EXTRA_MOUNT=$M")
./flash-x.sh x1-c1-mtp1 1 153537 2048 "${C[@]}" MAX_SEQS=8 LOWC=4 "EXTRA_ENV=QSA_ROWCACHE_SLOTS=8192 QSA_ROWCACHE_REQS=8"
./flash-x.sh x2-c1-mtp0 1 153537 2048 "${C[@]}" MAX_SEQS=12 MTP_TOKENS=0 PP_LAYER_PARTITION=24,24 LOWC=4 \
  "EXTRA_ENV=QSA_ROWCACHE_SLOTS=12288 QSA_ROWCACHE_REQS=12"
