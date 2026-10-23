#!/bin/bash
# w7d: 10 agents at the native context (chunk 2048 chosen by the stall test). w7a: 8 x 258k at 72.1 %, w7c: 9 at 81.1 %.
#   w7c: 9 x (250,000 + 8,192), pool 1.8 GB, 12,288 slots
#   w7d: 10 x (250,000 + 8,192), pool 1.8 GB, 10,240 slots (keeps row-cache VRAM ~ as at 8-9 streams)
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
C=(DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False MTP_TOKENS=0 PP_LAYER_PARTITION=24,24
   PREFIX_CACHING=1 QUICK_WINDOW_S=90 PREFILL_CHUNK=2048 "EXTRA_MOUNT=$M")
./flash-x.sh w7d-262x10-pc 10 250000 8192 "${C[@]}" MAX_SEQS=10 "EXTRA_ENV=QSA_ROWCACHE_SLOTS=10240 QSA_ROWCACHE_REQS=10"
