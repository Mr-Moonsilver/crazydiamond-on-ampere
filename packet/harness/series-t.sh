#!/bin/bash
# Identity-table row cache: eager mapping probe, then the real 32k hit rate/speed + churn, then a no-cache churn reference.
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
C=(DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False GPU_KV_BYTES=1250000000 MAX_SEQS=8)
./flash-quick.sh t0-dbg 8 2048 64 EAGER=1 "${C[@]}" "EXTRA_MOUNT=$M" "EXTRA_ENV=QSA_ROWCACHE_SLOTS=4096 QSA_ROWCACHE_REQS=10 QSA_ROWCACHE_DEBUG=1"
./flash-quick.sh t1-32k-id 8 32768 2048 "${C[@]}" CHURN=24 "EXTRA_MOUNT=$M" "EXTRA_ENV=QSA_ROWCACHE_SLOTS=8192 QSA_ROWCACHE_REQS=10"
./flash-quick.sh t3-nocache-churn 8 2048 64 "${C[@]}" CHURN=24
