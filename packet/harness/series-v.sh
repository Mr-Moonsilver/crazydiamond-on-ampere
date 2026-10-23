#!/bin/bash
# Text check at the t1 setup: same 8 x 32k run without and with the cache, saving outputs and a repetition
# score (8-gram repeats), then the 150k finale (series-u).
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
C=(DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False GPU_KV_BYTES=1250000000 MAX_SEQS=8)
./flash-quick.sh v1-32k-nocache 8 32768 2048 "${C[@]}"
./flash-quick.sh v2-32k-cache 8 32768 2048 "${C[@]}" "EXTRA_MOUNT=$M" "EXTRA_ENV=QSA_ROWCACHE_SLOTS=8192 QSA_ROWCACHE_REQS=10"
[ -e /data/vllm/gpu-tuning-20261021/flash-quick/STOP-AFTER-V ] && exit 0
./series-u.sh
