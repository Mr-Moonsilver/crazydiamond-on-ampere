#!/bin/bash
# 8 x 153,537 prompt, 8192 output, custom all-reduce, full 1.8 GB pool.
#   m: MTP-1 + 8192 slots   (more slots than k2's 2048: does the hit rate / speed grow?)
#   n: MTP off + 16384 slots (draft weights and the draft KV layer gone: speed, blocks/request)
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
C=(DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False MAX_SEQS=8)
./flash-quick.sh m-150rc8k 8 153537 8192 "${C[@]}" "EXTRA_MOUNT=$M" "EXTRA_ENV=QSA_ROWCACHE_SLOTS=8192 QSA_ROWCACHE_REQS=8"
./flash-quick.sh n-150mtp0rc16k 8 153537 8192 "${C[@]}" MTP_TOKENS=0 "EXTRA_MOUNT=$M" "EXTRA_ENV=QSA_ROWCACHE_SLOTS=16384 QSA_ROWCACHE_REQS=8"
