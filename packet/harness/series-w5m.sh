#!/bin/bash
# Operator decision 2026-10-22: if prefix caching works with MTP off, MTP stays off and the machine is tuned
# for total throughput. Prefill chunk sizes on the MTP-off configuration: 12 streams, layers 24/24,
# prefix caching on, row cache 12,288 slots. 12 x 32k prefill bursts + 12 x 32k decode per chunk size.
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
C=(DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False MAX_SEQS=12 MTP_TOKENS=0 PP_LAYER_PARTITION=24,24
   PREFIX_CACHING=1 QUICK_WINDOW_S=90 "EXTRA_MOUNT=$M" "EXTRA_ENV=QSA_ROWCACHE_SLOTS=12288 QSA_ROWCACHE_REQS=12")
for chunk in 512 1024 2048; do
  ./flash-x.sh w5m-chunk$chunk 12 32768 4096 "${C[@]}" PREFILL_CHUNK=$chunk PREFILL_PROBE=1
done
./flash-x.sh x2-c1-mtp0 1 153537 2048 "${C[@]}" LOWC=4
