#!/bin/bash
# Batch W (operator, 2026-10-22): prefix caching (MTP off, then MTP-1), MTP-2 with the row cache, larger
# prefill chunks. All with custom all-reduce, identity-keyed row cache, full GPU pool unless noted.
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
P=/data/vllm/gpu-tuning-20261021/probe
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
M="$P/qsa.rowcache.py:$D/qsa.py:ro $P/ops_qsa.rowcache.py:$D/ops/qsa.py:ro"
# QUICK_WINDOW_S: stop each decode run 90 s after the last stream started (the measured window)
C=(DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False MAX_SEQS=8 QUICK_WINDOW_S=90 "EXTRA_MOUNT=$M")
RC8="EXTRA_ENV=QSA_ROWCACHE_SLOTS=8192 QSA_ROWCACHE_REQS=8"
# reference for the loop question: 8 x 153k, 8192 out, no row cache, outputs saved (compare repetition with u1)
./flash-w.sh w0-150x8-nocache 8 153537 8192 DISABLE_CUSTOM_ALL_REDUCE=0 ALLOC_CONF=expandable_segments:False MAX_SEQS=8
# prefix caching: multi-turn 150k conversations (TTFT per turn + answers), 8 x 32k decode, churn
./flash-w.sh w1-pc-mtp0 8 32768 2048 "${C[@]}" MTP_TOKENS=0 PREFIX_CACHING=1 PREFIX_TEST=4 CHURN=24 "$RC8"
./flash-w.sh w2-pc-mtp1 8 32768 2048 "${C[@]}" PREFIX_CACHING=1 PREFIX_TEST=4 CHURN=24 "$RC8"
# MTP-2 with the row cache (32k vs v2's 612 with MTP-1; then 150k vs u1)
./flash-w.sh w3-mtp2-32k 8 32768 2048 "${C[@]}" MTP_TOKENS=2 GPU_KV_BYTES=1250000000 "$RC8"
./flash-w.sh w4-mtp2-150k 8 153537 8192 "${C[@]}" MTP_TOKENS=2 "$RC8"
# prefill chunk size: 8 x 32k one-output-token bursts (+ decode) at 512 / 1024 / 2048
for chunk in 512 1024 2048; do
  ./flash-w.sh w5-chunk$chunk 8 32768 2048 "${C[@]}" GPU_KV_BYTES=1250000000 PREFILL_CHUNK=$chunk PREFILL_PROBE=1 "$RC8"
done
