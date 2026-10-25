#!/usr/bin/env bash
# Container entry only, no service management. Inputs are validated by the queue.
set -Eeuo pipefail
rung=$1;pair=$2;split=$3;rows=$4;out=$5
args=(python3 -m torch.distributed.run --nproc_per_node 2 /h/s5_variants.py
      --rung "$rung" --pair "$pair" --split "$split" --rows "$rows" --config /q/model-config.json --out "$out")
if [[ $rung == bandwidth ]];then
  # Sample device-wide DRAM/SM metrics; never use replay-based Nsight Compute to
  # infer concurrent throughput. Missing profiler/metrics is not proof of DRAM.
  command -v nsys >/dev/null || { echo 'INCONCLUSIVE nsys unavailable for sampled DRAM metrics';exit 2; }
  exec nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx \
    --gpu-metrics-device=all --gpu-metrics-frequency=1000 --output "${out%.json}-metrics" "${args[@]}"
fi
exec "${args[@]}"
