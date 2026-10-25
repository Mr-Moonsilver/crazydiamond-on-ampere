#!/usr/bin/env bash
# Container entry only, no service management. Inputs are validated by the queue.
set -Eeuo pipefail
rung=$1;pair=$2;split=$3;rows=$4;out=$5
if [[ ${FLASH_MARLIN_GREEN_CAP:-0} == 1 ]];then
  # New case directory only. No wheel rebuild or installed-file modification.
  export S5_MARLIN_CAP_LIBRARY="${out%.json}-marlin-cap.so"
  [[ ! -e $S5_MARLIN_CAP_LIBRARY ]] || exit 2
  timeout 90 g++ -std=c++17 -O2 -shared -fPIC -I/usr/local/cuda/include \
    /h/s5_marlin_cap.cpp -L/usr/local/cuda/lib64/stubs -lcuda -ldl \
    -o "$S5_MARLIN_CAP_LIBRARY"
  sha256sum /h/s5_marlin_cap.cpp "$S5_MARLIN_CAP_LIBRARY" > "${out%.json}-marlin-cap.sha256"
  export LD_PRELOAD="$S5_MARLIN_CAP_LIBRARY${LD_PRELOAD:+:$LD_PRELOAD}"
fi
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
