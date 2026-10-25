#!/usr/bin/env bash
# S5 production-kernel slice, both TP pairs (Codex round 6 draft, packaged as harness/s5_queue.sh; run by q8 with the engine
# already stopped. This file neither starts nor stops vllm-flash.
# Inputs: H=checksummed harness dir, Q=new artifact parent, S5_CONFIG=model config,
# S5_RUN_ID=unique run id. Optional S5_DISPATCH_EVIDENCE is a reviewed JSON record.
set -u
: "${H:?checksummed harness directory}" "${Q:?artifact parent}" "${S5_CONFIG:?model config.json}"
: "${S5_RUN_ID:?unique run id, e.g. s5-20261024a}"
[[ $S5_RUN_ID =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
S5_OVERLAY=${S5_OVERLAY:-/data/vllm/flash-next-tp4/prod/overlay-v16}
IMG=$(cat /tmp/flash-image.txt)
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
(cd "$H" && sha256sum --quiet -c SHA256SUMS) || exit 1
(cd "$S5_OVERLAY" && sha256sum --quiet -c SHA256SUMS) || exit 1
source "$H/qlib.sh"
gpu_idle || { echo 'S5 INCONCLUSIVE: engine/GPU still in use'; exit 2; }
# qlib's gpu_idle alone treats nvidia-smi failure like empty output; close that here.
users=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 2
[[ -z $users ]] || exit 2
OUT="$Q/$S5_RUN_ID"
[[ ! -e $OUT ]] || { echo 'refusing to overwrite prior S5 artifacts'; exit 2; }
mkdir -p "$OUT"
cp "$S5_CONFIG" "$OUT/model-config.json"
sha256sum "$S5_CONFIG" "$S5_OVERLAY/ops_qsa.py" > "$OUT/input-sha256.txt"
docker image inspect "$IMG" --format '{{.Id}} {{json .RepoDigests}}' > "$OUT/image.txt" || exit 2
extra=(--modes "${S5_MODES:-decode,prefill,serial,prio,green24,green34,green44}"
       --mode-timeout "${S5_MODE_TIMEOUT:-120}")
if [[ -n ${S5_DISPATCH_EVIDENCE:-} ]]; then
  cp "$S5_DISPATCH_EVIDENCE" "$OUT/dispatch-evidence.json"
  extra+=(--dispatch-evidence /q/dispatch-evidence.json)
fi
worst=0
active=''
cleanup() { [[ -z $active ]] || docker rm -f "$active" > "$OUT/interrupted-cleanup.log" 2>&1 || true; }
trap cleanup EXIT
trap 'exit 130' INT TERM
for pair in 1,3 0,2; do
  n="s5-${S5_RUN_ID}-${pair/,/-}"
  [[ -z $(docker ps -aq -f "name=^${n}$") ]] || exit 2
  gpu_idle || exit 2
  users=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || exit 2
  [[ -z $users ]] || exit 2
  active=$n
  # Cleanup applies only to this newly-created test container. No engine restore.
  timeout 2400 docker run --rm --name "$n" --gpus all --ipc=host --network host \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="$pair" \
    -e NCCL_P2P_LEVEL=SYS -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN \
    -e VLLM_GDN_DECODE_KERNEL=cuda -e QSA_RETAIN_HISTORY=0 -e QSA_STAGE_TIMING=0 \
    -e VLLM_CACHE_ROOT=/q/cache/vllm -e TRITON_CACHE_DIR=/q/cache/triton \
    -e TORCH_EXTENSIONS_DIR=/q/cache/extensions \
    -v "$H:/h:ro" -v "$OUT:/q" \
    -v "$S5_OVERLAY/qsa.py:$D/qsa.py:ro" -v "$S5_OVERLAY/ops_qsa.py:$D/ops/qsa.py:ro" \
    --entrypoint python3 "$IMG" -m torch.distributed.run --nproc_per_node 2 \
    /h/s5_segment.py --config /q/model-config.json --pair "$pair" --layers 4 --steps "${S5_STEPS:-400}" \
    --gdn-decode "${S5_GDN_DECODE:-cuda}" --out "/q/pair-${pair/,/-}.json" "${extra[@]}" \
    > "$OUT/pair-${pair/,/-}.log" 2>&1
  rc=$?
  docker rm -f "$n" > "$OUT/cleanup-${pair/,/-}.log" 2>&1 || true
  [[ -z $(docker ps -aq -f "name=^${n}$") ]] || { echo "S5 container still present: $n"; exit 1; }
  active=''
  gpu_idle || exit 2
  echo "S5 pair=$pair exit=$rc (0 GO, 1 FAIL, 2 INCONCLUSIVE, 3 NO-GO; torchrun may wrap nonzero exits)"
  [[ $rc == 0 ]] || worst=1
done
# Re-grade artifacts; launcher exit codes are not a substitute for both-rank evidence.
python3 "$H/s5_segment.py" --combine "$OUT/pair-0-2.json" "$OUT/pair-1-3.json" \
  --out "$OUT/combined.json"
combined=$?
[[ $combined != 0 || $worst == 0 ]] || exit 2
exit "$combined"
