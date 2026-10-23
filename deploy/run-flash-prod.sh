#!/usr/bin/env bash
# Production launcher: Qwen3.8-Flash-Next W4A16 on CT <ct-id>, TP2 x PP2 / EP2, host QSA KV + PLE, custom
# all-reduce, per-request VRAM row cache for selected QSA rows (overlay in $OVERLAY, checksummed).
# Settings come from /etc/default/vllm-flash (see vllm-flash.service). Foreground: systemd owns the lifetime.
set -euo pipefail
task_dir=/data/vllm/flash-next-tp4
expected_image=sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b
name=qwen-flash-prod
OVERLAY=${OVERLAY:-$task_dir/prod/overlay}
[[ $(sudo docker image inspect "$expected_image" --format '{{.Id}}') == "$expected_image" ]] || exit 2
(cd "$OVERLAY" && sha256sum --quiet -c SHA256SUMS) || { echo "overlay checksum mismatch" >&2; exit 2; }
sudo docker rm -f "$name" >/dev/null 2>&1 || true
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
default_compilation='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,6,8,9,12,15,18,21,24]}'
compilation=${COMPILATION_CONFIG:-$default_compilation}
spec=()
if [[ ${MTP_TOKENS:-0} != 0 ]]; then
  printf -v spec_json '{"method":"mtp","num_speculative_tokens":%d}' "$MTP_TOKENS"
  spec=(--speculative-config "$spec_json")
fi
exec sudo docker run --rm --init --name "$name" --label qwen.flash-next-production=true \
  --gpus all --ipc=host --network host \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="${GPU_ORDER:-0,2,1,3}" \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN \
  -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e OMP_NUM_THREADS=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_QSA_KV_OFFLOAD=1 -e VLLM_QSA_KV_OFFLOAD_MAX_GIB="${HOST_KV_GIB_PER_RANK:-28}" \
  -e VLLM_QSA_KVO_ARENA=100663296 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_QSA_KVO_SPARSE_PREFILL=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e VLLM_PP_LAYER_PARTITION="${PP_LAYER_PARTITION:-26,22}" \
  -e QSA_ROWCACHE_SLOTS="${QSA_ROWCACHE_SLOTS:-8192}" -e QSA_ROWCACHE_REQS="${MAX_SEQS:-8}" \
  -e VLLM_CACHE_ROOT=/work/cache/vllm -e TRITON_CACHE_DIR=/work/cache/triton \
  -e FLASHINFER_WORKSPACE_BASE=/work/cache/flashinfer \
  -v "$task_dir/models/intel-w4a16:/model:ro" -v "$task_dir:/work" \
  -v "$OVERLAY/qsa.py:$D/qsa.py:ro" -v "$OVERLAY/ops_qsa.py:$D/ops/qsa.py:ro" \
  "$expected_image" /model \
  --served-model-name qwen38-w8a8-tp4 Qwen3.8-Flash-Next Intel/Qwen3.8-Flash-Next-W4A16-AutoRound \
    cyankiwi/Qwen3.8-27B-AWQ-INT4 cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit \
  --tensor-parallel-size 2 --pipeline-parallel-size 2 --enable-expert-parallel \
  --dtype bfloat16 --language-model-only --engram-config '{"cpu_offload":true}' \
  --kv-cache-dtype auto --kv-cache-memory-bytes "${GPU_KV_BYTES:-1800000000}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" --max-num-seqs "${MAX_SEQS:-8}" \
  --max-num-batched-tokens "${PREFILL_CHUNK:-512}" \
  --gpu-memory-utilization 0.95 --mamba-cache-mode align \
  --no-enable-prefix-caching --enable-chunked-prefill \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --max-logprobs 20 --host 0.0.0.0 --port 8000 \
  --compilation-config "$compilation" \
  "${spec[@]}"
