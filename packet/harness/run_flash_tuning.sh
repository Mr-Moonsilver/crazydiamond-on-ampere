#!/usr/bin/env bash
set -euo pipefail
task_dir=/data/vllm/flash-next-tp4
expected_image=sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b
[[ $(sudo docker image inspect "${IMAGE:?Set the validated image}" --format '{{.Id}}') == "$expected_image" ]] || exit 2
name=${CONTAINER_NAME:?Set a unique gpu-tuning-flash-20261021 container name}
[[ "$name" == gpu-tuning-flash-20261021-* ]] || exit 2
mkdir -p "$task_dir/cache/vllm" "$task_dir/cache/triton" "$task_dir/cache/flashinfer"
if sudo docker inspect "$name" >/dev/null 2>&1; then
  echo "Refusing to replace existing container $name" >&2
  exit 1
fi
extra=()
if [[ -n ${PROFILER_CONFIG:-} ]]; then
  extra+=(--profiler-config "$PROFILER_CONFIG")
fi
if [[ ${DISABLE_CUSTOM_ALL_REDUCE:-0} == 1 ]]; then
  extra+=(--disable-custom-all-reduce)
fi
if [[ ${EAGER:-1} == 1 ]]; then
  extra+=(--enforce-eager)
else
  default_compilation='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}'
  extra+=(--compilation-config "${COMPILATION_CONFIG:-$default_compilation}")
fi
if [[ ${MTP_TOKENS:-0} != 0 ]]; then
  printf -v spec_config '{"method":"mtp","num_speculative_tokens":%d}' "$MTP_TOKENS"
  extra+=(--speculative-config "$spec_config")
fi
docker_extra=()
for mount in ${EXTRA_MOUNT:-}; do
  docker_extra+=(-v "$mount")
done
for kv in ${EXTRA_ENV:-}; do
  docker_extra+=(-e "$kv")
done
if [[ -n ${QSA_OVERLAP_PROBE:-} ]]; then
  docker_extra+=(-e "QSA_OVERLAP_PROBE=$QSA_OVERLAP_PROBE")
fi
if [[ -n ${PP_LAYER_PARTITION:-} ]]; then
  docker_extra+=(-e "VLLM_PP_LAYER_PARTITION=$PP_LAYER_PARTITION")
fi
exec sudo docker run --init --name "$name" --label qwen.flash-next-experiment=true \
  --gpus all --ipc=host --network host --restart=no \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_VISIBLE_DEVICES="${GPU_ORDER:-0,1,2,3}" \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
  -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 -e OMP_NUM_THREADS=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_QSA_KV_OFFLOAD="${KV_OFFLOAD:-1}" -e VLLM_QSA_KV_OFFLOAD_MAX_GIB="${HOST_KV_GIB_PER_RANK:-16}" \
  -e VLLM_QSA_KVO_ARENA=100663296 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_QSA_KVO_SPARSE_PREFILL="${SPARSE_PREFILL:-1}" \
  -e PYTORCH_CUDA_ALLOC_CONF="${ALLOC_CONF:-expandable_segments:True}" \
  -e VLLM_CACHE_ROOT=/work/cache/vllm -e TRITON_CACHE_DIR=/work/cache/triton \
  -e FLASHINFER_WORKSPACE_BASE=/work/cache/flashinfer \
  "${docker_extra[@]}" \
  -v "$task_dir/models/${MODEL_PROFILE:-intel-w4a16}:/model:ro" -v "$task_dir:/work" \
  "${IMAGE:-qwen-flash-next-tp4:local}" /model \
  --served-model-name Intel/Qwen3.8-Flash-Next-W4A16-AutoRound Qwen3.8-Flash-Next \
  --tensor-parallel-size "${TP:-4}" --pipeline-parallel-size "${PP:-1}" --enable-expert-parallel \
  --dtype bfloat16 --language-model-only \
  --engram-config '{"cpu_offload":true}' \
  --kv-cache-dtype auto --kv-cache-memory-bytes "${GPU_KV_BYTES:-150000000}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" --max-num-seqs "${MAX_SEQS:-4}" \
  --max-num-batched-tokens "${PREFILL_CHUNK:-512}" \
  --gpu-memory-utilization 0.95 --mamba-cache-mode align \
  $( [[ ${PREFIX_CACHING:-0} == 1 ]] && echo --enable-prefix-caching || echo --no-enable-prefix-caching ) --enable-chunked-prefill \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --max-logprobs 20 \
  --host 127.0.0.1 --port "${PORT:-8001}" "${extra[@]}"
