#!/usr/bin/env bash
# Production launcher: Qwen3.8-Flash-Next W4A16 on CT <ct-id>, TP2 x PP2 / EP2, host QSA KV + PLE, custom
# all-reduce, per-request VRAM row cache for selected QSA rows (overlay in $OVERLAY, checksummed).
# Settings come from /etc/default/vllm-flash (see vllm-flash.service); EXTRA_ARGS is appended word-split. Foreground: systemd owns the lifetime.
set -euo pipefail
task_dir=/data/vllm/flash-next-tp4
expected_image=sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b
name=qwen-flash-prod
OVERLAY=${OVERLAY:-$task_dir/prod/overlay}
[[ $(sudo docker image inspect "$expected_image" --format '{{.Id}}') == "$expected_image" ]] || exit 2
(cd "$OVERLAY" && sha256sum --quiet -c SHA256SUMS) || { echo "overlay checksum mismatch" >&2; exit 2; }
sudo docker rm -f "$name" >/dev/null 2>&1 || true
# test-only switches of the row-cache overlay must not exist in production
sudo -n rm -f "$task_dir/rowcache.off" "$task_dir/rowcache.report" "$task_dir/retain.off" 2>/dev/null || true
D=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
default_compilation='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,6,8,9,12,15,18,21,24]}'
compilation=${COMPILATION_CONFIG:-$default_compilation}
# CORE_OVERLAY=1: patched vllm/v1/core files (evict superseded Mamba partial tails first), checksummed.
core=()
if [[ ${CORE_OVERLAY:-0} == 1 ]]; then
  C=/usr/local/lib/python3.12/dist-packages/vllm/v1/core
  (cd "$OVERLAY/core" && sha256sum --quiet -c SHA256SUMS) || { echo "core overlay checksum mismatch" >&2; exit 2; }
  core=(-v "$OVERLAY/core/block_pool.py:$C/block_pool.py:ro"
        -v "$OVERLAY/core/single_type_kv_cache_manager.py:$C/single_type_kv_cache_manager.py:ro")
  # the core overlay may also carry the scheduler (PD alternation under PP, FLASH_PD_ALTERNATE=1)
  [[ -f $OVERLAY/core/scheduler.py ]] && core+=(-v "$OVERLAY/core/scheduler.py:$C/sched/scheduler.py:ro")
  # Optional PP grouping / pool telemetry overlay; each mounted file must be in the manifest.
  for f in kv_cache_utils.py kv_cache_manager.py; do
    if [[ -f $OVERLAY/core/$f ]]; then
      grep -q "  *$f\$" "$OVERLAY/core/SHA256SUMS" || { echo "$f not checksummed" >&2; exit 2; }
      core+=(-v "$OVERLAY/core/$f:$C/$f:ro")
    fi
  done
fi
# An overlay that carries model_runner_v2.py (row-cache request ids for Model Runner V2, overlay-v2) mounts it too;
# its checksum is in the overlay SHA256SUMS checked above.
runner=()
if [[ -f $OVERLAY/model_runner_v2.py ]]; then
  grep -q " model_runner_v2.py$" "$OVERLAY/SHA256SUMS" || { echo "runner not checksummed" >&2; exit 2; }
  runner=(-v "$OVERLAY/model_runner_v2.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py:ro")
fi
# E2 ordinary-engine test payload. Always checksummed; switches remain opt-in.
e2=()
if [[ -d $OVERLAY/e2 ]]; then
  [[ ${CORE_OVERLAY:-0} == 1 ]] || { echo "E2 needs the declared core overlay" >&2; exit 2; }
  for f in e2/__init__.py e2/retention_contract.py e2/arena_contract.py e2/block_mirror.py e2/serial_dma.py e2/arena_dma.py e2/runner_lifecycle.py e2/startup.py e2/exactness.py e2_sources/scheduler_output.py e2_sources/block_table.py; do
    grep -q "  $f$" "$OVERLAY/SHA256SUMS" || { echo "E2 mount not checksummed" >&2; exit 2; }
  done
  V=/usr/local/lib/python3.12/dist-packages/vllm/v1
  e2=(-v "$OVERLAY/e2:$V/flash_e2:ro"
      -v "$OVERLAY/e2_sources/scheduler_output.py:$V/core/sched/output.py:ro"
      -v "$OVERLAY/e2_sources/block_table.py:$V/worker/gpu/block_table.py:ro")
fi
# VISION=1: image input (no --language-model-only), $VISION_ARGS appended, first-stage-only tower if the overlay has
# vision/model.py (checksummed in vision/SHA256SUMS).
vision=(); lmo=(--language-model-only)
if [[ ${VISION:-0} == 1 ]]; then
  lmo=()
  if [[ -f $OVERLAY/vision/model.py ]]; then
    (cd "$OVERLAY/vision" && sha256sum --quiet -c SHA256SUMS) || { echo "vision overlay checksum mismatch" >&2; exit 2; }
    vision=(-v "$OVERLAY/vision/model.py:$D/model.py:ro")
  fi
fi
# EXTRA_ENV="K=V K2=V2": experiment switches passed to the container as-is (e.g. QSA_STAGE_TIMING=1,
# FLASH_PD_DECODE_SHARE=0.5); only K=V words with an upper-case name are accepted.
extra_env=()
for kv in ${EXTRA_ENV:-}; do
  [[ $kv =~ ^[A-Z][A-Z0-9_]*=[^[:space:]]*$ ]] || { echo "bad EXTRA_ENV word: $kv" >&2; exit 2; }
  extra_env+=(-e "$kv")
done
# MPS_PIPE=dir: diagnostic only -- connect to an MPS control daemon started in the CT with that pipe directory
# (e.g. to cap SMs with EXTRA_ENV=CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50). Unset in production.
mps=()
if [[ -n ${MPS_PIPE:-} ]]; then
  [[ -S $MPS_PIPE/control ]] || { echo "no MPS daemon at $MPS_PIPE" >&2; exit 2; }
  mps=(-v "$MPS_PIPE:$MPS_PIPE" -e CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE")
fi
# MODEL_DIR: another checkpoint directory under $task_dir/models (FP8 PLE phase: intel-w4a16-fp8ple, made by
# tools/ple_fp8_convert.py, hard links + one rewritten table file).
model_dir=${MODEL_DIR:-$task_dir/models/intel-w4a16}
[[ $model_dir == "$task_dir"/models/* && -f $model_dir/config.json ]] || { echo "bad MODEL_DIR: $model_dir" >&2; exit 2; }
# An overlay that carries ngram_embedding.py (make_plefp8.py: byte-view lookup for 1-byte PLE tables) mounts it too.
ngram=()
if [[ -f $OVERLAY/ngram_embedding.py ]]; then
  grep -q " ngram_embedding.py$" "$OVERLAY/SHA256SUMS" || { echo "ngram_embedding.py not checksummed" >&2; exit 2; }
  ngram=(-v "$OVERLAY/ngram_embedding.py:$D/ngram_embedding.py:ro")
fi
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
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_USE_BREAKABLE_CUDAGRAPH="${BREAKABLE_CG:-0}" \
  -e VLLM_QSA_KV_OFFLOAD=1 -e VLLM_QSA_KV_OFFLOAD_MAX_GIB="${HOST_KV_GIB_PER_RANK:-28}" \
  -e VLLM_QSA_KVO_ARENA=100663296 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \
  -e VLLM_QSA_KVO_SPARSE_PREFILL="${SPARSE_PREFILL:-1}" \
  ${QSA_COW_VERIFY:+-e QSA_COW_VERIFY="$QSA_COW_VERIFY"} \
  ${FLASH_SCHED_STATS:+-e FLASH_SCHED_STATS="$FLASH_SCHED_STATS"} \
  ${FLASH_PARITY_BALANCE:+-e FLASH_PARITY_BALANCE="$FLASH_PARITY_BALANCE"} \
  -e FLASH_PD_ALTERNATE="${FLASH_PD_ALTERNATE:-0}" \
  ${QSA_SPARSE_MIN_PAGES:+-e QSA_SPARSE_MIN_PAGES="$QSA_SPARSE_MIN_PAGES"} \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e VLLM_PP_LAYER_PARTITION="${PP_LAYER_PARTITION:-26,22}" \
  -e QSA_ROWCACHE_SLOTS="${QSA_ROWCACHE_SLOTS:-8192}" -e QSA_ROWCACHE_REQS="${MAX_SEQS:-8}" \
  -e VLLM_CACHE_ROOT=/work/cache/vllm -e TRITON_CACHE_DIR=/work/cache/triton \
  -e FLASHINFER_WORKSPACE_BASE=/work/cache/flashinfer \
  "${extra_env[@]}" "${mps[@]}" \
  -v "$model_dir:/model:ro" -v "$task_dir:/work" \
  -v "$OVERLAY/qsa.py:$D/qsa.py:ro" -v "$OVERLAY/ops_qsa.py:$D/ops/qsa.py:ro" \
  "${core[@]}" \
  "${runner[@]}" "${e2[@]}" \
  "${vision[@]}" "${ngram[@]}" \
  "$expected_image" /model \
  --served-model-name qwen38-w8a8-tp4 Qwen3.8-Flash-Next Intel/Qwen3.8-Flash-Next-W4A16-AutoRound \
    cyankiwi/Qwen3.8-27B-AWQ-INT4 cyankiwi/Qwen3.5-122B-A10B-AWQ-4bit \
  --tensor-parallel-size 2 --pipeline-parallel-size 2 --enable-expert-parallel \
  --dtype bfloat16 "${lmo[@]}" --engram-config '{"cpu_offload":true}' \
  --kv-cache-dtype auto --kv-cache-memory-bytes "${GPU_KV_BYTES:-1800000000}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" --max-num-seqs "${MAX_SEQS:-8}" \
  --max-num-batched-tokens "${PREFILL_CHUNK:-512}" \
  --gpu-memory-utilization 0.95 --mamba-cache-mode align \
  $( [[ ${PREFIX_CACHING:-1} == 1 ]] && echo --enable-prefix-caching || echo --no-enable-prefix-caching ) --enable-chunked-prefill \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --max-logprobs 20 --host 0.0.0.0 --port 8000 \
  --compilation-config "$compilation" \
  "${spec[@]}" ${EXTRA_ARGS:-} ${VISION_ARGS:-}
