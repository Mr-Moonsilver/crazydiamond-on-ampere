#!/bin/bash
# Short AWQ-BF16-INT4 vs W8A8 check on CT <ct-id> (2026-10-22).
#   phase perf : production flags, model swapped  -> quality smoke + the 4-case screen (same script/corpus/matrix as candidate-power-screen)
#   phase eval : August eval flags, model swapped -> PPL EN/FR, GSM8K N 500, IFEval N 541, MMLU-Pro N 10/subject
#   then production vllm-qwen38 is restarted, whatever happened.
# The test server runs as transient unit gpu-tuning-benchmark.service (stopped by the host thermal guard), bound to 127.0.0.1.
set -uo pipefail
ROOT=/data/vllm/gpu-tuning-20261021/awq-bf16
MODEL=cyankiwi/Qwen3.8-27B-AWQ-BF16-INT4
REV=cf6076fcb2d840d459a93297c7c5dd01bb429931
PY=/home/<service-user>/vllm-new/bin/python
U=http://127.0.0.1:8000
mkdir -p "$ROOT"; chown <service-user>:<service-user> "$ROOT"
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/run.log"; }

restore(){
  log "restore: stopping test server, starting vllm-qwen38"
  systemctl stop gpu-tuning-benchmark.service 2>/dev/null; systemctl reset-failed gpu-tuning-benchmark.service 2>/dev/null
  systemctl start vllm-qwen38.service
  for i in $(seq 1 90); do curl -fs --max-time 5 $U/v1/models >/dev/null && { log "production healthy"; return; }; sleep 10; done
  log "PRODUCTION NOT HEALTHY AFTER 15 MIN"
}
trap restore EXIT

serve(){ # $1=label, rest = flags
  local label=$1; shift
  systemd-run --unit=gpu-tuning-benchmark --uid=<service-user> --gid=<service-user> -p WorkingDirectory=/home/<service-user> -p LimitMEMLOCK=infinity \
    --setenv=HOME=/home/<service-user> --setenv=HF_HOME=/data/huggingface --setenv=HF_HUB_OFFLINE=1 --setenv=TMPDIR=/home/<service-user>/tmp \
    --setenv=VLLM_CACHE_ROOT=/home/<service-user>/.cache/vllm-awqbf16 --setenv=TORCHINDUCTOR_CACHE_DIR=/home/<service-user>/.cache/torchinductor-awqbf16 \
    --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID --setenv=NCCL_P2P_LEVEL=SYS --setenv=NCCL_CUMEM_ENABLE=0 \
    --setenv=VLLM_WORKER_MULTIPROC_METHOD=spawn --setenv=OMP_NUM_THREADS=1 \
    -p StandardOutput=append:$ROOT/vllm-$label.log -p StandardError=append:$ROOT/vllm-$label.log \
    /home/<service-user>/vllm-new/bin/vllm serve $MODEL --revision $REV --served-model-name qwen38-w8a8-tp4 \
    --tensor-parallel-size 4 --dtype bfloat16 \
    --attention-backend FLASHINFER --mamba-cache-mode align --mamba-backend flashinfer --mamba-ssu-algorithm auto \
    --kv-cache-dtype fp8_e4m3 --enable-chunked-prefill \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --host 127.0.0.1 --port 8000 --max-logprobs 20 --trust-remote-code "$@"
  for i in $(seq 1 90); do
    curl -fs --max-time 5 $U/health >/dev/null && { log "$label server healthy"; return 0; }
    systemctl is-active -q gpu-tuning-benchmark.service || { log "$label server died"; return 1; }
    sleep 10
  done
  log "$label server not healthy in 15 min"; return 1
}
stop_test(){ systemctl stop gpu-tuning-benchmark.service; systemctl reset-failed gpu-tuning-benchmark.service 2>/dev/null; sleep 5; }

log "start; prod unit sha $(sha256sum /etc/systemd/system/vllm-qwen38.service | cut -c1-16)"
systemctl stop vllm-qwen38.service; sleep 5

# --- perf: production flags
if serve perf --enable-prefix-caching --gpu-memory-utilization 0.90 --max-model-len 163840 --max-num-seqs 8 --max-num-batched-tokens 8192; then
  grep -E "KV cache|Maximum concurrency|kernel|Marlin|Machete|WNA16|CompressedTensors" $ROOT/vllm-perf.log | head -20 > $ROOT/startup-notes.txt
  sudo -u <service-user> $PY /data/vllm/gpu-tuning-20261021/tools/quality_gpu_tuning.py --output $ROOT/quality-smoke \
    --reference-summary /data/vllm/gpu-tuning-20261021/baseline-quality/summary.json >> $ROOT/run.log 2>&1 || log "quality smoke reported failures"
  sudo -u <service-user> $PY /home/<service-user>/bench_gpu_tuning.py --matrix /home/<service-user>/GPU-TUNING-MATRIX-2026-10-21.json \
    --corpus /home/<service-user>/p2p-tests/corpus.txt --output $ROOT/screen --screen \
    --stop-at "$(date -u -d '+3 hours' +%FT%T+00:00)" >> $ROOT/run.log 2>&1 || log "screen failed"
fi
stop_test

# --- eval: August eval profile
if serve eval --no-enable-prefix-caching --gpu-memory-utilization 0.75 --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 1024; then
  cd /home/<service-user>/p2p-tests
  sudo -u <service-user> python3 ppl.py $U qwen38-w8a8-tp4 corpus.txt "awqbf16 PPL-EN" 2048 6 >> $ROOT/evals.log 2>&1
  sudo -u <service-user> python3 ppl.py $U qwen38-w8a8-tp4 corpus_fr.txt "awqbf16 PPL-FR" 2048 6 >> $ROOT/evals.log 2>&1
  LM=/home/<service-user>/lmeval/bin/lm_eval; R=$ROOT/lmeval
  MA="model=qwen38-w8a8-tp4,base_url=$U/v1/chat/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,max_gen_toks=4096,timeout=1800"
  GK="temperature=0,reasoning_effort=none,max_gen_toks=4096"
  ev(){ echo "== $1 ==" >> $ROOT/evals.log; shift; sudo -u <service-user> env HF_HUB_OFFLINE=0 HF_HOME=/home/<service-user>/.cache/hf-eval "$@" >> $ROOT/evals.log 2>&1; }
  ev "N gsm8k"    $LM --model local-chat-completions --model_args "$MA" --tasks gsm8k --limit 500 --apply_chat_template --fewshot_as_multiturn --num_fewshot 5 --gen_kwargs $GK --output_path $R --seed 1234 --log_samples
  ev "N ifeval"   $LM --model local-chat-completions --model_args "$MA" --tasks ifeval --apply_chat_template --gen_kwargs $GK --output_path $R --seed 1234 --log_samples
  ev "N mmlu_pro" $LM --model local-chat-completions --model_args "$MA" --tasks mmlu_pro --limit 10 --apply_chat_template --fewshot_as_multiturn --gen_kwargs $GK --output_path $R --seed 1234 --log_samples
fi
stop_test
log "tests done"
