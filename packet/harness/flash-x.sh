#!/bin/bash
# Quick Flash-Next probes on CT <ct-id>: each config = one container launch + one short 8-stream run.
#   flash-quick.sh <tag> <streams> <prompt> <output> [ENV=VALUE ...]   (ENV overrides prod/flash-tuning.env values)
# Production (vllm-qwen38) must already be stopped by the caller; this script never starts it.
set -uo pipefail
TAG=$1 N=$2 PT=$3 OT=$4; shift 4
ROOT=/data/vllm/gpu-tuning-20261021/flash-quick; mkdir -p $ROOT; chmod 777 $ROOT
T=/data/vllm/gpu-tuning-20261021/tools
set -a; . $T/flash-tuning.env; set +a
CONTAINER_NAME=gpu-tuning-flash-20261021-quick-$TAG
for kv in "$@"; do export "$kv"; done
export CONTAINER_NAME
log(){ echo "$(date -u +%FT%TZ) [$TAG] $*" | tee -a $ROOT/quick.log; }
log "launch $* (N=$N prompt=$PT out=$OT)"
setsid bash $T/run_flash_tuning.sh > $ROOT/$TAG.server.log 2>&1 < /dev/null &
ok=0
for i in $(seq 1 80); do
  curl -fs --max-time 5 http://127.0.0.1:${PORT}/health >/dev/null && { ok=1; break; }
  docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null | grep -q true || { [ $i -gt 3 ] && break; }
  sleep 10
done
if [ $ok = 1 ]; then
  log "healthy after $((i*10)) s; $(docker logs $CONTAINER_NAME 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | tail -1)"
  # warm-up (JIT), then the measured run
  sudo -u <service-user> python3 $T/quick_c8.py http://127.0.0.1:$PORT Qwen3.8-Flash-Next $N 512 64 warm $ROOT/$TAG.warm.json >/dev/null 2>&1
  # peak VRAM per GPU during the measured run (MiB, nvidia-smi order)
  ( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd, -; sleep 2; done ) > $ROOT/$TAG.vram.csv &
  sampler=$!
  sudo -u <service-user> env QUICK_WINDOW_S=${QUICK_WINDOW_S:-0} python3 $T/quick_c8.py http://127.0.0.1:$PORT Qwen3.8-Flash-Next $N $PT $OT $TAG $ROOT/$TAG.json 2>&1 | tee -a $ROOT/quick.log
  kill $sampler 2>/dev/null
  log "peak VRAM MiB per GPU: $(python3 -c "import sys; rows=[list(map(int,l.split(','))) for l in open(sys.argv[1]) if l.strip()]; print([max(c) for c in zip(*rows)])" $ROOT/$TAG.vram.csv)"
  # one tiny request = one prefill step, which makes the row cache log its hit counters
  touch /data/vllm/flash-next-tp4/rowcache.report
  sudo -u <service-user> python3 $T/quick_c8.py http://127.0.0.1:$PORT Qwen3.8-Flash-Next 1 64 4 probe $ROOT/$TAG.probe.json >/dev/null 2>&1
  rm -f /data/vllm/flash-next-tp4/rowcache.report
  sleep 2
  hits=$(docker logs $CONTAINER_NAME 2>&1 | grep QSA_ROWCACHE_HITS | tail -4 | sed -E 's/.*(layer=[^ ]+ .*)/\1/' | sort -u | tr '\n' ';')
  [ -n "$hits" ] && log "row cache: $hits"
  if [ "${LOWC:-0}" != 0 ]; then
    # a second concurrency level in the same server (e.g. 4 streams after the 1-stream run)
    log "c$LOWC: $(sudo -u <service-user> env QUICK_WINDOW_S=${QUICK_WINDOW_S:-0} python3 $T/quick_c8.py http://127.0.0.1:$PORT Qwen3.8-Flash-Next $LOWC $PT $OT $TAG-c$LOWC $ROOT/$TAG.c$LOWC.json 2>&1 | tail -1)"
  fi
  if [ "${PREFIX_TEST:-0}" != 0 ]; then
    log "prefix test ($PREFIX_TEST conversations x 3 turns, ${PREFIX_RECORDS:-8000} records): $(sudo -u <service-user> python3 $T/prefix_test.py http://127.0.0.1:$PORT Qwen3.8-Flash-Next $PREFIX_TEST ${PREFIX_RECORDS:-8000} $ROOT/$TAG.prefix.json 2>&1 | tail -1)"
    log "prefix cache: $(docker logs $CONTAINER_NAME 2>&1 | grep -oE 'Prefix cache hit rate: [0-9.]+%' | tail -1)"
  fi
  if [ "${PREFILL_PROBE:-0}" != 0 ]; then
    # prefill only: N x PT with 1 output token; aggregate prefill = all prompt tokens / time to the last first token
    sudo -u <service-user> python3 $T/quick_c8.py http://127.0.0.1:$PORT Qwen3.8-Flash-Next $N $PT 1 $TAG-prefill $ROOT/$TAG.prefill.json >/dev/null 2>&1
    log "prefill: $(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); t=max(d['ttft_s']); print({'streams': d['streams'], 'prompt_tokens': sum(d['prompt_tokens']), 'last_ttft_s': t, 'aggregate_prefill_tps': round(sum(d['prompt_tokens'])/t, 1), 'first_ttft_s': min(d['ttft_s'])})" $ROOT/$TAG.prefill.json)"
  fi
  if [ "${CHURN:-0}" != 0 ]; then
    log "churn retrieval ($CHURN requests, concurrency $((MAX_SEQS + 4))): $(sudo -u <service-user> python3 $T/churn_test.py http://127.0.0.1:$PORT Qwen3.8-Flash-Next $CHURN $((MAX_SEQS + 4)) $ROOT/$TAG.churn.json 2>&1 | tail -1)"
  fi
else
  log "FAILED to start: $(docker logs $CONTAINER_NAME 2>&1 | grep -E 'Error|error|raise' | tail -3)"
fi
docker stop -t 30 $CONTAINER_NAME >/dev/null 2>&1
log "stopped"
