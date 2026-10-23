#!/bin/bash
# One Flash container with the row cache; A/B inside it via /work/rowcache.off:
#   greedy outputs off vs on (must match), then 8-stream speed on vs off.
#   flash-ab.sh <tag> <streams> <prompt> <output> [ENV=VALUE ...]
set -uo pipefail
TAG=$1 N=$2 PT=$3 OT=$4; shift 4
ROOT=/data/vllm/gpu-tuning-20261021/flash-quick; T=/data/vllm/gpu-tuning-20261021/tools
FLAG=/data/vllm/flash-next-tp4/rowcache.off
set -a; . $T/flash-tuning.env; set +a
CONTAINER_NAME=gpu-tuning-flash-20261021-quick-$TAG
for kv in "$@"; do export "$kv"; done
export CONTAINER_NAME
U=http://127.0.0.1:$PORT; MODEL=Qwen3.8-Flash-Next
log(){ echo "$(date -u +%FT%TZ) [$TAG] $*" | tee -a $ROOT/quick.log; }
log "launch A/B $* (N=$N prompt=$PT out=$OT)"
rm -f $FLAG
setsid bash $T/run_flash_tuning.sh > $ROOT/$TAG.server.log 2>&1 < /dev/null &
ok=0
for i in $(seq 1 80); do
  curl -fs --max-time 5 $U/health >/dev/null && { ok=1; break; }
  docker inspect -f '{{.State.Running}}' $CONTAINER_NAME 2>/dev/null | grep -q true || { [ $i -gt 3 ] && break; }
  sleep 10
done
if [ $ok = 1 ]; then
  log "healthy after $((i*10)) s; $(docker logs $CONTAINER_NAME 2>&1 | grep -oE 'GPU KV cache size: [0-9,]+ tokens' | tail -1)"
  run(){ sudo -u <service-user> python3 "$@"; }
  touch $FLAG; run $T/quick_c8.py $U $MODEL $N 512 64 warm $ROOT/$TAG.warm-off.json >/dev/null 2>&1
  rm -f $FLAG; run $T/quick_c8.py $U $MODEL $N 512 64 warm $ROOT/$TAG.warm-on.json >/dev/null 2>&1
  for pass in 1 2; do
    touch $FLAG; log "greedy off$pass: $(run $T/greedy_ab.py $U $MODEL $ROOT/$TAG.greedy-off$pass.json 2>&1 | tail -1)"
    rm -f $FLAG; log "greedy on$pass:  $(run $T/greedy_ab.py $U $MODEL $ROOT/$TAG.greedy-on$pass.json 2>&1 | tail -1)"
  done
  log "compare off1/off2 (engine determinism): $(python3 $T/greedy_ab.py --compare $ROOT/$TAG.greedy-off1.json $ROOT/$TAG.greedy-off2.json)"
  log "compare on1/on2: $(python3 $T/greedy_ab.py --compare $ROOT/$TAG.greedy-on1.json $ROOT/$TAG.greedy-on2.json)"
  log "compare off1/on1 (cache): $(python3 $T/greedy_ab.py --compare $ROOT/$TAG.greedy-off1.json $ROOT/$TAG.greedy-on1.json)"
  [ "${SKIP_SPEED:-0}" = 1 ] && { docker stop -t 30 $CONTAINER_NAME >/dev/null 2>&1; log "stopped"; exit 0; }
  rm -f $FLAG; log "speed on:  $(run $T/quick_c8.py $U $MODEL $N $PT $OT $TAG-on $ROOT/$TAG.on.json 2>&1 | tail -1)"
  touch $FLAG; log "speed off: $(run $T/quick_c8.py $U $MODEL $N $PT $OT $TAG-off $ROOT/$TAG.off.json 2>&1 | tail -1)"
  rm -f $FLAG
else
  log "FAILED to start: $(docker logs $CONTAINER_NAME 2>&1 | grep -E 'Error|error|raise' | tail -3)"
fi
docker stop -t 30 $CONTAINER_NAME >/dev/null 2>&1
log "stopped"
