#!/usr/bin/env bash
# Final-engine SERVE stop (b16's stop, unchanged steps), inside CT <ct-id> as root:  bash <packet>/serve-stop.sh <OUT>
# Snapshot the logs, stop vllm-flash cleanly, put the found launcher and defaults back byte-for-byte, and prove the box is
# in the found state the production handback expects: vllm-flash inactive + disabled, vllm-qwen38 inactive + enabled
# (untouched), no container, GPUs < 1000 MiB, root >= 28 GB, no /etc/default/vllm-flash.next, launcher = found 45cc5d81,
# defaults = found.  Then run the EXISTING handback (pinned prodback + image canary, unchanged) if production is to return.
# It never starts, enables or edits vllm-qwen38 and never touches the boot guard.  Idempotent: safe to run twice.
# R (default empty) prefixes every box path, for the hub self-run only.
set -uo pipefail
PK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=${R:-}
source "$PK/serve-params.env" || { echo "serve-params.env missing"; exit 2; }
OUT=${1:-${OUT:-}}
[[ -n $OUT ]] || { echo "usage: serve-stop.sh <OUT> (the launch's out dir, e.g. /data/serve-$SERVE_TAG-engine)"; exit 2; }
H=${H:-$PK/harness}
TASK=$R/data/vllm/flash-next-tp4/prod; LAUNCHER=$TASK/run-flash-prod.sh
DEFAULTS=$R/etc/default/vllm-flash; NEXT=$DEFAULTS.next
SERVICE=vllm-flash; PROD=vllm-qwen38; CONTAINER=qwen-flash-prod
FOUND_SHA=$FOUND_LAUNCHER_SHA
PORT=${SERVE_PORT:-8000}; URL=http://127.0.0.1:$PORT
OWNER_UID=${SERVE_OWNER_UID:-0}
TS=$(date -u +%Y%m%dT%H%M%SZ)
[[ $(id -u) == "$OWNER_UID" ]] || { echo "run as root in CT <ct-id>"; exit 2; }
[[ -d $OUT/private ]] || { echo "no serve run at $OUT"; exit 2; }
log() { printf '%s STOP %-18s %s\n' "$(date +%T)" "$1" "$2" | tee -a "$OUT/stop-$TS.log"; }
log start "out=$OUT"
# 1. snapshots before the stop (the container runs --rm)
if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  timeout 180s docker logs --timestamps "$CONTAINER" > "$OUT/engine-final-$TS.log" 2>&1
  log docker-logs "rc=$? $(du -h "$OUT/engine-final-$TS.log" | cut -f1)"
  docker inspect "$CONTAINER" --format '{{.Id}} {{.State.Status}} started {{.State.StartedAt}}' > "$OUT/container-$TS.txt" 2>&1
else
  log docker-logs "no container (already stopped)"
fi
AUTH_ENV=${AUTH_ENV:-$R/root/.flashnext-serve-$SERVE_TAG/vllm-api.env}
source "$OUT/arm.env" 2>/dev/null; log arm "${ARM:-unknown}"
[[ -f $OUT/floor-breach.json ]] && log floor "BREACH latched during the serve: $(head -c 300 "$OUT/floor-breach.json")" 
if [[ -z ${VLLM_API_KEY:-} && -f $AUTH_ENV ]]; then VLLM_API_KEY=$(sed -n 's/^VLLM_API_KEY=//p' "$AUTH_ENV"); export VLLM_API_KEY; fi
export FLASH_S5_DECODE_GROUPS=chunk PYTHONPATH="$H/e3_builder:$H"
if [[ -f $OUT/pids.env ]]; then
  source "$OUT/pids.env"
  if [[ ${ARM:-} == engine && -n ${RUN_ID:-} && -n ${VLLM_API_KEY:-} ]]; then
    timeout 90s python3 -B "$PK/serve_lane.py" status --h "$H" --url "$URL" --run-id "$RUN_ID" > "$OUT/final-status-$TS.json" 2>"$OUT/final-status-$TS.err"
    log final-status "rc=$? (lane counters)"
  fi
fi
cp -p "$OUT/recorder.jsonl" "$OUT/recorder-at-stop-$TS.jsonl" 2>/dev/null
journalctl -u "$SERVICE" --since "-12h" --no-pager > "$OUT/journal-$TS.log" 2>&1
curl -s --max-time 10 "$URL/metrics" > "$OUT/metrics-final-$TS.txt" 2>/dev/null; log metrics "final /metrics $(wc -c < "$OUT/metrics-final-$TS.txt" 2>/dev/null || echo 0) bytes" 
# 2. recorder: stop file, then its verified pid
touch "$OUT/recorder.stop"
if [[ -n ${RECORDER_PID:-} ]] && grep -qa serve_recorder.py "/proc/$RECORDER_PID/cmdline" 2>/dev/null; then
  for i in $(seq 1 75); do [[ -d /proc/$RECORDER_PID ]] || break; sleep 1; done
  grep -qa serve_recorder.py "/proc/$RECORDER_PID/cmdline" 2>/dev/null && kill -TERM "$RECORDER_PID"
fi
log recorder "stopped ($( (wc -l < "$OUT/recorder.jsonl") 2>/dev/null || echo 0) ticks)"
# 3. stop vllm-flash (never disable/enable: it is disabled and stays so)
N1=$(systemctl show "$SERVICE" -p NRestarts --value)
timeout --signal=TERM --kill-after=20s 240s systemctl stop "$SERVICE" > "$OUT/systemctl-stop-$TS.log" 2>&1
log systemctl-stop "rc=$? NRestarts=$N1 is-active=$(systemctl is-active $SERVICE)"
for i in $(seq 1 90); do docker inspect "$CONTAINER" >/dev/null 2>&1 || break; sleep 2; done
docker inspect "$CONTAINER" >/dev/null 2>&1 && log container "STILL PRESENT after 180 s" || log container "gone"
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
  busy=0; for u in $used; do (( u < 1000 )) || busy=1; done
  (( busy == 0 )) && break; sleep 2
done
log gpus "MiB used: $used"
if [[ -n ${FOLLOWER_PID:-} ]] && grep -qa 'docker' "/proc/$FOLLOWER_PID/cmdline" 2>/dev/null; then
  for i in $(seq 1 20); do [[ -d /proc/$FOLLOWER_PID ]] || break; sleep 1; done
fi
# 4. found launcher and defaults back (only once the engine is proven down)
state=$(systemctl is-active "$SERVICE")
if [[ $state == inactive || $state == failed ]] && ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  source "$OUT/found.env" 2>/dev/null
  if [[ -f $OUT/private/launcher.found ]]; then
    if cmp -s "$LAUNCHER" "$OUT/private/launcher.found"; then log launcher "already the found launcher"
    else
      read -r lm lu lg <<<"$LAUNCHER_MODE"
      install -m "$lm" -o "$lu" -g "$lg" "$OUT/private/launcher.found" "$LAUNCHER.q18a-restore" && mv "$LAUNCHER.q18a-restore" "$LAUNCHER" &&
        cmp -s "$LAUNCHER" "$OUT/private/launcher.found" && log launcher "restored [$LAUNCHER_MODE]" || log launcher "RESTORE FAILED"
    fi
  fi
  if [[ -f $OUT/private/defaults.found ]]; then
    if cmp -s "$DEFAULTS" "$OUT/private/defaults.found"; then log defaults "already the found defaults"
    else
      read -r dm du dg <<<"$DEFAULTS_MODE"
      install -m "$dm" -o "$du" -g "$dg" "$OUT/private/defaults.found" "$DEFAULTS.q18a-restore" && mv "$DEFAULTS.q18a-restore" "$DEFAULTS" &&
        cmp -s "$DEFAULTS" "$OUT/private/defaults.found" && log defaults "restored [$DEFAULTS_MODE]" || log defaults "RESTORE FAILED"
    fi
  fi
  if [[ -e $NEXT || -L $NEXT ]]; then mv -n "$NEXT" "$OUT/private/defaults.next.left-$TS" && log next "moved into $OUT/private"; fi
else
  log restore "WITHHELD: engine state $state or container present; found launcher/defaults NOT restored"
fi
# 5. the w14 chain's start state (box_ok + the pinned stop script's own refusals)
bad=()
[[ $(systemctl is-active $SERVICE) != active ]] || bad+=("$SERVICE active")
[[ $(systemctl is-enabled $SERVICE 2>&1) == disabled ]] || bad+=("$SERVICE not disabled")
[[ $(systemctl is-active $PROD) != active ]] || bad+=("$PROD active")
[[ $(systemctl is-enabled $PROD 2>&1) == enabled ]] || bad+=("$PROD not enabled")
[[ -z $(systemctl show $PROD -p DropInPaths --value) ]] || bad+=("$PROD drop-ins present")
docker inspect "$CONTAINER" >/dev/null 2>&1 && bad+=("container present")
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
for u in $used; do (( u < 1000 )) || { bad+=("GPU busy: $used"); break; }; done
root_gb=$(df -BG --output=avail "${R:-/}" | tail -1 | tr -dc 0-9); (( root_gb >= 28 )) || bad+=("root ${root_gb}G < 28G")
[[ $(sha256sum "$LAUNCHER" | cut -c1-64) == "$FOUND_SHA" ]] || bad+=("launcher is not the found ${FOUND_SHA:0:12}")
[[ -f $OUT/private/defaults.found ]] && ! cmp -s "$DEFAULTS" "$OUT/private/defaults.found" && bad+=("defaults are not the found file")
[[ ! -e $NEXT && ! -L $NEXT ]] || bad+=("$NEXT present")
[[ -d "$R/home/<service-user>/s0" ]] || bad+=("/home/<service-user>/s0 missing")
log state "$SERVICE=$(systemctl is-active $SERVICE)/$(systemctl is-enabled $SERVICE 2>&1) $PROD=$(systemctl is-active $PROD)/$(systemctl is-enabled $PROD 2>&1) GPU [$used] root ${root_gb}G"
if (( ${#bad[@]} == 0 )); then echo "SERVE_STOP_STATE READY $(date -u +%FT%TZ) (found state; production handback may run)" | tee -a "$OUT/stop-$TS.log"; exit 0; fi
printf 'SERVE_STOP_STATE NOT_READY: %s\n' "$(IFS=';'; echo "${bad[*]}")" | tee -a "$OUT/stop-$TS.log"; exit 1
