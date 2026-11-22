#!/usr/bin/env bash
# Final-engine SERVE drain, inside CT <ct-id> as root:  bash <packet>/serve-drain.sh <OUT> [deadline_s]
# The client side (the A/B driver) stops opening sessions first.  Then this waits until the engine has no running and no
# waiting request on DRAIN_QUIET consecutive /metrics reads (DRAIN_EVERY_S apart), up to the deadline (default 900 s),
# printing one line per read; then it runs serve-stop.sh <OUT>.  On the deadline it stops anyway and says so (requests
# still running then are cut: the A/B reader counts them as unfinished).  Never signals the engine itself.
set -uo pipefail
PK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${1:-}; DEADLINE_S=${2:-${DRAIN_DEADLINE_S:-900}}
[[ -n $OUT && -d $OUT ]] || { echo "usage: serve-drain.sh <OUT> [deadline_s]"; exit 2; }
PORT=${SERVE_PORT:-8000}; URL=http://127.0.0.1:$PORT
QUIET=${DRAIN_QUIET:-3}; EVERY=${DRAIN_EVERY_S:-5}
TS=$(date -u +%Y%m%dT%H%M%SZ)
log() { printf '%s DRAIN %-10s %s\n' "$(date +%T)" "$1" "$2" | tee -a "$OUT/drain-$TS.log"; }
gauge() { curl -s --max-time 10 "$URL/metrics" 2>/dev/null | awk -v n="vllm:$1" '$1 ~ "^"n"({|$)" {s+=$2} END {printf "%d", s+0}'; }
log start "out=$OUT deadline ${DEADLINE_S}s quiet ${QUIET}x${EVERY}s"
quiet=0; t0=$SECONDS; result=deadline
while (( SECONDS - t0 < DEADLINE_S )); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$URL/health" 2>/dev/null) || true
  if [[ $code != 200 ]]; then result=engine-down; log read "health $code: engine not answering"; break; fi
  run=$(gauge num_requests_running); wait=$(gauge num_requests_waiting)
  if (( run == 0 && wait == 0 )); then quiet=$((quiet + 1)); else quiet=0; fi
  log read "running $run waiting $wait quiet $quiet/$QUIET t+$((SECONDS - t0))s"
  if (( quiet >= QUIET )); then result=drained; break; fi
  sleep "$EVERY"
done
echo "SERVE_DRAIN $result after $((SECONDS - t0))s" | tee -a "$OUT/drain-$TS.log"
bash "$PK/serve-stop.sh" "$OUT"
