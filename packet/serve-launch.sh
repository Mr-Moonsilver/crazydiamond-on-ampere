#!/usr/bin/env bash
# Final-engine SERVE launch, inside CT <ct-id> as root:  bash <packet>/serve-launch.sh engine|stock
#   engine = the G0 candidate cxfx42 (overlay overlay-v16s5-g4-headroom-20261118a, launcher 0bf5b9a9, the b1
#            settings' S5_SETTINGS) with the serve changes only: EXTRA_ARGS without "--host 127.0.0.1" (binds 0.0.0.0:8000),
#            MAX_MODEL_LEN / MAX_SEQS / QSA_ROWCACHE_SLOTS from serve-params.env, FLASH_S5_AUTH_ENV_FILE = the serve key.
#            Then cxay12/cxfx42's n3 lane (n3.g2-live-0, auto-arm released) is selected once and held (b16's cx order).
#   stock  = deployment stock: the FOUND defaults (pinned) with MAX_MODEL_LEN / MAX_SEQS from serve-params.env,
#            QSA_ROWCACHE_SLOTS = STOCK_ROWCACHE_SLOTS (10240 = the found value) and SERVE_AUTH_ENV_FILE = the serve key; the
#            launcher is run-flash-stock-keyed.sh (c packet, RULE-TEST-SERVE-KEYED-V2): the FOUND launcher 45cc5d81 plus ONE
#            pinned change, docker --env-file "$SERVE_AUTH_ENV_FILE" (the engine arm's key mechanism; it refuses a missing
#            file and any argv/EXTRA_ENV key).  No lane.
# Both arms: the same fresh key (0600 file, value never printed) and one unauthenticated GET /v1/models that must answer 401
# before any test traffic (readback: auth=key).
# Both: guards, per-rank VRAM budget + floor gate after start and after the smoke, effective-settings readback, recorder,
# smoke.  Prerequisite: production stopped by the PINNED box-stop-qwen38-20261109.sh (its MANIFEST is the restore EXPECT).
# vllm-flash is started, never enabled.  Any failure after the first change runs serve-stop.sh (KEEP_ON_FAIL=1 skips).
# R (default empty) prefixes every box path; it exists only for the hub self-run (tools/final_engine_serve_selfrun.py).
set -uo pipefail
ARM=${1:-}
[[ $ARM == engine ]] || { echo "usage: serve-launch.sh engine|stock"; exit 2; }
PK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=${R:-}
source "$PK/serve-params.env" || { echo "serve-params.env missing"; exit 2; }
H=${H:-$PK/harness}
OUT=${OUT:-$R/data/serve-$SERVE_TAG-$ARM}
AUTH_ENV=${AUTH_ENV:-$R/root/.flashnext-serve-$SERVE_TAG/vllm-api.env}
OVERLAY=$R/data/vllm/flash-next-tp4/prod/overlay-v16s5-g4-headroom-20261118a
TASK=$R/data/vllm/flash-next-tp4/prod; LAUNCHER=$TASK/run-flash-prod.sh
DEFAULTS=$R/etc/default/vllm-flash; NEXT=$DEFAULTS.next; SETCFG=$R/tmp/setcfg.py
SERVICE=vllm-flash; PROD=vllm-qwen38; CONTAINER=qwen-flash-prod; SERVICE_USER='<service-user>'
OWNER_UID=${SERVE_OWNER_UID:-0}; OWNER=${SERVE_OWNER:-root}
PORT=${SERVE_PORT:-8001}; URL=http://127.0.0.1:$PORT
RUN_ID="serve-$SERVE_TAG-$ARM-$(date -u +%Y%m%dT%H%M%SZ)-$$"
HEALTH_DEADLINE_S=${HEALTH_DEADLINE_S:-1500}
step() { printf '%s STEP %-16s rc=%s %s\n' "$(date +%T)" "$1" "$2" "${3:-}" | tee -a "$OUT/steps.log"; }
die() { step "$1" 1 "$2"; echo "SERVE_LAUNCH FAILED at $1"; exit 1; }
fail_stop() {  # after any change: put the box back (logs, stop, launcher/defaults) unless KEEP_ON_FAIL=1
  step "$1" 1 "$2"
  if [[ ${KEEP_ON_FAIL:-0} == 1 ]]; then echo "SERVE_LAUNCH FAILED at $1 (KEEP_ON_FAIL=1: engine left as is; run serve-stop.sh $OUT)"; exit 1; fi
  echo "SERVE_LAUNCH FAILED at $1: running serve-stop.sh"
  bash "$PK/serve-stop.sh" "$OUT"; exit 1
}
own() { [[ $(stat -c '%a %u' "$1") == "$2 $OWNER_UID" ]]; }

# ---- guards (no change yet) ----
[[ $(id -u) == "$OWNER_UID" ]] || { echo "run as root in CT <ct-id>"; exit 2; }
[[ ! -e $OUT ]] || { echo "OUT exists: $OUT (fresh name needed: OUT=$OUT-2 bash serve-launch.sh $ARM)"; exit 2; }
mkdir -p "$OUT" && mkdir -m 0700 "$OUT/private" || { echo "cannot create $OUT"; exit 2; }
echo "SERVE_LAUNCH START $(date -u +%FT%TZ) arm=$ARM run_id=$RUN_ID out=$OUT params=$PARAMS_STATUS" | tee "$OUT/steps.log"
printf 'ARM=%s\nRUN_ID=%s\n' "$ARM" "$RUN_ID" > "$OUT/arm.env"
[[ $(sha256sum "$PK/serve-params.env" | cut -c1-64) == $(sed -n "s/  serve-params.env\$//p" "$PK/PACKET-SHA256SUMS") ]] ||
  die params "serve-params.env differs from PACKET-SHA256SUMS (parameters are part of the approved packet)"
step params 0 "max_model_len $MAX_MODEL_LEN max_num_seqs $MAX_SEQS rowcache engine $ENGINE_ROWCACHE_SLOTS stock $STOCK_ROWCACHE_SLOTS status $PARAMS_STATUS"
root_gb=$(df -BG --output=avail "${R:-/}" | tail -1 | tr -dc 0-9)
(( root_gb >= 28 )) || die root "root free ${root_gb}G < 28G"
step root 0 "root free ${root_gb}G >= 28G"
[[ $(systemctl is-active $SERVICE) != active ]] || die units "$SERVICE already active"
[[ $(systemctl is-enabled $SERVICE 2>&1) == disabled ]] || die units "$SERVICE not disabled"
[[ $(systemctl is-active $PROD) != active ]] || die units "$PROD active: run the pinned box-stop-qwen38-20261109.sh first (its log is the restore EXPECT)"
[[ $(systemctl is-enabled $PROD 2>&1) == enabled ]] || die units "$PROD not enabled (production setup changed; not touching it)"
docker inspect "$CONTAINER" >/dev/null 2>&1 && die units "container $CONTAINER exists"
[[ ! -e $NEXT && ! -L $NEXT ]] || die units "$NEXT exists"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
for u in $used; do (( u < 1000 )) || die gpus "GPU MiB used: $used"; done
step units 0 "$SERVICE inactive/disabled, $PROD inactive/enabled, no container, GPUs idle ($used)"
[[ $(sha256sum "$LAUNCHER" | cut -c1-64) == "$FOUND_LAUNCHER_SHA" ]] || die found "found launcher is not ${FOUND_LAUNCHER_SHA:0:12}"
[[ $(sha256sum "$SETCFG" | cut -c1-64) == "$SETCFG_SHA" ]] || die found "setcfg is not ${SETCFG_SHA:0:12}"
python3 -I -B "$PK/serve_readback.py" found --defaults "$DEFAULTS" --params "$PK/serve-params.env" > "$OUT/found-defaults.txt" 2>&1 ||
  die found "found defaults are not the pinned deployment file (see found-defaults.txt)"
step found 0 "found launcher ${FOUND_LAUNCHER_SHA:0:12}, setcfg ${SETCFG_SHA:0:12}, found defaults = deployment ${FOUND_DEFAULTS_SHA:0:12}"
SERVE_SETTINGS=("MAX_MODEL_LEN=$MAX_MODEL_LEN" "MAX_SEQS=$MAX_SEQS")
[[ $(sha256sum "$H/SHA256SUMS" | cut -c1-64) == "$BUNDLE_SUMS_SHA" ]] && (cd "$H" && sha256sum --quiet -c SHA256SUMS) ||
  die bundle "harness bundle is not cxfx42-b1 ${BUNDLE_SUMS_SHA:0:12}"
# ---- the serve key, BOTH arms (RULE-TEST-SERVE-KEYED-V2: no keyless test serve on :8000) ----
[[ -f $AUTH_ENV && ! -L $AUTH_ENV ]] && own "$AUTH_ENV" 600 || die auth "auth env missing or not 0600 $OWNER: keyless test serve refused"
VLLM_API_KEY=$(sed -n 's/^VLLM_API_KEY=//p' "$AUTH_ENV"); export VLLM_API_KEY
export PYTHONPATH="$H/e3_builder:$H" FLASH_S5_DECODE_GROUPS=chunk RUN_ID
python3 "$H/e3_private_auth.py" check "$AUTH_ENV" > "$OUT/auth-check.log" 2>&1 || die auth "private auth check failed: keyless test serve refused"
(( ${#VLLM_API_KEY} == 64 )) || die auth "key length ${#VLLM_API_KEY} != 64: keyless test serve refused"
step auth 0 "key file 0600 $OWNER, key length ${#VLLM_API_KEY} (arm $ARM)"
if [[ $ARM == engine ]]; then
  [[ $(sha256sum "$OVERLAY/SHA256SUMS" | cut -c1-64) == "$OVERLAY_SUMS_SHA" ]] && (cd "$OVERLAY" && sha256sum --quiet -c SHA256SUMS) &&
    [[ $(sha256sum "$OVERLAY/run-flash-s5-test.sh" | cut -c1-64) == "$S5_LAUNCHER_SHA" ]] &&
    [[ $(sha256sum "$OVERLAY/e3/native/flash_s5_green.so" | cut -c1-64) == "$GREEN_SHA" ]] &&
    [[ $(sha256sum "$OVERLAY/e3/native/flash_s5_replay_observer.so" | cut -c1-64) == "$OBSERVER_SHA" ]] ||
    die overlay "cxfx42-b1 overlay pins differ (or the overlay is not installed at $OVERLAY)"
  cmp -s "$H/native30_overlay/SHA256SUMS" "$OVERLAY/SHA256SUMS" || die overlay "installed overlay is not the bundle's native30_overlay"
  step pins 0 "bundle ${BUNDLE_SUMS_SHA:0:12}, overlay sums ${OVERLAY_SUMS_SHA:0:12}, launcher ${S5_LAUNCHER_SHA:0:12}, green ${GREEN_SHA:0:12}, observer ${OBSERVER_SHA:0:12}"
  # ---- the b1 settings file, the serve changes only ----
  S5_OVERLAY=${OVERLAY#$R} S5_AUTH_ENV_FILE=${AUTH_ENV#$R} S5_GREEN_LIB_SHA256=$GREEN_SHA S5_REPLAY_LIB_SHA256=$OBSERVER_SHA \
    Q18A_TARGET_LABEL=s5-overlap-80-80 FLASH_NATIVE30_SYNC=none
  export FLASH_NATIVE30_SYNC=none
  source "$H/runs/q18a_parity_n3s24q2024p_j2_cxfx42_b1_settings.sh" || die settings "b1 settings refused"
  SERVE_SETTINGS=()
  for kv in "${S5_SETTINGS[@]}"; do
    case $kv in
      "EXTRA_ARGS=$TRUNK_PROD_EXTRA_ARGS --host 127.0.0.1") SERVE_SETTINGS+=("EXTRA_ARGS=$TRUNK_PROD_EXTRA_ARGS") ;;
      EXTRA_ARGS=*) die settings "unexpected EXTRA_ARGS in the b1 settings" ;;
      QSA_ROWCACHE_SLOTS=*) [[ $kv == "QSA_ROWCACHE_SLOTS=$ENGINE_ROWCACHE_SLOTS" ]] || SERVE_ROWCACHE_CHANGED=1
                            SERVE_SETTINGS+=("QSA_ROWCACHE_SLOTS=$ENGINE_ROWCACHE_SLOTS") ;;
      EXTRA_ENV=*) [[ $kv != *FLASH_S5_NATIVE_MM* ]] || die settings "duplicate MM switch"
                   SERVE_SETTINGS+=("$kv FLASH_S5_NATIVE_MM=1 FLASH_S5_G4_HEADROOM=1") ;;
      MAX_MODEL_LEN=*|MAX_SEQS=*) die settings "the b1 settings set $kv (not expected)" ;;
      *) SERVE_SETTINGS+=("$kv") ;;
    esac
  done
  SERVE_SETTINGS+=("MAX_MODEL_LEN=$MAX_MODEL_LEN" "MAX_SEQS=$MAX_SEQS" "VISION=1" 'VISION_ARGS=--enable-mm-embeds --limit-mm-per-prompt {"image":0,"video":0} --mm-processor-cache-gb 0 --skip-mm-profiling')
  (( ${#SERVE_SETTINGS[@]} == ${#S5_SETTINGS[@]} + 4 )) || die settings "settings count"
  step settings 0 "b1 S5_SETTINGS (${#S5_SETTINGS[@]}): EXTRA_ARGS host dropped, MAX_MODEL_LEN=$MAX_MODEL_LEN, MAX_SEQS=$MAX_SEQS, QSA_ROWCACHE_SLOTS=$ENGINE_ROWCACHE_SLOTS${SERVE_ROWCACHE_CHANGED:+ (CHANGED from cxfx42 6144)}"
else
  [[ $(sha256sum "$PK/run-flash-stock-keyed.sh" | cut -c1-64) == "$STOCK_LAUNCHER_SHA" ]] ||
    die pins "keyed stock launcher is not ${STOCK_LAUNCHER_SHA:0:12}"
  SERVE_SETTINGS+=("QSA_ROWCACHE_SLOTS=$STOCK_ROWCACHE_SLOTS" "SERVE_AUTH_ENV_FILE=${AUTH_ENV#$R}")
  step settings 0 "found defaults with MAX_MODEL_LEN=$MAX_MODEL_LEN, MAX_SEQS=$MAX_SEQS, QSA_ROWCACHE_SLOTS=$STOCK_ROWCACHE_SLOTS, SERVE_AUTH_ENV_FILE (keyed stock launcher ${STOCK_LAUNCHER_SHA:0:12} = found 45cc5d81 + --env-file)"
fi
printf '%s\n' "${SERVE_SETTINGS[@]}" | sed 's/SHA256=[0-9a-f]*/SHA256=<pin>/g' | cut -c1-200 > "$OUT/serve-settings.txt"

# ---- snapshot and install (changes start here) ----
DEFAULTS_MODE=$(stat -c '%a %U %G' "$DEFAULTS"); LAUNCHER_MODE=$(stat -c '%a %U %G' "$LAUNCHER")
cp -p "$DEFAULTS" "$OUT/private/defaults.found" && chmod 600 "$OUT/private/defaults.found" &&
  cp -p "$LAUNCHER" "$OUT/private/launcher.found" && chmod 600 "$OUT/private/launcher.found" || die snapshot "snapshot failed"
printf 'DEFAULTS_MODE="%s"\nLAUNCHER_MODE="%s"\nDEFAULTS_FOUND_SHA=%s\nLAUNCHER_FOUND_SHA=%s\n' "$DEFAULTS_MODE" "$LAUNCHER_MODE" \
  "$(sha256sum "$OUT/private/defaults.found" | cut -c1-64)" "$(sha256sum "$OUT/private/launcher.found" | cut -c1-64)" > "$OUT/found.env"
step snapshot 0 "found launcher [$LAUNCHER_MODE] and defaults [$DEFAULTS_MODE] kept in $OUT/private"
INSTALL_OWN=(-o "$OWNER" -g "$(id -gn "$OWNER" 2>/dev/null || echo "$OWNER")")
if [[ $ARM == engine ]]; then
  install -m 0755 "${INSTALL_OWN[@]}" "$OVERLAY/run-flash-s5-test.sh" "$LAUNCHER.q18a-restore" && mv "$LAUNCHER.q18a-restore" "$LAUNCHER" &&
    cmp -s "$LAUNCHER" "$OVERLAY/run-flash-s5-test.sh" && runuser -u "$SERVICE_USER" -- test -r "$LAUNCHER" ||
    fail_stop launcher-install "overlay launcher not installed"
  step launcher-install 0 "cxfx42 S5 launcher installed 0755 $OWNER"
else
  install -m 0755 "${INSTALL_OWN[@]}" "$PK/run-flash-stock-keyed.sh" "$LAUNCHER.q18a-restore" && mv "$LAUNCHER.q18a-restore" "$LAUNCHER" &&
    cmp -s "$LAUNCHER" "$PK/run-flash-stock-keyed.sh" && runuser -u "$SERVICE_USER" -- test -r "$LAUNCHER" ||
    fail_stop launcher-install "keyed stock launcher not installed"
  step launcher-install 0 "keyed stock launcher ${STOCK_LAUNCHER_SHA:0:12} installed 0755 $OWNER (found 45cc5d81 + --env-file)"
fi
python3 "$SETCFG" "${SERVE_SETTINGS[@]}" > /dev/null 2>&1 && [[ -f $NEXT && ! -L $NEXT ]] || fail_stop configure "setcfg failed"
python3 -I -B "$PK/serve_readback.py" rendered --arm "$ARM" --defaults "$NEXT" --found "$OUT/private/defaults.found" \
  --params "$PK/serve-params.env" --overlay "${OVERLAY#$R}" --auth "${AUTH_ENV#$R}" > "$OUT/serve-defaults.txt" 2>&1 ||
  fail_stop configure "rendered defaults differ (serve-defaults.txt)"
install -m 0600 "${INSTALL_OWN[@]}" "$NEXT" "$DEFAULTS.q18a-restore" && mv "$DEFAULTS.q18a-restore" "$DEFAULTS" && cmp -s "$DEFAULTS" "$NEXT" &&
  mv -n "$NEXT" "$OUT/private/defaults.serve" && [[ ! -e $NEXT ]] || fail_stop configure "serve defaults not installed"
step configure 0 "serve defaults installed 0600 ($(grep -c . "$OUT/serve-defaults.txt") readback lines; staged .next moved into $OUT/private)"

# ---- engine start (start, never enable; Conflicts=vllm-qwen38 is already satisfied: prod stopped) ----
setsid nohup bash -c 'for i in $(seq 1 600); do docker inspect "$1" >/dev/null 2>&1 && exec docker logs -f --timestamps "$1"; sleep 1; done' \
  serve-log-follower "$CONTAINER" > "$OUT/engine.log" 2>&1 < /dev/null &
echo "FOLLOWER_PID=$!" > "$OUT/pids.env"
systemctl reset-failed "$SERVICE" > /dev/null 2>&1
N0=$(systemctl show "$SERVICE" -p NRestarts --value); echo "N0=$N0" >> "$OUT/pids.env"
timeout --signal=TERM --kill-after=20s 300s systemctl start "$SERVICE" > "$OUT/start.log" 2>&1 || fail_stop start "systemctl start failed"
rc=1; t0=$SECONDS; end=$((SECONDS + HEALTH_DEADLINE_S))
while (( SECONDS < end )); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$URL/health" 2>/dev/null) || true
  [[ $code == 200 ]] && { rc=0; break; }
  systemctl is-failed -q "$SERVICE" && break
  [[ $(systemctl show "$SERVICE" -p NRestarts --value) == "$N0" ]] || break
  sleep 2
done
[[ $rc == 0 ]] || fail_stop health "no /health 200 within ${HEALTH_DEADLINE_S}s (or failed/auto-restarted)"
step health 0 "/health 200 after $((SECONDS - t0))s (readiness: owner proof latched)"
sleep 3
python3 -I -B "$PK/serve_readback.py" engine-args --log "$OUT/engine.log" --params "$PK/serve-params.env" > "$OUT/engine-args.txt" 2>&1 ||
  fail_stop engine-args "engine args differ (engine-args.txt)"
if command -v ss >/dev/null 2>&1; then
  ss -ltnH "sport = :$PORT" > "$OUT/listen.txt" 2>&1; grep -q "127.0.0.1:$PORT" "$OUT/listen.txt" || fail_stop listen "not listening on 127.0.0.1:$PORT"
else
  grep -qE "^ *[0-9]+: 0100007F:$(printf '%04X' "$PORT") 00000000:0000 0A" /proc/net/tcp || fail_stop listen "not listening on 127.0.0.1:$PORT (/proc/net/tcp)"
fi
step engine-args 0 "$(tail -1 "$OUT/engine-args.txt") listen 127.0.0.1:$PORT"
python3 -I -B "$PK/serve_readback.py" container --arm "$ARM" --container "$CONTAINER" --params "$PK/serve-params.env" \
  --g0-env "$PK/vision-container-env.json" --overlay "${OVERLAY#$R}" --url "$URL" --out "$OUT/readback.json" > "$OUT/readback.txt" 2>&1 ||
  fail_stop readback "effective settings readback differs: $(grep -h '^SERVE_READBACK' "$OUT/readback.txt" | cut -c1-300)"
step readback 0 "$(grep -h '^SERVE_READBACK' "$OUT/readback.txt" | cut -c1-200)"
python3 -I -B "$PK/serve_vram.py" gate --arm "$ARM" --params "$PK/serve-params.env" --phase startup --out "$OUT/vram-startup.json" > "$OUT/vram-startup.txt" 2>&1 ||
  fail_stop vram-startup "$(grep -h '^SERVE_VRAM' "$OUT/vram-startup.txt" | cut -c1-300)"
step vram-startup 0 "$(grep -h '^SERVE_VRAM' "$OUT/vram-startup.txt" | cut -c1-240)"

if [[ $ARM == engine ]]; then
  # ---- the n3 lane (b16's cx order): startup grade binds the run_id, typed all-rank FULL, n3 select, auto-arm released ----
  timeout 600s python3 -B "$PK/serve_lane.py" cx-step --h "$H" -- e3_split_startup_grade.py --url "$URL" --run-id "$RUN_ID" \
    --package "${OVERLAY}" --out "$OUT/startup-grade.json" > "$OUT/grade.log" 2>&1
  rc=$?
  grep -h '^SERVE_IDENTITY_DIFF' "$OUT/grade.log" >> "$OUT/steps.log"
  EPOCH=$(python3 - "$OUT/startup-grade.json" 2>>"$OUT/grade.log" <<'PY_EPOCH'
import json,sys
d=json.load(open(sys.argv[1]));e=d.get('engine_epoch')
assert type(e) is int and e>0 and d.get('verdict')=='PASS'
print(e)
PY_EPOCH
) || fail_stop startup "startup grade rc=$rc not PASS: $(grep -h 'NATIVE_STARTUP\|SERVE_IDENTITY_DIFF' "$OUT/grade.log" | tr '\n' ' ' | cut -c1-300)"
  step startup 0 "cx startup grade PASS, engine_epoch $EPOCH, session run_id bound to $RUN_ID"
  python3 -B "$PK/tools/vision_mm_memory_read.py" --startup-grade "$OUT/startup-grade.json" --out "$OUT/mm-memory.json" > "$OUT/mm-memory.log" 2>&1 ||
    fail_stop mm-memory "all-rank MM construction budget not proven"
  step mm-memory 0 "$(tail -1 "$OUT/mm-memory.log")"
  echo "EPOCH=$EPOCH RUN_ID=$RUN_ID" >> "$OUT/pids.env"
  timeout 450s python3 -B "$PK/serve_lane.py" cx-step --h "$H" -- e3_native39_trunk_checks.py full-idle --url "$URL" --run-id "$RUN_ID" \
    --epoch "$EPOCH" --control-timeout-s 60 --timeout-s 120 --out "$OUT/entry-full-idle.json" > "$OUT/entry-full-idle.log" 2>&1 ||
    fail_stop full-idle "typed all-rank FULL not proven: $(grep -h 'SERVE_IDENTITY_DIFF' "$OUT/entry-full-idle.log" | tr '\n' ' ' | cut -c1-200)"
  step full-idle 0 "typed all-rank FULL, idle, lease taken (epoch $EPOCH)"
  timeout 300s python3 -B "$PK/serve_lane.py" select --h "$H" --url "$URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --out "$OUT/lane-select.json" > "$OUT/lane-select.log" 2>&1 || fail_stop lane-select "$(grep -h 'SERVE_LANE\|SERVE_IDENTITY_DIFF' "$OUT/lane-select.log" | tr '\n' ' ' | cut -c1-300)"
  step lane-select 0 "$(grep -h '^SERVE_LANE ' "$OUT/lane-select.log" | tr '\n' ' ' | cut -c1-200)"
  LANE_ARGS=(--h "$H" --run-id "$RUN_ID")
else
  echo "RUN_ID=$RUN_ID" >> "$OUT/pids.env"
  LANE_ARGS=(--no-lane)
fi

# ---- recorder (60 s ticks, VRAM floor every 5 s) and smoke ----
setsid nohup python3 -B "$PK/serve_recorder.py" --out "$OUT" --url "$URL" --engine-log "$OUT/engine.log" --params "$PK/serve-params.env" \
  --arm "$ARM" --root-path "${R:-/}" "${LANE_ARGS[@]}" > "$OUT/recorder.log" 2>&1 < /dev/null &
echo "RECORDER_PID=$!" >> "$OUT/pids.env"
step recorder 0 "pid $! -> $OUT/recorder.jsonl every 60 s, VRAM floor ${VRAM_FLOOR_MIB} MiB every ${VRAM_SAMPLE_S} s"
python3 -B "$PK/serve_smoke.py" --arm "$ARM" --url "$URL" --max-model-len "$MAX_MODEL_LEN" --max-seqs "$MAX_SEQS" "${LANE_ARGS[@]}" \
  --out "$OUT/smoke.json" 2>&1 | tee "$OUT/smoke.log"
SMOKE=$(grep -h '^SERVE_SMOKE' "$OUT/smoke.log" | tail -1)
[[ $SMOKE == 'SERVE_SMOKE PASS' ]] || fail_stop smoke "$SMOKE"
step smoke 0 "$SMOKE; $(grep -h '^SERVE_MESSAGES' "$OUT/smoke.log")"
python3 -I -B "$PK/serve_vram.py" gate --arm "$ARM" --params "$PK/serve-params.env" --phase after-smoke --out "$OUT/vram-after-smoke.json" > "$OUT/vram-after-smoke.txt" 2>&1 ||
  fail_stop vram-after-smoke "$(grep -h '^SERVE_VRAM' "$OUT/vram-after-smoke.txt" | cut -c1-300)"
step vram-after-smoke 0 "$(grep -h '^SERVE_VRAM' "$OUT/vram-after-smoke.txt" | cut -c1-240)"
n=$(systemctl show "$SERVICE" -p NRestarts --value)
echo "SERVE_UP $(date -u +%FT%TZ) arm=$ARM 127.0.0.1:$PORT model Qwen3.8-Flash-Next max_model_len $MAX_MODEL_LEN max_num_seqs $MAX_SEQS rowcache $([[ $ARM == engine ]] && echo "$ENGINE_ROWCACHE_SLOTS" || echo "$STOCK_ROWCACHE_SLOTS") auth $(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['auth'])" "$OUT/readback.json") NRestarts $N0->$n params $PARAMS_STATUS run_id $RUN_ID out $OUT; $SMOKE" | tee -a "$OUT/steps.log"
echo "drain+stop: bash $PK/serve-drain.sh $OUT   (stop at once: bash $PK/serve-stop.sh $OUT)"
