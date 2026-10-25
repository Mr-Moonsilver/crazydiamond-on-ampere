# Shared experiment lifecycle. Source only from an owner-allocated GPU queue.
# Inputs: H, Q (new), BASE, CAND. No production restore, no auto-arming.
set -uo pipefail
umask 077
P=/data/vllm/flash-next-tp4/prod
U=http://127.0.0.1:8000; M=qwen38-w8a8-tp4
[[ ! -e $Q ]] || exit 2
mkdir -p "$Q/private" || exit 1
(cd "$H" && sha256sum --quiet -c SHA256SUMS) || exit 1
source "$H/qlib.sh"
CHANGED=0; DONE=0; LAUNCHER_SWAPPED=0
idle_wait() { local i; for i in $(seq 1 36); do gpu_idle && return 0; sleep 5; done; return 1; }
cleanup_variant() {
  trap - EXIT INT TERM HUP
  [[ $DONE == 1 ]] || step interrupted 1
  if [[ $CHANGED == 1 ]]; then
    systemctl stop vllm-flash > "$Q/stop.log" 2>&1; step stop "$?"
    idle_wait; step gpu-handoff "$?"
    if [[ $LAUNCHER_SWAPPED == 1 ]]; then
      cp -p "$Q/private/launcher" "$P/run-flash-prod.sh.experiment-back" &&
      mv "$P/run-flash-prod.sh.experiment-back" "$P/run-flash-prod.sh" &&
      cmp -s "$P/run-flash-prod.sh" "$Q/private/launcher"; step launcher-restore "$?"
    fi
    cp -p "$Q/private/defaults" /etc/default/vllm-flash.experiment-back &&
      mv /etc/default/vllm-flash.experiment-back /etc/default/vllm-flash &&
      cmp -s /etc/default/vllm-flash "$Q/private/defaults"; step defaults-restore "$?"
  fi
  finish EXPERIMENT-LEFT-STOPPED
}
trap cleanup_variant EXIT; trap 'exit 130' INT TERM HUP
run_step() { local n=$1 rc; shift; "$@" > "$Q/$n.log" 2>&1; rc=$?; step "$n" "$rc"; return "$rc"; }
capture_variant() {
  local name=$1 out=$2 request=$3 rc
  identity "$out" "$request" --timeout 1800 > "$Q/$name.log" 2>&1; rc=$?
  step "$name" "$rc" "$(identity_cost "$out")"; return "$rc"
}
prepare_variant() {
  [[ $(sha256sum /tmp/setcfg.py | cut -d' ' -f1) == f3cae1379fa9079785c03a5555dbf739c687c5dcde9b1c6e058b38e8849754fe ]] || return 1
  [[ $(sha256sum "$H/run-flash-prod.sh.q9" | cut -d' ' -f1) == 98d6d2a21927b8943f1ae9b0b43616757e80adac0c628e65f482e4332e3cf126 ]] || return 1
  cp -p /etc/default/vllm-flash "$Q/private/defaults" && cp -p "$P/run-flash-prod.sh" "$Q/private/launcher" || return 1
  CHANGED=1
  systemctl stop vllm-flash && idle_wait || return 1
  LAUNCHER_SWAPPED=1
  cp "$H/run-flash-prod.sh.q9" "$P/run-flash-prod.sh.experiment-new" &&
    mv "$P/run-flash-prod.sh.experiment-new" "$P/run-flash-prod.sh" && cmp -s "$P/run-flash-prod.sh" "$H/run-flash-prod.sh.q9"
}
# Declare sources AFTER the planned launcher install, BEFORE any engine creation.
# Extra settings are explicit per case. Inputs never come from a running engine.
start_variant() {
  local name=$1 overlay=$2 extra=$3; shift 3
  systemctl stop vllm-flash && idle_wait || return 1
  local settings=("OVERLAY=$overlay" CORE_OVERLAY=1 PP_LAYER_PARTITION=24,24 MAX_SEQS=10
    MODEL_DIR=- VISION=- VISION_ARGS=- BREAKABLE_CG=- COMPILATION_CONFIG=- MTP_TOKENS=0
    FLASH_SCHED_STATS=200 PREFILL_CHUNK=1024 FLASH_PD_ALTERNATE=1 "QSA_COW_VERIFY=${COW_VERIFY:--}"
    "EXTRA_ENV=QSA_STAGE_TIMING=${STAGE_TIMING:-0} FLASH_PP_GROUPING=1 FLASH_POOL_STATS=200 $extra" "$@")
  run_step "$name.declare" python3 "$H/engine_identity.py" declare "$Q/$name.request.json" \
    --launcher "$H/run-flash-prod.sh.q9" --mount-policy "$H/pp_release_mount_policy.json" --settings "${settings[@]}" || return 1
  python3 /tmp/setcfg.py "${settings[@]}" > /dev/null 2>&1 || return 1
  systemctl reset-failed vllm-flash
  run_step "$name.restart" /usr/local/sbin/flash-reconfig.sh || return 1
  CURRENT=$name
  capture_variant "$name.identity" "$Q/$name.identity.json" "$Q/$name.request.json"
}
measured() {
  local n=$1; shift
  capture_variant "$n.before" "$Q/$n.before.json" "$Q/$CURRENT.request.json" || return 1
  local rc=0 ar=0
  run_step "$n" "$@" || rc=$?
  capture_variant "$n.after" "$Q/$n.after.json" "$Q/$CURRENT.request.json" || ar=$?
  run_step "$n.stable" identity_check "$Q/$n.before.json" "$Q/$CURRENT.request.json" "$Q/$n.after.json" || ar=$?
  [[ $rc == 0 && $ar == 0 ]]
}
w2_variant() { measured "$1" timeout 1800 python3 "$H/stall_trace2.py" "$U" "$M" 8 153537 200000 "$1" "$Q/$1.json"; }
leak_variant() {
  measured "$1" timeout 1800 env MODE=inside GEN=200 TIMEOUT=900 NEED_PARTIAL=1 \
    python3 "$H/leak_partial_tail.py" "$U" "$M" 10 4 7568 "$Q/$1.json" || return 1
  run_step "$1.grade" python3 "$H/pp_release_grade.py" leak "$Q/$1.json" 10 4 7568 0 1
}
