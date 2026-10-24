# Shared queue helpers (Codex round 4), sourced by the weekend queues from the checksummed harness directory:
#   step NAME RC [TEXT]   prints "STEP NAME PASS|INCONCLUSIVE|FAIL rc=N TEXT" and counts the verdict
#   finish NAME           prints "NAME DONE <time> fails=F inconclusive=I"; exit 1 on any FAIL, 2 on any INCONCLUSIVE
#   installed OVL CHUNK WORDS..  the running container is the configuration a case asked for: its qsa.py mount comes
#                         from overlay OVL, it runs --max-num-batched-tokens CHUNK, every K=V word is in its environment, and every candidate switch (SWITCHES)
#                         not named in WORDS is absent -- a rolled-back restart leaves the previous settings, and a
#                         measurement on them must not carry the new case's name
#   gpu_idle              the serving engine is stopped, its container gone and no process holds a GPU
#   stall_case NAME OVL tCHUNK XCFG XENV   /tmp/stall2.sh (restart + W1/W2 trace) into $Q/NAME, its STEP lines prefixed
#                         and counted here; returns 0 only when the restart passed and `installed` holds
#   ident                 container id + start time (a restart between two points invalidates what lies between)
FAILS=0; INCS=0
SWITCHES="QSA_HOST_META QSA_HOST_META_VERIFY QSA_ROWCACHE_MIXED QSA_RETAIN_HISTORY QSA_RETAIN_PAGES QSA_RETAIN_VERIFY
QSA_COW_VERIFY FLASH_SHORT_FIRST FLASH_STEP_TARGET_MS FLASH_CHUNK_BUSY FLASH_PD_DECODE_SHARE"
step() {
  local v
  case $2 in (0) v=PASS;; (2) v=INCONCLUSIVE; INCS=$((INCS + 1));; (*) v="FAIL rc=$2"; FAILS=$((FAILS + 1));; esac
  echo "STEP $1 $v ${3:-}"
}
finish() {
  echo "$1 DONE $(date -u +%T) fails=$FAILS inconclusive=$INCS"
  [[ $FAILS == 0 ]] || exit 1; [[ $INCS == 0 ]] || exit 2; exit 0
}
ident() { docker inspect qwen-flash-prod --format '{{.Id}} {{.State.StartedAt}}' 2>/dev/null; }
installed() {
  local ovl=$1 chunk=$2 env mounts kv k; shift 2
  env=$(docker inspect qwen-flash-prod --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null)
  mounts=$(docker inspect qwen-flash-prod --format '{{range .Mounts}}{{println .Source}}{{end}}' 2>/dev/null)
  [[ -n $env ]] || { echo "no engine container"; return 1; }
  grep -qx -- "/data/vllm/flash-next-tp4/prod/$ovl/qsa.py" <<<"$mounts" || { echo "qsa.py not from $ovl"; return 1; }
  docker inspect qwen-flash-prod --format '{{json .Args}}' | grep -qF "\"--max-num-batched-tokens\",\"$chunk\"" ||
    { echo "chunk is not $chunk"; return 1; }
  for kv in "$@"; do grep -qxF -- "$kv" <<<"$env" || { echo "missing $kv"; return 1; }; done
  for k in $SWITCHES; do
    [[ " $* " == *" $k="* ]] && continue
    grep -q "^$k=" <<<"$env" && { echo "unexpected $(grep "^$k=" <<<"$env")"; return 1; }
  done
  return 0
}
gpu_idle() {
  systemctl is-active -q vllm-flash && return 1
  [[ -z $(docker ps -q -f name=^qwen-flash-prod$) ]] || return 1
  [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]] || return 1
  # processes outside this container may not be listed inside the LXC: every GPU's used memory must be near zero too
  local m; m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  [[ -n $m ]] && ! awk '$1 > 500 {bad=1} END {exit !bad}' <<<"$m"
}
stall_case() {
  local name=$1 ovl=$2 c=$3 xcfg=$4 xenv=$5 why
  OUT=$Q/$name XCFG="$xcfg" XENV="$xenv" OVL=$ovl bash /tmp/stall2.sh $c > $Q/$name.out 2>&1
  sed "s/^STEP /STEP $name /" $Q/$name.out
  FAILS=$((FAILS + $(grep -c '^STEP .* FAIL' $Q/$name.out))); INCS=$((INCS + $(grep -c '^STEP .* INCONCLUSIVE' $Q/$name.out)))
  grep -q "^STEP restart-$c PASS" $Q/$name.out || return 1
  why=$(installed $ovl ${c#t} QSA_STAGE_TIMING=1 $xcfg $xenv); step installed-$name $? "$why"
  [[ -z $why ]]
}
