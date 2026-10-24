#!/bin/bash
# Release check for Flash-Next cache/kernel/scheduler changes on CT <ct-id> (run inside the CT after a restart).
# Every gate ends PASS / FAIL / INCONCLUSIVE / NOT_RUN; the run is PASS (exit 0) only if every gate PASSed.
# Any FAIL -> exit 1; otherwise any INCONCLUSIVE or NOT_RUN -> exit 2. Gates:
#   harness     HARNESS_DIR's SHA256SUMS verify (only versioned harness files run; no /tmp copies)
#   identity    engine container id/start identical at start and end (a restart invalidates the run)
#   cow-hook    "host KV views" line from 4 distinct worker ranks with views > 0
#   inside      leak_partial_tail.py MODE=inside GEN=200 10 x 6 (+ 3rd requests)   (its exit 0/1/2 maps to the verdict)
#   after       leak_partial_tail.py MODE=after 10 x 4
#   steer-*     vllm-leaktest.py agent-like 3 seeds x 10 x 3, standard 2 seeds x 6 x 3 (parsed: "bad 0")
#   coverage    copy counters (verify off: "CoW count", verify on: "CoW verify") advanced on all 4 ranks DURING
#               this run (log lines since the run start)
#   bytes       verify mode only: cow_invariants.py on this run's verify lines -- prefix equal after every copy, sources
#               and control blocks unchanged, re-checks ran on all 4 ranks (else NOT_RUN)
#   errors      no Traceback / CUDA error / Xid / AssertionError since the container started
#   bench       quick_c8 8 x 60k: valid JSON, 8 streams > 0; >= BENCH_MIN tok/s if set (NOT_RUN with skip-bench)
# usage: HARNESS_DIR=/home/<service-user>/harness/<ver> verify-release.sh OUTDIR [skip-bench]
set -u
OUT=${1:?outdir}; mkdir -p "$OUT"; H=${HARNESS_DIR:?set HARNESS_DIR}
URL=http://127.0.0.1:8000; MODEL=qwen38-w8a8-tp4
declare -A V; ORDER=()
put() { V[$1]=$2; ORDER+=("$1"); echo "$2 $1${3:+ - $3}"; }
map() { case $1 in 0) echo PASS;; 2) echo INCONCLUSIVE;; *) echo FAIL;; esac; }
ident() { docker inspect qwen-flash-prod --format '{{.Id}} {{.State.StartedAt}}' 2>/dev/null; }
RUN_START=$(date -u +%FT%TZ)

if (cd "$H" && sha256sum --quiet -c SHA256SUMS); then put harness PASS "$H"; else put harness FAIL "checksum mismatch in $H"; fi
ID0=$(ident); [[ -n $ID0 ]] || { put identity FAIL "no engine container"; }
HARNESS_DIR=$H bash "$H/manifest.sh" "$OUT/manifest-start.txt" > /dev/null
STARTED=$(docker inspect qwen-flash-prod --format '{{.State.StartedAt}}')

ranks=$(docker logs --since "$STARTED" qwen-flash-prod 2>&1 | grep -E 'QSA host-page CoW: [1-9][0-9]* host KV views' |
        grep -oE 'Worker_PP[0-9]+_TP[0-9]+' | sort -u | wc -l)
[[ $ranks -eq 4 ]] && put cow-hook PASS "4 ranks" || put cow-hook FAIL "$ranks distinct ranks"

# GEN=200 keeps every 2nd request inside its page, so it hits its partial tail (a longer request 1 output crosses the page
# end and the stock hybrid lookup falls back to the previous full block: no copy, no coverage)
timeout 3600 env MODE=inside GEN=200 TIMEOUT=1800 python3 "$H/leak_partial_tail.py" $URL $MODEL 10 6 7568 "$OUT/inside.json" > "$OUT/inside.log" 2>&1
put inside "$(map $?)" "$(tail -1 "$OUT/inside.log" | cut -c1-160)"
timeout 3000 env MODE=after python3 "$H/leak_partial_tail.py" $URL $MODEL 10 4 7568 "$OUT/after.json" > "$OUT/after.log" 2>&1
put after "$(map $?)" "$(tail -1 "$OUT/after.log" | cut -c1-160)"

for s in 31 32 33; do
  timeout 1200 env RULES=45 FILL=60 SEED=$s GAP=2 python3 "$H/vllm-leaktest.py" 10 3 > "$OUT/steer-agent-$s.log" 2>&1; rc=$?
  r=$(tail -1 "$OUT/steer-agent-$s.log")
  [[ $rc -eq 0 && $r =~ ^sessions\ 10\ rounds\ 3\ bad\ 0\  ]] && put steer-agent-$s PASS "$r" || put steer-agent-$s FAIL "rc=$rc $r"
done
for s in 41 42; do
  timeout 1500 env SEED=$s python3 "$H/vllm-leaktest.py" 6 3 > "$OUT/steer-std-$s.log" 2>&1; rc=$?
  r=$(tail -1 "$OUT/steer-std-$s.log")
  [[ $rc -eq 0 && $r =~ ^sessions\ 6\ rounds\ 3\ bad\ 0\  ]] && put steer-std-$s PASS "$r" || put steer-std-$s FAIL "rc=$rc $r"
done

docker logs --since "$RUN_START" qwen-flash-prod 2>&1 | grep -E 'QSA host-page CoW (count|verify)' > "$OUT/cow-run.log"
cov=$(grep -oE 'Worker_PP[0-9]+_TP[0-9]+' "$OUT/cow-run.log" | sort -u | wc -l)
[[ $cov -eq 4 ]] && put coverage PASS "copies logged on 4 ranks during this run" || put coverage FAIL "copies on $cov ranks during this run"
if docker inspect qwen-flash-prod --format '{{.Config.Env}}' | grep -q 'QSA_COW_VERIFY=1'; then
  grep 'CoW verify' "$OUT/cow-run.log" > "$OUT/cow-verify.log"
  r=$(python3 "$H/cow_invariants.py" "$OUT/cow-verify.log"); put bytes "${r%% *}" "${r#* }"
else
  put bytes NOT_RUN "verify mode off (byte check needs QSA_COW_VERIFY=1)"
fi

errs=$(docker logs --since "$STARTED" qwen-flash-prod 2>&1 | grep -E 'Traceback|AssertionError|CUDA error|Xid' | head -5)
[[ -z $errs ]] && put errors PASS || { put errors FAIL; echo "$errs"; }

if [[ ${2:-} == skip-bench ]]; then put bench NOT_RUN "skip-bench"; else
  QUICK_WINDOW_S=30 timeout 900 python3 "$H/quick_c8.py" $URL $MODEL 8 60000 2048 release-8x60k "$OUT/bench.json" > "$OUT/bench.log" 2>&1; rc=$?
  b=$(python3 - "$OUT/bench.json" "${BENCH_MIN:-0}" <<'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1])); tps = float(d['aggregate_decode_tps']); per = d['per_stream_tps']
    ok = len(per) == 8 and all(p > 0 for p in per) and tps >= float(sys.argv[2])
    print(('PASS' if ok else 'FAIL'), f"{tps} tok/s (min {sys.argv[2]})")
except Exception as e:  # noqa: BLE001
    print('FAIL', f'invalid bench result: {e}')
EOF
)
  [[ $rc -eq 0 ]] || b="FAIL rc=$rc"
  put bench "${b%% *}" "${b#* }"
fi

ID1=$(ident); [[ -n $ID0 && $ID0 == "$ID1" ]] && put identity PASS "$ID0" || put identity FAIL "container changed: $ID0 -> $ID1"
HARNESS_DIR=$H bash "$H/manifest.sh" "$OUT/manifest-end.txt" > /dev/null
{ for k in "${ORDER[@]}"; do echo "$k ${V[$k]}"; done; echo "END $(cat /proc/sys/kernel/random/uuid 2>/dev/null || date +%s%N)"; } > "$OUT/verdicts.txt"
if grep -q ' FAIL$' "$OUT/verdicts.txt"; then echo "RELEASE CHECK FAIL"; exit 1; fi
if grep -qE ' (INCONCLUSIVE|NOT_RUN)$' "$OUT/verdicts.txt"; then echo "RELEASE CHECK INCONCLUSIVE"; exit 2; fi
echo "RELEASE CHECK PASS"; exit 0
