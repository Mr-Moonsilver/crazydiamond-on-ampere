#!/bin/bash
# Release check for Flash-Next cache/kernel/scheduler changes on CT <ct-id> (run inside the CT after a restart).
# Exit 0 only if every gate passes; any failed test, missing result, engine error or unequal copied page fails it.
# 1. mounts and the QSA host-page CoW hook (one "host KV views" line per rank, views > 0);
# 2. partial-copy tests (leak_partial_tail.py): MODE=inside 10 x 6 with a 3rd request, MODE=after 10 x 4;
# 3. <review-group>'s concurrent leak test (vllm-leaktest.py): agent-like 3 seeds x 10 x 3, standard 2 seeds x 6 x 3;
# 4. with QSA_COW_VERIFY=1: every logged copy compared equal after the copy, on every rank;
# 5. no Traceback / CUDA error / Xid in the engine log; optional decode bench 8 x 60k.
# The real-route canary (harness/flash_canary.sh on the hub) is run separately.
# usage: verify-release.sh OUTDIR [skip-bench]
set -u
OUT=${1:?outdir}; mkdir -p "$OUT"
URL=http://127.0.0.1:8000; MODEL=qwen38-w8a8-tp4; H=/home/<service-user>
C=$(docker ps -q --filter name=qwen-flash-prod | head -1)
[[ -n $C ]] || { echo "no engine container"; exit 1; }
since=$(docker inspect "$C" --format '{{.State.StartedAt}}')
fail=0
gate() { local name=$1; shift; if "$@"; then echo "PASS $name"; else echo "FAIL $name"; fail=1; fi; }

echo "== deployment"
docker inspect "$C" --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' | grep -E 'overlay' | tee "$OUT/mounts.txt"
docker inspect "$C" --format '{{.Config.Env}}' | grep -oE 'QSA_[A-Z_]+=[^ ]+|FLASH_PD_ALTERNATE=[^ ]+' | tee "$OUT/env.txt"
docker inspect "$C" --format '{{.Args}}' | grep -oE 'prefix-match-unit [0-9]+' | tee -a "$OUT/env.txt"
for f in $(grep -E 'overlay' "$OUT/mounts.txt"); do sha256sum "$f"; done > "$OUT/sha256.txt"

views=$(docker logs --since "$since" "$C" 2>&1 | grep -c 'QSA host-page CoW: [1-9][0-9]* host KV views')
gate "cow-hook on 4 ranks ($views)" test "$views" -ge 4

echo "== partial-copy tests"
gate "inside 10x6 +req3" env MODE=inside timeout 3000 python3 $H/leak_partial_tail2.py $URL $MODEL 10 6 7568 "$OUT/inside.json"
gate "after 10x4" env MODE=after timeout 2400 python3 $H/leak_partial_tail2.py $URL $MODEL 10 4 7568 "$OUT/after.json"

echo "== <review-group> leak test"
for s in 31 32 33; do
  r=$(RULES=45 FILL=60 SEED=$s GAP=2 timeout 1200 python3 $H/vllm-leaktest.py 10 3 2>&1 | tee "$OUT/steer-agent-$s.log" | tail -1)
  gate "steer agent seed $s: $r" grep -q '^sessions 10 rounds 3 bad 0 ' <<<"$r"
done
for s in 41 42; do
  r=$(SEED=$s timeout 1500 python3 $H/vllm-leaktest.py 6 3 2>&1 | tee "$OUT/steer-std-$s.log" | tail -1)
  gate "steer std seed $s: $r" grep -q '^sessions 6 rounds 3 bad 0 ' <<<"$r"
done

if grep -q 'QSA_COW_VERIFY=1' "$OUT/env.txt"; then
  echo "== copied host pages"
  docker logs --since "$since" "$C" 2>&1 | grep 'CoW verify' > "$OUT/cow-verify.log"
  n=$(wc -l < "$OUT/cow-verify.log"); bad=$(grep -vc ', after 0 (' "$OUT/cow-verify.log")
  tail -4 "$OUT/cow-verify.log" | cut -c1-240
  gate "cow verify lines $n, unequal after copy $bad" test "$n" -gt 0 -a "$bad" -eq 0
fi

echo "== engine errors"
errs=$(docker logs --since "$since" "$C" 2>&1 | grep -E 'Traceback|AssertionError|CUDA error|Xid' | head -5)
gate "no engine errors" test -z "$errs"; [[ -z $errs ]] || echo "$errs"

if [[ ${2:-} != skip-bench ]]; then
  echo "== decode 8 x 60k"
  QUICK_WINDOW_S=30 timeout 900 python3 /tmp/quick_c8.py $URL $MODEL 8 60000 2048 release-8x60k "$OUT/bench-8x60k.json" 2>&1 | tail -4
fi
[[ $fail == 0 ]] && { echo "RELEASE CHECK PASS"; exit 0; }
echo "RELEASE CHECK FAIL"; exit 1
