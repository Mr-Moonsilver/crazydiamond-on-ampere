#!/bin/bash
# Release check for the row cache v2 on CT <ct-id> (run inside the CT after a restart into overlay-v2).
# 1. the runner overlay is mounted and the cache is really active (hit counters in the log);
# 2. the concurrent cross-session leak test: 3 seeds x 6 sessions x 3 turns, 2 seeds x 10 x 4;
# 3. single-session warm == cold greedy;
# 4. decode speed, 8 streams x 60k (compare with the same run on the cache-off setting).
# usage: verify-rowcache-v2.sh [skip-bench]
set -u
URL=http://localhost:8000; MODEL=qwen38-w8a8-tp4; W=/data/vllm/flash-next-tp4
C=$(docker ps -q --filter name=flash | head -1)
echo "== mounts"
docker inspect "$C" --format '{{range .Mounts}}{{.Destination}}{{"\n"}}{{end}}' | grep -E 'model_runner|qsa|core/'
echo "== leak test (pass = bad 0 everywhere)"
touch $W/rowcache.report
fail=0
for s in 501 502 503; do r=$(SEED=$s GAP=5 timeout 900 python3 /tmp/leaktest_local.py 6 3 | tail -1); echo "seed $s: $r"; [[ $r == *"bad 0 "* ]] || fail=1; done
for s in 504 505; do r=$(SEED=$s GAP=20 timeout 1500 python3 /tmp/leaktest_local.py 10 4 | tail -1); echo "seed $s: $r"; [[ $r == *"bad 0 "* ]] || fail=1; done
echo "== cache active (hit counters while the leak test ran)"
docker logs --since 30m "$C" 2>&1 | grep QSA_ROWCACHE_HITS | tail -4
n=$(docker logs --since 30m "$C" 2>&1 | grep -c QSA_ROWCACHE_HITS)
rm -f $W/rowcache.report
echo "hit lines: $n"; [[ $n -gt 0 ]] || { echo "CACHE NOT ACTIVE"; fail=1; }
echo "== single-session warm == cold"
python3 /tmp/salt_ab.py $URL $MODEL 6 1500 2>&1 | grep -E "warm==cold|RESULT"
if [[ ${1:-} != skip-bench ]]; then
  echo "== decode 8 x 60k"
  QUICK_WINDOW_S=30 timeout 900 python3 /tmp/quick_c8.py $URL $MODEL 8 60000 2048 v2-8x60k /tmp/v2-8x60k.json 2>&1 | tail -4
fi
echo "== errors"
docker logs --since 30m "$C" 2>&1 | grep -E "Traceback|AssertionError|CUDA error|Xid" | head -5
[[ $fail == 0 ]] && echo "RELEASE CHECK PASS" || echo "RELEASE CHECK FAIL"
