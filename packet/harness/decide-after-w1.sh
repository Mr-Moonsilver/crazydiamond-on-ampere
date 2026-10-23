#!/bin/bash
# Waits for w1, then picks the branch: prefix reuse works (turn 2/3 TTFT well below turn 1, every answer
# correct) -> MTP-off chunk series; otherwise the rest of the original batch W. Then series X is not needed
# separately (the MTP-off low-concurrency run is part of w5m; the original X stays for the other branch).
R=/data/vllm/gpu-tuning-20261021/flash-quick
until grep -qE "w1-pc-mtp0\] (stopped|FAILED)" $R/quick.log; do sleep 15; done
sleep 5
verdict=$(python3 - <<'PY'
import json, statistics, sys
try:
    rows = json.load(open("/data/vllm/gpu-tuning-20261021/flash-quick/w1-pc-mtp0.prefix.json"))
except Exception:
    print("fail no-prefix-file"); sys.exit()
t1 = statistics.median(r["turns"][0]["ttft_s"] for r in rows)
t2 = statistics.median(r["turns"][1]["ttft_s"] for r in rows)
t3 = statistics.median(r["turns"][2]["ttft_s"] for r in rows)
ok = all(t["ok"] for r in rows for t in r["turns"])
print(("works" if ok and t2 < 0.3 * t1 and t3 < 0.3 * t1 else "fail") + f" turn1={t1} turn2={t2} turn3={t3} all_ok={ok}")
PY
)
echo "$(date -u +%FT%TZ) [decide] prefix caching MTP off: $verdict" >> $R/quick.log
case "$verdict" in
  works*) exec /data/vllm/gpu-tuning-20261021/tools/series-w5m.sh ;;
  *) /data/vllm/gpu-tuning-20261021/tools/series-wrest.sh; exec /data/vllm/gpu-tuning-20261021/tools/series-x.sh ;;
esac
