#!/bin/bash
# Drop w6 (12 x 150k) from the running chunk series and run w7 (8 x 262k) instead.
R=/data/vllm/gpu-tuning-20261021/flash-quick
until grep -q "w5m-chunk2048\] launch" $R/quick.log; do sleep 10; done
for p in $(pgrep -f "tools/series-w5m\.s[h]") $(pgrep -f "/after-w1\.s[h]"); do kill $p; done
echo "$(date -u +%FT%TZ) [plan] w6 (12 x 150k) dropped for w7 (8 x 262k, agentic coding)" >> $R/quick.log
until grep -qE "w5m-chunk2048\] (stopped|FAILED)" $R/quick.log; do sleep 15; done
sleep 5
exec /data/vllm/gpu-tuning-20261021/tools/series-w7.sh
