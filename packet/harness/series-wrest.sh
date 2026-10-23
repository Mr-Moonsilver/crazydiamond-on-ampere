#!/bin/bash
# Prefix caching with MTP off did not work: continue the original batch W after w1.
cd /data/vllm/gpu-tuning-20261021/tools || exit 1
sed -n '/^\.\/flash-w\.sh w2-/,$p' series-w.sh > /tmp/w-rest-body.sh
head -8 series-w.sh > /tmp/w-rest.sh; cat /tmp/w-rest-body.sh >> /tmp/w-rest.sh
exec bash /tmp/w-rest.sh
