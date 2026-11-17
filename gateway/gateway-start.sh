#!/bin/bash
set -euo pipefail
PK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PK"
sha256sum --strict --quiet -c PACKET-SHA256SUMS
(cd harness && sha256sum --strict --quiet -c SHA256SUMS)
(cd harness/native30_overlay && sha256sum --strict --quiet -c SHA256SUMS)
AUTH=/root/.flashnext-serve-vision-prod-w27-20261117a/vllm-api.env
ENC=/root/.flashnext-serve-vision-prod-w27-20261117a/vision-encoder.env
for f in "$AUTH" "$ENC"; do
  [[ -f $f && ! -L $f && $(stat -c '%a %u' "$f") == '600 0' ]] || { echo 'GATEWAY HOLD private0600 auth missing'; exit 2; }
done
PIN=$(sha256sum PACKET-SHA256SUMS); PIN=${PIN:0:64}
if docker inspect flashnext-vision-gateway-w27 >/dev/null 2>&1; then
  [[ $(docker inspect -f '{{index .Config.Labels "flashnext.vision.packet"}} {{.State.Running}} {{.Image}}' flashnext-vision-gateway-w27) == "$PIN false sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b" ]] || { echo 'GATEWAY HOLD retained-container identity/state differs'; exit 2; }
  exec docker start --attach flashnext-vision-gateway-w27
fi
exec docker run --pull never --runtime runc --name flashnext-vision-gateway-w27 \
  --label flashnext.vision.gateway=vision-prod-w27-20261117a --label "flashnext.vision.packet=$PIN" --network host --read-only \
  --cap-drop ALL --security-opt no-new-privileges --pids-limit 128 --cpus 2 --memory 2g --memory-swap 2g \
  --log-driver local --log-opt max-size=16m --log-opt max-file=2 --tmpfs /tmp:rw,nosuid,nodev,size=64m \
  -v "$PK:/packet:ro" -v "$AUTH:/key/vllm-api.env:ro" -v "$ENC:/key/vision-encoder.env:ro" \
  -e PYTHONDONTWRITEBYTECODE=1 --entrypoint python3 \
  sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b \
  -B /packet/harness/vision_public_gateway.py --bind 0.0.0.0 --port 8000 --backend http://127.0.0.1:8001 \
  --encoder http://<encoder-host>:9041 --key-file /key/vllm-api.env --encoder-key-file /key/vision-encoder.env
