#!/bin/bash
set -euo pipefail
name=flashnext-vision-gateway-w27
if docker inspect "$name" >/dev/null 2>&1; then
 [[ $(docker inspect -f '{{index .Config.Labels "flashnext.vision.gateway"}}' "$name") == vision-prod-w27-20261117a ]] || { echo 'GATEWAY_STOP HOLD foreign container'; exit 2; }
 docker stop --time 30 "$name" >/dev/null
 [[ $(docker inspect -f '{{.State.Running}}' "$name") == false ]] || exit 2
fi
echo 'GATEWAY_STOP OFF owned-container retained'
