#!/bin/bash
# OWNER CT<ct-id> ONLY. Changes the gateway, never launches/stops the engine.
set -euo pipefail
[[ ${1:-} == --owner-approved && $EUID == 0 ]] || { echo 'GATEWAY_SWAP HOLD owner approval/root required'; exit 2; }
PK=/data/flash-next-packets/serve-vision-prod-w28-20261118b
Q=/data/flash-next-packets/gateway-queue-w28-20261118-build-7
[[ $(systemctl is-active vllm-flash.service) == active ]]
[[ $(systemctl is-active vllm-qwen38.service || :) != active ]]
ENGINE_ID=$(systemctl show vllm-flash.service -p InvocationID --value)
ENGINE_PID=$(systemctl show vllm-flash.service -p MainPID --value)
cd "$PK"
printf '%s  PACKET-SHA256SUMS\n' 344cb0fd98c1520b386117bc80de20e8f07eacd5b2589c029e80e80d287b2345 | sha256sum --strict -c -
sha256sum --strict --quiet -c PACKET-SHA256SUMS
cd "$Q";sha256sum --strict --quiet -c DELIVERY-SHA256SUMS;sha256sum --strict --quiet -c GATEWAY-SHA256SUMS
for f in /root/.flashnext-serve-vision-prod-w27-20261117a/{vllm-api,vision-encoder}.env;do
 [[ -f $f && ! -L $f && $(stat -c '%a %u' "$f") == '600 0' ]]
done
umask 077
FOUND=/data/flash-next-control/gateway-queue7-found-$(date -u +%Y%m%dT%H%M%S)-$$
mkdir -m 0700 "$FOUND"
for f in /etc/systemd/system/flashnext-vision-gateway.service /etc/default/flashnext-vision-gateway;do
 [[ ! -L $f ]]
 if [[ -f $f ]];then cp -p "$f" "$FOUND/$(basename "$f")";fi
done
systemctl stop flashnext-vision-gateway.service
install -m 0644 "$Q/flashnext-vision-gateway.service" /etc/systemd/system/flashnext-vision-gateway.service
install -m 0644 "$Q/gateway-defaults.env" /etc/default/flashnext-vision-gateway
systemctl daemon-reload
systemctl start flashnext-vision-gateway.service
UP=0
for _ in {1..60};do
 if [[ $(curl -s --max-time 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health || :) == 200 ]];then UP=1;break;fi
 sleep 1
done
[[ $UP == 1 ]]
for endpoint in health metrics v1/models;do
 [[ $(curl -sS --max-time 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:8000/$endpoint") == 200 ]]
done
[[ $(curl -sS --max-time 5 -H 'Content-Type: application/json' -d '{}' -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/chat/completions) == 401 ]]
[[ $(systemctl show vllm-flash.service -p InvocationID --value) == "$ENGINE_ID" ]]
[[ $(systemctl show vllm-flash.service -p MainPID --value) == "$ENGINE_PID" ]]
[[ $(docker inspect -f '{{.State.Running}} {{.HostConfig.Runtime}} {{.Config.Image}}' flashnext-vision-gateway-w28-queue7) == 'true runc sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b' ]]
systemctl is-active vllm-flash.service flashnext-vision-gateway.service
journalctl -u flashnext-vision-gateway.service --no-pager -n 30 | grep 'VISION_GATEWAY_READY '
echo "GATEWAY_SWAP PASS engine InvocationID/MainPID unchanged; found files retained $FOUND"
