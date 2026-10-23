#!/bin/bash
# Switch vllm-flash to /etc/default/vllm-flash.next with automatic rollback to a copy of the current settings if the new
# configuration is not healthy within 12 minutes. Log: /data/vllm/flash-next-tp4/logs/reconfig-<stamp>.log
set -u
NEXT=/etc/default/vllm-flash.next; CUR=/etc/default/vllm-flash; BAK=/etc/default/vllm-flash.bak-$(date -u +%Y%m%dT%H%M%SZ); cp -p "$CUR" "$BAK"
LOG=/data/vllm/flash-next-tp4/logs/reconfig-$(date -u +%Y%m%dT%H%M%SZ).log
log(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
wait_healthy(){ for i in $(seq 1 72); do curl -fs --max-time 5 http://127.0.0.1:8000/health >/dev/null && return 0; systemctl is-failed -q vllm-flash && return 1; sleep 10; done; return 1; }
[ -f "$NEXT" ] && [ -f "$BAK" ] || { log "missing $NEXT or $BAK"; exit 2; }
log "switching to $NEXT"; cp -p "$NEXT" "$CUR"
systemctl restart vllm-flash
if wait_healthy; then
  log "HEALTHY with new settings"; docker logs qwen-flash-prod 2>&1 | grep -oE "GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|max_num_batched_tokens=[0-9]+" | sort -u | tee -a "$LOG"
  exit 0
fi
log "NOT HEALTHY - rolling back to $BAK"; docker logs --tail 40 qwen-flash-prod >> "$LOG" 2>&1
cp -p "$BAK" "$CUR"; systemctl reset-failed vllm-flash 2>/dev/null; systemctl restart vllm-flash
if wait_healthy; then log "ROLLED BACK and healthy"; exit 1; fi
log "ROLLBACK ALSO NOT HEALTHY"; exit 3
