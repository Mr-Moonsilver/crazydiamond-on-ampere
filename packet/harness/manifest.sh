#!/bin/bash
# Run manifest for a Flash-Next experiment (run inside CT <ct-id>): exactly what the engine runs, not what a config says.
# - container id, start time, image id; engine args; settings file (redacted)
# - model identity: sha256 of config/index/processor json and FULL-CONTENT sha256 of every *.safetensors shard,
#   computed once into $W/models/intel-w4a16.sha256 and reused only while every shard's size+mtime are unchanged
#   (the manifest states whether it was recomputed)
# - mounted overlay files with their sha256; env by allowlist (VLLM_/QSA_/FLASH_/CUDA_/NCCL_/PYTORCH_), secrets masked
# - observed GPU state now: clocks.sm/clocks.mem (current, not max), temperature, power draw/limit, memory, link width
# - optional HARNESS_DIR: sha256 of the harness files used
# usage: manifest.sh OUT.txt
set -u
OUT=${1:?out}; W=/data/vllm/flash-next-tp4; M=$W/models/intel-w4a16; HC=$W/models/intel-w4a16.sha256
C=$(docker ps -q --filter name=qwen-flash-prod | head -1)
mask() { sed -E 's/((KEY|TOKEN|SECRET|PASSWORD)[A-Za-z_]*=).*/\1***/I'; }
{
  echo "# manifest $(date -u +%FT%TZ) host $(hostname)"
  if [[ -z $C ]]; then echo "NO ENGINE CONTAINER"; else
    echo "## container"; docker inspect "$C" --format 'id={{.Id}} started={{.State.StartedAt}} image={{.Image}}'
    echo "## args"; docker inspect "$C" --format '{{join .Args " "}}' | mask
    echo "## mounted overlay files"
    docker inspect "$C" --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' | grep -E 'overlay' |
      while read -r src _ dst; do echo "$(sha256sum "$src" | cut -c1-64) $src -> $dst"; done
    echo "## env (allowlist)"; docker inspect "$C" --format '{{range .Config.Env}}{{println .}}{{end}}' |
      grep -E '^(VLLM_|QSA_|FLASH_|CUDA_|NCCL_|PYTORCH_)' | mask
  fi
  echo "## settings"; grep -vE '^\s*(#|$)' /etc/default/vllm-flash | mask
  echo "## model identity"
  for f in "$M"/*.json; do echo "$(sha256sum "$f" | cut -c1-64) $(basename "$f")"; done
  stamp=$(stat -c '%n %s %Y' "$M"/*.safetensors | sha256sum | cut -c1-64)
  if [[ -f $HC && $(head -1 "$HC") == "stat $stamp" ]]; then echo "shards: cached full-content hashes (size+mtime unchanged)"
  else echo "shards: full-content hashes recomputed now"
    { echo "stat $stamp"; sha256sum "$M"/*.safetensors | awk '{print $1, $2}' | sed "s|$M/||"; } > "$HC.tmp" && mv "$HC.tmp" "$HC"; fi
  tail -n +2 "$HC" | sha256sum | cut -c1-64 | sed 's/^/weights digest /'
  echo "## gpus (observed now)"
  nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.width.current,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,memory.used,memory.total --format=csv,noheader
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | sed 's/^/driver /'
  if [[ -n ${HARNESS_DIR:-} ]]; then echo "## harness"; (cd "$HARNESS_DIR" && sha256sum *.py *.sh 2>/dev/null); fi
} > "$OUT" 2>&1
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
