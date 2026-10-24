#!/bin/bash
# Run manifest for a Flash-Next experiment (run inside CT <ct-id>): exactly what the engine runs, not what a config says.
# Image id, model revision, mounted overlay files + sha256, container env (secrets masked), engine args, PP partition,
# GPU order/clocks/power/memory, driver, settings file. usage: manifest.sh OUT.txt
set -u
OUT=${1:?out}
C=$(docker ps -q --filter name=qwen-flash-prod | head -1)
{
  echo "# manifest $(date -u +%FT%TZ) host $(hostname)"
  [[ -n $C ]] || { echo "NO ENGINE CONTAINER"; exit 0; }
  echo "## container"; docker inspect "$C" --format 'image={{.Image}} started={{.State.StartedAt}} name={{.Name}}'
  echo "## model"; cat /data/vllm/flash-next-tp4/models/intel-w4a16/config.json 2>/dev/null | python3 -c \
    'import json,sys; d=json.load(sys.stdin); print("architectures", d.get("architectures"), "quant", (d.get("quantization_config") or {}).get("quant_method"))'
  ls -la --time-style=+%FT%T /data/vllm/flash-next-tp4/models/intel-w4a16/*.safetensors 2>/dev/null | awk '{print $6, $5, $7}' | md5sum | sed 's/^/weights listing md5 /'
  echo "## mounted overlay files"
  docker inspect "$C" --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' | grep -E 'overlay|core' |
    while read -r src _ dst; do echo "$(sha256sum "$src" | cut -c1-16) $src -> $dst"; done
  echo "## env"; docker inspect "$C" --format '{{range .Config.Env}}{{println .}}{{end}}' |
    grep -E '^(VLLM_|QSA_|FLASH_|CUDA_|NCCL_|PYTORCH_)' | sed -E 's/((KEY|TOKEN|SECRET)[A-Z_]*=).*/\1***/'
  echo "## args"; docker inspect "$C" --format '{{join .Args " "}}'
  echo "## settings"; grep -vE '^\s*(#|$)' /etc/default/vllm-flash | sed -E 's/((KEY|TOKEN|SECRET)[A-Z_]*=).*/\1***/'
  echo "## gpus"; nvidia-smi --query-gpu=index,pci.bus_id,name,pcie.link.width.current,clocks.max.sm,clocks.max.mem,power.limit,memory.used,memory.total --format=csv,noheader
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | sed 's/^/driver /'
} > "$OUT" 2>&1
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
