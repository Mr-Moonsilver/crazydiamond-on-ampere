#!/usr/bin/env bash
# OFFLINE UNTIL THE QUEUE OWNER LAUNCHES IT. No service/production control.
# Usage: canary_hub_runner.sh CT_QUEUE_DIR CT_HARNESS_DIR NEW_HUB_OUTPUT_DIR
set -euo pipefail
H=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
(cd "$H" && sha256sum --quiet -c SHA256SUMS)
exec python3 "$H/canary_hub_runner.py" "$@"
