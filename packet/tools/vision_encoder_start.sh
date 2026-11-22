#!/usr/bin/env bash
# Run on VM<vm-id> only, after infra clearance; foreground owner, no auto-restart.
set -euo pipefail
S=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if (( $# < 2 || $# > 3 )); then
    echo 'usage: vision_encoder_start.sh RUN_ID CLEARANCE_JSON [--service]' >&2
    exit 2
fi
RUN_ID=$1
CLEARANCE=$2
if [[ ${3:-} != '' && ${3:-} != '--service' ]]; then exit 2; fi
(cd "$S" && sha256sum --strict -c PACKET-SHA256SUMS)
exec python3 -B "$S/harness/vision_encoder_job.py" \
    --model /opt/flashnext/models/flash-next-vision-8b333e23 \
    --run-dir "/opt/flashnext/runs/vision-$RUN_ID" --run-id "$RUN_ID" \
    --clearance "$CLEARANCE" --key-file /opt/flashnext/keys/vision-encoder.env \
    --expected-asr "$S/asr-expected.txt" \
    --asr-url 'http://<encoder-host>:9030/asr?language=en&output_format=json' \
    --asr-max-seconds 10 --speech-health 'http://<encoder-host>:9030/health' \
    --fixture '<hub-dir>/scratch/infra/asr-fixture/asr10.wav' \
    --images "$S/harness/vision_assets/red.png" "$S/harness/vision_assets/table.png" "$S/harness/vision_assets/diagram.png" "$S/harness/vision_assets/screenshot.png" "$S/harness/vision_assets/red1024.png" ${3:+"$3"}
