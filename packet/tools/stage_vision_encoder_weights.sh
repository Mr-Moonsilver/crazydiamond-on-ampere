#!/usr/bin/env bash
# Owner-run CPU/storage copy only. No GPU, service control, overwrite or deletion.
set -euo pipefail
archive=${1:?usage: stage_vision_encoder_weights.sh vision.tar}
tool=$(dirname "$(readlink -f "$0")")/vision_encoder_bundle.py
[[ -f $archive && -f $tool ]] || { echo 'archive/tool missing' >&2; exit 2; }
archive_sha=$(sha256sum "$archive" | cut -d' ' -f1)
tool_sha=$(sha256sum "$tool" | cut -d' ' -f1)
remote=<service-user>@<encoder-host>
stage=/opt/flashnext/stage/vision-${archive_sha:0:12}-$(date -u +%Y%m%dT%H%M%SZ)
ssh_args=(-i "$HOME/.ssh/<ssh-key>" -o BatchMode=yes -o IdentitiesOnly=yes)
ssh "${ssh_args[@]}" "$remote" "test ! -e '$stage' && mkdir -p '$stage'"
scp "${ssh_args[@]}" -- "$archive" "$remote:$stage/vision.tar"
scp "${ssh_args[@]}" -- "$tool" "$remote:$stage/vision_encoder_bundle.py"
ssh "${ssh_args[@]}" "$remote" "cd '$stage' && printf '%s  %s\n%s  %s\n' '$archive_sha' vision.tar '$tool_sha' vision_encoder_bundle.py | sha256sum -c - && python3 -B vision_encoder_bundle.py unpack --archive vision.tar --parent /opt/flashnext/models"
echo "VISION_COPY PASS archive_sha256=$archive_sha tool_sha256=$tool_sha stage=$stage GPU_NOT_USED"
