#!/usr/bin/env python3
# Add VISION=1 support to run-flash-prod.sh (idempotent): drop --language-model-only, append $VISION_ARGS, mount the
# checksummed $OVERLAY/vision/model.py (first-PP-stage tower guard). With VISION unset the launch is unchanged.
import sys
p = sys.argv[1]; s = open(p).read()
if 'VISION' in s:
    print('already has VISION support'); sys.exit(0)
def sub(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
sub('''spec=()
''', '''# VISION=1: image input (no --language-model-only), $VISION_ARGS appended, first-stage-only tower if the overlay has
# vision/model.py (checksummed in vision/SHA256SUMS).
vision=(); lmo=(--language-model-only)
if [[ ${VISION:-0} == 1 ]]; then
  lmo=()
  if [[ -f $OVERLAY/vision/model.py ]]; then
    (cd "$OVERLAY/vision" && sha256sum --quiet -c SHA256SUMS) || { echo "vision overlay checksum mismatch" >&2; exit 2; }
    vision=(-v "$OVERLAY/vision/model.py:$D/model.py:ro")
  fi
fi
spec=()
''')
sub('''  --dtype bfloat16 --language-model-only --engram-config''', '''  --dtype bfloat16 "${lmo[@]}" --engram-config''')
sub('''  "${runner[@]}" \\
''', '''  "${runner[@]}" \\
  "${vision[@]}" \\
''')
sub('''  "${spec[@]}" ${EXTRA_ARGS:-}''', '''  "${spec[@]}" ${EXTRA_ARGS:-} ${VISION_ARGS:-}''')
open(p, 'w').write(s); print('patched', p)
