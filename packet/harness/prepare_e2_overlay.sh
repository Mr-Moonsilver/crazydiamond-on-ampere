#!/bin/bash
# CPU-only E2 packaging entrypoint, never an engine queue. The slot owner runs
# its separately reviewed q12-style gate after reviewing the generated payload.
set -euo pipefail
umask 077
: "${H:?checked harness bundle}"
: "${BASE:?approved overlay-v16 baseline}"
: "${OUT:?new output directory}"
: "${BASE_MANIFEST_SHA:?independently approved full baseline manifest SHA256}"
: "${CORE_MANIFEST_SHA:?independently approved full core manifest SHA256}"
[[ ! -e $OUT ]] || { echo 'new output directory required' >&2; exit 1; }
(cd "$H" && sha256sum --quiet -c SHA256SUMS)
python3 "$H/e2_builder/make_e2_arena.py" "$BASE" "$OUT" \
  --reference "$H/e2_reference" --launcher "$H/run-flash-prod.sh.q9" \
  --base-manifest-sha256 "$BASE_MANIFEST_SHA" --core-manifest-sha256 "$CORE_MANIFEST_SHA"
(cd "$OUT" && sha256sum --quiet -c SHA256SUMS)
(cd "$OUT/core" && sha256sum --quiet -c SHA256SUMS)
bash -n "$OUT/run-flash-e2-test.sh"
echo 'E2 package created and checksummed; no engine action performed; owner review/GPU gates required'
