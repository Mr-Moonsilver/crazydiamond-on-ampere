#!/usr/bin/env bash
# Write the record of a passed (or failed) dry run of one exact head.
# Usage: dryrun-record.sh <worktree> <rc> <log-path> <out.json>
# Record: {"schema":"dryrun-record/1","head","dirty","rc","log","log_sha256","utc"}
#   head = git rev-parse HEAD of the worktree; dirty = number of `git status --short` lines;
#   log = absolute path of the dry-run log, log_sha256 = its sha256 at record time.
# Refuses (exit 2) to overwrite an existing out.json. Pair: tools/require-dryrun.sh.
set -uo pipefail
[[ $# -eq 4 ]] || { echo "usage: dryrun-record.sh <worktree> <rc> <log-path> <out.json>" >&2; exit 2; }
WT=$1; RC=$2; LOG=$3; OUT=$4
[[ $RC =~ ^[0-9]+$ ]] || { echo "dryrun-record: rc '$RC' is not an integer" >&2; exit 2; }
[[ -f $LOG ]] || { echo "dryrun-record: log '$LOG' not found" >&2; exit 2; }
[[ ! -e $OUT && ! -L $OUT ]] || { echo "dryrun-record: refusing to overwrite $OUT" >&2; exit 2; }
HEAD=$(git -C "$WT" rev-parse HEAD) || { echo "dryrun-record: '$WT' is not a git worktree" >&2; exit 2; }
DIRTY=$(git -C "$WT" status --short | wc -l) || exit 2
SHA=$(sha256sum "$LOG" | cut -d' ' -f1)
exec python3 -I - "$HEAD" "$DIRTY" "$RC" "$(readlink -f "$LOG")" "$SHA" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$OUT" <<'PY'
import json, sys
head, dirty, rc, log, sha, utc, out = sys.argv[1:]
rec = {"schema": "dryrun-record/1", "head": head, "dirty": int(dirty), "rc": int(rc),
       "log": log, "log_sha256": sha, "utc": utc}
try:
    with open(out, "x") as f:  # exclusive create: never overwrites
        json.dump(rec, f, indent=1)
        f.write("\n")
except FileExistsError:
    print(f"dryrun-record: refusing to overwrite {out}", file=sys.stderr)
    sys.exit(2)
print(f"dryrun-record: {out} head {head[:12]} dirty {dirty} rc {rc}")
PY
