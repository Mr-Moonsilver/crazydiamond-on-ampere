#!/usr/bin/env bash
# Launch guard: exit 0 only if the record proves a passing dry run of this exact head.
# Usage: require-dryrun.sh <worktree> <record.json>      (record from tools/dryrun-record.sh)
# Accepts only if: the record exists with schema dryrun-record/1 and rc 0, its head equals the
# worktree's current HEAD, dirty was 0, the worktree has no tracked changes now (untracked files are
# ignored, so a record file inside the worktree is fine), and the log still has the recorded sha256.
# Otherwise exit 3 with one line on stderr naming the reason.
set -uo pipefail
[[ $# -eq 2 ]] || { echo "require-dryrun: REFUSED usage: require-dryrun.sh <worktree> <record.json>" >&2; exit 3; }
exec python3 -I - "$1" "$2" <<'PY'
import hashlib, json, subprocess, sys

wt, rec_path = sys.argv[1:]

def refuse(why):
    print(f"require-dryrun: REFUSED {why}", file=sys.stderr)
    sys.exit(3)

try:
    with open(rec_path) as f:
        rec = json.load(f)
except FileNotFoundError:
    refuse(f"record {rec_path} does not exist")
except (OSError, ValueError) as e:
    refuse(f"record {rec_path} unreadable ({type(e).__name__})")
if not isinstance(rec, dict) or rec.get("schema") != "dryrun-record/1":
    refuse("record schema is not dryrun-record/1")
if type(rec.get("rc")) is not int or rec["rc"] != 0:
    refuse(f"dry run rc is {rec.get('rc')!r}, not 0")
if type(rec.get("dirty")) is not int or rec["dirty"] != 0:
    refuse(f"dry run worktree was dirty ({rec.get('dirty')!r} status lines)")
try:
    head = subprocess.run(["git", "-C", wt, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
except (subprocess.CalledProcessError, OSError):
    refuse(f"cannot read HEAD of worktree {wt}")
if rec.get("head") != head:
    refuse(f"head mismatch: dry run was {str(rec.get('head'))[:12]}, worktree is at {head[:12]}")
try:
    changed = subprocess.run(["git", "-C", wt, "status", "--short", "--untracked-files=no"], check=True,
                             capture_output=True, text=True).stdout.splitlines()
except (subprocess.CalledProcessError, OSError):
    refuse(f"cannot read git status of worktree {wt}")
if changed:
    refuse(f"worktree has tracked changes now ({len(changed)} status lines, first: {changed[0].strip()})")
log = rec.get("log")
if not isinstance(log, str) or not isinstance(rec.get("log_sha256"), str):
    refuse("record has no log path / log_sha256")
try:
    h = hashlib.sha256()
    with open(log, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
except OSError:
    refuse(f"dry-run log {log} is missing or unreadable")
if h.hexdigest() != rec["log_sha256"]:
    refuse(f"dry-run log {log} changed since the record (sha256 mismatch)")
print(f"require-dryrun: OK head {head[:12]} rc 0 clean log sha256 {h.hexdigest()[:12]}")
PY
