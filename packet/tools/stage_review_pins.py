#!/usr/bin/env python3
"""Refresh the review's source and harness pins for the C0/C1 branch's commits.

Moves only `source_shas` (every pinned engine file, plus the C0 modules) and `harness_source_shas` (tracked harness files),
the two tables the readiness gate compares with the tree. It claims no review: the CPU receipts and the candidate's review key
are written by the package build. Run with harness/ tracked. No GPU, no box action.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'patches/s5_engine/native_candidate_review.json'
NEW = ('stage_wire.py', 'stage_progress.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review = json.loads(REVIEW.read_text())
    names = sorted(set(review['source_shas']) | set(NEW))
    review['source_shas'] = {name: sha(ROOT / 'patches/s5_engine' / name) for name in names}
    tracked = subprocess.check_output(['git', 'ls-files', 'harness'], cwd=ROOT, text=True).splitlines()
    review['harness_source_shas'] = {str(Path(name).relative_to('harness')): sha(ROOT / name)
                                     for name in tracked if (ROOT / name).is_file()}
    REVIEW.write_text(json.dumps(review, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(source=len(review['source_shas']), harness=len(review['harness_source_shas']))))


if __name__ == '__main__':
    main()
