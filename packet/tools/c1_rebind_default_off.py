#!/usr/bin/env python3
"""Default-off proof for FLASH_S5_STAGE_OVERLAP_DEFER_BIND and FLASH_S5_STAGE_OVERLAP_OBSERVE (C1 rebind).

Against C1's delivered commit (``--c1``, default cf120738):

1. every file under ``patches/s5_engine`` except ``stage_overlap.py`` is byte-identical;
2. ``stage_overlap.py``, after the declared projections below (exact text, exact count, in
   order), has the same AST as C1's file.

Each projection removes one added statement, block or definition.  Every removed statement is
either a definition or import that nothing on the flags-off path calls, or code that runs only
when ``self.defer_bind`` is True, ``self.overlap_observer`` is not None, or a frame carries
``deferred_bind`` (set only under ``self.defer_bind``); the overrides it removes
(``_segment_gate``, ``_apply``, ``_ingest_report``) delegate to ``super()`` with the same
arguments when those are off.  With both switches unset (``defer_bind`` False, observer None)
every statement the engine executes is C1's.  Prints a JSON receipt; exit 0 only when both hold.

usage: c1_rebind_default_off.py [--c1 REV] [--out FILE]
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = 'patches/s5_engine'
TARGET = 'stage_overlap.py'

PROJECTIONS = [
    ("from dataclasses import replace\nimport json\nimport logging\nimport os\nimport time\n",
     "from dataclasses import replace\nimport logging\n"),
    ("from wire_v1 import CancelRequest, PollRequest, ReleaseAck, SegmentKind\n",
     "from wire_v1 import CancelRequest, PollRequest, ReleaseAck\n"),
    # module-level additions: the two switches, the observer class
    ('SPAN', "# C1 rebind (review 2 Nov, section 2, option (a))", "def overlap_refusal(manager,", ''),
    ("        self.defer_bind = stage_overlap_defer_bind_enabled(os.environ)\n"
     "        self.defer_counters = dict(deferred_opens=0, deferred_binds=0)\n"
     "        self.overlap_observer = OverlapObserver() if stage_overlap_observe_enabled(os.environ) else None\n", ''),
    ("        if self.overlap_observer is not None:\n"
     "            frame.stats['obs_successor_of'] = older.chunk['tx'].txn_seq\n", ''),
    ("        if self.defer_bind:\n"
     "            for frame in tuple(self._frames):\n"
     "                self._deferred_bind_due(frame)\n", ''),
    ("            if self.overlap_observer is not None and self._scoped_stage is not None:\n"
     "                self.overlap_observer.note_submit(sequence)\n", ''),
    ('SPAN', "    # -- overlap observation (FLASH_S5_STAGE_OVERLAP_OBSERVE)", "    # -- open ---", ''),
    ("        if self.defer_bind:\n"
     "            # n+1's own compute needs no PP binding; the bind waits in front of its PP_SEND.\n"
     "            self.phase = 'segments'\n"
     "            frame.deferred_bind = True\n"
     "            self.defer_counters['deferred_opens'] += 1\n", ''),
    ("        if self.overlap_observer is not None:\n"
     "            frame.stats['obs_early'] = True\n"
     "            frame.stats['obs_deferred'] = bool(getattr(frame, 'deferred_bind', False))\n", ''),
    ('SPAN', "    # -- deferred PP binding (FLASH_S5_STAGE_OVERLAP_DEFER_BIND)", "    def _chunk_opened(self, frame):", ''),
    ('SPAN', "        observer = self.overlap_observer\n        if observer is not None:\n            info = dict(",
     "        if self.progress is None or not self.overlap_selected:", ''),
    ("        if self.defer_bind:\n"
     "            result['overlap']['defer_bind'] = dict(counters=dict(self.defer_counters))\n"
     "        if self.overlap_observer is not None:\n"
     "            result['overlap']['observe'] = dict(counters=dict(self.overlap_observer.counters))\n", ''),
    ("__all__ = ['DEFER_BIND_ENV', 'OBSERVE_ENV', 'OBSERVE_LINE', 'OBSERVE_SCHEMA', 'OverlapObserver', "
     "'OverlapSegmentDriver', 'REFUSALS', 'STAGE_OVERLAP_ENV', 'StagedAdmissionController',\n"
     "           'overlap_refusal', 'provisional_fields', 'stage_overlap_defer_bind_enabled', 'stage_overlap_enabled',\n"
     "           'stage_overlap_observe_enabled']",
     "__all__ = ['OverlapSegmentDriver', 'REFUSALS', 'STAGE_OVERLAP_ENV', 'StagedAdmissionController',\n"
     "           'overlap_refusal', 'provisional_fields', 'stage_overlap_enabled']"),
]


def project(text):
    applied = []
    for item in PROJECTIONS:
        if item[0] == 'SPAN':
            _tag, start, end, new = item
            if text.count(start) != 1 or text.count(end) != 1:
                raise SystemExit('span markers not unique: %r .. %r' % (start, end))
            a, b = text.index(start), text.index(end)
            if b <= a:
                raise SystemExit('span out of order: %r' % start)
            removed = text[a:b]
            text = text[:a] + new + text[b:]
            applied.append(dict(kind='span', start=start, end=end, removed_lines=removed.count('\n')))
        else:
            old, new = item
            if text.count(old) != 1:
                raise SystemExit('projection does not match exactly once: %r' % old[:80])
            text = text.replace(old, new)
            applied.append(dict(kind='text', removed_lines=old.count('\n') - new.count('\n')))
    return text, applied


def git_show(rev, path):
    return subprocess.run(['git', '-C', str(ROOT), 'show', '%s:%s' % (rev, path)],
                          check=True, capture_output=True).stdout


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--c1', default='cf120738')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    listed = subprocess.run(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', args.c1, ENGINE + '/'],
                            check=True, capture_output=True, text=True).stdout.split()
    here = sorted(str(p.relative_to(ROOT)) for p in (ROOT / ENGINE).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts)
    differing = []
    for path in sorted(set(listed) | set(here)):
        if path == ENGINE + '/' + TARGET:
            continue
        mine = (ROOT / path).read_bytes() if (ROOT / path).exists() else None
        theirs = git_show(args.c1, path) if path in listed else None
        if mine != theirs:
            differing.append(path)
    current = (ROOT / ENGINE / TARGET).read_text()
    original = git_show(args.c1, ENGINE + '/' + TARGET).decode()
    projected, applied = project(current)
    same_ast = ast.dump(ast.parse(projected)) == ast.dump(ast.parse(original))
    receipt = dict(schema='c1-rebind-default-off/1', c1=args.c1,
                   engine_files_compared=len(set(listed) | set(here)) - 1,
                   other_engine_files_differing=differing,
                   target=ENGINE + '/' + TARGET,
                   target_sha256=hashlib.sha256(current.encode()).hexdigest(),
                   projections=applied, projected_ast_equals_c1=same_ast,
                   passed=not differing and same_ast)
    text = json.dumps(receipt, indent=1)
    if args.out:
        with open(args.out, 'x') as out:
            out.write(text + '\n')
    print(text)
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
