#!/usr/bin/env python3
"""M3 queue deltas: the next lifecycle queue writes per-phase allocator receipts after its engine log capture.

Declared text deltas over a queue generated from the l-f texts.  The next candidate's generator applies them after
its own swaps, under its own QNAME.  This tool never writes into harness/runs, and every replacement must match
exactly once.  The tool's exit code is logged and never feeds a verdict.
  M1 preflight interfaces: native43_allocator_phase_deltas.py is checked with the other harness tools;
  M2 declaration pins: it is pinned with the other harness tools;
  M3 capture_logs: right after the engine-log step it reads the just-captured current log and the
     engine-watch-<name>.json records.
"""
M1_OLD = 'for _interface in native43_row1_qualification.py '
M1_NEW = 'for _interface in native43_allocator_phase_deltas.py native43_row1_qualification.py '
M2_OLD = 'native43_lifecycle_gate.py native43_lifecycle_grade.py native43_engine_death.py '
M2_NEW = 'native43_lifecycle_gate.py native43_lifecycle_grade.py native43_engine_death.py native43_allocator_phase_deltas.py '
M3_OLD = 'step engine-log "$lrc"\n'
M3_NEW = ('step engine-log "$lrc"\n'
          '# M3 receipt only: per-phase allocator counters and indexer score sizes; never a verdict input.\n'
          'if [[ -n ${H:-} && $lrc == 0 ]]; then\n'
          '  python3 "$H/native43_allocator_phase_deltas.py" --stream "$Q/engine-current.log" --watch-dir "$Q" \\\n'
          '    --out "$Q/allocator-phase-deltas.json" >> "$Q/allocator-phase-deltas.log" 2>&1\n'
          '  echo "allocator-phase-deltas rc=$?" >> "$Q/allocator-phase-deltas.log"\n'
          'fi\n')
DRAFT_DELTAS = (('M1', M1_OLD, M1_NEW), ('M3', M3_OLD, M3_NEW))
DECLARE_DELTAS = (('M2', M2_OLD, M2_NEW),)


def swap(text, old, new, label):
    if text.count(old) != 1:
        raise ValueError('%s: anchor matches %d times' % (label, text.count(old)))
    return text.replace(old, new)


def apply_draft(text):
    for label, old, new in DRAFT_DELTAS:
        text = swap(text, old, new, label)
    return text


def apply_declare(text):
    for label, old, new in DECLARE_DELTAS:
        text = swap(text, old, new, label)
    return text
