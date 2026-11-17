#!/usr/bin/env python3
"""Native43 le-fix queue deltas (a)+(c): a dead engine fails the phase within seconds and the gate names it.

l-e (q18a-native43-lifecycle run 1, 1 Nov 2026) hung its row1 client for 25 min on a dead engine.  Its
engine-errors gate also PASSed over an unplanned executor teardown.  These declared text deltas apply to a
queue generated from the native39 lifecycle texts (the l-e draft/declare are the reference input).  The
next lifecycle candidate's generator applies them after its own swaps, under its own new QNAME.  This tool
never writes into harness/runs.  Every replacement must match exactly once.
  D1 preflight interfaces: native43_engine_death.py is checked with the other harness tools;
  D2 declaration pins: native43_engine_death.py is pinned with the other harness tools;
  D3 host_crash_detected (c): after the two existing checks, `native43_engine_death.py scan` over the retained
     host stream and the queue's stop intent.  A DEAD scan (an executor/worker teardown or a broadcast-block
     stall before the first orderly shutdown trigger, or a hard death marker) sets ENGINE_CRASHED=1 with
     ENGINE_CRASH_SOURCE=engine-death-scan.  capture_logs calls host_crash_detected before stop_engine, so the
     engine-errors step FAILs on it ('retained host stream proves engine crash');
  D4 engine_watched (a): a new function that runs a client under `native43_engine_death.py watch`.  The watch
     polls every 5 s: the stream scan, any GPU below 1 GiB for 2 polls, and the container not running for 2
     polls.  On DEAD it stops the client, sets ENGINE_CRASHED=1 with ENGINE_CRASH_SOURCE=engine-death-watch
     and returns 1.  Otherwise the client's own exit code passes through unchanged;
  D5-D7 the three long engine-facing clients run under engine_watched, with argv and redirections unchanged:
     the qualification phase client (row1), the lifecycle client, and the typed FULL hygiene return.
The bounded instruments checks (30 s / 10 s control timeouts) are left as they are.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'harness/runs'
REFERENCE = 'q18a_native43_lifecycle'

D1_OLD = 'for _interface in native43_lifecycle_gate.py '
D1_NEW = 'for _interface in native43_engine_death.py native43_lifecycle_gate.py '
D2_OLD = '      native39_lifecycle_gate.py native35_lifetime_grade.py native43_lifecycle_gate.py native43_lifecycle_grade.py \\\n'
D2_NEW = ('      native39_lifecycle_gate.py native35_lifetime_grade.py native43_lifecycle_gate.py native43_lifecycle_grade.py '
          'native43_engine_death.py \\\n')
D3_OLD = ('  if grep -Eqi \'DistBackendError|illegal memory access|Worker proc .* died unexpectedly\' \\\n'
          '      "$Q/engine-stream.log" "$Q/engine-stream.err" 2>/dev/null; then\n'
          '    ENGINE_CRASHED=1; ENGINE_CRASH_SOURCE=host-engine-stream\n'
          '    return 0\n'
          '  fi\n'
          '  return 1\n'
          '}\n')
D3_NEW = ('  if grep -Eqi \'DistBackendError|illegal memory access|Worker proc .* died unexpectedly\' \\\n'
          '      "$Q/engine-stream.log" "$Q/engine-stream.err" 2>/dev/null; then\n'
          '    ENGINE_CRASHED=1; ENGINE_CRASH_SOURCE=host-engine-stream\n'
          '    return 0\n'
          '  fi\n'
          '  # Native43 le-fix (c): an executor/worker teardown or a broadcast-block stall before the first orderly\n'
          '  # shutdown trigger (or the queue\'s stop intent), or a hard death marker, is an engine crash (l-e 11:03:08Z).\n'
          '  python3 "${H:-}/native43_engine_death.py" scan --log "$Q/engine-stream.log" --log "$Q/engine-stream.err" \\\n'
          '    --stop-intent "$Q/host-stop-intent.json" --out "$Q/engine-death-scan.json" >> "$Q/engine-death-scan.log" 2>&1\n'
          '  if [[ $? == 1 ]]; then\n'
          '    ENGINE_CRASHED=1; ENGINE_CRASH_SOURCE=engine-death-scan\n'
          '    return 0\n'
          '  fi\n'
          '  return 1\n'
          '}\n'
          '# Native43 le-fix (a): a dead engine fails the running client phase within seconds instead of hanging it\n'
          '# (l-e: 25 min).  The watch polls the stream, the GPUs and the container every 5 s; on DEAD it stops the\n'
          '# client and returns 1 (record engine-watch-<name>.json); otherwise the client\'s own code passes through.\n'
          'engine_watched() {\n'
          '  local name=$1 rc\n'
          '  shift\n'
          '  python3 "$H/native43_engine_death.py" watch --record "$Q/engine-watch-$name.json" \\\n'
          '    --log "$Q/engine-stream.log" --log "$Q/engine-stream.err" --stop-intent "$Q/host-stop-intent.json" \\\n'
          '    --container "$CONTAINER" -- "$@"\n'
          '  rc=$?\n'
          '  if [[ $rc == 1 ]] && python3 -c \'import json,sys; raise SystemExit(json.load(open(sys.argv[1])).get("verdict") != "DEAD")\' \\\n'
          '      "$Q/engine-watch-$name.json" 2>/dev/null; then\n'
          '    ENGINE_CRASHED=1; ENGINE_CRASH_SOURCE=engine-death-watch\n'
          '    echo "ENGINE_DEATH_WATCH $name: engine dead, client stopped (engine-watch-$name.json)" >&2\n'
          '  fi\n'
          '  return "$rc"\n'
          '}\n')
D5_OLD = '  python3 "$H/$cli" --phase "$name" --url "$PROBE_URL" \\\n'
D5_NEW = '  engine_watched "qual-$name" python3 "$H/$cli" --phase "$name" --url "$PROBE_URL" \\\n'
D6_OLD = '  python3 "$H/e3_q18a_lifecycle.py" --url "$PROBE_URL" --model "$MODEL" \\\n'
D6_NEW = '  engine_watched lifecycle python3 "$H/e3_q18a_lifecycle.py" --url "$PROBE_URL" --model "$MODEL" \\\n'
D7_OLD = '  python3 - "$PROBE_URL" "$RUN_ID" "$EPOCH" "$Q/hygiene-full.json" <<\'PY_HYGIENE\'\n'
D7_NEW = '  engine_watched hygiene-full python3 - "$PROBE_URL" "$RUN_ID" "$EPOCH" "$Q/hygiene-full.json" <<\'PY_HYGIENE\'\n'

DRAFT_DELTAS = (('D1', D1_OLD, D1_NEW), ('D3+D4', D3_OLD, D3_NEW), ('D5', D5_OLD, D5_NEW),
                ('D6', D6_OLD, D6_NEW), ('D7', D7_OLD, D7_NEW))
DECLARE_DELTAS = (('D2', D2_OLD, D2_NEW),)


def swap(text, old, new, label):
    assert text.count(old) == 1, (label, text.count(old))
    assert 'native43_engine_death' not in old
    return text.replace(old, new)


def apply_draft(text):
    assert 'native43_engine_death' not in text and 'engine_watched' not in text, 'deltas already applied'
    for label, old, new in DRAFT_DELTAS:
        text = swap(text, old, new, label)
    assert text.count('engine_watched ') == 3 and text.count('engine_watched() {') == 1
    return text


def apply_declare(text):
    assert 'native43_engine_death' not in text, 'deltas already applied'
    for label, old, new in DECLARE_DELTAS:
        text = swap(text, old, new, label)
    return text


def reference_texts():
    return ((RUNS / (REFERENCE + '.draft.sh')).read_text(), (RUNS / (REFERENCE + '_declare.sh')).read_text())


if __name__ == '__main__':
    # --preview DIR: the deltas over the l-e reference texts, written to a new directory outside harness/runs.
    if len(sys.argv) != 3 or sys.argv[1] != '--preview':
        raise SystemExit('usage: native43_le_fix_queue_deltas.py --preview NEW_DIR')
    out = Path(sys.argv[2])
    out.mkdir()
    draft, declare = reference_texts()
    (out / 'le-fix-preview.draft.sh').write_text(apply_draft(draft))
    (out / 'le-fix-preview_declare.sh').write_text(apply_declare(declare))
    print('native43 le-fix deltas applied to the l-e reference texts -> ' + str(out))
