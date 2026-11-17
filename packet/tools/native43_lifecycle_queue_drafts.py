#!/usr/bin/env python3
"""Generate the q18a-native43-lifecycle queue from the box-run q18a-native39-lifecycle texts (short scope).

Base: harness/runs/q18a_native39_lifecycle{.draft,_declare,_settings}.sh as they ran on the box
(held-a run 1, bundle b90f1e292956, fn-main results/box/q18a-native39-lifecycle-run-1), on the
native43 dense engine (the frozen n43-f overlay).  Kept byte for byte: preflight, snapshot,
launcher/defaults install, restart and bind receipt, startup grade, bind epoch, ledger, workload
plan, W2 length guard, host headroom, row1 (the lane proof), the lifetime gates, both instruments
receipts, the native lifetime (e3_q18a_lifecycle: cancel/drain/reuse), the typed FULL hygiene
return and the whole cleanup/restore machinery.  Declared deltas only:
  1. engine settings: the native43 candidate policy FLASH_S5_NATIVE_QSA_POLICY=dense replaces the
     native39 lifecycle's FLASH_POOL_STATS=200 telemetry delta (needed only by the pressure-lifetime
     oracle, not run here).  The resulting EXTRA_ENV equals Codex's q18a_native43_dense_settings.sh
     EXTRA_ENV without its two diagnostic entries FLASH_S5_NATIVE_QSA_TRACE=1 and
     FLASH_S5_NATIVE_QSA_STOP_AFTER=101200 (asserted below): absent, the engine defaults are trace 0
     and no stop frontier (qsa_operation.native_qsa_settings, engine_driver STOP_AFTER '0'), so the
     reuse prefill runs to completion.  Asserted in the preflight and the declaration's PY_DECL;
     the settings comment names the reason;
  2. short scope (lead, 1 Nov 2026): the pool-telemetry record and, after the typed FULL hygiene
     return, the leaktest, pressure2/partial-tail leak, pressure completion and pressure lifetime
     steps are not run (their verdicts were for the native39 engine and do not carry over; this
     queue answers only whether cancel/drain/reuse completes on native43);
  3. on_exit: the native35 lifetime grade lines become the native43 lifecycle grade
     (native43_lifecycle_grade: native35_lifetime_grade's lifecycle/hygiene/crash checks, leak and
     pressure NOT_RUN);
  4. window: native43_lifecycle_gate (the derived short budget), outer timeout 5000 s with the
     unchanged 900 s kill margin;
  5. interfaces/pins: native43_lifecycle_gate.py (and its parent native39_lifecycle_gate.py) and
     native43_lifecycle_grade.py checked at preflight and pinned at declaration;
  plus names and the title/header lines.  Every replacement must match exactly once.
"""
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'harness/runs'
BASE, LIFECYCLE, DENSE = 'q18a_native39_lifecycle', 'q18a_native43_lifecycle', 'q18a_native43_dense'

BASE_TITLE = '# q18a-native39-lifecycle: fresh native lifecycle/leak queue; unrun, no GO from this file.\n'
TITLE = '# q18a-native39-lifecycle: fresh native lifecycle queue (short scope); unrun, no GO from this file.\n'  # names() renames
BASE_HEADER_PREFIX = '# Native39 lifecycle: the trunk queue and engine with the replay observer and engine offers off, pool telemetry on: '
HEADER = ('# Native43 lifecycle (short scope): the box-run native39 lifecycle queue on the native43 dense engine '
          '(FLASH_S5_NATIVE_QSA_POLICY=dense), observer and offers off: native28 startup -> bind receipt -> W2 length guard '
          '-> row1 (lane proof) -> lifetime gates -> instruments -> native cancel/drain/reuse -> typed FULL -> instruments\n')
BASE_TIMEOUT = '  exec timeout --signal=TERM --kill-after=900s 9600s env Q18A_OVERLAP_BOUNDED=1 bash "$0" "$@"\n'
TIMEOUT = '  exec timeout --signal=TERM --kill-after=900s 5000s env Q18A_OVERLAP_BOUNDED=1 bash "$0" "$@"\n'
BASE_SLOT_NOTE = '# entry unless its 9600 s timeout plus 900 s kill margin fits that window (native39_lifecycle_gate derivation).\n'
SLOT_NOTE = '# entry unless its 5000 s timeout plus 900 s kill margin fits that window (native43_lifecycle_gate derivation).\n'
BASE_SLOT = 'from native39_lifecycle_gate import lifecycle_window_ok as window_ok\n'
SLOT = 'from native43_lifecycle_gate import lifecycle_window_ok as window_ok\n'
BASE_INTERFACES = 'for _interface in native39_lifecycle_gate.py native35_lifetime_grade.py '
INTERFACES = ('for _interface in native43_lifecycle_gate.py native39_lifecycle_gate.py native43_lifecycle_grade.py '
              'native35_lifetime_grade.py ')
BASE_GRADE = ('  python3 "$H/native35_lifetime_grade.py" --queue "$Q" --out "$Q/native35-lifetime-grade.json" > "$Q/native35-lifetime-grade.log" 2>&1\n'
              "  record native35-lifetime \"$?\" 'native cancel/drain/reuse plus unchanged hygiene/leak oracles; no performance claim'\n")
GRADE = ('  python3 "$H/native43_lifecycle_grade.py" --queue "$Q" --out "$Q/native43-lifecycle-grade.json" > "$Q/native43-lifecycle-grade.log" 2>&1\n'
         "  record native43-lifecycle \"$?\" 'native cancel/drain/reuse and the typed FULL return on the native43 dense engine; "
         "leak and pressure oracles not run (short scope); no performance claim'\n")
BASE_PREFLIGHT = ("# Native39 lifecycle telemetry delta: the pressure-lifetime oracle's pool counters (lead, 1 Nov 2026).\n"
                  "assert env.get('FLASH_POOL_STATS') == '200'\n")
PREFLIGHT = ("# Native43 lifecycle: the candidate dense policy; trace and the 101200 stop frontier absent (engine defaults:\n"
             "# trace 0, no stop), so the reuse prefill runs to completion; no pool telemetry (pressure not run).\n"
             "assert env.get('FLASH_S5_NATIVE_QSA_POLICY') == 'dense'\n"
             "assert 'FLASH_S5_NATIVE_QSA_TRACE' not in env and 'FLASH_S5_NATIVE_QSA_STOP_AFTER' not in env\n"
             "assert 'FLASH_POOL_STATS' not in env\n")
BASE_DECL = "assert env.get('FLASH_POOL_STATS')=='200', 'pool telemetry (FLASH_POOL_STATS=200) must be on for the pressure-lifetime oracle'\n"
DECL = ("assert env.get('FLASH_S5_NATIVE_QSA_POLICY')=='dense', 'native43 lifecycle runs the candidate dense policy'\n"
        "assert 'FLASH_S5_NATIVE_QSA_TRACE' not in env and 'FLASH_S5_NATIVE_QSA_STOP_AFTER' not in env, "
        "'native43 lifecycle: no trace and no stop frontier (engine defaults), the reuse prefill must complete'\n"
        "assert 'FLASH_POOL_STATS' not in env, 'native43 lifecycle: no pool telemetry delta (pressure not run)'\n")
BASE_DECL_PINS = '      native39_lifecycle_gate.py native35_lifetime_grade.py \\\n'
DECL_PINS = '      native39_lifecycle_gate.py native35_lifetime_grade.py native43_lifecycle_gate.py native43_lifecycle_grade.py \\\n'
BASE_SETTINGS_NOTE = ('# Native39 lifecycle telemetry delta (lead, 1 Nov 2026): FLASH_POOL_STATS=200 below. The pressure-lifetime\n'
                      '# oracle (pp_release_grade lifetime) needs the FLASH_PP_LAYOUT state-group layout and the FLASH_POOL_STATS\n'
                      '# state-group eviction counters; the S5 core overlay logs both only when FLASH_POOL_STATS > 0\n'
                      '# (patches/s5_engine/telemetry_hooks.py: counters read and logged every 200 scheduler steps, no allocation or\n'
                      '# eviction change). The E2 gate value (q16_settings.sh); no speed claim.\n')
SETTINGS_NOTE = ('# Native43 lifecycle (short scope): FLASH_S5_NATIVE_QSA_POLICY=dense below, the native43 candidate policy;\n'
                 "# EXTRA_ENV is Codex's q18a_native43_dense_settings.sh EXTRA_ENV without the two diagnostic entries\n"
                 '# FLASH_S5_NATIVE_QSA_TRACE=1 and FLASH_S5_NATIVE_QSA_STOP_AFTER=101200 (engine defaults: trace 0, no stop\n'
                 '# frontier), so the reuse prefill runs to completion. No FLASH_POOL_STATS: the pressure-lifetime oracle that\n'
                 '# needed it is not run in the short scope. No speed claim.\n')
BASE_EXTRA_ENV = ' FLASH_S5_ENGINE_OFFERS=0 FLASH_POOL_STATS=200 FLASH_S5_REPLAY_LIB='
EXTRA_ENV = ' FLASH_S5_ENGINE_OFFERS=0 FLASH_S5_NATIVE_QSA_POLICY=dense FLASH_S5_REPLAY_LIB='
DENSE_DIAGNOSTIC = ' FLASH_S5_NATIVE_QSA_TRACE=1 FLASH_S5_NATIVE_QSA_STOP_AFTER=101200'

# Short scope: the pool-telemetry record (from its heredoc to its record line) and the post-hygiene leak/pressure block.
POOL_BEGIN = "python3 - \"$Q/engine.identity-start.json\" \"$Q/engine-stream.log\" \"$Q/engine-stream.err\" > \"$Q/pool-telemetry.log\" 2>&1 <<'PY_POOL_TELEMETRY'\n"
POOL_END = ("record pool-telemetry \"$?\" 'engine env FLASH_POOL_STATS=200 as captured at identity-start; the startup FLASH_PP_LAYOUT "
            "line with a MambaSpec group in the followed engine log'\n")
POOL_NOTE = '# Native43 lifecycle (short scope): no pool-telemetry record (no FLASH_POOL_STATS delta; pressure not run).\n'
LEAK_BEGIN = '  if [[ $hrc == 0 ]]; then\n    URL=$PROBE_URL/v1/chat/completions SEED=7 GAP=20 \\\n'
LEAK_END = ("    record pressure-lifetime \"$?\" 'unchanged state eviction and request-completion coverage; preemption reported'\n"
            '  fi\n')
LEAK_NOTE = ('  # Native43 lifecycle (short scope, lead 1 Nov 2026): the leaktest, pressure2/partial-tail leak, pressure\n'
             '  # completion and pressure lifetime steps are not run; native43_lifecycle_grade reports them NOT_RUN.\n')


def swap(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new)


def cut(text, begin, end, note):
    assert text.count(begin) == 1 and text.count(end) == 1, (begin[:60], end[:60])
    start, stop = text.index(begin), text.index(end) + len(end)
    assert start < stop
    return text[:start] + note + text[stop:]


def swap_header(text):
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith(BASE_HEADER_PREFIX)]
    assert len(hits) == 1, hits
    return ''.join(lines[:hits[0]]) + HEADER + ''.join(lines[hits[0] + 1:])


def names(text):
    assert 'q18a-native43-lifecycle' not in text and 'q18a_native43_lifecycle' not in text
    return text.replace('q18a-native39-lifecycle', 'q18a-native43-lifecycle').replace(BASE, LIFECYCLE)


def extra_env(settings):
    rows = [w for line in settings.splitlines() if '"EXTRA_ENV=' in line for w in shlex.split(line) if w.startswith('EXTRA_ENV=')]
    assert len(rows) == 1, rows
    return rows[0]


def derive_draft():
    text = (RUNS / (BASE + '.draft.sh')).read_text()
    text = swap_header(text)
    for old, new in ((BASE_TITLE, TITLE), (BASE_TIMEOUT, TIMEOUT), (BASE_SLOT_NOTE, SLOT_NOTE), (BASE_SLOT, SLOT),
                     (BASE_INTERFACES, INTERFACES), (BASE_GRADE, GRADE), (BASE_PREFLIGHT, PREFLIGHT)):
        text = swap(text, old, new)
    text = cut(text, POOL_BEGIN, POOL_END, POOL_NOTE)
    text = cut(text, LEAK_BEGIN, LEAK_END, LEAK_NOTE)
    text = names(text)
    # Invocations only: the bare names stay in the inherited preflight interface list.
    for gone in ('"$H/leaktest.py"', '"$H/pressure2.py"', '"$H/leak_partial_tail.py"', '"$H/pressure_wait.py"',
                 '"$H/pp_release_grade.py"',
                 'FLASH_POOL_STATS)', "'FLASH_POOL_STATS') ==", 'record pool-telemetry', 'record cross-session-leak',
                 'record partial-leak', 'record pressure-', 'native35-lifetime-grade.json'):
        assert gone not in text, gone
    for kept in ('e3_q18a_lifecycle.py', 'record lifecycle "$?"', 'record hygiene-full', 'record instruments-lifecycle',
                 'record instruments-end', 'step lifetime-gates', 'qualification_phase row1'):
        assert kept in text, kept
    return text


def derive_settings():
    text = (RUNS / (BASE + '_settings.sh')).read_text()
    for old, new in ((BASE_SETTINGS_NOTE, SETTINGS_NOTE), (BASE_EXTRA_ENV, EXTRA_ENV)):
        text = swap(text, old, new)
    dense = extra_env((RUNS / (DENSE + '_settings.sh')).read_text())
    assert dense.count(DENSE_DIAGNOSTIC) == 1
    assert extra_env(text) == dense.replace(DENSE_DIAGNOSTIC, ''), 'EXTRA_ENV must be the dense arm without trace/stop'
    return names(text)


def derive_declare():
    text = (RUNS / (BASE + '_declare.sh')).read_text()
    for old, new in ((BASE_DECL, DECL), (BASE_DECL_PINS, DECL_PINS)):
        text = swap(text, old, new)
    return names(text)


def generate():
    return {RUNS / (LIFECYCLE + '.draft.sh'): derive_draft(),
            RUNS / (LIFECYCLE + '_settings.sh'): derive_settings(),
            RUNS / (LIFECYCLE + '_declare.sh'): derive_declare()}


if __name__ == '__main__':
    check = '--check' in sys.argv
    for path, text in generate().items():
        if check:
            assert path.read_text() == text, path.name
        else:
            assert not path.exists(), path.name + ' exists; regenerate only under --check'
            path.write_text(text)
            path.chmod(0o664)
    print('native43 lifecycle queue drafts ' + ('verified' if check else 'written'))
