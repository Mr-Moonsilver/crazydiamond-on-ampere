#!/usr/bin/env python3
"""Generate the q18a-native39-lifecycle queue from the q18a-native39-trunk texts and the native35-L block.

Lifecycle = the trunk queue (held-b) on the trunk engine, whose settings already run
the replay observer and the engine offers off (lead, 1 Nov 2026: the native36-wd
witness showed synchronous observer grading inside before_route_change froze every
observed split->FULL return, fn-main results/box/q18a-native36-wd-run-1).  Kept from
the trunk, byte for byte: preflight (with its observer/offers 0/0 assertions),
snapshot, launcher/defaults install, restart with the per-second /health loop and the
bind receipt, effective host, startup grade, bind epoch, ledger, workload plan, the
W2 length guard, host headroom, row1 (the lane proof) and the whole cleanup/restore
machinery; settings and declaration keep the trunk's observer/offers 0/0.  Not run:
rule4, row4, W1, the plain W1 control and the W2 speed rounds.  Declared deltas only:
  1. the trunk text from 'qualification_phase rule4_controls' through the trunk
     summary step is replaced by native35-L's gates and lifecycle block, taken
     from q18a_native35_l.draft.sh (whose block is native28's, test_native35_queue),
     with two changes: the gates read startup and row1 only (RULE4_CONTINUATION=0
     exported, so L's PY_GATES continuation branch stays off), and the lifecycle
     workload is the trunk's plannable W2 manifest (q18a-w2-overlap-native39.json,
     the fresh count the w2-length guard checked; P-c's length is refused by the
     band rule on this engine);
  2. on_exit: the trunk's 'native35-diagnostic NOT_RUN' line becomes L's
     native35 lifetime grade lines;
  3. observer and offers off proven, not only declared: the trunk's instruments check
     (e3_native39_trunk_checks.py instruments: per-rank /proc/<pid>/maps from a paused
     all-rank barrier) before the lifecycle block, which runs only on its PASS as the
     trunk's W1 does, and after the block with --same-as (the same rank processes);
     the settings comment names the lifecycle reason instead of the speed one;
  4. window: native39_lifecycle_gate (derived budget), outer timeout 9600 s with
     the trunk's 900 s kill margin;
  5. interfaces/pins: native39_lifecycle_gate.py and native35_lifetime_grade.py
     checked at preflight and pinned at declaration;
  6. declared telemetry delta against the trunk engine (lead, 1 Nov 2026, option a):
     FLASH_POOL_STATS=200 in EXTRA_ENV, so the S5 core overlay logs the startup
     FLASH_PP_LAYOUT line and the FLASH_POOL_STATS counters the unchanged
     pressure-lifetime oracle (pp_release_grade lifetime) reads; asserted in the
     declaration's PY_DECL and the preflight, and from the engine's own view by a
     pool-telemetry record (the identity-start capture of the container env and the
     startup FLASH_PP_LAYOUT line in the followed engine log) before the gates;
     the settings comment names the reason.  No speed claim;
  plus names, the title line and the header comment.  Every replacement must match exactly once.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'harness/runs'
TRUNK, L, LIFECYCLE = 'q18a_native39_trunk', 'q18a_native35_l', 'q18a_native39_lifecycle'
MANIFEST_L, MANIFEST = 'workloads/q18a-w2-overlap.json', 'workloads/q18a-w2-overlap-native39.json'

TRUNK_TITLE = '# q18a-native39-trunk: fresh native performance queue; unrun, no GO from this file.\n'
LIFECYCLE_TITLE = '# q18a-native39-trunk: fresh native lifecycle/leak queue; unrun, no GO from this file.\n'  # names() renames it
TRUNK_HEADER_PREFIX = '# Native39 trunk: '
TRUNK_HEADER_NEXT = '# Inherited cleanup machinery below; lifecycle/leak (L) and canary phases are absent.\n'
LIFECYCLE_HEADER = ('# Native39 lifecycle: the trunk queue and engine with the replay observer and engine offers off, pool telemetry on: native28 startup -> bind receipt -> W2 length guard -> row1 (lane proof) -> lifetime gates (startup, row1) -> native lifetime (cancel/drain/reuse) -> hygiene FULL -> cross-session leak -> pressure + partial-leak tail -> pressure completion/lifetime -> owned restore.\n'
                    '# Inherited cleanup machinery below; rule4, row4, W1/W2 performance and canary phases are absent.\n')
TRUNK_TIMEOUT = '  exec timeout --signal=TERM --kill-after=900s 16200s env Q18A_OVERLAP_BOUNDED=1 bash "$0" "$@"\n'
LIFECYCLE_TIMEOUT = '  exec timeout --signal=TERM --kill-after=900s 9600s env Q18A_OVERLAP_BOUNDED=1 bash "$0" "$@"\n'
TRUNK_SLOT_NOTE = '# entry unless its 16200 s timeout plus 900 s kill margin fits that window (native39_trunk_gate derivation).\n'
LIFECYCLE_SLOT_NOTE = '# entry unless its 9600 s timeout plus 900 s kill margin fits that window (native39_lifecycle_gate derivation).\n'
TRUNK_SLOT = 'from native39_trunk_gate import trunk_window_ok as window_ok\n'
LIFECYCLE_SLOT = 'from native39_lifecycle_gate import lifecycle_window_ok as window_ok\n'
TRUNK_INTERFACES = 'for _interface in e3_native_startup_grade.py '
LIFECYCLE_INTERFACES = 'for _interface in native39_lifecycle_gate.py native35_lifetime_grade.py e3_native_startup_grade.py '
TRUNK_DIAGNOSTIC = "  record native35-diagnostic 2 'NOT_RUN: LOCALIZE=0 declared; no receipts to read'\n"
L_GRADE = ('  python3 "$H/native35_lifetime_grade.py" --queue "$Q" --out "$Q/native35-lifetime-grade.json" > "$Q/native35-lifetime-grade.log" 2>&1\n'
           "  record native35-lifetime \"$?\" 'native cancel/drain/reuse plus unchanged hygiene/leak oracles; no performance claim'\n")
TRUNK_CUT_BEGIN = 'qualification_phase rule4_controls\n'
TRUNK_CUT_END = "step trunk-summary \"$?\" 'descriptive: W2 beside the native W1 FULL and plain W1 baselines; 80/80 is a marker'\n"
L_BLOCK_BEGIN = "Q18A_PHASE_VERDICTS='{'\n"
ENDING = ('\nstop_pressure || true\nstop_sampler || true\ncapture_logs || true\ncapture_end "$Q18A_IDENTITY_TIMEOUT_S" || true\n'
          'stop_engine || true\nrestore_launcher || true\nrestore_defaults || true\narchive_owned_next || true\nwrite_cleanup\nDONE=1\n'
          '[[ $FAILS == 0 ]] || exit 1\n[[ $INCS == 0 ]] || exit 2\nexit 0\n')
L_GATES_LOOP = 'for _gate in startup row1 rule4_controls rule4_candidate row4; do\n'
L_GATES_STEP = 'gate_rc=$?;step lifetime-gates "$gate_rc"\n'
L_CONDITION = 'if [[ $gate_rc == 0 && $ENGINE_CRASHED == 0 ]]; then\n'
LIFECYCLE_CONDITION = 'if [[ $gate_rc == 0 && $ENGINE_CRASHED == 0 && $INSTRUMENTS_LIFECYCLE == 0 ]]; then\n'
L_NOT_RUN = "  record lifecycle 2 'native lifetime and hygiene NOT_RUN after missing gates or engine crash'\n"
LIFECYCLE_NOT_RUN = ("  record lifecycle 2 'native lifetime and hygiene NOT_RUN after missing gates, engine crash or "
                     "observer and offers off not proven (instruments-lifecycle)'\n")
# The trunk's instruments lines (q18a_native39_trunk.draft.sh, held-b), unchanged but for the receipt names.
INSTRUMENTS_BEFORE = (
    'INSTRUMENTS_LIFECYCLE=2\n'
    + L_CONDITION
    + '  # Native39 lifecycle: observer and offers off, proven on every rank process before the lifecycle (the flags\n'
      '  # are not the proof).\n'
      "  _cid=$(docker inspect --format '{{.Id}}' \"$CONTAINER\" 2>/dev/null) || _cid=''\n"
      '  python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \\\n'
      '    --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 \\\n'
      '    --out "$Q/instruments-lifecycle.json" > "$Q/instruments-lifecycle.log" 2>&1\n'
      '  INSTRUMENTS_LIFECYCLE=$?; record instruments-lifecycle "$INSTRUMENTS_LIFECYCLE" '
      "'replay observer library mapped in no rank process (actual state; observer and offers off)'\n"
      'fi\n')
INSTRUMENTS_AFTER = (
    '# Native39 lifecycle: the end instruments receipt brackets the lifecycle block.\n'
    'if [[ $INSTRUMENTS_LIFECYCLE == 0 ]] && health_ok; then\n'
    '  python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \\\n'
    '    --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 --same-as "$Q/instruments-lifecycle.json" \\\n'
    '    --out "$Q/instruments-end.json" > "$Q/instruments-end.log" 2>&1\n'
    '  record instruments-end "$?" \'after the lifecycle: the same rank processes, replay observer mapped in none\'\n'
    'fi\n')
LIFECYCLE_GATES_LOOP = 'for _gate in startup row1; do\n'
L_WORKLOAD = '--workload "$H/workloads/q18a-w2-overlap.json"'
LIFECYCLE_WORKLOAD = '--workload "$H/workloads/q18a-w2-overlap-native39.json"'
LIFECYCLE_PREAMBLE = (
    '# Native39 lifecycle: rule4, row4, W1 and the W2 speed rounds are not part of this queue (lead, 1 Nov 2026);\n'
    '# row1 above is the lane proof.  Below: native35-L\'s gates over startup and row1 and its lifecycle block\n'
    '# (native28\'s), unchanged except the W2 manifest: the trunk\'s plannable one, whose fresh count the\n'
    '# w2-length guard checked against this engine\'s /tokenize.  No rule4 continuation exists here.\n'
    'RULE4_CONTINUATION=0\n'
    'export RULE4_CONTINUATION\n')
# Declared telemetry delta (lead, 1 Nov 2026): the engine as it runs, not only as declared.  The identity-start capture
# is the container env (engine_identity compares it to the declaration); the startup FLASH_PP_LAYOUT line is logged
# only by a block pool that read FLASH_POOL_STATS > 0 (core/scheduler.py, telemetry_hooks.LAYOUT).
POOL_TELEMETRY = (
    "python3 - \"$Q/engine.identity-start.json\" \"$Q/engine-stream.log\" \"$Q/engine-stream.err\" > \"$Q/pool-telemetry.log\" 2>&1 <<'PY_POOL_TELEMETRY'\n"
    'import json, sys\n'
    "env = json.load(open(sys.argv[1]))['identity']['env']\n"
    "if env.get('FLASH_POOL_STATS') != '200':\n"
    "    print('engine env FLASH_POOL_STATS %r, declared 200' % (env.get('FLASH_POOL_STATS'),)); raise SystemExit(1)\n"
    'layouts, bad = [], 0\n'
    'for name in sys.argv[2:]:\n'
    '    try:\n'
    "        text = open(name, errors='replace').read()\n"
    '    except OSError:\n'
    '        continue\n'
    '    for line in text.splitlines():\n'
    "        if 'FLASH_PP_LAYOUT ' in line:\n"
    '            try:\n'
    "                layouts.append(json.loads(line.split('FLASH_PP_LAYOUT ', 1)[1]))\n"
    '            except ValueError:\n'
    '                bad += 1\n'
    "kinds = sorted({str(g.get('kind')) for row in layouts for g in row.get('groups', [])})\n"
    "print(json.dumps({'flash_pool_stats': env['FLASH_POOL_STATS'], 'layout_lines': len(layouts), 'malformed': bad,\n"
    "                  'kinds': kinds}, sort_keys=True))\n"
    "raise SystemExit(0 if layouts and not bad and 'MambaSpec' in kinds else 2)\n"
    'PY_POOL_TELEMETRY\n'
    "record pool-telemetry \"$?\" 'engine env FLASH_POOL_STATS=200 as captured at identity-start; the startup FLASH_PP_LAYOUT "
    "line with a MambaSpec group in the followed engine log'\n")

TRUNK_SETTINGS_NOTE = ('# Native39 speed engine: FLASH_S5_REPLAY_OBSERVER=0 and FLASH_S5_ENGINE_OFFERS=0 below (q18b-w1ab2\n'
                       '# measured ~5 % decode for the two together); row4 placement needs both and is declared NOT_RUN.\n')
LIFECYCLE_SETTINGS_NOTE = ('# Native39 lifecycle: FLASH_S5_REPLAY_OBSERVER=0 and FLASH_S5_ENGINE_OFFERS=0 below, as on the trunk.\n'
                           '# Synchronous replay-observer grading inside before_route_change froze every observed split->FULL\n'
                           '# return (fn-main results/box/q18a-native36-wd-run-1); the lifecycle block reads neither.\n'
                           '# Native39 lifecycle telemetry delta (lead, 1 Nov 2026): FLASH_POOL_STATS=200 below. The pressure-lifetime\n'
                           '# oracle (pp_release_grade lifetime) needs the FLASH_PP_LAYOUT state-group layout and the FLASH_POOL_STATS\n'
                           '# state-group eviction counters; the S5 core overlay logs both only when FLASH_POOL_STATS > 0\n'
                           '# (patches/s5_engine/telemetry_hooks.py: counters read and logged every 200 scheduler steps, no allocation or\n'
                           '# eviction change). The E2 gate value (q16_settings.sh); no speed claim.\n')
TRUNK_PREFLIGHT_OFFERS = "assert env.get('FLASH_S5_ENGINE_OFFERS') == '0'\n"
LIFECYCLE_PREFLIGHT_OFFERS = (TRUNK_PREFLIGHT_OFFERS
                              + "# Native39 lifecycle telemetry delta: the pressure-lifetime oracle's pool counters (lead, 1 Nov 2026).\n"
                                "assert env.get('FLASH_POOL_STATS') == '200'\n")
TRUNK_DECL_OFFERS = "assert env.get('FLASH_S5_ENGINE_OFFERS')=='0', 'engine offers must be off for the speed engine'\n"
LIFECYCLE_DECL_OFFERS = (TRUNK_DECL_OFFERS
                         + "assert env.get('FLASH_POOL_STATS')=='200', 'pool telemetry (FLASH_POOL_STATS=200) must be on for the pressure-lifetime oracle'\n")
TRUNK_EXTRA_ENV = ' FLASH_S5_ENGINE_OFFERS=0 FLASH_S5_REPLAY_LIB='
LIFECYCLE_EXTRA_ENV = ' FLASH_S5_ENGINE_OFFERS=0 FLASH_POOL_STATS=200 FLASH_S5_REPLAY_LIB='
TRUNK_DECL_NATIVE = '      e3_native39_w2_length.py e3_native39_trunk_checks.py native39_trunk_gate.py w1_trace.py \\\n'
LIFECYCLE_DECL_NATIVE = TRUNK_DECL_NATIVE + '      native39_lifecycle_gate.py native35_lifetime_grade.py \\\n'


def swap(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new)


def swap_header(text):
    lines = text.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.startswith(TRUNK_HEADER_PREFIX)]
    assert len(hits) == 1 and lines[hits[0] + 1] == TRUNK_HEADER_NEXT, hits
    return ''.join(lines[:hits[0]]) + LIFECYCLE_HEADER + ''.join(lines[hits[0] + 2:])


def l_block():
    """native35-L's gates and lifecycle block: from its phase-verdict loop to its ending (exclusive)."""
    text = (RUNS / (L + '.draft.sh')).read_text()
    assert text.count(L_BLOCK_BEGIN) == 1 and text.endswith('\n' + ENDING[1:]) and text.count(ENDING) == 1
    return text[text.index(L_BLOCK_BEGIN):text.index(ENDING)]


def lifecycle_region():
    block = l_block()
    for old, new in ((L_GATES_LOOP, LIFECYCLE_GATES_LOOP), (L_WORKLOAD, LIFECYCLE_WORKLOAD), (L_CONDITION, LIFECYCLE_CONDITION),
                     (L_NOT_RUN, LIFECYCLE_NOT_RUN), (L_GATES_STEP, L_GATES_STEP + INSTRUMENTS_BEFORE)):
        block = swap(block, old, new)
    assert MANIFEST_L not in block and block.endswith('fi\n')
    return LIFECYCLE_PREAMBLE + POOL_TELEMETRY + block + INSTRUMENTS_AFTER


def cut(text):
    assert text.count(TRUNK_CUT_BEGIN) == 1 and text.count(TRUNK_CUT_END) == 1
    start, end = text.index(TRUNK_CUT_BEGIN), text.index(TRUNK_CUT_END) + len(TRUNK_CUT_END)
    assert start < end and text[end:] == ENDING, 'the trunk ending must follow the trunk summary step'
    return text[:start] + lifecycle_region() + text[end:]


def names(text):
    # Only the q18a-prefixed queue names; e3_native39_trunk_* modules and native39_trunk_gate stay.
    assert 'q18a-native39-lifecycle' not in text and 'q18a_native39_lifecycle' not in text
    return text.replace('q18a-native39-trunk', 'q18a-native39-lifecycle').replace(TRUNK, LIFECYCLE)


def derive_draft():
    text = (RUNS / (TRUNK + '.draft.sh')).read_text()
    text = swap_header(text)
    for old, new in ((TRUNK_TITLE, LIFECYCLE_TITLE), (TRUNK_TIMEOUT, LIFECYCLE_TIMEOUT), (TRUNK_SLOT_NOTE, LIFECYCLE_SLOT_NOTE), (TRUNK_SLOT, LIFECYCLE_SLOT),
                     (TRUNK_INTERFACES, LIFECYCLE_INTERFACES), (TRUNK_DIAGNOSTIC, L_GRADE),
                     (TRUNK_PREFLIGHT_OFFERS, LIFECYCLE_PREFLIGHT_OFFERS)):
        text = swap(text, old, new)
    text = names(cut(text))
    for gone in ('qualification_phase rule4', 'qualification_phase row4', 'e3_native_w1.py "$PROBE_URL"',
                 'e3_native39_w1.py "$PROBE_URL"', 'w1_trace.py" "$PROBE_URL"', 'overlap-round-', 'trunk-summary',
                 'ROW4_DECLARED_NOT_RUN', 'instruments-w1', 'ingress'):
        assert gone not in text, gone
    return text


def derive(name, swaps):
    text = (RUNS / (TRUNK + name)).read_text()
    for old, new in swaps:
        text = swap(text, old, new)
    return names(text)


def generate():
    return {RUNS / (LIFECYCLE + '.draft.sh'): derive_draft(),
            RUNS / (LIFECYCLE + '_settings.sh'): derive('_settings.sh', ((TRUNK_SETTINGS_NOTE, LIFECYCLE_SETTINGS_NOTE),
                                                                         (TRUNK_EXTRA_ENV, LIFECYCLE_EXTRA_ENV))),
            RUNS / (LIFECYCLE + '_declare.sh'): derive('_declare.sh', ((TRUNK_DECL_NATIVE, LIFECYCLE_DECL_NATIVE),
                                                                       (TRUNK_DECL_OFFERS, LIFECYCLE_DECL_OFFERS)))}


if __name__ == '__main__':
    check = '--check' in sys.argv
    for path, text in generate().items():
        if check:
            assert path.read_text() == text, path.name
        else:
            assert not path.exists(), path.name + ' exists; regenerate only under --check'
            path.write_text(text)
            path.chmod(0o664)
    print('native39 lifecycle queue drafts ' + ('verified' if check else 'written'))
