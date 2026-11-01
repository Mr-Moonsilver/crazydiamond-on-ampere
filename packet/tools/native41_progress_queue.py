#!/usr/bin/env python3
"""Derive held-f's box scripts from the committed held-e (818a4a7) native41 queue texts.

Kept from held-e: pins, preflight, startup, bind, host headroom, the unarmed
check with its CT-side clock proof, the end identity and the whole owned
cleanup/restore tail. Changed: names (QNAME q18a-native41-progress), the one
phase budget (native41_progress_gate PHASES through the draft's budget
function), the progress switch FLASH_S5_NATIVE_PROGRESS=burst in EXTRA_ENV
(asserted in declaration and preflight), the length guard on the held-f
manifest, and the body after the unarmed check: held-e's four W2 arms are
replaced by eight auto-only arms (current | burst) x (live | solo) x 2 in ABBA
order, each behind a typed all-rank FULL entry, with instruments, per-arm
ingress, a typed FULL at the end and the progress grade over the followed
engine log. No model setting, gate threshold or restore change.
"""
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import native41_box_queue as held_e  # noqa: E402

OLD = 'q18a_native41_autoarm'
NEW = 'q18a_native41_progress'
OLD_Q = 'q18a-native41-autoarm-e'
NEW_Q = 'q18a-native41-progress'
SWITCH = 'FLASH_S5_NATIVE_PROGRESS=burst'
MANIFEST = 'workloads/q18a-native41-progress.json'
ARMS = ('current-solo-0', 'burst-solo-0', 'burst-live-0', 'current-live-0',
        'current-live-1', 'burst-live-1', 'burst-solo-1', 'current-solo-1')
HEADER_OLD = held_e.HEADER.replace(held_e.NEW_Q, OLD_Q)
HEADER = '''# q18a-native41-progress: held-e's startup, bind and unarmed check with FLASH_S5_AUTO_ARM=1 and
# FLASH_S5_NATIVE_PROGRESS=burst, then eight auto-only arms and a typed FULL. Unrun, no GO from this file.
#
# Held-f: matched native progress timing. Each arm is held-e's auto arm (plain W2 traffic, the policy
# alone runs the fresh prompt natively) with one typed native_progress {current|burst} between pause and
# release_auto_arm, and either all eight standing decoders open through the prefill (live) or cancelled
# at the first observed native chunk (solo). Order (ABBA): current-solo, burst-solo, burst-live,
# current-live, current-live, burst-live, burst-solo, current-solo; a typed all-rank FULL before each.
# The engine and worker progress lines in the followed engine log are the receipts; speed is descriptive.
# The inherited held-e/native40 description below covers the kept startup and owned cleanup; held-e's W2
# arms are replaced by the held-f arms. Every cap is native41_progress_gate.PHASES.
'''
BODY = r'''# Held-f: eight auto-only progress arms. Receipts only; speed is descriptive, never a gate.
# Every timeout below is one native41_progress_gate.PHASES entry (instruments, arm_entries, progress_arms,
# ingress, end_full_idle, grade). _cid is the engine container id read before the unarmed check.
timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
  --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 \
  --out "$Q/instruments-start.json" > "$Q/instruments-start.log" 2>&1
INSTRUMENTS_START=$?
record instruments-start "$INSTRUMENTS_START" 'replay observer library mapped in no rank process (observer and offers off)'
GRADE_ARGS=()
if [[ $INSTRUMENTS_START == 0 ]]; then
  for _label in current-solo-0 burst-solo-0 burst-live-0 current-live-0 current-live-1 burst-live-1 burst-solo-1 current-solo-1; do
    IFS=- read -r _mode _cohort _pair <<< "$_label"
    n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown
    if [[ $n != "$N0" ]] || host_crash_detected || ! health_ok; then
      record "arm-$_label" 2 'NOT_RUN: engine crashed, unhealthy or auto-restarted; no restart attempted'
      continue
    fi
    # The typed all-rank FULL takes the manual lease the previous arm released.
    timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
      --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/entry-$_label.json" > "$Q/entry-$_label.log" 2>&1
    _entry_rc=$?; record "entry-$_label" "$_entry_rc" 'typed all-rank FULL, resumed, idle, workers empty; lease taken'
    if [[ $_entry_rc != 0 ]]; then
      record "arm-$_label" 2 'NOT_RUN: arm entry not a typed idle FULL'
      continue
    fi
    READINESS=$Q18A_READINESS GEN=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \
    S5_OVERLAY=$S5_OVERLAY S5_LAUNCHER_NAME=$S5_LAUNCHER_NAME \
    PACKAGE_MANIFEST=$S5_OVERLAY/SHA256SUMS GENERATION=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \
    timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_native41_progress_box.py" arm \
      --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH" \
      --mode "$_mode" --cohort "$_cohort" --index "$_pair" \
      --workload "$H/workloads/q18a-native41-progress.json" --control-timeout-s 60 \
      --out "$Q/arm-$_label.json" > "$Q/arm-$_label.log" 2>&1
    record "arm-$_label" "$?" 'auto arm: 8 standing decoders, native_progress mode, fresh, append; live keeps them, solo cancels'
    GRADE_ARGS+=(--arm "$Q/arm-$_label.json")
    _span=$(python3 - "$Q/arm-$_label.json" <<'PY_SPAN'
import json,sys
try:d=json.load(open(sys.argv[1]))
except (OSError,ValueError):raise SystemExit(0)
s,e=d.get('wall_start_s'),d.get('wall_end_s')
if type(s) in (int,float) and type(e) in (int,float):print(repr(float(s)),repr(float(e)))
PY_SPAN
)
    if [[ -n $_span ]]; then
      read -r _s _e <<< "$_span"
      timeout 60s python3 "$H/e3_native39_trunk_checks.py" ingress --q "$Q" --label "$_label" --start "$_s" --end "$_e" \
        --generation 10 --out "$Q/ingress-$_label.json" > "$Q/ingress-$_label.log" 2>&1
      record "ingress-$_label" "$?" 'access log: every request in the arm span is the arm (8 standing, fresh, append), else INVALID'
    else
      record "ingress-$_label" 2 'arm span missing; ingress not checked'
    fi
    GRADE_ARGS+=(--ingress "$_label=$Q/ingress-$_label.json")
  done
  n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown
  if [[ $n != "$N0" ]] || ! health_ok; then
    ENGINE_CRASHED=1; record arms-engine 1 'engine unhealthy or auto-restarted; no restart attempted'
  else
    record arms-engine 0 'engine healthy with unchanged NRestarts'
  fi
else
  record arms 2 'NOT_RUN: observer and offers off not proven on every rank process (instruments-start)'
fi
if [[ $INSTRUMENTS_START == 0 ]] && health_ok; then
  timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 --same-as "$Q/instruments-start.json" \
    --out "$Q/instruments-end.json" > "$Q/instruments-end.log" 2>&1
  record instruments-end "$?" 'after the arms: the same rank processes, replay observer mapped in none'
  timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/end-full-idle.json" > "$Q/end-full-idle.log" 2>&1
  record end-full-idle "$?" 'typed all-rank FULL at the end: resumed, idle, workers empty'
fi
[[ ${#GRADE_ARGS[@]} -gt 0 ]] || GRADE_ARGS=(--arm "$Q/arm-none.json")
timeout 300s python3 "$H/e3_native41_progress_box.py" grade "${GRADE_ARGS[@]}" \
  --check "instruments-start=$Q/instruments-start.json" --check "instruments-end=$Q/instruments-end.json" \
  --check "end-full-idle=$Q/end-full-idle.json" --check "unarmed-stock-prefill=$Q/unarmed-stock-prefill.json" \
  --log "$Q/engine-stream.log" --log "$Q/engine-stream.err" \
  --out "$Q/native41-progress-grade.json" > "$Q/native41-progress-grade.log" 2>&1
record native41-progress "$?" 'receipts and complete progress lines for all eight arms; per-chunk breakdown descriptive'
'''
GUARD_OLD = '''# Native39: the W2 counts the trunk manifest declares, from this engine's /tokenize, must plan.
timeout 60s python3 "$H/e3_native39_w2_length.py" --url "$PROBE_URL" --model "$MODEL" \\
  --workload "$H/workloads/q18a-w2-overlap-native39.json" --out "$Q/w2-length.json" > "$Q/w2-length.log" 2>&1
rc=$?; record w2-length "$rc" 'declared W2 fresh/append counts equal /tokenize and both have a native plan'
'''
GUARD_NEW = '''# Held-f: the fresh/append counts the progress manifest declares, from this engine's /tokenize, must plan.
timeout 60s python3 "$H/e3_native39_w2_length.py" --url "$PROBE_URL" --model "$MODEL" \\
  --workload "$H/workloads/q18a-native41-progress.json" --out "$Q/w2-length.json" > "$Q/w2-length.log" 2>&1
rc=$?; record w2-length "$rc" 'declared progress fresh/append counts equal /tokenize and both have a native plan'
'''


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:120]
    return text.replace(old, new)


def derive():
    out = {}
    for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
        text = (ROOT / 'harness/runs' / (OLD + suffix)).read_text()
        text = text.replace(OLD, NEW).replace(OLD_Q, NEW_Q)
        if suffix == '.draft.sh':
            text = replace_once(text, HEADER_OLD.replace(OLD_Q, NEW_Q), HEADER)
            text = replace_once(text, held_e.BODY, BODY)
            text = replace_once(text, GUARD_OLD, GUARD_NEW)
            # One phase budget: the draft's budget function and window read native41_progress_gate.
            text = replace_once(text, 'from native41_box_gate import ACTIVE_S,CLEANUP_S,PHASES',
                                'from native41_progress_gate import ACTIVE_S,CLEANUP_S,PHASES')
            text = replace_once(text, 'from native41_box_gate import window_ok',
                                'from native41_progress_gate import window_ok')
            text = replace_once(text, '  e3_native41_unarmed_box.py e3_native41_autoarm_box.py native41_box_gate.py; do',
                                '  e3_native41_unarmed_box.py e3_native41_autoarm_box.py native41_box_gate.py \\\n'
                                '  e3_native41_progress_box.py native41_progress_gate.py; do')
            text = replace_once(text, '   -r $H/agent_replay.py && -r $H/workloads/q18a-w2-overlap-native39.json &&',
                                '   -r $H/agent_replay.py && -r $H/workloads/q18a-w2-overlap-native39.json &&\n'
                                '   -r $H/workloads/q18a-native41-progress.json &&')
            old = "assert env.get('FLASH_S5_AUTO_ARM') == '1', 'native41 auto-arm switch'\n"
            text = replace_once(text, old, old + "assert env.get('FLASH_S5_NATIVE_PROGRESS') == 'burst', "
                                                 "'held-f native progress switch'\n")
            for old, new in (("perf=p.parent/'native41-autoarm-grade.json'",
                              "perf=p.parent/'native41-progress-grade.json'"),
                             ("'missing':['native41-autoarm-grade-absent']",
                              "'missing':['native41-progress-grade-absent']")):
                text = replace_once(text, old, new)
        elif suffix == '_settings.sh':
            text = replace_once(text, ' FLASH_S5_ENGINE_OFFERS=0 FLASH_S5_AUTO_ARM=1 ',
                                ' FLASH_S5_ENGINE_OFFERS=0 FLASH_S5_AUTO_ARM=1 ' + SWITCH + ' ')
            note = '# Native41: FLASH_S5_AUTO_ARM=1 below, the only setting added to native40.\n'
            text = replace_once(text, note, note + '# Held-f: FLASH_S5_NATIVE_PROGRESS=burst below, the only setting '
                                                   'added to held-e (receipts on\n# auto-owned native chunks; each '
                                                   'arm selects current or burst by typed control).\n')
        else:
            old = "assert env.get('FLASH_S5_AUTO_ARM')=='1', 'native41 auto-arm switch must be 1'\n"
            text = replace_once(text, old, old + "assert env.get('FLASH_S5_NATIVE_PROGRESS')=='burst', "
                                                 "'held-f native progress switch must be burst'\n")
        out['harness/runs/' + NEW + suffix] = text
    return out


def derive_tools():
    text = (ROOT / 'tools/native41_box_cpu_probe.py').read_text()
    line = "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native41_unarmed_box.py'))"
    text = replace_once(text, line, line + "\nsuite.addTests(unittest.defaultTestLoader.discover('/source/harness',"
                                           "pattern='test_native41_progress_queue.py'))")
    text = re.sub(r'^EXPECTED_TESTS=(\d+).*$', lambda m: 'EXPECTED_TESTS=' + str(int(m.group(1)) + PROGRESS_QUEUE_TESTS)
                  + '  # held-e cases plus %d held-f progress queue cases' % PROGRESS_QUEUE_TESTS,
                  text, count=1, flags=re.M)
    text = replace_once(text, "owner.QNAME==%r" % OLD_Q, "owner.QNAME==%r" % NEW_Q)
    text = replace_once(text, 'q18a_native41_autoarm_settings.sh', 'q18a_native41_progress_settings.sh')
    text = replace_once(text, 'import native41_box_gate\nactive_and_cleanup=native41_box_gate.ACTIVE_S+native41_box_gate.CLEANUP_S',
                        'import native41_progress_gate\nactive_and_cleanup=native41_progress_gate.ACTIVE_S+'
                        'native41_progress_gate.CLEANUP_S')
    text = replace_once(text, "+sys.modules['test_native41_box_queue'].SLOT_RECEIPTS",
                        "+sys.modules['test_native41_box_queue'].SLOT_RECEIPTS"
                        "+sys.modules['test_native41_progress_queue'].SLOT_RECEIPTS")
    wrapper = (ROOT / 'tools/native41_box_selfrun.py').read_text()
    wrapper = replace_once(wrapper, 'native41_box_cpu_probe.py', 'native41_progress_cpu_probe.py')
    wrapper = replace_once(wrapper, "schema='native41-box-bundle-selfrun/1'", "schema='native41-progress-bundle-selfrun/1'")
    return {'tools/native41_progress_cpu_probe.py': text, 'tools/native41_progress_selfrun.py': wrapper}


PROGRESS_QUEUE_TESTS = 19  # test_native41_progress_queue testsRun


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rewrite', action='store_true')
    a = p.parse_args()
    for name, text in (derive() | derive_tools()).items():
        path = ROOT / name
        if a.rewrite:
            path.write_text(text)
        else:
            assert path.read_text() == text, name
    print('held-f queue derivation PASS')


if __name__ == '__main__':
    main()
