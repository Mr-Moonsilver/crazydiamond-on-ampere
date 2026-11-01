#!/usr/bin/env python3
"""Derive native41's box scripts from the committed native40 box texts.

Kept from native40: pins, preflight, startup, bind, W2 length guard, host
headroom, the unarmed-hold check and the whole owned cleanup/restore tail.
Changed: names, the budget-derived window (native41_box_gate), the auto-arm
switch FLASH_S5_AUTO_ARM=1 in EXTRA_ENV (asserted in declaration and preflight),
and the body after the unarmed check: native40's row1/rule4/W1/W2 trunk is
replaced by four plain W2 arms (stock, auto, stock, auto) with instruments,
idle/typed-FULL entries, per-arm ingress, a typed FULL at the end and the
native41 grade. No model setting, workload, gate threshold or restore change.
"""
import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OLD = 'q18a_native40_unarmed_hold'
NEW = 'q18a_native41_autoarm'
OLD_Q = 'q18a-native40-unarmed-hold'
NEW_Q = 'q18a-native41-autoarm'
SWITCH = 'FLASH_S5_AUTO_ARM=1'
BODY_START = 'python3 "$H/e3_native_request_plans.py"'
BODY_END = ("step trunk-summary \"$?\" 'descriptive: W2 beside the native W1 FULL and plain W1 baselines; "
            "80/80 is a marker'\n")
HEADER = '''# q18a-native41-autoarm: native40's startup, bind and unarmed-hold check with FLASH_S5_AUTO_ARM=1,
# then plain W2 traffic in four arms (stock, auto, stock, auto) and a typed FULL. Unrun, no GO from this file.
#
# Native41: ordinary plain-client W2 requests (no native client, no prepare) must auto-arm the native
# lane. The startup identity and the unarmed check take the manual lease; stock arms keep it (auto-arm
# suppressed, same engine), auto arms pause/release/resume once with their standing cohort open.
# The inherited native39/native40 description below covers the kept startup and owned cleanup; its
# row1 -> rule4 -> W1 -> W2 trunk body is replaced by the native41 arms.
'''
BODY = r'''# Native41: plain W2 arms beside the auto-arm switch. Receipts only; speed is descriptive, never a gate.
_cid=$(docker inspect --format '{{.Id}}' "$CONTAINER" 2>/dev/null) || _cid=''
python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
  --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 \
  --out "$Q/instruments-start.json" > "$Q/instruments-start.log" 2>&1
INSTRUMENTS_START=$?
record instruments-start "$INSTRUMENTS_START" 'replay observer library mapped in no rank process (observer and offers off)'
GRADE_ARGS=()
if [[ $INSTRUMENTS_START == 0 ]]; then
  for _pair in 0 1; do
    for _kind in stock auto; do
      _label=$_kind-$_pair
      n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown
      if [[ $n != "$N0" ]] || host_crash_detected || ! health_ok; then
        record "arm-$_label" 2 'NOT_RUN: engine crashed, unhealthy or auto-restarted; no restart attempted'
        continue
      fi
      if [[ $_label == stock-1 ]]; then
        # The typed FULL re-takes the manual lease that auto-0 released.
        python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
          --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/entry-$_label.json" > "$Q/entry-$_label.log" 2>&1
        _entry_rc=$?; record "entry-$_label" "$_entry_rc" 'typed all-rank FULL, resumed, idle, workers empty; lease taken'
      else
        python3 "$H/e3_native39_trunk_checks.py" idle --url "$PROBE_URL" --timeout-s "$IDLE_TIMEOUT_S" \
          --out "$Q/entry-$_label.json" > "$Q/entry-$_label.log" 2>&1
        _entry_rc=$?; record "entry-$_label" "$_entry_rc" 'engine idle (running 0, waiting 0) before the arm'
      fi
      if [[ $_entry_rc != 0 ]]; then
        record "arm-$_label" 2 'NOT_RUN: arm entry not idle'
        continue
      fi
      READINESS=$Q18A_READINESS GEN=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \
      S5_OVERLAY=$S5_OVERLAY S5_LAUNCHER_NAME=$S5_LAUNCHER_NAME \
      PACKAGE_MANIFEST=$S5_OVERLAY/SHA256SUMS GENERATION=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \
      timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_native41_autoarm_box.py" arm \
        --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH" --kind "$_kind" --index "$_pair" \
        --workload "$H/workloads/q18a-w2-overlap-native39.json" --control-timeout-s 60 \
        --out "$Q/arm-$_label.json" > "$Q/arm-$_label.log" 2>&1
      record "arm-$_label" "$?" 'plain W2: 8 standing decoders, fresh, append; stock keeps the lease, auto releases it'
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
        python3 "$H/e3_native39_trunk_checks.py" ingress --q "$Q" --label "$_label" --start "$_s" --end "$_e" \
          --generation 10 --out "$Q/ingress-$_label.json" > "$Q/ingress-$_label.log" 2>&1
        record "ingress-$_label" "$?" 'access log: every request in the arm span is the arm (8 standing, fresh, append), else INVALID'
      else
        record "ingress-$_label" 2 'arm span missing; ingress not checked'
      fi
      GRADE_ARGS+=(--ingress "$_label=$Q/ingress-$_label.json")
    done
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
  python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 --same-as "$Q/instruments-start.json" \
    --out "$Q/instruments-end.json" > "$Q/instruments-end.log" 2>&1
  record instruments-end "$?" 'after the arms: the same rank processes, replay observer mapped in none'
  python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/end-full-idle.json" > "$Q/end-full-idle.log" 2>&1
  record end-full-idle "$?" 'typed all-rank FULL at the end: resumed, idle, workers empty'
fi
[[ ${#GRADE_ARGS[@]} -gt 0 ]] || GRADE_ARGS=(--arm "$Q/arm-none.json")
python3 "$H/e3_native41_autoarm_box.py" grade "${GRADE_ARGS[@]}" \
  --check "instruments-start=$Q/instruments-start.json" --check "instruments-end=$Q/instruments-end.json" \
  --check "end-full-idle=$Q/end-full-idle.json" --check "unarmed-stock-prefill=$Q/unarmed-stock-prefill.json" \
  --out "$Q/native41-autoarm-grade.json" > "$Q/native41-autoarm-grade.log" 2>&1
record native41-autoarm "$?" 'auto engagement from plain W2 traffic; descriptive speed beside the same-queue stock arms'
'''


def derive():
    out = {}
    for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
        text = (ROOT / 'harness/runs' / (OLD + suffix)).read_text()
        text = text.replace(OLD, NEW).replace(OLD_Q, NEW_Q)
        if suffix == '.draft.sh':
            first, rest = text.split('\n', 1)
            assert first == '#!/usr/bin/env bash'
            line2 = rest.split('\n', 1)[0]
            assert line2.startswith('# ' + NEW_Q + ': ')
            text = first + '\n' + HEADER + rest.split('\n', 1)[1]
            old = 'exec timeout --signal=TERM --kill-after=900s 2400s'
            assert text.count(old) == 1
            text = text.replace(old, 'exec timeout --signal=TERM --kill-after=900s 10860s')
            text = text.replace('its 2400 s timeout', 'its 10860 s timeout')
            old = 'from native40_box_gate import window_ok'
            assert text.count(old) == 1
            text = text.replace(old, 'from native41_box_gate import window_ok')
            interface = ' e3_native40_unarmed_box.py native40_box_gate.py; do'
            assert text.count(interface) == 1
            text = text.replace(interface, ' e3_native40_unarmed_box.py native40_box_gate.py \\\n'
                                           '  e3_native41_autoarm_box.py native41_box_gate.py; do')
            start, end = text.index(BODY_START), text.index(BODY_END)
            assert text.count(BODY_START) == 1 and text.count(BODY_END) == 1 and start < end
            text = text[:start] + BODY + text[end + len(BODY_END):]
            old = "assert env.get('FLASH_S5_ENGINE_OFFERS') == '0'\n"
            assert text.count(old) == 1
            text = text.replace(old, old + "assert env.get('FLASH_S5_AUTO_ARM') == '1', 'native41 auto-arm switch'\n")
            for old, new in (("perf=p.parent/'native35-performance-grade.json'",
                              "perf=p.parent/'native41-autoarm-grade.json'"),
                             ("'missing':['native35-performance-grade-absent']",
                              "'missing':['native41-autoarm-grade-absent']")):
                assert text.count(old) == 1, old
                text = text.replace(old, new)
        elif suffix == '_settings.sh':
            old = ' FLASH_S5_REPLAY_OBSERVER=0 FLASH_S5_ENGINE_OFFERS=0 '
            assert text.count(old) == 1
            text = text.replace(old, old + SWITCH + ' ')
            note = '# Native39 speed engine: FLASH_S5_REPLAY_OBSERVER=0 and FLASH_S5_ENGINE_OFFERS=0 below'
            assert text.count(note) == 1
            text = text.replace(note, '# Native41: FLASH_S5_AUTO_ARM=1 below, the only setting added to native40.\n' + note)
        else:
            old = "assert env.get('FLASH_S5_ENGINE_OFFERS')=='0', 'engine offers must be off for the speed engine'\n"
            assert text.count(old) == 1
            text = text.replace(old, old + "assert env.get('FLASH_S5_AUTO_ARM')=='1', 'native41 auto-arm switch must be 1'\n")
        out['harness/runs/' + NEW + suffix] = text
    return out


def derive_tools():
    text = (ROOT / 'tools/native40_box_cpu_probe.py').read_text()
    line = "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native40_box_queue.py'))"
    assert text.count(line) == 1
    text = text.replace(line, line + "\nsuite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native41_box_queue.py'))")
    text = re.sub(r'^EXPECTED_TESTS=(\d+).*$', lambda m: 'EXPECTED_TESTS=' + str(int(m.group(1)) + NATIVE41_QUEUE_TESTS)
                  + '  # inherited native40 cases plus %d native41 queue cases' % NATIVE41_QUEUE_TESTS, text, flags=re.M)
    text = text.replace("owner.QNAME=='q18a-native40-unarmed-hold'", "owner.QNAME=='q18a-native41-autoarm'")
    text = text.replace('q18a_native40_unarmed_hold_settings.sh', 'q18a_native41_autoarm_settings.sh')
    text = text.replace('import native40_box_gate\nactive_and_cleanup=native40_box_gate.ACTIVE_S+native40_box_gate.CLEANUP_S',
                        'import native41_box_gate\nactive_and_cleanup=native41_box_gate.ACTIVE_S+native41_box_gate.CLEANUP_S')
    text = text.replace("sys.modules['test_native40_box_queue'].SLOT_RECEIPTS",
                        "sys.modules['test_native40_box_queue'].SLOT_RECEIPTS+sys.modules['test_native41_box_queue'].SLOT_RECEIPTS")
    wrapper = (ROOT / 'tools/native40_box_selfrun.py').read_text().replace(
        'native40_box_cpu_probe.py', 'native41_box_cpu_probe.py').replace(
        "schema='native40-box-bundle-selfrun/1'", "schema='native41-box-bundle-selfrun/1'")
    return {'tools/native41_box_cpu_probe.py': text, 'tools/native41_box_selfrun.py': wrapper}


NATIVE41_QUEUE_TESTS = 10  # test_native41_box_queue cases (testsRun)


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
    print('native41 queue derivation PASS')


if __name__ == '__main__':
    main()
