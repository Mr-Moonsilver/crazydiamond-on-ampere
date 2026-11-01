#!/usr/bin/env python3
"""Derive native40's box scripts from committed native39 held-b texts.

Targeted startup/bind/unarmed/row1/one-fresh-W2 regression on unchanged
held-b model settings. The phase budget drives watchdog and window.
"""
import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OLD = 'q18a_native39_trunk'
NEW = 'q18a_native40_unarmed_hold'
TEST_CASES = 9
BUDGET = '''native40_budget() {
  python3 - "$1" <<'PY_NATIVE40_BUDGET'
import os,sys
sys.path.insert(0,os.environ.get('HARNESS_DIR') or os.environ.get('H') or '')
from native40_box_gate import ACTIVE_S,CLEANUP_S,PHASES
print(ACTIVE_S if sys.argv[1]=='active' else CLEANUP_S if sys.argv[1]=='cleanup' else PHASES[sys.argv[1]])
PY_NATIVE40_BUDGET
}
'''
TARGET = '''# Native40 targeted regression: rule4, row4, W1 and solo/append/repeated W2 are NOT_RUN.
[[ ${PHASE_VERDICT[row1]:-} == PASS ]] || { DONE=1; exit 2; }
_cid=$(docker inspect --format '{{.Id}}' "$CONTAINER" 2>/dev/null) || _cid=''
python3 "$H/e3_native39_trunk_checks.py" idle --url "$PROBE_URL" --timeout-s 120 --out "$Q/w2-idle.json"
rc=$?; record w2-idle "$rc"; [[ $rc == 0 ]] || { DONE=1; exit "$rc"; }
timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \\
  --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 --out "$Q/instruments-begin.json"
rc=$?; record instruments-begin "$rc"; [[ $rc == 0 ]] || { DONE=1; exit "$rc"; }
_round_start=$EPOCHREALTIME
READINESS=$Q18A_READINESS GEN=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \\
S5_OVERLAY=$S5_OVERLAY S5_LAUNCHER_NAME=$S5_LAUNCHER_NAME \\
PACKAGE_MANIFEST=$S5_OVERLAY/SHA256SUMS GENERATION=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \\
timeout --signal=TERM --kill-after=15s "$(native40_budget w2_fresh)s" \\
  python3 "$H/e3_native40_regression.py" --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH" \\
  --workload "$H/workloads/q18a-w2-overlap-native39.json" --row1 "$Q/qual-row1.json" --out "$Q/w2-fresh.json" > "$Q/w2-fresh.log" 2>&1
rc=$?; record w2-fresh "$rc" 'one retained fresh native overlap, regression only; no speed claim'
_round_end=$EPOCHREALTIME
timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \\
  --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 --same-as "$Q/instruments-begin.json" --out "$Q/instruments-end.json"
record instruments-end "$?"
python3 "$H/e3_native39_trunk_checks.py" ingress --q "$Q" --label native40-w2-fresh --start "$_round_start" --end "$_round_end" \\
  --generation 9 --out "$Q/ingress-fresh.json" > "$Q/ingress-fresh.log" 2>&1
record ingress-fresh "$?" 'eight standing requests plus one fresh prefill; no foreign ingress'
health_ok && [[ $(systemctl show "$SERVICE" -p NRestarts --value) == "$N0" ]]
_alive_rc=$?
record regression-engine "$_alive_rc" 'engine alive, restart count unchanged'
python3 - "$Q/engine-alive.json" "$_alive_rc" <<'PY_NATIVE40_ALIVE'
import json,sys
with open(sys.argv[1],'x') as f:json.dump({'verdict':'PASS' if sys.argv[2]=='0' else 'FAIL'},f)
PY_NATIVE40_ALIVE
python3 "$H/e3_native40_regression.py" --grade-dir "$Q" --out "$Q/regression-grade.json"
record native40-regression "$?" 'descriptive raw rates against held-b manifest; no denominator or speed claim'

'''
ADD = '''# Native40: no native admission has been armed in this queue. A plain
# stock prefill must remain held on split until the automatic typed FULL return.
timeout --signal=TERM --kill-after=15s 180s python3 "$H/e3_native40_unarmed_box.py" \\
  --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH" \\
  --timeout-s 100 --control-timeout-s 30 --out "$Q/unarmed-stock-prefill.json" \\
  > "$Q/unarmed-stock-prefill.log" 2>&1
rc=$?; record unarmed-stock-prefill "$rc" 'plain prefill: unarmed split hold -> automatic FULL -> HTTP completion, all-rank empty FULL'
[[ $rc == 0 ]] || { DONE=1; exit "$rc"; }

'''


def derive():
    out = {}
    for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
        text = (ROOT / 'harness/runs' / (OLD + suffix)).read_text()
        text = text.replace(OLD, NEW).replace('q18a-native39-trunk', 'q18a-native40-unarmed-hold')
        if suffix == '.draft.sh':
            old = 'exec timeout --signal=TERM --kill-after=900s 16200s'
            assert text.count(old) == 1
            text = text.replace('if [[ ${Q18A_OVERLAP_BOUNDED:-0} != 1 ]]; then',
                                BUDGET + 'if [[ ${Q18A_OVERLAP_BOUNDED:-0} != 1 ]]; then')
            text = text.replace(old, 'exec timeout --signal=TERM --kill-after="$(native40_budget cleanup)s" "$(native40_budget active)s"')
            text = text.replace('from native39_trunk_gate import trunk_window_ok as window_ok',
                                'from native40_box_gate import window_ok')
            mark = 'python3 "$H/e3_native_request_plans.py"'
            assert text.count(mark) == 1
            text = text.replace(mark, ADD + mark)
            start = text.index('qualification_phase rule4_controls\n')
            end = text.index('stop_pressure || true\nstop_sampler || true\n', start)
            text = text[:start] + TARGET + text[end:]
            text = '\n'.join('# Native40 targeted: startup/bind -> unarmed stock hold -> row1 -> one fresh W2 -> owned cleanup; no speed claim.' if line.startswith('# Native39 trunk:') else line for line in text.split('\n'))
            text = text.replace('240s systemctl restart', '"$(native40_budget restart)s" systemctl restart')
            text = text.replace('python3 "$H/e3_native_startup_grade.py"',
                                'timeout "$(native40_budget bootstrap)s" python3 "$H/e3_native_startup_grade.py"')
            text = text.replace('python3 "$H/e3_native39_w2_length.py"',
                                'timeout 60s python3 "$H/e3_native39_w2_length.py"')
            text = text.replace('python3 "$H/e3_native_request_plans.py"',
                                'timeout "$(native40_budget row1_plan)s" python3 "$H/e3_native_request_plans.py"')
            text = text.replace('  python3 "$H/$cli"', '  timeout --signal=TERM --kill-after=15s "$(native40_budget row1)s" python3 "$H/$cli"')
            text = text.replace('--timeout-s 900 --snapshot-max-bytes', '--timeout-s "$(native40_budget row1)" --snapshot-max-bytes')
            text = text.replace("  record native30-localize \"$?\" 'all-rank localization; no qualification or performance claim'",
                '  _diag_rc=$?; if [[ $_diag_rc == 1 ]]; then record native30-localize 1; else PHASE_VERDICT[native30-localize]=NOT_RUN; fi')
            text = text.replace("  record native35-diagnostic 2 'NOT_RUN: LOCALIZE=0 declared; no receipts to read'",
                                '  PHASE_VERDICT[native35-diagnostic]=NOT_RUN')
            interface = '  e3_native39_w1.py e3_native39_round.py; do'
            assert text.count(interface) == 1
            text = text.replace(interface, interface.replace('; do', ' e3_native40_unarmed_box.py e3_native40_regression.py native40_box_gate.py; do'))
            text = text.replace('HEALTH_DEADLINE_S=540', 'HEALTH_DEADLINE_S=$(native40_budget health)')
            text = text.replace('--timeout "$Q18A_IDENTITY_TIMEOUT_S"', '--timeout "$(native40_budget identity)"')
            text = text.replace('capture_end "$Q18A_IDENTITY_TIMEOUT_S"', 'capture_end "$(native40_budget end_identity)"')
            text = text.replace('its 16200 s timeout', 'its derived active timeout')
        out['harness/runs/' + NEW + suffix] = text
    return out


def derive_tools():
    text = (ROOT / 'tools/native39_trunk_cpu_probe.py').read_text()
    line = "suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native39_trunk_queue.py'))"
    assert text.count(line) == 1
    text = text.replace(line, line + "\nsuite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native40_box_queue.py'))")
    text = re.sub(r'^EXPECTED_TESTS=(\d+).*$', lambda m: 'EXPECTED_TESTS=' + str(int(m.group(1)) + TEST_CASES)
                  + '  # inherited native39 cases plus native40 queue/CLI/grade cases', text, flags=re.M)
    text = text.replace("owner.QNAME=='q18a-native39-trunk'", "owner.QNAME=='q18a-native40-unarmed-hold'")
    text = text.replace('q18a_native39_trunk_settings.sh', 'q18a_native40_unarmed_hold_settings.sh')
    text = text.replace('active_and_cleanup=native39_trunk_gate.TRUNK_ACTIVE_AND_CLEANUP',
                        'import native40_box_gate\nactive_and_cleanup=native40_box_gate.ACTIVE_S+native40_box_gate.CLEANUP_S')
    text = text.replace('slot_receipts=slot_receipts+trunk_slots',
                        "slot_receipts=slot_receipts+trunk_slots+sys.modules['test_native40_box_queue'].SLOT_RECEIPTS")
    wrapper = (ROOT / 'tools/native39_trunk_bundle_selfrun.py').read_text().replace(
        'native39_trunk_cpu_probe.py', 'native40_box_cpu_probe.py').replace(
        "schema='native39-trunk-bundle-selfrun/1'", "schema='native40-box-bundle-selfrun/1'")
    return {'tools/native40_box_cpu_probe.py': text, 'tools/native40_box_selfrun.py': wrapper}


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
    print('native40 queue derivation PASS')


if __name__ == '__main__':
    main()
