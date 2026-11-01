#!/usr/bin/env python3
"""Held-b guarded startup/restore, twice; replace only its workload phase.

The pair requires two fresh predeclared request directories next to output
paths. Packaging derives declarations from each literal policy's settings;
owner tooling supplies fresh root-only API keys. No box action at build time.
"""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'harness'))
import native43_boundary_gate as gate


def derive():
    parent = (ROOT/'harness/runs/q18a_native39_trunk.draft.sh').read_text()
    prefix = parent[:parent.index('SNAPSHOT_MAX_BYTES=$(python3 - "$R/request-plan.json"')]
    cleanup = parent[parent.index('stop_pressure || true\nstop_sampler || true\ncapture_logs || true\ncapture_end'):]
    settings = (ROOT/'harness/runs/q18a_native39_trunk_settings.sh').read_text()
    declare = (ROOT/'harness/runs/q18a_native39_trunk_declare.sh').read_text()
    rows = {}
    for policy in ('current','dense'):
        name, stem = 'q18a-native43-'+policy, 'q18a_native43_'+policy
        text = prefix + '''# Native43: no row1/rule4/W1/W2 cohort or full speed grade.
python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \\
  --out "$Q/instruments-before.json" > "$Q/instruments-before.log" 2>&1
rc=$?; record instruments-before "$rc" 'observer/offers/localize/diagnostics off; trace is the declared native43 exception'
[[ $rc == 0 ]] || { DONE=1; exit "$rc"; }
timeout --signal=TERM --kill-after=20s 360s python3 "$H/native43_boundary_client.py" \\
  --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH" \\
  --pair-id "${NATIVE43_PAIR_ID:?pair identity required}" --policy POLICY \\
  --workload "$H/workloads/q18a-w2-overlap-native39.json" --timeout-s 300 \\
  --out "$Q/native43-client.json" > "$Q/native43-client.log" 2>&1
rc=$?; record native43-boundary "$rc" 'fresh 197270, no cohort; engine stops after 101200; typed cancel/drain and FULL'
''' .replace('POLICY', policy) + cleanup
        text = text.replace('q18a-native39-trunk', name).replace('q18a_native39_trunk', stem)
        text = text.replace('from native39_trunk_gate import trunk_window_ok as window_ok',
                            'from native43_boundary_gate import arm_window_ok as window_ok')
        text = text.replace('16200s',str(gate.ARM_ACTIVE_S)+'s').replace('--kill-after=900s','--kill-after=%ds'%gate.ARM_KILL_MARGIN_S)
        text = text.replace('16200 s timeout plus 900 s', '%d s timeout plus %d s'%(gate.ARM_ACTIVE_S,gate.ARM_KILL_MARGIN_S))
        # Keep the same independent launcher/defaults/auth/declaration checks.
        anchor = "assert env.get('FLASH_S5_ENGINE_OFFERS') == '0'\n"
        assert text.count(anchor) == 1
        text = text.replace(anchor,anchor+"assert env.get('FLASH_S5_NATIVE_QSA_POLICY') == %r\n"%policy+
            "assert env.get('FLASH_S5_NATIVE_QSA_TRACE') == '1'\nassert env.get('FLASH_S5_NATIVE_QSA_STOP_AFTER') == '101200'\n")
        rows['harness/runs/'+stem+'.draft.sh'] = text
        config = settings.replace('q18a-native39-trunk',name).replace('q18a_native39_trunk',stem)
        anchor = 'FLASH_S5_REPLAY_OBSERVER=0 FLASH_S5_ENGINE_OFFERS=0 '
        assert config.count(anchor)==1
        config = config.replace(anchor,anchor+'FLASH_S5_NATIVE_QSA_POLICY='+policy+
             ' FLASH_S5_NATIVE_QSA_TRACE=1 FLASH_S5_NATIVE_QSA_STOP_AFTER=101200 ')
        rows['harness/runs/'+stem+'_settings.sh'] = config
        rows['harness/runs/'+stem+'_declare.sh'] = declare.replace('q18a-native39-trunk',name).replace('q18a_native39_trunk',stem)
    wrapper = '''#!/usr/bin/env bash
# Draft, owner-run only. One pair; fresh process + fresh root-only key per arm.
set -uo pipefail
umask 077
H=${HARNESS_DIR:?frozen bundle required}
BASE=${1:?fresh absolute output prefix required}
[[ $BASE == /* ]] || exit 2
[[ ! -e $BASE-pair-grade.json && ! -L $BASE-pair-grade.json ]] || exit 2
export PYTHONPATH="$H/e3_builder:$H"
python3 - <<'PY_WINDOW' || exit 2
import time
from native43_boundary_gate import window_ok
assert window_ok(time.time()), 'native43 pair cannot fit the owner window'
PY_WINDOW
if [[ ${NATIVE43_PAIR_BOUNDED:-0} != 1 ]]; then
  exec timeout --signal=TERM --kill-after=PAIR_KILLs PAIR_ACTIVEs env NATIVE43_PAIR_BOUNDED=1 bash "$0" "$@"
fi
export NATIVE43_PAIR_ID="native43-$(date -u +%Y%m%dT%H%M%SZ)-$$"
for policy in current dense; do
  output="$BASE-$policy"
  export REQUESTS="$output-requests"
  [[ ! -e $output && ! -L $output && -d $REQUESTS ]] || exit 2
  # Predeclared by the owner runner using this policy's frozen settings.
  # Each arm retains held-b preflight, identity/bind/startup, crash checks,
  # stop and exact launcher/defaults restore. No production start occurs.
  env -u Q18A_OVERLAP_BOUNDED bash "$H/runs/q18a_native43_${policy}.draft.sh" "$output"
  rc=$?
  # Do not pick a winner or erase a failed arm. A dirty/unsafe cleanup stops
  # the pair; a clean HARNESS-INC remains retained beside the second arm.
  python3 - "$output" <<'PY_CLEAN' || exit 2
import json,sys
from pathlib import Path
q=Path(sys.argv[1]);grade=json.loads((q/'queue-grade.json').read_text())
cleanup=json.loads((q/'cleanup.json').read_text())
assert not grade['engine_crashed'] and grade['engine_state'] in ('inactive','failed')
assert cleanup.get('engine_stopped') is True and cleanup.get('restore_withheld') is False
for kind in ('launcher','defaults'):
    assert cleanup.get(kind+'_found_sha256') and cleanup[kind+'_found_sha256']==cleanup.get(kind+'_restored_sha256')
PY_CLEAN
done
python3 "$H/native43_dense_grade.py" \\
  --current "$BASE-current/native43-client.json" --dense "$BASE-dense/native43-client.json" \\
  --current-log "$BASE-current/engine-stream.log" --current-log "$BASE-current/engine-stream.err" \\
  --dense-log "$BASE-dense/engine-stream.log" --dense-log "$BASE-dense/engine-stream.err" \\
  --out "$BASE-pair-grade.json"
'''
    wrapper = wrapper.replace('PAIR_KILLs',str(gate.PAIR_KILL_MARGIN_S)+'s').replace('PAIR_ACTIVEs',str(gate.PAIR_ACTIVE_S)+'s')
    rows['harness/runs/q18a_native43_dense_boundary.draft.sh'] = wrapper
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    for name,text in derive().items():
        path=ROOT/name
        if a.check:
            assert path.read_text()==text,name
        else:
            path.write_text(text)
if __name__=='__main__':main()
