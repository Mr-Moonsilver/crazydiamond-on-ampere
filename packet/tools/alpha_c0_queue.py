#!/usr/bin/env python3
"""Derive the q18a-alpha-c0 box scripts from the committed alpha-perf (40a4cea) texts. Nothing a past queue ran is edited.

Kept from alpha-perf, byte for byte: pins, preflight, startup, bind, host headroom, the unarmed check, the end identity,
both band manifests and their /tokenize guards (band 2 is not run here; its guard stays, it costs seconds),
the le-fix death scan and watch, the M3 allocator receipts and the whole
owned cleanup/restore tail. Declared deltas, each applied exactly once:
  names    q18a_alpha_perf -> q18a_alpha_c0, q18a-alpha-perf -> q18a-alpha-c0 (the band 2 manifest keeps its file name);
  budget   the draft reads alpha_c0_gate (ACTIVE_S is the 9,000 s box budget less cleanup, LAST_ARM_ENTRY_S the rule);
  window   none: alpha-perf's PY_SLOT wall-clock refusal is replaced by SLOT_NEW (operator, 2 Nov: launch time is the lead's);
  switch   FLASH_S5_STAGE_LOCAL=1 in EXTRA_ENV, asserted in declaration and preflight beside the kept dense assert;
  body     eight band 1 arms, ABBA, each behind a typed FULL; no optional arm; the grade is e3_alpha_c0_box.
Labels carry round 0 or 1, so every cache salt stays unique and the band follows from the index as in alpha-perf.
No model setting, gate threshold, launcher or restore change. --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import alpha_perf_queue as alpha  # noqa: E402

OLD, NEW = 'q18a_alpha_perf', 'q18a_alpha_c0'
OLD_Q, NEW_Q = 'q18a-alpha-perf', 'q18a-alpha-c0'
SWITCH = 'FLASH_S5_STAGE_LOCAL=1'
ARMS = ('serial-solo-0', 'local-solo-0', 'local-live-0', 'serial-live-0',
        'serial-live-1', 'local-live-1', 'local-solo-1', 'serial-solo-1')
HEADER_NOTE = ('# C0: this queue compares stage-local native progress with the frozen serial path on one engine, matched per arm by the\n'
               '# typed native_stage_local control: band 1 ABBA at 65,445 tokens. Mechanism and numerics gate (crash-free, engagement,\n'
               '# same generated tokens); speed and memory are reported; release readiness is not assessed.\n#\n')
BODY = r'''# C0: band 1 ABBA (serial-solo-0 .. serial-solo-1). Each arm selects
# burst and its serial or local path by typed control while paused idle FULL. Speed and memory are descriptive. Every
# timeout below is one alpha_c0_gate cap (instruments, arm_entries, alpha_c0_arms, ingress, end_full_idle, grade); each arm
# enters only inside LAST_ARM_ENTRY_S. _cid is the engine container id read before the unarmed check. Every engine-facing
# client runs under the le-fix engine watch.
timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
  --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 \
  --out "$Q/instruments-start.json" > "$Q/instruments-start.log" 2>&1
INSTRUMENTS_START=$?
record instruments-start "$INSTRUMENTS_START" 'replay observer library mapped in no rank process (observer and offers off)'
GRADE_ARGS=()
LAST_ARM_ENTRY_S=$(native41_budget last_arm_entry)
if [[ $INSTRUMENTS_START == 0 ]]; then
  for _label in serial-solo-0 local-solo-0 local-live-0 serial-live-0 serial-live-1 local-live-1 local-solo-1 serial-solo-1; do
    IFS=- read -r _mode _cohort _pair <<< "$_label"
    _workload=$H/workloads/q18a-native41-progress.json
    if (( SECONDS > LAST_ARM_ENTRY_S )); then
      record "arm-$_label" 2 "NOT_RUN: box budget, elapsed ${SECONDS}s > ${LAST_ARM_ENTRY_S}s"
      continue
    fi
    n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown
    if [[ $n != "$N0" ]] || host_crash_detected || ! health_ok; then
      record "arm-$_label" 2 'NOT_RUN: engine crashed, unhealthy or auto-restarted; no restart attempted'
      continue
    fi
    # The typed all-rank FULL takes the manual lease the previous arm released.
    engine_watched "entry-$_label" timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
      --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/entry-$_label.json" > "$Q/entry-$_label.log" 2>&1
    _entry_rc=$?; record "entry-$_label" "$_entry_rc" 'typed all-rank FULL, resumed, idle, workers empty; lease taken'
    if [[ $_entry_rc != 0 ]]; then
      record "arm-$_label" 2 'NOT_RUN: arm entry not a typed idle FULL'
      continue
    fi
    READINESS=$Q18A_READINESS GEN=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \
    S5_OVERLAY=$S5_OVERLAY S5_LAUNCHER_NAME=$S5_LAUNCHER_NAME \
    PACKAGE_MANIFEST=$S5_OVERLAY/SHA256SUMS GENERATION=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \
    engine_watched "arm-$_label" timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_alpha_c0_box.py" arm \
      --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH" \
      --mode "$_mode" --cohort "$_cohort" --index "$_pair" \
      --workload "$_workload" --control-timeout-s 60 \
      --out "$Q/arm-$_label.json" > "$Q/arm-$_label.log" 2>&1
    record "arm-$_label" "$?" 'auto arm: 8 standing decoders, burst plus the serial or local selection, fresh, append; live keeps them, solo cancels'
    GRADE_ARGS+=(--arm "$Q/arm-$_label.json" --watch "$_label=$Q/engine-watch-arm-$_label.json")
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
  engine_watched end-full-idle timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/end-full-idle.json" > "$Q/end-full-idle.log" 2>&1
  record end-full-idle "$?" 'typed all-rank FULL at the end: resumed, idle, workers empty'
fi
[[ ${#GRADE_ARGS[@]} -gt 0 ]] || GRADE_ARGS=(--arm "$Q/arm-none.json")
timeout 600s python3 "$H/e3_alpha_c0_box.py" grade "${GRADE_ARGS[@]}" \
  --check "instruments-start=$Q/instruments-start.json" --check "instruments-end=$Q/instruments-end.json" \
  --check "end-full-idle=$Q/end-full-idle.json" --check "unarmed-stock-prefill=$Q/unarmed-stock-prefill.json" \
  --log "$Q/engine-stream.log" --log "$Q/engine-stream.err" \
  --report "$Q/alpha-c0-report.md" --out "$Q/alpha-c0-grade.json" > "$Q/alpha-c0-grade.log" 2>&1
record alpha-c0 "$?" 'per-arm mechanism (crash-free, TP commitment, engagement), complete progress lines; speed and memory descriptive'
'''
BUDGET_OLD = 'from alpha_perf_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES\n'
# Alpha-perf's absolute-time slot, replaced: C0 has no wall-clock window (operator, 2 Nov: "we don't put in code what you
# govern"). The relative budget (the outer timeout and LAST_ARM_ENTRY_S, from the run's own start) is unchanged.
SLOT_OLD = ('# The owner supplies the UTC hard end at bundle build time. The queue refuses\n'
            '# entry unless its derived active timeout plus 900 s kill margin fits that window (native39_trunk_gate derivation).\n'
            '# Production restoration remains with the owner. No workload/gate changes.\n'
            "python3 - <<'PY_SLOT' || fail_early night-window-worst-case-exceeds-restore-margin\n"
            'import os,sys,time\n'
            "sys.path.insert(0,os.environ.get('HARNESS_DIR') or os.environ.get('H') or '')\n"
            'from alpha_perf_gate import window_ok\n'
            "assert window_ok(time.time()), 'queue outside owner window or cleanup margin'\n"
            'PY_SLOT\n')
SLOT_NEW = ('# C0: no absolute-time window (operator, 2 Nov): launch time is the lead\'s, and no start is refused by wall clock.\n'
            '# The time limits are relative to this run\'s own start: the outer timeout above (ACTIVE_S, then CLEANUP_S for owned\n'
            '# cleanup) and LAST_ARM_ENTRY_S before each arm. Production restoration remains with the owner.\n')
BUDGET_NEW = 'from alpha_c0_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES\n'
SETTINGS_DENSE = 'FLASH_S5_NATIVE_PROGRESS=burst ' + alpha.DENSE + ' '
SETTINGS_ANCHOR = '# Alpha-perf: FLASH_S5_NATIVE_QSA_POLICY=dense below,'
SETTINGS_NOTE = ('# C0: FLASH_S5_STAGE_LOCAL=1 below, the only setting added to alpha-perf; each arm selects serial or local by typed '
                 'control.\n')
DRAFT_SWITCH = ("assert env.get('FLASH_S5_STAGE_LOCAL') == '1', 'alpha-c0 stage-local switch'\n")
DECLARE_SWITCH = ("assert env.get('FLASH_S5_STAGE_LOCAL')=='1', 'alpha-c0 runs with the stage-local switch installed'\n")
INTERFACE_OLD = '  e3_alpha_perf_box.py alpha_perf_gate.py native43_engine_death.py native43_allocator_phase_deltas.py; do'
INTERFACE_NEW = ('  e3_alpha_perf_box.py alpha_perf_gate.py native43_engine_death.py native43_allocator_phase_deltas.py \\\n'
                 '  e3_alpha_c0_box.py alpha_c0_gate.py; do')
PINS_OLD = '      e3_alpha_perf_box.py alpha_perf_gate.py native43_engine_death.py native43_allocator_phase_deltas.py '
PINS_NEW = PINS_OLD + 'e3_alpha_c0_box.py alpha_c0_gate.py '
# Text kept verbatim through the name swap: the band 2 manifest's file name and its workload name.
KEEP = ('workloads/q18a-alpha-perf-band2.json',)


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:120], text.count(old))
    return text.replace(old, new)


def rename(text):
    held = {}
    for index, keep in enumerate(KEEP):
        held['\0%d\0' % index] = keep
        text = text.replace(keep, '\0%d\0' % index)
    text = text.replace(OLD, NEW).replace(OLD_Q, NEW_Q)
    for mark, keep in held.items():
        text = text.replace(mark, keep)
    return text


def derive():
    out = {}
    for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
        text = (ROOT / 'harness/runs' / (OLD + suffix)).read_text()
        if suffix == '.draft.sh':
            text = replace_once(text, alpha.BODY, BODY)
            text = replace_once(text, BUDGET_OLD, BUDGET_NEW)
            text = replace_once(text, SLOT_OLD, SLOT_NEW)
            text = replace_once(text, INTERFACE_OLD, INTERFACE_NEW)
            old = alpha.DRAFT_DENSE.splitlines(True)[0]
            text = replace_once(text, old, old + DRAFT_SWITCH)
            for a, b in (("perf=p.parent/'alpha-perf-grade.json'", "perf=p.parent/'alpha-c0-grade.json'"),
                         ("'missing':['alpha-perf-grade-absent']", "'missing':['alpha-c0-grade-absent']"),
                         ('Every cap is alpha_perf_gate.PHASES.', 'Every cap is alpha_c0_gate.PHASES.'),
                         ('# number. Speed is descriptive. Every cap is alpha_perf_gate.\n',
                          '# number. Speed is descriptive. Alpha-perf text above; every cap here is alpha_c0_gate.\n')):
                text = replace_once(text, a, b)
            first = text.index('# Alpha-perf: this queue measures')
            text = text[:first] + HEADER_NOTE + text[first:]
        elif suffix == '_settings.sh':
            text = replace_once(text, SETTINGS_DENSE, SETTINGS_DENSE + SWITCH + ' ')
            text = replace_once(text, SETTINGS_ANCHOR, SETTINGS_NOTE + SETTINGS_ANCHOR)
        else:
            old = alpha.DECLARE_DENSE.splitlines(True)[0]
            text = replace_once(text, old, old + DECLARE_SWITCH)
            text = replace_once(text, PINS_OLD, PINS_NEW)
        out['harness/runs/' + NEW + suffix] = rename(text)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    for name, text in derive().items():
        path = ROOT / name
        if a.check:
            assert path.read_text() == text, name
        else:
            path.write_text(text)
            path.chmod(0o664)
    print('alpha-c0 queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
