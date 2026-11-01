#!/usr/bin/env python3
"""Derive the q18a-step-a box scripts from the committed alpha-perf (40a4cea) texts. Nothing a past queue ran is edited.

Kept from alpha-perf, byte for byte: pins, preflight, startup, bind, host headroom, the unarmed check, the end identity,
both band manifests and their /tokenize guards, the le-fix death scan and watch, the M3 allocator receipts and the whole
owned cleanup/restore tail. Declared deltas, each applied exactly once:
  names    q18a_alpha_perf -> q18a_step_a, q18a-alpha-perf -> q18a-step-a (the band 2 manifest keeps its file name);
  budget   draft and window read step_a_gate (ACTIVE_S is the 9,000 s box budget less cleanup, LAST_ARM_ENTRY_S the rule);
  switch   FLASH_S5_QSA_FUSED=1 in EXTRA_ENV, asserted in declaration and preflight beside the kept dense assert;
  body     twelve arms in the lead's order, each behind a typed FULL; no optional arm; the grade is e3_step_a_box.
Band 1 labels round 0 and 1; band 2 uses round index 2 in its labels, so every cache salt stays unique and the band follows
from the index as in alpha-perf. No model setting, gate threshold, launcher or restore change. --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import alpha_perf_queue as alpha  # noqa: E402

OLD, NEW = 'q18a_alpha_perf', 'q18a_step_a'
OLD_Q, NEW_Q = 'q18a-alpha-perf', 'q18a-step-a'
FUSED = 'FLASH_S5_QSA_FUSED=1'
ARMS_BAND1 = ('stepped-solo-0', 'fused-solo-0', 'fused-live-0', 'stepped-live-0',
              'stepped-live-1', 'fused-live-1', 'fused-solo-1', 'stepped-solo-1')
ARMS_BAND2 = ('fused-live-2', 'stepped-live-2', 'stepped-solo-2', 'fused-solo-2')
ARMS = ARMS_BAND1 + ARMS_BAND2
HEADER_NOTE = ('# Step A: this queue compares one bounded dense QSA command per layer (fused) with the stepped protocol on one '
               'engine, matched\n# per arm by the typed native_qsa_fused control: band 1 ABBA at 65,445 tokens, band 2 at '
               '131,094 tokens past the 97,152-row\n# arena. Mechanism gates (crash-free, TP commitment, engagement); speed and '
               'memory are reported; release readiness is not assessed.\n#\n')
BODY = r'''# Step A: band 1 ABBA (stepped-solo-0 .. stepped-solo-1) then band 2 matched (fused-live-2 .. fused-solo-2). Each arm selects
# burst and its fused or stepped path by typed control while paused idle FULL. Speed and memory are descriptive. Every
# timeout below is one step_a_gate cap (instruments, arm_entries, step_a_arms, ingress, end_full_idle, grade); each arm
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
  for _label in stepped-solo-0 fused-solo-0 fused-live-0 stepped-live-0 stepped-live-1 fused-live-1 fused-solo-1 stepped-solo-1 fused-live-2 stepped-live-2 stepped-solo-2 fused-solo-2; do
    IFS=- read -r _mode _cohort _pair <<< "$_label"
    if [[ $_pair == [01] ]]; then _workload=$H/workloads/q18a-native41-progress.json
    else _workload=$H/workloads/q18a-alpha-perf-band2.json; fi
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
    engine_watched "arm-$_label" timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_step_a_box.py" arm \
      --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH" \
      --mode "$_mode" --cohort "$_cohort" --index "$_pair" \
      --workload "$_workload" --control-timeout-s 60 \
      --out "$Q/arm-$_label.json" > "$Q/arm-$_label.log" 2>&1
    record "arm-$_label" "$?" 'auto arm: 8 standing decoders, burst plus the fused or stepped selection, fresh, append; live keeps them, solo cancels'
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
timeout 600s python3 "$H/e3_step_a_box.py" grade "${GRADE_ARGS[@]}" \
  --check "instruments-start=$Q/instruments-start.json" --check "instruments-end=$Q/instruments-end.json" \
  --check "end-full-idle=$Q/end-full-idle.json" --check "unarmed-stock-prefill=$Q/unarmed-stock-prefill.json" \
  --log "$Q/engine-stream.log" --log "$Q/engine-stream.err" \
  --report "$Q/step-a-report.md" --out "$Q/step-a-grade.json" > "$Q/step-a-grade.log" 2>&1
record step-a "$?" 'per-arm mechanism (crash-free, TP commitment, engagement), complete progress lines; speed and memory descriptive'
'''
BUDGET_OLD = 'from alpha_perf_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES\n'
BUDGET_NEW = 'from step_a_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES\n'
SETTINGS_DENSE = 'FLASH_S5_NATIVE_PROGRESS=burst ' + alpha.DENSE + ' '
SETTINGS_ANCHOR = '# Alpha-perf: FLASH_S5_NATIVE_QSA_POLICY=dense below,'
SETTINGS_NOTE = ('# Step A: FLASH_S5_QSA_FUSED=1 below, the only setting added to alpha-perf; each arm selects fused or stepped by typed '
                 'control.\n')
DRAFT_FUSED = ("assert env.get('FLASH_S5_QSA_FUSED') == '1', 'step-a fused switch'\n")
DECLARE_FUSED = ("assert env.get('FLASH_S5_QSA_FUSED')=='1', 'step-a runs with the fused switch installed'\n")
INTERFACE_OLD = '  e3_alpha_perf_box.py alpha_perf_gate.py native43_engine_death.py native43_allocator_phase_deltas.py; do'
INTERFACE_NEW = ('  e3_alpha_perf_box.py alpha_perf_gate.py native43_engine_death.py native43_allocator_phase_deltas.py \\\n'
                 '  e3_step_a_box.py step_a_gate.py; do')
PINS_OLD = '      e3_alpha_perf_box.py alpha_perf_gate.py native43_engine_death.py native43_allocator_phase_deltas.py '
PINS_NEW = PINS_OLD + 'e3_step_a_box.py step_a_gate.py '
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
            text = replace_once(text, 'from alpha_perf_gate import window_ok', 'from step_a_gate import window_ok')
            text = replace_once(text, INTERFACE_OLD, INTERFACE_NEW)
            old = alpha.DRAFT_DENSE.splitlines(True)[0]
            text = replace_once(text, old, old + DRAFT_FUSED)
            for a, b in (("perf=p.parent/'alpha-perf-grade.json'", "perf=p.parent/'step-a-grade.json'"),
                         ("'missing':['alpha-perf-grade-absent']", "'missing':['step-a-grade-absent']"),
                         ('Every cap is alpha_perf_gate.PHASES.', 'Every cap is step_a_gate.PHASES.'),
                         ('# number. Speed is descriptive. Every cap is alpha_perf_gate.\n',
                          '# number. Speed is descriptive. Alpha-perf text above; every cap here is step_a_gate.\n')):
                text = replace_once(text, a, b)
            first = text.index('# Alpha-perf: this queue measures')
            text = text[:first] + HEADER_NOTE + text[first:]
        elif suffix == '_settings.sh':
            text = replace_once(text, SETTINGS_DENSE, SETTINGS_DENSE + FUSED + ' ')
            text = replace_once(text, SETTINGS_ANCHOR, SETTINGS_NOTE + SETTINGS_ANCHOR)
        else:
            old = alpha.DECLARE_DENSE.splitlines(True)[0]
            text = replace_once(text, old, old + DECLARE_FUSED)
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
    print('step-a queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
