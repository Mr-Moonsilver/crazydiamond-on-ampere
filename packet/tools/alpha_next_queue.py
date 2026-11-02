#!/usr/bin/env python3
"""Derive the q18a-alpha-next box scripts from the committed q18a-step-b texts. Nothing a past queue ran is edited.

Kept from Step B, byte for byte: pins, preflight, startup, bind, host headroom, the unarmed check, the end identity, both band
manifests and their /tokenize guards, the le-fix death scan and watch, the M3 allocator receipts and the whole owned cleanup and
restore tail. Declared deltas, each applied exactly once:
  names      q18a_step_b -> q18a_alpha_next, q18a-step-b -> q18a-alpha-next (the band 2 manifest keeps its file name);
  budget     the draft reads alpha_next_gate: relative to the run's own start only (ACTIVE_S, LAST_ARM_ENTRY_S for core arms,
             LAST_TAIL_ENTRY_S for the g4 tail); Step B's size 4 repeat trim is gone (the table orders g4 last);
  window     none (Step B's text already says so; restated for alpha-next);
  switches   FLASH_S5_STAGE_LOCAL=1 and FLASH_S5_HANDLER_ANATOMY=1 in EXTRA_ENV after FLASH_S5_LAYER_GROUPS=1, asserted in
             declaration and preflight, with no FLASH_S5_PROBE_* key and FLASH_S5_UNION_RETIRE still unset;
  stop race  host_crash_detected runs alpha_next_stop_race.py when the frozen death scan says DEAD: the declared STOP_RACE class
             (lead, 2 Nov 13:30) is recorded and counted, engine_crashed stays false; anything else stays a crash;
  pins       the overlay manifest and package pins, PLACEHOLDER until the packaging step fills tools/alpha_next_queue_pins.json
             and re-derives (a placeholder refuses the declaration);
  body       the core arms (alpha_next_arms, parts 'core' and 'extra'), each behind a typed FULL; a typed FULL and the core
             grade, the queue's verdict; then the g4 tail (part 'tail'), reads only, each behind its own typed FULL; instruments-end and the end
             FULL (counted); the read over every arm (not counted).
Labels carry the C0 selection, the size, the cohort and the round (``c0on.g2-live-0``); the band follows from the round index
as in Step A, and the label is in every cache salt. No model setting, gate threshold, launcher or restore change. The arm lists
come from harness/alpha_next_arms.py; --check verifies the files.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import step_b_queue as step_b  # noqa: E402
import alpha_next_arms as table  # noqa: E402

OLD, NEW = step_b.NEW, 'q18a_alpha_next'
OLD_Q, NEW_Q = step_b.NEW_Q, 'q18a-alpha-next'
SWITCHES = 'FLASH_S5_STAGE_LOCAL=1 FLASH_S5_HANDLER_ANATOMY=1'
CORE = tuple('%s:%s' % (r['part'], r['label']) for r in table.arms(table.GRADED))
TAIL = tuple(table.labels('tail'))
PINS_FILE = ROOT / 'tools/alpha_next_queue_pins.json'
PLACEHOLDER = 'PLACEHOLDER-filled-by-the-packaging-step'
HEADER_NOTE = ('# alpha-next: this queue compares C0 stage-local native progress with the serial path on the composed Step A + Step B\n'
               '# engine (fused in every arm, one layer group size per arm), matched per arm by the typed native_stage_local control:\n'
               '# a core of %d arms (ABBA C0 on/off, live and solo, band 1 at size 2 over %d rounds, band 2 at size 2), then %d extra\n'
               '# band 1 arms if the budget allows, whose grade is the queue verdict, then a g4 tail of %d C0-on arms, reads only.\n'
               '# Mechanism gates (crash-free, TP commitment, group and C0 engagement); numerics on the greedy standing streams, speed,\n'
               '# decode and memory are reported; release readiness is not assessed.\n#\n' % (
                   len(table.arms('core')), table.BAND1_REQUIRED_ROUNDS, len(table.arms('extra')), len(TAIL)))

BODY = r'''# alpha-next: the core arms (CORE below) then, once the core grade is written, the g4 tail (TAIL below, reads only). Each arm
# selects burst, fused, its group size and its C0 selection by typed control while paused idle FULL. Speed, decode and memory are
# descriptive. Every timeout below is one alpha_next_gate cap (instruments, arm_entries, alpha_next_arms, ingress,
# core_full_idle, grade, end_full_idle, read); a core arm enters only inside LAST_ARM_ENTRY_S, a tail arm only inside
# LAST_TAIL_ENTRY_S, both counted from this run's own start (bash SECONDS). _cid is the engine container id read before the
# unarmed check. Every engine-facing client runs under the le-fix engine watch.
# A tail step is a read: read_step prints and retains it and never counts it. A tail arm still counts for crash accounting (the
# engine watch, the death scan and the health checks set ENGINE_CRASHED) and for leak accounting (instruments-end and the end
# FULL run after the tail and are counted). An extra arm (band 1's round past the required two) runs only while the core entry
# limit allows; once entered it counts as a core arm, and past the limit its NOT_RUN is a read.
read_step() {
  local v
  case $2 in (0) v=PASS;; (2) v=INCONCLUSIVE;; (*) v="FAIL rc=$2";; esac
  PHASE_VERDICT[$1]="READ-${v%% *}"
  echo "STEP $1 READ-$v ${3:-}"
}
alpha_next_arm() {
  local _part=$1 _label=$2 _rec=record _skip=record _limit=$LAST_ARM_ENTRY_S _sel _cohort _pair _c0 _group _workload _entry_rc
  local _span _s _e
  if [[ $_part == tail ]]; then _rec=read_step; _skip=read_step; _limit=$LAST_TAIL_ENTRY_S; fi
  [[ $_part != extra ]] || _skip=read_step
  IFS=- read -r _sel _cohort _pair <<< "$_label"
  IFS=. read -r _c0 _group <<< "$_sel"
  if [[ $_pair == @BAND1@ ]]; then _workload=$H/workloads/q18a-native41-progress.json
  else _workload=$H/workloads/q18a-alpha-perf-band2.json; fi
  if (( SECONDS > _limit )); then
    "$_skip" "arm-$_label" 2 "NOT_RUN: box budget, elapsed ${SECONDS}s > ${_limit}s"
    return 0
  fi
  n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown
  if [[ $n != "$N0" ]] || host_crash_detected || ! health_ok; then
    "$_rec" "arm-$_label" 2 'NOT_RUN: engine crashed, unhealthy or auto-restarted; no restart attempted'
    return 0
  fi
  # The typed all-rank FULL takes the manual lease the previous arm released.
  engine_watched "entry-$_label" timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/entry-$_label.json" > "$Q/entry-$_label.log" 2>&1
  _entry_rc=$?; "$_rec" "entry-$_label" "$_entry_rc" 'typed all-rank FULL, resumed, idle, workers empty; lease taken'
  if [[ $_entry_rc != 0 ]]; then
    "$_rec" "arm-$_label" 2 'NOT_RUN: arm entry not a typed idle FULL'
    return 0
  fi
  READINESS=$Q18A_READINESS GEN=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \
  S5_OVERLAY=$S5_OVERLAY S5_LAUNCHER_NAME=$S5_LAUNCHER_NAME \
  PACKAGE_MANIFEST=$S5_OVERLAY/SHA256SUMS GENERATION=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \
  engine_watched "arm-$_label" timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_alpha_next_box.py" arm \
    --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --stage-local "${_c0#c0}" --size "${_group#g}" --cohort "$_cohort" --index "$_pair" \
    --workload "$_workload" --control-timeout-s 60 \
    --out "$Q/arm-$_label.json" > "$Q/arm-$_label.log" 2>&1
  "$_rec" "arm-$_label" "$?" 'auto arm: 8 standing decoders, burst, fused, the group size and the C0 selection, fresh, append; live keeps them, solo cancels'
  [[ $_part == tail ]] || GRADE_ARGS+=(--arm "$Q/arm-$_label.json" --watch "$_label=$Q/engine-watch-arm-$_label.json")
  READ_ARGS+=(--arm "$Q/arm-$_label.json" --watch "$_label=$Q/engine-watch-arm-$_label.json")
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
    "$_rec" "ingress-$_label" "$?" 'access log: every request in the arm span is the arm (8 standing, fresh, append), else INVALID'
  else
    "$_rec" "ingress-$_label" 2 'arm span missing; ingress not checked'
  fi
  [[ $_part == tail ]] || GRADE_ARGS+=(--ingress "$_label=$Q/ingress-$_label.json")
  READ_ARGS+=(--ingress "$_label=$Q/ingress-$_label.json")
  return 0
}
timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
  --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 \
  --out "$Q/instruments-start.json" > "$Q/instruments-start.log" 2>&1
INSTRUMENTS_START=$?
record instruments-start "$INSTRUMENTS_START" 'replay observer library mapped in no rank process (observer and offers off)'
GRADE_ARGS=(); READ_ARGS=()
LAST_ARM_ENTRY_S=$(native41_budget last_arm_entry)
LAST_TAIL_ENTRY_S=$(native41_budget last_tail_entry)
if [[ $INSTRUMENTS_START == 0 ]]; then
  for _arm in @CORE@; do
    alpha_next_arm "${_arm%%:*}" "${_arm#*:}"
  done
  n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown
  if [[ $n != "$N0" ]] || ! health_ok; then
    ENGINE_CRASHED=1; record arms-engine 1 'engine unhealthy or auto-restarted; no restart attempted'
  else
    record arms-engine 0 'engine healthy with unchanged NRestarts'
    engine_watched core-full-idle timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
      --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/core-full-idle.json" > "$Q/core-full-idle.log" 2>&1
    record core-full-idle "$?" 'typed all-rank FULL after the core arms: resumed, idle, workers empty'
  fi
else
  record arms 2 'NOT_RUN: observer and offers off not proven on every rank process (instruments-start)'
fi
# The core grade is the queue's verdict and is written before any tail arm starts.
[[ ${#GRADE_ARGS[@]} -gt 0 ]] || GRADE_ARGS=(--arm "$Q/arm-none.json")
timeout 600s python3 "$H/e3_alpha_next_box.py" grade "${GRADE_ARGS[@]}" \
  --check "instruments-start=$Q/instruments-start.json" --check "core-full-idle=$Q/core-full-idle.json" \
  --check "unarmed-stock-prefill=$Q/unarmed-stock-prefill.json" \
  --log "$Q/engine-stream.log" --log "$Q/engine-stream.err" \
  --report "$Q/alpha-next-report.md" --out "$Q/alpha-next-grade.json" > "$Q/alpha-next-grade.log" 2>&1
record alpha-next "$?" 'core arms: per-arm mechanism (crash-free, TP commitment, group and C0 engagement), numerics envelope, complete progress lines; speed, decode and memory descriptive'
# The g4 tail: reads only, after the core grade, while the engine is healthy.
if [[ $INSTRUMENTS_START == 0 && -s $Q/alpha-next-grade.json && $ENGINE_CRASHED == 0 ]] && health_ok; then
  for _label in @TAIL@; do
    alpha_next_arm tail "$_label"
  done
  n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown
  if [[ $n != "$N0" ]] || ! health_ok; then
    ENGINE_CRASHED=1; record tail-engine 1 'engine unhealthy or auto-restarted after the g4 tail; no restart attempted'
  else
    record tail-engine 0 'engine healthy with unchanged NRestarts after the g4 tail'
  fi
else
  read_step tail 2 'NOT_RUN: no core grade, instruments-start not proven, or the engine crashed or is unhealthy'
fi
if [[ $INSTRUMENTS_START == 0 ]] && health_ok; then
  timeout 120s python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --container-id "${_cid:-none}" --control-timeout-s 30 --exec-timeout-s 10 --same-as "$Q/instruments-start.json" \
    --out "$Q/instruments-end.json" > "$Q/instruments-end.log" 2>&1
  record instruments-end "$?" 'after every arm, the g4 tail included: the same rank processes, replay observer mapped in none'
  engine_watched end-full-idle timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \
    --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/end-full-idle.json" > "$Q/end-full-idle.log" 2>&1
  record end-full-idle "$?" 'typed all-rank FULL at the end, after the g4 tail: resumed, idle, workers empty'
fi
[[ ${#READ_ARGS[@]} -gt 0 ]] || READ_ARGS=(--arm "$Q/arm-none.json")
timeout 600s python3 "$H/e3_alpha_next_box.py" read "${READ_ARGS[@]}" \
  --check "instruments-start=$Q/instruments-start.json" --check "instruments-end=$Q/instruments-end.json" \
  --check "end-full-idle=$Q/end-full-idle.json" --check "unarmed-stock-prefill=$Q/unarmed-stock-prefill.json" \
  --log "$Q/engine-stream.log" --log "$Q/engine-stream.err" \
  --report "$Q/alpha-next-read.md" --out "$Q/alpha-next-read.json" > "$Q/alpha-next-read.log" 2>&1
read_step alpha-next-read "$?" 'every arm, the g4 tail included: the same rules as the core grade, a read, never the queue verdict'
'''


def body():
    return BODY.replace('@CORE@', ' '.join(CORE)).replace('@TAIL@', ' '.join(TAIL)).replace('@BAND1@', table.band1_glob())


BUDGET_OLD = step_b.BUDGET_NEW
BUDGET_NEW = 'from alpha_next_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,LAST_TAIL_ENTRY_S,PHASES\n'
PRINT_OLD = step_b.PRINT_NEW
PRINT_NEW = ("else LAST_ARM_ENTRY_S if sys.argv[1]=='last_arm_entry' else LAST_TAIL_ENTRY_S if sys.argv[1]=='last_tail_entry' "
             "else PHASES[sys.argv[1]])\n")
WINDOW_OLD = step_b.WINDOW_NEW
WINDOW_NEW = (
    "# Launch time is governed by the lead at launch (operator, 2 Nov 2026); no absolute window or hard end in this queue. The run's\n"
    "# budget is relative to its own start: the watchdog and the core and tail arm entry limits (alpha_next_gate, bash SECONDS).\n"
    "# Production restoration remains with the owner. No workload/gate changes.\n")
INTERFACE_OLD = step_b.INTERFACE_NEW
INTERFACE_NEW = ('  e3_step_a_box.py step_a_gate.py e3_step_b_box.py step_b_gate.py \\\n'
                 '  e3_alpha_next_box.py alpha_next_gate.py alpha_next_arms.py alpha_next_stop_race.py; do')
PINS_OLD = step_b.PINS_NEW
PINS_NEW = PINS_OLD + 'e3_alpha_next_box.py alpha_next_gate.py alpha_next_arms.py alpha_next_stop_race.py '
SETTINGS_OLD = step_b.SETTINGS_FUSED + step_b.GROUPS + ' '
SETTINGS_ANCHOR = '# Step B: FLASH_S5_LAYER_GROUPS=1 below,'
SETTINGS_NOTE = ('# alpha-next: FLASH_S5_STAGE_LOCAL=1 (C0) and FLASH_S5_HANDLER_ANATOMY=1 (stage-1 anatomy lines) below, the only\n'
                 '# settings added to Step B; each arm selects C0 on or off by typed control. FLASH_S5_UNION_RETIRE stays unset and no\n'
                 '# FLASH_S5_PROBE_* key is set.\n')
DRAFT_SWITCHES = ("assert env.get('FLASH_S5_STAGE_LOCAL') == '1', 'alpha-next stage-local switch (C0)'\n"
                  "assert env.get('FLASH_S5_HANDLER_ANATOMY') == '1', 'alpha-next stage-1 handler anatomy switch'\n"
                  "assert not [k for k in env if k.startswith('FLASH_S5_PROBE_')], 'alpha-next: no probe switch'\n")
DECLARE_SWITCHES = ("assert env.get('FLASH_S5_STAGE_LOCAL')=='1', 'alpha-next runs with the stage-local switch installed'\n"
                    "assert env.get('FLASH_S5_HANDLER_ANATOMY')=='1', 'alpha-next runs with the handler anatomy switch installed'\n"
                    "assert not [k for k in env if k.startswith('FLASH_S5_PROBE_')], 'alpha-next: no probe switch'\n")
CRASH_OLD = (
    "  if [[ $? == 1 ]]; then\n"
    "    ENGINE_CRASHED=1; ENGINE_CRASH_SOURCE=engine-death-scan\n"
    "    return 0\n"
    "  fi\n"
    "  return 1\n"
    "}\n")
CRASH_NEW = (
    "  if [[ $? == 1 ]]; then\n"
    "    # alpha-next STOP_RACE (lead, 2 Nov 2026 13:30): one output-handler EngineDeadError after the queue's stop intent and\n"
    "    # within 500 ms after 'shutdown triggered', no other hard marker, no worker failure record, the end FULL all empty.\n"
    "    # Recorded and counted; engine_crashed stays false. Anything else stays a crash (alpha_next_stop_race.py).\n"
    "    python3 \"${H:-}/alpha_next_stop_race.py\" --log \"$Q/engine-stream.log\" --log \"$Q/engine-stream.err\" \\\n"
    "      --stop-intent \"$Q/host-stop-intent.json\" --end-full-idle \"$Q/end-full-idle.json\" --out \"$Q/stop-race.json\" \\\n"
    "      >> \"$Q/stop-race.log\" 2>&1\n"
    "    if [[ $? == 0 ]]; then\n"
    "      STOP_RACE=1; PHASE_VERDICT[stop-race]=STOP_RACE\n"
    "      return 1\n"
    "    fi\n"
    "    ENGINE_CRASHED=1; ENGINE_CRASH_SOURCE=engine-death-scan\n"
    "    return 0\n"
    "  fi\n"
    "  return 1\n"
    "}\n")
STATE_OLD = "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; N0=''\n"
STATE_NEW = STATE_OLD + 'STOP_RACE=0\n'
FINAL_READ_OLD = "  rule4_continuation=None\n"
FINAL_READ_NEW = (
    "  stop_race=None\n"
    "  race=p.parent/'stop-race.json'\n"
    "  if race.is_file():\n"
    "   try:stop_race=json.load(race.open())\n"
    "   except (OSError,ValueError):stop_race={'stop_race':False,'unavailable':'record-unreadable'}\n"
    "  read=None\n"
    "  tail=p.parent/'alpha-next-read.json'\n"
    "  if tail.is_file():\n"
    "   try:read={k:v for k,v in json.load(tail.open()).items() if k in ('schema','verdict','failures','missing','read_only')}\n"
    "   except (OSError,ValueError):read={'verdict':'INCONCLUSIVE','unavailable':'read-unreadable'}\n"
    "  rule4_continuation=None\n")
FINAL_DICT_OLD = "engine_crash_source=sys.argv[9] or None,"
FINAL_DICT_NEW = ("engine_crash_source=sys.argv[9] or None,stop_race=(stop_race or {}).get('stop_race') is True,"
                  "stop_race_record=stop_race,alpha_next_read=read,")
PINS_ANCHOR = '{\n  for p in "$R/startup.json"'


def pins():
    data = json.loads(PINS_FILE.read_text())
    return data['overlay_manifest_sha256'], data['package_sha256']


def pins_block():
    manifest, package = pins()
    return (
        "# alpha-next bundle and manifest pins, from tools/alpha_next_queue_pins.json. PLACEHOLDER until the packaging step fills\n"
        "# that file and re-derives this text (tools/alpha_next_queue.py); a placeholder refuses the declaration. The harness bundle\n"
        "# pin is held outside the bundle (the harness SHA256SUMS covers this text, so it cannot hold its own digest).\n"
        "ALPHA_NEXT_OVERLAY_MANIFEST_SHA256=%s\n"
        "ALPHA_NEXT_PACKAGE_SHA256=%s\n"
        "[[ $ALPHA_NEXT_OVERLAY_MANIFEST_SHA256 =~ ^[0-9a-f]{64}$ && $ALPHA_NEXT_PACKAGE_SHA256 =~ ^[0-9a-f]{64}$ ]] ||\n"
        "  fail \"alpha-next bundle pins are placeholders (filled by the packaging step)\"\n"
        "[[ $(sha256sum \"$S5_OVERLAY/SHA256SUMS\" | cut -c1-64) == \"$ALPHA_NEXT_OVERLAY_MANIFEST_SHA256\" ]] ||\n"
        "  fail \"overlay manifest differs from the alpha-next pin\"\n"
        "python3 - \"$S5_OVERLAY/S5-OVERLAP-GENERATION.json\" \"$ALPHA_NEXT_PACKAGE_SHA256\" <<'PY_ALPHA_NEXT_PIN' || "
        "fail \"overlay package differs from the alpha-next pin\"\n"
        "import json,sys\n"
        "assert json.load(open(sys.argv[1])).get('package_sha256')==sys.argv[2]\n"
        "PY_ALPHA_NEXT_PIN\n" % (manifest, package))


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:120], text.count(old))
    return text.replace(old, new)


def rename(text):
    held = {}
    for index, keep in enumerate(step_b.KEEP):
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
            text = replace_once(text, step_b.BODY, body())
            text = replace_once(text, BUDGET_OLD, BUDGET_NEW)
            text = replace_once(text, PRINT_OLD, PRINT_NEW)
            text = replace_once(text, WINDOW_OLD, WINDOW_NEW)
            text = replace_once(text, INTERFACE_OLD, INTERFACE_NEW)
            text = replace_once(text, step_b.DRAFT_GROUPS, step_b.DRAFT_GROUPS + DRAFT_SWITCHES)
            text = replace_once(text, CRASH_OLD, CRASH_NEW)
            text = replace_once(text, STATE_OLD, STATE_NEW)
            text = replace_once(text, FINAL_READ_OLD, FINAL_READ_NEW)
            text = replace_once(text, FINAL_DICT_OLD, FINAL_DICT_NEW)
            for a, b in (("perf=p.parent/'step-b-grade.json'", "perf=p.parent/'alpha-next-grade.json'"),
                         ("'missing':['step-b-grade-absent']", "'missing':['alpha-next-grade-absent']"),
                         ('Every cap is step_b_gate.PHASES.', 'Every cap is alpha_next_gate.PHASES.'),
                         ('Step A text above; every cap here is step_b_gate.\n',
                          'Step A and Step B text above; every cap here is alpha_next_gate.\n')):
                text = replace_once(text, a, b)
            first = text.index('# Step B: this queue compares')
            text = text[:first] + HEADER_NOTE + text[first:]
        elif suffix == '_settings.sh':
            text = replace_once(text, SETTINGS_OLD, SETTINGS_OLD + SWITCHES + ' ')
            text = replace_once(text, SETTINGS_ANCHOR, SETTINGS_NOTE + SETTINGS_ANCHOR)
        else:
            text = replace_once(text, step_b.DECLARE_GROUPS, step_b.DECLARE_GROUPS + DECLARE_SWITCHES)
            text = replace_once(text, PINS_OLD, PINS_NEW)
            text = replace_once(text, PINS_ANCHOR, pins_block() + PINS_ANCHOR)
        out['harness/runs/' + NEW + suffix] = rename(text)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--check', action='store_true')
    a = p.parse_args(argv)
    for name, text in derive().items():
        path = ROOT / name
        if a.check:
            assert path.read_text() == text, name
        else:
            path.write_text(text)
            path.chmod(0o664)
    print('alpha-next queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
