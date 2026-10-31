#!/usr/bin/env python3
"""Generate the q18a-native35-p queue from the frozen native35 O texts plus native28's measurement wiring.

P = O under a fresh name with: the native33-form real row4 placement (O's
causal --arm O branch needs FLASH_NATIVE34_DIAG=1 and never qualifies row4;
the W2 gates need a qualified row4), diagnostics off exactly as L
(FLASH_NATIVE34_DIAG=0), native28's declared 1200 s first-advance read bound
(the only value e3_native_performance accepts; O's 480 s is a row4-only
leftover that W2 refuses), native28's same-queue qualification -> W1 ->
gates.json block verbatim, two retained W2 rounds with the round salt
namespace, per-round unchanged memory grade, the harness-only aggregate grade
into queue-grade.json.measured_performance, and P's own declared window.
Every replacement must match exactly once; nothing else in the texts changes.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'harness/runs'

O_HEADER=('# Native30-e: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 -> owned restore.\n'
          '# Inherited cleanup machinery below; later qualification/performance phases are absent.\n')
P_HEADER=('# Native35-p: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 -> W1 -> two retained W2 rounds -> owned restore.\n'
          '# Inherited cleanup machinery below; lifecycle/leak (L) and canary phases are absent.\n')
O_TIMEOUT='  exec timeout --signal=TERM --kill-after=420s 2100s env Q18A_OVERLAP_BOUNDED=1 bash "$0" "$@"\n'
P_TIMEOUT='  exec timeout --signal=TERM --kill-after=900s 14400s env Q18A_OVERLAP_BOUNDED=1 bash "$0" "$@"\n'
O_SLOT_NOTE='# entry unless its 2100 s timeout plus 420 s kill margin fits that window.\n'
P_SLOT_NOTE='# entry unless its 14400 s timeout plus 900 s kill margin fits that window (native35p_gate derivation).\n'
O_SLOT_IMPORT='from native34_gate import window_ok\n'
P_SLOT_IMPORT='from native35p_gate import speed_window_ok as window_ok\n'
O_ROW4_BRANCH=('  if [[ $name == row4 ]]; then\n'
    '    python3 "$H/e3_native34_causal.py" --url "$PROBE_URL" --model "$MODEL" \\\n'
    '      --run-id "$RUN_ID" --epoch "$EPOCH" --out "$Q/qual-row4.json" \\\n'
    '      --store "$Q18A_STORE" --workload "$H/workloads/q17b-correctness-precontrols.json" \\\n'
    '      --prefill-plan "$H/workloads/q18a-native-placement-prefill.json" \\\n'
    '      --snapshot-max-bytes "$SNAPSHOT_MAX_BYTES" --timeout-s 90 --arm O \\\n'
    '      > "$Q/qual-row4.log" 2>&1\n'
    '  else\n')
O_INTERFACES=('  e3_q18a_lifecycle.py e3_auth_exec.py pp_release_grade.py pp_canary_bridge.py \\\n'
              '  pressure2.py pressure_wait.py leak_partial_tail.py leaktest.py; do\n')
P_INTERFACES=('  e3_q18a_lifecycle.py e3_auth_exec.py pp_release_grade.py pp_canary_bridge.py \\\n'
              '  pressure2.py pressure_wait.py leak_partial_tail.py leaktest.py \\\n'
              '  e3_native35_performance.py native35_performance_grade.py native35p_gate.py; do\n')
O_FINAL_METRICS=("  metrics=None\n"
    "  perf=p.parent/'overlap.json'\n"
    "  if perf.is_file():\n"
    "   try:metrics=json.load(open(perf)).get('grade')\n"
    "   except (OSError,ValueError):metrics={'verdict':'INCONCLUSIVE','missing':['performance-artifact-unreadable']}\n")
P_FINAL_METRICS=("  metrics={'verdict':'NOT_RUN','missing':['native35-performance-grade-absent'],'target_is_gate':False}\n"
    "  perf=p.parent/'native35-performance-grade.json'\n"
    "  if perf.is_file():\n"
    "   try:metrics=json.load(open(perf))\n"
    "   except (OSError,ValueError):metrics={'verdict':'INCONCLUSIVE','missing':['performance-artifact-unreadable'],'target_is_gate':False}\n")
O_STOP=("# Stop after native28 row4; no W1/performance/lifecycle queue follows.\n"
    'rc=2\n'
    'case ${PHASE_VERDICT[row4]:-INCONCLUSIVE} in PASS) rc=0;; FAIL) rc=1;; esac\n'
    'if [[ $rc == 0 ]]; then\n'
    '  capture_end "$Q18A_IDENTITY_TIMEOUT_S" || rc=$?\n'
    'else\n'
    "  step engine.identity-end 2 'not run: row4 did not complete; stop/restore takes priority'\n"
    'fi\n'
    'EVIDENCE_DONE=1\n'
    'DONE=1\n'
    'exit "$rc"\n')
P_MEASUREMENT=('# Native35-p: two retained fixed W2 schedules after the unchanged same-queue\n'
    '# gates. The second round is part of the declared schedule, not a retry: it\n'
    '# runs whatever the first round graded, unless the engine is no longer\n'
    '# healthy. Each round uses its own cache-salt namespace; both are retained.\n'
    'if [[ $gate_rc == 0 ]]; then\n'
    '  for _round in 0 1; do\n'
    '    if [[ $_round != 0 ]]; then\n'
    '      n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown\n'
    '      if [[ $n != "$N0" ]] || host_crash_detected || ! health_ok; then\n'
    "        record \"overlap-round-$_round\" 2 'NOT_RUN: engine crashed, unhealthy or auto-restarted after the previous round; no restart attempted'\n"
    '        continue\n'
    '      fi\n'
    '    fi\n'
    '    READINESS=$Q18A_READINESS GEN=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \\\n'
    '    S5_OVERLAY=$S5_OVERLAY S5_LAUNCHER_NAME=$S5_LAUNCHER_NAME \\\n'
    '    PACKAGE_MANIFEST=$S5_OVERLAY/SHA256SUMS GENERATION=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \\\n'
    '    python3 "$H/e3_native35_performance.py" --url "$PROBE_URL" --model "$MODEL" \\\n'
    '      --run-id "$RUN_ID" --epoch "$EPOCH" --round "$_round" --workload "$H/workloads/q18a-w2-overlap.json" \\\n'
    '      --w1 "$Q/w1.json" --gates "$Q/gates.json" --out "$Q/overlap-round-$_round.json" > "$Q/overlap-round-$_round.log" 2>&1\n'
    "    record \"overlap-round-$_round\" \"$?\" 'actual native fresh/append in full-solo, matched split-solo and decode-during-prefill; round salt namespace; 80/80 reported target'\n"
    '    python3 "$H/e3_native_memory_grade.py" --performance "$Q/overlap-round-$_round.json" \\\n'
    '      --samples "$Q/client-samples.timed.jsonl" --out "$Q/memory-grade-round-$_round.json" > "$Q/memory-grade-round-$_round.log" 2>&1\n'
    "    record \"memory-round-$_round\" \"$?\" 'unchanged 512 MiB physical reserve, four queried allocator peaks'\n"
    '  done\n'
    '  n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown\n'
    '  if [[ $n != "$N0" ]] || ! health_ok; then\n'
    "    ENGINE_CRASHED=1;record overlap-engine 1 'engine unhealthy or auto-restarted; no restart attempted'\n"
    '  else\n'
    "    record overlap-engine 0 'engine healthy with unchanged NRestarts'\n"
    '  fi\n'
    'else\n'
    "  record overlap 2 'native performance NOT_RUN: same-queue gates refused continuation'\n"
    'fi\n'
    'python3 "$H/native35_performance_grade.py" --w1 "$Q/w1.json" --gates "$Q/gates.json" \\\n'
    '  --round "$Q/overlap-round-0.json" --round "$Q/overlap-round-1.json" \\\n'
    '  --memory "$Q/memory-grade-round-0.json" --memory "$Q/memory-grade-round-1.json" \\\n'
    '  --out "$Q/native35-performance-grade.json" > "$Q/native35-performance-grade.log" 2>&1\n'
    "record native35-performance \"$?\" 'both retained rounds beside the W1 baseline; raw rates, counts and durations; 80/80 is a marker, never a gate'\n"
    '\n')
P_ENDING=('stop_pressure || true\n'
    'stop_sampler || true\n'
    'capture_logs || true\n'
    'capture_end "$Q18A_IDENTITY_TIMEOUT_S" || true\n'
    'stop_engine || true\n'
    'restore_launcher || true\n'
    'restore_defaults || true\n'
    'archive_owned_next || true\n'
    'write_cleanup\n'
    'DONE=1\n'
    '[[ $FAILS == 0 ]] || exit 1\n'
    '[[ $INCS == 0 ]] || exit 2\n'
    'exit 0\n')
O_DIAG=' FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=1 FLASH_NATIVE34_ARM=O '
P_DIAG=' FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=0 '
O_READ='Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=480\n'
P_READ='Q18A_FIRST_ADVANCE_READ_TIMEOUT_S=1200\n'
O_DECLARE_PINS=('      leak_partial_tail.py leaktest.py pressure2.py pressure_wait.py pp_release_grade.py pp_canary_bridge.py \\\n')
P_DECLARE_PINS=('      leak_partial_tail.py leaktest.py pressure2.py pressure_wait.py pp_release_grade.py pp_canary_bridge.py \\\n'
                '      e3_native35_performance.py native35_performance_grade.py native35p_gate.py \\\n')


def swap(text,old,new,count=1):
    assert text.count(old)==count,(old[:80],text.count(old));return text.replace(old,new)


def names(text):
    return text.replace('q18a-native35-o','q18a-native35-p').replace('q18a_native35_o','q18a_native35_p')


def native28_gates_block():
    """native28's qualification-ready -> W1 -> gates.json block, exact bytes."""
    text=(RUNS/'q18a_native28.draft.sh').read_text()
    start=text.index('RULE4_CONTINUATION=0\n');end=text.index('gate_rc=$?;step overlap-gates "$gate_rc"\n',start)
    return text[start:end]+'gate_rc=$?;step overlap-gates "$gate_rc"\n'


def generate():
    draft=names((RUNS/'q18a_native35_o.draft.sh').read_text())
    draft=swap(draft,O_HEADER,P_HEADER)
    draft=swap(draft,O_TIMEOUT,P_TIMEOUT);draft=swap(draft,O_SLOT_NOTE,P_SLOT_NOTE)
    draft=swap(draft,O_SLOT_IMPORT,P_SLOT_IMPORT)
    draft=swap(draft,O_ROW4_BRANCH,'');draft=swap(draft,'  fi\n  rc=$?\n','  rc=$?\n')
    draft=swap(draft,O_INTERFACES,P_INTERFACES)
    draft=swap(draft,O_FINAL_METRICS,P_FINAL_METRICS)
    draft=swap(draft,O_STOP,native28_gates_block()+P_MEASUREMENT+P_ENDING)
    settings=names((RUNS/'q18a_native35_o_settings.sh').read_text())
    settings=swap(settings,O_DIAG,P_DIAG);settings=swap(settings,O_READ,P_READ)
    declare=names((RUNS/'q18a_native35_o_declare.sh').read_text())
    declare=swap(declare,O_DECLARE_PINS,P_DECLARE_PINS)
    return {RUNS/'q18a_native35_p.draft.sh':draft,RUNS/'q18a_native35_p_settings.sh':settings,RUNS/'q18a_native35_p_declare.sh':declare}


if __name__=='__main__':
    import sys
    check='--check' in sys.argv
    for path,text in generate().items():
        if check:assert path.read_text()==text,path.name
        else:path.write_text(text)
    print('native35 P queue drafts '+('verified' if check else 'written'))
