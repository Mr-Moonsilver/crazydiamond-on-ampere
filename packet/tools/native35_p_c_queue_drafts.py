#!/usr/bin/env python3
"""Generate the q18a-native35-p-c queue from the q18a-native35-p-b texts.

P-c = P-b under fresh names with FLASH_NATIVE30_LOCALIZE=0, the second declared
diagnostic switch vs O (lead decision 31 Oct 2026). The four native30 plugins
stay in the overlay and on PYTHONPATH; with LOCALIZE=0, NATIVE33_DIAG=0 and
NATIVE34_DIAG=0 each install() returns before any hook. Declared deltas only:
  1. settings: FLASH_NATIVE30_LOCALIZE=1 -> 0;
  2. cleanup: native35-diagnostic is an explicit NOT_RUN record (rc 2) instead
     of native35_grade.py, which would read zero rows and print a vacuous PASS;
  3. the 30 s post-row4 gap stays; only its step note changes;
  4. row4 runs e3_native35_p_c_placement.py (the L2 runner with the decode
     cohort stopped at the 20-token window boundary; lead decision 31 Oct
     2026, option A + observer_end spool), and the preflight interface list
     names it and e3_observer_census.py;
  5. the declared row4 capacity continuation (lead decision 31 Oct 2026, gate
     design, P-c only): when row4 is INCONCLUSIVE the queue runs
     e3_native35_p_c_continuation.py over qual-row4.json with its own engine
     probes (host_crash_detected, NRestarts, health_ok, the engine-errors marker
     ERE over the live engine-stream files); an eligible receipt lets the gate
     loop skip row4 for W1, and PY_GATES records row4_capacity_continuation,
     the receipt sha256 and the matched reasons (row4 stays INCONCLUSIVE). The
     interface list and the declaration pins add the continuation module (the
     pins also the P-c runner and census);
  plus the header comment. native30-localize keeps running unchanged: zero
  begins on every rank is the box-side proof the switch was off. Every
  replacement must match exactly once.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'harness/runs'

PB_HEADER=('# Native35-p-b: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 (L2 client window at the accepted resume receipt) -> W1 -> two retained W2 rounds -> owned restore.\n')
PC_HEADER=('# Native35-p-c: P-b with FLASH_NATIVE30_LOCALIZE=0 (plugins present, inert): native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 (L2 client window at the accepted resume receipt; decode cohort stopped at the window boundary) -> W1 -> two retained W2 rounds -> owned restore.\n')
PB_DIAGNOSTIC=('  python3 "$H/native35_grade.py" --log "$Q/engine-stream.err" --log "$Q/engine-stream.log" --arm-record "$Q/qual-row4.json" --out "$Q/native35-grade.json" > "$Q/native35-grade.log" 2>&1\n'
               "  record native35-diagnostic \"$?\" 'unfenced O-style arm; no qualification claim'\n")
PC_DIAGNOSTIC="  record native35-diagnostic 2 'NOT_RUN: LOCALIZE=0 declared; no receipts to read'\n"
PB_TAIL_NOTE="  step native30-receipt-tail 0 '30 seconds after row4 client return; original clients already joined'\n"
PC_TAIL_NOTE="  step native30-receipt-tail 0 '30 s post-row4 gap kept from P-b; LOCALIZE=0: no receipts, no receipt tail'\n"
PB_INTERFACES='  e3_native35_l2_placement.py; do\n'
PC_INTERFACES='  e3_native35_l2_placement.py e3_native35_p_c_placement.py e3_observer_census.py; do\n'  # row4 delta (4); 5 adds the continuation
PB_ROW4=('  # L2: row4 uses the resume-receipt window collector; argv and record are unchanged.\n'
         '  if [[ $name == row4 ]]; then cli=e3_native35_l2_placement.py; fi\n')
PC_ROW4=('  # P-c: row4 uses the L2 collector with the decode cohort stopped at the window boundary; argv and record are unchanged.\n'
         '  if [[ $name == row4 ]]; then cli=e3_native35_p_c_placement.py; fi\n')
PB_READY=('export RULE4_CONTINUATION\n'
          'qualification_ready=1\n'
          'for _gate in startup row1 rule4_controls rule4_candidate row4; do\n'
          '  if [[ $RULE4_CONTINUATION == 1 &&\n'
          '        ( $_gate == rule4_controls || $_gate == rule4_candidate ) ]]; then continue; fi\n'
          '  [[ ${PHASE_VERDICT[$_gate]:-INCONCLUSIVE} == PASS ]] || qualification_ready=0\n'
          'done\n')
PC_READY=('export RULE4_CONTINUATION\n'
          '# P-c: declared row4 capacity continuation (lead decision 31 Oct 2026).  A\n'
          '# capacity-only row4 INCONCLUSIVE (observer rings/lists or the observer_end reply\n'
          '# over the spool ceiling) with a complete client window, no failures, no crash\n'
          '# or error marker, unchanged NRestarts and a typed full cleanup lets W1/W2 run.\n'
          '# row4 itself stays INCONCLUSIVE.  Any other row4 outcome withholds as before.\n'
          'ROW4_CAPACITY_CONTINUATION=0\n'
          'if [[ ${PHASE_VERDICT[row4]:-INCONCLUSIVE} == INCONCLUSIVE ]]; then\n'
          '  _cap_n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || _cap_n=unknown\n'
          '  _cap_crashed=0; host_crash_detected && _cap_crashed=1\n'
          '  _cap_healthy=0; health_ok && _cap_healthy=1\n'
          '  _cap_markers=unavailable\n'
          '  if [[ -r $Q/engine-stream.log && -r $Q/engine-stream.err ]]; then\n'
          "    _cap_markers=$(cat \"$Q/engine-stream.log\" \"$Q/engine-stream.err\" | grep -Ec 'Traceback|AssertionError|CUDA error|Xid|ContractError|EngineDeadError|FLASH_S5_.*FAIL')\n"
          '  fi\n'
          '  python3 "$H/e3_native35_p_c_continuation.py" --row4 "$Q/qual-row4.json" \\\n'
          '    --engine-crashed "$_cap_crashed" --healthy "$_cap_healthy" \\\n'
          '    --nrestarts-start "$N0" --nrestarts "$_cap_n" --error-markers "$_cap_markers" \\\n'
          '    --out "$Q/row4-capacity-continuation.json" > "$Q/row4-capacity-continuation.log" 2>&1\n'
          '  if [[ $? == 0 ]]; then\n'
          '    ROW4_CAPACITY_CONTINUATION=1\n'
          "    step row4-capacity-continuation 0 'row4 INCONCLUSIVE on observer capacity only; W1/W2 continue under the declared P-c rule; row4 stays INCONCLUSIVE'\n"
          '  else\n'
          "    step row4-capacity-continuation 2 'withheld: row4 INCONCLUSIVE is not capacity-only or a precondition failed (row4-capacity-continuation.json)'\n"
          '  fi\n'
          'fi\n'
          'export ROW4_CAPACITY_CONTINUATION\n'
          'qualification_ready=1\n'
          'for _gate in startup row1 rule4_controls rule4_candidate row4; do\n'
          '  if [[ $RULE4_CONTINUATION == 1 &&\n'
          '        ( $_gate == rule4_controls || $_gate == rule4_candidate ) ]]; then continue; fi\n'
          '  if [[ $ROW4_CAPACITY_CONTINUATION == 1 && $_gate == row4 ]]; then continue; fi\n'
          '  [[ ${PHASE_VERDICT[$_gate]:-INCONCLUSIVE} == PASS ]] || qualification_ready=0\n'
          'done\n')
PB_GATES_VALID=("continuation=os.environ.get('RULE4_CONTINUATION')=='1'\n"
                "allowed={'rule4_controls','rule4_candidate'} if continuation else set()\n"
                "valid=all(v=='PASS' or (k in allowed and v in ('INCONCLUSIVE','FAIL')) for k,v in d.items())\n")
PC_GATES_VALID=("continuation=os.environ.get('RULE4_CONTINUATION')=='1'\n"
                "capacity=os.environ.get('ROW4_CAPACITY_CONTINUATION')=='1'\n"
                "allowed={'rule4_controls','rule4_candidate'} if continuation else set()\n"
                "valid=all(v=='PASS' or (k in allowed and v in ('INCONCLUSIVE','FAIL'))\n"
                "          or (capacity and k=='row4' and v=='INCONCLUSIVE') for k,v in d.items())\n")
PB_GATES_DUMP=("with open(sys.argv[1],'x') as f:json.dump({'schema':1,'outcomes':d,'no_retry':True,\n"
               "    'calibration_frozen':True,'rule4_control_continuation':continuation,\n"
               "    'rule4_continuation_sha256':continuation_sha},f)\n")
PC_GATES_DUMP=("capacity_sha=None;capacity_matched=[]\n"
               "if capacity:\n"
               "    data=Path(sys.argv[1]).with_name('row4-capacity-continuation.json').read_bytes()\n"
               "    receipt=json.loads(data)\n"
               "    assert receipt.get('eligible') is True and receipt.get('row4_verdict')==d['row4']=='INCONCLUSIVE'\n"
               "    capacity_sha=hashlib.sha256(data).hexdigest();capacity_matched=[row['reason'] for row in receipt['matched']]\n"
               "with open(sys.argv[1],'x') as f:json.dump({'schema':1,'outcomes':d,'no_retry':True,\n"
               "    'calibration_frozen':True,'rule4_control_continuation':continuation,\n"
               "    'rule4_continuation_sha256':continuation_sha,\n"
               "    'row4_capacity_continuation':capacity,'row4_capacity_continuation_sha256':capacity_sha,\n"
               "    'row4_capacity_matched':capacity_matched},f)\n")
PC_INTERFACES_CONT='  e3_native35_l2_placement.py e3_native35_p_c_placement.py e3_observer_census.py e3_native35_p_c_continuation.py; do\n'
PB_PINS='      e3_native35_performance.py native35_performance_grade.py native35p_gate.py e3_native35_l2_placement.py \\\n'
PC_PINS=('      e3_native35_performance.py native35_performance_grade.py native35p_gate.py e3_native35_l2_placement.py \\\n'
         '      e3_native35_p_c_placement.py e3_observer_census.py e3_native35_p_c_continuation.py \\\n')
PB_LOCALIZE=' FLASH_NATIVE30_LOCALIZE=1 '
PC_LOCALIZE=' FLASH_NATIVE30_LOCALIZE=0 '


def swap(text,old,new):
    assert text.count(old)==1,(old[:80],text.count(old));return text.replace(old,new)


def names(text):
    # Only the q18a-prefixed queue names; native35_performance_grade etc. stay.
    assert 'q18a-native35-p-c' not in text and 'q18a_native35_p_c' not in text
    return text.replace('q18a-native35-p-b','q18a-native35-p-c').replace('q18a_native35_p_b','q18a_native35_p_c')


def generate():
    draft=(RUNS/'q18a_native35_p_b.draft.sh').read_text()
    for old,new in ((PB_HEADER,PC_HEADER),(PB_DIAGNOSTIC,PC_DIAGNOSTIC),(PB_TAIL_NOTE,PC_TAIL_NOTE),(PB_INTERFACES,PC_INTERFACES_CONT),(PB_ROW4,PC_ROW4),
                    (PB_READY,PC_READY),(PB_GATES_VALID,PC_GATES_VALID),(PB_GATES_DUMP,PC_GATES_DUMP)):draft=swap(draft,old,new)
    settings=names(swap((RUNS/'q18a_native35_p_b_settings.sh').read_text(),PB_LOCALIZE,PC_LOCALIZE))
    declare=names(swap((RUNS/'q18a_native35_p_b_declare.sh').read_text(),PB_PINS,PC_PINS))
    return {RUNS/'q18a_native35_p_c.draft.sh':names(draft),RUNS/'q18a_native35_p_c_settings.sh':settings,RUNS/'q18a_native35_p_c_declare.sh':declare}


if __name__=='__main__':
    check='--check' in sys.argv
    for path,text in generate().items():
        if check:assert path.read_text()==text,path.name
        else:
            assert not path.exists(),path.name+' exists; regenerate only under --check'
            path.write_text(text);path.chmod(0o664)
    print('native35 P-c queue drafts '+('verified' if check else 'written'))
