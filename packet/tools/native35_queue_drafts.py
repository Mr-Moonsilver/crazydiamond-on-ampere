#!/usr/bin/env python3
"""Generate the q18a-native35 R/O/L queue drafts from the frozen native34 O3/I4 texts.

R and O are the native34 O3 queue under fresh names (R adds READBACK=1 and the
native-writer grade); both add the startup ledger phase. L is the native33-form
bounded queue (real row4 placement qualification) followed by the native28
lifetime, hygiene and leak phases verbatim, with the longer L window slot.
Every replacement must match exactly once; nothing else in the texts changes.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'harness/runs'

LEDGER=('export EPOCH\n',
        'export EPOCH\n'
        'python3 "$H/native35_ledger_grade.py" --startup "$Q/startup-grade.json" --out "$Q/native35-ledger.json" > "$Q/native35-ledger.log" 2>&1\n'
        "record native35-ledger \"$?\" 'per-rank native-private builder bytes allocated once; fail-closed field classification; no headroom claim'\n")
O3_GRADE=('  python3 "$H/native34_grade.py" --log "$Q/engine-stream.err" --log "$Q/engine-stream.log" --arm-record "$Q/qual-row4.json" --out "$Q/native34-grade.json" > "$Q/native34-grade.log" 2>&1\n'
          "  record native34-diagnostic \"$?\" 'bounded causal arm and indexer discriminator; no qualification claim'\n")
O_GRADE=('  python3 "$H/native35_grade.py" --log "$Q/engine-stream.err" --log "$Q/engine-stream.log" --arm-record "$Q/qual-row4.json" --out "$Q/native35-grade.json" > "$Q/native35-grade.log" 2>&1\n'
         "  record native35-diagnostic \"$?\" 'unfenced O-style arm; no qualification claim'\n")
R_GRADE=('  python3 "$H/native35_grade.py" --require-readback --require-native-writer --log "$Q/engine-stream.err" --log "$Q/engine-stream.log" --arm-record "$Q/qual-row4.json" --out "$Q/native35-grade.json" > "$Q/native35-grade.log" 2>&1\n'
         "  record native35-diagnostic \"$?\" 'unfenced O-style arm with evidence-only readback and native-private writer catalogue; no qualification claim'\n")
L_GRADE=('  python3 "$H/native35_lifetime_grade.py" --queue "$Q" --out "$Q/native35-lifetime-grade.json" > "$Q/native35-lifetime-grade.log" 2>&1\n'
         "  record native35-lifetime \"$?\" 'native cancel/drain/reuse plus unchanged hygiene/leak oracles; no performance claim'\n")
I4_ROW4_BRANCH=('  if [[ $name == row4 ]]; then\n'
    '    python3 "$H/e3_native34_causal.py" --url "$PROBE_URL" --model "$MODEL" \\\n'
    '      --run-id "$RUN_ID" --epoch "$EPOCH" --out "$Q/qual-row4.json" \\\n'
    '      --store "$Q18A_STORE" --workload "$H/workloads/q17b-correctness-precontrols.json" \\\n'
    '      --prefill-plan "$H/workloads/q18a-native-placement-prefill.json" \\\n'
    '      --snapshot-max-bytes "$SNAPSHOT_MAX_BYTES" --timeout-s 90 --arm I \\\n'
    '      > "$Q/qual-row4.log" 2>&1\n'
    '  else\n')
I4_TAIL_START='\n# The real phase producer has joined/stopped its owned clients before returning.\n'
I4_STOP=("# Stop after native28 row4; no W1/performance/lifecycle queue follows.\n"
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
L_ENDING=('stop_pressure || true\n'
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


def swap(text,old,new,count=1):
    assert text.count(old)==count,(old[:80],text.count(old));return text.replace(old,new)


def names(text,src,dst):
    return text.replace('q18a-native34-'+src,'q18a-native35-'+dst).replace('q18a_native34_'+src,'q18a_native35_'+dst)


def native28_lifetime_block():
    """native28's lifetime/hygiene/leak phases, exact bytes, without the canary tail."""
    text=(RUNS/'q18a_native28.draft.sh').read_text()
    start=text.index('# Native cancellation and persistent reuse are measured separately from the\n')
    end=text.index('    identity "$Q/canary.before.json"',start)
    block=text[start:end]
    assert block.endswith("    record pressure-lifetime \"$?\" 'unchanged state eviction and request-completion coverage; preemption reported'\n")
    return block+"  fi\nelse\n  record lifecycle 2 'native lifetime and hygiene NOT_RUN after missing gates or engine crash'\nfi\n\n"


def native28_gates_block():
    text=(RUNS/'q18a_native28.draft.sh').read_text()
    start=text.index('RULE4_CONTINUATION=0\n');end=text.index('gate_rc=$?;step overlap-gates "$gate_rc"\n',start)
    block=text[start:end]
    # No W1 phase in the L arm: drop native28's W1 block between the rule4
    # continuation export and the gates record; keep both neighbours verbatim.
    w1_start=block.index('qualification_ready=1\n');w1_end=block.index("Q18A_PHASE_VERDICTS='{'\n")
    block=block[:w1_start]+block[w1_end:]
    block=swap(block,'for _gate in startup row1 rule4_controls rule4_candidate row4 w1_controls; do\n','for _gate in startup row1 rule4_controls rule4_candidate row4; do\n')
    return block+'gate_rc=$?;step lifetime-gates "$gate_rc"\n'


def r_and_o():
    out={}
    for arm,grade,readback in (('r',R_GRADE,True),('o',O_GRADE,False)):
        draft=names((RUNS/'q18a_native34_o3.draft.sh').read_text(),'o3',arm)
        draft=swap(draft,O3_GRADE,grade);draft=swap(draft,*LEDGER)
        settings=names((RUNS/'q18a_native34_o3_settings.sh').read_text(),'o3',arm)
        if readback:settings=swap(settings,' FLASH_NATIVE34_ARM=O ',' FLASH_NATIVE34_ARM=O FLASH_NATIVE34_READBACK=1 ')
        declare=names((RUNS/'q18a_native34_o3_declare.sh').read_text(),'o3',arm)
        out[arm]=dict(draft=draft,settings=settings,declare=declare)
    return out


def lifetime():
    draft=names((RUNS/'q18a_native34_i4.draft.sh').read_text(),'i4','l')
    draft=swap(draft,'# Native30-e: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 -> owned restore.\n'
        '# Inherited cleanup machinery below; later qualification/performance phases are absent.\n',
        '# Native35-l: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 -> native lifetime -> hygiene/leak -> owned restore.\n'
        '# Inherited cleanup machinery below; W1/performance/canary phases are absent.\n')
    draft=swap(draft,'  exec timeout --signal=TERM --kill-after=420s 2100s env Q18A_OVERLAP_BOUNDED=1 bash "$0" "$@"\n',
        '  exec timeout --signal=TERM --kill-after=420s 7200s env Q18A_OVERLAP_BOUNDED=1 bash "$0" "$@"\n')
    draft=swap(draft,'# entry unless its 2100 s timeout plus 420 s kill margin fits that window.\n','# entry unless its 7200 s timeout plus 420 s kill margin fits that window.\n')
    draft=swap(draft,'from native34_gate import window_ok\n','from native35_gate import lifetime_window_ok as window_ok\n')
    draft=swap(draft,I4_ROW4_BRANCH,'');draft=swap(draft,'  fi\n  rc=$?\n','  rc=$?\n')
    draft=swap(draft,O3_GRADE.replace('"$H/native34_grade.py" --log','"$H/native34_grade.py" --require-readback --log'),L_GRADE)
    draft=swap(draft,*LEDGER)
    tail=draft.index(I4_TAIL_START);head=draft[:tail]
    receipt=draft[tail:draft.index(I4_STOP)]
    draft=head+receipt+native28_gates_block()+native28_lifetime_block()+L_ENDING
    settings=names((RUNS/'q18a_native34_i4_settings.sh').read_text(),'i4','l')
    settings=swap(settings,' FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=1 FLASH_NATIVE34_ARM=I FLASH_NATIVE34_READBACK=1 ',' FLASH_NATIVE33_DIAG=0 FLASH_NATIVE34_DIAG=0 ')
    declare=names((RUNS/'q18a_native34_i4_declare.sh').read_text(),'i4','l')
    return dict(draft=draft,settings=settings,declare=declare)


def generate():
    files={}
    for arm,parts in {**r_and_o(),'l':lifetime()}.items():
        files[RUNS/('q18a_native35_'+arm+'.draft.sh')]=parts['draft']
        files[RUNS/('q18a_native35_'+arm+'_settings.sh')]=parts['settings']
        files[RUNS/('q18a_native35_'+arm+'_declare.sh')]=parts['declare']
    return files


if __name__=='__main__':
    import sys
    check='--check' in sys.argv
    for path,text in generate().items():
        if check:assert path.read_text()==text,path.name
        else:path.write_text(text)
    print('native35 queue drafts '+('verified' if check else 'written'))
