#!/usr/bin/env python3
"""Generate the q18a-native35-p-b queue from the q18a-native35-p (P-a) texts.

P-b = P-a under fresh names plus the approved L2 row4 window fix, taken exactly:
the row4 qualification phase calls e3_native35_l2_placement.py (the
resume-receipt window collector, claude/native35-l2) with the identical argv,
using L2's own phase-function replacement text (imported from
native35_l2_queue_draft.py, not re-typed). Every other phase still calls
e3_q18a_qualification.py. P's packaging rule (every executable interface is
checked present and pinned into the run declaration) adds the new CLI to the
draft interface loop and the declaration pins. Settings differ from P-a by the
queue name only (FLASH_NATIVE34_DIAG=0, LOCALIZE=1 and the 1200 s first-advance
bound unchanged). Every replacement must match exactly once.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'harness/runs'
sys.path.insert(0,str(ROOT/'tools'))
from native35_l2_queue_draft import L_PHASE,L2_PHASE

P_HEADER=('# Native35-p: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 -> W1 -> two retained W2 rounds -> owned restore.\n')
PB_HEADER=('# Native35-p-b: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 (L2 client window at the accepted resume receipt) -> W1 -> two retained W2 rounds -> owned restore.\n')
P_INTERFACE='  e3_native35_performance.py native35_performance_grade.py native35p_gate.py; do\n'
PB_INTERFACE='  e3_native35_performance.py native35_performance_grade.py native35p_gate.py \\\n  e3_native35_l2_placement.py; do\n'
P_PINS='      e3_native35_performance.py native35_performance_grade.py native35p_gate.py \\\n'
PB_PINS='      e3_native35_performance.py native35_performance_grade.py native35p_gate.py e3_native35_l2_placement.py \\\n'


def swap(text,old,new):
    assert text.count(old)==1,(old[:80],text.count(old));return text.replace(old,new)


def names(text):
    # Only the q18a-prefixed queue names; native35_performance_grade etc. stay.
    assert 'q18a-native35-p-b' not in text and 'q18a_native35_p_b' not in text
    return text.replace('q18a-native35-p','q18a-native35-p-b').replace('q18a_native35_p','q18a_native35_p_b')


def generate():
    draft=(RUNS/'q18a_native35_p.draft.sh').read_text()
    draft=swap(draft,P_HEADER,PB_HEADER);draft=swap(draft,L_PHASE,L2_PHASE);draft=swap(draft,P_INTERFACE,PB_INTERFACE);draft=names(draft)
    settings=names((RUNS/'q18a_native35_p_settings.sh').read_text())
    declare=names(swap((RUNS/'q18a_native35_p_declare.sh').read_text(),P_PINS,PB_PINS))
    return {RUNS/'q18a_native35_p_b.draft.sh':draft,RUNS/'q18a_native35_p_b_settings.sh':settings,RUNS/'q18a_native35_p_b_declare.sh':declare}


if __name__=='__main__':
    check='--check' in sys.argv
    for path,text in generate().items():
        if check:assert path.read_text()==text,path.name
        else:
            assert not path.exists(),path.name+' exists; regenerate only under --check'
            path.write_text(text)
    print('native35 P-b queue drafts '+('verified' if check else 'written'))
