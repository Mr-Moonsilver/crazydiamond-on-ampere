#!/usr/bin/env python3
"""Generate the q18a-native35-l2 queue from the frozen q18a-native35-l texts.

L2 = L under a fresh name plus one declared change: the row4 qualification
phase calls e3_native35_l2_placement.py (the resume-receipt window collector)
with the identical argv; every other phase still calls e3_q18a_qualification.py.
Settings and declare differ from L by the queue name only (FLASH_NATIVE34_DIAG=0
unchanged). Every replacement must match exactly once.
"""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUNS=ROOT/'harness/runs'

L_HEADER=('# Native35-l: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 -> native lifetime -> hygiene/leak -> owned restore.\n')
L2_HEADER=('# Native35-l2: native28 startup -> row1 -> rule4_controls -> rule4_candidate -> row4 (client window at the accepted resume receipt) -> native lifetime -> hygiene/leak -> owned restore.\n')
L_PHASE=('  local name=$1 rc\n'
         '  python3 "$H/e3_q18a_qualification.py" --phase "$name" --url "$PROBE_URL" \\\n')
L2_PHASE=('  local name=$1 rc cli=e3_q18a_qualification.py\n'
          '  # L2: row4 uses the resume-receipt window collector; argv and record are unchanged.\n'
          '  if [[ $name == row4 ]]; then cli=e3_native35_l2_placement.py; fi\n'
          '  python3 "$H/$cli" --phase "$name" --url "$PROBE_URL" \\\n')


def swap(text,old,new):
    assert text.count(old)==1,(old[:80],text.count(old));return text.replace(old,new)


def names(text):
    # Only the q18a-prefixed queue names; native35_lifetime_grade etc. stay.
    assert 'q18a-native35-l2' not in text and 'q18a_native35_l2' not in text
    return text.replace('q18a-native35-l','q18a-native35-l2').replace('q18a_native35_l','q18a_native35_l2')


def generate():
    draft=(RUNS/'q18a_native35_l.draft.sh').read_text()
    draft=swap(draft,L_HEADER,L2_HEADER);draft=swap(draft,L_PHASE,L2_PHASE);draft=names(draft)
    settings=names((RUNS/'q18a_native35_l_settings.sh').read_text())
    declare=names((RUNS/'q18a_native35_l_declare.sh').read_text())
    return {RUNS/'q18a_native35_l2.draft.sh':draft,RUNS/'q18a_native35_l2_settings.sh':settings,RUNS/'q18a_native35_l2_declare.sh':declare}


if __name__=='__main__':
    import sys
    check='--check' in sys.argv
    for path,text in generate().items():
        if check:assert path.read_text()==text,path.name
        else:
            assert not path.exists(),path.name+' exists; regenerate only under --check'
            path.write_text(text)
    print('native35 l2 queue drafts '+('verified' if check else 'written'))
