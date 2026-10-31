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
PC_INTERFACES='  e3_native35_l2_placement.py e3_native35_p_c_placement.py e3_observer_census.py; do\n'
PB_ROW4=('  # L2: row4 uses the resume-receipt window collector; argv and record are unchanged.\n'
         '  if [[ $name == row4 ]]; then cli=e3_native35_l2_placement.py; fi\n')
PC_ROW4=('  # P-c: row4 uses the L2 collector with the decode cohort stopped at the window boundary; argv and record are unchanged.\n'
         '  if [[ $name == row4 ]]; then cli=e3_native35_p_c_placement.py; fi\n')
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
    for old,new in ((PB_HEADER,PC_HEADER),(PB_DIAGNOSTIC,PC_DIAGNOSTIC),(PB_TAIL_NOTE,PC_TAIL_NOTE),(PB_INTERFACES,PC_INTERFACES),(PB_ROW4,PC_ROW4)):draft=swap(draft,old,new)
    settings=names(swap((RUNS/'q18a_native35_p_b_settings.sh').read_text(),PB_LOCALIZE,PC_LOCALIZE))
    declare=names((RUNS/'q18a_native35_p_b_declare.sh').read_text())
    return {RUNS/'q18a_native35_p_c.draft.sh':names(draft),RUNS/'q18a_native35_p_c_settings.sh':settings,RUNS/'q18a_native35_p_c_declare.sh':declare}


if __name__=='__main__':
    check='--check' in sys.argv
    for path,text in generate().items():
        if check:assert path.read_text()==text,path.name
        else:
            assert not path.exists(),path.name+' exists; regenerate only under --check'
            path.write_text(text);path.chmod(0o664)
    print('native35 P-c queue drafts '+('verified' if check else 'written'))
