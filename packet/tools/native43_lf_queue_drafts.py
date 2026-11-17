#!/usr/bin/env python3
"""Generate the q18a-native43-lifecycle-f (l-f) queue from the box-run l-e texts.

Base: harness/runs/q18a_native43_lifecycle{.draft,_declare,_settings}.sh as they ran on the box (l-e,
q18a-native43-lifecycle run 1, bundle 296adc0886b6, 1 Nov 2026).  Kept byte for byte except the declared
deltas, each applied exactly once:
  D1-D7  the le-fix queue deltas (tools/native43_le_fix_queue_deltas.py, applied unchanged): engine-death scan in
         host_crash_detected, engine_watched around the row1, lifecycle and typed FULL hygiene clients, and the
         death tool's preflight interface and declaration pin;
  D8     qualification_phase: row1 runs native43_row1_qualification.py --qsa-policy dense, i.e.
         e3_q18a_qualification with the route coverage of the dense policy the preflight asserts
         (native43_row1_grade); the client, the rest of the argv and the record are unchanged; row4 keeps
         e3_native35_p_c_placement.py and every other phase e3_q18a_qualification.py, without the option;
  D9     native43_row1_qualification.py and native43_row1_grade.py are checked with the other harness tools at
         preflight and pinned at declaration;
  names  q18a-native43-lifecycle -> q18a-native43-lifecycle-f (QNAME default, RUN_ID, title) and the file stem
         q18a_native43_lifecycle -> q18a_native43_lifecycle_f; one header line names the l-f deltas.
Engine and workload settings are l-e's (EXTRA_ENV unchanged: FLASH_S5_NATIVE_QSA_POLICY=dense, no trace, no stop).
The window is l-e's (native43_lifecycle_gate: 5000 s + 900 s kill margin).  --check verifies the committed files.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'harness/runs'
sys.path.insert(0, str(ROOT / 'tools'))
import native43_le_fix_queue_deltas as le_fix  # noqa: E402

BASE, LF = 'q18a_native43_lifecycle', 'q18a_native43_lifecycle_f'
BASE_NAME, LF_NAME = 'q18a-native43-lifecycle', 'q18a-native43-lifecycle-f'

HEADER_ANCHOR = ('# Native43 lifecycle (short scope): the box-run native39 lifecycle queue on the native43 dense engine '
                 '(FLASH_S5_NATIVE_QSA_POLICY=dense), observer and offers off: native28 startup -> bind receipt -> W2 '
                 'length guard -> row1 (lane proof) -> lifetime gates -> instruments -> native cancel/drain/reuse -> typed '
                 'FULL -> instruments\n')
HEADER_NOTE = ('# Native43 l-f: the l-e queue (q18a-native43-lifecycle run 1) with the le-fix deltas (engine-death scan in '
               'host_crash_detected; row1, lifecycle and FULL hygiene clients under engine_watched) and row1 graded with '
               'the dense policy\'s route coverage (native43_row1_qualification); engine: bounded broadcast enqueue, '
               'logged worker fatal record, row1 reference on its own dense branch beyond the arena\n')
D8_OLD = ('  local name=$1 rc cli=e3_q18a_qualification.py\n'
          '  # P-c: row4 uses the L2 collector with the decode cohort stopped at the window boundary; argv and record are unchanged.\n'
          '  if [[ $name == row4 ]]; then cli=e3_native35_p_c_placement.py; fi\n'
          '  engine_watched "qual-$name" python3 "$H/$cli" --phase "$name" --url "$PROBE_URL" \\\n')
D8_NEW = ('  local name=$1 rc cli=e3_q18a_qualification.py\n'
          '  local -a policy=()\n'
          '  # P-c: row4 uses the L2 collector with the decode cohort stopped at the window boundary; argv and record are unchanged.\n'
          '  if [[ $name == row4 ]]; then cli=e3_native35_p_c_placement.py; fi\n'
          '  # Native43 l-f: row1 is graded with the route coverage of the policy the preflight asserts (dense); the client,\n'
          '  # the rest of the argv and the record are e3_q18a_qualification\'s (native43_row1_qualification).\n'
          '  if [[ $name == row1 ]]; then cli=native43_row1_qualification.py; policy=(--qsa-policy dense); fi\n'
          '  engine_watched "qual-$name" python3 "$H/$cli" ${policy[@]+"${policy[@]}"} --phase "$name" --url "$PROBE_URL" \\\n')
D9_DRAFT_OLD = 'for _interface in native43_engine_death.py native43_lifecycle_gate.py '
D9_DRAFT_NEW = ('for _interface in native43_row1_qualification.py native43_row1_grade.py native43_engine_death.py '
                'native43_lifecycle_gate.py ')
D9_DECLARE_OLD = ('      native39_lifecycle_gate.py native35_lifetime_grade.py native43_lifecycle_gate.py native43_lifecycle_grade.py '
                  'native43_engine_death.py \\\n')
D9_DECLARE_NEW = ('      native39_lifecycle_gate.py native35_lifetime_grade.py native43_lifecycle_gate.py native43_lifecycle_grade.py '
                  'native43_engine_death.py native43_row1_qualification.py native43_row1_grade.py \\\n')


def swap(text, old, new, label):
    assert text.count(old) == 1, (label, text.count(old))
    return text.replace(old, new)


def names(text):
    assert LF_NAME not in text and LF not in text, 'l-f names already present'
    return text.replace(BASE_NAME, LF_NAME).replace(BASE, LF)


def unnames(text):
    assert BASE_NAME + '-' not in text.replace(LF_NAME, '') and BASE + '_f' == LF
    return text.replace(LF_NAME, BASE_NAME).replace(LF, BASE)


def base_texts():
    return {kind: (RUNS / (BASE + suffix)).read_text()
            for kind, suffix in (('draft', '.draft.sh'), ('declare', '_declare.sh'), ('settings', '_settings.sh'))}


def derive_draft(text):
    text = le_fix.apply_draft(text)
    text = swap(text, D8_OLD, D8_NEW, 'D8')
    text = swap(text, D9_DRAFT_OLD, D9_DRAFT_NEW, 'D9 draft')
    text = names(text)
    # After the rename: the note names the l-e run by its own QNAME.
    text = swap(text, names(HEADER_ANCHOR), names(HEADER_ANCHOR) + HEADER_NOTE, 'header')
    assert text.count('native43_row1_qualification.py') == 2 and text.count('--qsa-policy dense') == 1
    return text


def derive_declare(text):
    text = le_fix.apply_declare(text)
    text = swap(text, D9_DECLARE_OLD, D9_DECLARE_NEW, 'D9 declare')
    return names(text)


def derive_settings(text):
    return names(text)


def generate():
    texts = base_texts()
    return {RUNS / (LF + '.draft.sh'): derive_draft(texts['draft']),
            RUNS / (LF + '_declare.sh'): derive_declare(texts['declare']),
            RUNS / (LF + '_settings.sh'): derive_settings(texts['settings'])}


if __name__ == '__main__':
    check = '--check' in sys.argv
    for path, text in generate().items():
        if check:
            assert path.read_text() == text, path.name
        else:
            assert not path.exists(), path.name + ' exists; regenerate only under --check'
            path.write_text(text)
            path.chmod(0o664)
    print('native43 l-f queue drafts ' + ('verified' if check else 'written'))
