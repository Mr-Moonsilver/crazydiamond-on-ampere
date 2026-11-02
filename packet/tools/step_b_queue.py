#!/usr/bin/env python3
"""Derive the q18a-step-b box scripts from the committed q18a-step-a texts. Nothing a past queue ran is edited.

Kept from Step A, byte for byte: pins, preflight, startup, bind, host headroom, the unarmed check, the end identity, both band
manifests and their /tokenize guards, the le-fix death scan and watch, the M3 allocator receipts and the whole owned cleanup and
restore tail. Declared deltas, each applied exactly once:
  names    q18a_step_a -> q18a_step_b, q18a-step-a -> q18a-step-b (the band 2 manifest keeps its file name);
  budget   draft and window read step_b_gate (the same budget, hard end and last arm entry as Step A, plus the size 4 trim);
  switch   FLASH_S5_LAYER_GROUPS=1 in EXTRA_ENV beside FLASH_S5_QSA_FUSED=1, asserted in declaration and preflight;
  body     sixteen arms in the lead's order, each behind a typed FULL; fused in all; the grade is e3_step_b_box.
Band 1 labels round 0 and 1; band 2 uses round index 2 in its labels, so every cache salt stays unique and the band follows from
the index as in Step A. No model setting, gate threshold, launcher or restore change. --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import step_a_queue as step_a  # noqa: E402

OLD, NEW = step_a.NEW, 'q18a_step_b'
OLD_Q, NEW_Q = step_a.NEW_Q, 'q18a-step-b'
GROUPS = 'FLASH_S5_LAYER_GROUPS=1'
ARMS_BAND1 = ('g1-solo-0', 'g2-solo-0', 'g4-solo-0', 'g4-live-0', 'g2-live-0', 'g1-live-0',
              'g1-live-1', 'g2-live-1', 'g4-live-1', 'g4-solo-1', 'g2-solo-1', 'g1-solo-1')
ARMS_BAND2 = ('g2-live-2', 'g1-live-2', 'g1-solo-2', 'g2-solo-2')
ARMS = ARMS_BAND1 + ARMS_BAND2
HEADER_NOTE = ('# Step B: this queue compares layer group sizes on one engine, fused selected in every arm, matched per arm by the typed\n'
               '# native_layer_group control: band 1 sizes 1, 2, 4 at 65,445 tokens, band 2 sizes 1, 2 at 131,094 tokens past the 97,152-row\n'
               '# arena. Mechanism gates (crash-free, TP commitment, group engagement); speed, decode and memory are reported; release\n'
               '# readiness is not assessed.\n#\n')


def body():
    text = step_a.BODY
    swaps = (
        ('# Step A: band 1 ABBA (stepped-solo-0 .. stepped-solo-1) then band 2 matched (fused-live-2 .. fused-solo-2). Each arm selects\n'
         '# burst and its fused or stepped path by typed control while paused idle FULL. Speed and memory are descriptive. Every\n'
         '# timeout below is one step_a_gate cap (instruments, arm_entries, step_a_arms, ingress, end_full_idle, grade); each arm\n',
         '# Step B: band 1 ABBA-style (g1-solo-0 .. g1-solo-1) then band 2 matched (g2-live-2 .. g2-solo-2). Each arm selects burst, fused\n'
         '# and its group size by typed control while paused idle FULL. Speed, decode and memory are descriptive. Every timeout\n'
         '# below is one step_b_gate cap (instruments, arm_entries, step_b_arms, ingress, end_full_idle, grade); each arm\n'),
        ('LAST_ARM_ENTRY_S=$(native41_budget last_arm_entry)\n',
         'LAST_ARM_ENTRY_S=$(native41_budget last_arm_entry)\nTRIM_G4_REPEAT_S=$(native41_budget trim_g4_repeat)\n'),
        ('for _label in ' + ' '.join(step_a.ARMS) + '; do\n', 'for _label in ' + ' '.join(ARMS) + '; do\n'),
        ('      record "arm-$_label" 2 "NOT_RUN: box budget, elapsed ${SECONDS}s > ${LAST_ARM_ENTRY_S}s"\n      continue\n    fi\n',
         '      record "arm-$_label" 2 "NOT_RUN: box budget, elapsed ${SECONDS}s > ${LAST_ARM_ENTRY_S}s"\n      continue\n    fi\n'
         '    # The lead\'s order of what goes first: a size 4 repeat gives way to band 2.\n'
         '    if [[ $_label == g4-*-1 ]] && (( SECONDS > TRIM_G4_REPEAT_S )); then\n'
         '      record "arm-$_label" 2 "NOT_RUN: size 4 repeat trimmed for band 2, elapsed ${SECONDS}s > ${TRIM_G4_REPEAT_S}s"\n'
         '      continue\n    fi\n'),
        ('python3 "$H/e3_step_a_box.py" arm \\\n', 'python3 "$H/e3_step_b_box.py" arm \\\n'),
        ('      --mode "$_mode" --cohort "$_cohort" --index "$_pair" \\\n', '      --size "${_mode#g}" --cohort "$_cohort" --index "$_pair" \\\n'),
        ("'auto arm: 8 standing decoders, burst plus the fused or stepped selection, fresh, append; live keeps them, solo cancels'",
         "'auto arm: 8 standing decoders, burst, fused and the group size, fresh, append; live keeps them, solo cancels'"),
        ('timeout 600s python3 "$H/e3_step_a_box.py" grade "${GRADE_ARGS[@]}" \\\n', 'timeout 600s python3 "$H/e3_step_b_box.py" grade "${GRADE_ARGS[@]}" \\\n'),
        ('--report "$Q/step-a-report.md" --out "$Q/step-a-grade.json" > "$Q/step-a-grade.log" 2>&1\n',
         '--report "$Q/step-b-report.md" --out "$Q/step-b-grade.json" > "$Q/step-b-grade.log" 2>&1\n'),
        ("record step-a \"$?\" 'per-arm mechanism (crash-free, TP commitment, engagement), complete progress lines; speed and memory descriptive'",
         "record step-b \"$?\" 'per-arm mechanism (crash-free, TP commitment, group engagement), complete progress lines; speed, decode and memory descriptive'"),
    )
    for old, new in swaps:
        assert text.count(old) == 1, (old[:80], text.count(old))
        text = text.replace(old, new)
    return text


BODY = body()
BUDGET_OLD = 'from step_a_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES\n'
BUDGET_NEW = 'from step_b_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,PHASES,TRIM_G4_REPEAT_S\n'
PRINT_OLD = "else LAST_ARM_ENTRY_S if sys.argv[1]=='last_arm_entry' else PHASES[sys.argv[1]])\n"
PRINT_NEW = "else LAST_ARM_ENTRY_S if sys.argv[1]=='last_arm_entry' else TRIM_G4_REPEAT_S if sys.argv[1]=='trim_g4_repeat' else PHASES[sys.argv[1]])\n"
SETTINGS_FUSED = step_a.SETTINGS_DENSE + step_a.FUSED + ' '
SETTINGS_ANCHOR = '# Step A: FLASH_S5_QSA_FUSED=1 below,'
SETTINGS_NOTE = ('# Step B: FLASH_S5_LAYER_GROUPS=1 below, the only setting added to Step A; each arm selects its size by typed control.\n')
DRAFT_GROUPS = "assert env.get('FLASH_S5_LAYER_GROUPS') == '1', 'step-b group switch'\n"
DECLARE_GROUPS = "assert env.get('FLASH_S5_LAYER_GROUPS')=='1', 'step-b runs with the group switch installed'\n"
INTERFACE_OLD = '  e3_step_a_box.py step_a_gate.py; do'
INTERFACE_NEW = '  e3_step_a_box.py step_a_gate.py e3_step_b_box.py step_b_gate.py; do'
PINS_OLD = 'e3_step_a_box.py step_a_gate.py '
PINS_NEW = PINS_OLD + 'e3_step_b_box.py step_b_gate.py '
KEEP = step_a.KEEP


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


# Step A's absolute owner-window slot, taken out: launch time is governed by the lead at launch (operator, 2 Nov 2026: "we don't
# put in code what you govern"). The run's own budget stays relative to its start (bash SECONDS against step_b_gate's caps).
WINDOW_OLD = (
    "# The owner supplies the UTC hard end at bundle build time. The queue refuses\n"
    "# entry unless its derived active timeout plus 900 s kill margin fits that window (native39_trunk_gate derivation).\n"
    "# Production restoration remains with the owner. No workload/gate changes.\n"
    "python3 - <<'PY_SLOT' || fail_early night-window-worst-case-exceeds-restore-margin\n"
    "import os,sys,time\n"
    "sys.path.insert(0,os.environ.get('HARNESS_DIR') or os.environ.get('H') or '')\n"
    "from step_a_gate import window_ok\n"
    "assert window_ok(time.time()), 'queue outside owner window or cleanup margin'\n"
    "PY_SLOT\n")
WINDOW_NEW = (
    "# Launch time is governed by the lead at launch (operator, 2 Nov 2026); no absolute window or hard end in this queue. The run's\n"
    "# budget is relative to its own start: the watchdog, the arm entry limit and the size 4 repeat trim (step_b_gate, bash SECONDS).\n"
    "# Production restoration remains with the owner. No workload/gate changes.\n")


def derive():
    out = {}
    for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
        text = (ROOT / 'harness/runs' / (OLD + suffix)).read_text()
        if suffix == '.draft.sh':
            text = replace_once(text, step_a.BODY, BODY)
            text = replace_once(text, BUDGET_OLD, BUDGET_NEW)
            text = replace_once(text, PRINT_OLD, PRINT_NEW)
            text = replace_once(text, WINDOW_OLD, WINDOW_NEW)
            text = replace_once(text, INTERFACE_OLD, INTERFACE_NEW)
            text = replace_once(text, step_a.DRAFT_FUSED, step_a.DRAFT_FUSED + DRAFT_GROUPS)
            for a, b in (("perf=p.parent/'step-a-grade.json'", "perf=p.parent/'step-b-grade.json'"),
                         ("'missing':['step-a-grade-absent']", "'missing':['step-b-grade-absent']"),
                         ('Every cap is step_a_gate.PHASES.', 'Every cap is step_b_gate.PHASES.'),
                         ('Alpha-perf text above; every cap here is step_a_gate.\n', 'Step A text above; every cap here is step_b_gate.\n')):
                text = replace_once(text, a, b)
            first = text.index('# Step A: this queue compares')
            text = text[:first] + HEADER_NOTE + text[first:]
        elif suffix == '_settings.sh':
            text = replace_once(text, SETTINGS_FUSED, SETTINGS_FUSED + GROUPS + ' ')
            text = replace_once(text, SETTINGS_ANCHOR, SETTINGS_NOTE + SETTINGS_ANCHOR)
        else:
            text = replace_once(text, step_a.DECLARE_FUSED, step_a.DECLARE_FUSED + DECLARE_GROUPS)
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
    print('step-b queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
