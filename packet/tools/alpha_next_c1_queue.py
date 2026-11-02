#!/usr/bin/env python3
"""Derive the q18a-alpha-next-c1 box scripts from the committed q18a-alpha-next texts. Nothing a past queue ran is edited.

Kept from alpha-next, byte for byte: everything alpha-next kept from Step B (pins, preflight, startup, bind, host headroom, the
unarmed check, the end identity, both band manifests and their /tokenize guards, the le-fix death scan and watch, the M3 allocator
receipts, the owned cleanup and restore tail), alpha-next's switches and asserts, its STOP_RACE hook (alpha_next_stop_race.py,
unchanged), the core-then-tail structure and the read. Declared deltas, each applied exactly once (counts asserted):
  names      the queue's own files and records: runs/q18a_alpha_next_{settings,declare}.sh and .draft.sh -> q18a_alpha_next_c1_*,
             QNAME and RUN_ID q18a-alpha-next -> q18a-alpha-next-c1, the grade and read records alpha-next-{grade,read,report}
             -> alpha-next-c1-*, the box e3_alpha_next_box -> e3_alpha_next_c1_box, the gate alpha_next_gate ->
             alpha_next_c1_gate (the budget reader and the cap comments); the history comments keep their names;
  budget     the draft reads alpha_next_c1_gate: alpha-next's formulas, relative to the run's own start only (ACTIVE_S,
             LAST_ARM_ENTRY_S for core arms, LAST_TAIL_ENTRY_S for the g4 tail), no window;
  switches   FLASH_S5_STAGE_OVERLAP=1 in EXTRA_ENV after alpha-next's FLASH_S5_HANDLER_ANATOMY=1, asserted in the declaration and
             the preflight beside alpha-next's asserts;
  pins       the declaration pins alpha-next's files and adds e3_alpha_next_c1_box.py alpha_next_c1_gate.py
             alpha_next_c1_arms.py (the draft's interface list too). No bundle, manifest or package digest is written here
             (alpha-next's way);
  body       alpha-next's body with the C1 selection in place of the C0 selection: labels ``c1on.g2-live-0``, the arm call
             ``--c1 on|off`` (C0 local is selected by the box in every arm), the core arms (alpha_next_c1_arms part 'core'),
             the core grade (the queue's verdict), then the g4 tail (part 'tail', reads only) and the read over every arm.
No model setting, gate threshold, launcher or restore change. The arm lists come from harness/alpha_next_c1_arms.py; --check
verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import alpha_next_queue as alpha_next  # noqa: E402
import alpha_next_c1_arms as table  # noqa: E402

OLD, NEW = alpha_next.NEW, 'q18a_alpha_next_c1'
OLD_Q, NEW_Q = alpha_next.NEW_Q, 'q18a-alpha-next-c1'
SWITCH = 'FLASH_S5_STAGE_OVERLAP=1'
CORE = tuple('%s:%s' % (r['part'], r['label']) for r in table.arms(table.GRADED))
TAIL = tuple(table.labels('tail'))
HEADER_NOTE = ('# alpha-next-c1: this queue compares C1 (two chunks in flight, one command in flight per stage) with C0\'s local\n'
               '# path on the alpha-next engine (fused, layer groups and C0 stage-local installed, plus FLASH_S5_STAGE_OVERLAP=1),\n'
               '# matched per arm by the typed native_stage_local {local: true, overlap} control: a core of %d arms (band 1 ABBA C1\n'
               '# on/off, live and solo, at size 2 over %d rounds; band 2 matched at size 2), whose grade is the queue verdict, then a\n'
               '# g4 tail of %d C1-on arms, reads only. Mechanism gates (crash-free, TP commitment, group, C0 and C1 engagement, one\n'
               '# command in flight per stage, OOM); numerics on the greedy standing streams, speed, decode and memory are reported;\n'
               '# release readiness is not assessed.\n#\n' % (
                   len(table.arms('core')), table.BAND1_ROUNDS, len(TAIL)))

# The body: alpha-next's, with these edits (each old text must be there; every occurrence replaced).
BODY_EDITS = (
    ("# alpha-next: the core arms (CORE below) then, once the core grade is written, the g4 tail (TAIL below, reads only). Each arm\n"
     "# selects burst, fused, its group size and its C0 selection by typed control while paused idle FULL.",
     "# alpha-next-c1: the core arms (CORE below) then, once the core grade is written, the g4 tail (TAIL below, reads only). Each\n"
     "# arm selects burst, fused, its group size, C0 local and its C1 selection by typed control while paused idle FULL."),
    ("# descriptive. Every timeout below is one alpha_next_gate cap (instruments, arm_entries, alpha_next_arms, ingress,",
     "# descriptive. Every timeout below is one alpha_next_c1_gate cap (instruments, arm_entries, alpha_next_c1_arms, ingress,"),
    ("# FULL run after the tail and are counted). An extra arm (band 1's round past the required two) runs only while the core entry\n"
     "# limit allows; once entered it counts as a core arm, and past the limit its NOT_RUN is a read.",
     "# FULL run after the tail and are counted). This table has no extra arm; the extra branch below is alpha-next's, unused."),
    ('_pair _c0 _group _workload', '_pair _c1 _group _workload'),
    ('IFS=. read -r _c0 _group <<< "$_sel"', 'IFS=. read -r _c1 _group <<< "$_sel"'),
    ('python3 "$H/e3_alpha_next_box.py"', 'python3 "$H/e3_alpha_next_c1_box.py"'),
    ('--stage-local "${_c0#c0}"', '--c1 "${_c1#c1}"'),
    ("'auto arm: 8 standing decoders, burst, fused, the group size and the C0 selection, fresh, append; live keeps them, solo cancels'",
     "'auto arm: 8 standing decoders, burst, fused, the group size, C0 local and the C1 selection, fresh, append; live keeps them, solo cancels'"),
    ('$Q/alpha-next-', '$Q/alpha-next-c1-'),
    ("record alpha-next \"$?\" 'core arms: per-arm mechanism (crash-free, TP commitment, group and C0 engagement), numerics envelope, "
     "complete progress lines; speed, decode and memory descriptive'",
     "record alpha-next-c1 \"$?\" 'core arms: per-arm mechanism (crash-free, TP commitment, group, C0 and C1 engagement, one command in "
     "flight per stage, OOM), numerics envelope, complete progress lines; speed, decode and memory descriptive'"),
    ('read_step alpha-next-read "$?"', 'read_step alpha-next-c1-read "$?"'),
)


def body():
    text = alpha_next.BODY
    for old, new in BODY_EDITS:
        assert old in text, old[:100]
        text = text.replace(old, new)
    return text.replace('@CORE@', ' '.join(CORE)).replace('@TAIL@', ' '.join(TAIL)).replace('@BAND1@', table.band1_glob())


# Name deltas outside the body: (old, new, occurrences) per file suffix.
DRAFT_NAMES = (
    ('from alpha_next_gate import ACTIVE_S', 'from alpha_next_c1_gate import ACTIVE_S', 1),
    ('(alpha_next_gate, bash SECONDS)', '(alpha_next_c1_gate, bash SECONDS)', 1),
    ('every cap here is alpha_next_gate.\n', 'every cap here is alpha_next_c1_gate.\n', 1),
    ('Every cap is alpha_next_gate.PHASES.', 'Every cap is alpha_next_c1_gate.PHASES.', 1),
    ('"$H/runs/q18a_alpha_next_settings.sh"', '"$H/runs/q18a_alpha_next_c1_settings.sh"', 1),
    ('-r $H/runs/q18a_alpha_next_settings.sh &&', '-r $H/runs/q18a_alpha_next_c1_settings.sh &&', 1),
    ('RUN_ID="q18a-alpha-next-$(date', 'RUN_ID="q18a-alpha-next-c1-$(date', 1),
    ("perf=p.parent/'alpha-next-grade.json'", "perf=p.parent/'alpha-next-c1-grade.json'", 1),
    ("'missing':['alpha-next-grade-absent']", "'missing':['alpha-next-c1-grade-absent']", 1),
    ("tail=p.parent/'alpha-next-read.json'", "tail=p.parent/'alpha-next-c1-read.json'", 1),
)
DECLARE_NAMES = (
    ('QNAME=${QNAME:-q18a-alpha-next}', 'QNAME=${QNAME:-q18a-alpha-next-c1}', 1),
    ('"$H/runs/q18a_alpha_next_settings.sh"', '"$H/runs/q18a_alpha_next_c1_settings.sh"', 2),
    ('"$H/runs/q18a_alpha_next.draft.sh"', '"$H/runs/q18a_alpha_next_c1.draft.sh"', 2),
    ('"$H/runs/q18a_alpha_next_declare.sh"', '"$H/runs/q18a_alpha_next_c1_declare.sh"', 1),
)
SETTINGS_NAMES = (
    ('QNAME=${QNAME:-q18a-alpha-next}', 'QNAME=${QNAME:-q18a-alpha-next-c1}', 1),
)
C1_FILES = 'e3_alpha_next_c1_box.py alpha_next_c1_gate.py alpha_next_c1_arms.py'
INTERFACE_OLD = alpha_next.INTERFACE_NEW
INTERFACE_NEW = INTERFACE_OLD[:-len('; do')] + ' \\\n  ' + C1_FILES + '; do'
PINS_OLD = alpha_next.PINS_NEW
PINS_NEW = PINS_OLD + C1_FILES + ' '
SETTINGS_OLD = alpha_next.SETTINGS_OLD + alpha_next.SWITCHES + ' '
SETTINGS_ANCHOR = alpha_next.SETTINGS_NOTE.splitlines()[0]
SETTINGS_NOTE = ('# alpha-next-c1: FLASH_S5_STAGE_OVERLAP=1 (C1) below, the only setting added to alpha-next; each arm selects C1 on\n'
                 '# or off (with C0 local) by typed control.\n')
DRAFT_SWITCHES = alpha_next.DRAFT_SWITCHES + "assert env.get('FLASH_S5_STAGE_OVERLAP') == '1', 'alpha-next-c1 stage-overlap switch (C1)'\n"
DECLARE_SWITCHES = (alpha_next.DECLARE_SWITCHES
                    + "assert env.get('FLASH_S5_STAGE_OVERLAP')=='1', 'alpha-next-c1 runs with the stage-overlap switch installed'\n")


def replace_once(text, old, new, count=1):
    assert text.count(old) == count, (old[:120], text.count(old), count)
    return text.replace(old, new)


def derive():
    out = {}
    for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
        text = (ROOT / 'harness/runs' / (OLD + suffix)).read_text()
        if suffix == '.draft.sh':
            text = replace_once(text, alpha_next.body(), body())
            text = replace_once(text, INTERFACE_OLD, INTERFACE_NEW)
            text = replace_once(text, alpha_next.DRAFT_SWITCHES, DRAFT_SWITCHES)
            for old, new, count in DRAFT_NAMES:
                text = replace_once(text, old, new, count)
            first = text.index('# alpha-next: this queue compares')
            text = text[:first] + HEADER_NOTE + text[first:]
        elif suffix == '_settings.sh':
            text = replace_once(text, SETTINGS_OLD, SETTINGS_OLD + SWITCH + ' ')
            text = replace_once(text, SETTINGS_ANCHOR, SETTINGS_NOTE + SETTINGS_ANCHOR)
            for old, new, count in SETTINGS_NAMES:
                text = replace_once(text, old, new, count)
        else:
            text = replace_once(text, alpha_next.DECLARE_SWITCHES, DECLARE_SWITCHES)
            text = replace_once(text, PINS_OLD, PINS_NEW)
            for old, new, count in DECLARE_NAMES:
                text = replace_once(text, old, new, count)
        out['harness/runs/' + NEW + suffix] = text
    return out


def table_text():
    """The arm table as the draft runs it, with each part's relative entry limit and the expected runtime (descriptive)."""
    import alpha_next_c1_gate as gate
    lines = ['q18a-alpha-next-c1 arm table (harness/alpha_next_c1_arms.py); entry limits relative to the run start, never a clock',
             '%-3s %-16s %-5s %-4s %-4s %-6s %-4s %-5s %s' % ('#', 'label', 'part', 'c1', 'size', 'cohort', 'band', 'round',
                                                          'enters while SECONDS <=')]
    for k, row in enumerate(table.arms(), 1):
        lines.append('%-3d %-16s %-5s %-4s %-4d %-6s %-4d %-5d %s %d s' % (
            k, row['label'], row['part'], row['c1'], row['size'], row['cohort'], row['band'], row['index'], row['entry_limit'],
            gate.ENTRY_LIMIT_S[row['entry_limit']]))
    exp = gate.EXPECTED_RUNTIME_S
    e = exp['nominal']
    lines.append('arms %d (core %d, tail %d); C0 local in every arm; standing decoders %d: greedy %s at temperature %s, sampled %s; '
                 'box budget %d s, cleanup included' % (exp['arms'], exp['core_arms'], exp['tail_arms'], table.STANDING_DECODERS,
                                                       list(table.STANDING_GREEDY), table.GREEDY_TEMPERATURE,
                                                       list(table.STANDING_SAMPLED), gate.BOX_BUDGET_S))
    lines.append('expected (lead walls): arms entering %d (core %d, tail %d); last core arm enters +%d s, last arm ends +%d s, about '
                 '%.1f min with the end; smallest entry margin %d s' % (
                     e['arms_entering'], e['core_arms_entering'], e['tail_arms_entering'], e['last_core_arm_enters_s'],
                     e['last_arm_ends_s'], e['total_min'], e['min_margin_s']))
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--check', action='store_true')
    p.add_argument('--table', action='store_true', help='print the arm table with its entry limits; write nothing')
    a = p.parse_args(argv)
    if a.table:
        print(table_text())
        return
    for name, text in derive().items():
        path = ROOT / name
        if a.check:
            assert path.read_text() == text, name
        else:
            assert not path.exists() or path.read_text() == text, 'refusing to change an existing derived text: ' + name
            path.write_text(text)
            path.chmod(0o664)
    print('alpha-next-c1 queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
