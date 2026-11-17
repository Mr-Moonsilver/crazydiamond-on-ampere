#!/usr/bin/env python3
"""Derive the q18a-stage1-step box scripts from the committed q18a-alpha-next-c1 texts. Nothing a past queue ran is edited.

Kept from alpha-next-c1, byte for byte: everything c1 kept from alpha-next (pins, preflight, startup, bind, host headroom, the unarmed check, the end identity, both band manifests and their
/tokenize guards, the death watch, the M3 allocator receipts, the owned cleanup and restore tail, STOP_RACE), c1's switches and asserts, the core-then-tail structure and the read. Attn-meta's form
by name (claude/attn-meta-c1b: tools/attn_meta_queue{,_b}.py, never edited there). Declared deltas, each applied exactly once (counts asserted) or, inside the body, on every occurrence:
  names      runs/q18a_alpha_next_c1_{settings,declare}.sh and .draft.sh -> q18a_stage1_step_*, QNAME and RUN_ID q18a-alpha-next-c1 -> q18a-stage1-step, the records
             alpha-next-c1-{grade,read} -> stage1-step-*, the box e3_alpha_next_c1_box -> e3_stage1_step_box, the gate alpha_next_c1_gate -> stage1_step_gate;
  switch     FLASH_S5_LAZY_RECV=1 FLASH_S5_DECODE_MERGE=1 in EXTRA_ENV after c1's switches (installed, not selected), asserted in the declaration and the preflight beside c1's;
  pins       the declaration pins c1's files and adds this box, its gate, table and reads and the modules it imports (e3_alpha_next_c1b_box.py, alpha_next_c1b_arms.py, e3_c1_open_rows.py), the _h
             death files too (the old files stay pinned: old tests in the bundle use them); the draft's interface list the same;
  death      the lead's health-scoped rule (3 Nov 16:49; returning-append 14904f18, byte for byte): every name of native43_engine_death.py and alpha_next_stop_race.py in the draft -> their _h names;
  body       c1's body with: this table's core list (no tail: the list is empty), the arm's --c1 is the lever selection (soff, sl1, sl12, sl12a), the grade and read of e3_stage1_step_box.
No model setting, gate threshold, launcher or restore change. The arm lists come from harness/stage1_step_arms.py; --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import alpha_next_c1_queue as c1q  # noqa: E402
import alpha_next_queue as alpha_next  # noqa: E402
import stage1_step_arms as table  # noqa: E402

OLD, NEW = c1q.NEW, 'q18a_stage1_step'
OLD_Q, NEW_Q = c1q.NEW_Q, 'q18a-stage1-step'
SWITCH_NAMES = ('FLASH_S5_LAZY_RECV', 'FLASH_S5_DECODE_MERGE')
SWITCH = ' '.join(name + '=1' for name in SWITCH_NAMES)
CORE = tuple('%s:%s' % (r['part'], r['label']) for r in table.arms(table.GRADED))
TAIL = tuple(table.labels('tail'))
HEADER_NOTE = ('# stage1-step: this queue compares L1 lazy PP receive and L2 decode merge (results/hub/stage1-step/DESIGN-stage1-step.md) against the base, on the alpha-next-c1b engine plus\n'
               '# FLASH_S5_LAZY_RECV=1 FLASH_S5_DECODE_MERGE=1 (installed, not selected), every arm C1 defer and selecting its levers (native_stage_local {local, overlap, defer_bind, lazy_recv,\n'
               '# decode_merge}), its group size and burst and fused by typed control while paused idle FULL: %d core arms, band 1 and band 2, live and solo, matched, then one band 1 live\n'
               '# always-merged pair; no tail. Mechanism gates are c1\'s plus the selection and the counters. NUMERICS ARE NOT COVERED: selected arms are speed reads; batch 8 against 2 x 4 changes\n'
               '# batch composition, so OUTSIDE on merged arms is expected and is not a defect; speed, decode, TTFT, FIFO wait, handler time, pending replies and memory are reads only;\n'
               '# release readiness is not assessed.\n#\n' % len(table.arms('core')))
BODY_EDITS = (
    ("# alpha-next-c1: the core arms (CORE below) then, once the core grade is written, the g4 tail (TAIL below, reads only). Each\n"
     "# arm selects burst, fused, its group size, C0 local and its C1 selection by typed control while paused idle FULL.",
     "# stage1-step: the core arms (CORE below), then the core grade; this table has no tail (the list below is empty). Each arm\n"
     "# selects burst, fused, its group size, C0 local, C1 defer and its lever selection by typed control while paused idle FULL."),
    ("# descriptive. Every timeout below is one alpha_next_c1_gate cap (instruments, arm_entries, alpha_next_c1_arms, ingress,",
     "# descriptive. Every timeout below is one stage1_step_gate cap (instruments, arm_entries, stage1_step_arms, ingress,"),
    ('python3 "$H/e3_alpha_next_c1_box.py"', 'python3 "$H/e3_stage1_step_box.py"'),
    ("'auto arm: 8 standing decoders, burst, fused, the group size, C0 local, the C1 selection and defer_bind, fresh, append; live keeps them, solo cancels'",
     "'auto arm: 8 standing decoders, burst, fused, the group size, C0 local, C1 defer, the lever selection, fresh, append; live keeps them, solo cancels'"),
    ('$Q/alpha-next-c1-', '$Q/stage1-step-'),
    ("record alpha-next-c1 \"$?\" 'core arms: per-arm mechanism (crash-free, TP commitment, group, C0 and C1 engagement, one command in "
     "flight per stage, OOM), numerics envelope, complete progress lines; speed, decode and memory descriptive'",
     "record stage1-step \"$?\" 'core arms: per-arm mechanism (crash-free, TP commitment, group, C0 and C1 engagement, one command in "
     "flight per stage, OOM), the lever selection and counters, complete progress lines; speed, decode, FIFO wait and memory reads, numerics not covered'"),
    ('read_step alpha-next-c1-read "$?"', 'read_step stage1-step-read "$?"'),
)
DRAFT_DEATH = (('native43_engine_death.py', 'native43_engine_death_h.py', 3),
               ('alpha_next_stop_race.py', 'alpha_next_stop_race_h.py', 3))
DRAFT_NAMES = (
    ('from alpha_next_c1_gate import ACTIVE_S', 'from stage1_step_gate import ACTIVE_S', 1),
    ('(alpha_next_c1_gate, bash SECONDS)', '(stage1_step_gate, bash SECONDS)', 1),
    ('every cap here is alpha_next_c1_gate.\n', 'every cap here is stage1_step_gate.\n', 1),
    ('Every cap is alpha_next_c1_gate.PHASES.', 'Every cap is stage1_step_gate.PHASES.', 1),
    ('"$H/runs/q18a_alpha_next_c1_settings.sh"', '"$H/runs/q18a_stage1_step_settings.sh"', 1),
    ('-r $H/runs/q18a_alpha_next_c1_settings.sh &&', '-r $H/runs/q18a_stage1_step_settings.sh &&', 1),
    ('RUN_ID="q18a-alpha-next-c1-$(date', 'RUN_ID="q18a-stage1-step-$(date', 1),
    ("perf=p.parent/'alpha-next-c1-grade.json'", "perf=p.parent/'stage1-step-grade.json'", 1),
    ("'missing':['alpha-next-c1-grade-absent']", "'missing':['stage1-step-grade-absent']", 1),
    ("tail=p.parent/'alpha-next-c1-read.json'", "tail=p.parent/'stage1-step-read.json'", 1),
)
DECLARE_NAMES = (
    ('QNAME=${QNAME:-q18a-alpha-next-c1}', 'QNAME=${QNAME:-q18a-stage1-step}', 1),
    ('"$H/runs/q18a_alpha_next_c1_settings.sh"', '"$H/runs/q18a_stage1_step_settings.sh"', 2),
    ('"$H/runs/q18a_alpha_next_c1.draft.sh"', '"$H/runs/q18a_stage1_step.draft.sh"', 2),
    ('"$H/runs/q18a_alpha_next_c1_declare.sh"', '"$H/runs/q18a_stage1_step_declare.sh"', 1),
)
SETTINGS_NAMES = (('QNAME=${QNAME:-q18a-alpha-next-c1}', 'QNAME=${QNAME:-q18a-stage1-step}', 1),)
FILES = 'e3_stage1_step_box.py stage1_step_gate.py stage1_step_arms.py stage1_step_read.py'
BOX_FILES = 'e3_alpha_next_c1b_box.py alpha_next_c1b_arms.py e3_c1_open_rows.py'
H_FILES = 'native43_engine_death_h.py alpha_next_stop_race_h.py'
INTERFACE_OLD = c1q.INTERFACE_NEW
INTERFACE_NEW = INTERFACE_OLD[:-len('; do')] + ' \\\n  ' + FILES + ' \\\n  ' + BOX_FILES + '; do'
PINS_OLD = c1q.PINS_NEW
PINS_NEW = PINS_OLD + FILES + ' ' + BOX_FILES + ' ' + H_FILES + ' '
SETTINGS_OLD = c1q.SETTINGS_OLD + c1q.SWITCH + ' '
SETTINGS_ANCHOR = c1q.SETTINGS_NOTE.splitlines()[0]
SETTINGS_NOTE = ('# stage1-step: FLASH_S5_LAZY_RECV=1 FLASH_S5_DECODE_MERGE=1 below (installed, not selected), for the whole run, the only settings\n'
                 '# added to alpha-next-c1; each arm selects its levers by the typed native_stage_local control.\n')
DRAFT_SWITCHES = c1q.DRAFT_SWITCHES + ''.join("assert env.get('%s') == '1', 'stage1-step lever installed'\n" % n for n in SWITCH_NAMES)
DECLARE_SWITCHES = c1q.DECLARE_SWITCHES + ''.join("assert env.get('%s')=='1', 'stage1-step runs with the lever installed'\n" % n for n in SWITCH_NAMES)


def c1_template():
    """c1's body before its arm lists were filled in (c1q.body's own steps, placeholders kept)."""
    text = alpha_next.BODY
    for old, new in c1q.BODY_EDITS:
        assert old in text, old[:100]
        text = text.replace(old, new)
    return text


def body():
    text = c1_template()
    for old, new in BODY_EDITS:
        assert old in text, old[:100]
        text = text.replace(old, new)
    return text.replace('@CORE@', ' '.join(CORE)).replace('@TAIL@', ' '.join(TAIL)).replace('@BAND1@', table.band1_glob())


def replace_once(text, old, new, count=1):
    assert text.count(old) == count, (old[:120], text.count(old), count)
    return text.replace(old, new)


def derive():
    out = {}
    for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
        text = (ROOT / 'harness/runs' / (OLD + suffix)).read_text()
        if suffix == '.draft.sh':
            text = replace_once(text, c1q.body(), body())
            text = replace_once(text, INTERFACE_OLD, INTERFACE_NEW)
            text = replace_once(text, c1q.DRAFT_SWITCHES, DRAFT_SWITCHES)
            for old, new, count in DRAFT_NAMES + DRAFT_DEATH:
                text = replace_once(text, old, new, count)
            first = text.index('# alpha-next-c1: this queue compares')
            text = text[:first] + HEADER_NOTE + text[first:]
        elif suffix == '_settings.sh':
            text = replace_once(text, SETTINGS_OLD, SETTINGS_OLD + SWITCH + ' ')
            text = replace_once(text, SETTINGS_ANCHOR, SETTINGS_NOTE + SETTINGS_ANCHOR)
            for old, new, count in SETTINGS_NAMES:
                text = replace_once(text, old, new, count)
        else:
            text = replace_once(text, c1q.DECLARE_SWITCHES, DECLARE_SWITCHES)
            text = replace_once(text, PINS_OLD, PINS_NEW)
            for old, new, count in DECLARE_NAMES:
                text = replace_once(text, old, new, count)
        out['harness/runs/' + NEW + suffix] = text
    return out


def table_text():
    import stage1_step_gate as gate
    lines = ['q18a-stage1-step arm table (harness/stage1_step_arms.py); entry limits relative to the run start, never a clock',
             '%-3s %-16s %-6s %-5s %-6s %-4s %-6s %-4s %-5s %s' % ('#', 'label', 'part', 'lazy', 'merge', 'size', 'cohort', 'band', 'round', 'enters while SECONDS <=')]
    for k, row in enumerate(gate.rows(), 1):
        lines.append('%-3d %-16s %-6s %-5s %-6s %-4d %-6s %-4s %-5s %s %d s' % (
            k, row['label'], row['part'], 'on' if row['lazy_recv'] else 'off', row['decode_merge'], row['size'], row['cohort'], row['band'], row['index'],
            row['entry_limit'], gate.ENTRY_LIMIT_S[row['entry_limit']]))
    exp = gate.EXPECTED_RUNTIME_S
    e, slow = exp['nominal'], exp['slower']
    lines.append('arms %d (core %d, tail %d); box budget %d s, cleanup included; worst case of the caps %d s' % (
        exp['arms'], exp['core_arms'], exp['tail_arms'], gate.BOX_BUDGET_S, gate.WORST_CASE_SUM_S))
    lines.append('expected (c1 walls): %d arms entering, last arm ends +%d s, about %.1f min with startup and the end; smallest entry margin %d s' % (
        e['arms_entering'], e['last_arm_ends_s'], e['total_min'], e['min_margin_s']))
    lines.append('at %.1fx the walls: %d arms entering, about %.1f min; not run: %s' % (
        gate.SLOWER, slow['arms_entering'], slow['total_min'], ', '.join(slow['not_run']) or 'none'))
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--check', action='store_true')
    p.add_argument('--table', action='store_true', help='print the arm table with its entry limits; write nothing')
    p.add_argument('--replace-derived', action='store_true',
                   help='write a derived text that differs from the one on disk (only for a text no queue has run)')
    a = p.parse_args(argv)
    if a.table:
        print(table_text())
        return
    for name, text in derive().items():
        path = ROOT / name
        if a.check:
            assert path.read_text() == text, name
        else:
            assert a.replace_derived or not path.exists() or path.read_text() == text, 'refusing to change an existing derived text: ' + name
            path.write_text(text)
            path.chmod(0o664)
    print('stage1-step queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
