#!/usr/bin/env python3
"""Derive the q18a-compose-2 box scripts from the committed q18a-stage1-step texts. Nothing a past queue ran is edited. Discovery mode.

Declared deltas over stage1-step's three texts, each by exact count:
  names     q18a_stage1_step -> q18a_compose_2, q18a-stage1-step -> q18a-compose-2, the records stage1-step-* -> compose-2-*, the box, gate and table names;
  switch    FLASH_ATTN_META_GDN_BATCH=1 in EXTRA_ENV after FLASH_S5_DECODE_MERGE=1 (all installed, none selected), asserted in the declaration and the preflight beside them;
  pins      the declaration and the draft's interface list add this box and table and attn-meta's box and table;
  body      this table's core list and band 1 glob; the two decode-only baselines first, attn-meta's function by name through this box's decode-arm.
The arm lists come from harness/compose2_arms.py; --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import attn_meta_queue as aq  # noqa: E402
import compose2_arms as table  # noqa: E402
import stage1_step_queue as sq  # noqa: E402

OLD, NEW = sq.NEW, 'q18a_compose_2'
OLD_Q, NEW_Q = sq.NEW_Q, 'q18a-compose-2'
SWITCH_NAME = 'FLASH_ATTN_META_GDN_BATCH'
OLD_CORE = ' '.join(sq.CORE)
CORE = ' '.join('%s:%s' % (r['part'], r['label']) for r in table.arms(table.GRADED))
DECODE = ' '.join(d['label'] for d in table.DECODE_ARMS)
# stage1-step's list stays with its gate renamed, so that list holds compose2_gate.py; its box, table and reads stay: this box imports them.
FILES = 'e3_compose2_box.py compose2_arms.py e3_attn_meta_box.py attn_meta_arms.py'
NAMES = ((OLD, NEW), (OLD_Q, NEW_Q), ('e3_stage1_step_box.py" ', 'e3_compose2_box.py" '), ('stage1_step_gate', 'compose2_gate'), ('$Q/stage1-step-', '$Q/compose-2-'),
         ("p.parent/'stage1-step-", "p.parent/'compose-2-"), ("'stage1-step-grade-absent'", "'compose-2-grade-absent'"),
         ('record stage1-step "$?"', 'record compose-2 "$?"'), ('read_step stage1-step-read', 'read_step compose-2-read'),
         ('stage1_step_arms, ingress', 'compose2_arms, ingress'))
HEADER_NOTE = ('# compose-2: attn-meta and stage1-step L1 + L2 stacked, discovery mode: every lever flag installed for the whole run, each selected per arm by typed control while paused idle FULL: b0 both off,\n'
               '# am attn-meta only, mg L1 + L2 only, amg both; two decode-only baselines first; band 1 live x2 and solo x1 for b0 and amg, band 2 live b0 and amg (harness/compose2_arms.py). NUMERICS ARE NOT\n'
               '# COVERED: speed, decode, handler time and memory are reads only.\n#\n')
SETTINGS_NOTE = ('# compose-2: FLASH_ATTN_META_GDN_BATCH=1 below, beside FLASH_S5_LAZY_RECV=1 FLASH_S5_DECODE_MERGE=1, for the whole run (installed, not selected), the only setting added to\n'
                 '# stage1-step; each arm selects each lever by typed control.\n')
SETTINGS_ANCHOR = '# stage1-step: FLASH_S5_LAZY_RECV=1'
DRAFT_ASSERT = "assert env.get('FLASH_S5_DECODE_MERGE') == '1', 'stage1-step lever installed'\n"
DECLARE_ASSERT = "assert env.get('FLASH_S5_DECODE_MERGE')=='1', 'stage1-step runs with the lever installed'\n"
LOOP = '  for _arm in ' + CORE + '; do\n'
LOOP_NEW = '  for _label in ' + DECODE + '; do\n    compose2_decode_arm "$_label"\n  done\n' + LOOP
FUNCTION_ANCHOR = 'GRADE_ARGS=(); READ_ARGS=()\n'
GLOB_OLD, GLOB_NEW = 'if [[ $_pair == [013] ]]; then', 'if [[ $_pair == %s ]]; then' % table.band1_glob()


def once(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)


def renames(text):
    for old, new in NAMES:
        text = text.replace(old, new)
    return text


def decode_function():
    """attn-meta's decode arm function, renamed: this box's decode-arm subcommand, this table's labels."""
    text = once(aq.DECODE_FUNCTION, '@DECODE_CASE@', '\n'.join('    (%s) _attn=%s;;' % (d['label'], 'on' if d['attn'] else 'off') for d in table.DECODE_ARMS))
    text = once(text, 'attn_meta_decode_arm() {', 'compose2_decode_arm() {')
    text = once(text, '# attn-meta: the standalone decode baselines (attn_meta_arms.DECODE_ARMS)', '# compose-2: the standalone decode baselines (compose2_arms.DECODE_ARMS, attn-meta\'s)')
    text = once(text, 'python3 "$H/e3_attn_meta_box.py" decode-arm', 'python3 "$H/e3_compose2_box.py" decode-arm')
    return once(text, 'then e3_attn_meta_box decode-arm', 'then e3_compose2_box decode-arm')


def body():
    text = once(sq.body(), OLD_CORE, CORE)
    text = once(text, LOOP, LOOP_NEW)
    text = once(text, GLOB_OLD, GLOB_NEW)
    text = once(text, FUNCTION_ANCHOR, decode_function() + FUNCTION_ANCHOR)
    return renames(text)


def draft(text):
    text = once(text, sq.BOX_FILES + '; do', sq.BOX_FILES + ' \\\n  ' + FILES + '; do')
    text = once(text, sq.body(), body())
    text = once(text, DRAFT_ASSERT, DRAFT_ASSERT + "assert env.get('%s') == '1', 'attn-meta installed'\n" % SWITCH_NAME)
    text = once(text, '# stage1-step: this queue compares', HEADER_NOTE + '# stage1-step: this queue compares')
    return renames(text)


def settings(text):
    text = once(text, sq.SWITCH + ' FLASH_S5_REPLAY_LIB=', sq.SWITCH + ' ' + SWITCH_NAME + '=1 FLASH_S5_REPLAY_LIB=')
    text = once(text, SETTINGS_ANCHOR, SETTINGS_NOTE + SETTINGS_ANCHOR)
    return renames(text)


def declare(text):
    text = once(text, DECLARE_ASSERT, DECLARE_ASSERT + "assert env.get('%s')=='1', 'compose-2 runs with attn-meta installed'\n" % SWITCH_NAME)
    text = once(text, sq.H_FILES + ' ', sq.H_FILES + ' ' + FILES + ' ')
    return renames(text)


def derive():
    out = {}
    for suffix, edit in (('.draft.sh', draft), ('_settings.sh', settings), ('_declare.sh', declare)):
        out['harness/runs/' + NEW + suffix] = edit((ROOT / 'harness/runs' / (OLD + suffix)).read_text())
    return out


def table_text():
    import compose2_gate as gate
    lines = ['q18a-compose-2 arm table (harness/compose2_arms.py); entry limits relative to the run start, never a clock',
             '%-3s %-16s %-6s %-5s %-6s %-5s %-4s %-6s %-4s %-5s %s' % ('#', 'label', 'part', 'attn', 'lazy', 'merge', 'size', 'cohort', 'band', 'round', 'enters while SECONDS <=')]
    for k, row in enumerate(gate.rows(), 1):
        lines.append('%-3d %-16s %-6s %-5s %-6s %-5s %-4d %-6s %-4s %-5s %s %d s' % (
            k, row['label'], row['part'], 'on' if row['attn'] else 'off', 'on' if row.get('lazy_recv') else 'off', row.get('decode_merge', 'off'), row['size'], row['cohort'], row['band'],
            row.get('index', '-'), row['entry_limit'], gate.ENTRY_LIMIT_S[row['entry_limit']]))
    exp = gate.EXPECTED_RUNTIME_S
    e, slow = exp['nominal'], exp['slower']
    lines.append('arms %d (decode %d, core %d, tail %d); box budget %d s, cleanup included; worst case of the caps %d s' % (
        exp['arms'], exp['decode_arms'], exp['core_arms'], exp['tail_arms'], gate.BOX_BUDGET_S, gate.WORST_CASE_SUM_S))
    lines.append('expected (c1 walls): %d arms entering, last arm ends +%d s, about %.1f min with startup and the end; smallest entry margin %d s' % (
        e['arms_entering'], e['last_arm_ends_s'], e['total_min'], e['min_margin_s']))
    lines.append('at %.1fx the walls: %d arms entering, about %.1f min; not run: %s' % (gate.SLOWER, slow['arms_entering'], slow['total_min'], ', '.join(slow['not_run']) or 'none'))
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
            path.write_text(text)
            path.chmod(0o664)
    print('compose-2 queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
