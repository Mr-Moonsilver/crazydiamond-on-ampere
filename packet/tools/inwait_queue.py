#!/usr/bin/env python3
"""Derive the q18a-inwait box scripts from the committed q18a-compose-2 texts. Nothing a past queue ran is edited. Discovery mode.

Declared deltas over compose-2's three texts, each by exact count:
  names     every compose-2 name -> inwait (the queue, records, box, gate, table);
  switch    FLASH_S5_INWAIT=1 in EXTRA_ENV after FLASH_ATTN_META_GDN_BATCH=1 (installed, not selected), asserted in the declaration and the preflight beside them;
  body      this table's core list; one decode-only baseline (decode-am1), not two.
The arm lists come from harness/inwait_arms.py; --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import compose2_queue as cq  # noqa: E402
import inwait_arms as table  # noqa: E402

OLD, NEW = cq.NEW, 'q18a_inwait'
OLD_Q, NEW_Q = cq.NEW_Q, 'q18a-inwait'
SWITCH_NAME = 'FLASH_S5_INWAIT'
CORE = ' '.join('%s:%s' % (r['part'], r['label']) for r in table.arms(table.GRADED))
DECODE = ' '.join(d['label'] for d in table.DECODE_ARMS)
NAMES = ((OLD, NEW), (OLD_Q, NEW_Q), ('COMPOSE2', 'INWAIT'), ('compose2', 'inwait'), ('compose-2', 'inwait'))
FILES = 'e3_inwait_box.py inwait_arms.py e3_attn_meta_box.py attn_meta_arms.py'
HEADER_NOTE = ('# inwait: native handoffs inside the stock wait, discovery mode: FLASH_S5_INWAIT installed for the whole run beside attn-meta and L1 + L2, all selected per arm by typed control while paused idle FULL: iw0 off,\n'
               '# iw1 on; one decode-only baseline first; band 1 live x2 and solo x1 each (harness/inwait_arms.py). NUMERICS ARE NOT COVERED by this queue either: speed, decode, handler time and memory are reads only.\n#\n')
SETTINGS_NOTE = '# inwait: FLASH_S5_INWAIT=1 below (installed, not selected), for the whole run, the only setting added to compose-2; each arm selects it by typed control.\n'
SETTINGS_ANCHOR = '# compose-2: FLASH_ATTN_META_GDN_BATCH=1'
DRAFT_ASSERT = "assert env.get('FLASH_ATTN_META_GDN_BATCH') == '1', 'attn-meta installed'\n"
DECLARE_ASSERT = "assert env.get('FLASH_ATTN_META_GDN_BATCH')=='1', 'compose-2 runs with attn-meta installed'\n"
AM0_CASE = '    (decode-am0) _attn=off;;\n'


def once(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)


def renames(text):
    for old, new in NAMES:
        text = text.replace(old, new)
    return text


def body():
    text = once(cq.body(), cq.CORE, CORE)
    text = once(text, '  for _label in ' + cq.DECODE + '; do', '  for _label in ' + DECODE + '; do')
    text = once(text, AM0_CASE, '')
    return renames(text)


def draft(text):
    text = once(text, cq.body(), body())
    text = once(text, DRAFT_ASSERT, DRAFT_ASSERT + "assert env.get('%s') == '1', 'in-wait installed'\n" % SWITCH_NAME)
    text = once(text, '# compose-2: attn-meta and stage1-step', HEADER_NOTE + '# compose-2: attn-meta and stage1-step')
    return renames(text)


def settings(text):
    text = once(text, 'FLASH_ATTN_META_GDN_BATCH=1 FLASH_S5_REPLAY_LIB=', 'FLASH_ATTN_META_GDN_BATCH=1 ' + SWITCH_NAME + '=1 FLASH_S5_REPLAY_LIB=')
    text = once(text, SETTINGS_ANCHOR, SETTINGS_NOTE + SETTINGS_ANCHOR)
    return renames(text)


def declare(text):
    text = once(text, DECLARE_ASSERT, DECLARE_ASSERT + "assert env.get('%s')=='1', 'inwait runs with the lever installed'\n" % SWITCH_NAME)
    return renames(text)


def derive():
    out = {}
    for suffix, edit in (('.draft.sh', draft), ('_settings.sh', settings), ('_declare.sh', declare)):
        out['harness/runs/' + NEW + suffix] = edit((ROOT / 'harness/runs' / (OLD + suffix)).read_text())
    return out


def table_text():
    import inwait_gate as gate
    lines = ['q18a-inwait arm table (harness/inwait_arms.py); entry limits relative to the run start, never a clock',
             '%-3s %-16s %-6s %-6s %-5s %-4s %-6s %-5s %s' % ('#', 'label', 'part', 'inwait', 'attn', 'size', 'cohort', 'round', 'enters while SECONDS <=')]
    for k, row in enumerate(gate.rows(), 1):
        lines.append('%-3d %-16s %-6s %-6s %-5s %-4d %-6s %-5s %s %d s' % (
            k, row['label'], row['part'], 'on' if row.get('inwait') else 'off', 'on' if row['attn'] else 'off', row['size'], row['cohort'], row.get('index', '-'),
            row['entry_limit'], gate.ENTRY_LIMIT_S[row['entry_limit']]))
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
    print('inwait queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
