#!/usr/bin/env python3
"""Derive the q18a-a1-trunk box scripts from q18a-inwait's committed texts (tools/inwait_queue.py). Nothing a past queue ran is edited. Discovery mode.

Declared deltas over inwait's three texts, each by exact count:
  names     q18a-inwait -> q18a-a1-trunk and every inwait name of the queue, records, box, gate and table -> a1_trunk; FLASH_S5_INWAIT and the engine's in-wait stay;
  switch    the four FLASH_S5_APPEND_YIELD_* launch bounds in EXTRA_ENV after FLASH_S5_INWAIT=1 (installed; the yield is selected per arm), asserted in the declaration and the preflight;
  body      this table's core list; no tail arm; the ingress check's generation count read from the table (a live arm has its returners on top of 10);
  modules   the modules the box imports beyond inwait's (e3_inwait_box, inwait_arms, the refusal client) in the preflight's and the declaration's lists.
The arm lists come from harness/a1_trunk_arms.py; --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import inwait_queue as parent  # noqa: E402
import a1_trunk_arms as table  # noqa: E402

OLD, NEW = parent.NEW, 'q18a_a1_trunk'
OLD_Q, NEW_Q = parent.NEW_Q, 'q18a-a1-trunk'
OLD_CORE = parent.CORE
CORE = ' '.join('%s:%s' % (r['part'], r['label']) for r in table.arms(table.GRADED) if r['label'] != table.SMOKE_ARM)
BOUNDS = ' '.join('%s=%d' % kv for kv in table.LAUNCH_ENV.items())
NAMES = ((OLD, NEW), (OLD_Q, NEW_Q), ('inwait-', 'a1-trunk-'), ('inwait', 'a1_trunk'))
GEN_OLD = '--generation 10 --out "$Q/ingress-$_label.json"'
GEN_NEW = '--generation "$(python3 "$H/a1_trunk_arms.py" generations "$_label")" --out "$Q/ingress-$_label.json"'
GEN_TEXT = ("'access log: every request in the arm span is the arm (8 standing, fresh, append), else INVALID'", "'access log: every request in the arm span is the arm (8 standing, fresh, append, the returners), else INVALID'")
ARM_TEXT = ('fresh, append; live keeps them, solo cancels', 'the yield selection, fresh with the returners inside it, append; live keeps the rest, solo cancels')
GRADE_TEXT = ("'core arms: per-arm mechanism (crash-free, TP commitment, group, C0 and C1 engagement, one command in flight per stage, OOM), the lever selection and counters, complete progress lines; speed, decode, FIFO wait and memory reads, numerics not covered'",
              "'the arms: crash-free, attn-meta and startup, the yield selected only where on and served where on with returners, route generations add up; every speed and the budget lines are reads, numerics not covered'")
HEADER_NOTE = ('# a1-trunk: the returning-agent yield over build-3b\'s ps1 levers, discovery mode: the four FLASH_S5_APPEND_YIELD_* bounds installed for the whole run, the yield selected per arm by typed control at the arm\'s paused idle FULL entry.\n'
               '# The exact-path smoke first (its failure stops the table), one decode-only baseline, matched off/on live pairs with returning agents in the live replay, one solo (harness/a1_trunk_arms.py). NUMERICS ARE NOT COVERED: speed, decode and the budget lines are reads only.\n#\n')
SETTINGS_NOTE = '# a1-trunk: the four FLASH_S5_APPEND_YIELD_* launch bounds below (installed, not selected), for the whole run, the only setting added to the parent queue; each arm selects the yield by typed control.\n'
DRAFT_ASSERT = "assert env.get('FLASH_S5_INWAIT') == '1', 'in-wait installed'\n"
DECLARE_ASSERT = "assert env.get('FLASH_S5_INWAIT')=='1', 'inwait runs with the lever installed'\n"
MODULES = ' e3_inwait_box.py inwait_arms.py e3_native_refusal_client.py'
DRAFT_FILES = 'e3_inwait_box.py inwait_arms.py e3_attn_meta_box.py attn_meta_arms.py; do'
DECLARE_FILE = 'e3_inwait_box.py'


def once(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)


def renames(text):
    # The engine's own names stay: FLASH_S5_INWAIT is upper case and 'in-wait' has a hyphen. The parent modules the box imports keep their names through protect().
    for old, new in NAMES:
        text = text.replace(old, new)
    return text


def asserts(env_form):
    return ''.join("assert env.get('%s')%s'%d', 'the yield bound installed'\n" % (name, env_form, value) for name, value in table.LAUNCH_ENV.items())


def body():
    text = once(parent.body(), OLD_CORE, CORE)
    text = once(text, GRADE_TEXT[0], GRADE_TEXT[1])
    text = once(text, 'for _label in cap.g2-live-0; do', 'for _label in ; do')
    text = text.replace('cap arm', 'tail arm')
    text = once(text, GEN_OLD, GEN_NEW)
    text = once(text, GEN_TEXT[0], GEN_TEXT[1])
    text = once(text, ARM_TEXT[0], ARM_TEXT[1])
    return text


def draft(text):
    text = once(text, parent.body(), body())
    text = once(text, DRAFT_ASSERT, DRAFT_ASSERT + asserts(' == '))
    text = once(text, DRAFT_FILES, DRAFT_FILES.replace('; do', MODULES + '; do').replace('e3_inwait_box.py inwait_arms.py e3_attn', 'e3_a1_trunk_box.py a1_trunk_arms.py e3_attn', 1))
    text = once(text, 'inwait_gate.py', 'a1_trunk_gate.py inwait_gate.py')
    text = once(text, '# inwait: native handoffs inside the stock wait', HEADER_NOTE + '# inwait: native handoffs inside the stock wait')
    # The switch names and the parent's modules survive the rename by a marker.
    return restore(renames(protect(text)))


def settings(text):
    text = once(text, 'FLASH_ATTN_META_GDN_BATCH=1 FLASH_S5_INWAIT=1 ', 'FLASH_ATTN_META_GDN_BATCH=1 FLASH_S5_INWAIT=1 ' + BOUNDS + ' ')
    text = once(text, '# inwait: FLASH_S5_INWAIT=1 below', SETTINGS_NOTE + '# inwait: FLASH_S5_INWAIT=1 below')
    return restore(renames(protect(text)))


def declare(text):
    text = once(text, DECLARE_ASSERT, DECLARE_ASSERT + asserts('=='))
    text = once(text, DECLARE_FILE, 'e3_a1_trunk_box.py a1_trunk_arms.py a1_trunk_gate.py inwait_gate.py' + MODULES, 1)
    return restore(renames(protect(text)))


KEEP = ('e3_inwait_box.py inwait_arms.py e3_native_refusal_client.py', 'a1_trunk_gate.py inwait_gate.py')
MARK = '@KEEP%d@'


def protect(text):
    for k, name in enumerate(KEEP):
        text = text.replace(name, MARK % k)
    return text


def restore(text):
    for k, name in enumerate(KEEP):
        text = text.replace(MARK % k, name)
    return text


def final_body():
    """The body as it stands in the draft, its names renamed."""
    return restore(renames(protect(body())))


def derive():
    out = {}
    for suffix, edit in (('.draft.sh', draft), ('_settings.sh', settings), ('_declare.sh', declare)):
        out['harness/runs/' + NEW + suffix] = edit((ROOT / 'harness/runs' / (OLD + suffix)).read_text())
    return out


def table_text():
    import a1_trunk_gate as gate
    lines = ['q18a-a1-trunk arm table (harness/a1_trunk_arms.py); entry limits relative to the run start, never a clock',
             '%-3s %-16s %-6s %-6s %-6s %-5s %-5s %s' % ('#', 'label', 'part', 'yield', 'cohort', 'round', 'reqs', 'enters while SECONDS <=')]
    for k, row in enumerate(gate.rows(), 1):
        lines.append('%-3d %-16s %-6s %-6s %-6s %-5s %-5s %s %d s' % (k, row['label'], row['part'], {True: 'on', False: 'off'}.get(row.get('yield_on'), '-'), row['cohort'], row.get('index', '-'), row.get('generations', 8), row['entry_limit'], gate.ENTRY_LIMIT_S[row['entry_limit']]))
    exp = gate.EXPECTED_RUNTIME_S
    e, slow = exp['nominal'], exp['slower']
    lines.append('arms %d (decode %d, core %d, tail %d); box budget %d s, cleanup included; worst case of the caps %d s' % (exp['arms'], exp['decode_arms'], exp['core_arms'], exp['tail_arms'], gate.BOX_BUDGET_S, gate.WORST_CASE_SUM_S))
    lines.append('expected: %d arms entering, last arm ends +%d s, about %.1f min with startup and the end; smallest entry margin %d s' % (e['arms_entering'], e['last_arm_ends_s'], e['total_min'], e['min_margin_s']))
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
    print('a1-trunk queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
