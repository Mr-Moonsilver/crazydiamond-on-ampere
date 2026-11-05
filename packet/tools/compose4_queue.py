#!/usr/bin/env python3
"""Derive the q18a-compose-4 box scripts from q18a-a1-trunk's committed texts. Nothing a past queue ran is edited. Discovery mode.

From a1's committed texts, not from tools/a1_trunk_queue.py: that tool derives from inwait's queue, and the merged inwait queue is 5d's, whose lists its anchors no longer find. The two texts differ only in lists, notes and the yield's lines (READY-compose-4.md).

Declared deltas over a1's three texts, each by exact count:
  names     q18a-a1-trunk -> q18a-compose-4 and every a1 name of the queue, records, box, gate and table -> compose4; FLASH_S5_APPEND_YIELD_* and the engine's names stay;
  switch    connections 32 and the two PLE launch switches in EXTRA_ENV after the yield's bounds (installed for the whole run, both arms; the PLE is selected per arm), asserted in the declaration and the preflight. FLASH_S5_CHUNK_TOKENS stays unset, so 1024, the default;
  body      this table's core list; the ingress check's generation count read from the table, and for a replay from its own record;
  modules   the modules the box imports beyond a1's (a1's box and table, the replay client and agent_replay) in the preflight's and the declaration's lists.
The arm list comes from harness/compose4_arms.py; --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import compose4_arms as table  # noqa: E402

OLD, NEW = 'q18a_a1_trunk', 'q18a_compose_4'
OLD_Q, NEW_Q = 'q18a-a1-trunk', 'q18a-compose-4'
OLD_CORE = 'core:on.g2-live-0 core:off.g2-live-0 core:off.g2-live-1 core:on.g2-live-1 core:mn.g2-live-0 core:mo.g2-live-0 core:on.g2-solo-0'
CORE = ' '.join('%s:%s' % (r['part'], r['label']) for r in table.arms(table.GRADED) if r['label'] != table.SMOKE_ARM)
NAMES = ((OLD, NEW), (OLD_Q, NEW_Q), ('a1-trunk-', 'compose4-'), ('a1_trunk', 'compose4'))
SWITCHES = table.CONNECTIONS_ENV + ' ' + table.PLE_ENV
LAST_BOUND = 'FLASH_S5_APPEND_YIELD_REENTRY_RETRIES=4 '
GEN_OLD = 'generations "$_label")"'
GEN_NEW = 'generations "$_label" "$Q/arm-$_label.json")"'
GEN_TEXT = ('(8 standing, fresh, append, the returners)', '(the standing streams, fresh, append, the returners; a replay arm the turns of its agents)')
ARM_TEXT = ("'auto arm: 8 standing decoders, burst, fused, the group size, C0 local, C1 defer, the lever selection, the yield selection, fresh with the returners inside it, append; live keeps the rest, solo cancels'",
            "'the composed levers selected, the auto-arm released for the arm; dec1 one standing stream and one decode window, cmp and ctl eight replay agents for 18 minutes'")
GRADE_TEXT = ("'the arms: crash-free, attn-meta and startup, the yield selected only where on and served where on with returners, route generations add up; every speed and the budget lines are reads, numerics not covered'",
              "'the arms: the record is whole, crash-free, attn-meta and startup, the yield selected only where on; the run rule is compose4_rule.py on the fetched record, numerics not covered'")
HEADER_ANCHOR = '# a1-trunk: the returning-agent yield over build-3b\'s ps1 levers'
HEADER_NOTE = ('# compose-4: the four lever tracks on one engine under the closed-loop agent replay, discovery mode: inwait 5d full, the private PLE buffer, the returning-agent yield, connections 32 for the whole run. '
               'The exact-path smoke first (its failure stops the table), one decode-only baseline, the one-stream arm, the composed replay arm, its control (harness/compose4_arms.py). NUMERICS ARE NOT COVERED.\n#\n')
SETTINGS_ANCHOR = '# a1-trunk: the four FLASH_S5_APPEND_YIELD_* launch bounds below'
SETTINGS_NOTE = '# compose-4: connections 32 and the two PLE launch switches below, the only settings added to a1-trunk\'s, for the whole run; each arm selects its levers by typed control.\n'
DRAFT_ASSERT = "assert env.get('FLASH_S5_APPEND_YIELD_REENTRY_RETRIES') == '4', 'the yield bound installed'\n"
DECLARE_ASSERT = "assert env.get('FLASH_S5_APPEND_YIELD_REENTRY_RETRIES')=='4', 'the yield bound installed'\n"
MODULES = ' e3_a1_trunk_box.py a1_trunk_arms.py compose4_replay.py agent_replay.py'
DRAFT_FILES = 'e3_compose4_box.py compose4_arms.py e3_attn_meta_box.py attn_meta_arms.py e3_inwait_box.py inwait_arms.py e3_native_refusal_client.py; do'
DECLARE_FILES = 'e3_compose4_box.py compose4_arms.py compose4_gate.py inwait_gate.py e3_inwait_box.py inwait_arms.py e3_native_refusal_client.py'


def once(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)


def renames(text):
    for old, new in NAMES:
        text = text.replace(old, new)
    return text


def asserts(env_form):
    pairs = (tok.split('=', 1) for tok in SWITCHES.split())
    return ''.join("assert env.get('%s')%s'%s', 'installed for the whole run'\n" % (name, env_form, value) for name, value in pairs) + "assert 'FLASH_S5_CHUNK_TOKENS' not in env, 'the chunk stays the default 1024'\n"


def source(suffix):
    return (ROOT / 'harness/runs' / (OLD + suffix)).read_text()


def draft(text):
    text = once(text, DRAFT_ASSERT, DRAFT_ASSERT + asserts(' == '))
    text = once(text, HEADER_ANCHOR, HEADER_NOTE + HEADER_ANCHOR)
    text = once(text, OLD_CORE, CORE)
    text = once(text, ARM_TEXT[0], ARM_TEXT[1])
    text = once(text, GRADE_TEXT[0], GRADE_TEXT[1])
    text = once(text, GEN_TEXT[0], GEN_TEXT[1])
    text = renames(text)
    text = once(text, GEN_OLD, GEN_NEW)
    return once(text, DRAFT_FILES, DRAFT_FILES.replace('; do', MODULES + '; do'))


def settings(text):
    text = once(text, LAST_BOUND, LAST_BOUND + SWITCHES + ' ')
    text = once(text, SETTINGS_ANCHOR, SETTINGS_NOTE + SETTINGS_ANCHOR)
    return renames(text)


def declare(text):
    text = once(text, DECLARE_ASSERT, DECLARE_ASSERT + asserts('=='))
    text = renames(text)
    return once(text, DECLARE_FILES, DECLARE_FILES + MODULES)


def derive():
    out = {}
    for suffix, edit in (('.draft.sh', draft), ('_settings.sh', settings), ('_declare.sh', declare)):
        out['harness/runs/' + NEW + suffix] = edit(source(suffix))
    return out


def final_body():
    """The arm loop through the read, as it stands in the draft."""
    text = derive()['harness/runs/' + NEW + '.draft.sh']
    start = text.index('alpha_next_arm() {')
    stop = text.index('\n', text.index('read_step compose4-read', start))
    return text[start:stop + 1]


def table_text():
    import compose4_gate as gate
    lines = ['q18a-compose-4 arm table (harness/compose4_arms.py); entry limits relative to the run start, never a clock',
             '%-3s %-16s %-10s %-6s %-5s %s' % ('#', 'label', 'kind', 'yield', 'ple', 'enters while SECONDS <=')]
    for k, row in enumerate(gate.rows(), 1):
        lines.append('%-3d %-16s %-10s %-6s %-5s %s %d s' % (k, row['label'], row.get('kind', 'decode'), {True: 'on', False: 'off'}.get(row.get('yield_on'), '-'), {True: 'on', False: 'off'}.get(row.get('ple_private'), '-'), row['entry_limit'], gate.ENTRY_LIMIT_S[row['entry_limit']]))
    exp = gate.EXPECTED_RUNTIME_S
    e, slow = exp['nominal'], exp['slower']
    lines.append('box budget %d s, cleanup included; worst case of the caps %d s' % (gate.BOX_BUDGET_S, gate.WORST_CASE_SUM_S))
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
    print('compose-4 queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
