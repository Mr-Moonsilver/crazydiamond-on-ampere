#!/usr/bin/env python3
"""Derive the q18a-compose-soak-fix box scripts from q18a-compose-4's committed texts. Nothing a past queue ran is edited. Discovery mode.

Declared deltas over compose-4's three texts, each by exact count:
  names     q18a-compose-4 -> q18a-compose-soak-fix and every compose-4 name of the queue, records, box, gate and table -> compose5. The box imports compose-4's box, table and replay, so those three keep their names;
  body      this table's core list; the arm and grade texts that name the arms;
  modules   compose-4's box and table added to the preflight's and the declaration's lists: compose-5's box binds its table over them;
  chunk     FLASH_S5_CHUNK_TOKENS=2048 in EXTRA_ENV for the whole run, as dp8d's launch; both preflights assert it, where compose-4's asserted it unset.
The other switches, the yield's bounds and the ingress check stay compose-4's. The arm list comes from harness/compose5_arms.py; --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import compose5_arms as table  # noqa: E402

OLD, NEW = 'q18a_compose_4', 'q18a_compose_soak_fix'
OLD_CORE = 'core:dec1.g2-live-0 core:cmp.g2-live-0 core:ctl.g2-live-0'
CORE = ' '.join('%s:%s' % (r['part'], r['label']) for r in table.arms(table.GRADED) if r['label'] != table.SMOKE_ARM)
ARM_TEXT = ('dec1 one standing stream and one decode window, cmp and ctl eight replay agents for 18 minutes', 'dec1 one standing stream and one decode window, nr and ny eight replay agents for %d s, ABBA' % table.REPLAY['DURATION'])
HEADER_ANCHOR = '# compose-4: the four lever tracks on one engine under the closed-loop agent replay'
HEADER_NOTE = ('# coverage-soak-fix: the soak (nr against ny on dnr\'s closed-loop turn mix, ABBA nr ny ny nr, 480 s of traffic an arm), discovery mode, '
               "with the geometry-tail handback (2 <= num_tokens % 4048 <= 64: native to the last block boundary, stock the r rows, graded as its own class) "
               "and per Worker_PP RSS failing nr (>= 256 MiB, or >= 32 B per native-append history token); the grade fails nr on any other stock prefill row, "
               "decoder live or idle, and on a request still open after the drain (harness/compose5_arms.py, tmp/a1-trunk/READY-coverage-soak-fix.md). "
               "The reader is coverage-soak-fix/scratch/soak_fix_read.py on the hub.\n")
NAMES = ((OLD, NEW), ('q18a-compose-4', 'q18a-compose-soak-fix'), ('compose-4', 'compose-5'), ('compose4', 'compose5'))
# compose-5's box imports compose-4's box, table and replay by their names; the rename above must not reach them.
SWITCHES = table.CONNECTIONS_ENV + ' ' + table.PLE_ENV + ' ' + table.CHUNK_ENV
OLD_LAST_SWITCH = table.PLE_ENV + ' '
DRAFT_FILES = 'e3_compose5_box.py compose5_arms.py e3_attn_meta_box.py attn_meta_arms.py e3_inwait_box.py inwait_arms.py e3_native_refusal_client.py; do'
DECLARE_FILES = 'e3_compose5_box.py compose5_arms.py compose5_gate.py inwait_gate.py e3_inwait_box.py inwait_arms.py e3_native_refusal_client.py'
MODULES_AFTER = ' e3_a1_trunk_box.py a1_trunk_arms.py compose5_replay.py agent_replay.py'
MODULES = ' e3_a1_trunk_box.py a1_trunk_arms.py compose4_replay.py agent_replay.py e3_compose4_box.py compose4_arms.py'


def once(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)


def renames(text):
    for old, new in NAMES:
        text = text.replace(old, new)
    return text


def source(suffix):
    return (ROOT / 'harness/runs' / (OLD + suffix)).read_text()


def chunk_assert(text, form):
    return once(text, "assert 'FLASH_S5_CHUNK_TOKENS' not in env, 'the chunk stays the default 1024'", "assert env.get('FLASH_S5_CHUNK_TOKENS')%s'%d', 'native chunk rows installed'" % (form, table.CHUNK_TOKENS))


def draft(text):
    text = chunk_assert(text, ' == ')
    # dnr: a replay arm doesn't read the workload, and index 2-9 would have named band 2's; one file for every index, the one sm has.
    text = once(text, 'if [[ $_pair == [01] ]]; then _workload=', 'if [[ $_pair == [0-9] ]]; then _workload=')
    text = once(text, OLD_CORE, CORE)
    text = once(text, ARM_TEXT[0], ARM_TEXT[1])
    text = once(text, HEADER_ANCHOR, HEADER_NOTE + HEADER_ANCHOR)
    text = renames(text)
    return once(text, MODULES_AFTER, MODULES)


def settings(text):
    text = once(text, OLD_LAST_SWITCH, OLD_LAST_SWITCH + table.CHUNK_ENV + ' ')
    return renames(text)


def declare(text):
    text = chunk_assert(text, '==')
    text = renames(text)
    return once(text, MODULES_AFTER, MODULES)


def derive():
    out = {}
    for suffix, edit in (('.draft.sh', draft), ('_settings.sh', settings), ('_declare.sh', declare)):
        out['harness/runs/' + NEW + suffix] = edit(source(suffix))
    return out


def final_body():
    """The arm loop through the read, as it stands in the draft."""
    text = derive()['harness/runs/' + NEW + '.draft.sh']
    start = text.index('alpha_next_arm() {')
    stop = text.index('\n', text.index('read_step compose5-read', start))
    return text[start:stop + 1]


def table_text():
    import compose5_gate as gate
    lines = ['q18a-compose-soak-fix arm table (harness/compose5_arms.py); entry limits relative to the run start, never a clock',
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
    print('compose-5 queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
