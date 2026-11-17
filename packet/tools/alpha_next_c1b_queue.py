#!/usr/bin/env python3
"""Derive the q18a-alpha-next-c1b box scripts from the committed q18a-alpha-next-c1 texts. Nothing a past queue ran is edited.

The parent texts are verified first (tools/alpha_next_c1_queue.py derive() must equal the committed c1 texts), then each delta below
is applied exactly once (counts asserted). Kept from alpha-next-c1, byte for byte: everything it kept from alpha-next and Step B
(pins, preflight, startup, bind, host headroom, the unarmed check, the end identity, both band manifests and their /tokenize guards,
the le-fix death scan and watch, the M3 allocator receipts, the owned cleanup and restore tail), its three switches
(FLASH_S5_STAGE_OVERLAP=1, FLASH_S5_STAGE_OVERLAP_DEFER_BIND=1, FLASH_S5_STAGE_OVERLAP_OBSERVE=1) and their asserts, the STOP_RACE
hook, the arm function, the core grade (the queue's verdict) and the read. Declared deltas:
  names      runs/q18a_alpha_next_c1_{settings,declare}.sh and .draft.sh -> q18a_alpha_next_c1b_*, QNAME and RUN_ID
             q18a-alpha-next-c1 -> q18a-alpha-next-c1b, the grade and read records alpha-next-c1-{grade,read,report} ->
             alpha-next-c1b-*, the box e3_alpha_next_c1_box -> e3_alpha_next_c1b_box, the gate alpha_next_c1_gate ->
             alpha_next_c1b_gate (the budget reader and the cap comments); the history comments keep their names;
  budget     alpha_next_c1b_gate: alpha-next's formulas over 7,800 s, relative to the run's own start only, no window;
  pins       the declaration pins alpha-next-c1's files and adds e3_alpha_next_c1b_box.py alpha_next_c1b_gate.py
             alpha_next_c1b_arms.py e3_c1_open_rows.py (the draft's interface list too); no bundle, manifest or package digest here;
  body       the six core arms S1 S2 L1 L2 D S3 of alpha_next_c1b_arms (labels c0off.g2-solo-0 ...), no tail (an empty tail list;
             the tail health record stays), band 1's rounds 0 1 3 on band 1's workload, and the arm call ``--c1 serial|off|defer``
             mapped from the label's prefix (c0off -> serial, c1off -> off, c1defer -> defer; anything else is passed as
             ``unknown`` and the box refuses it).
No model setting, switch, gate threshold, launcher or restore change. --check verifies the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import alpha_next_c1_queue as c1q  # noqa: E402
import alpha_next_c1b_arms as table  # noqa: E402

OLD, NEW = c1q.NEW, 'q18a_alpha_next_c1b'
OLD_Q, NEW_Q = c1q.NEW_Q, 'q18a-alpha-next-c1b'
CORE = tuple('%s:%s' % (r['part'], r['label']) for r in table.arms(table.GRADED))
TAIL = tuple(table.labels('tail'))
assert TAIL == () and len(CORE) == 6
HEADER_NOTE = ('# alpha-next-c1b: the fixed-schedule numerics queue (DESIGN-c1-next (c), band 1 only) on alpha-next-c1\'s engine,\n'
               '# byte for byte, with its three switches for the whole run. Six solo core arms in the order S1 S2 L1 L2 D S3, each\n'
               '# selected by the typed native_stage_local {local, overlap, defer_bind} control: S1 S2 S3 serial (C0 off), L1 L2 C0\n'
               '# local (C1 off), D C1 defer. The fresh request of every arm is the numerics probe: one fixed prompt recorded by sha,\n'
               '# solo, temperature 0, 256 tokens, top-20 logprobs per position. The rule is declared before the run (bit-equal if\n'
               '# S1 = S2, else the S1-S2 bound; no retry), and its verdict joins the core grade, the queue verdict. No tail.\n#\n')

BODY_EDITS = (
    ("# alpha-next-c1: the core arms (CORE below) then, once the core grade is written, the g4 tail (TAIL below, reads only). Each\n"
     "# arm selects burst, fused, its group size, C0 local and its C1 selection by typed control while paused idle FULL.",
     "# alpha-next-c1b: the six core arms S1 S2 L1 L2 D S3 (CORE below), then the core grade; this table has no tail (the tail\n"
     "# list below is empty). Each arm selects burst, fused, its group size, C0 serial or local and its C1 selection by typed\n"
     "# control while paused idle FULL; its fresh request is the numerics probe (e3_alpha_next_c1b_box.NUMERICS)."),
    ("# descriptive. Every timeout below is one alpha_next_c1_gate cap (instruments, arm_entries, alpha_next_c1_arms, ingress,",
     "# descriptive. Every timeout below is one alpha_next_c1b_gate cap (instruments, arm_entries, alpha_next_c1b_arms, ingress,"),
    ('  IFS=. read -r _c1 _group <<< "$_sel"\n',
     '  IFS=. read -r _c1 _group <<< "$_sel"\n'
     '  case $_c1 in (c0off) _c1=serial;; (c1off) _c1=off;; (c1defer) _c1=defer;; (*) _c1=unknown;; esac\n'),
    ('  if [[ $_pair == [01] ]]; then _workload=$H/workloads/q18a-native41-progress.json',
     '  if [[ $_pair == %s ]]; then _workload=$H/workloads/q18a-native41-progress.json' % table.band1_glob()),
    ('python3 "$H/e3_alpha_next_c1_box.py"', 'python3 "$H/e3_alpha_next_c1b_box.py"'),
    ('--c1 "${_c1#c1}"', '--c1 "$_c1"'),
    ("'auto arm: 8 standing decoders, burst, fused, the group size, C0 local, the C1 selection and defer_bind, fresh, append; live "
     "keeps them, solo cancels'",
     "'auto arm (solo): 8 standing decoders cancelled at the first native chunk, burst, fused, the group size, C0 serial or "
     "local, the C1 selection and defer_bind, the numerics fresh (temperature 0, 256 tokens, top-20 logprobs), append'"),
    ('$Q/alpha-next-c1-', '$Q/alpha-next-c1b-'),
    ("record alpha-next-c1 \"$?\" 'core arms: per-arm mechanism (crash-free, TP commitment, group, C0 and C1 engagement, one command "
     "in flight per stage, OOM), numerics envelope, complete progress lines; speed, decode and memory descriptive'",
     "record alpha-next-c1b \"$?\" 'core arms S1 S2 L1 L2 D S3: per-arm mechanism (crash-free, TP commitment, group, C0 and C1 "
     "engagement, one command in flight per stage, OOM), the numerics rule (bit-equal or the S1-S2 bound), complete progress "
     "lines (C1 open rows carried); speed, decode and memory descriptive'"),
    ('read_step alpha-next-c1-read "$?"', 'read_step alpha-next-c1b-read "$?"'),
    ("'every arm, the g4 tail included: the same rules as the core grade, a read, never the queue verdict'",
     "'every arm (no tail): the same rules as the core grade, a read, never the queue verdict'"),
    ('# The g4 tail: reads only, after the core grade, while the engine is healthy.',
     '# No tail in this table: the list below is empty; the health record after it stays (alpha-next-c1\'s structure).'),
)


def c1_body():
    return c1q.body()


def body():
    text = c1_body()
    old_core = ' '.join(c1q.CORE)
    old_tail = ' '.join(c1q.TAIL)
    for old, new in BODY_EDITS:
        assert old in text, old[:100]
        text = text.replace(old, new)
    text = c1q.replace_once(text, 'for _arm in %s; do' % old_core, 'for _arm in %s; do' % ' '.join(CORE))
    text = c1q.replace_once(text, 'for _label in %s; do' % old_tail, 'for _label in %s; do' % ' '.join(TAIL))
    return text


DRAFT_NAMES = (
    ('from alpha_next_c1_gate import ACTIVE_S', 'from alpha_next_c1b_gate import ACTIVE_S', 1),
    ('(alpha_next_c1_gate, bash SECONDS)', '(alpha_next_c1b_gate, bash SECONDS)', 1),
    ('every cap here is alpha_next_c1_gate.\n', 'every cap here is alpha_next_c1b_gate.\n', 1),
    ('Every cap is alpha_next_c1_gate.PHASES.', 'Every cap is alpha_next_c1b_gate.PHASES.', 1),
    ('"$H/runs/q18a_alpha_next_c1_settings.sh"', '"$H/runs/q18a_alpha_next_c1b_settings.sh"', 1),
    ('-r $H/runs/q18a_alpha_next_c1_settings.sh &&', '-r $H/runs/q18a_alpha_next_c1b_settings.sh &&', 1),
    ('RUN_ID="q18a-alpha-next-c1-$(date', 'RUN_ID="q18a-alpha-next-c1b-$(date', 1),
    ("perf=p.parent/'alpha-next-c1-grade.json'", "perf=p.parent/'alpha-next-c1b-grade.json'", 1),
    ("'missing':['alpha-next-c1-grade-absent']", "'missing':['alpha-next-c1b-grade-absent']", 1),
    ("tail=p.parent/'alpha-next-c1-read.json'", "tail=p.parent/'alpha-next-c1b-read.json'", 1),
)
DECLARE_NAMES = (
    ('QNAME=${QNAME:-q18a-alpha-next-c1}', 'QNAME=${QNAME:-q18a-alpha-next-c1b}', 1),
    ('"$H/runs/q18a_alpha_next_c1_settings.sh"', '"$H/runs/q18a_alpha_next_c1b_settings.sh"', 2),
    ('"$H/runs/q18a_alpha_next_c1.draft.sh"', '"$H/runs/q18a_alpha_next_c1b.draft.sh"', 2),
    ('"$H/runs/q18a_alpha_next_c1_declare.sh"', '"$H/runs/q18a_alpha_next_c1b_declare.sh"', 1),
)
SETTINGS_NAMES = (
    ('QNAME=${QNAME:-q18a-alpha-next-c1}', 'QNAME=${QNAME:-q18a-alpha-next-c1b}', 1),
)
C1B_FILES = 'e3_alpha_next_c1b_box.py alpha_next_c1b_gate.py alpha_next_c1b_arms.py e3_c1_open_rows.py'
INTERFACE_OLD = c1q.INTERFACE_NEW
INTERFACE_NEW = INTERFACE_OLD[:-len('; do')] + ' \\\n  ' + C1B_FILES + '; do'
PINS_OLD = c1q.PINS_NEW
PINS_NEW = PINS_OLD + C1B_FILES + ' '
SETTINGS_ANCHOR = c1q.SETTINGS_NOTE.splitlines()[0]
SETTINGS_NOTE = ('# alpha-next-c1b: alpha-next-c1\'s settings unchanged (the same engine and the same three switches below); each arm\n'
                 '# selects serial (C0 off), C0 local or C1 defer by typed control (alpha_next_c1b_arms).\n')


def parent_texts():
    """alpha-next-c1's committed texts, after checking they are exactly what its tool derives."""
    derived = c1q.derive()
    for name, text in derived.items():
        assert (ROOT / name).read_text() == text, 'alpha-next-c1 text differs from its derivation: ' + name
    return derived


def derive():
    parent = parent_texts()
    out = {}
    for suffix in ('.draft.sh', '_settings.sh', '_declare.sh'):
        text = parent['harness/runs/' + OLD + suffix]
        if suffix == '.draft.sh':
            text = c1q.replace_once(text, c1_body(), body())
            text = c1q.replace_once(text, INTERFACE_OLD, INTERFACE_NEW)
            for old, new, count in DRAFT_NAMES:
                text = c1q.replace_once(text, old, new, count)
            first = text.index('# alpha-next-c1: this queue compares')
            text = text[:first] + HEADER_NOTE + text[first:]
        elif suffix == '_settings.sh':
            text = c1q.replace_once(text, SETTINGS_ANCHOR, SETTINGS_NOTE + SETTINGS_ANCHOR)
            for old, new, count in SETTINGS_NAMES:
                text = c1q.replace_once(text, old, new, count)
        else:
            text = c1q.replace_once(text, PINS_OLD, PINS_NEW)
            for old, new, count in DECLARE_NAMES:
                text = c1q.replace_once(text, old, new, count)
        out['harness/runs/' + NEW + suffix] = text
    return out


def table_text():
    """The arm table as the draft runs it, with the relative entry limit and the expected runtime (descriptive)."""
    import alpha_next_c1b_gate as gate
    lines = ['q18a-alpha-next-c1b arm table (harness/alpha_next_c1b_arms.py); entry limits relative to the run start, never a clock',
             '%-3s %-4s %-17s %-6s %-5s %-5s %-5s %-4s %-6s %-4s %-5s %s' % (
                 '#', 'role', 'label', 'c1', 'c0', 'ovlp', 'defer', 'size', 'cohort', 'band', 'round', 'enters while SECONDS <=')]
    for k, row in enumerate(table.arms(), 1):
        lines.append('%-3d %-4s %-17s %-6s %-5s %-5s %-5s %-4d %-6s %-4d %-5d %s %d s' % (
            k, row['role'], row['label'], row['c1'], row['c0'], str(row['overlap']).lower(), str(row['defer_bind']).lower(),
            row['size'], row['cohort'], row['band'], row['index'], row['entry_limit'], gate.ENTRY_LIMIT_S[row['entry_limit']]))
    exp = gate.EXPECTED_RUNTIME_S
    e = exp['nominal']
    lines.append('arms %d (core %d, tail %d); every arm solo, band 1; box budget %d s, cleanup included' % (
        exp['arms'], exp['core_arms'], exp['tail_arms'], gate.BOX_BUDGET_S))
    lines.append('expected (lead walls + %d s probe decode per arm, assumed): arms entering %d; last arm enters +%d s, ends +%d s '
                 '(arms %.1f min); about %.1f min with startup and the end; smallest entry margin %d s' % (
                     gate.NUMERICS_DECODE_S, e['arms_entering'], e['last_arm_enters_s'], e['last_arm_ends_s'], e['arms_s'] / 60,
                     e['total_min'], e['min_margin_s']))
    slow = exp['slower']
    lines.append('at %.1fx the walls: arms entering %d (about %.1f min); not run: %s' % (
        gate.SLOWER, slow['arms_entering'], slow['total_min'], ', '.join(slow['not_run']) or 'none'))
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--check', action='store_true')
    p.add_argument('--table', action='store_true', help='print the arm table with its entry limits; write nothing')
    p.add_argument('--replace-derived', action='store_true',
                   help='write a derived text that differs from the one on disk (only for a text no queue has run: the old one '
                        'stays in git)')
    a = p.parse_args(argv)
    if a.table:
        print(table_text())
        return
    for name, text in derive().items():
        path = ROOT / name
        if a.check:
            assert path.read_text() == text, name
        else:
            assert a.replace_derived or not path.exists() or path.read_text() == text, (
                'refusing to change an existing derived text: ' + name)
            path.write_text(text)
            path.chmod(0o664)
    print('alpha-next-c1b queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
