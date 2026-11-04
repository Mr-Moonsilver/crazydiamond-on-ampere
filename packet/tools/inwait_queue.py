#!/usr/bin/env python3
"""Derive the q18a-inwait box scripts from the committed q18a-compose-2 texts. Nothing a past queue ran is edited. Discovery mode.

Declared deltas over compose-2's three texts, each by exact count:
  names     every compose-2 name -> inwait (the queue, records, box, gate, table);
  switch    FLASH_S5_INWAIT=1 in EXTRA_ENV after FLASH_ATTN_META_GDN_BATCH=1 (installed, not selected), asserted in the declaration and the preflight beside them;
  body      this table's core list; one decode-only baseline (decode-am1), not two; the exact-path smoke before them, whose failure stops every later arm; the cap arm as the tail, after the core grade.
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
CORE = ' '.join('%s:%s' % (r['part'], r['label']) for r in table.arms(table.GRADED) if r['label'] != table.SMOKE_ARM)
DECODE = ' '.join(d['label'] for d in table.DECODE_ARMS)
TAIL = ' '.join(table.labels('tail'))
NAMES = ((OLD, NEW), (OLD_Q, NEW_Q), ('COMPOSE2', 'INWAIT'), ('compose2', 'inwait'), ('compose-2', 'inwait'))
FILES = 'e3_inwait_box.py inwait_arms.py e3_attn_meta_box.py attn_meta_arms.py'
HEADER_NOTE = ('# inwait: native handoffs inside the stock wait, discovery mode: FLASH_S5_INWAIT installed for the whole run beside attn-meta and L1 + L2, all selected per arm by typed control while paused idle FULL: ps1 off,\n'
               '# hf base, against both, cap, w0 and wa; the exact-path smoke first, then one decode-only baseline; band 1 live x2 each and solo x1 for hf and both, top-20 logprobs on band 1's fresh and append, w0 at band 2 last, after the core grade (harness/inwait_arms.py). NUMERICS ARE NOT COVERED by this queue either: speed, decode, handler time, memory and the logprob spread are reads only.\n#\n')
SETTINGS_NOTE = '# inwait: FLASH_S5_INWAIT=1 below (installed, not selected), for the whole run, the only setting added to compose-2; each arm selects it by typed control.\n'
SETTINGS_ANCHOR = '# compose-2: FLASH_ATTN_META_GDN_BATCH=1'
DRAFT_ASSERT = "assert env.get('FLASH_ATTN_META_GDN_BATCH') == '1', 'attn-meta installed'\n"
DECLARE_ASSERT = "assert env.get('FLASH_ATTN_META_GDN_BATCH')=='1', 'compose-2 runs with attn-meta installed'\n"
AM0_CASE = '    (decode-am0) _attn=off;;\n'
SMOKE_ANCHOR = '  for _label in ' + DECODE + '; do\n'
SMOKE_CALL = '  inwait_smoke ' + table.SMOKE_ARM + '\n'
SMOKE_DEFS_ANCHOR = 'GRADE_ARGS=(); READ_ARGS=()\n'
SMOKE_DEFS = '''# inwait build-4: arm 1 is the exact-path smoke (review section 6). Real request shape, real control, positive engagement, one complete
# observation, clean teardown: if not, SMOKE_ABORT stops every later arm, each recorded NOT_RUN.
SMOKE_ABORT=
inwait_smoke() {
  local _label=$1 _rc
  alpha_next_arm core "$_label"
  timeout 60s python3 "$H/e3_inwait_box.py" smoke --arm "$Q/arm-$_label.json" --out "$Q/smoke.json" > "$Q/smoke.log" 2>&1
  _rc=$?
  n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown
  if [[ $_rc == 0 ]] && { [[ $n != "$N0" ]] || host_crash_detected || ! health_ok; }; then _rc=2; fi
  record smoke "$_rc" 'exact-path smoke: complete arm, real control acknowledged, native engaged on every rank, engine healthy after it; else the table aborts'
  [[ $_rc == 0 ]] || SMOKE_ABORT=1
  return 0
}
'''
ARM_GUARD_ANCHOR = '  if (( SECONDS > _limit )); then\n'
ARM_GUARD = '''  if [[ -n $SMOKE_ABORT ]]; then
    "$_skip" "arm-$_label" 2 'NOT_RUN: the exact-path smoke did not engage; table aborted'
    return 0
  fi
'''
DECODE_GUARD_ANCHOR = '  if (( SECONDS > LAST_ARM_ENTRY_S )); then\n    record "arm-$_label" 2 "NOT_RUN: box budget'
DECODE_GUARD = '''  if [[ -n $SMOKE_ABORT ]]; then
    record "arm-$_label" 2 'NOT_RUN: the exact-path smoke did not engage; table aborted'
    return 0
  fi
'''


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
    text = once(text, '  for _label in ; do\n    alpha_next_arm tail', '  for _label in ' + TAIL + '; do\n    alpha_next_arm tail')
    text = text.replace('g4 tail', 'cap arm')
    text = once(text, SMOKE_ANCHOR, SMOKE_CALL + SMOKE_ANCHOR)
    text = once(text, SMOKE_DEFS_ANCHOR, SMOKE_DEFS + SMOKE_DEFS_ANCHOR)
    text = once(text, ARM_GUARD_ANCHOR, ARM_GUARD + ARM_GUARD_ANCHOR)
    text = once(text, DECODE_GUARD_ANCHOR, DECODE_GUARD + DECODE_GUARD_ANCHOR)
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
