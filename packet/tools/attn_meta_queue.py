#!/usr/bin/env python3
"""Derive the q18a-attn-meta box scripts from the committed q18a-alpha-next-c1 texts. Nothing a past queue ran is edited.

Kept from alpha-next-c1, byte for byte: everything c1 kept from alpha-next (pins, preflight, startup, bind, host headroom, the
unarmed check, the end identity, both band manifests and their /tokenize guards, the le-fix death scan and watch, the M3
allocator receipts, the owned cleanup and restore tail, STOP_RACE), c1's switches and asserts, the core-then-tail structure and
the read. Declared deltas, each applied exactly once (counts asserted) or, inside the body, on every occurrence:
  names      runs/q18a_alpha_next_c1_{settings,declare}.sh and .draft.sh -> q18a_attn_meta_*, QNAME and RUN_ID
             q18a-alpha-next-c1 -> q18a-attn-meta, the records alpha-next-c1-{grade,read} -> attn-meta-*, the box
             e3_alpha_next_c1_box -> e3_attn_meta_box, the gate alpha_next_c1_gate -> attn_meta_gate;
  switch     FLASH_ATTN_META_GDN_BATCH=1 in EXTRA_ENV after c1's switches (attn_meta_select installs the per-arm Selector at
             startup, not selected), asserted in the declaration and the preflight beside c1's;
  pins       the declaration pins c1's files and adds e3_attn_meta_box.py attn_meta_gate.py attn_meta_arms.py (the draft's
             interface list too);
  body       c1's body with: the band 1 workload glob from attn_meta_arms (round indexes 0, 1, 3, 4), the arm's attn-meta
             selection from its round index (``--attn on|off``), the decode-only baselines first (attn_meta_decode_arm: entry
             as an arm, e3_attn_meta_box decode-arm, ingress with the 8 standing requests), this table's core and tail lists,
             the grade and read of e3_attn_meta_box.
No model setting, gate threshold, launcher or restore change. The arm lists come from harness/attn_meta_arms.py; --check verifies
the files.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'harness')]
import alpha_next_c1_queue as c1q  # noqa: E402
import alpha_next_queue as alpha_next  # noqa: E402
import attn_meta_arms as table  # noqa: E402

OLD, NEW = c1q.NEW, 'q18a_attn_meta'
OLD_Q, NEW_Q = c1q.NEW_Q, 'q18a-attn-meta'
SWITCH_NAME = 'FLASH_ATTN_META_GDN_BATCH'
SWITCH = SWITCH_NAME + '=1'
CORE = tuple('%s:%s' % (r['part'], r['label']) for r in table.arms(table.GRADED))
TAIL = tuple(table.labels('tail'))
DECODE = tuple(d['label'] for d in table.DECODE_ARMS)
ATTN_ON_GLOB = '[%s]' % ''.join(str(i) for i, on in sorted(table.ATTN.items()) if on)
HEADER_NOTE = ('# attn-meta: this queue is the 2x2 at group size 1 (attn-meta GDN metadata batching x C1 with the deferred PP bind)\n'
               '# on the alpha-next-c1 engine plus FLASH_ATTN_META_GDN_BATCH=1 (the per-arm Selector installed, not selected), every\n'
               '# arm selecting its C1 selection and defer_bind (native_stage_local), its group size (native_layer_group) and its\n'
               '# attn-meta selection (qualification worker action attn_meta_select {on}) by typed control while paused idle FULL:\n'
               '# %d decode-only baselines (attn-meta off, on), a core of %d arms (R M P MP at G1, band 1 live over two rounds, B = A\n'
               '# reversed, solo once, MP band 2 live), whose grade is the queue verdict, then a G2 tail of %d arms (P2 MP2), reads\n'
               '# only. Mechanism gates are c1\'s plus attn-meta\'s (selection leak FAIL, not engaged INC, the order lines); speed,\n'
               '# decode (three retentions), memory and the overlap observation are reported; release readiness is not assessed.\n#\n'
               % (len(DECODE), len(table.arms('core')), len(TAIL)))

DECODE_FUNCTION = '''# attn-meta: the standalone decode baselines (attn_meta_arms.DECODE_ARMS), before the core arms. Entry as an arm (budget, crash,
# the typed all-rank FULL), then e3_attn_meta_box decode-arm (8 standing decoders, burst, fused, G1, C0 local, C1 off, the
# attn-meta selection; one decode window, no bulk, the end read), then the ingress check over its 8 standing requests.
attn_meta_decode_arm() {
  local _label=$1 _attn _entry_rc _span _s _e
  case $_label in
@DECODE_CASE@
    (*) record "arm-$_label" 1 'decode arm label not declared'; return 0;;
  esac
  if (( SECONDS > LAST_ARM_ENTRY_S )); then
    record "arm-$_label" 2 "NOT_RUN: box budget, elapsed ${SECONDS}s > ${LAST_ARM_ENTRY_S}s"
    return 0
  fi
  n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown
  if [[ $n != "$N0" ]] || host_crash_detected || ! health_ok; then
    record "arm-$_label" 2 'NOT_RUN: engine crashed, unhealthy or auto-restarted; no restart attempted'
    return 0
  fi
  engine_watched "entry-$_label" timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \\
    --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/entry-$_label.json" > "$Q/entry-$_label.log" 2>&1
  _entry_rc=$?; record "entry-$_label" "$_entry_rc" 'typed all-rank FULL, resumed, idle, workers empty; lease taken'
  if [[ $_entry_rc != 0 ]]; then
    record "arm-$_label" 2 'NOT_RUN: arm entry not a typed idle FULL'
    return 0
  fi
  READINESS=$Q18A_READINESS GEN=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \\
  S5_OVERLAY=$S5_OVERLAY S5_LAUNCHER_NAME=$S5_LAUNCHER_NAME \\
  PACKAGE_MANIFEST=$S5_OVERLAY/SHA256SUMS GENERATION=$S5_OVERLAY/S5-OVERLAP-GENERATION.json \\
  engine_watched "arm-$_label" timeout --signal=TERM --kill-after=30s 1500s python3 "$H/e3_attn_meta_box.py" decode-arm \\
    --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH" --label "$_label" --attn "$_attn" \\
    --workload "$H/workloads/q18a-native41-progress.json" --control-timeout-s 60 \\
    --out "$Q/arm-$_label.json" > "$Q/arm-$_label.log" 2>&1
  record "arm-$_label" "$?" 'decode-only baseline: 8 standing decoders, burst, fused, G1, C0 local, C1 off, the attn-meta selection; no bulk'
  GRADE_ARGS+=(--arm "$Q/arm-$_label.json" --watch "$_label=$Q/engine-watch-arm-$_label.json")
  READ_ARGS+=(--arm "$Q/arm-$_label.json" --watch "$_label=$Q/engine-watch-arm-$_label.json")
  _span=$(python3 - "$Q/arm-$_label.json" <<'PY_SPAN'
import json,sys
try:d=json.load(open(sys.argv[1]))
except (OSError,ValueError):raise SystemExit(0)
s,e=d.get('wall_start_s'),d.get('wall_end_s')
if type(s) in (int,float) and type(e) in (int,float):print(repr(float(s)),repr(float(e)))
PY_SPAN
)
  if [[ -n $_span ]]; then
    read -r _s _e <<< "$_span"
    timeout 60s python3 "$H/e3_native39_trunk_checks.py" ingress --q "$Q" --label "$_label" --start "$_s" --end "$_e" \\
      --generation 8 --out "$Q/ingress-$_label.json" > "$Q/ingress-$_label.log" 2>&1
    record "ingress-$_label" "$?" 'access log: every request in the arm span is the arm (8 standing), else INVALID'
  else
    record "ingress-$_label" 2 'arm span missing; ingress not checked'
  fi
  GRADE_ARGS+=(--ingress "$_label=$Q/ingress-$_label.json")
  READ_ARGS+=(--ingress "$_label=$Q/ingress-$_label.json")
  return 0
}
'''

# The body: c1's (alpha-next's with c1's edits), with these edits; every occurrence replaced, each old text must be there.
BODY_EDITS = (
    ("# alpha-next-c1: the core arms (CORE below) then, once the core grade is written, the g4 tail (TAIL below, reads only). Each\n"
     "# arm selects burst, fused, its group size, C0 local and its C1 selection by typed control while paused idle FULL.",
     "# attn-meta: the decode-only baselines (DECODE below), the core arms (CORE below) then, once the core grade is written, the G2\n"
     "# tail (TAIL below, reads only). Each arm selects burst, fused, its group size, C0 local, its C1 selection and its attn-meta\n"
     "# selection (from its round index: attn_meta_arms.ATTN) by typed control while paused idle FULL."),
    ("# descriptive. Every timeout below is one alpha_next_c1_gate cap (instruments, arm_entries, alpha_next_c1_arms, ingress,",
     "# descriptive. Every timeout below is one attn_meta_gate cap (instruments, arm_entries, attn_meta_arms, ingress,"),
    ('local _part=$1 _label=$2 _rec=record _skip=record _limit=$LAST_ARM_ENTRY_S _sel _cohort _pair _c1 _group _workload _entry_rc',
     'local _part=$1 _label=$2 _rec=record _skip=record _limit=$LAST_ARM_ENTRY_S _sel _cohort _pair _c1 _group _workload _entry_rc _attn'),
    ('IFS=. read -r _c1 _group <<< "$_sel"\n',
     'IFS=. read -r _c1 _group <<< "$_sel"\n  if [[ $_pair == @ATTN_ON@ ]]; then _attn=on; else _attn=off; fi\n'),
    ('python3 "$H/e3_alpha_next_c1_box.py"', 'python3 "$H/e3_attn_meta_box.py"'),
    ('python3 "$H/e3_attn_meta_box.py" arm \\\n', 'python3 "$H/e3_attn_meta_box.py" arm --attn "$_attn" \\\n'),
    ("'auto arm: 8 standing decoders, burst, fused, the group size, C0 local, the C1 selection and defer_bind, fresh, append; live keeps them, solo cancels'",
     "'auto arm: 8 standing decoders, burst, fused, the group size, C0 local, the C1 selection and defer_bind, the attn-meta selection, fresh, append; live keeps them, solo cancels'"),
    ('$Q/alpha-next-c1-', '$Q/attn-meta-'),
    ("record alpha-next-c1 \"$?\" 'core arms: per-arm mechanism (crash-free, TP commitment, group, C0 and C1 engagement, one command in "
     "flight per stage, OOM), numerics envelope, complete progress lines; speed, decode and memory descriptive'",
     "record attn-meta \"$?\" 'decode-only and core arms: per-arm mechanism (crash-free, TP commitment, group, C0 and C1 engagement, one "
     "command in flight per stage, OOM, attn-meta selection leak and engagement, the order lines), numerics envelope, complete progress "
     "lines; speed, decode and memory descriptive'"),
    ('read_step alpha-next-c1-read "$?"', 'read_step attn-meta-read "$?"'),
    ('GRADE_ARGS=(); READ_ARGS=()\n', '@DECODE_FUNCTION@GRADE_ARGS=(); READ_ARGS=()\n'),
    ('  for _arm in @CORE@; do\n', '  for _label in @DECODE@; do\n    attn_meta_decode_arm "$_label"\n  done\n  for _arm in @CORE@; do\n'),
    ('g4 tail', 'G2 tail'),
)


def c1_template():
    """c1's body before its arm lists were filled in (c1q.body's own steps, placeholders kept)."""
    text = alpha_next.BODY
    for old, new in c1q.BODY_EDITS:
        assert old in text, old[:100]
        text = text.replace(old, new)
    return text


def decode_case():
    return '\n'.join('    (%s) _attn=%s;;' % (d['label'], 'on' if d['attn'] else 'off') for d in table.DECODE_ARMS)


def body():
    text = c1_template()
    for old, new in BODY_EDITS:
        assert old in text, old[:100]
        text = text.replace(old, new)
    text = text.replace('@DECODE_FUNCTION@', DECODE_FUNCTION.replace('@DECODE_CASE@', decode_case()))
    return (text.replace('@CORE@', ' '.join(CORE)).replace('@TAIL@', ' '.join(TAIL)).replace('@BAND1@', table.band1_glob())
            .replace('@DECODE@', ' '.join(DECODE)).replace('@ATTN_ON@', ATTN_ON_GLOB))


DRAFT_NAMES = (
    ('from alpha_next_c1_gate import ACTIVE_S', 'from attn_meta_gate import ACTIVE_S', 1),
    ('(alpha_next_c1_gate, bash SECONDS)', '(attn_meta_gate, bash SECONDS)', 1),
    ('every cap here is alpha_next_c1_gate.\n', 'every cap here is attn_meta_gate.\n', 1),
    ('Every cap is alpha_next_c1_gate.PHASES.', 'Every cap is attn_meta_gate.PHASES.', 1),
    ('"$H/runs/q18a_alpha_next_c1_settings.sh"', '"$H/runs/q18a_attn_meta_settings.sh"', 1),
    ('-r $H/runs/q18a_alpha_next_c1_settings.sh &&', '-r $H/runs/q18a_attn_meta_settings.sh &&', 1),
    ('RUN_ID="q18a-alpha-next-c1-$(date', 'RUN_ID="q18a-attn-meta-$(date', 1),
    ("perf=p.parent/'alpha-next-c1-grade.json'", "perf=p.parent/'attn-meta-grade.json'", 1),
    ("'missing':['alpha-next-c1-grade-absent']", "'missing':['attn-meta-grade-absent']", 1),
    ("tail=p.parent/'alpha-next-c1-read.json'", "tail=p.parent/'attn-meta-read.json'", 1),
)
DECLARE_NAMES = (
    ('QNAME=${QNAME:-q18a-alpha-next-c1}', 'QNAME=${QNAME:-q18a-attn-meta}', 1),
    ('"$H/runs/q18a_alpha_next_c1_settings.sh"', '"$H/runs/q18a_attn_meta_settings.sh"', 2),
    ('"$H/runs/q18a_alpha_next_c1.draft.sh"', '"$H/runs/q18a_attn_meta.draft.sh"', 2),
    ('"$H/runs/q18a_alpha_next_c1_declare.sh"', '"$H/runs/q18a_attn_meta_declare.sh"', 1),
)
SETTINGS_NAMES = (
    ('QNAME=${QNAME:-q18a-alpha-next-c1}', 'QNAME=${QNAME:-q18a-attn-meta}', 1),
)
FILES = 'e3_attn_meta_box.py attn_meta_gate.py attn_meta_arms.py'
INTERFACE_OLD = c1q.INTERFACE_NEW
INTERFACE_NEW = INTERFACE_OLD[:-len('; do')] + ' \\\n  ' + FILES + '; do'
PINS_OLD = c1q.PINS_NEW
PINS_NEW = PINS_OLD + FILES + ' '
SETTINGS_OLD = c1q.SETTINGS_OLD + c1q.SWITCH + ' '
SETTINGS_ANCHOR = c1q.SETTINGS_NOTE.splitlines()[0]
SETTINGS_NOTE = ('# attn-meta: FLASH_ATTN_META_GDN_BATCH=1 below (attn_meta_select installs the per-arm Selector over attn_meta_batch at\n'
                 '# initialize_kv_cache, before handler anatomy, not selected), for the whole run, the only setting added to\n'
                 '# alpha-next-c1; each arm selects attn-meta off or on by the typed qualification worker action attn_meta_select.\n')
DRAFT_SWITCHES = c1q.DRAFT_SWITCHES + "assert env.get('%s') == '1', 'attn-meta per-arm selector installed'\n" % SWITCH_NAME
DECLARE_SWITCHES = c1q.DECLARE_SWITCHES + "assert env.get('%s')=='1', 'attn-meta runs with the per-arm selector installed'\n" % SWITCH_NAME


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
            for old, new, count in DRAFT_NAMES:
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
    import attn_meta_gate as gate
    lines = ['q18a-attn-meta arm table (harness/attn_meta_arms.py); entry limits relative to the run start, never a clock',
             '%-3s %-18s %-6s %-4s %-5s %-5s %-4s %-6s %-4s %-5s %s' % ('#', 'label', 'part', 'cell', 'attn', 'c1', 'size', 'cohort',
                                                                 'band', 'round', 'enters while SECONDS <=')]
    for k, row in enumerate(gate.rows(), 1):
        lines.append('%-3d %-18s %-6s %-4s %-5s %-5s %-4d %-6s %-4s %-5s %s %d s' % (
            k, row['label'], row['part'], row['cell'], 'on' if row['attn'] else 'off', row['c1'], row['size'], row['cohort'],
            row.get('band', '-'), row.get('index', '-'), row['entry_limit'], gate.ENTRY_LIMIT_S[row['entry_limit']]))
    exp = gate.EXPECTED_RUNTIME_S
    e, slow = exp['nominal'], exp['slower']
    lines.append('arms %d (decode %d, core %d, tail %d); box budget %d s, cleanup included; worst case of the caps %d s' % (
        exp['arms'], exp['decode_arms'], exp['core_arms'], exp['tail_arms'], gate.BOX_BUDGET_S, gate.WORST_CASE_SUM_S))
    lines.append('expected (c1 walls): %d arms entering, last arm ends +%d s, about %.1f min with startup and the end; smallest '
                 'entry margin %d s; the lead\'s basis 17 x 2.1 min = %.1f min of arms' % (
                     e['arms_entering'], e['last_arm_ends_s'], e['total_min'], e['min_margin_s'], exp['per_arm_all_in_min']))
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
            assert a.replace_derived or not path.exists() or path.read_text() == text, (
                'refusing to change an existing derived text: ' + name)
            path.write_text(text)
            path.chmod(0o664)
    print('attn-meta queue ' + ('verified' if a.check else 'written'))


if __name__ == '__main__':
    main()
