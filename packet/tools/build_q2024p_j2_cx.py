#!/usr/bin/env python3
"""Compose one block of a registered q2024p-j2 composition cohort (lead 12 Nov 2026, branch claude/n3-compose).

A cohort is four blocks cx<tag>-b1 .. cx<tag>-b4 (tools/q2024p_j2_cx_render.py), each an ordinary four-arm j2 packet
(sp.g2-live-0, n3.g2-live-0, n3.g2-live-1, sp.g2-live-1 at 480 s, j2's seeds: the V1 cohort's shape), all from ONE
source commit and ONE flag set given on the command line; a block differs from its siblings only in its name.

Method: the night-pack tool (tools/build_q2024p_j2_np.py, through it the ing, yr, yrs and host-gap batch tools) with
  * the flag set: the night pack's five flags and the ingress verify settings, plus
    FLASH_S5_APPEND_NO_BLOCK_STOP (lever A), FLASH_S5_DECODE_GROUPS (chunk|off|off-small; harness/n3_decode_groups) and
    FLASH_S5_DECODE_LIVENESS (the engine's decode liveness observer the arm guard reads); all in the settings line and
    asserted by the declaration and the draft;
  * the bundle's n3j2_packet.configure wrapped by n3_decode_groups (marked tail), the draft exporting
    FLASH_S5_DECODE_GROUPS before its first python, so every process the draft starts reads the same table;
  * the arm liveness guard (harness/decode_liveness_guard: ends an arm only on a still-stuck decoder) bound in the
    bundle's n3j2_box with its status-row line, and the arm gate closed by ENGINE_CRASHED (the dmp-b review fix);
    the binding declares the packet's own FLASH_S5_DECODE_LIVENESS (``observer='on'|'off'``, GUARD_INSTALL, brief 11):
    with the observer off the liveness reading is NOT_APPLICABLE and not a missing guard (window 13: all seven cx
    packets INC only for guard-blind on every arm); with it on a blind guard still makes the arm INC;
  * the engine modules the composition moved (stage_overlap, native_overlap_control) bound as well, and the CPU suites
    of every lever (test_decode_merge_probe stays a hub suite: its bundle test needs the dmp table and budget).
  * FLASH_S5_CONTROL_INWAIT (Codex, a15bed1d: FULL-entry identity observed beside decode, the duplicate barrier
    skipped) in the settings line and asserted; its CPU suite test_control_inwait runs in the frozen CPU with its frozen
    off-path parent (46d28946's native_auto_arm.py, sha-pinned by the suite) copied into the bundle and named by
    CONTROL_INWAIT_PARENT_SOURCE (the container has no git).
  * FLASH_S5_CONTROL_O3_INWAIT (Codex, a59b1e7f, scope 2: a kept-SPLIT re-entry or entry arms on a finite applied-output
    prefix while later decode stays queued; a refused census drains before the FULL fallback) in the settings line and
    asserted; it needs CONTROL_INWAIT=1 (refused otherwise) and AUTO_ARM=1 and O3=1 in the parent's settings (asserted at
    build time); its suite test_control_o3_inwait runs in the frozen CPU with a15bed1d's native_auto_arm.py as its
    frozen OFF parent (CONTROL_O3_PARENT_SOURCE).
  * FLASH_S5_DECODE_SAMPLE_FAST (lever D) was blocked by Codex's engine review (13 Nov 13:23) until a commit marked as
    the reviewed fix is on the line. That fix is 2eba0dd0 (Codex re-review 13 Nov, no new blocker), merged onto the
    compose line with the watermark-sampler refusal kept. D=1 now builds only from a source commit that has
    DECODE_SAMPLE_FAST_REVIEWED_FIX as an ancestor AND whose regenerated overlay decode_step.py carries both the
    exact overflow route and the watermark refusal (DECODE_SAMPLE_FAST_MARKERS); anything else is refused.
    FLASH_S5_DECODE_ANATOMY stays available.
  * FLASH_S5_DECODE_TIMELINE (Codex 84135895 + 1469bd85, log-only, default off: bounded execute/sample samples inside and
    outside native progress with each rank's own queried forward completion) in the settings line and asserted; it
    needs FLASH_S5_HANDLER_ANATOMY=1, AUTO_ARM=1, NATIVE_PROGRESS=burst and FUSED_PHASE_EVERY in 1..4096 in the parent's
    settings (TIMELINE_NEEDS, asserted at build time; decode_timeline.check_startup refuses the engine otherwise); its
    module is a new overlay file and bound; its suite test_decode_timeline runs in the frozen CPU.
Refused flag sets: settle with the yield window (settle sits where the window hands off), a decode-group mode other than
chunk without the liveness observer (the guard would be blind), verify settings without ingress, and CONTROL_INWAIT with
any of settle, the small-piece batch, ingress, a decode-group mode other than chunk or the liveness observer (none of
these pairs is tested together; test_control_inwait covers A/Y1/Y2, test_n3_compose_pairs A x Y2 on the whole window).
No box command, GPU, VM, deletion or push.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


np = _load('build_q2024p_j2_np', 'tools/build_q2024p_j2_np.py')
dmp = _load('build_q2024p_j2_dmp', 'tools/build_q2024p_j2_dmp.py')
ing, yr, yrs, hgb = np.ing, np.yr, np.yrs, np.hgb

PARENT_NAME = hgb.PARENT_NAME
FAMILY = 'q2024p-j2-cx'
# brief 12: fixed work (harness/compose4_replay, REPLAY_FIXED_WORK): name -> (per-seed targets, cap s). n480: per agent
# the most turns EVERY stock arm of the twelve j2 records with seeds 1 and 2 (cxay12 b1-b4, V1 b1-b4, cxaycis,
# cxaing, cxayc2s, cxayc2t2) completed by 480 s (cx-20261114d/scratch/nchoice); seed 1 121 turns, 6 session rolls,
# seed 2 129 turns, 4 rolls. The cap: four replay arms at the cap still enter inside n3j2_gate's LAST_ARM_ENTRY_S.
FIXED_WORK = {'n480': ('1=15,17,14,17,18,14,12,14/2=18,15,16,17,15,16,16,16', 640)}
# flag -> (VARIANTS key, allowed values, default); the order is the settings line's
FLAGS = (
    ('FLASH_S5_SMALL_PIECE_BATCH', 'small_piece_batch', ('0', '1'), '0'),
    ('FLASH_S5_YIELD_REENTRY_SETTLE', 'yield_reentry_settle', ('0', '1'), '0'),
    ('FLASH_S5_YIELD_NO_APPEND_OWNER', 'no_append_owner', ('0', '1'), '0'),
    ('FLASH_S5_YIELD_WINDOW', 'yield_window', ('0', '1'), '0'),
    ('FLASH_S5_INGRESS_INCREMENTAL', 'ingress_incremental', ('0', '1'), '0'),
    ('FLASH_S5_INGRESS_VERIFY_EVERY', 'verify_every', None, None),          # positive int; only with ingress
    ('FLASH_S5_INGRESS_VERIFY_ASYNC', 'verify_async', ('0', '1'), None),    # only with ingress
    ('FLASH_S5_APPEND_NO_BLOCK_STOP', 'append_no_block_stop', ('0', '1'), '0'),
    ('FLASH_S5_DECODE_GROUPS', 'decode_groups', ('chunk', 'off', 'off-small'), 'chunk'),
    ('FLASH_S5_DECODE_LIVENESS', 'decode_liveness', ('0', '1'), '0'),
    ('FLASH_S5_CONTROL_INWAIT', 'control_inwait', ('0', '1'), '0'),
    ('FLASH_S5_CONTROL_O3_INWAIT', 'control_o3_inwait', ('0', '1'), '0'),   # scope 2, kept-SPLIT (Codex a59b1e7f)
    ('FLASH_S5_DECODE_SAMPLE_FAST', 'decode_sample_fast', ('0', '1'), '0'),   # lever D (decode_step, 13 Nov)
    ('FLASH_S5_DECODE_ANATOMY', 'decode_anatomy', ('0', '1'), '0'),           # clean-window anatomy (decode_step)
    ('FLASH_S5_DECODE_TIMELINE', 'decode_timeline', ('0', '1'), '0'),         # log-only stamp (decode_timeline, 13 Nov)
    ('REPLAY_REPLY_TEXT', 'replay_reply_text', ('model', 'seeded'), 'model'),  # harness only: compose4_replay (brief 11)
    ('QUIET_SPLIT_CONTROL', 'quiet_split_control', ('0', '1'), '0'),     # harness only: the tail read (brief 11 C)
    ('REPLAY_FIXED_WORK', 'replay_fixed_work', ('0',) + tuple(sorted(FIXED_WORK)), '0'),   # harness only (brief 12)
    ('REPLAY_SEED_ORDER', 'replay_seed_order', ('plain', 'swap', 'alternate'), 'plain'),   # harness only (brief 12)
)
# harness flags: not on the engine's settings line and not in its env asserts; the draft exports them (when not default)
HARNESS_FLAGS = ('REPLAY_REPLY_TEXT', 'QUIET_SPLIT_CONTROL', 'REPLAY_FIXED_WORK', 'REPLAY_SEED_ORDER')
REPLY_TEXT_EXPORT = ('export REPLAY_REPLY_TEXT=seeded  # brief 11 B.1: compose4_replay puts a seeded synthetic reply '
                     'text in the history (every python started below)\n')
FIXED_WORK_EXPORT = ('export REPLAY_FIXED_WORK=%s REPLAY_FIXED_CAP_S=%s  # brief 12: every agent sends until the '
                     'last has completed its target turns; the cap is a guard (an arm at it is INC)\n')
SEED_ORDER_EXPORT = ('export REPLAY_SEED_ORDER=%s  # brief 12: harness/fixed_work_packet sets the seed of each replay '
                     'pair (n3j2_packet.configure, the mechanism rule)\n')
# alternate: the cohort's blocks carry the two orders in turn, so each seed runs stock first and engine first
SEED_ORDER_OF = dict(plain={}, swap={}, alternate=dict(b2='swap', b4='swap'))
SEED_PACKET_TAIL = ('\n# brief 12 (q2024p-j2-cx): REPLAY_SEED_ORDER sets the seed each replay pair carries (harness/fixed_work_packet;\n'
                    '# plain, the default, is j2\'s dict unchanged). The draft exports it.\n'
                    'import fixed_work_packet as _fwp\n'
                    'configure = _fwp.wrap_env(configure)\n')
# the mechanism rule's 480 s literals and its seed of a label, asked of harness/fixed_work_packet (env-gated) instead
RULE_PATCH = (
    ('import native43_engine_death_h as death\n',
     'import native43_engine_death_h as death\nimport fixed_work_packet as _fwp  # brief 12: fixed work, seed order\n'),
    ("and row['t'] <= 480 and row['seconds'] >= row['ttft']",
     "and row['t'] <= _fwp.send_limit_s() and row['seconds'] >= row['ttft']"),
    ("dict(table.REPLAY, SEED=1 + int(label[-1]))", "dict(table.REPLAY, SEED=_fwp.expected_seed(label))"),
    ("arm['wall_start_s'] <= arm['replay_start_s'] < arm['replay_start_s'] + 480 <= arm['replay_end_s'] <= arm['wall_end_s']",
     "arm['wall_start_s'] <= arm['replay_start_s'] < arm['replay_end_s'] <= arm['wall_end_s'] and _fwp.replay_window_ok(arm)"),
)


def seed_order_of(flag, variant):
    """The block's own seed order (plain or swap) under the cohort's REPLAY_SEED_ORDER."""
    if flag == 'alternate':
        return SEED_ORDER_OF['alternate'].get(variant, 'plain')
    return flag


def packet_edits_on(flags):
    """The seed-order tail and the rule patch go in only when fixed work or a seed order is set."""
    return flags['REPLAY_FIXED_WORK'] != '0' or flags['REPLAY_SEED_ORDER'] != 'plain'


def rule_patch(text):
    for old, new in RULE_PATCH:
        text = hgb.once(text, old, new)
    return text


QUIET_SPLIT_EXPORT = ('export QUIET_SPLIT_CONTROL=1  # brief 11 C: the tail runs the quiet-SPLIT control '
                      '(e3_quiet_split_box), a read\n')
# brief 11 C: the quiet-SPLIT control as a tail READ (never the queue verdict; the core grade is written before it), on a
# healthy engine inside LAST_TAIL_ENTRY_S (the tail arm's reserve, 1.5 x 519 s, holds it: about 3 x 35 s), after the
# typed all-rank FULL entry that takes the manual lease (no auto-arm cycle, no native work). The tail-engine check,
# instruments-end and end-full-idle follow it unchanged and are counted, so a control that leaves the engine off FULL
# or changed its rank processes still fails the packet.
QUIET_SPLIT_ANCHOR = ('  for _label in ; do\n'
                      '    alpha_next_arm tail "$_label"\n'
                      '  done\n')
QUIET_SPLIT_STEP = '''  # brief 11 C: the quiet-SPLIT control (e3_quiet_split_box): 8, 4 and 1 live decoders on the split route, no native
  # work, decode_merge=always, 25 s each. A read: never counted. Gated by the export above.
  if [[ ${QUIET_SPLIT_CONTROL:-0} == 1 ]]; then
    n=$(systemctl show "$SERVICE" -p NRestarts --value 2>/dev/null) || n=unknown
    if (( SECONDS > LAST_TAIL_ENTRY_S )); then
      read_step quiet-split 2 "NOT_RUN: box budget, elapsed ${SECONDS}s > ${LAST_TAIL_ENTRY_S}s"
    elif [[ $n != "$N0" ]] || [[ $ENGINE_CRASHED != 0 ]] || host_crash_detected || ! health_ok; then
      read_step quiet-split 2 'NOT_RUN: engine crashed, unhealthy or auto-restarted; no restart attempted'
    else
      engine_watched entry-quiet-split timeout 450s python3 "$H/e3_native39_trunk_checks.py" full-idle --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \\
        --control-timeout-s 60 --timeout-s "$IDLE_TIMEOUT_S" --out "$Q/entry-quiet-split.json" > "$Q/entry-quiet-split.log" 2>&1
      _qs_rc=$?; read_step entry-quiet-split "$_qs_rc" 'typed all-rank FULL, resumed, idle, workers empty; lease taken'
      if [[ $_qs_rc == 0 ]]; then
        engine_watched quiet-split timeout --signal=TERM --kill-after=30s 360s python3 "$H/e3_quiet_split_box.py" \\
          --url "$PROBE_URL" --model "$MODEL" --run-id "$RUN_ID" --epoch "$EPOCH" --stages 8,4,1 --hold-s 25 \\
          --decode-merge always --control-timeout-s 60 --out "$Q/quiet-split.json" > "$Q/quiet-split.log" 2>&1
        read_step quiet-split "$?" 'quiet-SPLIT control: 8, 4, 1 live decoders on split, no native work, decode_merge=always, 25 s each'
      else
        read_step quiet-split 2 'NOT_RUN: quiet-split entry not a typed idle FULL'
      fi
    fi
  fi
'''
QUIET_SPLIT_NEEDS = ('FLASH_S5_AUTO_ARM=1 ', 'FLASH_S5_DECODE_MERGE=1 ', 'FLASH_S5_LAZY_RECV=1 ',
                     'FLASH_S5_STAGE_OVERLAP_DEFER_BIND=1 ')
EXTRA = ('append_no_block_stop', 'decode_groups', 'decode_liveness', 'control_inwait', 'control_o3_inwait',
         'decode_sample_fast', 'decode_anatomy', 'decode_timeline')                                                     # added on top of np's queues
# CONTROL_INWAIT=1 is refused with each of these (flag, value that is NOT refused): untested together
CONTROL_INWAIT_UNTESTED = (('FLASH_S5_YIELD_REENTRY_SETTLE', '0'), ('FLASH_S5_SMALL_PIECE_BATCH', '0'),
                           ('FLASH_S5_INGRESS_INCREMENTAL', '0'), ('FLASH_S5_DECODE_GROUPS', 'chunk'),
                           ('FLASH_S5_DECODE_LIVENESS', '0'))
CONTROL_INWAIT_PARENT = ('46d28946', 'patches/s5_engine/native_auto_arm.py',
                         '3d5c813856f0c7acd68225a2df69c157d7d2aca2b653ea19ee0586ea4e1f9672')
CONTROL_INWAIT_PARENT_FILE = 'control_inwait_parent_native_auto_arm.py.src'   # bundle root = /src/harness; not importable
# scope 2's suite (test_control_o3_inwait) compares its OFF kept-SPLIT trace with a15bed1d's native_auto_arm.py
CONTROL_O3_PARENT = ('a15bed1d', 'patches/s5_engine/native_auto_arm.py',
                     '8090de4a9c349c89666916358b0c26f61c3ec0e8a6366545a4d990b4e89f5ee5')
CONTROL_O3_PARENT_FILE = 'control_o3_parent_native_auto_arm.py.src'
CONTROL_INWAIT_ENV = ("'-e','PYTHONDONTWRITEBYTECODE=1',",
                      "'-e','PYTHONDONTWRITEBYTECODE=1','-e','CONTROL_INWAIT_PARENT_SOURCE=/src/harness/%s',"
                      "'-e','CONTROL_O3_PARENT_SOURCE=/src/harness/%s',"
                      % (CONTROL_INWAIT_PARENT_FILE, CONTROL_O3_PARENT_FILE))
# scope 2 needs these in the parent's settings line (s5_startup refuses the engine otherwise; refused here first)
CONTROL_O3_NEEDS = ('FLASH_S5_AUTO_ARM=1 ', 'FLASH_S5_O3=1 ')
# the decode-timeline stamp needs these in the parent's settings line (decode_timeline.check_startup); refused here first
TIMELINE_NEEDS = ('FLASH_S5_HANDLER_ANATOMY=1 ', 'FLASH_S5_AUTO_ARM=1 ', 'FLASH_S5_NATIVE_PROGRESS=burst ')
TIMELINE_FUSED = re.compile(r'(?<![A-Z_])FLASH_S5_FUSED_PHASE_EVERY=([0-9]+) ')
# lever D (FLASH_S5_DECODE_SAMPLE_FAST): blocked 13 Nov 13:23 (Codex's engine review
# REVIEW-codex-decode-step-20261113.md: more than WIDTH - K ties at the K-th value sampled a truncated support, counted
# but never a fallback). The reviewed fix is 2eba0dd0 on claude/n3-decode-step ("lever D reviewed-fix"; Codex re-review
# REVIEW-codex-decode-step-2eba0dd0-20261113.md: previous blockers fixed, no new blocker). D=1 is accepted only when
# (1) this full sha is an ancestor of the source commit (git merge-base --is-ancestor) and (2) the bundle's regenerated
# overlay e3/decode_step.py contains every DECODE_SAMPLE_FAST_MARKERS line: the fix's per-row overflow select and its
# exact route, and the compose line's watermark-sampler refusal (1cfc2166). Set to None to block D again.
DECODE_SAMPLE_FAST_REVIEWED_FIX = '2eba0dd0d85f341ac9378c438ac4803c946104d8'
DECODE_SAMPLE_FAST_MARKERS = (
    '        return torch.where(overflow.view(-1), exact, fast)\n',            # 2eba0dd0: overflow rows exact, pre-draw
    '    def sample_overflow(self, values, ids, pivot, ties, tie_count, k, top_p):\n',
    "                if owner != 'Sampler' or getattr(sampler, 'watermarking', None) is not None:\n",   # 1cfc2166
)


def decode_sample_fast_refusals(source_text, is_ancestor):
    """Why lever D may not build from this source ([] = it may): no reviewed fix set, the fix not an ancestor of the
    source commit (``is_ancestor(sha)``), or a DECODE_SAMPLE_FAST_MARKERS line missing from ``source_text`` (the
    source's, then the regenerated overlay's decode_step.py)."""
    if DECODE_SAMPLE_FAST_REVIEWED_FIX is None:
        return ['no-reviewed-fix']
    out = [] if is_ancestor(DECODE_SAMPLE_FAST_REVIEWED_FIX) else ['reviewed-fix-not-an-ancestor']
    return out + ['missing-marker-%d' % i for i, m in enumerate(DECODE_SAMPLE_FAST_MARKERS) if m not in source_text]


BLOCKS = ('b1', 'b2', 'b3', 'b4')
DRY = 'dry'
# decode_timeline is new; Codex's stamp also moves handler_anatomy and native_measurement in the overlay (their
# timeline hooks), so the parent's regenerate refuses them undeclared (first ayc2t build, 13 Nov 19:15)
BOUND_MODULES = ing.BOUND_MODULES + ('stage_overlap', 'native_overlap_control', 'decode_step', 'decode_timeline',
                                     'handler_anatomy', 'native_measurement')
ENGINE_MOVED = {'e3/' + n + '.py' for n in BOUND_MODULES}
# decode_step (13 Nov) and decode_timeline (Codex 1469bd85): new in the overlay
NEW_OVERLAY_FILES = set(np.NEW_OVERLAY_FILES) | {'e3/decode_step.py', 'e3/decode_timeline.py'}
SUITES = np.SUITES + ('test_append_no_block_stop', 'test_append_no_block_stop_stack', 'test_n3_decode_groups',
                      'test_dmp_reader', 'test_q2024p_j2_cx', 'test_n3_compose_pairs', 'test_control_inwait',
                      'test_decode_step', 'test_decode_step_torch', 'test_control_o3_inwait', 'test_decode_timeline',
                      'test_replay_reply_text', 'test_quiet_split_box', 'test_replay_fixed_work',
                      'test_fixed_work_packet', 'test_q2024p_j2_long_reader')
TOOLS = np.TOOLS + ('tools/build_q2024p_j2_dmp.py', 'tools/q2024p_j2_dmp_reader.py', 'tools/build_q2024p_j2_cx.py', 'tools/q2024p_j2_cx_finish.py', 'tools/q2024p_j2_cx_render.py',
                    'tools/q2024p_j2_cx_reader.py', 'tools/q2024p_j2_long_reader.py')
CX_CONTAINER = "'q2024p-j2-cx-frozen-cpu-'"
PACKET_TAIL = ('\n# composition branch (q2024p-j2-cx): FLASH_S5_DECODE_GROUPS sets the n3 selection\'s decode_merge\n'
               '# (harness/n3_decode_groups; chunk, the default, is j2\'s table unchanged). The draft exports it.\n'
               'import n3_decode_groups as _n3dg\n'
               'configure = _n3dg.wrap_env(configure)\n')
DRAFT_EXPORT = ('export FLASH_S5_DECODE_GROUPS=%s  # composition branch: n3j2_packet.configure reads it in every python '
                'started below\n')
VARIANTS = np.VARIANTS          # filled by configure(); np's helpers read this dict


def parse_flags(pairs):
    """KEY=VALUE pairs over the defaults; refuses unknown keys, bad values and the refused combinations."""
    names = {f[0]: f for f in FLAGS}
    flags = {f[0]: f[3] for f in FLAGS}
    for pair in pairs:
        key, sep, value = pair.partition('=')
        if not sep or key not in names:
            raise SystemExit('unknown flag %r (known: %s)' % (pair, ' '.join(names)))
        allowed = names[key][2]
        if allowed is None:
            if not re.fullmatch(r'[1-9][0-9]{0,4}', value):
                raise SystemExit('%s must be a positive integer' % key)
        elif value not in allowed:
            raise SystemExit('%s must be one of %s' % (key, '|'.join(allowed)))
        flags[key] = value
    ingress = flags['FLASH_S5_INGRESS_INCREMENTAL'] == '1'
    for key in ('FLASH_S5_INGRESS_VERIFY_EVERY', 'FLASH_S5_INGRESS_VERIFY_ASYNC'):
        if flags[key] is not None and not ingress:
            raise SystemExit('%s needs FLASH_S5_INGRESS_INCREMENTAL=1' % key)
        if flags[key] is None and ingress:
            raise SystemExit('FLASH_S5_INGRESS_INCREMENTAL=1 needs %s set explicitly (np-ing: 1, np-all16: 16)' % key)
    if flags['FLASH_S5_YIELD_REENTRY_SETTLE'] == '1' and flags['FLASH_S5_YIELD_WINDOW'] == '1':
        raise SystemExit('refused: FLASH_S5_YIELD_REENTRY_SETTLE=1 with FLASH_S5_YIELD_WINDOW=1 (settle sits where the '
                         'yield window hands off; keep it out of a Y2 build)')
    if flags['FLASH_S5_DECODE_GROUPS'] != 'chunk' and flags['FLASH_S5_DECODE_LIVENESS'] != '1':
        raise SystemExit('refused: FLASH_S5_DECODE_GROUPS=%s needs FLASH_S5_DECODE_LIVENESS=1 (the arm guard reads it)'
                         % flags['FLASH_S5_DECODE_GROUPS'])
    if flags['FLASH_S5_CONTROL_O3_INWAIT'] == '1' and flags['FLASH_S5_CONTROL_INWAIT'] != '1':
        raise SystemExit('refused: FLASH_S5_CONTROL_O3_INWAIT=1 needs FLASH_S5_CONTROL_INWAIT=1 (scope 2 extends scope 1; '
                         'it also needs AUTO_ARM=1 and O3=1, checked in the settings at build time)')
    if flags['FLASH_S5_DECODE_SAMPLE_FAST'] == '1' and DECODE_SAMPLE_FAST_REVIEWED_FIX is None:
        raise SystemExit('refused: FLASH_S5_DECODE_SAMPLE_FAST=1 is blocked (Codex engine review 13 Nov: more than '
                         'WIDTH - K ties at the K-th value truncate the support without a fallback); no reviewed fix '
                         'is on the line')
    if flags['REPLAY_FIXED_WORK'] != '0' and flags['REPLAY_REPLY_TEXT'] != 'seeded':
        raise SystemExit('refused: REPLAY_FIXED_WORK=%s needs REPLAY_REPLY_TEXT=seeded (the first N turns are the same '
                         'work on both engines only then)' % flags['REPLAY_FIXED_WORK'])
    if flags['FLASH_S5_CONTROL_INWAIT'] == '1':
        for key, safe in CONTROL_INWAIT_UNTESTED:
            if flags[key] != safe:
                raise SystemExit('refused: FLASH_S5_CONTROL_INWAIT=1 with %s=%s (untested together; only lever A, Y1 '
                                 'and Y2 are composed with it)' % (key, flags[key]))
    return flags


def variant_record(flags):
    return {f[1]: flags[f[0]] for f in FLAGS}


def name_of(variant, tag=None):
    tag = CONFIG['tag'] if tag is None else tag
    assert variant in BLOCKS + (DRY,), variant
    assert re.fullmatch(r'[a-z0-9]{0,8}', tag), tag
    return PARENT_NAME + '-cx' + tag + '-' + variant


CONFIG = dict(tag='')


def configure(flags, tag):
    """Point np's (and through it ing's, yr's, yrs's and hgb's) helpers at this cohort."""
    CONFIG['tag'] = tag
    VARIANTS.clear()
    VARIANTS.update({v: variant_record(flags) for v in BLOCKS + (DRY,)})
    ing.BOUND_MODULES, ing.ENGINE_MOVED = BOUND_MODULES, ENGINE_MOVED
    np.name_of = name_of
    np.SUITES, np.TOOLS, np.NP_CONTAINER = SUITES, TOOLS, CX_CONTAINER
    np.NEW_OVERLAY_FILES = ing.NEW_OVERLAY_FILES = NEW_OVERLAY_FILES
    np.queues = queues
    hgb.BOX_IMPORT_NEW = box_import(flags['FLASH_S5_DECODE_LIVENESS'])           # + the arm liveness guard
    hgb.BOX_ROW_NEW = dmp.BOX_ROW_NEW
    np.bind()


_np_queues = np.queues

GUARD_INSTALL = 'decode_liveness_guard.install(e3_compose5_box)  # decode-merge probe\n'
GUARD_INSTALL_NEW = ("decode_liveness_guard.install(e3_compose5_box, observer='%s')  # cx: the packet's "
                     "FLASH_S5_DECODE_LIVENESS\n")
OBSERVER_OF = {'0': 'off', '1': 'on'}


def box_import(liveness):
    """dmp's box import with the guard's observer declaration from the packet's own liveness setting."""
    assert dmp.BOX_IMPORT_NEW.count(GUARD_INSTALL) == 1, 'guard install line moved in the dmp binding'
    return dmp.BOX_IMPORT_NEW.replace(GUARD_INSTALL, GUARD_INSTALL_NEW % OBSERVER_OF[liveness])


def queues(variant):
    """np's triple for the variant's five flags (+ verify), then lever A, decode groups and liveness after the ingress
    setting with their assertions; the draft also exports the decode-group mode and closes its arm gate on a crash."""
    v = VARIANTS[variant]
    setting = '%s=%s ' % (ing.ING_ENV, v['ingress_incremental'])
    check = "assert env.get('%s') == '%s', 'ingress setting differs'\n" % (ing.ING_ENV, v['ingress_incremental'])
    if v['ingress_incremental'] == '1':
        setting = '%s=%s ' % (np.VERIFY_ASYNC_ENV, v['verify_async'])
        check = "assert env.get('%s') == '%s', 'ingress verify mode differs'\n" % (np.VERIFY_ASYNC_ENV, v['verify_async'])
    extra = [(f[0], v[f[1]]) for f in FLAGS if f[1] in EXTRA]
    result = {}
    for rel, text in _np_queues(variant).items():
        if rel.endswith('_settings.sh'):
            if v['decode_anatomy'] == '1':      # decode_step.check_startup refuses the engine otherwise; refuse here first
                for need in ('FLASH_S5_HANDLER_ANATOMY=1 ', 'FLASH_S5_NATIVE_PROGRESS=burst '):
                    assert need in text, 'FLASH_S5_DECODE_ANATOMY=1 needs %sin the settings' % need
            if v['control_o3_inwait'] == '1':   # s5_startup refuses the engine without them; refuse here first
                for need in CONTROL_O3_NEEDS:
                    assert need in text, 'FLASH_S5_CONTROL_O3_INWAIT=1 needs %sin the settings' % need
            if v['quiet_split_control'] == '1':  # the selection it sends is refused by the engine otherwise
                for need in QUIET_SPLIT_NEEDS:
                    assert need in text, 'QUIET_SPLIT_CONTROL=1 needs %sin the settings' % need
            if v['decode_timeline'] == '1':     # decode_timeline.check_startup refuses the engine otherwise
                for need in TIMELINE_NEEDS:
                    assert need in text, 'FLASH_S5_DECODE_TIMELINE=1 needs %sin the settings' % need
                every = set(TIMELINE_FUSED.findall(text))
                assert len(every) == 1 and 1 <= int(every.pop()) <= 4096, \
                    'FLASH_S5_DECODE_TIMELINE=1 needs one FLASH_S5_FUSED_PHASE_EVERY in 1..4096 in the settings'
            text = hgb.once(text, setting, setting + ''.join('%s=%s ' % kv for kv in extra))
        else:
            text = hgb.once(text, check, check + ''.join(
                "assert env.get('%s') == '%s', 'composition setting differs: %s'\n" % (k, x, k) for k, x in extra))
        if rel.endswith('.draft.sh'):
            text = hgb.once(text, dmp.ARM_GATE, dmp.ARM_GATE_NEW.replace('decode-merge probe (dmp-b)', 'dmp-b review fix'))
            head, sep, _ = text.partition(dmp.DRAFT_ANCHOR)
            code = [line for line in head.splitlines() if not line.lstrip().startswith('#')]
            assert sep and not any(m in line for line in code for m in ('n3j2_packet', 'n3j2_arms', 'python')), \
                'table use before the export'
            text = hgb.once(text, dmp.DRAFT_ANCHOR, dmp.DRAFT_ANCHOR + DRAFT_EXPORT % v['decode_groups'])
            assert 'REPLAY_REPLY_TEXT' not in text, 'the parent draft names REPLAY_REPLY_TEXT already'
            if v['replay_reply_text'] == 'seeded':
                text = hgb.once(text, DRAFT_EXPORT % v['decode_groups'], DRAFT_EXPORT % v['decode_groups'] + REPLY_TEXT_EXPORT)
            assert 'QUIET_SPLIT_CONTROL' not in text and 'e3_quiet_split_box' not in text, 'the parent draft has a quiet split'
            if v['quiet_split_control'] == '1':
                text = hgb.once(text, DRAFT_EXPORT % v['decode_groups'], DRAFT_EXPORT % v['decode_groups'] + QUIET_SPLIT_EXPORT)
                text = hgb.once(text, QUIET_SPLIT_ANCHOR, QUIET_SPLIT_ANCHOR + QUIET_SPLIT_STEP)
            assert not any(n in text for n in ('REPLAY_FIXED_WORK', 'REPLAY_SEED_ORDER', 'fixed_work_packet')), \
                'the parent draft names fixed work or a seed order already'
            if v['replay_fixed_work'] != '0':
                text = hgb.once(text, DRAFT_EXPORT % v['decode_groups'],
                                DRAFT_EXPORT % v['decode_groups'] + FIXED_WORK_EXPORT % FIXED_WORK[v['replay_fixed_work']])
            if v['replay_seed_order'] != 'plain':
                text = hgb.once(text, DRAFT_EXPORT % v['decode_groups'], DRAFT_EXPORT % v['decode_groups'] +
                                SEED_ORDER_EXPORT % seed_order_of(v['replay_seed_order'], variant))
        result[rel] = text
    return result


def copy_delta(b, commit, bundle):
    copied = np.copy_delta(b, commit, bundle)
    for name in ('harness/n3_decode_groups.py', 'harness/decode_liveness_guard.py', 'harness/test_n3_decode_groups.py',
                 'harness/test_append_no_block_stop.py', 'harness/test_append_no_block_stop_stack.py',
                 'harness/test_q2024p_j2_cx.py', 'harness/test_n3_compose_pairs.py',
                 'harness/test_control_inwait.py', 'harness/control_inwait_fixture_leaves.json',
                 'harness/test_control_o3_inwait.py', 'harness/control_o3_inwait_fixture_leaves.json',
                 'patches/s5_engine/native_auto_arm.py',
                 'patches/s5_engine/s5_startup.py', 'patches/s5_engine/executor_hooks.py',
                 'harness/test_decode_step.py', 'harness/test_decode_step_torch.py', 'patches/s5_engine/decode_step.py',
                 'patches/s5_engine/stage_overlap.py',
                 'patches/s5_engine/native_overlap_control.py', 'patches/s5_engine/scheduler_hooks.py',
                 'patches/s5_engine/decode_timeline.py', 'patches/s5_engine/handler_anatomy.py',
                 'patches/s5_engine/native_measurement.py', 'harness/test_decode_timeline.py',
                 'harness/decode_timeline_fixture_leaves.json', 'harness/decode_timeline_parent_ast.json',
                 'harness/receipt_lines.py', 'harness/compose4_replay.py', 'harness/test_replay_reply_text.py',
                 'harness/e3_quiet_split_box.py', 'harness/test_quiet_split_box.py',
                 'harness/e3_compose4_box.py', 'harness/fixed_work_packet.py', 'harness/test_fixed_work_packet.py',
                 'harness/test_replay_fixed_work.py', 'harness/test_q2024p_j2_long_reader.py'):
        assert name in copied, name
    return copied


def control_inwait_parent(b, bundle, pin=CONTROL_INWAIT_PARENT, file=CONTROL_INWAIT_PARENT_FILE):
    """A control-inwait suite's frozen off-path parent, by bytes from git (its sha checked here and again by the
    suite): test_control_inwait's (46d28946) and test_control_o3_inwait's (a15bed1d)."""
    commit, name, digest = pin
    target = bundle / file
    assert not target.exists(), target
    target.write_bytes(b.git(ROOT, 'show', commit + ':' + name))
    assert b.sha(target) == digest, 'control-inwait parent sha differs: ' + file
    return dict(status='A', bundle=file, sha256=digest, source=commit + ':' + name)


def frozen_cpu_tool(bundle):
    """yrs's frozen CPU tool, the container also naming the control-inwait parent source."""
    yrs.frozen_cpu_tool(bundle)
    path = bundle / 'tools/n3j2_frozen_cpu.py'
    path.write_text(hgb.once(path.read_text(), *CONTROL_INWAIT_ENV))


def prepare(commit, variant, out, flags, tag=''):
    """np.prepare's steps in its order, with this cohort's queues, bundle edits, delta check and record."""
    configure(flags, tag)
    b = hgb.helper()
    name = name_of(variant)
    commit = b.git(ROOT, 'rev-parse', commit + '^{commit}').decode().strip()
    assert len(commit) == 40
    assert not out.exists() and not out.is_symlink(), 'fresh output required'
    out.mkdir(parents=True)
    bundle = out / (name + '-bundle-pending')
    shutil.copytree(hgb.PARENT, bundle)
    for p in (bundle, *bundle.rglob('*')):
        p.chmod(0o755 if p.is_dir() else 0o644)
    copied = copy_delta(b, commit, bundle)
    copied['control_inwait_parent'] = control_inwait_parent(b, bundle)
    copied['control_o3_parent'] = control_inwait_parent(b, bundle, CONTROL_O3_PARENT, CONTROL_O3_PARENT_FILE)
    if flags['FLASH_S5_DECODE_SAMPLE_FAST'] == '1':
        def is_ancestor(sha):
            return subprocess.run(['git', 'merge-base', '--is-ancestor', sha, commit], cwd=ROOT).returncode == 0
        source = b.git(ROOT, 'show', commit + ':patches/s5_engine/decode_step.py').decode()
        refusals = decode_sample_fast_refusals(source, is_ancestor)
        assert not refusals, 'lever D refused for %s: %s' % (commit, refusals)
    box = bundle / 'n3j2_box.py'
    box.write_text(hgb.once(hgb.once(box.read_text(), hgb.BOX_IMPORT, hgb.BOX_IMPORT_NEW), hgb.BOX_ROW, hgb.BOX_ROW_NEW))
    observer = OBSERVER_OF[flags['FLASH_S5_DECODE_LIVENESS']]
    assert box.read_text().count(GUARD_INSTALL_NEW % observer) == 1, 'the guard binding does not declare observer=' + observer
    packet = bundle / 'n3j2_packet.py'
    packet.write_text(packet.read_text() + PACKET_TAIL)
    if packet_edits_on(flags):          # brief 12: off, the packet and the rule are the earlier bundle's bytes
        packet.write_text(packet.read_text() + SEED_PACKET_TAIL)
        rule = bundle / 'tools/n3j2_junction_rule.py'
        rule.write_text(rule_patch(rule.read_text()))
    for rel, text in queues(variant).items():
        (bundle / rel).write_text(text)
        subprocess.run(['bash', '-n', str(bundle / rel)], check=True)
    proof = hgb.regenerate(b, bundle, out)
    assert set(NEW_OVERLAY_FILES) <= set(proof['overlay_changed']), 'new overlay files not in the regenerated overlay'
    if flags['FLASH_S5_DECODE_SAMPLE_FAST'] == '1':
        built = (bundle / 'native30_overlay/e3/decode_step.py').read_text()
        refusals = decode_sample_fast_refusals(built, lambda sha: True)
        assert not refusals, 'lever D refused: the overlay decode_step.py %s' % refusals
    frozen_cpu_tool(bundle)
    hgb.queue_adapter(bundle, variant)
    for p in bundle.rglob('*.py'):
        compile(p.read_bytes(), str(p), 'exec')
    manifest = b.manifest(bundle)
    final = out / (name + '-bundle-' + manifest[:12])
    bundle.rename(final)
    runner = out / (name + '-owner-run-' + manifest[:12] + '.py')
    runner.write_text(hgb.runner_text(b, final, manifest, proof, commit, variant))
    compile(runner.read_bytes(), str(runner), 'exec')
    for p in (final, *final.rglob('*')):
        p.chmod(0o555 if p.is_dir() else 0o444)
    runner.chmod(0o444)
    record = dict(proof, schema='q2024p-j2-cx-build/1', name=name, variant=variant, family=FAMILY, tag=tag,
                  flags=dict(flags), diagnostic='always-on', liveness_guard='bound (decode_liveness_guard, observer=%s)' % observer,
                  new_overlay_files=sorted(NEW_OVERLAY_FILES), source_commit=commit, base_commit=hgb.BASE_COMMIT,
                  bundle=str(final), manifest=manifest, runner=str(runner), runner_sha256=b.sha(runner),
                  runner_size=runner.stat().st_size, held=True, hardware_qualified=False, exploratory=True,
                  parent_manifest=hgb.PARENT_SHA, parent_runner_sha256=hgb.PARENT_RUNNER_SHA,
                  parent_overlay=hgb.PARENT_OVERLAY, source_changes=copied, cpu_suites=SUITES,
                  hgb_modules=BOUND_MODULES, box_action=False, gpu_action=False)
    b.dump(out / 'build.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--variant', choices=BLOCKS + (DRY,), required=True)
    parser.add_argument('--tag', default='')
    parser.add_argument('--set', dest='flags', action='append', default=[], metavar='FLAG=VALUE')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source_commit, args.variant, args.out.resolve(), parse_flags(args.flags), args.tag)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_changes'}, sort_keys=True))
