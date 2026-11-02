"""alpha-next-c1 queue: the declared 14-arm table, the budget, the derived texts, the arm loop in bash, the arm's C1 control, and the
grade (alpha-next's plus C1's regroup, engagement, one command in flight per stage, OOM, memory-peak against the matched C1-off arm
and tears) over alpha-next's real emitters with C1's two chunks in flight. CPU only; no pass here is box evidence.

Classes:

* Table / Budget: the table, its order and matched pairs; 14 arms fit the relative budget with margin at the lead's box walls.
* ArmControl: the arm's seams over a fake control client: the stage command carries ``overlap``, the reply must carry it, the status
  needs the overlap block, the receipts check the overlap selection both ways, the record keeps C0 'on'; alpha-next's names are
  restored. (The real control plane forwards ``overlap`` only with FLASH_S5_STAGE_OVERLAP=1: native_overlap_session, checked in
  Forwarding below by reading the session's own action.)
* Grade: alpha-next's AlphaEmit (fused, the size's layer commands split over the two C0 stages, a release per stage, the stage-local
  line, memory and memory-peak receipts) with C1Emit's overlap arms: stage 0 opens chunk n+1 and runs its stage-0 commands while
  stage 1 runs chunk n's, one engine flush per commit holding the successor's early rows, and one ``flash-s5 stage-overlap`` line
  per chunk naming its sequences. The 12 core arms PASS; each C1 rule has a mutant it catches.
* InFlight: the rule on fused and grouped C0-local arms (g2 and g4, C1 off) passes; a stage command submitted before the previous
  one's apply on the same stage FAILs ENGINE; commands of the two stages interleaving do not.
* Numerics: alpha-next's envelope with the C1-off arms as controls; one control left in a band stays UNPROVEN (no one-witness
  DIFFERS: the 51ba618c fix is not needed here).
* ArmLoop: the derived body under real bash with the draft's own functions over stub arm and check clients and the real grade.
* Derivation / NoClock / Commands / C1Parity: the derived texts against alpha-next's, every python3 call parses with its module's
  real parser, no absolute time, C1's restated pieces equal to e3_alpha_c1_box's source.

Fixture leaves: alpha_next_c1_queue_fixture_leaves.json.
"""
import argparse
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import alpha_next_c1_arms as T  # noqa: E402
import alpha_next_c1_gate as gate  # noqa: E402
import alpha_next_c1_queue as queue  # noqa: E402
import alpha_next_c1_report as report_tool  # noqa: E402
import alpha_next_queue as an_queue  # noqa: E402
import e3_alpha_next_box as an  # noqa: E402
import e3_alpha_next_c1_box as box  # noqa: E402
import test_alpha_next_queue as tan  # noqa: E402
import test_alpha_perf_queue as tap  # noqa: E402

progress, ap, sa, sb = box.progress, box.ap, box.sa, box.b
CORE = T.labels('core')
TAIL = T.labels('tail')
ALL_LABELS = T.labels()
# Module-level receipts, the shape of test_alpha_next_queue's (the pack's probe derive reads both): no slot (no absolute-time
# window in the draft), and one row per python call of the derived texts parsed by its real parser (QueueArgparse).
SLOT_RECEIPTS = []
ARGPARSE_RECEIPTS = []
DRAFT = 'harness/runs/q18a_alpha_next_c1.draft.sh'
SETTINGS = 'harness/runs/q18a_alpha_next_c1_settings.sh'
DECLARE = 'harness/runs/q18a_alpha_next_c1_declare.sh'
LAYER_COMMANDS = sb.LAYER_COMMANDS_PER_CHUNK
GIB, MIB = 2 ** 30, 2 ** 20


# --- the table and the budget -------------------------------------------------------------------------------

class Table(unittest.TestCase):
    def test_the_declared_fourteen_arms_in_order(self):
        self.assertEqual(ALL_LABELS, [
            'c1off.g2-solo-0', 'c1on.g2-solo-0', 'c1on.g2-live-0', 'c1off.g2-live-0',
            'c1off.g2-live-1', 'c1on.g2-live-1', 'c1on.g2-solo-1', 'c1off.g2-solo-1',
            'c1off.g2-solo-2', 'c1on.g2-solo-2', 'c1on.g2-live-2', 'c1off.g2-live-2',
            'c1on.g4-solo-0', 'c1on.g4-live-0'])
        rows = T.arms()
        self.assertEqual([r['part'] for r in rows], ['core'] * 12 + ['tail'] * 2)
        self.assertEqual([r['band'] for r in rows], [1] * 8 + [2] * 4 + [1] * 2)
        self.assertEqual({(r['stage_local'], r['c0']) for r in rows}, {(True, 'on')})
        self.assertEqual([r['overlap'] for r in rows], [r['c1'] == 'on' for r in rows])
        self.assertEqual(T.GRADED, ('core',))
        self.assertEqual((T.STANDING_DECODERS, T.STANDING_GREEDY, T.STANDING_SAMPLED),
                         (8, (0, 1), (2, 3, 4, 5, 6, 7)))

    def test_every_graded_on_arm_has_its_matched_off_arm_and_the_tail_none(self):
        for row in T.arms('core'):
            if row['c1'] == 'on':
                off = T.by_label()[T.matched_off(row)]
                self.assertEqual((off['c1'], off['size'], off['cohort'], off['index'], off['band']),
                                 ('off', row['size'], row['cohort'], row['index'], row['band']))
        self.assertEqual([T.matched_off(r) for r in T.arms('tail')], [None, None])
        # ABBA: the second band 1 round mirrors the first.
        band1 = [(r['c1'], r['cohort']) for r in T.arms('core') if r['band'] == 1]
        self.assertEqual(band1[4:], band1[:4][::-1])

    def test_the_table_refuses_a_malformed_order(self):
        for name, value in (('ARM_TABLE', T.ARM_TABLE[-1:] + T.ARM_TABLE[:-1]),
                            ('ARM_TABLE', T.ARM_TABLE + (('tail', 'off', 'live', 0),)),
                            ('ARM_TABLE', tuple(r for r in T.ARM_TABLE if r[1:] != ('off', 'live', 2)))):
            with self.subTest(name=name), mock.patch.object(T, name, value), self.assertRaises(AssertionError):
                T.check()
        self.assertTrue(T.check())

    def test_the_view_gives_alpha_next_the_c1_axis(self):
        view = box.table_view()
        self.assertEqual([r['c0'] for r in view.arms()], [r['c1'] for r in T.arms()])
        self.assertEqual(view.matched_off(view.by_label()['c1on.g2-live-2']), 'c1off.g2-live-2')
        self.assertEqual(view.parse_selection('c1on.g4'), ('on', 4))
        arm = box.table_view('off')
        self.assertEqual((arm.label('on', 2, 'solo', 1), arm.selection('on', 2), arm.C0), ('c1off.g2-solo-1', 'c1off.g2', {'on': True}))


class Budget(unittest.TestCase):
    def test_the_box_budget_and_the_entry_limits_are_relative_and_alpha_nexts(self):
        import alpha_next_gate as an_gate
        self.assertEqual((gate.BOX_BUDGET_S, gate.CLEANUP_S, gate.ACTIVE_S), (9000, 900, 8100))
        self.assertEqual((gate.LAST_ARM_ENTRY_S, gate.LAST_TAIL_ENTRY_S), (an_gate.LAST_ARM_ENTRY_S, an_gate.LAST_TAIL_ENTRY_S))
        self.assertEqual((gate.LAST_ARM_ENTRY_S, gate.LAST_TAIL_ENTRY_S), (3420, 4470))
        self.assertEqual((gate.ARMS, gate.CORE_ARMS, gate.TAIL_ARMS), (14, 12, 2))
        self.assertEqual(gate.PHASES['alpha_next_c1_arms'], 14 * (1500 + 30))
        self.assertTrue(gate.arm_fits(3420) and not gate.arm_fits(3421) and gate.arm_fits(4470, 'tail'))

    def test_fourteen_arms_fit_with_margin_at_the_lead_walls(self):
        e = gate.EXPECTED_RUNTIME_S['nominal']
        self.assertEqual((e['arms_entering'], e['core_arms_entering'], e['tail_arms_entering']), (14, 12, 2))
        # first arm +480 s; band 1: 246 (cold) + 75 + 147 * 4 + 75 * 2; band 2: 128 * 2 + 273 * 2; core FULL and grade 150.
        self.assertEqual(e['last_core_arm_enters_s'], 480 + 1059 + 128 + 128 + 273)
        self.assertEqual(e['last_arm_enters_s'], 480 + 1059 + 802 + 150 + 75)
        self.assertEqual((e['last_arm_ends_s'], e['total_s']), (2713, 3313))
        self.assertGreaterEqual(e['min_margin_s'], 1300)       # every arm enters >= 1,300 s before its limit
        # At one and a half times the walls every arm still enters.
        slower = gate._estimate(lambda row, first: 1.5 * gate.arm_wall(row, first))
        self.assertEqual(slower['arms_entering'], 14)

    def test_the_tool_prints_the_table_with_its_limits(self):
        text = queue.table_text()
        for name in ALL_LABELS:
            self.assertIn(name, text)
        self.assertIn('LAST_TAIL_ENTRY_S 4470 s', text)
        self.assertIn('LAST_ARM_ENTRY_S 3420 s', text)


# --- the arm's control -----------------------------------------------------------------------------------

class FakeClient:
    """The control client's surface the release uses; the stage reply echoes ``overlap`` unless told otherwise."""

    def __init__(self, overlap_reply=True):
        self.calls, self.overlap_reply = [], overlap_reply

    def pause(self):
        self.calls.append(('pause', {}))
        return {'route': 'full', 'route_generation': 3}

    def resume(self):
        self.calls.append(('resume', {}))
        return {'resumed': True}

    def call(self, action, **fields):
        self.calls.append((action, fields))
        reply = {'native_progress': dict(mode=fields.get('mode'), installed='burst', route='full'),
                 'native_qsa_fused': dict(fused=True, route='full'),
                 'native_layer_group': dict(size=fields.get('size'), route='full'),
                 'release_auto_arm': dict(released=True)}.get(action)
        if action == 'native_stage_local':
            reply = dict(local=fields['local'], previous=True, route='full')
            if self.overlap_reply:
                reply.update(overlap=fields.get('overlap'), previous_overlap=False)
        return reply


def status(local=True, overlap=True, counters=None, block=True):
    stage = dict(selected=local, counters=dict.fromkeys(an.STAGE_COUNTERS, 0), ring_reserve=2, poll_reserve=1, ring={})
    if block:
        stage['overlap'] = dict(selected=overlap, counters=dict(dict.fromkeys(box.OVERLAP_COUNTERS, 0), **(counters or {})))
    return {'native_stage_local': stage}


class ArmControl(unittest.TestCase):
    def release(self, c1, client):
        with box._bound(an, box.arm_names(c1)):
            record = {}
            try:
                an.release_alpha_next(2, True)(client, record, 'burst')
                return record, None
            except ValueError as exc:
                return record, str(exc)

    def test_each_selection_sends_overlap_and_proves_it(self):
        for c1 in ('on', 'off'):
            client = FakeClient()
            record, error = self.release(c1, client)
            self.assertIsNone(error)
            sent = dict(client.calls)['native_stage_local']
            self.assertEqual(sent, dict(local=True, overlap=c1 == 'on'))
            self.assertEqual((record['stage_switch']['local'], record['stage_switch']['overlap']), (True, c1 == 'on'))
            self.assertEqual([a for a, _ in client.calls], ['pause', 'native_progress', 'native_qsa_fused', 'native_layer_group',
                                                            'native_stage_local', 'release_auto_arm', 'resume'])

    def test_a_reply_without_overlap_is_refused_and_still_resumes(self):
        # FLASH_S5_STAGE_OVERLAP unset on the box: the session drops the field and the reply has no overlap (C0's command).
        client = FakeClient(overlap_reply=False)
        record, error = self.release('on', client)
        self.assertEqual(error, 'native-stage-overlap-selection-unproved')
        self.assertEqual(client.calls[-1][0], 'resume')

    def test_the_status_needs_the_overlap_block_and_the_receipts_check_it(self):
        with self.assertRaises(ValueError) as caught:
            box.stage_view(status(block=False))
        self.assertEqual(str(caught.exception), 'native-stage-overlap-not-installed')
        view = box.stage_view(status(counters={'early_opens': 3}))
        self.assertEqual((view['overlap_selected'], view['overlap_early_opens'], view['selected']), (True, 3, True))
        before, after = box.stage_view(status()), box.stage_view(status(counters={'early_opens': 4, 'successor_frames': 4}))
        delta = box.stage_delta(after, before)
        self.assertEqual((delta['overlap_early_opens'], delta['overlap_successor_frames'], delta['chunks']), (4, 4, 0))
        record = dict(control_history=list(an.CONTROL_SEQUENCE), progress=[dict(label='before-fresh', native_stage_local=before),
                                                                       dict(label='after-fresh', native_stage_local=after)])
        self.assertEqual(box.stage_reasons_c1(True)(record, True), [])
        self.assertEqual(box.stage_reasons_c1(False)(dict(record), True), ['stage-overlap-selection-not-this-arm'])

    def test_the_cli_takes_c1_on_off_and_refuses_others_and_alpha_next_names_are_restored(self):
        names = ('T', 'release_alpha_next', 'stage_reasons', 'stage_view', 'stage_delta')
        before = [getattr(an, k) for k in names]
        argv = ['arm', '--url', 'http://127.0.0.1:9', '--model', 'm', '--run-id', 'r', '--epoch', '1', '--c1', 'on', '--size', '2',
                '--cohort', 'live', '--index', '0', '--workload', '/nonexistent.json', '--control-timeout-s', '1']
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'arm.json'
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                rc = box.main(argv + ['--out', str(out)])
            record = json.loads(out.read_text())
            self.assertEqual((rc, record['label'], record['selection'], record['c0'], record['stage_local'], record['part']),
                             (2, 'c1on.g2-live-0', 'c1on.g2', 'on', True, 'core'))
            self.assertEqual(printed.getvalue().split()[:3], ['ALPHA_NEXT_C1_ARM', 'c1on.g2-live-0', 'INCOMPLETE'])
            out2 = Path(tmp) / 'arm2.json'
            with contextlib.redirect_stdout(io.StringIO()):
                box.main(argv[:9] + ['--c1', 'off', '--size', '4'] + argv[13:] + ['--out', str(out2)])
            self.assertEqual(json.loads(out2.read_text())['error']['error_message'], 'arm-not-in-the-declared-table')
            for flag, value in (('--c1', 'maybe'), ('--size', '3'), ('--index', '3')):
                bad = list(argv)
                bad[bad.index(flag) + 1] = value
                with self.subTest(flag=flag), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    box.main(bad + ['--out', str(Path(tmp) / 'x.json')])
        self.assertEqual([getattr(an, k) for k in names], before)
        self.assertIs(an.T, tan.T)


class Forwarding(unittest.TestCase):
    def test_the_session_forwards_overlap_only_with_the_c1_switch(self):
        import native_overlap_session as nos
        source = Path(nos.__file__).read_text()
        body = source[source.index('async def _action_native_stage_local'):]
        body = body[:body.index('\n    async def ', 10)]
        self.assertIn("'overlap' in command and os.environ.get('FLASH_S5_STAGE_OVERLAP', '0') == '1'", body)
        self.assertIn('native_stage_overlap(command.get(\'local\'), command[\'overlap\'])', body)


# --- C1's two chunks in flight on alpha-next's emitters ----------------------------------------------------

def ranges(values):
    out, run = [], None
    for value in sorted(values):
        if run is not None and value == run[1] + 1:
            run[1] = value
            continue
        if run is not None:
            out.append(run)
        run = [value, value]
    if run is not None:
        out.append(run)
    return ','.join(str(a) if a == b else '%d-%d' % (a, b) for a, b in out) or '-'


class C1Emit(tan.AlphaEmit):
    """AlphaEmit plus an overlap arm (stage_overlap's order on alpha-next's commands and lines).

    Per chunk n: stage 1 opens n only after it released n-1; while stage 1 runs n's layer commands, releases n and sends the
    receipt, stage 0 opens n+1 (early: before stage 1 released n), runs n+1's layer commands and releases n+1, interleaved command
    by command, each stage one command at a time (each submitted after its previous one applied). At n's commit the stage-local
    line, the stage-overlap line naming n's sequences and the successor, then one engine flush holding every row recorded since the
    previous commit (n's own and n+1's early rows), as OverlapSegmentDriver._reset carries them. Memory receipts in the alpha-next
    engine's format (whole, then the PLAN 2 identity)."""

    def receipt_memory(self, c1, band, rank, txn, start, end, ooms=0):
        allocated, reserved, free = tan.GPU[('on' if c1 == 'on' else 'off', band)]
        for point, retries, shift in (('open', 5 + txn, 0), ('released', 6 + txn, 1)):
            self.log(self.now, rank, 'flash-s5 memory point=%s rank=%d txn_seq=%d chunk=%d-%d allocated=%d reserved=%d '
                                     'inactive_split=%d alloc_retries=%d ooms=%d free=%d total=25296044032 sync_all_streams=0 '
                                     'device_allocs=%d device_frees=%d pid=%d seq=%d sha=%016x' % (
                                         point, rank, txn, start, end, (allocated + shift) * MIB, reserved * MIB, 2 ** 26,
                                         retries, ooms, (free - shift) * MIB, 10 + txn, 9 + txn, 201 + rank, 2 * txn, txn))
        self.log(self.now, rank, 'flash-s5 memory-peak rank=%d txn_seq=%d chunk=%d-%d peak_allocated=%d peak_reserved=%d '
                                 'allocated_at_open=%d stage_timer=0' % (rank, txn, start, end,
                                                                         (allocated + 300 + (20 if c1 == 'on' else 0)) * MIB,
                                                                         (reserved + 10) * MIB, allocated * MIB))

    def overlap_arm(self, cuts, *, size, lives, c1, band, early=True, overlap_lines=True, breach=None, skip_flush=None):
        n = len(cuts) - 1
        want = LAYER_COMMANDS[size]
        half = want // 2
        ranks = {0: (0, 1), 1: (2, 3)}
        txns = [self.txn + 1 + c for c in range(n)]
        self.txn += n
        owned = {t: [] for t in txns}
        first = {}

        def command(kind, stage, t, origin='tick', direct=0):
            held = self.now
            if breach and (t, stage, kind) == breach[:3] and breach[3] == len(owned[t]):
                self.now -= 1_000_000         # this one command submitted before the previous command of its stage applied
            first.setdefault(t, self.now)
            self.stage_command(kind, 'layer' if kind == 'segment' else None, stage, ranks[0] if stage is None else ranks[stage],
                               origin, direct)
            self.now = max(self.now, held)
            owned[t].append(self.seq)
            self.tick(self.CADENCE)

        def stage0(c):
            t = txns[c]
            return ([('open', 0, t)] + [('segment', 0, t)] * half + [('release', 0, t)])

        def stage1(c):
            t = txns[c]
            return [('open', 1, t)] + [('segment', 1, t)] * (want - half) + [('release', 1, t), ('receipt', None, t)]

        for kind, stage, t in stage0(0):
            command(kind, stage, t, 'wait' if lives[0] and kind == 'segment' else 'tick', int(kind == 'segment'))
        for c in range(n):
            for rank in range(4):
                self.receipt_memory(c1, band, rank, txns[c], cuts[c], cuts[c + 1])
            a = stage1(c)
            b_ = stage0(c + 1) if c + 1 < n else []
            order = ([x for pair in zip(a, b_) for x in pair] + a[len(b_):] + b_[len(a):]) if early else a + b_
            for i, (kind, stage, t) in enumerate(order):
                command(kind, stage, t, 'wait' if lives[c] and kind == 'segment' and i <= 2 else 'tick', int(kind == 'segment'))
                if i == 1 and lives[c]:
                    self.stock('execute_model')
                    self.stock('sample_tokens')
                    self.engine.record('w', self.seq, 'sample_tokens', self.now, self.now + 300_000, 2, 0)
            t = txns[c]
            self.log(self.now, 'engine', 'flash-s5 stage-local txn=%d chunk=%d-%d commands=%d,%d both_stages_ns=2000000 '
                                         'first_ns=%d,%d last_ns=%d,%d local_release_ns=%d,%d headroom_refused=1 '
                                         'headroom_blocked_ns=0 pid=101 seq=%d sha=%016x' % (
                                             t, cuts[c], cuts[c + 1], half + 2, want - half + 3, first[t], first[t], self.now,
                                             self.now, self.now - 5000, self.now, t, t))
            if overlap_lines:
                self.log(self.tick(10), 'engine', 'flash-s5 stage-overlap txn=%d seqs=%s successor=%s' % (
                    t, ranges(owned[t]), txns[c + 1] if c + 1 < n else 'none'))
            texts = self.engine.lines(txn=t, request_generation=1, chunk_start=cuts[c], chunk_end=cuts[c + 1],
                                      decode_live=lives[c], mode='burst', open_rows=0)
            if skip_flush == c:
                continue                     # the whole engine flush lost from the log
            for text in texts:
                self.log(self.tick(100), 'engine', text)
        return n


def c1_counters(chunks, on, **changes):
    out = dict.fromkeys(box.OVERLAP_COUNTERS, 0)
    if on:
        out.update(successor_frames=chunks - 1, overlapped_commits=chunks - 1, early_opens=chunks - 1, lane_checks=chunks - 1)
    out.update(changes)
    return {'overlap_' + k: v for k, v in out.items()}


def synthetic_run(rows=None, *, outs=None, extra=None, arm_kwargs=None, counters=None, local_off_alpha=False):
    """The declared arms in order on one C1Emit: C1-on arms with two chunks in flight, C1-off arms alpha-next's C0-on chunk."""
    emit, arms = C1Emit(), []
    emit.spans = {}
    for row in (T.arms(T.GRADED) if rows is None else rows):
        name, band, size = row['label'], row['band'], row['size']
        cuts = tap.band_cuts(band)
        first_line = len(emit.lines)
        start = emit.tick(5_000_000)
        emit.tick(500_000)
        chunks = len(cuts) - 1
        lives = [8] * chunks if row['cohort'] == 'live' else [8] + [0] * (chunks - 1)
        if row['c1'] == 'on':
            emit.overlap_arm(cuts, size=size, lives=lives, c1='on', band=band, **(arm_kwargs or {}).get(name, {}))
        else:
            for c, live in enumerate(lives):
                for rank in range(4):
                    emit.receipt_memory('off', band, rank, emit.txn + 1, cuts[c], cuts[c + 1])
                emit.alpha_chunk(cuts[c], cuts[c + 1], size=size, live=live, local=True)
        for text in (extra or {}).get(name, ()):
            emit.log(emit.now, 1, text)
        end = emit.tick(1_000_000)
        emit.spans[name] = (first_line, len(emit.lines))
        streams = [40.0, 41.0, 42.0, 42.0, 43.0, 43.0, 44.0, 45.0]
        window = {'aggregate_tokens_per_s': 340.0, 'stream_rates_tokens_per_s': streams, 'gap_ms_p99': 60.0, 'gap_ms_max': 100.0}
        client = {'prefill_tokens_per_s': 900.0, 'all_open_share_of_span': 1.0 if row['cohort'] == 'live' else 0.1}
        if row['cohort'] == 'live':
            client['decode_window'] = window
        layer = LAYER_COMMANDS[size] * chunks
        fused = dict(selections=0, chunks=chunks, grants=12 * chunks, fused_commands=12 * chunks, run_ahead_polls=0, ledger_checks=50 * chunks)
        groups = dict(selections=0, chunks=chunks, plan_commands=layer, group_commands=layer, group_layers=sb.LAYERS_PER_CHUNK * chunks)
        stage = dict(selections=0, chunks=chunks, headroom_refused=chunks, local_releases=2 * chunks, both_stages_ns=2000000 * chunks)
        stage.update(c1_counters(chunks, row['c1'] == 'on', **(counters or {}).get(name, {})))
        arms.append(dict(label=name, selection=row['selection'], size=size, stage_local=True, c0='on', part=row['part'],
                         mode='burst', cohort=row['cohort'], index=row['index'], band=band, complete=True, reasons=[],
                         wall_start_s=emit.wall(start), wall_end_s=emit.wall(end),
                         progress_delta={'chunks': chunks, 'commands': layer + 5 * chunks}, fused_delta=fused,
                         group_delta=groups, group_max_after=size, stage_delta=stage,
                         ring_delta=dict(blocked_ns=0, episodes=0, blocked_max_ns_after=0, blocked_open_after=False,
                                         limiting={'2': 0}, limiting_sole={'2': 0}),
                         stop_delta=dict(partial_replays=0, bound_exhausted=0),
                         selection_switch=dict(fused=True, route='full'), group_switch=dict(size=size, route='full'),
                         stage_switch=dict(local=True, overlap=row['overlap'], route='full'),
                         outputs=(outs or {}).get(name) or tan.outputs(row),
                         fresh={'prompt_tokens_observed': cuts[-1] + 1, 'ttft_s': float(chunks)},
                         decode_only={'aggregate_tokens_per_s': 400.0, 'stream_rates_tokens_per_s': [50.0] * 8},
                         client_basis=client, append={'ttft_s': 2.0}, metrics={}))
    return emit, arms


class Grader:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()

    def __call__(self, emit, arms, ingress=None, watches=None, lines=None, part='core', checks=None):
        q = Path(tempfile.mkdtemp(dir=self.tmp.name))
        tap.write_streams(q, emit.lines if lines is None else lines)
        ingress = {a['label']: tap.PASS for a in arms} if ingress is None else ingress
        watches = {a['label']: tap.ALIVE for a in arms} if watches is None else watches
        logs = [str(q / 'engine-stream.log'), str(q / 'engine-stream.err')]
        return box.grade(arms, ingress, tap.CHECKS if checks is None else checks, logs, watches, part=part)


_RUNS = {}


def run_of(key, **kwargs):
    """One synthetic run per declared shape, built once (the emitters are deterministic)."""
    if key not in _RUNS:
        _RUNS[key] = synthetic_run(**kwargs)
    emit, arms = _RUNS[key]
    return emit, copy.deepcopy(arms)          # tests never change emit.lines in place (they grade edited copies)


class GradeBase(unittest.TestCase):
    def setUp(self):
        self.grade = Grader()
        self.addCleanup(self.grade.tmp.cleanup)

    @staticmethod
    def edited(arms, name, **fields):
        return [dict(a, **fields) if a['label'] == name else a for a in arms]

    @staticmethod
    def indexes(emit, name, mark):
        """Line indexes holding ``mark`` inside the arm's own lines (C1Emit.spans)."""
        lo, hi = emit.spans[name]
        return [i for i in range(lo, hi) if mark in emit.lines[i][1]]


def whole_memory(rank, txn, start, end, ooms):
    return ('flash-s5 memory point=released rank=%d txn_seq=%d chunk=%d-%d allocated=%d reserved=%d inactive_split=0 '
            'alloc_retries=0 ooms=%d free=%d total=25296044032 sync_all_streams=0 device_allocs=1 device_frees=1 pid=%d seq=900 '
            'sha=00000000000000ff' % (rank, txn, start, end, 21000 * MIB, 22000 * MIB, ooms, 700 * MIB, 201 + rank))


class Grade(GradeBase):
    def test_the_core_arms_pass_with_all_sections(self):
        emit, arms = run_of('core')
        result = self.grade(emit, arms)
        self.assertEqual((result['schema'], result['verdict'], result['missing'], result['failures'], result['classification']),
                         ('alpha-next-c1-grade/1', 'PASS', [], [], None), result['missing'][:5])
        self.assertEqual((result['part'], result['is_queue_verdict'], result['required_arms']), ('core', True, CORE))
        self.assertEqual({v['verdict'] for v in result['validity'].values()}, {'PASS'})
        on, off = result['validity']['c1on.g2-live-2'], result['validity']['c1off.g2-live-2']
        self.assertEqual((on['engagement']['c1']['engaged'], on['engagement']['c1']['overlap'], on['engagement']['c0']['engaged'],
                          on['engagement']['group_engaged'], on['engagement']['c1']['overlap_lines'],
                          on['engagement']['c1']['regrouped_chunks_checked'], on['engagement']['c1']['counters']['early_opens']),
                         (True, True, True, True, 5, 5, 4))
        self.assertGreater(on['regroup']['moved_rows'], 0)
        self.assertEqual((off['engagement']['c1']['engaged'], off['engagement']['c1']['overlap'], off['engagement']['c1']['overlap_lines'],
                          off['regroup']), (True, False, 0, None))
        for name in CORE:     # one command in flight per stage, on fused and grouped g2 arms, C1 on and off: counted, no breach
            flight = result['validity'][name]['in_flight']
            self.assertEqual([len(flight[s]['breaches']) for s in '01'], [0, 0], name)
            self.assertTrue(all(flight[s]['commands'] > 0 and flight[s]['unbounded'] == 0 for s in '01'), name)
        row = result['table']['c1on.g2-live-2']
        self.assertEqual((row['c0'], row['c1'], row['overlap'], row['stage_local'], row['size'], row['band']), ('on', 'on', True, True, 2, 2))
        self.assertEqual(result['table']['c1off.g2-solo-0']['c1'], 'off')
        self.assertEqual(sorted(result['summary']['band1']['over_c0_off']), ['c1on.g2-live', 'c1on.g2-solo'])
        self.assertEqual(result['memory_read']['c1on.g2-live-0']['against'], 'c1off.g2-live-0')
        self.assertEqual(sorted(result['memory_c0_on_vs_off']), sorted(x for x in CORE if x.startswith('c1on')))
        self.assertEqual((result['numerics']['verdict'], result['numerics']['bands']['1']['controls'],
                          result['numerics']['bands']['2']['controls']),
                         ('WITHIN', ['c1off.g2-solo-0', 'c1off.g2-live-0', 'c1off.g2-live-1', 'c1off.g2-solo-1'],
                          ['c1off.g2-solo-2', 'c1off.g2-live-2']))
        self.assertEqual((result['numerics']['is_gate'], result['receipt_integrity_is_gate'], result['oom_is_gate'],
                          result['in_flight_is_gate']), (False, False, True, True))
        self.assertIn('C1 on against C1 off', result['axis'] + ' C1 on against C1 off')
        self.assertEqual(sorted(result['declared_limits']), ['budget', 'in_flight', 'memory_peak', 'oom'])
        text = box.report(result)
        self.assertEqual(re.findall(r'^## (.+)$', text, flags=re.M), ['Mechanism', 'Performance', 'Release readiness'])
        self.assertTrue(text.startswith('# q18a-alpha-next-c1 report'))
        for head in ('### C1 engagement and one command in flight per stage',
                     '### Memory-peak per chunk, C1 on against the matched C1-off arm (a read)',
                     '### Memory margin per stage, C1 on against the matched C1-off arm (a read)'):
            self.assertIn(head, text)
        self.assertNotIn('C0-on', text)

    def test_memory_peak_per_chunk_and_stage_one_margin_against_the_matched_off_arm(self):
        emit, arms = run_of('all', rows=T.arms())
        result = self.grade(emit, arms, part='all')
        self.assertEqual((result['schema'], result['verdict']), ('alpha-next-c1-read/1', 'PASS'))
        peaks = result['memory_peak_vs_c1_off']['arms']
        self.assertEqual(sorted(peaks), sorted(x for x in ALL_LABELS if x.startswith('c1on')))
        band2 = peaks['c1on.g2-live-2']
        self.assertEqual((band2['against'], band2['chunks'], band2['unmatched_chunks']), ('c1off.g2-live-2', 5, []))
        self.assertEqual({x['allocated_delta_mib'] for x in band2['per_chunk']}, {70.0})     # GPU on/off +50, C1-on peak +20 MiB
        self.assertGreater(band2['past_arena']['chunks'], 0)
        self.assertEqual(band2['past_arena']['chunks'] + band2['below_arena']['chunks'], 5)
        self.assertEqual(peaks['c1on.g2-live-0']['past_arena']['chunks'], 0)
        self.assertEqual((peaks['c1on.g4-live-0']['against'], peaks['c1on.g4-live-0']['reported_only']), (None, True))
        margin = result['memory_stage_vs_c1_off']['arms']['c1on.g2-live-2']
        self.assertEqual(result['memory_stage_vs_c1_off']['stage_ranks'], {'0': [0, 1], '1': [2, 3]})
        self.assertEqual((margin['stages']['1']['min_free_below_off_mib'], margin['stages']['1']['peak_allocated_over_off_mib']),
                         (10.0, 70.0))
        self.assertTrue(result['memory_stage_vs_c1_off']['arms']['c1on.g4-solo-0']['reported_only'])
        # A read: a worse stage 1 never changes a verdict.
        self.assertEqual(result['memory_read']['c1on.g2-live-2']['verdict'], 'NOT_MET')

    def test_the_seams_are_this_call_only(self):
        def names():
            return [an.T, an.engagement, sa.SIGNATURES, sa.arm_verdict, box.torn_lines.settle_arm, progress.assemble,
                    progress.chunk_breakdown, ap.band_of_index, sb.engagement]
        before = names()
        emit, arms = run_of('core')
        self.grade(emit, arms[:2])
        self.assertEqual(names(), before)
        with mock.patch.object(an, 'grade', side_effect=RuntimeError('inside')), self.assertRaises(RuntimeError):
            self.grade(emit, arms)
        self.assertEqual(names(), before)
        self.assertEqual(box._RUN, {})

    def test_without_the_regroup_a_c1_on_arm_is_not_engaged(self):
        emit, arms = run_of('core')
        with mock.patch.object(box, 'regrouping', lambda original: original):
            result = self.grade(emit, arms)
        self.assertEqual(result['verdict'], 'INC')
        v = result['validity']['c1on.g2-live-0']
        self.assertIn('regrouped-chunk-without-two-opens-and-two-releases', v['reasons'])
        self.assertEqual(result['validity']['c1off.g2-live-0']['verdict'], 'PASS')

    def test_a_declared_arm_not_run_for_the_budget_is_inc_never_an_engine_result(self):
        emit, arms = run_of('core')
        kept = [a for a in arms if a['label'] != CORE[-1]]
        result = self.grade(emit, kept, watches={a['label']: tap.ALIVE for a in kept})
        self.assertEqual((result['verdict'], result['classification']), ('INC', None))
        self.assertEqual(result['validity'][CORE[-1]]['reasons'], ['no-arm-record'])
        self.assertIn('declared-arm-missing:' + CORE[-1], result['missing'])


class Engagement(GradeBase):
    def test_no_early_open_or_no_successor_is_inc(self):
        for change, reason in (({'early_opens': 0}, 'overlap-stage-zero-never-opened-a-successor-early'),
                               ({'successor_frames': 0}, 'overlap-ran-no-second-chunk'),
                               ({'overlapped_commits': 0}, 'overlap-ran-no-second-chunk')):
            emit, arms = run_of('core')
            arm = next(a for a in arms if a['label'] == 'c1on.g2-solo-1')
            arm['stage_delta'].update({'overlap_' + k: v for k, v in change.items()})
            with self.subTest(change=change):
                result = self.grade(emit, arms)
                v = result['validity']['c1on.g2-solo-1']
                self.assertEqual((result['verdict'], v['verdict']), ('INC', 'INC'))
                self.assertIn(reason, v['reasons'])

    def test_counters_missing_is_inc(self):
        emit, arms = run_of('core')
        arm = next(a for a in arms if a['label'] == 'c1on.g2-live-1')
        del arm['stage_delta']['overlap_lane_checks']
        result = self.grade(emit, arms)
        self.assertIn('overlap-counters-missing', result['validity']['c1on.g2-live-1']['reasons'])

    def test_a_c1_off_arm_moving_an_overlap_counter_or_logging_its_line_is_inc(self):
        emit, arms = run_of('core')
        next(a for a in arms if a['label'] == 'c1off.g2-live-1')['stage_delta']['overlap_early_opens'] = 1
        result = self.grade(emit, arms)
        self.assertIn('c1-off-arm-moved-an-overlap-counter', result['validity']['c1off.g2-live-1']['reasons'])
        emit, arms = run_of('core')
        i = self.indexes(emit, 'c1off.g2-solo-2', 'flash-s5 stage-local ')[0]
        lines = list(emit.lines)
        lines.insert(i + 1, [lines[i][0], lines[i][1].split('flash-s5 ')[0] + 'flash-s5 stage-overlap txn=1 seqs=- successor=none\n'])
        result = self.grade(emit, arms, lines=lines)
        self.assertIn('c1-off-arm-logged-an-overlap-line', result['validity']['c1off.g2-solo-2']['reasons'])

    def test_a_missing_overlap_line_is_inc(self):
        emit, arms = run_of('core')
        i = self.indexes(emit, 'c1on.g2-live-1', box.OVERLAP_MARK)[2]
        lines = emit.lines[:i] + emit.lines[i + 1:]
        result = self.grade(emit, arms, lines=lines)
        v = result['validity']['c1on.g2-live-1']
        self.assertEqual(v['verdict'], 'INC')
        self.assertIn('overlap-lines-differ-from-the-auto-owned-chunks', v['reasons'])

    def test_a_sequence_named_by_two_chunks_is_inc(self):
        emit, arms = run_of('core')
        i, j = self.indexes(emit, 'c1on.g2-solo-0', box.OVERLAP_MARK)[:2]
        seq = box.seqs_of(box.OVERLAP_LINE.search(emit.lines[i][1]).group(2))[0]
        lines = [list(x) for x in emit.lines]
        lines[j][1] = lines[j][1].replace(' seqs=', ' seqs=%d,' % seq)
        result = self.grade(emit, arms, lines=lines)
        v = result['validity']['c1on.g2-solo-0']
        self.assertEqual(v['verdict'], 'INC')
        self.assertIn('progress:overlap-regroup:seq-%d-named-by-two-chunks' % seq, v['reasons'])


class Tears(GradeBase):
    def test_a_torn_overlap_line_drops_its_chunk_and_the_older_one_within_the_limits(self):
        emit, arms = run_of('core')
        i = self.indexes(emit, 'c1on.g2-live-0', box.OVERLAP_MARK)[2]
        lines = [list(x) for x in emit.lines]
        stream, text = lines[i]
        at = text.index(' successor=')
        lines[i] = [stream, text[:at] + ' (Worker_PP0_TP1_EP1 pid=102) ' + text[at:]]
        result = self.grade(emit, arms, lines=lines)
        v = result['validity']['c1on.g2-live-0']
        self.assertEqual((result['verdict'], v['verdict']), ('PASS', 'PASS'), v['reasons'])
        self.assertEqual((v['tears']['torn_overlap_lines'], v['tears']['lost_records'], len(v['tears']['dropped_chunk_starts'])),
                         (1, 1, 2))
        self.assertEqual(v['engagement']['c1']['lost_overlap_lines'], 1)

    def test_two_torn_overlap_lines_are_above_the_limit(self):
        emit, arms = run_of('core')
        lines = [list(x) for x in emit.lines]
        for i in self.indexes(emit, 'c1on.g2-live-0', box.OVERLAP_MARK)[1:3]:
            lines[i][1] = lines[i][1].replace(' successor=', ' (Worker_PP0_TP1_EP1 pid=102) successor=')
        result = self.grade(emit, arms, lines=lines)
        v = result['validity']['c1on.g2-live-0']
        self.assertEqual(v['verdict'], 'INC')
        self.assertTrue([x for x in v['reasons'] if 'lost-records-above-the-declared-limit:2>1' in x], v['reasons'])

    def test_a_lost_engine_flush_on_a_c1_on_arm_is_inc(self):
        emit, arms = synthetic_run(arm_kwargs={'c1on.g2-solo-2': dict(skip_flush=1)})
        result = self.grade(emit, arms)
        v = result['validity']['c1on.g2-solo-2']
        self.assertEqual((result['verdict'], v['verdict'], v['classification']), ('INC', 'INC', None))
        self.assertTrue({'progress:overlap-arm-engine-flush-lost', 'progress:overlap-rows-of-a-chunk-without-its-flush'}
                        & set(v['reasons']), v['reasons'])
        self.assertNotIn('FAIL', {x['verdict'] for x in result['validity'].values()})


class InFlight(GradeBase):
    def test_a_stage_command_submitted_before_the_previous_one_applied_fails_engine(self):
        first = synthetic_run(rows=T.arms('core')[:5])[0].txn         # transactions before the sixth arm
        txn = first + 2                                               # its second chunk, whose stage-0 commands overlap the first's
        emit, arms = synthetic_run(arm_kwargs={'c1on.g2-live-1': dict(breach=(txn, 0, 'segment', 3))})
        result = self.grade(emit, arms)
        v = result['validity']['c1on.g2-live-1']
        self.assertEqual((result['verdict'], v['verdict'], v['classification']), ('FAIL', 'FAIL', 'ENGINE'))
        self.assertIn('two-commands-in-flight-on-stage-0', v['signatures'])
        self.assertEqual(len(v['in_flight']['0']['breaches']), 1)
        self.assertEqual(v['in_flight']['1']['breaches'], [])

    def test_grouped_c0_local_arms_at_g2_and_g4_pass_the_rule(self):
        for size in (2, 4):
            emit = C1Emit()
            cuts = tap.band_cuts(1)
            for c in range(len(cuts) - 1):
                emit.alpha_chunk(cuts[c], cuts[c + 1], size=size, live=8, local=True)
            with tempfile.TemporaryDirectory() as tmp:
                tap.write_streams(Path(tmp), emit.lines)
                records, _ = progress.read_lines([str(Path(tmp) / 'engine-stream.log'), str(Path(tmp) / 'engine-stream.err')])
            flushes, _ = progress.assemble(records)
            segments = [box._segment(r, r['stage']) for f in flushes if f['side'] == 'engine' for r in f['rows'].get('n', [])
                        if r['kind'] == 'segment']
            got = box.in_flight(segments)
            with self.subTest(size=size):
                self.assertEqual({s: (got[s]['commands'], got[s]['breaches']) for s in got},
                                 {s: ((len(cuts) - 1) * LAYER_COMMANDS[size] // 2, []) for s in ('0', '1')})

    def test_the_rule_on_its_own(self):
        rows = [(0, 10, 20, 1), (1, 12, 30, 2), (0, 20, 25, 3), (1, 29, 40, 4), (0, 26, None, 5), (0, 27, 28, 6)]
        got = box.in_flight(rows)
        self.assertEqual(got['0']['breaches'], [])          # no end: unbounded, never a breach of the next
        self.assertEqual(got['0']['unbounded'], 1)
        self.assertEqual([b['seq'] for b in got['1']['breaches']], [4])


class Oom(GradeBase):
    def test_an_oom_in_an_intact_or_a_torn_receipt_fails_the_arm_as_engine(self):
        intact = whole_memory(2, 3, 0, 1024, 1)
        torn = intact.replace(' alloc_retries=', ' (Worker_PP1_TP0_EP2 pid=203) alloc_retries=')
        for name, text in (('intact', intact), ('torn', torn)):
            emit, arms = synthetic_run(extra={'c1on.g2-live-0': [text]})
            with self.subTest(receipt=name):
                result = self.grade(emit, arms)
                v = result['validity']['c1on.g2-live-0']
                self.assertEqual((result['verdict'], v['verdict'], v['classification']), ('FAIL', 'FAIL', 'ENGINE'))
                self.assertIn('oom', v['signatures'])
                self.assertEqual([(x['rank'], x['ooms'], x['torn']) for x in v['oom_rises']], [(2, 1, name == 'torn')])

    def test_an_oom_outside_every_arm_fails_the_run(self):
        emit, arms = run_of('core')
        lines = [list(x) for x in emit.lines]
        stamp_line = lines[-1][1]
        prefix = stamp_line[:stamp_line.index('flash-s5') if 'flash-s5' in stamp_line else 0]
        emit2 = C1Emit()
        emit2.now = emit.now + 60_000_000_000
        emit2.mono0 = emit.mono0
        emit2.log(emit2.now, 1, whole_memory(1, 99, 0, 1024, 2))
        result = self.grade(emit, arms, lines=lines + emit2.lines)
        self.assertEqual((result['verdict'], result['classification']), ('FAIL', ['ENGINE']))
        self.assertIn('oom-outside-every-arm-span', result['failures'])
        self.assertEqual(len(result['oom_outside_every_arm']), 1)

    def test_the_identity_suffix_reads_as_whole_and_fragments_still_read(self):
        text = whole_memory(3, 1, 0, 1024, 4)
        self.assertIsNotNone(box.MEMORY_WHOLE.search(text))
        self.assertIsNone(box.MEMORY_WHOLE.search(text.replace(' device_frees=1', '')))
        self.assertEqual(box._fragments('a flash-s5 memory rank=1 x ooms=2 flash-s5 memory ooms=3'), [(1, 2), (None, 3)])


class Signatures(GradeBase):
    def test_each_overlap_contract_error_fails_that_arm_as_engine(self):
        for name, text in box.OVERLAP_SIGNATURES[:3] + box.OVERLAP_SIGNATURES[-1:]:
            emit, arms = synthetic_run(extra={'c1on.g2-solo-2': ['ERROR native driver failed: DriverError: ' + text]})
            with self.subTest(signature=name):
                result = self.grade(emit, arms)
                v = result['validity']['c1on.g2-solo-2']
                self.assertEqual((v['verdict'], v['classification']), ('FAIL', 'ENGINE'))
                self.assertIn(name, v['signatures'])


class Numerics(GradeBase):
    def test_the_c1_off_arms_are_the_controls_and_a_far_c1_on_arm_is_outside_a_read(self):
        rows = T.arms('core')
        outs = {r['label']: tan.outputs(r) for r in rows}
        outs['c1on.g2-live-2'] = tan.outputs(T.by_label()['c1on.g2-live-2'], cut=3)
        emit, arms = synthetic_run(outs=outs)
        result = self.grade(emit, arms)
        self.assertEqual((result['verdict'], result['numerics']['verdict']), ('PASS', 'OUTSIDE'))
        self.assertEqual(result['numerics']['outside'], ['c1on.g2-live-2:stream-0', 'c1on.g2-live-2:stream-1'])
        self.assertIn('NUMERICS OUTSIDE', result['numerics_headline'])

    def test_one_control_left_in_band_two_is_unproven_never_differs(self):
        """The 51ba618c case (one serial witness) is UNPROVEN in alpha-next's envelope: it needs two controls per band and two full
        greedy controls per stream before it can read OUTSIDE."""
        rows = T.arms('core')
        outs = {r['label']: tan.outputs(r) for r in rows}
        outs['c1on.g2-live-2'] = tan.outputs(T.by_label()['c1on.g2-live-2'], cut=3)
        outs['c1off.g2-live-2'] = tan.outputs(T.by_label()['c1off.g2-live-2'], prefix=list(range(200)))   # short of 256
        emit, arms = synthetic_run(outs=outs)
        result = self.grade(emit, arms)
        self.assertEqual(result['numerics']['verdict'], 'UNPROVEN')
        self.assertFalse([x for x in result['numerics']['outside'] if x.startswith('c1on.g2-live-2')])
        self.assertIn('band2-stream-0-fewer-than-two-full-controls', result['numerics']['unproven'])
        kept = [a for a in arms if a['label'] != 'c1off.g2-live-2']
        result = self.grade(emit, kept, watches={a['label']: tap.ALIVE for a in kept})
        self.assertIn('band2-fewer-than-two-controls', result['numerics']['unproven'])
        self.assertEqual(result['verdict'], 'INC')          # the missing arm, not the numerics


# --- the arm loop under real bash ---------------------------------------------------------------------------

STUB_BOX = r'''#!/usr/bin/env python3
import json, os, shutil, sys, time
if sys.argv[1] in ('grade', 'read'):
    with open(os.environ['CALLS'], 'a') as f:
        f.write(json.dumps(dict(script='box', action=sys.argv[1])) + '\n')
    sys.path.insert(0, os.environ['REAL_HARNESS'])
    import e3_alpha_next_c1_box as real
    raise SystemExit(real.main(sys.argv[1:]))
a = sys.argv
out = a[a.index('--out') + 1]
label = 'c1%s.g%s-%s-%s' % (a[a.index('--c1') + 1], a[a.index('--size') + 1], a[a.index('--cohort') + 1], a[a.index('--index') + 1])
with open(os.environ['CALLS'], 'a') as f:
    f.write(json.dumps(dict(script='box', action='arm', label=label, workload=os.path.basename(a[a.index('--workload') + 1]))) + '\n')
if label in os.environ.get('TAIL_LABELS', '').split() and not os.path.exists(os.environ['ARMS'] + '/tail-appended'):
    for stream in ('log', 'err'):
        with open(os.environ['Q'] + '/engine-stream.' + stream, 'a') as f, open(os.environ['ARMS'] + '/tail.' + stream) as g:
            f.write(g.read())
    open(os.environ['ARMS'] + '/tail-appended', 'w').close()
if label == os.environ.get('DIE_IN'):
    with open(os.environ['Q'] + '/engine-stream.log', 'a') as f:
        f.write(os.environ['DEATH_LINE'])
    time.sleep(600)
shutil.copy(os.environ['ARMS'] + '/arm-%s.json' % label, out)
raise SystemExit(int(os.environ.get('RC_' + label.replace('.', '_').replace('-', '_'), '0')))
'''


class ArmLoop(unittest.TestCase):
    """queue.body() runs as written; arm and check clients are stubs (alpha-next's leaf arm-loop-clients), the watch, the grade
    and the read real (this queue's)."""

    def setUp(self):
        import shutil
        self.root = Path(tempfile.mkdtemp(prefix='alpha-next-c1-loop-'))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.q, self.h, self.bin, self.arms = (self.root / x for x in ('q', 'h', 'bin', 'arms'))
        for d in (self.q, self.h, self.bin, self.arms):
            d.mkdir()
        for name in ('native43_engine_death.py', 'alpha_next_c1_gate.py', 'alpha_next_c1_arms.py', 'alpha_next_arms.py',
                     'alpha_next_stop_race.py', 'native39_trunk_gate.py', 'native40_box_gate.py', 'native41_box_gate.py'):
            (self.h / name).symlink_to(HERE / name)
        (self.h / 'e3_native39_trunk_checks.py').write_text(tap.STUB_CHECKS)
        (self.h / 'e3_alpha_next_c1_box.py').write_text(STUB_BOX)
        self.tool('nvidia-smi', 'printf "0, 22925\\n1, 22835\\n2, 23037\\n3, 22947\\n"')
        self.tool('docker', 'case "$*" in *State.Running*) echo true;; *) echo 2026-11-01T10:48:09.000000000Z;; esac')
        self.tool('systemctl', 'echo 0')
        (self.q / 'unarmed-stock-prefill.json').write_text(json.dumps(tap.PASS))

    tool = tap.ArmLoop.tool

    def run_loop(self, *, jump_after=None, jump_to='LAST_ARM_ENTRY_S + 10', die_in=None, rcs=None, timeout=420):
        import signal
        emit, arms = run_of('all', rows=T.arms())
        first_tail = min(a['wall_start_s'] for a in arms if a['part'] == 'tail') * 1e9
        cut = next(i for i, (_, text) in enumerate(emit.lines) if (tan.sr.death.wall_ns(text.encode()) or 0) >= first_tail)
        tap.write_streams(self.q, emit.lines[:cut])
        tap.write_streams(self.arms, emit.lines[cut:])
        for stream in ('log', 'err'):
            os.replace(self.arms / ('engine-stream.' + stream), self.arms / ('tail.' + stream))
        for a in arms:
            (self.arms / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
        draft = queue.derive()[DRAFT]
        body = queue.body()
        self.assertIn(body, draft)
        script = ['set -uo pipefail', 'FAILS=0; INCS=0; declare -A PHASE_VERDICT=()',
                  "ENGINE_CRASHED=0; ENGINE_CRASH_SOURCE=''; STOP_RACE=0; CONTAINER=qwen-flash-prod; SERVICE=vllm-flash; N0=0",
                  'Q=%s; H=%s; PROBE_URL=http://127.0.0.1:1; MODEL=m; RUN_ID=r; EPOCH=1; _cid=c; IDLE_TIMEOUT_S=1' % (self.q, self.h),
                  'Q18A_READINESS=/r/ready; S5_OVERLAY=/o; S5_LAUNCHER_NAME=l.sh', 'health_ok() { return 0; }']
        script += [tap.le_fix_function(draft, name) for name in tap.LOOP_FUNCTIONS]
        if jump_after:
            for fn in ('record', 'read_step'):
                script += ['%s_hook() { [[ $1 == %s ]] && SECONDS=$((%s)); return 0; }' % (fn, jump_after, jump_to)]
            body = body.replace('read_step() {\n  local v\n', 'read_step() {\n  read_step_hook "$1"\n  local v\n')
            script += ['eval "$(declare -f record | sed \'1s/record/real_record/\')"',
                       'record() { real_record "$@"; local rc=$?; record_hook "$1"; return $rc; }']
        script += [body, 'echo "END FAILS=$FAILS INCS=$INCS CRASHED=$ENGINE_CRASHED SOURCE=$ENGINE_CRASH_SOURCE"']
        path = self.root / 'loop.sh'
        path.write_text('\n'.join(script) + '\n')
        env = {k: v for k, v in os.environ.items() if k != 'HARNESS_DIR'}
        env.update(PATH='%s:%s' % (self.bin, os.environ['PATH']), Q=str(self.q), CALLS=str(self.root / 'calls.jsonl'),
                   ARMS=str(self.arms), REAL_HARNESS=str(HERE), TAIL_LABELS=' '.join(TAIL), DEATH_LINE=tap.DEATH_LINE,
                   PYTHONDONTWRITEBYTECODE='1')
        for name, rc in (rcs or {}).items():
            env['RC_' + name.replace('.', '_').replace('-', '_')] = str(rc)
        if die_in:
            env['DIE_IN'] = die_in
        proc = subprocess.Popen(['bash', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                                start_new_session=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            raise
        calls = [json.loads(x) for x in (self.root / 'calls.jsonl').read_text().splitlines()]
        steps = {m.group(1): m.group(2) for m in re.finditer(r'^STEP (\S+) (\S+)', out, flags=re.M)}
        return out, err, calls, steps

    def grade_file(self, name='alpha-next-c1-grade.json'):
        return json.loads((self.q / name).read_text())

    def test_every_arm_in_order_the_core_grade_before_the_tail_then_the_read(self):
        out, err, calls, steps = self.run_loop()
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, (out[-1500:], err[-800:]))
        box_calls = [c.get('label') or c['action'] for c in calls if c['script'] == 'box']
        self.assertEqual(box_calls, CORE + ['grade'] + TAIL + ['read'])
        for c in calls:
            if c['script'] == 'box' and c['action'] == 'arm':
                self.assertEqual(c['workload'], ap.BAND_WORKLOAD[T.by_label()[c['label']]['band']] + '.json', c['label'])
        self.assertEqual([c['out'] for c in calls if c['script'] == 'checks' and c['action'] == 'full-idle'],
                         ['entry-%s.json' % x for x in CORE] + ['core-full-idle.json'] + ['entry-%s.json' % x for x in TAIL]
                         + ['end-full-idle.json'])
        for name in CORE:
            self.assertEqual(steps['arm-' + name], 'PASS', name)
        for name in TAIL:
            self.assertEqual(steps['arm-' + name], 'READ-PASS', name)
        self.assertEqual((steps['alpha-next-c1'], steps['alpha-next-c1-read'], steps['tail-engine'], steps['end-full-idle']),
                         ('PASS', 'READ-PASS', 'PASS', 'PASS'))
        result = self.grade_file()
        self.assertEqual((result['schema'], result['verdict'], result['missing'], result['required_arms']),
                         ('alpha-next-c1-grade/1', 'PASS', [], CORE))
        read = self.grade_file('alpha-next-c1-read.json')
        self.assertEqual((read['verdict'], read['required_arms']), ('PASS', ALL_LABELS))
        self.assertTrue((self.q / 'alpha-next-c1-report.md').read_text().startswith('# q18a-alpha-next-c1 report'))

    def test_past_the_core_limit_core_arms_are_inc_and_the_tail_still_runs(self):
        out, err, calls, steps = self.run_loop(jump_after='ingress-' + CORE[9])
        ran = [c['label'] for c in calls if c['script'] == 'box' and c['action'] == 'arm']
        self.assertEqual(ran, CORE[:10] + TAIL)
        for name in CORE[10:]:
            self.assertEqual(steps['arm-' + name], 'INCONCLUSIVE', name)
        self.assertIn('END FAILS=0 INCS=3 CRASHED=0', out, out[-800:])        # two core arms and the core grade
        result = self.grade_file()
        self.assertEqual((result['verdict'], result['classification']), ('INC', None))
        self.assertIn('declared-arm-missing:' + CORE[10], result['missing'])
        self.assertEqual(result['validity'][CORE[10]]['reasons'], ['no-arm-record'])

    def test_a_failing_tail_arm_is_a_read_and_never_counts(self):
        out, err, calls, steps = self.run_loop(rcs={TAIL[1]: 1})
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, out[-1200:])
        self.assertEqual(steps['arm-' + TAIL[1]], 'READ-FAIL')
        self.assertEqual(self.grade_file()['verdict'], 'PASS')

    def test_past_the_tail_limit_the_second_tail_arm_is_a_read_not_run(self):
        out, err, calls, steps = self.run_loop(jump_after='ingress-' + TAIL[0], jump_to='LAST_TAIL_ENTRY_S + 10')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box' and c['action'] == 'arm'], CORE + TAIL[:1])
        self.assertEqual(steps['arm-' + TAIL[1]], 'READ-INCONCLUSIVE')
        self.assertIn('END FAILS=0 INCS=0 CRASHED=0', out, out[-800:])
        self.assertEqual(steps['alpha-next-c1-read'], 'READ-INCONCLUSIVE')
        self.assertEqual(self.grade_file()['verdict'], 'PASS')

    def test_a_dead_engine_in_the_tail_is_crash_accounted_and_the_core_grade_stands(self):
        out, err, calls, steps = self.run_loop(die_in=TAIL[0])
        self.assertIn('CRASHED=1 SOURCE=engine-death', out, out[-800:])
        self.assertEqual(steps['arm-' + TAIL[0]], 'READ-FAIL')
        self.assertEqual(steps['arm-' + TAIL[1]], 'READ-INCONCLUSIVE')
        self.assertEqual(self.grade_file()['verdict'], 'PASS')

    def test_a_dead_engine_in_the_core_stops_the_run_and_skips_the_tail(self):
        out, err, calls, steps = self.run_loop(die_in=CORE[3])
        self.assertIn('CRASHED=1 SOURCE=engine-death', out, out[-800:])
        self.assertEqual(steps['arm-' + CORE[3]], 'FAIL')
        self.assertEqual([c['label'] for c in calls if c['script'] == 'box' and c['action'] == 'arm'], CORE[:4])
        self.assertEqual(steps['tail'], 'READ-INCONCLUSIVE')
        result = self.grade_file()
        self.assertEqual(result['verdict'], 'FAIL')
        self.assertIn(CORE[3] + ':engine-dead', result['failures'])


# --- commands, derivation, clock, parity, report ----------------------------------------------------------------

def body_commands(text):
    body = text[text.index(queue.body().splitlines()[0]):]
    body = body[:body.index('read_step alpha-next-c1-read ')]
    joined = re.sub(r'\\\n\s*', ' ', body)
    rows = []
    for line in joined.splitlines():
        match = re.search(r'python3 "\$H/([a-z0-9_]+)\.py"(.*)$', line)
        if match:
            rows.append((match.group(1), re.split(r'\s+>\s*"', match.group(2))[0].strip()))
    return rows


class Commands(unittest.TestCase):
    def test_every_body_command_parses_with_its_module(self):
        import e3_native39_trunk_checks as checks
        modules = {'e3_alpha_next_c1_box': box, 'e3_native39_trunk_checks': checks}
        rows = body_commands(queue.derive()[DRAFT])
        self.assertEqual(sorted({m for m, _ in rows}), sorted(modules))
        name = 'c1on.g2-live-2'
        sel, cohort, pair = name.split('-')
        c1, group = sel.split('.')
        values = dict(tap.QUEUE_VALUES, _label=name, _sel=sel, _cohort=cohort, _pair=pair, _c1=c1, _group=group,
                      **{'_c1#c1': c1[2:], '_group#g': group[1:]}, _workload='/h/workloads/q18a-alpha-perf-band2.json',
                      READ_ARGS=tap.QUEUE_VALUES['GRADE_ARGS'])
        actions = []
        original = argparse.ArgumentParser.parse_args

        def capture(parser, args=None, namespace=None):
            raise tan.held.Parsed(original(parser, args, namespace))
        for module, args in rows:
            argv = tan.held.expand(args, values)
            with mock.patch.object(argparse.ArgumentParser, 'parse_args', capture):
                with self.assertRaises(tan.held.Parsed) as parsed, contextlib.redirect_stderr(io.StringIO()):
                    modules[module].main(argv)
            actions.append((module, argv[0], vars(parsed.exception.args[0])))
        self.assertEqual([a[1] for a in actions], ['full-idle', 'arm', 'ingress', 'instruments', 'full-idle', 'grade', 'instruments',
                                                   'full-idle', 'read'])
        arm = next(ns for m, a, ns in actions if a == 'arm')
        self.assertEqual((arm['c1'], arm['size'], arm['cohort'], arm['index'], arm['control_timeout_s']), ('on', 2, 'live', 2, 60.0))
        grade = next(ns for m, a, ns in actions if a == 'grade')
        self.assertEqual((grade['out'], grade['report']), ('/q/alpha-next-c1-grade.json', '/q/alpha-next-c1-report.md'))
        read = actions[-1][2]
        self.assertEqual((read['out'], read['report']), ('/q/alpha-next-c1-read.json', '/q/alpha-next-c1-read.md'))

    def test_the_loop_lists_are_the_table_and_the_split_runs_in_bash(self):
        draft = queue.derive()[DRAFT]
        (core,) = re.findall(r'^\s*for _arm in ([a-z0-9 :.-]+); do$', draft, flags=re.M)
        (tail,) = re.findall(r'^\s*for _label in ([a-z0-9 .-]+); do$', draft, flags=re.M)
        self.assertEqual([x.split(':') for x in core.split()], [[r['part'], r['label']] for r in T.arms(T.GRADED)])
        self.assertEqual(tail.split(), TAIL)
        split = ('  IFS=- read -r _sel _cohort _pair <<< "$_label"\n  IFS=. read -r _c1 _group <<< "$_sel"\n'
                 '  if [[ $_pair == %s ]]; then' % T.band1_glob())
        self.assertIn(split, draft)
        script = ('for _arm in %s %s; do _label=${_arm#*:}\n%s echo 1; else echo 2; fi\n'
                  'printf "%%s %%s %%s %%s\\n" "${_c1#c1}" "${_group#g}" "$_cohort" "$_pair"\ndone') % (core, tail, split)
        done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=30)
        got = done.stdout.split('\n')
        self.assertEqual([int(x) for x in got[0::2] if x], [r['band'] for r in T.arms()])
        self.assertEqual([(c, int(s), co, int(i)) for c, s, co, i in (x.split() for x in got[1::2] if x)],
                         [(r['c1'], r['size'], r['cohort'], r['index']) for r in T.arms()])


ARGPARSE_VALUES = dict(tan.ARGPARSE_VALUES, QNAME='q18a-alpha-next-c1', RUN_ID='q18a-alpha-next-c1-fixture', _label='c1on.g2-live-2',
                       _sel='c1on.g2', _c1='c1on', _group='g2', _arm='core:c1on.g2-live-2')
ARGPARSE_VALUES.pop('_c0', None)


def bash_words(args, draft, values=None):
    """alpha-next's helper over this queue's settings and labels: the words bash gives one python call."""
    budget = tap.le_fix_function(draft, 'native41_budget')
    values = dict(ARGPARSE_VALUES, **(values or {}))
    lines = ['set -u', 'fail_early() { :; }', 'fail() { :; }', 'H=%s' % shlex.quote(str(HERE))]
    lines += ['%s=%s' % (k, shlex.quote(v)) for k, v in values.items()]
    arrays = ('--arm "$Q/arm-c1on.g2-live-2.json" --ingress "c1on.g2-live-2=$Q/ingress-c1on.g2-live-2.json" '
              '--watch "c1on.g2-live-2=$Q/engine-watch-arm-c1on.g2-live-2.json"')
    lines += ['GRADE_ARGS=(%s)' % arrays, 'READ_ARGS=(%s)' % arrays,
              'source "$H/runs/q18a_alpha_next_c1_settings.sh" || exit 97', budget, "printf '%s\\0' " + args]
    done = subprocess.run(['bash', '-c', '\n'.join(lines)], capture_output=True, timeout=60)
    return done.returncode, done.stdout.decode().split('\0')[:-1], done.stderr.decode()[-400:]


class QueueArgparse(unittest.TestCase):
    def test_every_queue_python_call_parses_with_its_real_parser(self):
        held = tan.held
        texts = queue.derive()
        draft = texts[DRAFT]
        calls = held.queue_python_calls(texts, (HERE / 'qlib.sh').read_text())
        parent = held.queue_python_calls(an_queue.derive(), (HERE / 'qlib.sh').read_text())
        # alpha-next's calls, one for one, with its box replaced by this queue's.
        swap = lambda s: 'e3_alpha_next_c1_box.py' if s == 'e3_alpha_next_box.py' else s
        self.assertEqual(sorted(swap(c[1]) for c in parent), sorted(c[1] for c in calls))
        self.assertNotIn('e3_alpha_next_box.py', {c[1] for c in calls})
        setcfg = next(p for p in (ROOT / 'deploy/setcfg.py', HERE / 'setcfg.reference.py') if p.is_file())
        rows = []
        for name, script, args, form in calls:
            with self.subTest(file=name.split('/')[-1], script=script, args=args[:60]):
                rc, words, err = bash_words(args, draft)
                self.assertEqual(rc, 0, err)
                if form != 'python3':
                    words = held.QLIB_CALLS[form][1](words)
                if script == 'SETCFG':
                    verdict, detail = held.setcfg_parse(setcfg, words)
                else:
                    verdict, detail = held.real_parse(HERE / script, words)
                self.assertEqual(verdict, 'parsed', (script, words[:4], detail))
                rows.append(dict(file=name.split('/')[-1], script=script, form=form,
                                 action=words[0] if words and not words[0].startswith('-') else None,
                                 words=len(words), verdict=verdict))
        self.assertEqual(len(rows), len(calls))
        ARGPARSE_RECEIPTS[:] = rows
        self.assertEqual(SLOT_RECEIPTS, [])

        def first(script, action):
            return next(a for n, s, a, f in calls if s == script and a.split()[0] == action)
        arm, grade = first('e3_alpha_next_c1_box.py', 'arm'), first('e3_alpha_next_c1_box.py', 'grade')
        for label, original, (old, new) in (('arm-unknown-option', arm, ('--control-timeout-s', '--control-timeout-sec')),
                                            ('arm-undeclared-size', arm, ('--size "${_group#g}"', '--size 3')),
                                            ('arm-undeclared-c1', arm, ('--c1 "${_c1#c1}"', '--c1 maybe')),
                                            ('arm-c0-option-gone', arm, ('--c1 "${_c1#c1}"', '--stage-local on')),
                                            ('arm-undeclared-round', arm, ('--index "$_pair"', '--index 6')),
                                            ('grade-without-out', grade, ('--out "$Q/alpha-next-c1-grade.json"', ''))):
            with self.subTest(control=label):
                self.assertEqual(original.count(old), 1)
                rc, words, err = bash_words(original.replace(old, new), draft)
                self.assertEqual(rc, 0, err)
                self.assertEqual(held.real_parse(HERE / 'e3_alpha_next_c1_box.py', words)[0], 'refused', label)


class Derivation(unittest.TestCase):
    def test_derived_texts_are_on_disk_and_check(self):
        for name, text in queue.derive().items():
            self.assertEqual((ROOT / name).read_text(), text, name)
            self.assertEqual(subprocess.run(['bash', '-n', str(ROOT / name)]).returncode, 0, name)
        queue.main(['--check'])

    def test_undoing_the_declared_deltas_gives_alpha_nexts_texts_back(self):
        texts = queue.derive()
        old = {s: (ROOT / 'harness/runs' / (queue.OLD + s)).read_text() for s in ('.draft.sh', '_settings.sh', '_declare.sh')}
        draft = texts[DRAFT].replace(queue.HEADER_NOTE, '', 1).replace(queue.body(), an_queue.body(), 1)
        draft = draft.replace(queue.INTERFACE_NEW, queue.INTERFACE_OLD, 1).replace(queue.DRAFT_SWITCHES, an_queue.DRAFT_SWITCHES, 1)
        for a, b_, count in queue.DRAFT_NAMES:
            self.assertEqual(draft.count(b_), count, b_)
            draft = draft.replace(b_, a)
        self.assertEqual(draft, old['.draft.sh'])
        settings = texts[SETTINGS].replace(queue.SETTINGS_NOTE, '', 1).replace(queue.SWITCH + ' ', '', 1)
        for a, b_, count in queue.SETTINGS_NAMES:
            settings = settings.replace(b_, a)
        self.assertEqual(settings, old['_settings.sh'])
        declare = texts[DECLARE].replace(queue.DECLARE_SWITCHES, an_queue.DECLARE_SWITCHES, 1).replace(queue.PINS_NEW, queue.PINS_OLD, 1)
        for a, b_, count in queue.DECLARE_NAMES:
            declare = declare.replace(b_, a)
        self.assertEqual(declare, old['_declare.sh'])

    def test_the_c1_switch_is_set_asserted_and_pinned(self):
        texts = queue.derive()
        self.assertIn('FLASH_S5_HANDLER_ANATOMY=1 FLASH_S5_STAGE_OVERLAP=1 ', texts[SETTINGS])
        self.assertIn("assert env.get('FLASH_S5_STAGE_OVERLAP') == '1'", texts[DRAFT])
        self.assertIn("assert env.get('FLASH_S5_STAGE_OVERLAP')=='1'", texts[DECLARE])
        for name in ('e3_alpha_next_c1_box.py', 'alpha_next_c1_gate.py', 'alpha_next_c1_arms.py', 'e3_alpha_next_box.py',
                     'alpha_next_arms.py', 'alpha_next_stop_race.py'):
            self.assertIn(name, texts[DECLARE])
        self.assertNotIn('--stage-local', queue.body())
        self.assertIn('from alpha_next_c1_gate import ACTIVE_S,CLEANUP_S,LAST_ARM_ENTRY_S,LAST_TAIL_ENTRY_S,PHASES', texts[DRAFT])

    def test_alpha_nexts_texts_still_derive_to_their_files(self):
        # This queue derives from alpha-next by import; alpha-next's own texts stay what its tool derives (nothing here edits them).
        for name, text in an_queue.derive().items():
            self.assertEqual((ROOT / name).read_text(), text, name)


NEW_FILES = ('harness/e3_alpha_next_c1_box.py', 'harness/alpha_next_c1_gate.py', 'harness/alpha_next_c1_arms.py',
             'tools/alpha_next_c1_queue.py', 'tools/alpha_next_c1_report.py')


class NoClock(unittest.TestCase):
    def test_no_absolute_time_or_window_in_the_new_files(self):
        for name in NEW_FILES:
            text = (ROOT / name).read_text()
            for line_no, line in enumerate(text.splitlines(), 1):
                for word in tan.FORBIDDEN:
                    self.assertNotIn(word, line, '%s:%d' % (name, line_no))
        draft = queue.derive()[DRAFT]
        self.assertEqual(draft.count('$(date'), 1)
        self.assertEqual(draft.count('time.time_ns()'), 1)
        self.assertIn('RUN_ID="q18a-alpha-next-c1-$(date -u +%Y%m%dT%H%M%SZ)-$$"', draft)
        self.assertNotIn('HARD_END', draft)

    def test_the_gate_table_and_box_import_no_clock(self):
        for name in NEW_FILES:
            tree = ast.parse((ROOT / name).read_text())
            imported = {a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
            imported |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
            self.assertFalse(imported & {'time', 'datetime', 'calendar', 'zoneinfo'}, name)


class C1Parity(unittest.TestCase):
    """C1's grade pieces restated in e3_alpha_next_c1_box (importing e3_alpha_c1_box imports e3_alpha_c0_box, which wraps
    chunk_breakdown process-wide): constants and functions equal to C1's source text."""

    def source(self, path):
        text = (HERE / path).read_text()
        tree = ast.parse(text)
        out = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                out[node.targets[0].id] = ast.get_source_segment(text, node.value)
            elif isinstance(node, ast.FunctionDef):
                out[node.name] = ast.get_source_segment(text, node)
        return out

    def test_constants_and_functions_equal_c1s(self):
        c1, mine = self.source('e3_alpha_c1_box.py'), self.source('e3_alpha_next_c1_box.py')
        for name in ('OVERLAP_COUNTERS', 'OVERLAP_SIGNATURES', 'OVERLAP_MARK', 'OVERLAP_LINE', 'FRAGMENT_FIELD',
                     'seqs_of', 'overlap_owners', 'regroup', 'overlap_rows', '_fragments', 'oom_rises', 'in_flight', '_segment',
                     'regrouping'):
            self.assertEqual(mine[name], c1[name], name)
        self.assertNotIn('e3_alpha_c1_box', sys.modules.get('e3_alpha_next_c1_box').__dict__)

    def test_memory_whole_is_c1s_plus_the_plan2_identity(self):
        whole = ('flash-s5 memory point=open rank=1 txn_seq=2 chunk=0-1024 allocated=1 reserved=2 inactive_split=3 alloc_retries=4 '
                 'ooms=0 free=5 total=6 sync_all_streams=0 device_allocs=7 device_frees=8')
        self.assertTrue(box.MEMORY_WHOLE.search(whole))
        self.assertTrue(box.MEMORY_WHOLE.search(whole + ' pid=201 seq=9 sha=00ab'))
        self.assertFalse(box.MEMORY_WHOLE.search(whole + ' pid=201 (Worker_TP1 pid=202) seq=9 sha=00ab'))

    def test_the_engine_line_formats_are_the_engines(self):
        so = (ROOT / 'patches/s5_engine/stage_overlap.py').read_text()
        self.assertIn("'flash-s5 stage-overlap txn=%d seqs=%s successor=%s'", so)
        for _, text in box.OVERLAP_SIGNATURES:
            self.assertIn(text, so)
        nm = (ROOT / 'patches/s5_engine/native_measurement.py').read_text()
        self.assertIn("'%s pid=%d seq=%d sha=%s\\n' % (text, self.pid, seq, receipt_sha([text]))", nm)


class Report(unittest.TestCase):
    def test_the_post_run_report_regroups_c1_on_arms_and_keys_by_c1(self):
        emit, arms = run_of('core')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tap.write_streams(root, emit.lines)
            os.replace(root / 'engine-stream.log', root / 'engine-current.log')
            for a in arms:
                (root / ('arm-%s.json' % a['label'])).write_text(json.dumps(a))
            result = report_tool.analyse(str(root), [str(root / 'engine-current.log')])
            plain = report_tool.anr.analyse(str(root), [str(root / 'engine-current.log')])
            out = root / 'report.json'
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(report_tool.main([str(root), str(out)]), 0)
            written = json.loads(out.read_text())
        on, off = result['arms']['c1on.g2-live-1'], result['arms']['c1off.g2-live-1']
        self.assertEqual((on['group'], on['overlap'], on['stage_local'], off['group'], off['overlap']),
                         ('fused.g2.c1on', True, True, 'fused.g2.c1off', False))
        self.assertEqual(result['unkeyed_arms'], [])
        # Regrouped: each C1-on chunk holds its own commands (the same count as the C1-off chunk); without the regroup it does not.
        counts = lambda arm: sorted({c['commands'] for c in arm['chunks']})
        self.assertEqual(counts(on), counts(off))
        self.assertNotEqual(counts(plain['arms']['c1on.g2-live-1']), counts(off))
        self.assertEqual((written['schema'], written['old'], written['new']), ('alpha-next-c1-report/1', 'fused.g2.c1off', 'fused.g2.c1on'))
        self.assertIn('fused.g2.c1on', written['memory_peak'])
        self.assertIs(report_tool.sar.box.assemble, progress.assemble)


if __name__ == '__main__':
    unittest.main()
