"""append-pieces lever A (lead 12 Nov 2026): FLASH_S5_APPEND_NO_BLOCK_STOP (default 0).

An append (at most 8192 prefill rows past the turn's cached start) runs a native piece through the Mamba block boundary
stock's split stops it at, continuing with stock's own next piece; every other stop (the prompt's last hash boundary, a
shared-prefix junction, the native end, the cap and chunk bound) is stock's.  At commit the block the piece left (it holds
the state at the piece's start) is withheld from cache_blocks.  CPU only, the hub rig: the real controller (admit, commit,
staged plan) over stock's own _mamba_block_aligned_split (the reference text, unchanged) at production geometry (block
4048, unit 176, native cap 2048, stock budget 1024, the dp6 chunk bound 2024 from 97152).  The real stock KV manager is
test_append_no_block_stop_stack (pinned container).
"""
import json
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'patches/s5_engine')]

import scheduler_hooks as sh  # noqa: E402
import test_coverage_residue as tcr  # noqa: E402
import test_e3_admission_native as adm  # noqa: E402
import test_geometry_tail as tgt  # noqa: E402

BLOCK, UNIT, FLOOR = tcr.BLOCK, tcr.UNIT, tcr.FLOOR
CHUNK = 2048                        # FLASH_S5_CHUNK_TOKENS on the box: native_chunk_cap(2048, 1024) = 2048
BOUND = (97152, 2024)               # decode-phase dp6 chunk bound as the V1 arms set it
ENV = sh.APPEND_NO_BLOCK_STOP_ENV
# V1 appends (block 1, n3.g2-live-0, from n3_rows.json): (prompt, cached start, recorded pieces)
V1 = ((88020, 84656, (352, 2048, 944)),
      (84813, 80432, (528, 2048, 1648, 156)),
      (91661, 88000, (1056, 2048, 416, 140)),
      (76670, 71984, (880, 2048, 1648, 109)))


class Group:
    """An align-mode Mamba group as commit sees it: the cached-block counts cache_blocks reads."""
    mamba_cache_mode = 'align'

    def __init__(self, log):
        self.num_cached_block, self.log = {}, log


class Other:
    mamba_cache_mode = None

    def __init__(self):
        self.num_cached_block = {}


def rig(n, start, flag, *, bound=BOUND, junction=0):
    with mock.patch.dict(os.environ, {ENV: flag}):
        request, scheduler, manager, controller = tgt.admission_rig(n, start=start)
    controller.chunk_tokens = CHUNK
    controller.chunk_bound = bound
    request.shared_prefix_boundary = junction
    manager.junction = junction                     # the residue lookup sets the request's junction from the manager
    log = []
    groups = (Group(log), Other())
    manager.coordinator = NS(single_type_managers=groups, find_longest_cache_hit=None)
    inner = manager.cache_blocks

    def cache_blocks(request, end):
        log.append(('cache_blocks', end, dict(groups[0].num_cached_block)))
        return inner(request, end)
    manager.cache_blocks = cache_blocks
    for g in groups:
        g.num_cached_block[request.request_id] = start // BLOCK
    return request, scheduler, manager, controller, groups, log


def model_ends(start, P, block_stop):
    """The seam model (builder-append-pieces scripts/seam_counts.py, after the reuse investigator's piece_model.py; 1,030 of
    1,042 V1 appends): cap 2048 (2024 from 97152), the 4048 stop for a piece starting mid-block (lever A: none), the last
    176 boundary below P - 1, then a terminal piece if more than 64 rows are left, else the native end is that boundary."""
    end = P - 1
    aligned = end // UNIT * UNIT
    pos, out = start, []
    while pos < aligned:
        nxt = min(pos + (2048 if pos < 97152 else 2024), aligned)
        b = (pos // BLOCK + 1) * BLOCK
        if block_stop and pos % BLOCK and b < nxt:
            nxt = b
        out.append(nxt)
        pos = nxt
    if end - pos > FLOOR:
        out.append(end)
    return out


def pieces(records):
    return tuple(r.transaction.chunk_end - r.transaction.chunk_start for r in records)


def ends(records):
    return [r.transaction.chunk_end for r in records]


class Setting(unittest.TestCase):
    def test_strict_zero_or_one(self):
        for value in ('2', 'on', ' 1', ''):
            with self.subTest(value=value), mock.patch.dict(os.environ, {ENV: value}):
                with self.assertRaises(sh.E3ContractError):
                    tgt.admission_rig(5000)

    def test_unset_and_zero_are_off_with_no_status_block(self):
        for env in ({}, {ENV: '0'}):
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=False):
                if not env:
                    os.environ.pop(ENV, None)
                _r, _s, _m, controller = tgt.admission_rig(5000)
                self.assertFalse(controller.append_no_block_stop)
                self.assertNotIn('append_pieces', controller.native_admission_record())

    def test_on_has_the_status_block(self):
        with mock.patch.dict(os.environ, {ENV: '1'}):
            _r, _s, _m, controller = tgt.admission_rig(5000)
        block = controller.native_admission_record()['append_pieces']
        self.assertEqual(block['max_new'], 8192)
        self.assertEqual((block['appends'], block['fused_pieces'], block['hist'], block['fallbacks']), (0, 0, {}, {}))


class Plan(unittest.TestCase):
    """Real admit + commit over stock's split: pieces, ends and the next turn's hit position (the last piece's end)."""

    def run_turn(self, n, start, flag, **kw):
        request, scheduler, manager, controller, groups, log = rig(n, start, flag, **kw)
        records = tgt.run_native(controller, scheduler)
        return records, controller, groups, log, request

    def test_v1_appends_reproduce_the_record_off_and_fuse_on(self):
        for n, start, recorded in V1:
            with self.subTest(n=n):
                off, *_ = self.run_turn(n, start, '0')
                on, controller, *_ = self.run_turn(n, start, '1')
                self.assertEqual(pieces(off), recorded)
                self.assertEqual(ends(on)[-1], ends(off)[-1], 'the native end, so the next turn\'s hit, is unchanged')
                self.assertLessEqual(len(on), len(recorded))
                self.assertEqual(controller.native_admission_record()['append_pieces']['hist'], {str(len(on)): 1})
                self.assertEqual(controller.append_pieces['fused_pieces'], 1)
        # the cut only moves where the cap count is unchanged (76670: 4,576 rows to the tail boundary are 3 cap pieces
        # either way); the other three lose their head piece
        self.assertEqual([len(self.run_turn(n, s, '1')[0]) for n, s, _ in V1], [2, 4, 3, 4])
        for n, s, _ in V1:
            self.assertEqual(ends(self.run_turn(n, s, '1')[0]), model_ends(s, n, False))
            self.assertEqual(ends(self.run_turn(n, s, '0')[0]), model_ends(s, n, True))

    def test_the_first_v1_append_runs_two_pieces_through_the_block_boundary(self):
        on, controller, groups, log, _ = self.run_turn(88020, 84656, '1')
        self.assertEqual(pieces(on), (2048, 1296))
        self.assertEqual(ends(on), [86704, 88000])
        a = controller.append_pieces
        self.assertEqual((a['appends'], a['pieces'], a['fused_pieces'], a['stops_skipped'], a['bulk_pieces']), (1, 2, 1, 1, 0))
        self.assertEqual(a['stale_states_unpublished'], 1)
        self.assertEqual(a['fallbacks'], {})

    def test_sweep_every_end_but_block_stops_is_stocks(self):
        # starts on the 176 grid inside one block, prompts up to MAX_NEW rows on: same native end, the kept stops kept,
        # at most one block boundary inside a piece, no more pieces than stock, every piece within cap and floor
        seen_fused = 0
        for start in range(20 * BLOCK + UNIT, 21 * BLOCK, UNIT * 5):
            for rows in (300, 2047, 2700, 4100, 7900):
                n = start + rows
                with self.subTest(start=start, rows=rows):
                    off, *_ = self.run_turn(n, start, '0')
                    on, controller, *_ = self.run_turn(n, start, '1')
                    self.assertEqual(ends(on)[-1], ends(off)[-1])
                    self.assertLessEqual(len(on), len(off))
                    tail = n // UNIT * UNIT
                    self.assertEqual(tail in ends(on), tail in ends(off), 'the last hash boundary stop is kept')
                    self.assertEqual(ends(on), model_ends(start, n, False))
                    self.assertEqual(ends(off), model_ends(start, n, True))
                    for r in on:
                        s, e = r.transaction.chunk_start, r.transaction.chunk_end
                        self.assertLessEqual(e - s, CHUNK)
                        self.assertLessEqual(e // BLOCK - s // BLOCK, 1)
                    seen_fused += controller.append_pieces['fused_pieces']
                    self.assertGreaterEqual(controller.append_pieces['fused_pieces'], len(off) - len(on))
                    self.assertEqual(controller.append_pieces['pieces'], len(on))
        self.assertGreater(seen_fused, 10)

    def test_a_bulk_turn_keeps_every_block_stop(self):
        off, *_ = self.run_turn(13000, 0, '0')             # 12,999 prefill rows from 0: more than MAX_NEW
        on, controller, *_ = self.run_turn(13000, 0, '1')
        self.assertEqual(pieces(on), pieces(off))
        self.assertEqual(pieces(off)[:2], (2048, 2000))
        a = controller.append_pieces
        self.assertEqual((a['appends'], a['fused_pieces'], a['bulk_pieces']), (0, 0, len(on)))

    def test_eight_thousand_one_hundred_ninety_two_new_rows_is_an_append_one_more_is_not(self):
        start = 20 * BLOCK + 500
        for rows, fused in ((8192, True), (8193, False)):
            with self.subTest(rows=rows):
                n = start + rows                         # prefill end = the prompt: new rows = n - start
                on, controller, *_ = self.run_turn(n, start, '1')
                a = controller.append_pieces
                self.assertEqual((a['fused_pieces'] > 0, a['appends'], a['bulk_pieces'] > 0), (fused, int(fused), not fused))

    def test_a_block_boundary_that_is_the_last_hash_boundary_is_kept(self):
        b = 22 * BLOCK                                   # every block boundary is a unit multiple (4048 = 23 x 176)
        start, n = b - 1000, b + 100                     # tail boundary = b; r = 100 > 64: a terminal piece past it
        off, *_ = self.run_turn(n, start, '0')
        on, controller, *_ = self.run_turn(n, start, '1')
        self.assertEqual(ends(on), ends(off))
        self.assertIn(b, ends(on))
        self.assertEqual(controller.append_pieces['fallbacks'], {'tail-boundary': 1})

    def test_a_junction_ahead_keeps_the_block_stop(self):
        b = 21 * BLOCK
        start, n = b - 1000, b + 2000
        off, *_ = self.run_turn(n, start, '0', junction=b + 500)
        on, controller, *_ = self.run_turn(n, start, '1', junction=b + 500)
        self.assertEqual(ends(on), ends(off))
        self.assertIn(b, ends(on))
        self.assertEqual(controller.append_pieces['fallbacks'].get('junction'), 1)

    def test_a_junction_already_behind_does_not_keep_the_stop(self):
        b = 21 * BLOCK
        start, n = b - 3000, b + 2000                    # cap piece, then the junction stop, then through b
        on, controller, *_ = self.run_turn(n, start, '1', junction=b - 600)
        self.assertIn(b - 600, ends(on))
        self.assertNotIn(b, ends(on))
        self.assertEqual(controller.append_pieces['fallbacks'], {})

    def test_a_geometry_tail_turn_still_ends_at_its_last_hash_boundary(self):
        n, start = 88020, 84656                          # r = 20: native ends at 88000, stock runs the 20 rows
        on, controller, *_ = self.run_turn(n, start, '1')
        self.assertEqual((on[-1].transaction.chunk_end, on[-1].remaining_tokens), (88000, 20))


class Commit(unittest.TestCase):
    def test_the_stale_block_is_withheld_before_cache_blocks_and_only_for_the_crossing_piece(self):
        request, scheduler, manager, controller, groups, log = rig(88020, 84656, '1')
        uid = request.request_id
        records = tgt.run_native(controller, scheduler)
        k = 84656 // BLOCK
        # piece 1 [84656, 86704) crosses 85008: the align group's count is k + 1 when cache_blocks runs; the other group
        # (attention) is untouched by the guard
        self.assertEqual(log[0], ('cache_blocks', 86704, {uid: k + 1}))
        self.assertEqual(groups[1].num_cached_block[uid], k)
        self.assertEqual(log[1][0:2], ('cache_blocks', 88000))
        self.assertEqual(len(records), 2)

    def test_off_commit_never_touches_the_counts(self):
        request, scheduler, manager, controller, groups, log = rig(88020, 84656, '0')
        tgt.run_native(controller, scheduler)
        self.assertEqual([entry[2] for entry in log], [{request.request_id: 84656 // BLOCK}] * len(log))

    def test_a_count_below_the_block_moves_past_it_and_is_counted(self):
        request, scheduler, manager, controller, groups, log = rig(88020, 84656, '1')
        groups[0].num_cached_block[request.request_id] = 3
        tgt.run_native(controller, scheduler)
        k = 84656 // BLOCK
        self.assertEqual(log[0][2], {request.request_id: k + 1})
        self.assertEqual(controller.append_pieces['extra_blocks_unpublished'], k - 3)

    def test_mutant_without_the_guard_publishes_the_crossing_piece_at_the_hit_block(self):
        request, scheduler, manager, controller, groups, log = rig(88020, 84656, '1')
        with mock.patch.object(sh.PrefillAdmissionController, '_append_unpublish_stale', lambda self, r, t: True):
            tgt.run_native(controller, scheduler)
        self.assertEqual(log[0][2], {request.request_id: 84656 // BLOCK},
                         'without the guard cache_blocks starts at the stale block (the stack suite shows the wrong hit)')


class Fallback(unittest.TestCase):
    def test_no_turn_start_keeps_stocks_size_and_logs_one_numbers_only_line(self):
        request, scheduler, manager, controller, groups, log = rig(88020, 84656, '1')
        view = sh._ProjectedPrefillRequest(request, 84700)               # mid-turn query, no first piece seen
        lines = []
        logger = NS(info=lambda fmt, text: lines.append(fmt % text))
        with mock.patch('engine_driver._progress_logger', lambda: logger):
            for _ in range(3):
                size = controller._aligned_split(scheduler, view, 2048)
        self.assertEqual(size, 85008 - 84700)
        self.assertEqual(controller.append_pieces['fallbacks'], {'no-turn-start': 1})
        self.assertEqual(len(lines), 1)
        head, body = lines[0].split(' ', 2)[:2], json.loads(lines[0].split(' ', 2)[2])
        self.assertEqual(head, ['flash-s5', 'append-pieces-fallback'])
        self.assertEqual(sorted(body), ['boundary', 'engine_epoch', 'observed_ns', 'position', 'prompt_tokens', 'reason', 'rows'])
        self.assertEqual((body['reason'], body['position'], body['boundary']), ('no-turn-start', 84700, 85008))

    def test_a_fault_is_stocks_size_and_counted(self):
        request, scheduler, manager, controller, groups, log = rig(88020, 84656, '1')
        bad = types.MethodType(lambda self, r, p, l: 1 // 0, controller)
        with mock.patch.object(controller, '_append_turn_start', bad), mock.patch('engine_driver._progress_logger',
                                                                                  lambda: NS(info=lambda *a: None)):
            size = controller._aligned_split(scheduler, request, 2048, num_new_local_computed_tokens=84656)
        self.assertEqual(size, 85008 - 84656)
        self.assertEqual((controller.append_pieces['errors'], controller.append_pieces['fallbacks']), (1, {'error': 1}))

    def test_shape_eagle_or_checkpoints_keep_the_stop(self):
        for field, value in (('use_eagle_block_drop', True), ('mamba_has_prefill_checkpoint_blocks', True),
                             ('mamba_partial_cache_hit', False)):
            with self.subTest(field=field):
                request, scheduler, manager, controller, groups, log = rig(88020, 84656, '1')
                setattr(scheduler, field, value)
                with mock.patch('engine_driver._progress_logger', lambda: NS(info=lambda *a: None)):
                    size = controller._aligned_split(scheduler, request, 2048, num_new_local_computed_tokens=84656)
                stock = scheduler._mamba_block_aligned_split(request, 2048, num_new_local_computed_tokens=84656)
                self.assertEqual(size, stock)
                if field != 'mamba_partial_cache_hit' or stock == 85008 - 84656:
                    self.assertEqual(controller.append_pieces['fallbacks'].get('shape', 0), int(stock == 85008 - 84656))


class OffIsParent(unittest.TestCase):
    def test_off_split_is_stocks_value_over_a_sweep(self):
        request, scheduler, manager, controller, groups, log = rig(88020, 84656, '0')
        for position in range(80000, 88000, 37):
            for limit in (64, 700, 2024, 2048):
                view = sh._ProjectedPrefillRequest(request, position)
                with self.subTest(position=position, limit=limit):
                    self.assertEqual(controller._aligned_split(scheduler, view, limit),
                                     scheduler._mamba_block_aligned_split(view, limit))
        self.assertEqual(controller._append_turns, {})


if __name__ == '__main__':
    unittest.main()
