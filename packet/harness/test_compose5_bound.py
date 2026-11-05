"""compose-5: b13 on this table's rows. The real plan search and the real overlap refusal's cap; the stage command's checks through the real driver method's validation; the box's client over every row.

From decode-phase dp6's suite, without what compose-5 leaves out (the witness, dp8d's other arms). CPU only.
"""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'patches', 's5_engine'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compose5_arms as T
from scheduler_hooks import bounded_limit, staged_chunk_plan


def stock_split(position, limit, end):
    """The stock alignment as the box's chunks show it: a chunk ends at a Mamba block when it crosses one, and at the last hash stop."""
    stop = end // T.HASH_BLOCK_ROWS * T.HASH_BLOCK_ROWS
    following = min(position + limit, end)
    if following < end and following // T.MAMBA_BLOCK_ROWS * T.MAMBA_BLOCK_ROWS > position:
        following = following // T.MAMBA_BLOCK_ROWS * T.MAMBA_BLOCK_ROWS
    if position < stop < following:
        following = stop
    return following - position


def real_plan(prompt, bound, cap=T.CHUNK_TOKENS, minimum=2):
    end = prompt - 1

    def split(position, limit):
        return stock_split(position, bounded_limit(bound, position, limit), end)
    return staged_chunk_plan(0, end, cap, minimum, split)


class Bound(unittest.TestCase):
    def test_none_is_the_unbounded_plan(self):
        self.assertEqual(bounded_limit(None, 99999, 2048), 2048)
        self.assertEqual(list(real_plan(65445, None)), T.planned_chunks(65445))
        self.assertEqual(len(T.planned_chunks(65445)), 34, "dp8d's cf")

    def test_position_decides(self):
        self.assertEqual(bounded_limit((52624, 1024), 52623, 2048), 2048)
        self.assertEqual(bounded_limit((52624, 1024), 52624, 2048), 1024)
        self.assertEqual(bounded_limit((52624, 1024), 60000, 700), 700)

    def test_the_real_search_gives_the_tables_plan_at_all_three_bands(self):
        for prompt in (65445, 98270, 131094):
            plan = list(real_plan(prompt, T.B13_BOUND))
            self.assertEqual(plan, T.planned_chunks(prompt, T.CHUNK_TOKENS, T.B13_BOUND), prompt)
            self.assertEqual(sum(plan), prompt - 1)

    def test_band_1_is_dp8ds_forty_chunks(self):
        self.assertEqual(len(real_plan(65445, T.B13_BOUND)), 40)

    def test_no_chunk_past_the_bound_exceeds_its_rows_and_none_before_exceeds_2048(self):
        end, rows = T.B13_BOUND
        position = 0
        for size in real_plan(131094, T.B13_BOUND):
            self.assertLessEqual(size, rows if position >= end else T.CHUNK_TOKENS)
            position += size

    def test_every_row_carries_it(self):
        self.assertEqual({tuple(r['chunk_bound']) for r in T.arms()}, {(52624, 1024)})
        self.assertEqual({(r['cow_host_filter'], r['cow_fast']) for r in T.arms()}, {(True, True)})
        self.assertFalse(any(r.get('score_split') or r.get('stock_rescue') for r in T.arms()), '5f is compose-6')


class Stage(unittest.TestCase):
    """The driver's stage command, stopped before anything else runs: the base class is stubbed; the field's handling is the real code."""

    def driver(self):
        from stage_overlap import OverlapSegmentDriver

        class Base:
            def _stage_control(self, command):
                return dict(local=True)
        d = OverlapSegmentDriver.__new__(OverlapSegmentDriver)
        d.controller = SimpleNamespace(chunk_tokens=T.CHUNK_TOKENS, chunk_bound=None)
        d.overlap_selected, d.defer_bind_installed, d.lazy_recv_installed, d.decode_merge_installed, d.defer_bind = True, False, False, False, True
        d.inwait_installed, d.inwait, d.early_successor, d.exact_successor, d._inwait_poll_ns = True, False, False, False, {}
        return d, Base

    def call(self, d, Base, **command):
        import stage_overlap
        parent = stage_overlap.StagedSegmentDriver._stage_control
        stage_overlap.StagedSegmentDriver._stage_control = lambda self, c: Base()._stage_control(c)
        try:
            return d._stage_control(dict(command, local=True))
        finally:
            stage_overlap.StagedSegmentDriver._stage_control = parent

    def test_set_and_cleared(self):
        d, Base = self.driver()
        reply = self.call(d, Base, chunk_bound=list(T.B13_BOUND))
        self.assertEqual(d.controller.chunk_bound, (52624, 1024))
        self.assertEqual(reply['chunk_bound'], T.B13_BOUND)
        reply = self.call(d, Base, chunk_bound=None)
        self.assertIsNone(d.controller.chunk_bound)
        self.assertIn('chunk_bound', reply)

    def test_absent_leaves_it(self):
        d, Base = self.driver()
        d.controller.chunk_bound = (0, 1536)
        reply = self.call(d, Base)
        self.assertEqual(d.controller.chunk_bound, (0, 1536))
        self.assertNotIn('chunk_bound', reply)

    def test_refused(self):
        d, Base = self.driver()
        for bad in ([1], [0, 512], [0, 4096], [-1, 1024], [0, 1024.0], 'x', True):
            with self.assertRaises(Exception) as caught:
                self.call(d, Base, chunk_bound=bad)
            self.assertIn('chunk-bound-invalid', str(caught.exception))
            self.assertIsNone(d.controller.chunk_bound)

    def test_the_rows_rows_refused_where_the_launch_has_no_2048(self):
        """A launch that lost FLASH_S5_CHUNK_TOKENS still takes [52624, 1024], so the declaration's assert and the smoke's chunk count are what catch it."""
        d, Base = self.driver()
        d.controller.chunk_tokens = 1024
        self.call(d, Base, chunk_bound=list(T.B13_BOUND))
        self.assertEqual(d.controller.chunk_bound, (52624, 1024))


class PleCounter(unittest.TestCase):
    """At 2048 the scratch has no lever, so the counter the smoke and the grade hold the arm to has to move without one. Real scratch, real path lookup, fake tensors."""

    def scratch(self, rows):
        import model_hooks as mh
        tensor = SimpleNamespace(numel=lambda: 1, element_size=lambda: 1, shape=(rows, 3, 4))
        embedding = SimpleNamespace(supports_prefetch=True, _prefetch_buffer=SimpleNamespace(shape=(1024, 3, 4), dtype=None), _prefetch_stream=object(), etp_data_parallel_size=1)
        layer = SimpleNamespace(ple=SimpleNamespace(ngram_embedding=None, ple_embedding=SimpleNamespace(ngram_embedding=embedding)))
        scratch = mh.PlePrivateScratch(embedding, rows, torch_module=SimpleNamespace(empty=lambda *a, **k: tensor), device='cpu', stream=SimpleNamespace(cuda_stream=9), selectable=True)
        return mh, scratch, SimpleNamespace(layers=[layer])

    def setUp(self):
        import model_hooks as mh
        saved = dict(mh.PLE_PRIVATE)
        self.addCleanup(mh.PLE_PRIVATE.update, saved)
        mh.PLE_PRIVATE.update(selected=False, scopes=0)

    def test_at_2048_it_engages_without_the_lever_and_counts(self):
        mh, scratch, model = self.scratch(T.CHUNK_TOKENS)
        self.assertFalse(scratch.selectable)
        self.assertTrue(scratch.engaged(model, 0))
        self.assertEqual(mh.PLE_PRIVATE['scopes'], 1)

    def test_at_2048_with_the_lever_sent_too_it_counts_once_per_segment(self):
        mh, scratch, model = self.scratch(T.CHUNK_TOKENS)
        mh.select_ple_private(True)
        scratch.engaged(model, 0), scratch.engaged(model, 0)
        self.assertEqual(mh.PLE_PRIVATE['scopes'], 2)

    def test_at_1024_nothing_moved_the_lever_decides_and_off_counts_nothing(self):
        mh, scratch, model = self.scratch(1024)
        self.assertFalse(scratch.engaged(model, 0))
        self.assertEqual(mh.PLE_PRIVATE['scopes'], 0)
        mh.select_ple_private(True)
        self.assertTrue(scratch.engaged(model, 0))
        self.assertEqual(mh.PLE_PRIVATE['scopes'], 1)


class Client(unittest.TestCase):
    """The box's arm client: the bound goes out on the arm's own command, for every row, and a row without the field sends nothing."""

    class Fake:
        def __init__(self):
            self.calls = []

        def call(self, action, **fields):
            self.calls.append((action, fields))
            return dict(inwait=fields.get('inwait'), early_successor=fields.get('early_successor'), exact_successor=fields.get('exact_successor'), chunk_bound=fields.get('chunk_bound'))

    def selecting(self, fake, record, row, *bound):
        import e3_inwait_box as box
        return box._InwaitSelecting(fake, row['inwait'], record, row['next_levers'], row['attn'], fake, row['prepare_slices'], row['score_warm'], row['cow_fast'], row['reserved_cap'], row['cow_host_filter'], row['exact_successor'],
                                    row.get('score_warm_rank0', False), row['allocator_flush'], row['cow_verify'], row['ple_private'], row['ple_witness'], *bound)

    def test_every_row(self):
        import e3_inwait_box as box
        for row in T.arms():
            fake, record = self.Fake(), {}
            self.selecting(fake, record, row, row['chunk_bound']).call(box.STAGE_CONTROL, local=True)
            self.assertEqual(fake.calls[-1][1]['chunk_bound'], T.B13_BOUND)
            self.assertEqual(record['inwait']['bound_acknowledged'], T.B13_BOUND)

    def test_a_row_without_the_field_sends_nothing(self):
        import compose4_arms as T4
        import e3_inwait_box as box
        fake, record = self.Fake(), {}
        row = T4.by_label()[T4.labels()[0]]
        self.assertNotIn('chunk_bound', row)
        self.selecting(fake, record, row).call(box.STAGE_CONTROL, local=True)
        self.assertNotIn('chunk_bound', fake.calls[-1][1])
        self.assertNotIn('bound_acknowledged', record['inwait'])


if __name__ == '__main__':
    unittest.main()
