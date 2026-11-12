"""yield-rules Y1 (lead, 12 Nov 2026): FLASH_S5_YIELD_NO_APPEND_OWNER. CPU only.

Box evidence (V1 j2 records, fable-levers-20261112/queue-sim trace; builder-tail-route-20261112/STATUS-yield-rules.md):
163 of 651 junction yields interrupt an append owner (owner rows left <= max_new) to serve another append; each pays
the yield-in, the yield-out and the owner's restart chunk.

  Y1  with the flag at 1, ``_yield_offer`` takes no yield (and does no candidate lookup) while the lane owner has at
      most ``append_yield.max_new`` rows left past its newest admitted or retained chunk; the waiting request then
      starts at the owner's end through the ordinary switch.  Counted once per junction with a request waiting
      (``queue_policy.no_append_owner``), with one 'flash-s5 n3-yield-owner-refused' receipt; a fault counts
      ``errors`` and today's offer runs.  Flag at 0: no change at all (journal, counters, lookups, status shape).

Rig: test_append_yield.Rig (real PrefillAdmissionController, segment driver, AutoArmPolicy, four bridge workers).
Its bulk has 12 tokens, chunks 4, 4, 3 (native end 11): rows left 7 after chunk 1, 3 after chunk 2.
No GPU, CUDA or box claim.
"""
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'patches/s5_engine')]
import test_append_yield as ay  # noqa: E402
import test_e3_admission_native as adm  # noqa: E402
import engine_driver as ed  # noqa: E402
import scheduler_hooks as sh  # noqa: E402
from e3_types import E3ContractError  # noqa: E402

NS = types.SimpleNamespace
Rig = ay.Rig
RECEIPT = 'flash-s5 n3-yield-owner-refused '


class Lines:
    """The driver progress logger, captured (receipts are its lines)."""
    def __init__(self, tc):
        self.lines = []
        logger = NS(info=lambda fmt, *a: self.lines.append(fmt % a), warning=lambda *a, **k: None)
        patch = mock.patch.object(ed, '_progress_logger', return_value=logger)
        patch.start()
        tc.addCleanup(patch.stop)

    def receipts(self):
        return [json.loads(line[len(RECEIPT):]) for line in self.lines if line.startswith(RECEIPT)]


def rig_on(on=True, **kw):
    rig = Rig(cached={}, **kw)
    rig.c.yield_no_append_owner = on
    return rig


def first_junction(rig, append_tokens=11, append_cached=8):
    """Bulk chunk 1 committed (frontier 4) with an append held; the driver offers at that boundary."""
    append = ay.run_to_yield(rig, append_tokens=append_tokens, append_cached=append_cached)
    for _ in range(3):
        rig.d.advance()
        rig.stack.complete_everything()
    return append


def state(rig):
    return dict(phase=rig.d.append_yield_phase(), kinds=[k for k in rig.kinds() if k.startswith('append_yield')],
                counters=dict(rig.d.append_yield_counters), lookups=list(rig.lookups),
                yield_counts=dict(rig.d._yield_counts))


class Setting(unittest.TestCase):
    def controller(self, env):
        with mock.patch.dict(os.environ, env):
            scheduler = NS(kv_cache_manager=NS(), max_num_scheduled_tokens=1024)
            return sh.PrefillAdmissionController(scheduler, leases=sh.SchedulerLeases(), frontier=adm.ah.CommittedFrontier(),
                                                 engine_epoch=7, chunk_tokens=4, ranks=(0, 1), max_history=64)

    def test_strict_boolean_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(sh.YIELD_NO_APPEND_OWNER_ENV, None)
            self.assertFalse(self.controller({}).yield_no_append_owner)
        self.assertFalse(self.controller({sh.YIELD_NO_APPEND_OWNER_ENV: '0'}).yield_no_append_owner)
        self.assertTrue(self.controller({sh.YIELD_NO_APPEND_OWNER_ENV: '1'}).yield_no_append_owner)
        for bad in ('2', 'yes', '', ' 1'):
            with self.assertRaises(E3ContractError):
                self.controller({sh.YIELD_NO_APPEND_OWNER_ENV: bad})

    def test_status_block_only_with_the_flag(self):
        off, on = rig_on(False), rig_on(True)
        self.assertNotIn('no_append_owner', off.c.native_admission_record().get('queue_policy') or {})
        self.assertEqual(on.c.native_admission_record()['queue_policy']['no_append_owner'],
                         dict(junctions=0, bulks=0, receipts=0, receipts_dropped=0, errors=0))


class Rule(unittest.TestCase):
    def setUp(self):
        self.log = Lines(self)

    def test_off_yields_into_the_append_owner_as_today(self):
        rig = rig_on(False)
        first_junction(rig)
        self.assertIn(rig.d.append_yield_phase(), ('closing', 'append'))
        self.assertEqual(rig.d.append_yield_counters['selected'], 1)
        self.assertEqual(self.log.receipts(), [])

    def test_on_refuses_at_rows_left_le_max_new_without_a_lookup(self):
        rig = rig_on(True)                         # max_new 8, rows left 7
        append = first_junction(rig)
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(rig.d.append_yield_counters['selected'], 0)
        self.assertNotIn(ay.APPEND_UID, rig.lookups)
        self.assertEqual(rig.c.yield_owner_counts, dict(junctions=1, bulks=1, receipts=1, receipts_dropped=0, errors=0))
        (receipt,) = self.log.receipts()
        self.assertEqual({k: receipt[k] for k in ('frontier', 'native_end', 'rows_left', 'max_new', 'waiting')},
                         dict(frontier=4, native_end=11, rows_left=7, max_new=8, waiting=1))
        self.assertEqual(receipt['bulk_uid'], ay.BULK_UID)
        self.assertNotIn(append.request_id, json.dumps(receipt))
        # the owner continues: its next chunk is admitted from the frontier
        txn = rig.admit()
        self.assertEqual((txn.request_uid, txn.chunk_start), (ay.BULK_UID, 4))

    def test_boundary_rows_left_equal_max_new_refuses(self):
        rig = rig_on(True, max_new=7)
        first_junction(rig, append_tokens=9, append_cached=4)
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(rig.c.yield_owner_counts['junctions'], 1)

    def test_rows_left_above_max_new_yields_exactly_as_off(self):
        on, off = rig_on(True, max_new=6), rig_on(False, max_new=6)
        for rig in (on, off):
            first_junction(rig)
            ay.to_append(rig)
        self.assertEqual(state(on), state(off))
        self.assertEqual(on.c.yield_owner_counts['junctions'], 0)

    def test_counted_once_per_junction_and_owner(self):
        rig = rig_on(True)
        first_junction(rig)
        for _ in range(5):                         # more ticks at the same junction
            rig.d.advance()
            rig.stack.complete_everything()
        self.assertEqual(rig.c.yield_owner_counts['junctions'], 1)
        rig.admit()
        rig.finish_chunk()                         # junction 2: frontier 8, rows left 3
        for _ in range(3):
            rig.d.advance()
            rig.stack.complete_everything()
        self.assertEqual(rig.c.yield_owner_counts, dict(junctions=2, bulks=1, receipts=2, receipts_dropped=0, errors=0))
        self.assertEqual([r['rows_left'] for r in self.log.receipts()], [7, 3])
        self.assertIsNone(rig.d.append_yield_phase())

    def test_no_waiting_request_counts_nothing_and_yields_nothing(self):
        rig = rig_on(True)
        rig.admit()
        rig.finish_chunk()
        for _ in range(3):
            rig.d.advance()
            rig.stack.complete_everything()
        self.assertEqual(rig.c.yield_owner_counts['junctions'], 0)
        self.assertEqual(self.log.receipts(), [])
        # a request arriving later at the same junction is counted then
        rig.add_held(ay.APPEND_UID, 11, cached=8)
        rig.d.advance()
        self.assertEqual(rig.c.yield_owner_counts['junctions'], 1)
        self.assertIsNone(rig.d.append_yield_phase())

    def test_a_fault_takes_todays_offer(self):
        rig = rig_on(True)
        real = rig.c.native_end

        def faulty(scheduler, request):
            if request is rig.bulk:
                raise RuntimeError('fault')
            return real(scheduler, request)
        with mock.patch.object(rig.c, 'native_end', side_effect=faulty):
            first_junction(rig)
        self.assertEqual(rig.d.append_yield_counters['selected'], 1)
        self.assertGreaterEqual(rig.c.yield_owner_counts['errors'], 1)
        self.assertEqual(rig.c.yield_owner_counts['junctions'], 0)

    def test_a_receipt_fault_still_refuses_and_counts(self):
        rig = rig_on(True)
        with mock.patch.object(sh, '_text', side_effect=E3ContractError('bad uid')):
            first_junction(rig)
        self.assertIsNone(rig.d.append_yield_phase())
        self.assertEqual(rig.c.yield_owner_counts['junctions'], 1)
        self.assertEqual(rig.c.yield_owner_counts['errors'], 1)

    def test_receipts_are_bounded(self):
        rig = rig_on(True)
        rig.c.yield_owner_counts['receipts'] = sh.YIELD_OWNER_RECEIPT_LIMIT
        first_junction(rig)
        self.assertEqual(self.log.receipts(), [])
        self.assertEqual(rig.c.yield_owner_counts['receipts_dropped'], 1)
        self.assertIsNone(rig.d.append_yield_phase())

    def test_quota_stop_comes_first_unchanged(self):
        rig = rig_on(True, per_bulk=1)
        txn = rig.admit()
        rig.d._yield_counts = {(txn.request_uid, txn.request_generation): 1}
        rig.add_held(ay.APPEND_UID, 11, cached=8)
        rig.finish_chunk()
        for _ in range(3):
            rig.d.advance()
            rig.stack.complete_everything()
        self.assertEqual(rig.c.yield_owner_counts['junctions'], 0)
        self.assertIsNone(rig.d.append_yield_phase())


class OffIsFrozen(unittest.TestCase):
    """Flag 0: the same run with the Y1 hook removed from the driver gives the same journal, counters and lookups."""
    def setUp(self):
        Lines(self)

    def scenario(self):
        rig = rig_on(False)
        append = first_junction(rig)
        ay.to_append(rig)
        ay.run_append(rig)
        ay.to_full(rig)
        rig.stock_handback(append)
        ay.to_resume(rig)
        return rig

    def test_off_equals_no_hook(self):
        with_hook = state(self.scenario())
        with mock.patch.object(sh.PrefillAdmissionController, 'yield_owner_refuses',
                               side_effect=AssertionError('called with the flag off')):
            again = state(self.scenario())
        self.assertEqual(with_hook, again)
        self.assertEqual(with_hook['counters']['reopened'], 0)
        self.assertEqual(with_hook['phase'], 'resume')


class Staged:
    def setUp(self):
        Rig.DRIVER = 'staged'
        self.addCleanup(setattr, Rig, 'DRIVER', 'native')
        super().setUp()


class RuleStaged(Staged, Rule):
    pass


class OffIsFrozenStaged(Staged, OffIsFrozen):
    pass


class Mutants(unittest.TestCase):
    def setUp(self):
        Lines(self)

    def run_mutant(self, source_old, source_new, test):
        path = ROOT / 'patches/s5_engine/scheduler_hooks.py'
        text = path.read_text()
        self.assertEqual(text.count(source_old), 1, source_old)
        code = compile(text.replace(source_old, source_new, 1), str(path), 'exec')
        mutant = type(sys)('scheduler_hooks_mutant')
        mutant.__file__ = str(path)
        exec(code, mutant.__dict__)
        with mock.patch.object(sh.PrefillAdmissionController, 'yield_owner_refuses',
                               mutant.PrefillAdmissionController.yield_owner_refuses):
            with self.assertRaises(AssertionError):
                test(self)

    def test_strict_boundary_mutant_caught(self):
        self.run_mutant('            if left > config.max_new:\n', '            if left >= config.max_new:\n',
                        Rule.test_boundary_rows_left_equal_max_new_refuses)

    def test_refusal_without_dedupe_caught(self):
        self.run_mutant('        if self._yield_owner_seen == mark:\n            return True\n',
                        '        if False:\n            return True\n', Rule.test_counted_once_per_junction_and_owner)

    def test_counting_without_a_waiter_caught(self):
        self.run_mutant('            if not waiting:\n                return True\n',
                        '            if False:\n                return True\n',
                        Rule.test_no_waiting_request_counts_nothing_and_yields_nothing)

    def test_driver_hook_after_the_lookup_caught(self):
        """The rule must run before the candidate lookup (no lookup cost on a refused junction)."""
        path = ROOT / 'patches/s5_engine/engine_driver.py'
        text = path.read_text()
        hook = text.index('controller.yield_owner_refuses(')
        lookup = text.index('candidate = controller.append_yield_candidate(self.scheduler, bulk,')
        quota = text.index('if self._yield_counts.get(key, 0) >= config.per_bulk:')
        self.assertLess(quota, hook)
        self.assertLess(hook, lookup)


if __name__ == '__main__':
    unittest.main()
