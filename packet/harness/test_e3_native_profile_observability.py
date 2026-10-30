"""Receipt-only profile fields and actual engine-hook bypass accounting."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import types
import unittest

import test_e3_native_schedule_profile as p
import executor_hooks as hooks
import e3_native27_profile_receipt as receipt


class ProfileObservability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p.ScheduleProfile.setUpClass()

    def setUp(self):
        self.case = p.ScheduleProfile('runTest')
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)

    def test_queried_fields_and_timing_do_not_change_agreement(self):
        world, driver, command = self.case.boundary((True, True))
        rows = self.case.query(world, driver)
        target = rows[0]
        self.assertEqual(target.selected_count, 4097)
        self.assertGreater(target.worker_host_ns, 0)
        altered = replace(target, worker_host_ns=target.worker_host_ns + 11,
                          selected_count=target.selected_count + 1)
        self.assertEqual(target, altered, 'telemetry must not become an agreement predicate')
        self.assertEqual(p.w.qsa_profile_agreement(driver._profile_query, rows, driver.ranks),
                         p.w.qsa_profile_agreement(driver._profile_query, (altered,) + rows[1:], driver.ranks))
        driver._apply_profile(rows)
        counters = driver.qsa_control.snapshot()['layers'][0]
        self.assertEqual(counters['profile_worker_host_ns'], sum(x.worker_host_ns for x in rows))

    def test_refusal_retains_both_local_profiles(self):
        self.case.selection_counts = (4097, 97153)
        world, driver, command = self.case.boundary((True, True))
        rows = self.case.query(world, driver)
        driver._apply_profile(rows)
        row = next(r for r in driver._journal if r['kind'] == 'schedule_profile_refused')
        peers = row['peer_profiles']
        self.assertEqual([r['rank'] for r in peers], [0, 1])
        self.assertEqual([r['selected_count'] for r in peers], [4097, 97153])
        self.assertEqual([r['bucket_waves'] for r in peers], [[17], [380, 1]])
        self.assertTrue(all(r['worker_host_ns'] > 0 for r in peers))

    def test_refusal_survives_ring_eviction_and_reaches_phase_and_grade(self):
        self.case.selection_counts = (4097, 97153)
        world, driver, command = self.case.boundary((True, True))
        driver._apply_profile(self.case.query(world, driver))
        for index in range(driver.journal_limit + 1):
            driver._record('receipt-test-padding', index=index)
        self.assertFalse(any(row['kind'] == 'schedule_profile_refused'
                             for row in driver._journal))
        status = driver._status_record()
        retained = status['schedule_profile_refusals']
        self.assertEqual(retained['count'], 1)
        self.assertEqual(retained['first'], retained['latest'])
        self.assertEqual([peer['selected_count'] for peer in retained['first']['peer_profiles']],
                         [4097, 97153])
        self.assertEqual([peer['bucket_waves'] for peer in retained['first']['peer_profiles']],
                         [[17], [380, 1]])
        cleanup = {'steps': {'leave_full': {
            'census_before_return': {'snapshot': status}}}}
        phase_receipt = receipt.from_cleanup(cleanup)
        self.assertEqual(phase_receipt, retained)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'qual-row1.json'
            path.write_text(json.dumps({'verdict': 'INCONCLUSIVE',
                                        'schedule_profile_refusals': phase_receipt}))
            grade = receipt.queue_grade_receipts(tmp)
        self.assertEqual(grade['phases']['row1'], retained)
        self.assertEqual(grade['unavailable'], {})

    def test_actual_hook_counts_bypass_once_and_not_utility_pumps(self):
        network = p.ProfileNetwork(self.case)
        engine = network.driver.engine
        engine.scheduler = network.scheduler
        # Startup route exchange makes before_step return a stock bypass.
        out = hooks.before_step_with_batch_queue(engine)
        self.assertIsNotNone(out)
        self.assertEqual(network.driver._stock_steps_skipped, 1)
        before = network.driver._stock_steps_skipped
        network.driver.advance(deliver_outputs=False)
        self.assertEqual(network.driver._stock_steps_skipped, before)
        network.driver.stop('abort')
        for _ in range(100):
            network.complete_everything()
            if not network.driver.active:
                break
            hooks.engine_idle_tick(engine)
        self.assertFalse(network.driver.active)
        self.assertGreaterEqual(network.driver._stock_steps_skipped, before)
        self.assertEqual(network.driver._status_record()['stock_steps_skipped'],
                         network.driver._stock_steps_skipped)


if __name__ == '__main__':
    unittest.main()
