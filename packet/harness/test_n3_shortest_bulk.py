"""Real AutoArmPolicy/_plan CPU checks; leaves in n3_shortest_bulk_fixture_leaves.json."""
import inspect
from pathlib import Path
import sys
import textwrap
import unittest
from unittest import mock
import weakref

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'patches/s5_engine')]
import native_auto_arm as auto
import scheduler_hooks as sh
import test_compose5_declines as prior

NOW = 1_800_000_000.0


def request(uid, tokens, age=0, **kwargs):
    return prior.req(uid, tokens, arrival_time=NOW-age, **kwargs)


def stage(*requests, on=True, cached=None):
    rig = prior.Stage(*requests)
    rig.controller.shortest_bulk = on
    rig.controller.native_hold_s = 30
    rig.controller.append_yield = sh.AppendYieldConfig(8192, 4, 8, 4)
    rig.lookups, rig.journal = [], []
    cached = cached or {}
    def lookup(req):
        rig.lookups.append(req.request_id)
        return prior.NS(blocks=()), cached.get(req.request_id, 0), 0
    rig.scheduler.kv_cache_manager.get_computed_blocks = lookup
    rig.driver._record = lambda kind, **fields: rig.journal.append(dict(kind=kind, **fields))
    rig.driver.append_yield_active = lambda: False
    return rig


def select(rig):
    with mock.patch.object(auto.time, 'time', return_value=NOW):
        return rig.policy._select()


class Picker(unittest.TestCase):
    def test_off_is_the_old_head_with_one_lookup_and_no_age_read(self):
        a, b = request('a', 60001), request('b', 20001, 90)
        for setting in (False, None):
            rig = stage(a, b, on=setting)
            if setting is None:
                del rig.controller.shortest_bulk
            with mock.patch.object(auto.time, 'time', side_effect=AssertionError('off read wall time')):
                self.assertTrue(rig.policy._select())
            self.assertIs(rig.policy._request(), a)
            self.assertEqual(rig.lookups, ['a'])
            self.assertEqual(rig.queue, [a, b])
            self.assertEqual(rig.policy.bulk_pick['scans'], 0)

    def test_remaining_rows_and_selected_plan_reuse(self):
        a, b = request('a', 60001), request('b', 100001)
        rig = stage(a, b, cached={'b': 80000})
        self.assertTrue(select(rig))
        self.assertIs(rig.policy._request(), b)
        self.assertEqual(rig.policy.plan['start'], 80000)
        self.assertEqual(rig.lookups, ['a', 'b'])
        self.assertEqual(rig.queue, [a, b])
        stats = rig.policy.decline_record()['shortest_bulk']
        self.assertEqual((stats['extra_lookup_calls'], stats['reordered']), (1, 1))
        self.assertGreaterEqual(stats['extra_lookup_ns'], 0)
        receipt = next(r for r in rig.journal if r['kind'] == 'bulk_pick')
        self.assertEqual((receipt['request_uid'], receipt['original_position'], receipt['remaining_rows'], receipt['aged']), ('b', 1, 20000, False))
        # This is the exact queue/object seam main's admission adapter must use.
        queue, target = sh.PrefillAdmissionController._waiting_object(rig.controller, rig.scheduler, rig.queue, b)
        self.assertIs(queue, rig.queue)
        self.assertIs(target, b)
        rig.policy.phase = 'native'
        self.assertTrue(rig.policy.permits(target))
        self.assertFalse(rig.policy.permits(a))

    def test_45_seconds_and_oldest_use_wall_clock(self):
        for age in (44.999, 45, 46):
            a, b = request('long', 90001, age), request('shorter', 20001)
            rig = stage(a, b)
            rig.driver.clock = lambda: 1.0
            self.assertTrue(select(rig))
            self.assertIs(rig.policy._request(), a if age >= 45 else b)
        a, b, c = request('a', 20001, 44.999), request('b', 80001, 45), request('c', 90001, 46)
        rig = stage(a, b, c)
        self.assertTrue(select(rig))
        self.assertIs(rig.policy._request(), c)
        self.assertEqual(rig.policy.bulk_pick['aged'], 1)

    def test_ties_keep_fcfs(self):
        for ages, sizes in (((0, 0), (40001, 40001)), ((46, 46), (90001, 20001))):
            a, b = request('a', sizes[0], ages[0]), request('b', sizes[1], ages[1])
            rig = stage(a, b)
            self.assertTrue(select(rig))
            self.assertIs(rig.policy._request(), a)

    def test_ready_short_head_or_single_bulk_never_waits_for_partner(self):
        for max_new in (8192, 10000):
            a, b = request('short', max_new+1), request('old', 90001, 90)
            rig = stage(a, b)
            rig.controller.append_yield = sh.AppendYieldConfig(max_new, 4, 8, 4)
            self.assertTrue(select(rig))
            self.assertIs(rig.policy._request(), a)
            self.assertEqual(rig.lookups, ['short'])
            self.assertEqual(rig.policy.bulk_pick['scans'], 0)
        alone = request('alone', 40001)
        rig = stage(alone)
        self.assertTrue(select(rig))
        self.assertIs(rig.policy._request(), alone)

    def test_scan_stops_at_short_barrier(self):
        a, b, c, d = request('a', 60001), request('b', 30001), request('short', 8001), request('old', 90001, 90)
        rig = stage(a, b, c, d)
        self.assertTrue(select(rig))
        self.assertIs(rig.policy._request(), b)
        self.assertEqual(rig.lookups, ['a', 'b', 'short'])
        self.assertEqual(rig.policy.bulk_pick['short_stop'], 1)
        self.assertEqual(rig.queue, [a, b, c, d])

    def test_unplannable_nonhead_is_skipped_without_releasing_it(self):
        a, bad, b = request('a', 60001), request('bad', 12001, ec_transfer_params={'blocked': True}), request('b', 30001)
        rig = stage(a, bad, b)
        rig.policy.holds['bad'] = (weakref.ref(bad), 0)
        self.assertTrue(select(rig))
        self.assertIs(rig.policy._request(), b)
        self.assertEqual(rig.lookups, ['a', 'b'])
        self.assertIn('bad', rig.policy.holds)
        self.assertEqual(rig.policy.bulk_pick['unplannable'], 1)

    def test_rejected_nonhead_is_not_retried_or_mutated(self):
        a, bad, b = request('a', 60001), request('bad', 12001, 90), request('b', 30001)
        rig = stage(a, bad, b)
        rig.policy.rejected['bad'] = weakref.ref(bad)
        rig.policy.refused_cohort['bad'] = ('other-cohort',)
        self.assertTrue(select(rig))
        self.assertIs(rig.policy._request(), b)
        self.assertEqual(rig.lookups, ['a', 'b'])
        self.assertIs(rig.policy.rejected['bad'](), bad)
        self.assertEqual(rig.policy.refused_cohort['bad'], ('other-cohort',))
        self.assertEqual(rig.policy.retries, {})

    def test_refused_head_and_cohort_keep_existing_disposition(self):
        for reason in ('rejected', 'unplannable', 'cohort'):
            a, b = request('a', 60001), request('b', 20001)
            rig = stage(a, b)
            if reason == 'rejected':
                rig.controller.native_residue = 1
                rig.policy.rejected['a'] = weakref.ref(a)
                rig.policy.refused_cohort['a'] = ('d1',)
            elif reason == 'unplannable':
                a.ec_transfer_params = {'blocked': True}
            else:
                rig.scheduler.running[0].is_prefill_chunk = True
            self.assertFalse(select(rig))
            self.assertEqual(rig.lookups, [])
            self.assertEqual(rig.policy.phase, 'idle')
            self.assertIsNone(rig.policy._request())

    def test_real_queue_selector_keeps_skipped_priority(self):
        a, b = request('a', 60001), request('b', 20001)
        rig = stage(a, b)
        skipped = request('skipped', 90001)
        rig.scheduler.requests['skipped'] = skipped
        rig.scheduler.waiting = rig.queue
        rig.scheduler.skipped_waiting = prior.Queue([skipped])
        rig.scheduler._select_waiting_queue_for_scheduling = lambda: rig.scheduler.skipped_waiting or rig.scheduler.waiting
        rig.controller._queue = lambda s: sh.PrefillAdmissionController._queue(rig.controller, s)
        self.assertTrue(select(rig))
        self.assertIs(rig.policy._request(), skipped)
        self.assertEqual(rig.lookups, ['skipped'])

    def test_ten_candidate_cap_is_visible(self):
        for count in (10, 11):
            requests = [request(str(i), 60001-i*1000) for i in range(count)]
            rig = stage(*requests)
            self.assertTrue(select(rig))
            self.assertIs(rig.policy._request(), requests[9])
            self.assertEqual(rig.lookups, [str(i) for i in range(10)])
            self.assertEqual(rig.policy.bulk_pick['capped'], int(count > 10))
            self.assertEqual(rig.policy.bulk_pick['candidates'], 10)

    def test_chosen_plan_side_state_survives_later_probes(self):
        a, chosen, last = request('a', 60001), request('chosen', 20001), request('last', 40001)
        bad = request('bad', 12001, ec_transfer_params={'blocked': True})
        rig = stage(a, bad, chosen, last)
        rig.controller.native_residue = 1
        def lookup(req):
            rig.lookups.append(req.request_id)
            return prior.NS(blocks=()), 0, 176 if req.request_id == chosen.request_id else 0
        rig.scheduler.kv_cache_manager.get_computed_blocks = lookup
        rig.controller._append_yield_split_view = lambda s, req: sh._ShortTailPrefillRequest(req, 20000) if req.request_id == chosen.request_id else req
        rig.policy._decline_why = 'before'
        self.assertTrue(select(rig))
        self.assertIs(rig.policy._request(), chosen)
        self.assertEqual(rig.policy._plan_residue, 'junction')
        self.assertTrue(rig.policy._plan_moved)
        self.assertEqual(rig.policy._decline_why, 'before')
        self.assertEqual(rig.policy.residue_counts['junction'], 1)
        self.assertEqual(rig.policy.hold_totals['moved_stop'], 1)

    def test_junction_probes_preserve_requests_and_plan_the_actual_junction_stop(self):
        import types
        import test_coverage_residue as residue

        a = request('head', 60001, shared_prefix_boundary=33)
        chosen = request('chosen', 30001, shared_prefix_boundary=11)
        last = request('last', 40001, shared_prefix_boundary=22)
        rig = stage(a, chosen, last)
        rig.controller.native_residue = 1
        rig.controller.minimum_stage_rows = 64
        rig.controller.chunk_tokens = 2048
        for name, value in residue.split_host(max_num_scheduled_tokens=1024).items():
            setattr(rig.scheduler, name, value)
        rig.scheduler._mamba_block_aligned_split = types.MethodType(residue.SPLIT, rig.scheduler)
        rig.controller._aligned_split = lambda s, req, limit, **kw: s._mamba_block_aligned_split(req, limit, **kw)
        def lookup(req):
            rig.lookups.append(req.request_id)
            return prior.NS(blocks=()), (0 if req is a else 4048), {'head': 0, 'chosen': 4576, 'last': 4752}[req.request_id]
        rig.scheduler.kv_cache_manager.get_computed_blocks = lookup
        before = [dict(vars(req)) for req in (a, chosen, last)]
        self.assertTrue(select(rig))
        self.assertEqual([dict(vars(req)) for req in (a, chosen, last)], before)
        self.assertEqual(rig.lookups, ['head', 'chosen', 'last'])
        self.assertIs(rig.policy._request(), chosen)
        self.assertEqual(rig.policy.plan['start'] + rig.policy.plan['plan'][0], 4576)
        self.assertEqual(sum(rig.policy.plan['plan']), rig.policy.plan['end'] - 4048)
        self.assertEqual(rig.policy._plan_residue, 'junction')
        self.assertFalse(rig.policy._plan_moved)
        self.assertEqual(rig.policy.hold_totals['moved_stop'], 0)


class Mutants(unittest.TestCase):
    def caught(self, method, old, new, test):
        original = getattr(auto.AutoArmPolicy, method)
        source = textwrap.dedent(inspect.getsource(original))
        self.assertIn(old, source)
        namespace = dict(vars(auto))
        exec(compile(source.replace(old, new, 1), '<shortest-bulk-mutant>', 'exec'), namespace)
        with mock.patch.object(auto.AutoArmPolicy, method, namespace[method]):
            result = unittest.TestResult()
            Picker(test).run(result)
        self.assertTrue(result.failures or result.errors, test)
        return result

    def test_total_prompt_instead_of_remaining(self):
        self.caught('_shortest_bulk', "candidate_plan['end'] - candidate_plan['start'])), age", "candidate.num_tokens)), age", 'test_remaining_rows_and_selected_plan_reuse')

    def test_monotonic_age(self):
        self.caught('_shortest_bulk', 'now = time.time()', 'now = self.driver.clock()', 'test_45_seconds_and_oldest_use_wall_clock')

    def test_age_boundary_and_oldest(self):
        self.caught('_shortest_bulk', 'age >= BULK_PICK_AGE_S else', 'age > BULK_PICK_AGE_S else', 'test_45_seconds_and_oldest_use_wall_clock')
        self.caught('_shortest_bulk', '(0, candidate.arrival_time)', '(0, -candidate.arrival_time)', 'test_45_seconds_and_oldest_use_wall_clock')

    def test_returner_barrier(self):
        self.caught('_shortest_bulk', "stats['short_stop'] += 1\n            break", "stats['short_stop'] += 1\n            continue", 'test_scan_stops_at_short_barrier')

    def test_side_state_restore(self):
        self.caught('_shortest_bulk', 'self._plan_moved, self._plan_residue, self._decline_why = state', 'pass', 'test_chosen_plan_side_state_survives_later_probes')

    def test_probe_request_mutation(self):
        result = self.caught('_plan', "if lookup_stats is None and not getattr(c, 'shortest_bulk', False):", 'if True:',
                             'test_junction_probes_preserve_requests_and_plan_the_actual_junction_stop')
        self.assertTrue(result.failures)
        self.assertEqual(result.errors, [])

    def test_rejected_filter_and_cap(self):
        self.caught('_shortest_bulk', "self.rejected.get(candidate.request_id, lambda: None)() is candidate", 'False', 'test_rejected_nonhead_is_not_retried_or_mutated')
        self.caught('_shortest_bulk', 'islice(queue, 1, BULK_PICK_CANDIDATES)', 'islice(queue, 1, None)', 'test_ten_candidate_cap_is_visible')

    def test_redundant_selected_lookup(self):
        self.caught('_select', 'request, plan = self._shortest_bulk(queue, request, plan)', 'request, plan = self._shortest_bulk(queue, request, plan)\n        plan = self._plan(request)', 'test_remaining_rows_and_selected_plan_reuse')


if __name__ == '__main__':
    unittest.main()
