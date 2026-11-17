#!/usr/bin/env python3
"""inwait build-4, review sections 2 and 6: earlier admission is earlier allocation, and the small invariant of the send paths. Discovery mode.

Real composition: test_inwait_engine's overlap stack (real controller, overlap driver, leases, frontier, worker bridges, ring model) with the early
successor selected; the fake manager records what each allocation owns. Each cleanup must release only what that admission owns:
  Admission: cancel after the early allocation, failed admission, allocation pressure, the partial-prefix boundary.
  Invariant: a refused submission leaves nothing in flight, on every send in-wait reaches; accepted work is never forgotten on a stop.
test_compose2_receipt_refused (ee4412e5's fault injection) stays in the suite list beside this file.
"""
import json
import logging
import unittest
from unittest import mock

import test_e3_stage_overlap as c1
import test_inwait_engine as ie
import e3_alpha_next_c1b_box as c1box
import e3_c1_open_rows as rule
import e3_inwait_box as box
import e3_native41_progress_box as pbox

import stage_progress as sp
from e3_types import E3ContractError
from reply_transport import Backpressure


class Network(ie.FlagOn, ie.Early, c1.OverlapBase):
    def early_point(self, network):
        """The early state: n's stage 0 finalize, before its release; nothing admitted yet."""
        controller, driver = network.controller, network.driver
        self.reach(network, lambda: len(driver._frames) == 1 and driver._frames[0].stage[0]['phase'] in ('finalize', 'release')
                   and 0 not in driver._frames[0].released and controller.successor is None and not driver._frames[0].dead)

    def older_state(self, network):
        """What the older chunk owns: it must read the same after the successor's cleanup."""
        controller = network.controller
        parked = controller.parked
        return dict(transaction=parked.transaction, ticket=parked.ticket, blocks=parked.admission.flat_blocks(), bound=parked.bound,
                    start=parked.transaction.chunk_start, end=parked.transaction.chunk_end)

    def stop(self, network, kind='abort'):
        driver = network.driver
        commits = network.controller.commits
        driver.stop(kind)
        network.frozen = set()
        for _ in range(3000):
            if not driver.active:
                break
            network.tick()
        self.assertEqual(network.controller.commits, commits)
        self.assert_closed(network)
        self.assertEqual(dict(driver._inflight), {})


class Admission(Network):
    def test_cancel_after_the_early_allocation_releases_the_successor_only(self):
        network = self.network()
        controller = network.controller
        self.early_point(network)
        manager = network.scheduler.kv_cache_manager
        before = list(manager.block_pool.freed)
        network.admit_next()
        successor = controller.successor
        self.assertIsNotNone(successor)
        older = self.older_state(network)
        mine = set(successor.admission.flat_blocks()) - set(older['blocks'])
        self.assertTrue(mine, 'the early admission allocated')
        request = controller.cancel(successor.transaction, never_submitted=True)
        self.assertIs(request, controller.parked.request)
        self.assertIsNone(controller.successor)
        self.assertEqual(self.older_state(network), older)
        self.assertEqual(controller.overlap_counters['successor_cancels'], 1)
        self.assertFalse(set(manager.block_pool.freed[len(before):]) & set(older['blocks']), 'nothing the older chunk owns was freed')
        network.frozen = set()
        network.driver.stop('abort')
        for _ in range(3000):
            if not network.driver.active:
                break
            network.tick()
        self.assert_closed(network)

    def test_the_driver_stop_with_an_admitted_successor_keeps_the_older_commit_count(self):
        network = self.network()
        self.early_point(network)
        network.admit_next()
        self.assertIsNotNone(network.controller.successor)
        self.stop(network)
        self.assertEqual(network.driver.overlap_counters['unbegun_successor_cancels'], 1)

    def test_a_failed_allocation_changes_nothing_and_the_request_goes_on(self):
        network = self.network()
        controller = network.controller
        self.early_point(network)
        manager = network.scheduler.kv_cache_manager
        older = self.older_state(network)
        request = controller.parked.request
        held = (request.num_computed_tokens, request.num_in_flight_tokens)
        receipts = (controller.refused, controller.alloc_failures, controller.refusal_seq, controller.last_refusal)
        row = manager.rows[request.request_id]
        manager.fail = True
        self.assertIsNone(network.admit_next())
        manager.fail = False
        self.assertIsNone(controller.successor)
        self.assertEqual(self.older_state(network), older)
        self.assertEqual((request.num_computed_tokens, request.num_in_flight_tokens), held, 'provisional fields restored')
        self.assertEqual((controller.refused, controller.alloc_failures, controller.refusal_seq, controller.last_refusal), receipts, 'not a serial event')
        self.assertEqual(manager.rows[request.request_id], row)
        self.assertGreaterEqual(controller.overlap_counters['alloc_wait'], 1)
        network.run_request()
        self.assertEqual(controller.commits, network.expected_commits)
        self.assertIsNone(network.driver.error)

    def test_an_exception_in_the_allocation_restores_the_older_chunk(self):
        network = self.network()
        controller = network.controller
        self.early_point(network)
        manager = network.scheduler.kv_cache_manager
        older = self.older_state(network)
        request = controller.parked.request
        held = (request.num_computed_tokens, request.num_in_flight_tokens)

        class OutOfBlocks(RuntimeError):
            pass
        with mock.patch.object(manager, 'allocate_slots', side_effect=OutOfBlocks()):
            with self.assertRaises(OutOfBlocks):
                controller.admit(network.scheduler)
        self.assertIsNone(controller.successor)
        self.assertEqual(self.older_state(network), older)
        self.assertEqual((request.num_computed_tokens, request.num_in_flight_tokens), held)
        self.assertIsNone(controller._successor_queue)
        network.run_request()
        self.assertEqual(controller.commits, network.expected_commits)

    def test_sustained_pressure_runs_the_request_serially_and_equal(self):
        network = self.network()
        manager = network.scheduler.kv_cache_manager
        real = manager.allocate_slots
        driver, controller = network.driver, network.controller

        def pressed(request, num_new, **kwargs):
            # Every early allocation fails: only after a commit may a chunk get its blocks.
            if controller.parked is None and driver._frames and not driver._frames[0].dead:
                manager.alloc_calls.append((request, num_new, kwargs))
                return None
            return real(request, num_new, **kwargs)
        manager.allocate_slots = pressed
        network.run_request(limit=20000)
        self.assertIsNone(driver.error)
        self.assertEqual(controller.commits, network.expected_commits)
        self.assertGreater(controller.overlap_counters['alloc_wait'], 0)
        self.assertEqual(driver.overlap_counters.get('overlapped_commits', 0), 0, 'pressure, so no chunk overlapped')

    def boundary(self, network, boundary):
        """The fake request's shared prefix boundary, and one real group the rule reads."""
        network.controller.parked.request.shared_prefix_boundary = boundary

    def test_at_the_partial_prefix_boundary_nothing_is_allocated_until_the_commit(self):
        network = self.network()
        controller = network.controller
        manager = network.scheduler.kv_cache_manager
        self.early_point(network)
        end = controller.parked.transaction.chunk_end
        self.boundary(network, end)
        calls = len(manager.alloc_calls)
        older = self.older_state(network)
        self.assertIsNone(network.admit_next())
        self.assertEqual(len(manager.alloc_calls), calls, 'read before any allocation')
        self.assertIsNone(controller.successor)
        self.assertEqual(self.older_state(network), older)
        self.assertEqual(controller.last_successor_refusal['reason'], 'older-at-shared-prefix-boundary')
        network.run_request()
        self.assertEqual(controller.commits, network.expected_commits)

    def test_a_partial_hit_pending_waits_too(self):
        network = self.network()
        controller = network.controller
        manager = network.scheduler.kv_cache_manager
        self.early_point(network)
        request = controller.parked.request
        group = c1.types.SimpleNamespace(use_eagle=False, _partial_hit_reqs={request.request_id}, _checkpoints={})
        manager.coordinator.single_type_managers = (group,)
        calls = len(manager.alloc_calls)
        self.assertIsNone(network.admit_next())
        self.assertEqual(len(manager.alloc_calls), calls)
        self.assertEqual(controller.last_successor_refusal['reason'], 'partial-tail-pending')
        group._partial_hit_reqs.clear()
        network.run_request()
        self.assertEqual(controller.commits, network.expected_commits)

    def test_the_older_chunk_may_not_be_cancelled_under_its_successor(self):
        network = self.network()
        controller = network.controller
        self.early_point(network)
        network.admit_next()
        with self.assertRaises(E3ContractError):
            controller.cancel(controller.parked.transaction, never_submitted=True)
        self.assertIsNotNone(controller.successor)


SENDS = ('flash_s5_open', 'flash_s5_bind', 'flash_s5_submit', 'flash_s5_receipt', 'flash_s5_release')
OF_KIND = dict(flash_s5_open='open', flash_s5_bind='bind', flash_s5_submit='segment', flash_s5_receipt='receipt', flash_s5_release='release')


class Invariant(ie.Base):
    """Every send an in-wait arm makes: open, bind, segment, receipt, release, on the wait's path and the tick's."""

    def setUp(self):
        super().setUp()

    def network_early(self):
        network = self.burst(self.network(direct=True))
        network.driver.early_successor = True
        return network

    def refuse(self, driver, name, times=1, in_wait=None):
        seen = dict(refused=0, inflight=[], native=[])
        real = driver._check_headroom

        def check(ring, envelope):
            if envelope[0] == name and seen['refused'] < times and (in_wait is None or driver._in_wait is in_wait):
                seen['refused'] += 1
                seen['inflight'].append(set(driver._inflight))
                seen['native'].append(set(driver.progress.native) if driver.progress is not None else set())
                with mock.patch.object(sp, 'ring_has_headroom', return_value=False), mock.patch.object(sp, 'ring_limiting_readers', return_value=()):
                    return real(ring, envelope)
            return real(ring, envelope)
        driver._check_headroom = check
        return seen

    def test_a_refused_send_leaves_nothing_in_flight_and_the_chunk_goes_on(self):
        for name in SENDS:
            with self.subTest(send=name):
                network = self.network_early()
                driver = network.driver
                seen = self.refuse(driver, name, times=3)
                submit = driver._submit
                after = []

                def spy(kind, envelope, ranks, **kwargs):
                    before = set(driver._inflight)
                    try:
                        return submit(kind, envelope, ranks, **kwargs)
                    except Backpressure:
                        # The invariant itself: refused here, so nothing was marked, on the driver and the progress record.
                        after.append((kind, set(driver._inflight) - before))
                        raise
                driver._submit = spy
                network.run_request(limit=20000, tick=self.in_wait_tick(network))
                self.assertGreater(seen['refused'], 0, name)
                self.assertTrue(after, name)
                self.assertEqual([row for row in after if row[1]], [], 'a refused submission left work in flight')
                self.same_as_serial(network)
                self.assertEqual(dict(driver._inflight), {})

    def stop_after_accept(self, kind, stop='abort'):
        network = self.network_early()
        driver = network.driver
        submit = driver._submit

        def freeze_after(sent, envelope, ranks, **kwargs):
            # No rank reads its ring after this send, so the accepted command is still out when the stop comes.
            sequence = submit(sent, envelope, ranks, **kwargs)
            if sent == kind:
                network.frozen = {0, 1, 2, 3}
            return sequence
        driver._submit = freeze_after
        self.reach(network, lambda: kind in driver._inflight.values(), limit=4000, tick=self.in_wait_tick(network))
        driver._submit = submit
        commits = network.controller.commits
        driver.stop(stop)
        network.frozen = set()
        for _ in range(4000):
            if not driver.active:
                break
            network.tick()
        self.assertFalse(driver.active, kind)
        self.assertEqual(network.controller.commits, commits, kind)
        self.assert_closed(network)
        self.assertEqual(dict(driver._inflight), {}, 'accepted work was forgotten')

    def test_a_stop_with_each_kind_of_accepted_work_out_closes_it(self):
        for kind in ('open', 'bind', 'segment', 'receipt', 'release'):
            with self.subTest(kind=kind):
                self.stop_after_accept(kind)


class StopInTheEarlyWindow(ie.Early, c1.StopStates):
    """c1's stop states, once more with allocation pressure on every successor after the first."""

    def network(self, **kwargs):
        network = super().network(**kwargs)
        manager = network.scheduler.kv_cache_manager
        controller = network.controller
        real, count = manager.allocate_slots, [0]

        def sometimes(request, num_new, **inner):
            if controller.parked is None:
                count[0] += 1
                if count[0] > 1 and count[0] % 2:
                    manager.alloc_calls.append((request, num_new, inner))
                    return None
            return real(request, num_new, **inner)
        manager.allocate_slots = sometimes
        return network

    # Pressure moves c1's reach to the prompt's last chunk, where the eligibility rule refuses: Admission and StopStatesEarly own this state.
    test_admitted_not_begun = None


class CommitBoundary(ie.Base):
    """The record through the real driver and the real line writer, then the grade's own parser and rule (review section 7)."""

    def run_and_read(self):
        network = self.network_run()
        records, flushes = [], None
        with self.assertLogs(level=logging.INFO) as logs:
            network.run_request(tick=self.in_wait_tick(network))
        self.same_as_serial(network)
        overlap_texts = []
        for line in logs.output:
            at = line.find(pbox.PROGRESS_LINE)
            if at >= 0:
                body = json.loads(line[at + len(pbox.PROGRESS_LINE):])
                if body.get('schema') == pbox.PROGRESS_SCHEMA:
                    records.append((None, body))
            if 'flash-s5 stage-overlap ' in line:
                overlap_texts.append(line)
        flushes, problems = pbox.assemble(records)
        self.assertEqual(problems, [])
        return [f for f in flushes if f['side'] == 'engine'], overlap_texts

    def network_run(self):
        network = self.burst(self.network(direct=True))
        network.driver.early_successor = True
        return network

    def test_every_flush_names_its_boundary_and_the_grade_reconciles_it(self):
        flushes, texts = self.run_and_read()
        self.assertTrue(flushes)
        for flush in flushes:
            identity = flush['identity']
            self.assertIs(type(identity['commit_ns']), int)
            self.assertEqual(identity['open_rows'], len(identity['open_seqs']))
        self.assertGreater(sum(f['identity']['open_rows'] for f in flushes), 0, 'the in-wait early run leaves rows out at a commit')
        overlap = rule.overlap_index(texts, c1box._parse_overlap)
        out = box.check_arm_inwait(flushes, overlap)
        self.assertEqual(out['reasons'], [], out['flagged'][:3])
        read = box.commit_boundary_read(dict(a=flushes))['a']
        self.assertEqual((read['flushes'], read['recorded']), (len(flushes), len(flushes)))

    def test_the_named_sequences_are_capped(self):
        import engine_driver as ed
        self.assertEqual(ed.NativeProgress.OPEN_SEQS_CAP, 32)


if __name__ == '__main__':
    unittest.main()
