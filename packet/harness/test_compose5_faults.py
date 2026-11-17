"""compose-5, Codex P1 and the re-entry plan. CPU only.

P1: a journal write that raised killed its agent outside the turn's handler, and `run` only looked at threads still alive, so the
arm read as finished, errors 0 and nothing open. Here the real replay meets a sink that fails, and the box arm's real reasons read it.
Re-entry: at a chunk above 1024 rows the policy's plan and what admission really does have to agree. These run the connected yield rig
with a bulk big enough to tell 2048 from 1024, 1024 being the control.
"""
import sys
import threading
import unittest
from unittest import mock

import compose4_arms as T
import compose4_replay as replay
import test_append_yield as ay
import test_compose4_box as tb

CFG = dict(T.REPLAY, AGENTS=2, DURATION=40.0, WARMUP=0.0, START_MIN=120000, START_MAX=120000)


class BrokenSink:
    """Fails from the write numbered `at` on, as a full disk does."""
    def __init__(self, at=1, error=OSError):
        self.at, self.error, self.calls, self.held, self.lock = at, error, 0, [], threading.Lock()

    def write(self, record):
        with self.lock:
            self.calls += 1
            if self.calls >= self.at:
                raise self.error('no space')
            self.held.append(record)


def play(sink=None, stream=None, cfg=CFG):
    clock, prompts = tb.Clock(), []
    return replay.run('http://x', 'm', 'k', 'run', tb.CMP, cfg=cfg, clock=clock, sleep=clock.tick, stream=stream or tb.scripted(clock, prompts), sink=sink)


class JournalFaults(unittest.TestCase):
    def test_a_clean_run_counts_nothing(self):
        _c, summary, turns, still_open = play(sink=BrokenSink(at=10 ** 9))
        self.assertTrue(turns)
        self.assertEqual((summary['journal_write_failures'], summary['agents_died'], summary['fault_kinds'], still_open), (0, 0, [], []))

    def test_every_failed_write_is_counted_and_the_load_goes_on(self):
        sink = BrokenSink(at=3)
        _c, summary, turns, still_open = play(sink=sink)
        self.assertGreater(len(turns), 3, 'the agents kept sending')
        self.assertEqual(sink.calls, len(turns), 'every turn tried the journal')
        self.assertEqual(summary['journal_write_failures'], len(turns) - 2)
        self.assertEqual((summary['agents_died'], summary['fault_kinds'], still_open), (0, ['OSError'], []))

    def test_an_agent_that_dies_is_counted_though_no_thread_is_open(self):
        clock, prompts = tb.Clock(), []
        good = tb.scripted(clock, prompts)
        with mock.patch.object(replay, 'draws', side_effect=KeyError('seed')):
            _c, summary, turns, still_open = replay.run('http://x', 'm', 'k', 'run', tb.CMP, cfg=CFG, clock=clock, sleep=clock.tick, stream=good)
        self.assertEqual((turns, still_open), ([], []), 'without the count this reads as a finished arm')
        self.assertEqual((summary['agents_died'], summary['fault_kinds']), (CFG['AGENTS'], ['KeyError']))

    def test_a_turn_failure_stays_a_turn_error_not_a_fault(self):
        def refused(*_a, **_k):
            raise ConnectionError('refused')
        _c, summary, turns, _o = play(stream=refused, cfg=dict(CFG, DURATION=4.0))
        self.assertTrue(turns)
        self.assertEqual(summary['errors'], len(turns))
        self.assertEqual((summary['journal_write_failures'], summary['agents_died']), (0, 0))

    def test_nothing_of_the_exception_text_reaches_the_summary(self):
        _c, summary, _t, _o = play(sink=BrokenSink(at=1, error=lambda _m: OSError('/secret/path')))
        self.assertNotIn('/secret/path', repr(summary))


class DeadAgentsAndTheArm(unittest.TestCase):
    def test_a_guard_that_itself_fails_still_reads_as_deaths(self):
        with mock.patch.object(replay, '_guarded_agent', side_effect=TypeError('bound wrong')):
            _c, summary, turns, still_open = play()
        self.assertEqual((turns, still_open, summary['agents_died']), ([], [], CFG['AGENTS']))

    def test_the_real_arm_fails_on_both_counts_through_its_reasons(self):
        for name, value, reason in (('journal_write_failures', 3, 'replay-journal-write-failures:3'), ('agents_died', 1, 'replay-agents-died:1')):
            with self.subTest(name):
                self.assertIn(reason, self.arm({name: value})['reasons'])
        clean = self.arm({})
        self.assertEqual((clean['reasons'], clean['complete']), ([], True))

    def arm(self, faults):
        import e3_compose4_box as box
        import os
        import tempfile
        import types

        def run(*_a, sink=None, **_k):
            sink.write(dict(t=1.0, kind='fresh'))
            return dict(T.REPLAY), dict(dict(errors=0, incomplete=0), **faults), [dict(t=1.0, kind='fresh')], []
        status = dict(route='full', scheduler_requests=0, route_generation=3)
        client = types.SimpleNamespace(epoch=1, history=[], call=lambda *_a, **_k: status)
        with tempfile.TemporaryDirectory() as scratch, \
                mock.patch.object(box.shared, '_package_gate', return_value={'ok': True}), \
                mock.patch.object(box, 'entry_status'), mock.patch.object(box, 'select_and_release'), mock.patch.object(box, 'Poller', mock.MagicMock(return_value=mock.MagicMock(rows=[], errors=0))), \
                mock.patch.object(box.progress, 'progress_view', return_value={}), mock.patch.object(box.progress, 'preemption_metrics', return_value={}), mock.patch.object(box.progress, 'delta', return_value=0), mock.patch.object(box.iw, 'stage_view', return_value=None), mock.patch.object(box, 'status_row', return_value={}):
            a = types.SimpleNamespace(run_id='r', url='http://x', model='m', out=os.path.join(scratch, 'arm'))
            return box.replay_arm(a, T.by_label()[tb.CMP], client, client, run=run)


class ReentryPlan(unittest.TestCase):
    """The policy's plan at re-entry against the chunks admission then really admits."""
    TOKENS = 6000

    def resume(self, chunk, driver='native', bound=None):
        real = ay.bridge_suite.Request

        def big(uid=None, tokens=None, **kwargs):
            return real(uid=uid, tokens=self.TOKENS if uid == ay.bridge_suite.UID else tokens, **kwargs)
        with mock.patch.object(ay.bridge_suite, 'Request', big):
            rig = ay.Rig(cached={}, driver=driver)
        rig.c.chunk_tokens = chunk
        rig.c.chunk_bound = bound
        append = ay.run_to_yield(rig)
        ay.to_append(rig)
        ay.run_append(rig)
        ay.to_full(rig)
        rig.stock_handback(append)
        ay.to_resume(rig)
        return rig

    def walk(self, rig):
        sizes = []
        while rig.bulk.num_computed_tokens < rig.bulk.num_tokens - 1:
            before = rig.bulk.num_computed_tokens
            self.assertIsNotNone(rig.admit(), rig.c.last_refusal)
            rig.finish_chunk()
            sizes.append(rig.bulk.num_computed_tokens - before)
        return sizes

    def agrees(self, chunk, driver='native', bound=None):
        rig = self.resume(chunk, driver, bound)
        plan = rig.p._reentry_plan(rig.bulk)['plan']
        self.assertEqual(sum(plan), self.TOKENS - 1 - rig.bulk.num_computed_tokens)
        self.assertEqual(list(plan), self.walk(rig))
        return plan

    def test_at_2048_the_plan_is_what_admission_does(self):
        self.assertEqual(self.agrees(2048), (2048, 1903))

    def test_at_1024_nothing_moved(self):
        self.assertEqual(self.agrees(1024), (1024, 1024, 1024, 1024, 879))

    def test_the_box_driver_too(self):
        self.assertEqual(self.agrees(2048, 'staged'), (2048, 1903))

    def test_with_the_bound_the_plan_is_what_admission_does(self):
        # b13's shape at this rig's size: from 2048 rows in, 2048 to a chunk past 3000, then 1024. The unbounded plan above is the control.
        self.assertEqual(self.agrees(2048, bound=(3000, 1024)), (2048, 1024, 879))

    def test_with_the_bound_on_the_box_driver_too(self):
        self.assertEqual(self.agrees(2048, 'staged', bound=(3000, 1024)), (2048, 1024, 879))

    def test_a_bound_past_the_prompt_changes_nothing(self):
        self.assertEqual(self.agrees(2048, bound=(52624, 1024)), (2048, 1903))


if __name__ == '__main__':
    unittest.main()
