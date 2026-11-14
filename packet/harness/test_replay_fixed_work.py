"""Fixed work (brief 12, 14 Nov 2026): compose4_replay's REPLAY_FIXED_WORK / REPLAY_FIXED_CAP_S.

* off (default): agent_loop and run() send exactly what they sent before (the model-mode sha of
  test_replay_reply_text, and the seeded sequence of the same driver without the setting); no fixed_work key;
* on: every agent keeps sending until the LAST agent has completed its own target (agents that got there early keep
  sending, so the load stays at AGENTS); then no new turn starts; each agent's time to its target is recorded; past the
  cap nothing is sent and cap_hit is True, which e3_compose4_box turns into an arm reason (the arm is not complete);
  a turn with an error does not count;
* refused before the first agent: no seeded reply text, a malformed spec, a seed without targets, a cap at or below
  DURATION, a cap without a spec, a spec without a cap;
* the first N requests of an agent are the same under any completion timing (the seeded property, under fixed work);
* mutants: the stop ignored, the stop at the first agent's target, errors counted, the cap ignored; each caught.
CPU only, fake server on a virtual clock shared by the real agent threads; no engine, network or box.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compose4_arms as T  # noqa: E402
import compose4_replay as rp  # noqa: E402
import test_replay_reply_text as rt  # noqa: E402

# Virtual time (SimClock): the agents are real threads, but the clock moves only when every live agent thread (and
# the starter, until every agent has started) waits in sleep() or in a fake turn, and then to the earliest wake. So a
# run is exact and does not depend on host load (two frozen CPU runs under parallel containers failed a real-time rig).
# Seconds are virtual: DURATION 60 s, turns 1-3 s, tool pauses compose4's 1-5 s. EPS: the 3-decimal rounding of stamps.
CFG = dict(T.REPLAY, DURATION=60.0, WARMUP=0.0, SEED=2, AGENTS=3)
SPEC = '1=4,5,6/2=2,14,4'
TARGETS = [2, 14, 4]
CAP = 400.0
EPS = 0.002
CYCLE = 3.0 + CFG['TOOL_MAX']          # the longest turn plus the longest tool pause
PREFIX = 'compose4-agent-'


class SimClock:
    """Shared virtual clock for the driver's agent threads (see above). ``close()`` stops its ticker."""

    def __init__(self, agents):
        self.now, self.agents = 1000.0, agents
        self.cond = threading.Condition()
        self.waits, self.members = {}, set()
        self.main = threading.current_thread()
        self.stale = {t for t in threading.enumerate() if t.name.startswith(PREFIX)}
        self.done = threading.Event()
        self.ticker = threading.Thread(target=self._tick, name='simclock-ticker', daemon=True)
        self.ticker.start()

    def _join(self):
        me = threading.current_thread()
        if me.name.startswith(PREFIX) and me not in self.members:
            self.members.add(me)
            self.cond.notify_all()

    def __call__(self):
        with self.cond:
            self._join()
            return self.now

    def sleep(self, seconds):
        me = threading.current_thread()
        with self.cond:
            self._join()
            if seconds <= 0:
                return
            self.waits[me] = self.now + seconds
            self.cond.notify_all()
            while me in self.waits:
                self.cond.wait(0.05)

    def _quiet(self):
        live = [t for t in threading.enumerate() if t.name.startswith(PREFIX) and t not in self.stale and t.is_alive()]
        if not self.waits or any(t not in self.members or t not in self.waits for t in live):
            return False
        return len(self.members) >= self.agents or self.main in self.waits

    def _tick(self):
        while not self.done.is_set():
            with self.cond:
                if self._quiet():
                    self.now = min(self.waits.values())
                    for th in [th for th, w in self.waits.items() if w <= self.now]:
                        del self.waits[th]
                    self.cond.notify_all()
                else:
                    self.cond.wait(0.0005)

    def close(self):
        self.done.set()
        self.ticker.join(5.0)


class Shared:
    """A fake engine for every agent thread on the SimClock: a turn takes ``duration(turn)`` virtual seconds."""

    def __init__(self, clock, duration=None, fail=None):
        self.clock, self.lock, self.turn, self.bodies = clock, threading.Lock(), 0, []
        self.duration = duration or (lambda turn: 1.0 + 2.0 * (turn % 2))
        self.fail = fail or (lambda turn, body: False)

    def __call__(self, url, body, key, origin):
        with self.lock:
            self.turn += 1
            turn = self.turn
            self.bodies.append(json.loads(json.dumps(body)))
        start = self.clock()
        self.clock.sleep(self.duration(turn))
        if self.fail(turn, body):
            raise OSError('reset')
        seconds = self.clock() - start
        n = body['max_tokens']
        return dict(text='ok %d' % turn, ttft=seconds / 3, seconds=seconds, completion_tokens=n,
                    prompt_tokens=rt.tokens_of(body['messages']), cached_tokens=0, finish='length',
                    arrivals_ms=[int((start - origin + seconds / 3) * 1000) + k for k in range(3)])


def run_with(env, module=rp, cfg=CFG, duration=None, fail=None):
    clock = SimClock(cfg['AGENTS'])
    stream = Shared(clock, duration, fail)
    try:
        with mock.patch.dict(os.environ, env, clear=False):
            for name in (rp.FIXED_WORK_ENV, rp.FIXED_CAP_ENV, rp.REPLY_TEXT_ENV):
                if name not in env:
                    os.environ.pop(name, None)
            out = module.run('u', 'm', None, 'run', 'sp.g2-live-0', cfg=cfg, clock=clock, sleep=clock.sleep,
                             stream=stream, finish_bound_s=1.0)
    finally:
        clock.close()
    return out, stream, clock


def bodies_of(stream, agent):
    return [b for b in stream.bodies if b['messages'][0]['content'].startswith('You are coding agent %d,' % agent)]


ON = {rp.REPLY_TEXT_ENV: 'seeded', rp.FIXED_WORK_ENV: SPEC, rp.FIXED_CAP_ENV: str(CAP)}


def per_agent(turns):
    out = {}
    for t in sorted(turns, key=lambda r: r['t']):
        out.setdefault(t['agent'], []).append(t)
    return out


def done_at(rows):
    """Each completed (error-free) turn's end time, in order."""
    return [round(r['t'] + r['seconds'], 3) for r in rows if 'error' not in r]


class Off(unittest.TestCase):
    def test_model_mode_is_the_parent_driver(self):
        bodies, records = rt.agent_run()
        self.assertEqual(rt.digest(bodies), rt.PARENT_SHA)

    def test_seeded_without_the_setting_is_seeded_today(self):
        self.assertIsNone(rp.fixed_work_targets(CFG, environ={}, reply_text='seeded'))
        (_cfg, summary, turns, _still), _s, _c = run_with({rp.REPLY_TEXT_ENV: 'seeded'})
        self.assertNotIn('fixed_work', summary)
        self.assertTrue(max(t['t'] for t in turns) < CFG['DURATION'])
        # agent_loop with fixed=None is the seeded loop: same bodies as the default call
        a, _ = rt.agent_run(reply_text='seeded')
        clock = rt.Clock()
        server = rt.Server(clock)
        rp.agent_loop(5, 'u', 'm', None, rt.CFG, 'run', 'sp.g2-live-1', clock.now, clock.now + rt.CFG['DURATION'], [],
                      threading.Lock(), clock=clock, sleep=clock.sleep, stream=server, reply_text='seeded', fixed=None)
        self.assertEqual(rt.digest(a), rt.digest(server.bodies))

    def test_off_run_is_unchanged_in_model_mode(self):
        (_cfg, summary, turns, _still), _s, _c = run_with({})
        self.assertNotIn('fixed_work', summary)
        self.assertNotIn('reply_text', summary)
        self.assertTrue(turns)


class On(unittest.TestCase):
    def assert_reached(self, reached, end):
        """The driver stamps the target when the turn returns: the row's own end (t + seconds), in virtual time exact
        up to the 3-decimal rounding of both stamps."""
        self.assertAlmostEqual(reached, end, delta=EPS)

    def test_all_agents_send_until_the_last_target_then_stop(self):
        (_cfg, summary, turns, still), stream, clock = run_with(ON)
        self.assertEqual(still, [])
        fw = summary['fixed_work']
        self.assertEqual((fw['targets'], fw['cap_s'], fw['all_reached'], fw['cap_hit']), (TARGETS, CAP, True, False))
        rows = per_agent(turns)
        self.assertEqual(sorted(rows), [0, 1, 2])
        for agent, target in enumerate(TARGETS):
            ends = done_at(rows[agent])
            self.assertGreaterEqual(len(ends), target)
            self.assert_reached(fw['reached_s'][agent], ends[target - 1])
        self.assertEqual(fw['end_s'], max(fw['reached_s']))
        self.assertGreater(fw['end_s'], CFG['DURATION'])          # it ran past DURATION to finish the work
        # early agents kept sending: every agent that got there well before the last started turns after its own
        # target and up to the stop
        early = [a for a in range(3) if fw['reached_s'][a] < fw['end_s'] - 2 * CYCLE]
        self.assertTrue(early, fw)
        for agent in early:
            after = [r['t'] for r in rows[agent] if r['t'] > fw['reached_s'][agent]]
            self.assertTrue(after and max(after) >= fw['end_s'] - CYCLE, (agent, after, fw))
        # and no turn starts after the stop
        self.assertTrue(all(t['t'] <= fw['end_s'] + EPS for t in turns), fw)
        self.assertEqual(fw['completed'], [len(done_at(rows[a])) for a in range(3)])

    def test_the_cap_ends_the_arm_and_is_marked(self):
        env = dict(ON, **{rp.FIXED_WORK_ENV: '2=3,90,5', rp.FIXED_CAP_ENV: '120'})
        (_cfg, summary, turns, still), _s, _c = run_with(env)
        fw = summary['fixed_work']
        self.assertEqual((fw['all_reached'], fw['cap_hit'], fw['end_s']), (False, True, None))
        self.assertIsNone(fw['reached_s'][1])
        self.assertTrue(all(t['t'] < 120.0 for t in turns))
        self.assertEqual(still, [])
        self.assertGreater(max(t['t'] for t in turns), CFG['DURATION'])

    def test_an_error_turn_does_not_count(self):
        (_cfg, summary, turns, _still), _s, _c = run_with(ON, fail=lambda turn, body: turn % 4 == 0)
        fw = summary['fixed_work']
        rows = per_agent(turns)
        self.assertTrue(any('error' in r for r in turns))
        self.assertEqual(fw['completed'], [len(done_at(rows[a])) for a in range(3)])
        for agent, target in enumerate(TARGETS):
            self.assert_reached(fw['reached_s'][agent], done_at(rows[agent])[target - 1])

    def test_the_first_n_requests_are_timing_free(self):
        bodies = []
        for duration in (None, lambda turn: 0.3, lambda turn: 2.0 + 1.0 * (turn % 3)):
            _out, stream, _c = run_with(ON, duration=duration)
            mine = bodies_of(stream, 1)
            bodies.append(mine[:TARGETS[1]])
        self.assertTrue(all(len(b) == TARGETS[1] for b in bodies))
        self.assertEqual(bodies[0], bodies[1])
        self.assertEqual(bodies[0], bodies[2])

    def test_refusals(self):
        cfg = dict(CFG)
        bad = [({rp.FIXED_WORK_ENV: SPEC, rp.FIXED_CAP_ENV: '400'}, 'model'),
               ({rp.FIXED_WORK_ENV: '2=3,9', rp.FIXED_CAP_ENV: '400'}, 'seeded'),
               ({rp.FIXED_WORK_ENV: '2=3,9,x', rp.FIXED_CAP_ENV: '400'}, 'seeded'),
               ({rp.FIXED_WORK_ENV: '2=3,0,5', rp.FIXED_CAP_ENV: '400'}, 'seeded'),
               ({rp.FIXED_WORK_ENV: '2=3,9,5/2=3,9,5', rp.FIXED_CAP_ENV: '400'}, 'seeded'),
               ({rp.FIXED_WORK_ENV: '3,9,5', rp.FIXED_CAP_ENV: '400'}, 'seeded'),
               ({rp.FIXED_WORK_ENV: '1=3,9,5', rp.FIXED_CAP_ENV: '400'}, 'seeded'),
               ({rp.FIXED_WORK_ENV: SPEC, rp.FIXED_CAP_ENV: '60'}, 'seeded'),
               ({rp.FIXED_WORK_ENV: SPEC, rp.FIXED_CAP_ENV: ''}, 'seeded'),
               ({rp.FIXED_WORK_ENV: SPEC, rp.FIXED_CAP_ENV: 'soon'}, 'seeded'),
               ({rp.FIXED_CAP_ENV: '400'}, 'seeded')]
        for env, mode in bad:
            with self.subTest(env=env, mode=mode):
                with self.assertRaises(ValueError):
                    rp.fixed_work_targets(cfg, environ=env, reply_text=mode)
        self.assertEqual(rp.fixed_work_targets(cfg, environ=ON, reply_text='seeded'), (TARGETS, CAP))
        # run() refuses before any agent sends
        clock = rt.Clock()
        sent = []
        with mock.patch.dict(os.environ, {rp.FIXED_WORK_ENV: SPEC, rp.FIXED_CAP_ENV: '400', rp.REPLY_TEXT_ENV: ''}):
            with self.assertRaises(ValueError):
                rp.run('u', 'm', None, 'run', 'x', cfg=cfg, clock=clock, sleep=clock.sleep, stream=lambda *a: sent.append(a))
        self.assertEqual(sent, [])

    def test_the_production_spec_parses_for_both_seeds(self):
        spec = '1=15,17,14,17,18,14,12,14/2=18,15,16,17,15,16,16,16'
        for seed in (1, 2):
            targets, cap = rp.fixed_work_targets(dict(T.REPLAY, SEED=seed, DURATION=480.0), environ={rp.FIXED_WORK_ENV: spec, rp.FIXED_CAP_ENV: '720'}, reply_text='seeded')
            self.assertEqual((len(targets), cap), (T.REPLAY['AGENTS'], 720.0))


class BoxReason(unittest.TestCase):
    """e3_compose4_box.replay_arm: a cap hit is a reason (the arm is not complete); a reached target is not."""

    def arm(self, summary):
        import e3_compose4_box as box
        status = dict(route='full', scheduler_requests=0)
        client = types.SimpleNamespace(history=[], call=lambda *_a, **_k: status)
        with tempfile.TemporaryDirectory() as tmp:
            a = types.SimpleNamespace(run_id='r', url='http://x', model='m', out=os.path.join(tmp, 'arm.json'))
            run = lambda *a, **k: (dict(T.REPLAY), dict(summary), [], [])  # noqa: E731
            poller = types.SimpleNamespace(start=lambda: None, join=lambda s: None, done=threading.Event(), rows=[], errors=0)
            with mock.patch.object(box.shared, '_package_gate', return_value={'ok': True}), \
                    mock.patch.object(box, 'entry_status', lambda *a: None), \
                    mock.patch.object(box, 'select_and_release', lambda *a: None), \
                    mock.patch.object(box, 'Poller', lambda c: poller), \
                    mock.patch.object(box.progress, 'preemption_metrics', lambda url: {}), \
                    mock.patch.object(box, 'base_record', lambda a, row, c: dict(reasons=[], statuses=[], progress=[])), \
                    mock.patch.object(box, 'status_row', lambda label, s: dict(label=label)):
                return box.replay_arm(a, T.by_label()[T.arms()[0]['label']], client, client, run=run)

    def test_cap_hit_is_a_reason(self):
        rec = self.arm(dict(fixed_work=dict(targets=[3, 9], reached_s=[10.0, None], cap_hit=True, all_reached=False, end_s=None)))
        self.assertIn('replay-fixed-work-cap-hit:1-of-2-reached', rec['reasons'])
        self.assertFalse(rec['complete'])

    def test_reached_and_off_are_not(self):
        rec = self.arm(dict(fixed_work=dict(targets=[3, 9], reached_s=[10.0, 50.0], cap_hit=False, all_reached=True, end_s=50.0)))
        self.assertEqual(rec['reasons'], [])
        self.assertTrue(rec['complete'])
        rec = self.arm({})
        self.assertEqual(rec['reasons'], [])

    def test_a_summary_without_the_flag_is_a_reason(self):
        rec = self.arm(dict(fixed_work=dict(targets=[3, 9], reached_s=[10.0, 50.0])))
        self.assertIn('replay-fixed-work-cap-hit:2-of-2-reached', rec['reasons'])


class Mutants(unittest.TestCase):
    def test_stop_ignored_is_caught(self):
        m = rt.load_mutant(("while clock() < t_end and (fixed is None or not fixed.stop.is_set()):", "while clock() < t_end:"))
        (_cfg, summary, turns, _still), _s, _c = run_with(ON, module=m)
        fw = summary['fixed_work']
        self.assertFalse(all(t['t'] <= fw['end_s'] + EPS for t in turns))   # what test_all_agents_send_until_... refuses

    def test_stop_at_the_first_target_is_caught(self):
        m = rt.load_mutant(("                if all(r is not None for r in self.reached) and self.end_s is None:",
                            "                if self.end_s is None:"))
        (_cfg, summary, _turns, _still), _s, _c = run_with(ON, module=m)
        fw = summary['fixed_work']
        self.assertTrue(None in fw['reached_s'] or fw['end_s'] != max(fw['reached_s']))

    def test_errors_counted_is_caught(self):
        m = rt.load_mutant(("            messages = []\n        else:\n            if fixed is not None:\n                fixed.completed(agent)\n",
                            "            messages = []\n        if fixed is not None:\n            fixed.completed(agent)\n"))
        (_cfg, summary, turns, _still), _s, _c = run_with(ON, module=m, fail=lambda turn, body: turn % 4 == 0)
        rows = per_agent(turns)
        self.assertNotEqual(summary['fixed_work']['completed'], [len(done_at(rows[a])) for a in range(3)])

    def test_cap_ignored_is_caught(self):
        m = rt.load_mutant(("    t_end = origin + (cfg['DURATION'] if work is None else work[1])\n",
                            "    t_end = origin + cfg['DURATION']\n"))
        env = dict(ON, **{rp.FIXED_WORK_ENV: '2=3,90,5', rp.FIXED_CAP_ENV: '120'})
        (_cfg, summary, turns, _still), _s, _c = run_with(env, module=m)
        self.assertFalse(max(t['t'] for t in turns) >= CFG['DURATION'])     # the work is cut at DURATION, not run on

    def test_box_reason_dropped_is_caught(self):
        import e3_compose4_box as box
        source = Path(box.__file__).read_text()
        self.assertEqual(source.count("record['reasons'].append('replay-fixed-work-cap-hit:"), 1)
        self.assertEqual(source.count("if fixed is not None and fixed.get('cap_hit') is not False:"), 1)


if __name__ == '__main__':
    unittest.main()
