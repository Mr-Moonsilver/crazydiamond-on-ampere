"""Replay determinism (brief 11 B.1, 14 Nov 2026): compose4_replay's REPLAY_REPLY_TEXT.

* model (default): the request sequence is byte-for-byte the earlier driver's (a sha pinned from 347e459c's
  compose4_replay over the same fake server), no reply_text key in a turn row or the summary;
* seeded: the request sequence of an agent depends on its seed alone: the same under any reply content the model
  returns (tool-call-shaped empty text, long text, random text) and any completion timing (a shorter run is a prefix
  of a longer one); the start/append/reply/pause/text draws are unchanged (seeded = a model that replied exactly the
  synthetic text); the forced decode length is the drawn reply; an unknown mode is refused before the agents start;
* mutants: the model text kept, the synthetic text from the shared text stream, a fixed synthetic length, the mode
  ignored; each caught.
CPU only, fake server and clock; no engine, network or box.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import agent_replay as ar  # noqa: E402
import compose4_arms as T  # noqa: E402
import compose4_replay as rp  # noqa: E402

CFG = dict(T.REPLAY, DURATION=480.0, WARMUP=0.0, SEED=2)
# sha256 over the request bodies of agent 5, seed 2 (model mode) with the fake server below, from 347e459c's
# compose4_replay.py (the driver before the reply-text setting).
PARENT_SHA = '271cc244486ff50feeee06745e9b5ef9fcb6a818a1a82a114944f12be957732d'


def tokens_of(messages):
    """The fake server's prompt count: a function of the history's text only (as the real tokenizer is)."""
    return int(sum(len(m['content'].split()) for m in messages) * ar.TOKENS_PER_WORD) + 7 * len(messages)


class Server:
    """A fake engine. ``reply(turn, body)`` gives the model's text; ``duration(turn)`` the turn's seconds."""

    def __init__(self, clock, reply=None, duration=None):
        self.clock, self.turn, self.bodies = clock, 0, []
        self.reply = reply or (lambda turn, body: 'ok %d' % turn)
        self.duration = duration or (lambda turn: 3.0)

    def __call__(self, url, body, key, origin):
        self.turn += 1
        self.bodies.append(json.loads(json.dumps(body)))
        seconds = self.duration(self.turn)
        start = self.clock.now - origin
        self.clock.now += seconds
        n = body['max_tokens']
        return dict(text=self.reply(self.turn, body), ttft=seconds / 3, seconds=seconds, completion_tokens=n,
                    prompt_tokens=tokens_of(body['messages']), cached_tokens=0, finish='length',
                    arrivals_ms=[int((start + seconds / 3) * 1000) + k for k in range(3)])


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s


def agent_run(module=rp, agent=5, reply_text='model', reply=None, duration=None, cfg=CFG):
    clock = Clock()
    server = Server(clock, reply, duration)
    records = []
    kwargs = dict(clock=clock, sleep=clock.sleep, stream=server)
    if reply_text != 'model':
        kwargs['reply_text'] = reply_text
    module.agent_loop(agent, 'u', 'm', None, cfg, 'run', 'sp.g2-live-1', clock.now, clock.now + cfg['DURATION'],
                      records, threading.Lock(), **kwargs)
    return server.bodies, records


def digest(bodies):
    return hashlib.sha256(json.dumps(bodies, sort_keys=True).encode()).hexdigest()


def prefix_equal(a, b):
    n = min(len(a), len(b))
    return n > 0 and a[:n] == b[:n]


TOOLISH = lambda turn, body: '' if turn % 3 == 0 else ' '.join(['word'] * (turn * 37 % 900))   # noqa: E731
NOISY = lambda turn, body: ' '.join(random.Random(turn * 7919).choice(('a', 'bb', 'ccc')) for _ in range(turn * 11 % 700))  # noqa: E731
FAST = lambda turn: 1.0 + turn % 4      # noqa: E731
SLOW = lambda turn: 6.0 + turn % 5      # noqa: E731


def seeded_holds(module=rp):
    """Same seed: the same request sequence whatever the model returns and whenever it completes."""
    runs = [agent_run(module, reply_text='seeded', reply=r, duration=d)[0]
            for r, d in ((None, None), (TOOLISH, FAST), (NOISY, SLOW), (TOOLISH, SLOW))]
    return all(prefix_equal(runs[0], x) for x in runs[1:]) and len({len(x) for x in runs}) > 1


def echo_holds(module=rp):
    """Seeded equals a model that replied exactly the seeded synthetic text: the other draws are untouched."""
    def echo_server():
        streams = {}

        def reply(turn, body):
            rng = streams.setdefault('r', random.Random('%d|%d|%s' % (CFG['SEED'], 5, 'reply_text')))
            return ar.text(rng, body['max_tokens'])
        return reply
    seeded = agent_run(module, reply_text='seeded')[0]
    echoed = agent_run(module, reply_text='model', reply=echo_server())[0]
    return seeded == echoed


def load_mutant(*edits):
    source = Path(rp.__file__).read_text()
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    module = types.ModuleType('compose4_replay_mutant')
    module.__file__ = rp.__file__
    exec(compile(source, 'compose4_replay_mutant', 'exec'), module.__dict__)
    return module


class ModelMode(unittest.TestCase):
    def test_default_is_the_parent_driver_byte_for_byte(self):
        bodies, records = agent_run()
        self.assertGreater(len(bodies), 10)
        self.assertEqual(digest(bodies), PARENT_SHA)
        self.assertFalse(any('reply_text' in r for r in records))

    def test_model_mode_follows_the_model_text(self):
        """The control: with the model's text in the history, reply content moves the rolls (the w13 bimodality)."""
        plain = agent_run(reply=TOOLISH, duration=FAST)[0]
        noisy = agent_run(reply=NOISY, duration=FAST)[0]
        self.assertFalse(prefix_equal(plain, noisy))

    def test_mode_from_the_environment(self):
        self.assertEqual(rp.reply_text_mode({}), 'model')
        self.assertEqual(rp.reply_text_mode({rp.REPLY_TEXT_ENV: ''}), 'model')
        self.assertEqual(rp.reply_text_mode({rp.REPLY_TEXT_ENV: 'seeded'}), 'seeded')
        for bad in ('1', 'on', 'Seeded', 'synthetic'):
            with self.assertRaises(ValueError):
                rp.reply_text_mode({rp.REPLY_TEXT_ENV: bad})


class SeededMode(unittest.TestCase):
    def test_the_seed_alone_sets_the_request_sequence(self):
        self.assertTrue(seeded_holds())

    def test_other_draws_and_the_forced_length_unchanged(self):
        self.assertTrue(echo_holds())
        bodies, records = agent_run(reply_text='seeded', reply=TOOLISH)
        rng = random.Random('%d|%d|%s' % (CFG['SEED'], 5, 'reply'))
        want = [rng.randint(CFG['REPLY_MIN'], CFG['REPLY_MAX']) for _ in bodies]
        self.assertEqual([b['max_tokens'] for b in bodies], want)
        self.assertEqual([b['min_tokens'] for b in bodies], want)
        self.assertTrue(records and all(r['reply_text'] == 'seeded' for r in records))

    def test_every_agent_and_both_seeds(self):
        for seed in (1, 2):
            cfg = dict(CFG, SEED=seed)
            for agent in range(8):
                a = agent_run(agent=agent, reply_text='seeded', reply=TOOLISH, duration=FAST, cfg=cfg)[0]
                b = agent_run(agent=agent, reply_text='seeded', reply=NOISY, duration=SLOW, cfg=cfg)[0]
                self.assertTrue(prefix_equal(a, b), (seed, agent))

    def test_run_reads_the_mode_once_and_records_it(self):
        clock = Clock()
        lock = threading.Lock()
        servers = []

        def stream(url, body, key, origin):
            with lock:
                if not servers:
                    servers.append(Server(clock))
                return servers[0](url, body, key, origin)
        cfg = dict(CFG, DURATION=4.0, AGENTS=2)
        with mock.patch.dict(os.environ, {rp.REPLY_TEXT_ENV: 'seeded'}):
            _, summary, turns, still = rp.run('u', 'm', None, 'run', 'x', cfg=cfg, clock=clock, sleep=clock.sleep,
                                              stream=stream, finish_bound_s=1.0)
        self.assertEqual(summary.get('reply_text'), 'seeded')
        self.assertTrue(turns and all(t.get('reply_text') == 'seeded' for t in turns))
        with mock.patch.dict(os.environ, {rp.REPLY_TEXT_ENV: ''}):
            _, summary, turns, _ = rp.run('u', 'm', None, 'run', 'x', cfg=cfg, clock=Clock(), sleep=lambda s: None,
                                          stream=lambda *a: (_ for _ in ()).throw(OSError('down')), finish_bound_s=0.0)
        self.assertNotIn('reply_text', summary)
        with mock.patch.dict(os.environ, {rp.REPLY_TEXT_ENV: 'on'}):
            with self.assertRaises(ValueError):
                rp.run('u', 'm', None, 'run', 'x', cfg=cfg, clock=Clock(), sleep=lambda s: None, stream=stream)


class Mutants(unittest.TestCase):
    def test_model_text_kept_is_caught(self):
        m = load_mutant(("got['text'] if synthetic is None else synthetic", "got['text']"))
        self.assertFalse(seeded_holds(m))

    def test_synthetic_from_the_shared_text_stream_is_caught(self):
        m = load_mutant(("ar.text(rng['reply_text'], reply)", "ar.text(rng['text'], reply)"))
        self.assertTrue(seeded_holds(m))          # still seed-only, but it moves the tool outputs and starts:
        self.assertFalse(echo_holds(m))

    def test_fixed_synthetic_length_is_caught(self):
        m = load_mutant(("ar.text(rng['reply_text'], reply)", "ar.text(rng['reply_text'], 500)"))
        self.assertFalse(echo_holds(m))

    def test_mode_ignored_is_caught(self):
        m = load_mutant(("    mode = reply_text_mode()\n", "    mode = 'model'\n"))
        clock = Clock()
        with mock.patch.dict(os.environ, {rp.REPLY_TEXT_ENV: 'seeded'}):
            _, summary, turns, _ = m.run('u', 'm', None, 'run', 'x', cfg=dict(CFG, DURATION=4.0, AGENTS=1), clock=clock,
                                         sleep=clock.sleep, stream=Server(clock), finish_bound_s=1.0)
        self.assertNotIn('reply_text', summary)   # what test_run_reads_the_mode_once_and_records_it refuses


if __name__ == '__main__':
    unittest.main()
