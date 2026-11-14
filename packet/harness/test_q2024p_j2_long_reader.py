"""Long-arm steady-state reader (tools/q2024p_j2_long_reader.py) on synthetic records; hub and bundle, no skip.

  * the primary statistic equals an inline copy of the review's prefix walk (norm.prefix_times + block's prefix_sum)
    on random pairs, and the from-180 s ratio its steady.py 'agentT>180' (the hub run reproduced the review's cohort
    values 1.0591 / 1.0627 / 1.0759 / 1.0510 and steady.py's 1.0240 / 1.0346 / 1.0717 / 1.0210 on the w13 cxay12
    records; READY-long-arms-20261114.md);
  * a divergence stops the walk at that turn; dropped (error) turns are not counted;
  * an engine twice as fast on the same turns reads R = 2 whole, from 180 s and on every slice;
  * curves at 60 s steps up to W (1800 s -> 30 cuts), slices of 300 s (6 at 1800 s, 2 at 480 s);
  * packets: pairs by index, one seed and one length a pair, the CLI writes a fresh JSON and refuses an existing one.
"""
import importlib.util
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
# source tree; the bundle keeps tools/ beside the suites. Loaded by path: sys.path stays as it is (the frozen CPU's
# queue probe imports q2024p_j2_queue later in the same process and must find its own /src/n3tools copy first).
TOOL = next(extra / 'q2024p_j2_long_reader.py' for extra in (HERE.parent / 'tools', HERE / 'tools')
            if (extra / 'q2024p_j2_long_reader.py').exists())
_spec = importlib.util.spec_from_file_location('q2024p_j2_long_reader_under_test', TOOL)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def review_prefix(sp, n3, W):
    """The review's norm.prefix_times, verbatim in logic (K = 0 agents excluded: the review indexes x[-1] there)."""
    def by(turns):
        d = {}
        for t in turns:
            d.setdefault(t['agent'], []).append(t)
        return d
    A, B = by(sp), by(n3)
    res = []
    for a in sorted(A):
        x, y = A[a], B.get(a, [])
        K = 0
        for i in range(min(len(x), len(y))):
            if x[i]['kind'] != y[i]['kind'] or x[i]['reply_target'] != y[i]['reply_target']:
                break
            if x[i]['end'] <= W and y[i]['end'] <= W:
                K = i + 1
            else:
                break
        if K:
            res.append((a, K, x[K - 1]['end'], y[K - 1]['end']))
    return res


def review_steady(sp, n3, W, T0):
    A, B = R.by_agent(sp), R.by_agent(n3)
    ds = dn = 0.0
    for a, K, Ts, Tn in review_prefix(sp, n3, W):
        k0 = next((k for k in range(K) if A[a][k]['end'] > T0 and B[a][k]['end'] > T0), None)
        if k0 is None or k0 >= K - 1:
            continue
        ds += Ts - A[a][k0]['end']
        dn += Tn - B[a][k0]['end']
    return ds / dn


def synth(rng, agents, W, speed=1.0, seed=1, diverge=None, errors=()):
    """Per agent a closed loop of turns; the same draws for both arms (seeded by agent), ``speed`` scales every time."""
    turns = []
    for a in range(agents):
        draw = random.Random(seed * 100 + a)
        t, k = draw.uniform(0, 0.1), 0
        prompt = 0
        while t < W * 1.1:
            k += 1
            kind = 'fresh' if k == 1 or draw.random() < 0.08 else 'append'
            reply = draw.randint(200, 1000)
            if diverge and diverge == (a, k):
                reply += 1
            prompt = draw.randint(60000, 131000) if kind == 'fresh' else prompt + draw.randint(500, 5000)
            ttft = draw.uniform(0.3, 12.0) / speed
            seconds = ttft + draw.uniform(3, 40) / speed
            turn = dict(agent=a, kind=kind, t=round(t, 3), reply_target=reply, ttft=ttft, seconds=seconds, prompt_tokens=prompt,
                        completion_tokens=reply, arrivals_ms=[0, seconds * 500, seconds * 1000])
            if (a, k) in errors:
                turn = dict(agent=a, kind=kind, t=round(t, 3), reply_target=reply, error='TimeoutError:x')
            turns.append(turn)
            t += seconds + draw.uniform(1, 5) / speed
    rng.shuffle(turns)
    return turns


class Statistic(unittest.TestCase):
    def test_matches_the_review_on_random_pairs(self):
        rng = random.Random(7)
        for case in range(12):
            W = (480.0, 1800.0)[case % 2]
            sp = R.prepare(synth(rng, 8, W, speed=1.0, seed=case))
            n3 = R.prepare(synth(rng, 8, W, speed=rng.uniform(0.9, 1.2), seed=case))
            read = R.pair_read(sp, n3, W)
            ref = review_prefix(sp, n3, W)
            self.assertAlmostEqual(read['whole']['R'], sum(x[2] for x in ref) / sum(x[3] for x in ref), places=5)
            self.assertEqual(read['matched_turns'], sum(x[1] for x in ref))
            self.assertAlmostEqual(read['from_180']['R'], review_steady(sp, n3, W, 180), places=5)

    def test_twice_as_fast_reads_two_everywhere(self):
        rng = random.Random(3)
        for W in (480.0, 1800.0):
            sp = R.prepare(synth(rng, 8, W, speed=1.0, seed=5))
            n3 = R.prepare(synth(rng, 8, 2 * W, speed=2.0, seed=5))
            read = R.pair_read(sp, n3, W)
            self.assertAlmostEqual(read['whole']['R'], 2.0, places=2)
            self.assertAlmostEqual(read['from_180']['R'], 2.0, places=2)
            got = [s['time_to_same_turns']['R'] for s in read['slices']]
            self.assertIsNotNone(got[0])
            for r in got:          # a slice holds the matched turns both arms ended inside it: none late at 2x
                if r is not None:
                    self.assertAlmostEqual(r, 2.0, places=2, msg=got)
            self.assertEqual(len(read['slices']), int(W // 300) + (1 if W % 300 else 0))
            self.assertEqual([c['cut'] for c in read['curves']['sp']['arm']][-1], W)
            self.assertEqual(len(read['curves']['n3']['arm']), int(W // 60))

    def test_divergence_and_dropped_turns(self):
        rng = random.Random(4)
        W = 1800.0
        sp = R.prepare(synth(rng, 2, W, seed=9))
        n3 = R.prepare(synth(rng, 2, W, seed=9, diverge=(1, 6)))
        read = R.pair_read(sp, n3, W)
        agents = {a['agent']: a for a in read['agents']}
        self.assertEqual(agents[1]['diverged_at'], 6)
        self.assertEqual(agents[1]['K'], 5)
        self.assertIsNone(agents[0]['diverged_at'])
        self.assertEqual(read['diverged_agents'], 1)
        raw = synth(rng, 2, W, seed=9, errors={(0, 3)})
        self.assertEqual(len(R.prepare(raw)), len(raw) - 1)
        self.assertNotIn('error', json.dumps(R.prepare(raw)))


def write_packet(root, pairs):
    root.mkdir()
    for label, cfg, turns in pairs:
        lines = [json.dumps(dict(label=label, header=True, replay_config=cfg))] + [json.dumps(t) for t in turns]
        (root / ('arm-%s.json.turns.jsonl' % label)).write_text('\n'.join(lines) + '\n')
    return root


class Packets(unittest.TestCase):
    def test_pairs_pool_and_cli(self):
        rng = random.Random(11)
        with tempfile.TemporaryDirectory(prefix='long-reader-') as tmp:
            tmp = Path(tmp)
            cfg = lambda seed: dict(DURATION=1800.0, SEED=seed, AGENTS=8)
            a = write_packet(tmp / 'longa', [('sp.g2-live-0', cfg(1), synth(rng, 8, 1800, seed=1)),
                                             ('n3.g2-live-0', cfg(1), synth(rng, 8, 1800, speed=1.05, seed=1))])
            b = write_packet(tmp / 'longb', [('n3.g2-live-1', cfg(2), synth(rng, 8, 1800, speed=1.05, seed=2)),
                                             ('sp.g2-live-1', cfg(2), synth(rng, 8, 1800, seed=2))])
            result = R.read([a, b])
            self.assertEqual([sorted(p['pairs']) for p in result['packets']], [['0'], ['1']])
            self.assertEqual(result['pooled']['agent_streams'], 16)
            whole = result['pooled']['whole']
            parts = [p['pairs'][i]['whole'] for p, i in zip(result['packets'], ('0', '1'))]
            self.assertAlmostEqual(whole['R'], sum(x['sp_s'] for x in parts) / sum(x['n3_s'] for x in parts), places=5)
            self.assertAlmostEqual(result['G']['whole'], math.sqrt(parts[0]['R'] * parts[1]['R']), places=5)
            self.assertEqual(len(result['pooled']['slices']), 6)
            out = tmp / 'read.json'
            done = subprocess.run([sys.executable, '-I', '-B', str(TOOL), '--packet', str(a), '--packet', str(b), '--json-out', str(out)],
                                  capture_output=True, text=True, timeout=120)
            self.assertEqual(done.returncode, 0, done.stderr[-1500:])
            self.assertIn('pooled: whole', done.stdout)
            self.assertEqual(json.loads(out.read_text())['pooled']['whole'], whole)
            again = subprocess.run([sys.executable, '-I', '-B', str(TOOL), '--packet', str(a), '--json-out', str(out)],
                                   capture_output=True, text=True, timeout=120)
            self.assertNotEqual(again.returncode, 0, 'an existing output is never overwritten')

    def test_a_pair_of_two_seeds_or_lengths_refuses(self):
        rng = random.Random(12)
        with tempfile.TemporaryDirectory(prefix='long-reader-') as tmp:
            tmp = Path(tmp)
            for name, c2 in (('seeds', dict(DURATION=1800.0, SEED=2)), ('lengths', dict(DURATION=480.0, SEED=1))):
                p = write_packet(tmp / name, [('sp.g2-live-0', dict(DURATION=1800.0, SEED=1), synth(rng, 2, 1800, seed=1)),
                                              ('n3.g2-live-0', c2, synth(rng, 2, 1800, seed=1))])
                self.assertRaises(AssertionError, R.packet_read, p)


if __name__ == '__main__':
    unittest.main()
