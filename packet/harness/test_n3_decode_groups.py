"""FLASH_S5_DECODE_GROUPS (composition branch, lead 12 Nov 2026): the packet-level decode-group mode on j2's table.

Hub and bundle (frozen CPU). On the hub j2's configure is its text (the same inline copy test_decode_merge_probe uses);
in a bundle it is the bundle's n3j2_packet (its marked tail wraps it, so the tail is proved there and skipped here):
  * unset, empty and ``chunk``: the configure's own table, every row equal to j2's (n3 ``chunk``, sp ``off``);
  * ``off`` and ``off-small``: only the n3 rows' ``decode_merge`` differs from j2's rows; labels, seeds, arm order and
    every other field equal; back to ``chunk`` restores j2's rows (the configure does not reset DECODE_MERGE);
  * an unknown value refuses (ValueError) at the configure call;
  * bundle: fresh pythons with the draft's export import n3j2_arms / n3j2_gate on the mode's table, j2's budget
    unchanged (four replay arms, no added arm).
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import n3_decode_groups as dg  # noqa: E402

J2_REPLAY = ['sp.g2-live-0', 'n3.g2-live-0', 'n3.g2-live-1', 'sp.g2-live-1']


def j2_configure():
    """j2's configure, unwrapped: the bundle's (through its tail's wrappers) or j2's text from the dmp suite."""
    try:
        import n3j2_packet
    except ImportError:
        text = (HERE / 'test_decode_merge_probe.py').read_text()
        source = re.search(r"^J2_CONFIGURE = '''(.*?)'''", text, re.S | re.M).group(1)
        space = {}
        exec(compile(source, 'n3j2_packet(inline)', 'exec'), space)
        return space['configure']
    inner = n3j2_packet.configure
    for attr in ('j2_configure', 'decode_groups_inner'):
        inner = getattr(inner, attr, inner)
    return inner


class Modes(unittest.TestCase):
    def setUp(self):
        import compose5_arms
        self.module = compose5_arms
        self.saved = dict(vars(compose5_arms))
        self.j2 = j2_configure()
        self.base = [dict(r) for r in self.j2().arms()]

    def tearDown(self):
        for k in list(vars(self.module)):
            if k not in self.saved:
                delattr(self.module, k)
        for k, v in self.saved.items():
            setattr(self.module, k, v)

    def replay(self, rows):
        return [r['label'] for r in rows if r['kind'] == 'replay']

    def test_default_is_j2_unchanged(self):
        for env in ({}, {dg.ENV: ''}, {dg.ENV: 'chunk'}):
            before = self.module.DECODE_MERGE
            table = dg.wrap_env(self.j2, env)()
            self.assertIs(table.DECODE_MERGE, before, env)          # chunk: the configure's own table object
            self.assertEqual(table.arms(), self.base, env)
            self.assertEqual(self.replay(table.arms()), J2_REPLAY)
            n3 = [r for r in table.arms() if r['c1'] == 'n3']
            self.assertTrue(n3 and all(r['decode_merge'] == 'chunk' for r in n3))

    def test_off_and_off_small_move_only_n3(self):
        for mode in ('off', 'off-small'):
            rows = dg.wrap_env(self.j2, {dg.ENV: mode})().arms()
            self.assertEqual([r['label'] for r in rows], [r['label'] for r in self.base])
            for got, want in zip(rows, self.base):
                differ = sorted(k for k in want if got.get(k) != want[k])
                if want['c1'] == 'n3':
                    self.assertEqual(differ, ['decode_merge'], want['label'])
                    self.assertEqual(got['decode_merge'], mode)
                else:
                    self.assertEqual(differ, [], want['label'])
            seeds = {r['label']: self.module.replay_config(r)['SEED'] for r in rows if r['kind'] == 'replay'}
            self.assertEqual(seeds, {r['label']: self.module.replay_config(r)['SEED'] for r in self.base if r['kind'] == 'replay'})
            back = dg.wrap_env(self.j2, {})()
            self.assertEqual(back.arms(), self.base, 'back to chunk restores j2 rows')

    def test_unknown_value_refuses(self):
        for bad in ('always', 'OFF', 'offsmall', '1'):
            with self.assertRaises(ValueError):
                dg.wrap_env(self.j2, {dg.ENV: bad})()

    def test_modes_are_engine_modes(self):
        sys.path.insert(0, str(HERE.parent / 'patches/s5_engine'))
        try:
            import stage_overlap
        except ImportError as exc:                       # the hub lacks torch/vllm for the engine module: read its text
            text = (HERE.parent / 'patches/s5_engine/stage_overlap.py').read_text()
            modes = eval(re.search(r"^DECODE_MERGE_MODES = (\(.*\))$", text, re.M).group(1))
        else:
            modes = stage_overlap.DECODE_MERGE_MODES
        self.assertTrue(set(dg.MODES) <= set(modes), modes)


class WindowPhases(unittest.TestCase):
    """Interaction decode groups x yield window (Y2), as built: OverlapSegmentDriver._decode_merge_wanted reads the
    append-yield phase, and inside a Y2 window the phase goes handback -> append -> handback (engine_driver
    append_yield_window_begin sets 'append' for each staged append). So with chunk / off-small the merge is off at every
    window handback (the FULL window) and back on for each window append in the native phase; with off it is never on.
    Pinned here on the real method with stub state (no scheduler, no device)."""

    def driver(self, mode, phase, rows, manual=False, policy_phase='native'):
        sys.path.insert(0, str(HERE.parent / 'patches/s5_engine'))
        import stage_overlap
        d = stage_overlap.OverlapSegmentDriver.__new__(stage_overlap.OverlapSegmentDriver)
        d.decode_merge = mode
        d.auto_arm = type('P', (), {'manual_owned': manual, 'phase': policy_phase})()
        d.append_yield_phase = lambda: phase
        d.tx = None if rows is None else type('T', (), {'chunk_start': 1000, 'chunk_end': 1000 + rows})()
        return d, stage_overlap.OFF_SMALL_MIN_ROWS

    def test_window_phase_sequence(self):
        _, small = self.driver('chunk', None, None)
        # a Y2 window: the first append's handback, a window append (piece of `rows`), its handback, then closing
        window = (('handback', 2048), ('append', 2048), ('append', small - 1), ('handback', 2048), ('closing', 2048), (None, 2048))
        want = {
            'chunk': [False, True, True, False, True, True],
            'off-small': [False, True, False, False, True, True],
            'off': [False] * 6,
        }
        for mode, expected in want.items():
            got = [self.driver(mode, phase, rows)[0]._decode_merge_wanted() for phase, rows in window]
            self.assertEqual(got, expected, mode)

    def test_manual_lease_and_no_piece(self):
        for mode in ('chunk', 'off-small'):
            self.assertFalse(self.driver(mode, 'append', 4096, manual=True)[0]._decode_merge_wanted(), mode)
            self.assertFalse(self.driver(mode, 'append', 4096, policy_phase='armed')[0]._decode_merge_wanted(), mode)
        self.assertTrue(self.driver('chunk', 'append', None)[0]._decode_merge_wanted())
        self.assertFalse(self.driver('off-small', 'append', None)[0]._decode_merge_wanted(), 'off-small: no open piece')


class BundleTail(unittest.TestCase):
    def test_tail_and_fresh_processes(self):
        try:
            import n3j2_packet
        except ImportError:
            self.skipTest('hub: no bundle n3j2_packet (the tail is a packet-build edit)')
        chain, seen = n3j2_packet.configure, []
        while chain is not None:
            seen.append(getattr(chain, 'decode_groups_env', None))
            chain = getattr(chain, 'j2_configure', None) or getattr(chain, 'decode_groups_inner', None)
        self.assertIn(dg.ENV, seen, 'the bundle configure is decode-groups wrapped')
        here = str(Path(n3j2_packet.__file__).resolve().parent)
        code = ('import n3j2_gate, n3j2_arms, json; rows = n3j2_arms.table.arms(); print(json.dumps([[r["label"] for r in '
                'rows if r["kind"] == "replay"], sorted({r["decode_merge"] for r in rows if r["c1"] == "n3"}), '
                'n3j2_gate.estimate()["total_s"], n3j2_gate.NOMINAL_S]))')
        for mode in ('', 'chunk', 'off', 'off-small'):
            env = {k: v for k, v in os.environ.items() if k != dg.ENV}
            env['PYTHONPATH'] = os.pathsep.join([here, os.environ.get('PYTHONPATH', '')])
            if mode:
                env[dg.ENV] = mode
            done = subprocess.run([sys.executable, '-B', '-c', code], env=env, cwd=here, capture_output=True, text=True,
                                  timeout=120)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            labels, n3_modes, total, nominal = json.loads(done.stdout.strip().splitlines()[-1])
            self.assertEqual(labels, J2_REPLAY, mode)
            self.assertEqual(n3_modes, [mode or 'chunk'], mode)
            self.assertLessEqual(total, nominal)


if __name__ == '__main__':
    unittest.main()
