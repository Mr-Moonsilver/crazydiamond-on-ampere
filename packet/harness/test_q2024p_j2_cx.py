#!/usr/bin/env python3
"""Composition cohort tooling (tools/build_q2024p_j2_cx.py, q2024p_j2_cx_render.py, q2024p_j2_cx_reader.py). CPU only.

  * flag sets: defaults; every flag accepted alone; the refused combinations (settle with the yield window, a
    decode-group mode other than chunk without the liveness observer, verify settings without ingress, ingress without
    explicit verify settings), unknown flags and values;
  * names: four blocks and the dry packet per tag, none a '-'-prefix of another or of a known QNAME; a clash refuses;
  * queues (hub only, needs the parent j2 bundle): the settings line carries the three composition settings after the
    ingress one, declaration and draft assert them, the draft exports FLASH_S5_DECODE_GROUPS right after its anchor
    and its arm gate reads ENGINE_CRASHED;
  * reader on synthetic records (the night-pack reader fixtures, records in fresh mkdtemp directories left in place):
    R per block pooled over the two seeds, G the geometric mean, a block without R named; the lever A append_pieces
    delta only where the block exists; identity NOTEs for a composition setting that differs from the cohort's.
No engine, torch or box access.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE.parent / 'tools', HERE / 'tools', HERE.parent / 'patches' / 's5_engine'):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))

TOOLS = next(p for p in (HERE.parent / 'tools', HERE / 'tools') if (p / 'build_q2024p_j2_cx.py').is_file())


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cx = _load('build_q2024p_j2_cx')
render = _load('q2024p_j2_cx_render')
reader = _load('q2024p_j2_cx_reader')
import test_q2024p_j2_np_reader as npt  # noqa: E402


class Flags(unittest.TestCase):
    def refuses(self, pairs):
        with self.assertRaises(SystemExit):
            cx.parse_flags(pairs)

    def test_defaults(self):
        flags = cx.parse_flags([])
        self.assertEqual(flags, reader.DEFAULTS)
        self.assertEqual(flags['FLASH_S5_DECODE_GROUPS'], 'chunk')
        self.assertEqual([f[0] for f in cx.FLAGS], list(reader.DEFAULTS))

    def test_each_flag_alone(self):
        for pairs in (['FLASH_S5_SMALL_PIECE_BATCH=1'], ['FLASH_S5_YIELD_REENTRY_SETTLE=1'],
                      ['FLASH_S5_YIELD_NO_APPEND_OWNER=1'], ['FLASH_S5_YIELD_WINDOW=1'],
                      ['FLASH_S5_APPEND_NO_BLOCK_STOP=1'], ['FLASH_S5_DECODE_LIVENESS=1'],
                      ['FLASH_S5_INGRESS_INCREMENTAL=1', 'FLASH_S5_INGRESS_VERIFY_EVERY=16', 'FLASH_S5_INGRESS_VERIFY_ASYNC=1'],
                      ['FLASH_S5_DECODE_GROUPS=off', 'FLASH_S5_DECODE_LIVENESS=1'],
                      ['FLASH_S5_DECODE_GROUPS=off-small', 'FLASH_S5_DECODE_LIVENESS=1']):
            flags = cx.parse_flags(pairs)
            for p in pairs:
                k, _, v = p.partition('=')
                self.assertEqual(flags[k], v)

    def test_all_compatible_on(self):
        flags = cx.parse_flags(['FLASH_S5_SMALL_PIECE_BATCH=1', 'FLASH_S5_YIELD_NO_APPEND_OWNER=1', 'FLASH_S5_YIELD_WINDOW=1',
                                'FLASH_S5_INGRESS_INCREMENTAL=1', 'FLASH_S5_INGRESS_VERIFY_EVERY=16',
                                'FLASH_S5_INGRESS_VERIFY_ASYNC=1', 'FLASH_S5_APPEND_NO_BLOCK_STOP=1',
                                'FLASH_S5_DECODE_GROUPS=off-small', 'FLASH_S5_DECODE_LIVENESS=1'])
        self.assertEqual(flags['FLASH_S5_YIELD_REENTRY_SETTLE'], '0')

    def test_refusals(self):
        self.refuses(['FLASH_S5_YIELD_REENTRY_SETTLE=1', 'FLASH_S5_YIELD_WINDOW=1'])
        self.refuses(['FLASH_S5_DECODE_GROUPS=off'])
        self.refuses(['FLASH_S5_DECODE_GROUPS=off-small', 'FLASH_S5_DECODE_LIVENESS=0'])
        self.refuses(['FLASH_S5_INGRESS_VERIFY_EVERY=1'])
        self.refuses(['FLASH_S5_INGRESS_VERIFY_ASYNC=1'])
        self.refuses(['FLASH_S5_INGRESS_INCREMENTAL=1'])
        self.refuses(['FLASH_S5_INGRESS_INCREMENTAL=1', 'FLASH_S5_INGRESS_VERIFY_EVERY=1'])
        self.refuses(['FLASH_S5_INGRESS_INCREMENTAL=1', 'FLASH_S5_INGRESS_VERIFY_EVERY=0', 'FLASH_S5_INGRESS_VERIFY_ASYNC=1'])
        self.refuses(['FLASH_S5_DECODE_GROUPS=always', 'FLASH_S5_DECODE_LIVENESS=1'])
        self.refuses(['FLASH_S5_TAIL_CUT_BACK=1'])          # lever B is not on this line
        self.refuses(['FLASH_S5_DECODE_MERGE=0'])           # the engine's install switch is not a cohort flag
        self.refuses(['FLASH_S5_APPEND_NO_BLOCK_STOP=2'])
        self.refuses(['FLASH_S5_YIELD_WINDOW'])


class Names(unittest.TestCase):
    def test_blocks_distinct_and_prefix_free(self):
        for tag in ('', 'a', 'w1', 'y12ab'):
            names = [cx.name_of(v, tag) for v in cx.BLOCKS + (cx.DRY,)]
            self.assertEqual(len(set(names)), 5)
            self.assertEqual(render.clashes(names + list(render.KNOWN)),
                             render.clashes(list(render.KNOWN)), tag)          # no clash involves a cx name
            self.assertTrue(all(n.startswith(render.PARENT_QNAME + '-cx' + tag + '-') for n in names))
        a, b = cx.name_of('b1', ''), cx.name_of('b1', 'a')
        self.assertFalse(b.startswith(a + '-') or a.startswith(b + '-'))

    def test_clash_refuses(self):
        self.assertEqual(render.clashes(['x-a', 'x-a-b1', 'x-b']), [('x-a', 'x-a-b1')])
        with self.assertRaises(SystemExit):
            render.main(['--source-commit', 'HEAD', '--tag', 'q', '--out', '/nonexistent/never-made',
                         '--known', cx.name_of('b2', 'q')])
        with self.assertRaises(SystemExit):
            render.main(['--source-commit', 'HEAD', '--tag', 'q', '--out', '/nonexistent/never-made',
                         '--known', cx.name_of('b2', 'q') + '-old'])
        with self.assertRaises(SystemExit):
            render.main(['--source-commit', 'HEAD', '--dry', '--set', 'FLASH_S5_YIELD_WINDOW=1',
                         '--out', '/nonexistent/never-made'])
        for bad in ('A', 'x-y', 'toolongtag'):
            with self.assertRaises(AssertionError):
                cx.name_of('b1', bad)


class Queues(unittest.TestCase):
    def test_settings_asserts_export_gate(self):
        if not Path(cx.hgb.PARENT).is_dir():
            self.skipTest('no parent j2 bundle here (bundle run): the queue edits are proved on the hub')
        flags = cx.parse_flags(['FLASH_S5_APPEND_NO_BLOCK_STOP=1', 'FLASH_S5_DECODE_GROUPS=off-small',
                                'FLASH_S5_DECODE_LIVENESS=1'])
        saved = dict(cx.CONFIG)
        try:
            cx.configure(flags, 'zt')
            texts = cx.queues('b3')
        finally:
            cx.CONFIG.update(saved)
        settings = next(t for r, t in texts.items() if r.endswith('_settings.sh'))
        tail = 'FLASH_S5_INGRESS_INCREMENTAL=0 FLASH_S5_APPEND_NO_BLOCK_STOP=1 FLASH_S5_DECODE_GROUPS=off-small FLASH_S5_DECODE_LIVENESS=1 '
        self.assertEqual(settings.count(tail), 1)
        for rel, text in texts.items():
            if rel.endswith('_settings.sh'):
                continue
            for env, value in (('FLASH_S5_APPEND_NO_BLOCK_STOP', '1'), ('FLASH_S5_DECODE_GROUPS', 'off-small'),
                               ('FLASH_S5_DECODE_LIVENESS', '1')):
                self.assertIn("assert env.get('%s') == '%s'" % (env, value), text, rel)
            if rel.endswith('.draft.sh'):
                self.assertIn(cx.dmp.DRAFT_ANCHOR + 'export FLASH_S5_DECODE_GROUPS=off-small ', text)
                self.assertIn('[[ $ENGINE_CRASHED != 0 ]]', text)
                self.assertNotIn(cx.dmp.ARM_GATE, text)
        self.assertTrue(any(cx.name_of('b3', 'zt') in t for t in texts.values()))


def pieces(n):
    return dict(appends=4 * n, pieces=6 * n, fused_pieces=3 * n, stops_skipped=5 * n, errors=0, stale_states_unpublished=n,
                extra_blocks_unpublished=2 * n, hist={'1': 2 * n, '2': 2 * n}, fallbacks={'junction': n} if n else {},
                max_new=8192)


def record(compose_env=None, append_pieces=False, done=None):
    """An np 'off' fixture record, composition settings in its identity, optional append_pieces blocks, and the
    replay summaries set to the given completed turns per arm."""
    root = npt.build('off', window=False)
    env = dict(FLASH_S5_APPEND_NO_BLOCK_STOP='0', FLASH_S5_DECODE_GROUPS='chunk', FLASH_S5_DECODE_LIVENESS='0')
    env.update(compose_env or {})
    for phase in ('start', 'end'):
        path = root / ('engine.identity-%s.json' % phase)
        doc = json.loads(path.read_text())
        doc['identity']['env'].update(env)
        path.write_text(json.dumps(doc))
    for label in reader.SP_ARMS + reader.N3_ARMS:
        path = root / ('arm-' + label + '.json')
        arm = json.loads(path.read_text())
        arm.setdefault('label', label)          # the box arm record carries its label (the dmp reader keys on it)
        if append_pieces and label.startswith('n3'):
            for row, n in zip(arm['statuses'], (0, 1)):
                row.setdefault('native_admission', {})['append_pieces'] = pieces(n)
        if done is not None:
            cfg = arm.setdefault('replay_config', {})
            cfg.setdefault('DURATION', 480.0)
            cfg.setdefault('WARMUP', 0.0)
            summary = arm.setdefault('replay_summary', {})
            summary['warm_turns'], summary['turns_finished_after_window'] = done[label], 0
        path.write_text(json.dumps(arm))
    return root


class Reader(unittest.TestCase):
    DONE1 = {'sp.g2-live-0': 40, 'n3.g2-live-0': 50, 'n3.g2-live-1': 70, 'sp.g2-live-1': 60}
    DONE2 = {'sp.g2-live-0': 50, 'n3.g2-live-0': 55, 'n3.g2-live-1': 45, 'sp.g2-live-1': 50}

    def test_ratio_and_geomean(self):
        flags = reader.flags_of([])
        r1 = reader.collect(record(done=self.DONE1), 'b1', flags)
        r2 = reader.collect(record(done=self.DONE2), 'b2', flags)
        self.assertEqual(r1['R'], round(120 / 100, 4))
        self.assertEqual(r2['R'], round(100 / 100, 4))
        c = reader.cohort_line([r1, r2])
        self.assertAlmostEqual(c['G'], round(math.sqrt(1.2 * 1.0), 4))
        self.assertEqual(c['blocks_without_R'], [])
        self.assertEqual(r1['compose_notes'], [n for n in r1['compose_notes'] if n.startswith('dmp-')])
        self.assertFalse([n for n in r1['compose_notes'] if n.startswith('identity-')])
        text = reader.render([r1, r2])
        self.assertIn('R(n3/sp pooled)=1.2', text)
        self.assertIn('G=%s' % c['G'], text)

    def test_unread_arm_has_no_ratio(self):
        root = record(done=self.DONE1)
        (root / 'arm-n3.g2-live-1.json').rename(root / 'arm-n3.g2-live-1.json.moved')
        r = reader.collect(root, 'b3', reader.flags_of([]))
        self.assertIsNone(r['R'])
        self.assertEqual(reader.cohort_line([r])['blocks_without_R'], ['b3'])

    def test_append_pieces_only_with_block(self):
        flags = reader.flags_of(['FLASH_S5_APPEND_NO_BLOCK_STOP=1'])
        r = reader.collect(record(dict(FLASH_S5_APPEND_NO_BLOCK_STOP='1'), append_pieces=True), 'b1', flags)
        got = r['arms']['n3.g2-live-0']['append_pieces']
        self.assertEqual({k: got[k] for k in ('appends', 'pieces', 'fused_pieces', 'stops_skipped')},
                         dict(appends=4, pieces=6, fused_pieces=3, stops_skipped=5))
        self.assertEqual(got['pieces_per_append'], 1.5)
        self.assertEqual(got['fallbacks'], {'junction': 1})
        self.assertNotIn('append_pieces', r['arms'].get('sp.g2-live-0') or {})
        self.assertFalse([n for n in r['compose_notes'] if n.startswith('identity-')])
        plain = reader.collect(record(), 'b1', reader.flags_of([]))
        self.assertIsNone(plain['arms']['n3.g2-live-0']['append_pieces'])

    def test_identity_note(self):
        flags = reader.flags_of(['FLASH_S5_DECODE_GROUPS=off-small', 'FLASH_S5_DECODE_LIVENESS=1'])
        r = reader.collect(record(dict(FLASH_S5_DECODE_LIVENESS='1')), 'b2', flags)
        notes = [n for n in r['compose_notes'] if n.startswith('identity-')]
        self.assertEqual(sorted(notes), ['identity-b2-FLASH_S5_DECODE_GROUPS-end-chunk',
                                         'identity-b2-FLASH_S5_DECODE_GROUPS-start-chunk'])
        r = reader.collect(record(dict(FLASH_S5_DECODE_GROUPS='off-small', FLASH_S5_DECODE_LIVENESS='1')), 'b2', flags)
        self.assertFalse([n for n in r['compose_notes'] if n.startswith('identity-')])

    def test_cohort_file_flags(self):
        import tempfile
        d = Path(tempfile.mkdtemp(prefix='cx-cohort-'))
        flags = cx.parse_flags(['FLASH_S5_APPEND_NO_BLOCK_STOP=1'])
        (d / 'cohort.json').write_text(json.dumps(dict(schema='q2024p-j2-cx-cohort/1', flags=flags)))
        self.assertEqual(reader.flags_of([], d / 'cohort.json'), flags)
        with self.assertRaises(ValueError):
            reader.flags_of(['FLASH_S5_TAIL_CUT_BACK=1'])


if __name__ == '__main__':
    unittest.main()
