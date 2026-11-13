#!/usr/bin/env python3
"""Composition cohort tooling (tools/build_q2024p_j2_cx.py, q2024p_j2_cx_render.py, q2024p_j2_cx_reader.py). CPU only.

  * flag sets: defaults; every flag accepted alone; the refused combinations (settle with the yield window, a
    decode-group mode other than chunk without the liveness observer, verify settings without ingress, ingress without
    explicit verify settings, FLASH_S5_CONTROL_INWAIT with settle, the batch, ingress, a decode-group mode other than
    chunk or the liveness observer), unknown flags and values; CONTROL_INWAIT with A, Y1 and Y2 accepted;
  * the control-inwait parent: the bundle copy's sha is the suite's pin, the frozen CPU's docker command names it;
  * names: four blocks and the dry packet per tag, none a '-'-prefix of another or of a known QNAME; a clash refuses;
  * queues: the settings line carries the three composition settings after the ingress one (or its verify mode),
    declaration and draft assert them, the draft exports FLASH_S5_DECODE_GROUPS right after its anchor and its arm gate
    reads ENGINE_CRASHED; lever A's MAX_NEW is at most the launch's append-yield max_new (the A x Y1 precondition). On
    the hub for a non-default flag set through the builder; in a bundle on the bundle's own built queue (no skip);
  * reader on synthetic records (the night-pack reader fixtures, records in fresh mkdtemp directories left in place):
    R per block pooled over the two seeds, G the geometric mean, a block without R named; the lever A append_pieces
    delta only where the block exists; lever D's fast and stock sampler steps per arm and rank from stamped
    engine-stream.log lines, with silent/present notes; the control_inwait delta per transition class on every arm (sp and n3) beside
    the auto-arm delta, 'off (block absent)' without it, NOTEs for a block absent under the flag or present without it;
    identity NOTEs for a composition setting that differs from the cohort's.
No engine, torch or box access.
"""
import importlib.util
import json
import math
import os
from pathlib import Path
import re
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
                      ['FLASH_S5_APPEND_NO_BLOCK_STOP=1'], ['FLASH_S5_DECODE_LIVENESS=1'], ['FLASH_S5_CONTROL_INWAIT=1'],
                      ['FLASH_S5_CONTROL_INWAIT=1', 'FLASH_S5_CONTROL_O3_INWAIT=1'], ['FLASH_S5_DECODE_ANATOMY=1'],
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
        ayci = cx.parse_flags(['FLASH_S5_APPEND_NO_BLOCK_STOP=1', 'FLASH_S5_YIELD_NO_APPEND_OWNER=1',
                               'FLASH_S5_YIELD_WINDOW=1', 'FLASH_S5_CONTROL_INWAIT=1'])
        self.assertEqual(ayci['FLASH_S5_CONTROL_INWAIT'], '1')
        ayc2 = cx.parse_flags(['FLASH_S5_APPEND_NO_BLOCK_STOP=1', 'FLASH_S5_YIELD_NO_APPEND_OWNER=1',
                               'FLASH_S5_YIELD_WINDOW=1', 'FLASH_S5_CONTROL_INWAIT=1', 'FLASH_S5_CONTROL_O3_INWAIT=1',
                               'FLASH_S5_DECODE_ANATOMY=1'])
        self.assertEqual((ayc2['FLASH_S5_CONTROL_O3_INWAIT'], ayc2['FLASH_S5_DECODE_SAMPLE_FAST']), ('1', '0'))

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
        ci = ['FLASH_S5_CONTROL_INWAIT=1']
        self.refuses(ci + ['FLASH_S5_YIELD_REENTRY_SETTLE=1'])
        self.refuses(ci + ['FLASH_S5_SMALL_PIECE_BATCH=1'])
        self.refuses(ci + ['FLASH_S5_INGRESS_INCREMENTAL=1', 'FLASH_S5_INGRESS_VERIFY_EVERY=16',
                           'FLASH_S5_INGRESS_VERIFY_ASYNC=1'])
        self.refuses(ci + ['FLASH_S5_DECODE_GROUPS=off', 'FLASH_S5_DECODE_LIVENESS=1'])
        self.refuses(ci + ['FLASH_S5_DECODE_LIVENESS=1'])
        self.refuses(['FLASH_S5_CONTROL_INWAIT=2'])
        for key, safe in cx.CONTROL_INWAIT_UNTESTED:           # each untested pair refused by its own rule
            self.assertEqual(cx.parse_flags([])[key], safe)
        self.refuses(['FLASH_S5_DECODE_SAMPLE_FAST=2'])
        self.refuses(['FLASH_S5_DECODE_ANATOMY=yes'])
        # scope 2 needs scope 1; its untested pairs are scope 1's (refused through CONTROL_INWAIT=1)
        self.refuses(['FLASH_S5_CONTROL_O3_INWAIT=1'])
        self.refuses(['FLASH_S5_CONTROL_O3_INWAIT=2', 'FLASH_S5_CONTROL_INWAIT=1'])
        o3 = ci + ['FLASH_S5_CONTROL_O3_INWAIT=1']
        self.refuses(o3 + ['FLASH_S5_YIELD_REENTRY_SETTLE=1'])
        self.refuses(o3 + ['FLASH_S5_DECODE_LIVENESS=1'])
        # lever D: blocked by Codex's engine review (lead note 13 Nov) until a reviewed fix is on the line
        self.assertIsNone(cx.DECODE_SAMPLE_FAST_REVIEWED_FIX)
        for extra in ([], ['FLASH_S5_DECODE_ANATOMY=1'], o3):
            self.refuses(['FLASH_S5_DECODE_SAMPLE_FAST=1'] + extra)


class ControlInWaitParent(unittest.TestCase):
    def test_parent_pin_and_container_env(self):
        commit, name, digest = cx.CONTROL_INWAIT_PARENT
        import hashlib
        import test_control_inwait as tci
        pinned = re.search(r"hexdigest\(\) == '([0-9a-f]{64})'", Path(tci.__file__).read_text()).group(1)
        self.assertEqual(digest, pinned)
        old, new = cx.CONTROL_INWAIT_ENV
        self.assertIn('CONTROL_INWAIT_PARENT_SOURCE=/src/harness/' + cx.CONTROL_INWAIT_PARENT_FILE, new)
        self.assertTrue(new.startswith(old))
        bundle_copy = HERE / cx.CONTROL_INWAIT_PARENT_FILE
        tool = HERE / 'tools' / 'n3j2_frozen_cpu.py'
        if (HERE / 'n3j2_packet.py').is_file() and tool.is_file():
            # a bundle: its own parent copy and frozen CPU tool, built by this builder
            self.assertEqual(hashlib.sha256(bundle_copy.read_bytes()).hexdigest(), digest)
            self.assertEqual(tool.read_text().count(new), 1)
            self.assertEqual(os.environ.get('CONTROL_INWAIT_PARENT_SOURCE'), '/src/harness/' + cx.CONTROL_INWAIT_PARENT_FILE)
        else:
            self.assertFalse(bundle_copy.exists())
            text = (Path(cx.hgb.PARENT) / 'tools/n3j2_frozen_cpu.py').read_text()
            self.assertIn(old, text)         # the anchor the builder rewrites, in the parent's tool (yrs adds no -e)


class ControlO3Parent(unittest.TestCase):
    def test_parent_pin_and_container_env(self):
        """test_control_o3_inwait's frozen OFF parent (a15bed1d's native_auto_arm.py): the builder's pin equals the
        suite's, the frozen CPU line names it, and in a bundle the copy and the env are there."""
        import hashlib
        import test_control_o3_inwait as to3
        commit, name, digest = cx.CONTROL_O3_PARENT
        text = Path(to3.__file__).read_text()
        self.assertIn("'%s:%s'" % (commit, name), text)
        self.assertIn("'%s'" % digest, text)
        old, new = cx.CONTROL_INWAIT_ENV
        self.assertIn('CONTROL_O3_PARENT_SOURCE=/src/harness/' + cx.CONTROL_O3_PARENT_FILE, new)
        bundle_copy = HERE / cx.CONTROL_O3_PARENT_FILE
        if (HERE / 'n3j2_packet.py').is_file() and (HERE / 'tools' / 'n3j2_frozen_cpu.py').is_file():
            self.assertEqual(hashlib.sha256(bundle_copy.read_bytes()).hexdigest(), digest)
            self.assertEqual(os.environ.get('CONTROL_O3_PARENT_SOURCE'), '/src/harness/' + cx.CONTROL_O3_PARENT_FILE)
        else:
            self.assertFalse(bundle_copy.exists())


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
        for argv in (['--dry', '--single', '--tag', 'q'], ['--single']):      # single: one tagged packet, b1 only
            with self.assertRaises(SystemExit):
                render.main(['--source-commit', 'HEAD', '--out', '/nonexistent/never-made'] + argv)
        with self.assertRaises(SystemExit):     # the single packet's name is checked like a cohort's
            render.main(['--source-commit', 'HEAD', '--single', '--tag', 'q', '--out', '/nonexistent/never-made',
                         '--known', cx.name_of('b1', 'q')])
        for bad in ('A', 'x-y', 'toolongtag'):
            with self.assertRaises(AssertionError):
                cx.name_of('b1', bad)


COMPOSE = ('FLASH_S5_APPEND_NO_BLOCK_STOP', 'FLASH_S5_DECODE_GROUPS', 'FLASH_S5_DECODE_LIVENESS',
           'FLASH_S5_CONTROL_INWAIT', 'FLASH_S5_CONTROL_O3_INWAIT', 'FLASH_S5_DECODE_SAMPLE_FAST',
           'FLASH_S5_DECODE_ANATOMY')
QUEUE = re.compile(r'^q18a_parity_n3s24q2024p_j2_cx([a-z0-9]{0,8})_(b[1-4]|dry)_settings\.sh$')


def engine_text(name):
    for path in (HERE.parent / 'patches/s5_engine' / name, HERE / 'e3_builder' / name):
        if path.is_file():
            return path.read_text()
    raise FileNotFoundError(name)


class NameOnly(unittest.TestCase):
    """render.name_only on two synthetic block builds (fresh mkdtemp, left in place): names in contents and in queue
    file names (QNAME and underscore forms) and the manifest are substituted; any other byte difference is listed."""

    def block(self, out, variant, manifest, extra=b''):
        name = cx.name_of(variant, 'zz')
        d = out / ('build-' + variant)
        bundle = d / (name + '-bundle-' + manifest[:12])
        (bundle / 'runs').mkdir(parents=True)
        q = name.replace('-', '_')
        (bundle / 'runs' / (q + '_settings.sh')).write_text('Q=%s FLASH_S5_DECODE_GROUPS=chunk\n' % name)
        (bundle / 'n3j2_packet.py').write_bytes(b'configure = None\n' + extra)
        (bundle / 'SHA256SUMS').write_text(manifest + '\n')
        runner = d / (name + '-owner-run-' + manifest[:12] + '.py')
        runner.write_text('BUNDLE = %r\nMANIFEST = %r\n' % (str(bundle), manifest))
        (d / 'build.json').write_text(json.dumps(dict(name=name, manifest=manifest, bundle=str(bundle), runner=str(runner),
                                                      flags=cx.parse_flags([]), source_commit='c' * 40, overlay_manifest='o' * 64)))

    def test_names_substituted_other_bytes_listed(self):
        import tempfile
        out = Path(tempfile.mkdtemp(prefix='cx-name-only-'))
        self.block(out, 'b1', 'a' * 64)
        self.block(out, 'b2', 'b' * 64)
        self.block(out, 'b3', 'c' * 64, extra=b'x = 1\n')
        rep = render.name_only(out, ['b1', 'b2', 'b3'])
        self.assertEqual(rep['b2']['differ'], [])                 # queue (renamed path), sums (manifest only)
        self.assertEqual(rep['b2']['equal_after_name'], 2)
        self.assertTrue(rep['b2']['runner_equal_after_name'] and rep['b2']['flags_equal'])
        self.assertEqual(rep['b3']['differ'], ['n3j2_packet.py'])


class Queues(unittest.TestCase):
    """On the hub the builder's queue edits for a non-default flag set; in a bundle the bundle's own built texts."""

    def check_texts(self, texts, flags, name):
        settings = [t for r, t in texts.items() if r.endswith('_settings.sh')]
        self.assertEqual(len(settings), 1)
        settings = settings[0]
        head = 'FLASH_S5_INGRESS_INCREMENTAL=%s ' % flags['FLASH_S5_INGRESS_INCREMENTAL']
        if flags['FLASH_S5_INGRESS_INCREMENTAL'] == '1':
            head = 'FLASH_S5_INGRESS_VERIFY_ASYNC=%s ' % flags['FLASH_S5_INGRESS_VERIFY_ASYNC']
        self.assertEqual(settings.count(head + ''.join('%s=%s ' % (k, flags[k]) for k in COMPOSE)), 1)
        self.assertEqual(cx.parse_flags(['%s=%s' % kv for kv in flags.items() if kv[1] is not None]), flags)
        drafts = 0
        for rel, text in texts.items():
            if rel.endswith('_settings.sh'):
                continue
            for env in COMPOSE:
                self.assertIn("assert env.get('%s') == '%s'" % (env, flags[env]), text, rel)
            if rel.endswith('.draft.sh'):
                drafts += 1
                self.assertIn(cx.dmp.DRAFT_ANCHOR + 'export FLASH_S5_DECODE_GROUPS=%s ' % flags['FLASH_S5_DECODE_GROUPS'], text)
                self.assertIn('[[ $ENGINE_CRASHED != 0 ]]', text)
                self.assertNotIn(cx.dmp.ARM_GATE, text)
        self.assertEqual(drafts, 1)
        self.assertTrue(any(name in t for t in texts.values()))
        # lever A x Y1 precondition: every turn lever A fuses (at most its MAX_NEW rows past the turn start) has at most
        # the launch's append-yield max_new rows left, so Y1 refuses its yield with or without lever A
        a_max = int(re.search(r'^APPEND_NO_BLOCK_STOP_MAX_NEW = (\d+)$', engine_text('scheduler_hooks.py'), re.M).group(1))
        y_max = re.findall(r'\bFLASH_S5_APPEND_YIELD_MAX_NEW=(\d+)\b', settings)
        self.assertTrue(y_max and len(set(y_max)) == 1, y_max)
        self.assertLessEqual(a_max, int(y_max[0]))

    def test_settings_asserts_export_gate(self):
        runs = HERE / 'runs'
        built = sorted(p.name for p in runs.glob('*_settings.sh') if QUEUE.match(p.name)) if runs.is_dir() else []
        # a bundle (the frozen CPU mounts it at /src/harness, beside /src/patches): its own built cx queue in runs/
        if built and (HERE / 'n3j2_packet.py').is_file():
            self.assertEqual(len(built), 1, built)                 # a bundle: its own built queue
            tag, variant = QUEUE.match(built[0]).groups()
            stem = built[0][:-len('_settings.sh')]
            texts = {'runs/' + n: (runs / n).read_text() for n in (stem + '_settings.sh', stem + '_declare.sh', stem + '.draft.sh')}
            settings = texts['runs/' + stem + '_settings.sh']
            flags = {}
            for f in cx.FLAGS:
                got = re.findall(r'\b%s=(\S+) ' % f[0], settings)
                self.assertLessEqual(len(set(got)), 1, f[0])
                flags[f[0]] = got[0] if got else None
            self.check_texts(texts, flags, cx.name_of(variant, tag))
            return
        self.assertTrue(Path(cx.hgb.PARENT).is_dir(), 'hub run needs the parent j2 bundle')
        flags = cx.parse_flags(['FLASH_S5_APPEND_NO_BLOCK_STOP=1', 'FLASH_S5_DECODE_GROUPS=off-small',
                                'FLASH_S5_DECODE_LIVENESS=1', 'FLASH_S5_INGRESS_INCREMENTAL=1',
                                'FLASH_S5_INGRESS_VERIFY_EVERY=16', 'FLASH_S5_INGRESS_VERIFY_ASYNC=1'])
        saved = dict(cx.CONFIG)
        try:
            cx.configure(flags, 'zt')
            texts = cx.queues('b3')
        finally:
            cx.CONFIG.update(saved)
        self.check_texts(texts, flags, cx.name_of('b3', 'zt'))
        self.assertIn('FLASH_S5_INGRESS_VERIFY_EVERY=16 FLASH_S5_INGRESS_VERIFY_ASYNC=1 FLASH_S5_APPEND_NO_BLOCK_STOP=1 ',
                      next(t for r, t in texts.items() if r.endswith('_settings.sh')))

    def test_decode_step_flags_in_the_queues(self):
        """The anatomy window and scope 2 ride the settings line and the asserts (lever D is refused, so its setting is
        written as 0). The anatomy window needs the settings' handler anatomy and burst progress, scope 2 the settings'
        AUTO_ARM=1 and O3=1: a parent without them is refused at render time, before the engine would."""
        if (HERE / 'n3j2_packet.py').is_file():
            return                                  # a bundle: its own queue is checked by the test above
        self.assertTrue(Path(cx.hgb.PARENT).is_dir(), 'hub run needs the parent j2 bundle')
        flags = cx.parse_flags(['FLASH_S5_CONTROL_INWAIT=1', 'FLASH_S5_CONTROL_O3_INWAIT=1', 'FLASH_S5_DECODE_ANATOMY=1'])
        saved = dict(cx.CONFIG)
        try:
            cx.configure(flags, 'zd')
            texts = cx.queues('b1')
            self.assertIn('e3/decode_step.py', cx.hgb.NEW_OVERLAY_FILES)
            self.assertIn('decode_step', cx.ing.BOUND_MODULES)
            self.assertIn('test_control_o3_inwait', cx.SUITES)
            settings = next(t for r, t in texts.items() if r.endswith('_settings.sh'))
            self.assertIn('FLASH_S5_CONTROL_INWAIT=1 FLASH_S5_CONTROL_O3_INWAIT=1 FLASH_S5_DECODE_SAMPLE_FAST=0 '
                          'FLASH_S5_DECODE_ANATOMY=1 ', settings)
            for need in cx.CONTROL_O3_NEEDS:
                self.assertIn(need, settings)
            self.check_texts(texts, flags, cx.name_of('b1', 'zd'))
            original = cx._np_queues
            for old, new in (('FLASH_S5_HANDLER_ANATOMY=1 ', 'FLASH_S5_HANDLER_ANATOMY=0 '),
                             ('FLASH_S5_O3=1 ', 'FLASH_S5_O3=0 '), ('FLASH_S5_AUTO_ARM=1 ', 'FLASH_S5_AUTO_ARM=0 ')):
                cx._np_queues = lambda variant, old=old, new=new: {
                    r: t.replace(old, new) for r, t in original(variant).items()}
                try:
                    with self.assertRaises(AssertionError, msg=old):
                        cx.queues('b1')
                finally:
                    cx._np_queues = original
        finally:
            cx.CONFIG.update(saved)


def pieces(n):
    return dict(appends=4 * n, pieces=6 * n, fused_pieces=3 * n, stops_skipped=5 * n, errors=0, stale_states_unpublished=n,
                extra_blocks_unpublished=2 * n, hist={'1': 2 * n, '2': 2 * n}, fallbacks={'junction': n} if n else {},
                max_new=8192)


def inwait(n):
    row = lambda m: dict(attempts=3 * m, refused=m, invalidated=m, armed=2 * m, closed=2 * m, drains_skipped=2 * m,
                         drains_kept={'applied-output-cut': 2 * m, 'typed-route-fence': 2 * m},
                         pause_to_arm_ms=10. * m, pause_to_arm_max_ms=7. + m, decoder_stall_ms=4. * m,
                         decoder_stall_max_ms=3. + m)
    return dict(schema='s5-control-inwait/1', enabled=True,
                transitions={'full-entry': row(n), 'yield-full-reentry': row(2 * n)} if n else {})


def record(compose_env=None, append_pieces=False, done=None, control_inwait=False):
    """An np 'off' fixture record, composition settings in its identity, optional append_pieces blocks, and the
    replay summaries set to the given completed turns per arm."""
    root = npt.build('off', window=False)
    env = dict(FLASH_S5_APPEND_NO_BLOCK_STOP='0', FLASH_S5_DECODE_GROUPS='chunk', FLASH_S5_DECODE_LIVENESS='0',
               FLASH_S5_CONTROL_INWAIT='0')
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
        for row, n in zip(arm['statuses'], (0, 1)):
            row['auto_arm'] = dict(active=False, completed=10 + 7 * n, manual_owned=False, phase='idle',
                                   request_uid=None, selection_seq=20 + 5 * n)
            if control_inwait:
                row.setdefault('native_admission', {})['control_inwait'] = inwait(n if label.startswith('n3') else 0)
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

    def test_control_inwait_per_arm_off_beside_on(self):
        flags = reader.flags_of(['FLASH_S5_CONTROL_INWAIT=1'])
        r = reader.collect(record(dict(FLASH_S5_CONTROL_INWAIT='1'), control_inwait=True), 'b1', flags)
        got = r['arms']['n3.g2-live-0']['control_inwait']
        self.assertEqual(sorted(got), ['full-entry', 'yield-full-reentry'])
        full = got['full-entry']
        self.assertEqual({k: full[k] for k in reader.INWAIT},
                         dict(attempts=3, refused=1, invalidated=1, armed=2, closed=2, drains_skipped=2))
        self.assertEqual(full['drains_kept'], {'applied-output-cut': 2, 'typed-route-fence': 2})
        self.assertEqual((full['pause_to_arm_ms'], full['pause_to_arm_per_armed_ms']), (10.0, 5.0))
        self.assertEqual((full['decoder_stall_ms'], full['decoder_stall_per_armed_ms']), (4.0, 2.0))
        self.assertEqual(full['pause_to_arm_max_ms_since_start'], 8.0)
        self.assertEqual(got['yield-full-reentry']['armed'], 4)
        self.assertEqual(r['arms']['n3.g2-live-0']['auto_arm_delta'], dict(selection_seq=5, completed=7))
        self.assertEqual(r['arms']['sp.g2-live-0']['control_inwait'], {})          # on, never engaged on sp
        self.assertFalse([n for n in r['compose_notes'] if n.startswith(('identity-', 'control-inwait-'))])
        text = reader.render([r])
        self.assertIn('  sp.g2-live-0  control_inwait on, no transition yet  auto-arm selected=5 completed=7', text)
        self.assertIn('  n3.g2-live-1  control_inwait on  auto-arm selected=5 completed=7', text)
        self.assertIn('full-entry         attempts=3 refused=1 invalidated=1 armed=2 closed=2 drains_skipped=2 '
                      'kept={"applied-output-cut":2,"typed-route-fence":2} pause_to_arm_ms=10.0/armed=5.0', text)
        off = reader.collect(record(), 'b1', reader.flags_of([]))
        self.assertIsNone(off['arms']['n3.g2-live-1']['control_inwait'])
        self.assertIn('  n3.g2-live-1  control_inwait off (block absent)  auto-arm selected=5 completed=7',
                      reader.render([off]))
        (off_root := record()).joinpath('arm-sp.g2-live-1.json').rename(off_root / 'arm-sp.g2-live-1.json.moved')
        self.assertIn('  sp.g2-live-1  control_inwait - (arm unread)',
                      reader.render([reader.collect(off_root, 'b4', reader.flags_of([]))]))
        self.assertFalse([n for n in off['compose_notes'] if n.startswith('control-inwait-')])
        absent = reader.collect(record(dict(FLASH_S5_CONTROL_INWAIT='1')), 'b2', flags)
        self.assertEqual(sorted(n for n in absent['compose_notes'] if n.startswith('control-inwait-')),
                         ['control-inwait-absent-b2-' + a for a in sorted(reader.SP_ARMS + reader.N3_ARMS)])
        present = reader.collect(record(control_inwait=True), 'b3', reader.flags_of([]))
        self.assertIn('control-inwait-present-b3-n3.g2-live-0', present['compose_notes'])

    def test_control_o3_inwait_kept_classes_and_lane_o3_line(self):
        """Scope 2: the kept-SPLIT classes' cut counters beside scope 1's rows, the lane-o3 line (install, stock O3
        counts and the refusal reasons a fallback ends in) per arm, and the install/cut NOTES."""
        flags = reader.flags_of(['FLASH_S5_CONTROL_INWAIT=1', 'FLASH_S5_CONTROL_O3_INWAIT=1'])
        root = record(dict(FLASH_S5_CONTROL_INWAIT='1', FLASH_S5_CONTROL_O3_INWAIT='1'), control_inwait=True)
        kept = lambda m, applied: dict(attempts=3 * m, refused=m, invalidated=0, armed=2 * m, closed=m,
                                       drains_skipped=3 * m,
                                       drains_kept={'applied-output-cut': 3 * m, 'typed-route-fence': m,
                                                    'fallback-full-cut': m},
                                       pause_to_arm_ms=6. * m, pause_to_arm_max_ms=4., decoder_stall_ms=1. * m,
                                       decoder_stall_max_ms=1., cut_frames_issued=4 * m, cut_frames_applied=applied,
                                       arms_with_later_frames=m)
        o3 = lambda m: dict(on=1, armed=2 * m, yield_kept=m, kept=3 * m, refused={'census': m} if m else {},
                            yield_full={'census': m} if m else {}, full={})
        for label in reader.SP_ARMS + reader.N3_ARMS:
            path = root / ('arm-' + label + '.json')
            arm = json.loads(path.read_text())
            for row, n in zip(arm['statuses'], (0, 1)):
                m = n if label.startswith('n3') else 0
                block = row['native_admission']['control_inwait']
                block.update(o3_enabled=True, cut_pending=0, cut_bound=2)
                if m:
                    block['transitions']['split-kept-yield'] = kept(m, 4 * m if label == 'n3.g2-live-0' else 3 * m)
                row.setdefault('append_yield', {}).setdefault('declines', {})['o3'] = o3(m)
            path.write_text(json.dumps(arm))
        r = reader.collect(root, 'b1', flags)
        n0 = r['arms']['n3.g2-live-0']
        k = n0['control_inwait']['split-kept-yield']
        self.assertEqual({x: k[x] for x in reader.INWAIT_CUT}, dict(cut_frames_issued=4, cut_frames_applied=4,
                                                                    arms_with_later_frames=1))
        self.assertEqual(k['drains_kept'], {'applied-output-cut': 3, 'fallback-full-cut': 1, 'typed-route-fence': 1})
        self.assertNotIn('cut_frames_issued', n0['control_inwait']['full-entry'])     # scope 1's rows unchanged
        self.assertEqual(n0['control_o3'], dict(o3_inwait=True, cut_pending_after=0, o3=dict(
            armed=2, yield_kept=1, kept=3, refused={'census': 1}, yield_full={'census': 1}, full={})))
        self.assertEqual(sorted(n for n in r['compose_notes'] if n.startswith('control-')),
                         ['control-o3-cut-unapplied-b1-n3.g2-live-1'])               # 4 issued, 3 applied
        text = reader.render([r])
        self.assertIn('cut issued=4 applied=4 arms_with_later_frames=1 (kept-SPLIT', text)
        self.assertIn('lane-o3            o3_inwait=on cut_pending_after=0 armed=2 yield_kept=1 kept=3 '
                      'refused={"census":1} yield_full={"census":1} full={}', text)
        off = reader.collect(record(dict(FLASH_S5_CONTROL_INWAIT='1'), control_inwait=True), 'b2',
                             reader.flags_of(['FLASH_S5_CONTROL_INWAIT=1']))
        self.assertEqual(off['arms']['n3.g2-live-0']['control_o3'], dict(o3_inwait=False, cut_pending_after=None, o3=None))
        self.assertIn('o3_inwait=off cut_pending_after=None - (no o3 record)', reader.render([off]))
        self.assertFalse([n for n in off['compose_notes'] if n.startswith('control-o3-')])
        absent = reader.collect(record(dict(FLASH_S5_CONTROL_INWAIT='1'), control_inwait=True), 'b3', flags)
        self.assertIn('control-o3-inwait-absent-b3-n3.g2-live-0', absent['compose_notes'])
        present = reader.collect(root, 'b4', reader.flags_of(['FLASH_S5_CONTROL_INWAIT=1']))
        self.assertIn('control-o3-inwait-present-b4-sp.g2-live-0', present['compose_notes'])

    def test_sample_fast_steps_per_arm_and_rank(self):
        import datetime
        root = record(dict(FLASH_S5_DECODE_SAMPLE_FAST='1'))
        t0 = 1794436590.0
        order = ('sp.g2-live-0', 'n3.g2-live-0', 'n3.g2-live-1', 'sp.g2-live-1')
        for i, name in enumerate(order):                       # sequential arms, 600 s each, 100 s apart
            path = root / ('arm-' + name + '.json')
            arm = json.loads(path.read_text())
            arm['replay_start_s'] = t0 + 700 * i
            arm.setdefault('replay_config', {})['DURATION'] = 600
            path.write_text(json.dumps(arm))

        def line(t, rank, fast, not_gated, mixed, tie=0):
            stamp = datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            body = dict(schema='s5-decode-step/1', rank=rank, tie_overflow_lagged=tie,
                        counters=dict(fast=fast, stock=not_gated + mixed, stock_not_gated=not_gated, stock_k_mixed=mixed,
                                      stock_dtype=0))
            return '%s (Worker_PP1_TP%d pid=1) INFO flash-s5 sample-fast %s\n' % (stamp, rank - 2, json.dumps(body))
        rows = []
        for rank in (2, 3):
            rows += [line(t0 + 750, rank, 1, 0, 0),                      # n3.g2-live-0 starts at t0+700
                     line(t0 + 1290, rank, 5000, 0, 24),                  # inside n3.g2-live-0 (ends t0+1300)
                     line(t0 + 1500, rank, 9000, 0, 30),                  # inside n3.g2-live-1
                     line(t0 + 1990, rank, 13096, 0, 40, tie=rank - 2),  # its last line
                     line(t0 + 2200, rank, 13096, 1024, 40)]              # sp.g2-live-1: stock, not gated
        with (root / 'engine-stream.log').open('a') as handle:
            handle.writelines(rows)
        flags = reader.flags_of(['FLASH_S5_DECODE_SAMPLE_FAST=1'])
        r = reader.collect(root, 'b1', flags)
        n0, n1 = r['arms']['n3.g2-live-0']['sample_fast'], r['arms']['n3.g2-live-1']['sample_fast']
        self.assertEqual(sorted(n0), ['2', '3'])
        self.assertEqual({k: n0['2'][k] for k in ('fast', 'stock', 'stock_by_reason', 'lines')},
                         dict(fast=5000, stock=24, stock_by_reason={'k_mixed': 24}, lines=2))
        self.assertEqual((n1['3']['fast'], n1['3']['stock'], n1['3']['tie_overflow_lagged']), (8096, 16, 1))
        self.assertEqual(n1['2']['fast_share'], round(8096 / 8112, 4))
        sp1 = r['arms']['sp.g2-live-1']['sample_fast']['2']
        self.assertEqual((sp1['fast'], sp1['stock_by_reason']), (0, {'not_gated': 1024}))
        self.assertEqual(r['arms']['sp.g2-live-0']['sample_fast'], {})     # before D's first line: none
        self.assertFalse([n for n in r['compose_notes'] if n.startswith('sample-fast-')])
        text = reader.render([r])
        self.assertIn('  n3.g2-live-0  sample_fast rank2 fast=5000 stock=24 share=0.9952 stock_by_reason={"k_mixed":24} '
                      'tie_overflow_lagged=0 lines=2', text)
        self.assertIn('  sp.g2-live-0  sample_fast no lines in the arm', text)
        silent = reader.collect(record(dict(FLASH_S5_DECODE_SAMPLE_FAST='1')), 'b2', flags)
        self.assertEqual(sorted(n for n in silent['compose_notes'] if n.startswith('sample-fast-')),
                         ['sample-fast-silent-b2-n3.g2-live-0', 'sample-fast-silent-b2-n3.g2-live-1'])
        present = reader.collect(root, 'b3', reader.flags_of([]))
        self.assertIn('sample-fast-present-b3-n3.g2-live-0', present['compose_notes'])

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
