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
from unittest import mock

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
                      ['FLASH_S5_DECODE_TIMELINE=1'],
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
        ayc2t = cx.parse_flags(['FLASH_S5_APPEND_NO_BLOCK_STOP=1', 'FLASH_S5_YIELD_NO_APPEND_OWNER=1',
                                'FLASH_S5_YIELD_WINDOW=1', 'FLASH_S5_CONTROL_INWAIT=1', 'FLASH_S5_CONTROL_O3_INWAIT=1',
                                'FLASH_S5_DECODE_ANATOMY=1', 'FLASH_S5_DECODE_TIMELINE=1'])
        self.assertEqual(dict(ayc2, FLASH_S5_DECODE_TIMELINE='1'), ayc2t)      # ayc2s plus the stamp, nothing else

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
        self.refuses(['FLASH_S5_DECODE_TIMELINE=2'])
        # scope 2 needs scope 1; its untested pairs are scope 1's (refused through CONTROL_INWAIT=1)
        self.refuses(['FLASH_S5_CONTROL_O3_INWAIT=1'])
        self.refuses(['FLASH_S5_CONTROL_O3_INWAIT=2', 'FLASH_S5_CONTROL_INWAIT=1'])
        o3 = ci + ['FLASH_S5_CONTROL_O3_INWAIT=1']
        self.refuses(o3 + ['FLASH_S5_YIELD_REENTRY_SETTLE=1'])
        self.refuses(o3 + ['FLASH_S5_DECODE_LIVENESS=1'])
        # lever D: blocked by Codex's engine review (lead note 13 Nov 13:23) until a reviewed fix is on the line;
        # brief 9: the reviewed fix 2eba0dd0 is merged, D=1 parses with the anatomy and with scope 2
        self.assertEqual(cx.DECODE_SAMPLE_FAST_REVIEWED_FIX, '2eba0dd0d85f341ac9378c438ac4803c946104d8')
        for extra in ([], ['FLASH_S5_DECODE_ANATOMY=1'], o3, o3 + ['FLASH_S5_DECODE_ANATOMY=1']):
            self.assertEqual(cx.parse_flags(['FLASH_S5_DECODE_SAMPLE_FAST=1'] + extra)['FLASH_S5_DECODE_SAMPLE_FAST'], '1')
        with mock.patch.object(cx, 'DECODE_SAMPLE_FAST_REVIEWED_FIX', None):     # None blocks D again
            for extra in ([], ['FLASH_S5_DECODE_ANATOMY=1'], o3):
                self.refuses(['FLASH_S5_DECODE_SAMPLE_FAST=1'] + extra)

    def test_decode_sample_fast_gate(self):
        """How the builder recognises the reviewed-fix tree for D=1 (prepare, before and after the overlay): the fix
        an ancestor of the source commit AND every marker in decode_step.py (the fix's overflow select and route, the
        compose line's watermark refusal). This tree's decode_step.py passes; each marker removed is refused."""
        text = next(p for p in (HERE.parent / 'patches/s5_engine/decode_step.py',          # worktree, frozen CPU /src
                                HERE / 'native30_overlay/e3/decode_step.py') if p.is_file()).read_text()
        yes, no = (lambda sha: True), (lambda sha: False)
        self.assertEqual(cx.decode_sample_fast_refusals(text, yes), [])
        self.assertEqual(cx.decode_sample_fast_refusals(text, no), ['reviewed-fix-not-an-ancestor'])
        for i, marker in enumerate(cx.DECODE_SAMPLE_FAST_MARKERS):
            self.assertEqual(text.count(marker), 1, marker)
            self.assertEqual(cx.decode_sample_fast_refusals(text.replace(marker, ''), yes), ['missing-marker-%d' % i])
        with mock.patch.object(cx, 'DECODE_SAMPLE_FAST_REVIEWED_FIX', None):
            self.assertEqual(cx.decode_sample_fast_refusals(text, yes), ['no-reviewed-fix'])


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
           'FLASH_S5_DECODE_ANATOMY', 'FLASH_S5_DECODE_TIMELINE')
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

    def check_texts(self, texts, flags, name, variant=None):
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
                # brief 11 C: the quiet-SPLIT tail read exactly when the flag is on
                quiet = flags.get('QUIET_SPLIT_CONTROL') == '1'
                self.assertEqual(text.count(cx.QUIET_SPLIT_STEP), int(quiet), rel)
                self.assertEqual(text.count(cx.QUIET_SPLIT_EXPORT), int(quiet), rel)
                self.assertEqual(text.count('e3_quiet_split_box'), int(quiet) * (
                    cx.QUIET_SPLIT_EXPORT.count('e3_quiet_split_box') + cx.QUIET_SPLIT_STEP.count('e3_quiet_split_box')), rel)
                # brief 12: fixed work and the block's seed order, each exported once when set, before any python
                fixed = flags.get('REPLAY_FIXED_WORK', '0')
                want = '' if fixed == '0' else cx.FIXED_WORK_EXPORT % cx.FIXED_WORK[fixed]
                self.assertEqual(text.count('REPLAY_FIXED_WORK'), int(fixed != '0'), rel)
                order = flags.get('REPLAY_SEED_ORDER', 'plain')
                self.assertEqual(text.count('REPLAY_SEED_ORDER'), int(order != 'plain'), rel)
                for line in [want] + ([cx.SEED_ORDER_EXPORT % cx.seed_order_of(order, variant)] if order != 'plain' and variant else []):
                    if line:
                        self.assertEqual(text.count(line), 1, rel)
                        self.assertLess(text.index(line), text.index('python3'), rel)
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
                if f[0] in cx.HARNESS_FLAGS:      # not on the settings line: the draft's export, else the default
                    got = re.findall(r'^export %s=(\S+) ' % f[0], texts['runs/' + stem + '.draft.sh'], re.M)
                    self.assertLessEqual(len(got), 1, f[0])
                    flags[f[0]] = got[0] if got else f[3]
                    if f[0] == 'REPLAY_FIXED_WORK' and got:       # brief 12: the export names the spec, the flag its name
                        flags[f[0]] = {v[0]: k for k, v in cx.FIXED_WORK.items()}[got[0]]
                    if f[0] == 'REPLAY_SEED_ORDER' and got == ['plain']:   # only alternate exports plain (b1, b3)
                        flags[f[0]] = 'alternate'
                    continue
                got = re.findall(r'\b%s=(\S+) ' % f[0], settings)
                self.assertLessEqual(len(set(got)), 1, f[0])
                flags[f[0]] = got[0] if got else None
            self.check_texts(texts, flags, cx.name_of(variant, tag), variant)
            # brief 12: the seed-order tail and the mechanism-rule patch exactly when fixed work or a seed order is set
            on = cx.packet_edits_on(flags)
            self.assertEqual((HERE / 'n3j2_packet.py').read_text().count(cx.SEED_PACKET_TAIL), int(on))
            rule_text = (HERE / 'tools' / 'n3j2_junction_rule.py').read_text()
            self.assertEqual('_fwp' in rule_text, on)
            if on:
                self.assertEqual(cx.rule_patch(cx.rule_unpatch(rule_text)), rule_text)
            else:
                for old, new in cx.RULE_PATCH:
                    self.assertEqual((rule_text.count(old), rule_text.count(new)), (1, 0), new)
            # brief 11: the built box binds the guard with the observer the settings line launches
            box = (HERE / 'n3j2_box.py').read_text()
            want = cx.GUARD_INSTALL_NEW % cx.OBSERVER_OF[flags['FLASH_S5_DECODE_LIVENESS']]
            self.assertEqual(box.count(want), 1, want)
            self.assertEqual(box.count('decode_liveness_guard.install('), 1)
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

    def test_replay_reply_text_option(self):
        """Brief 11 B.1: seeded exports REPLAY_REPLY_TEXT once, right after the decode-group export and before any
        python; model (the default) leaves every queue text as it was (no name anywhere); never on the settings line."""
        self.assertEqual(cx.parse_flags([])['REPLAY_REPLY_TEXT'], 'model')
        with self.assertRaises(SystemExit):
            cx.parse_flags(['REPLAY_REPLY_TEXT=on'])
        self.assertEqual(reader.DEFAULTS['REPLAY_REPLY_TEXT'], 'model')
        self.assertNotIn('REPLAY_REPLY_TEXT', reader.COMPOSE_ENVS)
        self.assertNotIn('replay_reply_text', cx.EXTRA)
        self.assertIn('test_replay_reply_text', cx.SUITES)
        if (HERE / 'n3j2_packet.py').is_file():
            return                                  # a bundle: its own queue is checked by test_settings_asserts_export_gate
        self.assertTrue(Path(cx.hgb.PARENT).is_dir(), 'hub run needs the parent j2 bundle')
        saved = dict(cx.CONFIG)
        try:
            cx.configure(cx.parse_flags([]), 'zr')
            plain = cx.queues('b1')
            cx.configure(cx.parse_flags(['REPLAY_REPLY_TEXT=seeded']), 'zr')
            seeded = cx.queues('b1')
        finally:
            cx.CONFIG.update(saved)
        self.assertFalse(any('REPLAY_REPLY_TEXT' in t for t in plain.values()))
        self.assertEqual(set(plain), set(seeded))
        for rel in plain:
            if not rel.endswith('.draft.sh'):
                self.assertEqual(plain[rel], seeded[rel], rel)
                continue
            self.assertEqual(seeded[rel].count('REPLAY_REPLY_TEXT'), 1)
            self.assertEqual(seeded[rel].replace(cx.REPLY_TEXT_EXPORT, ''), plain[rel])
            head = seeded[rel].partition(cx.REPLY_TEXT_EXPORT)[0]
            self.assertEqual(head.partition(cx.dmp.DRAFT_ANCHOR)[2], cx.DRAFT_EXPORT % 'chunk')   # nothing runs between

    def test_fixed_work_and_seed_order_options(self):
        """Brief 12: REPLAY_FIXED_WORK (0|n480, needs seeded reply text) and REPLAY_SEED_ORDER (plain|swap|alternate)
        are harness flags: each exports one line after the decode-group export and before any python; alternate gives
        b2 and b4 swap, b1, b3 and the dry block plain; the defaults leave every queue text as it was and the packet and
        rule untouched (packet_edits_on False)."""
        flags = cx.parse_flags([])
        self.assertEqual((flags['REPLAY_FIXED_WORK'], flags['REPLAY_SEED_ORDER']), ('0', 'plain'))
        self.assertFalse(cx.packet_edits_on(flags))
        for bad in (['REPLAY_FIXED_WORK=n480'], ['REPLAY_FIXED_WORK=n480', 'REPLAY_REPLY_TEXT=model'],
                    ['REPLAY_FIXED_WORK=15'], ['REPLAY_SEED_ORDER=reverse']):
            with self.assertRaises(SystemExit, msg=bad):
                cx.parse_flags(bad)
        self.assertTrue(cx.packet_edits_on(cx.parse_flags(['REPLAY_SEED_ORDER=swap'])))
        self.assertTrue(cx.packet_edits_on(cx.parse_flags(['REPLAY_FIXED_WORK=n480', 'REPLAY_REPLY_TEXT=seeded'])))
        for name in ('REPLAY_FIXED_WORK', 'REPLAY_SEED_ORDER'):
            self.assertIn(name, cx.HARNESS_FLAGS)
            self.assertNotIn(name, reader.COMPOSE_ENVS)
        self.assertEqual((reader.DEFAULTS['REPLAY_FIXED_WORK'], reader.DEFAULTS['REPLAY_SEED_ORDER']), ('0', 'plain'))
        self.assertEqual([cx.seed_order_of('alternate', v) for v in cx.BLOCKS + (cx.DRY,)],
                         ['plain', 'swap', 'plain', 'swap', 'plain'])
        self.assertEqual({cx.seed_order_of('swap', v) for v in cx.BLOCKS}, {'swap'})
        spec, cap = cx.FIXED_WORK['n480']
        self.assertEqual(spec, '1=15,17,14,17,18,14,12,14/2=18,15,16,17,15,16,16,16')
        self.assertEqual(cap, 640)
        for suite in ('test_replay_fixed_work', 'test_fixed_work_packet', 'test_q2024p_j2_long_reader'):
            self.assertIn(suite, cx.SUITES)
        self.assertIn('tools/q2024p_j2_long_reader.py', cx.TOOLS)
        if (HERE / 'n3j2_packet.py').is_file():
            return                                  # a bundle: its own queue is checked by test_settings_asserts_export_gate
        self.assertTrue(Path(cx.hgb.PARENT).is_dir(), 'hub run needs the parent j2 bundle')
        saved = dict(cx.CONFIG)
        got = {}
        try:
            for key, pairs in (('plain', ['REPLAY_REPLY_TEXT=seeded']),
                               ('fixed', ['REPLAY_REPLY_TEXT=seeded', 'REPLAY_FIXED_WORK=n480']),
                               ('alt', ['REPLAY_REPLY_TEXT=seeded', 'REPLAY_FIXED_WORK=n480', 'REPLAY_SEED_ORDER=alternate'])):
                cx.configure(cx.parse_flags(pairs), 'zw')
                got[key] = {v: cx.queues(v) for v in ('b1', 'b2')}
        finally:
            cx.CONFIG.update(saved)
        fixed_line = cx.FIXED_WORK_EXPORT % (spec, cap)
        self.assertIn('export REPLAY_FIXED_WORK=%s REPLAY_FIXED_CAP_S=640 ' % spec, fixed_line)
        for v in ('b1', 'b2'):
            plain = got['plain'][v]
            self.assertFalse(any(n in t for t in plain.values() for n in ('REPLAY_FIXED', 'REPLAY_SEED_ORDER', 'fixed_work_packet')))
            for rel in plain:
                fixed, alt = got['fixed'][v][rel], got['alt'][v][rel]
                if not rel.endswith('.draft.sh'):
                    self.assertEqual((fixed, alt), (plain[rel], plain[rel]), rel)
                    continue
                order_line = cx.SEED_ORDER_EXPORT % ('plain' if v == 'b1' else 'swap')
                self.assertEqual(fixed.replace(fixed_line, ''), plain[rel])
                self.assertEqual(alt.replace(order_line, ''), fixed)
                head = alt.partition(cx.dmp.DRAFT_ANCHOR)[2]
                self.assertTrue(head.startswith(cx.DRAFT_EXPORT % 'chunk' + order_line + fixed_line + cx.REPLY_TEXT_EXPORT), head[:400])
                import subprocess
                import tempfile
                with tempfile.NamedTemporaryFile('w', suffix='.sh') as f:
                    f.write(alt)
                    f.flush()
                    self.assertEqual(subprocess.run(['bash', '-n', f.name]).returncode, 0)

    def test_quiet_split_option(self):
        """Brief 11 C: QUIET_SPLIT_CONTROL=1 exports the gate once after the decode-group export and puts the quiet-SPLIT
        step once in the tail, after the (empty) tail-arm loop and before the tail-engine check, instruments-end and
        end-full-idle (which stay counted); a read_step, never record; 0 (the default) leaves every queue text as it was.
        A settings line without auto-arm, decode merge, lazy receive or the deferred bind is refused at render time."""
        self.assertEqual(cx.parse_flags([])['QUIET_SPLIT_CONTROL'], '0')
        with self.assertRaises(SystemExit):
            cx.parse_flags(['QUIET_SPLIT_CONTROL=on'])
        self.assertIn('QUIET_SPLIT_CONTROL', cx.HARNESS_FLAGS)
        self.assertNotIn('quiet_split_control', cx.EXTRA)
        self.assertEqual(reader.DEFAULTS['QUIET_SPLIT_CONTROL'], '0')
        self.assertIn('test_quiet_split_box', cx.SUITES)
        step = cx.QUIET_SPLIT_STEP
        self.assertNotIn('record ', step.replace('read_step', ''))
        self.assertIn('--hold-s 25 --max-tokens 4096', step)
        self.assertNotIn('--stages', step)                       # v2: the declared plan is the control's default
        self.assertIn('--decode-merge always', step)
        self.assertIn('(( SECONDS > LAST_TAIL_ENTRY_S ))', step)
        # brief 13: the control's timeout is the active budget left after the end reserve and the kill-after, <= 900 s
        self.assertIn('_qs_left=$(( $(native41_budget active) - SECONDS - 360 - 150 ))', step)
        self.assertIn('timeout --signal=TERM --kill-after=150s "${_qs_to}s" python3', step)
        self.assertIn('(( _qs_to < 450 ))', step)
        self.assertEqual((cx.QUIET_SPLIT_MINUTES, cx.minutes_of({'QUIET_SPLIT_CONTROL': '1'}), cx.minutes_of({})),
                         ((62, 93), (62, 93), (54, 81)))
        self.assertIn('NOMINAL_S = 62 * 60  #', cx.QUIET_SPLIT_GATE)
        self.assertTrue(cx.QUIET_SPLIT_GATE.endswith('\n') and cx.QUIET_SPLIT_HEADER.startswith(cx.dmp.J2_HEADER[:-1]))
        if (HERE / 'n3j2_gate.py').is_file() and 'quiet-SPLIT control v2' in (HERE / 'n3j2_gate.py').read_text():
            import n3j2_gate as gate                 # a quiet-split bundle: its gate is the edited one
            self.assertEqual((gate.NOMINAL_S, gate.BOX_BUDGET_S), (3720, 5580))
            self.assertEqual(int(gate.CAP * gate.END_S), 360)
        self.assertIn('[[ $ENGINE_CRASHED != 0 ]]', step)
        if (HERE / 'n3j2_packet.py').is_file():
            return                                  # a bundle: its own queue is checked by test_settings_asserts_export_gate
        self.assertTrue(Path(cx.hgb.PARENT).is_dir(), 'hub run needs the parent j2 bundle')
        saved = dict(cx.CONFIG)
        try:
            cx.configure(cx.parse_flags([]), 'zq')
            plain = cx.queues('b1')
            cx.configure(cx.parse_flags(['QUIET_SPLIT_CONTROL=1']), 'zq')
            quiet = cx.queues('b1')
            needs = cx.QUIET_SPLIT_NEEDS
            cx.QUIET_SPLIT_NEEDS = needs + ('FLASH_S5_NOT_IN_THE_PARENT=1 ',)
            try:
                with self.assertRaises(AssertionError):
                    cx.queues('b1')
            finally:
                cx.QUIET_SPLIT_NEEDS = needs
        finally:
            cx.CONFIG.update(saved)
        self.assertFalse(any('QUIET_SPLIT' in t or 'e3_quiet_split_box' in t for t in plain.values()))
        for rel in plain:
            if not rel.endswith('.draft.sh'):
                self.assertEqual(plain[rel], quiet[rel], rel)
                continue
            text = quiet[rel]
            self.assertEqual(text.count(cx.QUIET_SPLIT_HEADER), 1)
            self.assertEqual(text.replace(cx.QUIET_SPLIT_EXPORT, '').replace(cx.QUIET_SPLIT_STEP, '')
                             .replace(cx.QUIET_SPLIT_HEADER, cx.dmp.J2_HEADER), plain[rel])
            at = text.index(cx.QUIET_SPLIT_STEP)
            self.assertEqual(text.index(cx.QUIET_SPLIT_ANCHOR) + len(cx.QUIET_SPLIT_ANCHOR), at)
            after = text[at:]
            order = [after.index(k) for k in ('record tail-engine', 'record instruments-end', 'record end-full-idle')]
            self.assertEqual(order, sorted(order))
            self.assertLess(text.index('record compose5 '), at)            # the core grade is written before it
            self.assertLess(text.index(cx.QUIET_SPLIT_EXPORT), text.index('python3'))
            # the long command lines keep their bash continuations (a Python string escape once folded them)
            self.assertEqual(cx.QUIET_SPLIT_STEP.count(' \\\n'), 4)
            self.assertIn('"$H/e3_quiet_split_box.py" \\\n', text)
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile('w', suffix='.sh') as f:
                f.write(text)
                f.flush()
                self.assertEqual(subprocess.run(['bash', '-n', f.name]).returncode, 0)

    def test_quiet_split_reader(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertIsNone(reader.quiet_split(root))
            rec = dict(verdict='PASS', stages=[dict(live=8, window=dict(wall_s=25.1), aggregate_tokens_per_s=400.0,
                                                    merge_steps=900, polls=51,
                                                    streams=[dict(window_gap_ms_p50=20.0, window_gap_ms_max=55.0),
                                                             dict(window_gap_ms_p50=22.0, window_gap_ms_max=80.0)])])
            (root / 'quiet-split.json').write_text(json.dumps(rec))
            qs = reader.quiet_split(root)
            self.assertEqual(qs['stages'][0]['gap_ms_max'], 80.0)
            self.assertEqual(qs['stages'][0]['gap_ms_p50_median'], 22.0)
            self.assertEqual(qs['verdict'], 'PASS')
            (root / 'quiet-split.json').write_text('{')
            self.assertEqual(reader.quiet_split(root)['verdict'], 'UNREADABLE')

    def test_decode_timeline_flag_in_the_queues(self):
        """The timeline stamp rides the settings line and the asserts; its module is a new, bound overlay file and its suite
        runs in the frozen CPU. A parent settings line without handler anatomy, auto-arm, burst progress or one fused
        sampling period in 1..4096 is refused at render time (decode_timeline.check_startup would refuse the engine)."""
        if (HERE / 'n3j2_packet.py').is_file():
            return                                  # a bundle: its own queue is checked by test_settings_asserts_export_gate
        self.assertTrue(Path(cx.hgb.PARENT).is_dir(), 'hub run needs the parent j2 bundle')
        flags = cx.parse_flags(['FLASH_S5_DECODE_TIMELINE=1'])
        saved = dict(cx.CONFIG)
        try:
            cx.configure(flags, 'zl')
            texts = cx.queues('b1')
            self.assertIn('e3/decode_timeline.py', cx.hgb.NEW_OVERLAY_FILES)
            for name in ('decode_timeline', 'handler_anatomy', 'native_measurement'):   # every module the stamp moves
                self.assertIn(name, cx.ing.BOUND_MODULES)
            self.assertIn('test_decode_timeline', cx.SUITES)
            settings = next(t for r, t in texts.items() if r.endswith('_settings.sh'))
            self.assertIn('FLASH_S5_DECODE_ANATOMY=0 FLASH_S5_DECODE_TIMELINE=1 ', settings)
            self.check_texts(texts, flags, cx.name_of('b1', 'zl'))
            original = cx._np_queues
            for old, new in (('FLASH_S5_HANDLER_ANATOMY=1 ', 'FLASH_S5_HANDLER_ANATOMY=0 '),
                             ('FLASH_S5_AUTO_ARM=1 ', 'FLASH_S5_AUTO_ARM=0 '),
                             ('FLASH_S5_NATIVE_PROGRESS=burst ', 'FLASH_S5_NATIVE_PROGRESS=0 '),
                             ('FLASH_S5_FUSED_PHASE_EVERY=16 ', 'FLASH_S5_FUSED_PHASE_EVERY=0 '),
                             ('FLASH_S5_FUSED_PHASE_EVERY=16 ', 'FLASH_S5_FUSED_PHASE_EVERY=5000 '),
                             ('FLASH_S5_FUSED_PHASE_EVERY=16 ', '')):
                self.assertIn(old, settings)
                cx._np_queues = lambda variant, old=old, new=new: {
                    r: t.replace(old, new) for r, t in original(variant).items()}
                try:
                    with self.assertRaises(AssertionError, msg=old + '->' + new):
                        cx.queues('b1')
                finally:
                    cx._np_queues = original
            off = cx.parse_flags([])                 # the stamp off: none of its needs is asserted
            cx.configure(off, 'zl')
            cx._np_queues = lambda variant: {r: t.replace('FLASH_S5_FUSED_PHASE_EVERY=16 ', '')
                                             for r, t in original(variant).items()}
            try:
                self.assertIn('FLASH_S5_DECODE_TIMELINE=0 ', next(t for r, t in cx.queues('b1').items()
                                                                if r.endswith('_settings.sh')))
            finally:
                cx._np_queues = original
        finally:
            cx.CONFIG.update(saved)


import decode_liveness_guard as guard_mod  # noqa: E402
import types  # noqa: E402


def quiet_reply():
    """A w13 status reply with the observer off: the decode_step block has no liveness key."""
    return {'native_stage_local': {'overlap': {'decode_step': {'steps': 1}}}}


def live_reply(trips, live=0):
    return {'native_stage_local': {'overlap': {'decode_step': {'liveness': dict(trips=trips, live_max_stall_ms=live,
                                                                                stall_ms=10000)}}}}


class Replies:
    def __init__(self, replies):
        self.replies = list(replies)

    def call(self, action, **fields):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def w13_arm(polls=33, guard=guard_mod, observer='off', replies=None, before=None, after=None):
    """The w13 cx shape (cxay12-b1 sp.g2-live-0: 33 polls, no liveness block anywhere, no other reason)."""
    g = guard.Guard('sp.g2-live-0', clock=lambda: 10.0, observer=observer)
    client = guard.WatchedClient(Replies(replies if replies is not None else [quiet_reply()] * polls), g)
    for _ in range(len(client._client.replies)):
        try:
            client.call('status')
        except OSError:
            pass
    rows = [dict(label='entry'), dict(label='before'), dict(label='after')]
    if before is not None:
        rows[1]['decode_liveness'] = before
    if after is not None:
        rows[2]['decode_liveness'] = after
    rec = dict(reasons=[], complete=True, statuses=rows)
    g.settle(rec)
    return g, rec


def mutant(*edits):
    """The guard module with source edits, as a fresh module (the suite's own guard is untouched)."""
    source = Path(guard_mod.__file__).read_text()
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    module = types.ModuleType('decode_liveness_guard_mutant')
    exec(compile(source, 'decode_liveness_guard_mutant', 'exec'), module.__dict__)
    return module


def off_is_not_applicable(guard=guard_mod):
    g, rec = w13_arm(guard=guard)
    view = rec['decode_liveness_guard']
    return (rec['reasons'], rec['complete'], view['liveness'], view['absent_polls'], view['blind_polls'],
            view['observer']) == ([], True, 'NOT_APPLICABLE', 33, 0, 'off')


def on_blind_is_inc(guard=guard_mod):
    g, rec = w13_arm(guard=guard, observer='on')
    view = rec['decode_liveness_guard']
    return (rec['reasons'], rec['complete'], view['blind_polls'], view['liveness']) == (
        ['decode-liveness-guard:HARNESS:guard-blind'], False, 33, 'read')


def off_with_a_block_is_inc(guard=guard_mod):
    _, polled = w13_arm(guard=guard, replies=[quiet_reply(), live_reply(0), quiet_reply()])
    _, rowed = w13_arm(guard=guard, polls=2, after=dict(trips=0))
    want = ['decode-liveness-guard:HARNESS:observer-declared-off']
    return polled['reasons'] == want and rowed['reasons'] == want and not polled['complete'] and not rowed['complete']


class GuardObserver(unittest.TestCase):
    """Brief 11 (window 13: every cx packet INC only for guard-blind on all four arms, the observer off): the binding
    declares the observer; off reads NOT_APPLICABLE and is not missing, on still reads a blind guard as INC."""

    def test_off_is_not_applicable_and_not_missing(self):
        self.assertTrue(off_is_not_applicable())

    def test_on_blind_is_still_inc(self):
        self.assertTrue(on_blind_is_inc())
        g, rec = w13_arm(observer='on', polls=0)                # rows only, no poll: settle alone still reads blind
        self.assertEqual(rec['reasons'], ['decode-liveness-guard:HARNESS:guard-blind'])

    def test_off_with_a_liveness_block_is_inc(self):
        self.assertTrue(off_with_a_block_is_inc())

    def test_off_keeps_the_poll_watch(self):
        g, rec = w13_arm(replies=[OSError()] * guard_mod.POLL_FAILS)
        self.assertEqual(rec['reasons'], ['decode-liveness-guard:ENGINE:status-unanswered'])
        self.assertTrue(rec['decode_liveness_guard']['ended_replay'])
        g, rec = w13_arm(replies=[live_reply(0), live_reply(1, 12000)])   # a block seen off: still read and still ends
        self.assertEqual(rec['reasons'], ['decode-liveness-guard:ENGINE:engine-stuck',
                                          'decode-liveness-guard:HARNESS:observer-declared-off'])

    def test_on_reads_as_before(self):
        g, rec = w13_arm(observer='on', replies=[live_reply(0), live_reply(0)], before=dict(trips=0), after=dict(trips=0))
        self.assertEqual((rec['reasons'], rec['complete'], rec['decode_liveness_guard']['liveness']), ([], True, 'read'))

    def test_declaration_is_checked(self):
        for bad in ('0', '1', None, 'OFF'):
            with self.assertRaises(ValueError):
                guard_mod.Guard('x', observer=bad)
            with self.assertRaises(ValueError):
                guard_mod.guarded(lambda *a, **k: {}, lambda: None, bad)
        box = types.SimpleNamespace(replay_arm=lambda *a, **k: {}, _box=None)
        bound = guard_mod.install(box, observer='off')
        self.assertIs(guard_mod.install(box, observer='off'), bound)
        with self.assertRaises(ValueError):
            guard_mod.install(box)                              # a second declaration is refused
        self.assertEqual(guard_mod.install(types.SimpleNamespace(replay_arm=lambda *a, **k: {}, _box=None))
                         .decode_liveness_observer, 'on')       # the dmp binding's default

    def test_builder_binds_the_packet_setting(self):
        self.assertEqual(cx.OBSERVER_OF, {'0': 'off', '1': 'on'})
        for liveness, observer in cx.OBSERVER_OF.items():
            text = cx.box_import(liveness)
            self.assertEqual(text.count("decode_liveness_guard.install(e3_compose5_box, observer='%s')" % observer), 1)
            self.assertEqual(text.count('decode_liveness_guard.install('), 1)
            self.assertEqual(text.replace(cx.GUARD_INSTALL_NEW % observer, cx.GUARD_INSTALL), cx.dmp.BOX_IMPORT_NEW)
        with mock.patch.object(cx.dmp, 'BOX_IMPORT_NEW', cx.dmp.BOX_IMPORT_NEW.replace(cx.GUARD_INSTALL, '')):
            with self.assertRaises(AssertionError):
                cx.box_import('0')                              # the dmp line moved: refused, not silently unbound
        if (HERE / 'n3j2_packet.py').is_file():
            return                                              # a bundle: its own box is checked with its settings
        self.assertTrue(Path(cx.hgb.PARENT).is_dir(), 'hub run needs the parent j2 bundle')
        saved, imp = dict(cx.CONFIG), cx.hgb.BOX_IMPORT_NEW
        try:
            for liveness in ('0', '1'):
                flags = cx.parse_flags(['FLASH_S5_DECODE_LIVENESS=' + liveness])
                cx.configure(flags, 'zg')
                self.assertEqual(cx.hgb.BOX_IMPORT_NEW, cx.box_import(liveness))
        finally:
            cx.CONFIG.update(saved)
            cx.hgb.BOX_IMPORT_NEW = imp


class GuardObserverMutants(unittest.TestCase):
    """Each mutant breaks one direction and is caught by the rule it breaks."""
    OFF_POLL = ("                if self.observer == 'off':\n                    self.absent += 1\n"
                "                    return\n")

    def test_off_read_as_on_is_caught(self):
        m = mutant((self.OFF_POLL, ''))
        self.assertFalse(off_is_not_applicable(m))
        self.assertTrue(on_blind_is_inc(m))

    def test_settle_blind_without_the_declaration_is_caught(self):
        m = mutant(("'after' in rows and self.observer == 'on':", "'after' in rows:"))
        self.assertFalse(off_is_not_applicable(m))

    def test_on_read_as_off_is_caught(self):
        m = mutant((self.OFF_POLL, self.OFF_POLL.replace("self.observer == 'off'", 'True')),
                   ("'after' in rows and self.observer == 'on':", "'after' in rows and False:"))
        self.assertTrue(off_is_not_applicable(m))
        self.assertFalse(on_blind_is_inc(m))

    def test_contradiction_unread_is_caught(self):
        m = mutant(("            if self.observer == 'off':\n                self.harness = self.harness or "
                    "'observer-declared-off'\n", ''))
        self.assertFalse(off_with_a_block_is_inc(m))
        m = mutant(("        if self.observer == 'off' and self.harness is None and (isinstance(before, dict) or "
                    "isinstance(after, dict)):\n            self.harness = 'observer-declared-off'\n", ''))
        self.assertFalse(off_with_a_block_is_inc(m))

    def test_builder_always_on_is_caught(self):
        with mock.patch.dict(cx.OBSERVER_OF, {'0': 'on'}):
            self.assertNotIn("observer='off'", cx.box_import('0'))   # what the bundle assertion then refuses
        built = cx.dmp.BOX_IMPORT_NEW.replace(cx.GUARD_INSTALL, cx.GUARD_INSTALL_NEW % 'on')
        self.assertEqual(built.count(cx.GUARD_INSTALL_NEW % 'off'), 0)


import inspect  # noqa: E402


def work_turns():
    """Agent 0: a finished fresh and append; agent 1: the report's case, a 128k fresh sent at 454 s whose first token
    lands at 494 s; agent 2: an append-shaped first row open at W with half its chunks in; agent 3: an error row."""
    chunks = lambda a, b, n: [a + (b - a) * k // (n - 1) for k in range(n)]
    return [
        dict(agent=0, session=1, kind='fresh', t=0.0, ttft=10.0, seconds=20.0, prompt_tokens=100000, cached_tokens=0,
             completion_tokens=500, arrivals_ms=chunks(10000, 20000, 11)),
        dict(agent=0, session=1, kind='append', t=25.0, ttft=1.0, seconds=5.0, prompt_tokens=103000,
             cached_tokens=99968, completion_tokens=300, arrivals_ms=chunks(26000, 30000, 5)),
        dict(agent=1, session=3, kind='fresh', t=454.0, ttft=40.0, seconds=60.0, prompt_tokens=128000, cached_tokens=0,
             completion_tokens=400, arrivals_ms=chunks(494000, 514000, 11)),
        dict(agent=2, session=1, kind='append', t=470.0, ttft=2.0, seconds=20.0, prompt_tokens=5000, cached_tokens=4800,
             completion_tokens=200, arrivals_ms=chunks(472000, 490000, 10)),
        dict(agent=3, session=1, kind='fresh', t=100.0, error='URLError:boom'),
    ]


def work_rule_holds(fn):
    w = fn(work_turns(), 480.0, 0.02)
    return (w['rows_W'], w['out_W'], w['work_W'], w['turns_completed'], len(w['open']), w['open_rows'],
            w['open_rows_W'], w['errors'], round(w['frac_turns'], 3)) == (
        191200, 900, 4724.0, 2, 2, 133000, 88200, 1, round(2 + 1664 / 2960 + 200 / 300, 3))


def work_mutant(*edits, fn='arm_work'):
    """reader.<fn> with source edits in a copy of the reader's globals; arm_work rebuilt there to call it."""
    source = inspect.getsource(getattr(reader, fn))
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    space = dict(reader.__dict__)
    exec(compile(source, fn + '_mutant', 'exec'), space)
    if fn != 'arm_work':
        exec(compile(inspect.getsource(reader.arm_work), 'arm_work_copy', 'exec'), space)
    return space['arm_work']


def fake_block(label, R, works):
    """works: arm -> (seed, work_W, turns, seq_work)."""
    return dict(label=label, R=R, work={n: dict(seed=s, work_W=w, turns=t, seq_work=q, window_s=480.0, cost_ratio=0.02)
                                        for n, (s, w, t, q) in works.items()})


class WorkInWindow(unittest.TestCase):
    """Brief 11 B.2: work done inside the window per arm, R_w per block and G_w beside R and G."""

    def test_arm_work(self):
        self.assertTrue(work_rule_holds(reader.arm_work))
        w = reader.arm_work(work_turns(), 480.0, 0.02)
        big = next(x for x in w['open'] if x['agent'] == 1)
        self.assertEqual((big['rows'], big['rows_W'], big['out_W'], big['kind']), (128000, 83200, 0.0, 'fresh'))

    def test_open_turn_order_and_an_error_reset(self):
        rows = work_turns()
        rows.append(dict(agent=3, session=2, kind='fresh', t=200.0, ttft=5.0, seconds=9.0, prompt_tokens=60000,
                         completion_tokens=300, arrivals_ms=[205000, 209000]))
        shuffled = list(reversed(rows))
        self.assertEqual(reader.arm_work(rows, 480.0, 0.02), reader.arm_work(shuffled, 480.0, 0.02))
        rows.append(dict(agent=3, session=2, kind='append', t=300.0, ttft=1.0, seconds=3.0, prompt_tokens=62000,
                         completion_tokens=250, arrivals_ms=[301000, 303000]))
        w = reader.arm_work(rows, 480.0, 0.02)
        self.assertEqual(w['rows_W'], 191200 + 60000 + 2000)          # the append after the error's fresh: growth only

    def test_cohort_ratio_pinned_turn_work_flag_and_gap(self):
        blocks = [fake_block('b1', 1.10, {'sp.g2-live-0': (1, 900.0, 10, 9000.0), 'sp.g2-live-1': (2, 800.0, 10, 8000.0),
                                          'n3.g2-live-0': (1, 990.0, 10, 9000.0), 'n3.g2-live-1': (2, 880.0, 10, 8000.0)}),
                  fake_block('b2', 1.04, {'sp.g2-live-0': (1, 900.0, 10, 9000.0), 'sp.g2-live-1': (2, 800.0, 10, 8000.0),
                                          'n3.g2-live-0': (1, 1008.0, 10, 9000.0), 'n3.g2-live-1': (2, 896.0, 10, 8000.0)})]
        c = reader.work_cohort(blocks)
        self.assertEqual(c['turn_work'], {'1': 900.0, '2': 800.0})
        self.assertEqual(c['R_w'], {'b1': 1.1, 'b2': 1.12})
        self.assertEqual(c['G_w'], round(math.sqrt(1.1 * 1.12), 4))
        self.assertEqual(c['flagged'], ['b1', 'b2'])                   # R above / below its mean, R_w the other way
        self.assertEqual(c['gap'], {'b1': 0.0, 'b2': round(math.log(1.04 / 1.12), 4)})
        pinned = reader.work_cohort(blocks, {1: 450.0, 2: 800.0})
        self.assertEqual(pinned['turn_work'], {'1': 450.0, '2': 800.0})
        self.assertEqual(pinned['turn_work_pinned'], ['1', '2'])
        want = (990 / 450 + 880 / 800) / (900 / 450 + 800 / 800)
        self.assertEqual(pinned['R_w']['b1'], round(want, 4))
        blocks[1]['R'] = 1.12
        self.assertEqual(reader.work_cohort(blocks)['flagged'], [])     # same direction: not flagged
        blocks[1]['work']['sp.g2-live-1'] = None
        self.assertEqual(reader.work_cohort(blocks)['blocks_without_R_w'], ['b2'])

    def test_cost_ratio_is_named_in_the_output(self):
        self.assertEqual(reader.WORK_COST_RATIO, 0.0218)
        block = fake_block('b1', 1.1, {n: (1, 100.0, 1, 100.0) for n in reader.SP_ARMS + reader.N3_ARMS})
        self.assertEqual(reader.work_cohort([block])['cost_ratio'], 0.02)
        with self.assertRaises(SystemExit):
            reader.main(['b1=/nonexistent', '--cost-ratio', '0'])
        with self.assertRaises(SystemExit):
            reader.main(['b1=/nonexistent', '--turn-work', '2=-1'])


class WorkMutants(unittest.TestCase):
    def test_cached_tokens_for_rows_is_caught(self):
        self.assertFalse(work_rule_holds(work_mutant(
            ("need = t['prompt_tokens'] if", "need = t['prompt_tokens'] - t.get('cached_tokens', 0) if False and"),
            fn='turn_rows')))

    def test_no_proration_is_caught(self):
        self.assertFalse(work_rule_holds(work_mutant(
            ('pr = need * (1.0 if f <= window else', 'pr = need * (1.0 if True else'))))

    def test_arrivals_shifted_by_send_is_caught(self):
        self.assertFalse(work_rule_holds(work_mutant(('x / 1000.0 <= window', 't["t"] + x / 1000.0 <= window'))))

    def test_open_turn_counted_done_is_caught(self):
        self.assertFalse(work_rule_holds(work_mutant(('if e <= window:\n            done += 1', 'if s <= window:\n            done += 1'))))


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
        """Both overflow fields: the reviewed fix's overflow_rows_retired (2eba0dd0) and the pre-fix tie key."""
        for key in ('overflow_rows_retired', 'tie_overflow_lagged'):
            with self.subTest(key=key):
                self.OVERFLOW_KEY = key
                self.sample_fast_case()

    def test_sample_fast_overflow_in_flight_reads_none(self):
        """A retired copy not yet read (None, the CUDA event pending) is skipped; an arm without one reads None."""
        import datetime
        root = record(dict(FLASH_S5_DECODE_SAMPLE_FAST='1'))
        t0 = self.arm_times(root)

        def line(t, fast, seen):
            stamp = datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            body = dict(schema='s5-decode-step/1', rank=3, overflow_rows_retired=seen,
                        counters=dict(fast=fast, stock=0))
            return '%s (Worker_PP1_TP1 pid=1) INFO flash-s5 sample-fast %s\n' % (stamp, json.dumps(body))
        with (root / 'engine-stream.log').open('a') as handle:
            handle.writelines([line(t0 + 650, 10, 2), line(t0 + 750, 20, 3), line(t0 + 800, 30, None),
                               line(t0 + 1500, 40, None)])
        r = reader.collect(root, 'b1', reader.flags_of(['FLASH_S5_DECODE_SAMPLE_FAST=1']))
        self.assertEqual(r['arms']['n3.g2-live-0']['sample_fast']['3']['overflow_rows'], 1)   # 3 - 2, None skipped
        self.assertIsNone(r['arms']['n3.g2-live-1']['sample_fast']['3']['overflow_rows'])

    @staticmethod
    def arm_times(root):
        t0 = 1794436590.0
        order = ('sp.g2-live-0', 'n3.g2-live-0', 'n3.g2-live-1', 'sp.g2-live-1')
        for i, name in enumerate(order):                       # sequential arms, 600 s each, 100 s apart
            path = root / ('arm-' + name + '.json')
            arm = json.loads(path.read_text())
            arm['replay_start_s'] = t0 + 700 * i
            arm.setdefault('replay_config', {})['DURATION'] = 600
            path.write_text(json.dumps(arm))
        return t0

    def anatomy_log(self, root, t0, sample_ms, native_ms):
        """One clean window (core line in the sm smoke arm, before the graded arms) with sample_tokens rows on ranks
        2 and 3, an execute_model row, a row outside it in n3.g2-live-0 (native context), a torn handler line."""
        import datetime

        def stamp(t):
            return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        base, open_ns, close_ns = 10 ** 12, 10 ** 12 + 5 * 10 ** 9, 10 ** 12 + 25 * 10 ** 9
        core = dict(schema='s5-decode-step/1', kind='core-steps', window=1, open_ns=open_ns, close_ns=close_ns,
                    fields=['t_us', 'batch_queue', 'running', 'waiting'], counters={}, part=0, parts=1, stamps=[])
        lines = ['%s (EngineCore pid=1) INFO flash-s5 decode-anatomy %s\n' % (stamp(t0 - 60), json.dumps(core))]

        def row(seq, kind, t_us, dev_ms):
            return [seq, kind, 0, 0, t_us, t_us + 900, 900000, 800000, 1000, None if kind == 's' else 700000,
                    None if dev_ms is None else int(dev_ms * 1e6), None, None, None, []]
        for rank in (2, 3):
            m = [row(i, 's', 6 * 10 ** 6 + i * 1000, ms) for i, ms in enumerate(sample_ms[rank])]
            m.append(row(99, 'e', 7 * 10 ** 6, None))
            body = dict(schema='s5-handler-anatomy/1', rank=rank, line=1, base_ns=base, part=0, parts=1,
                        rows=dict(m=m, a=[]))
            lines.append('%s (Worker_PP1_TP%d pid=1) INFO flash-s5 handler %s\n' % (stamp(t0 - 50), rank - 2,
                                                                                  json.dumps(body)))
            native = dict(body, line=2, rows=dict(m=[row(200, 's', 40 * 10 ** 6, native_ms)], a=[]))
            lines.append('%s (Worker_PP1_TP%d pid=1) INFO flash-s5 handler %s\n' % (stamp(t0 + 800), rank - 2,
                                                                                  json.dumps(native)))
        lines.append('%s (Worker_PP1_TP0 pid=1) INFO flash-s5 handler {"schema": "s5-handler-anat\n' % stamp(t0 + 801))
        with (root / 'engine-stream.log').open('a') as handle:
            handle.writelines(lines)

    def test_handler_parse_pinned_to_the_writer_and_the_torn_rule(self):
        """The reader parses handler lines itself (stage1_anatomy_report is not in a bundle): its schema, main-row
        fields and dropped flags equal the writer's, its torn rule is step_a_torn_lines.INNER_HEADER."""
        import ast as ast_
        path = next(p for p in (HERE.parent / 'patches/s5_engine/handler_anatomy.py',
                                HERE / 'native30_overlay/e3/handler_anatomy.py',
                                HERE / 'e3_builder/handler_anatomy.py') if p.is_file())
        consts = {}
        for node in ast_.parse(path.read_text()).body:
            if isinstance(node, ast_.Assign):
                try:
                    value = ast_.literal_eval(node.value)
                except ValueError:
                    continue
                for target in node.targets:
                    if isinstance(target, ast_.Name):
                        consts[target.id] = value
                    elif isinstance(target, ast_.Tuple) and isinstance(value, tuple):
                        consts.update({e.id: v for e, v in zip(target.elts, value) if isinstance(e, ast_.Name)})
        self.assertEqual(consts['SCHEMA'], reader.HANDLER_SCHEMA)
        self.assertEqual(tuple(consts['M_FIELDS']), reader.HANDLER_M_FIELDS)
        self.assertEqual(consts['FLAG_FAILED'] | consts['FLAG_ABORTED'] | consts['FLAG_INSTALL'], reader.HANDLER_BAD_FLAGS)
        import step_a_torn_lines
        self.assertEqual(step_a_torn_lines.INNER_HEADER.pattern, reader.INNER_HEADER.pattern)
        whole = '2026-11-13T00:00:00.000000000Z (Worker_PP1_TP0 pid=1) INFO flash-s5 handler {"schema": "x"}'
        self.assertEqual(reader.handler_body(whole), {'schema': 'x'})
        self.assertIsNone(reader.handler_body(whole[:-1] + '(Worker_PP1_TP1 pid=2) INFO }'))
        self.assertIsNone(reader.handler_body(whole[:-3]))

    def test_dev_sample_clean_window_native_context_and_the_d_pair(self):
        out = {}
        for side, env, sample in (('off', {}, (1.0, 1.1, 1.2, 1.3)), ('on', {'FLASH_S5_DECODE_SAMPLE_FAST': '1'},
                                                                         (0.8, 0.9, 1.0, 1.1))):
            env = dict(env, FLASH_S5_DECODE_ANATOMY='1')
            root = record(env)
            t0 = self.arm_times(root)
            self.anatomy_log(root, t0, {2: sample, 3: tuple(x + 0.02 for x in sample)}, 2.0)
            r = reader.collect(root, side, reader.flags_of(['%s=%s' % kv for kv in sorted(env.items())]))
            out[side] = r
        d = out['on']['dev_sample']
        self.assertEqual(d['clean']['1']['arm'], 'before-arms')
        self.assertEqual(d['clean']['1']['close_minus_open_s'], 20.0)
        self.assertEqual(d['clean']['1']['ranks']['2'], dict(n=4, p50=0.9, p90=1.1))     # execute_model row left out
        self.assertEqual(d['native'], {'n3.g2-live-0': {'2': dict(n=1, p50=2.0, p90=2.0), '3': dict(n=1, p50=2.0, p90=2.0)}})
        self.assertEqual(d['torn_lines'], 1)
        self.assertFalse([n for n in out['on']['compose_notes'] if n.startswith('decode-anatomy-')])
        pair = reader.d_pair(out['off'], out['on'])
        self.assertEqual((pair['2']['drop_ms'], pair['2']['falls_by_0_1_ms']), (0.2, True))
        self.assertEqual(pair['3']['off']['p50'], 1.12)
        text = reader.render([out['on']])
        self.assertIn('    clean window 1 in before-arms (20.0s) {"2":{"n":4,"p50":0.9,"p90":1.1}', text)
        self.assertIn('    native-context n3.g2-live-0 {"2":', text)
        # no clean row: a NOTE; anatomy off: no dev_sample at all
        root = record(dict(FLASH_S5_DECODE_ANATOMY='1'))
        self.arm_times(root)
        quiet = reader.collect(root, 'b9', reader.flags_of(['FLASH_S5_DECODE_ANATOMY=1']))
        self.assertIn('decode-anatomy-no-clean-sample-rows-b9', quiet['compose_notes'])
        self.assertNotIn('dev_sample', reader.collect(record(), 'b8', reader.flags_of([])))
        same = reader.d_pair(out['off'], out['off'])
        self.assertEqual((same['2']['drop_ms'], same['2']['falls_by_0_1_ms']), (0.0, False))

    def test_d_pair_main_refuses_a_second_difference(self):
        import contextlib
        import io
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix='cx-dpair-'))
        docs = {}
        for side, flags in (('off', {'FLASH_S5_DECODE_ANATOMY': '1'}),
                            ('on', {'FLASH_S5_DECODE_ANATOMY': '1', 'FLASH_S5_DECODE_SAMPLE_FAST': '1'}),
                            ('bad', {'FLASH_S5_DECODE_ANATOMY': '1', 'FLASH_S5_DECODE_SAMPLE_FAST': '1',
                                     'FLASH_S5_YIELD_WINDOW': '1'})):
            docs[side] = tmp / (side + '.json')
            docs[side].write_text(json.dumps(dict(schema='q2024p-j2-cx-cohort/1', flags=flags)))
        roots = {}
        for side, env in (('off', {'FLASH_S5_DECODE_ANATOMY': '1'}),
                          ('on', {'FLASH_S5_DECODE_ANATOMY': '1', 'FLASH_S5_DECODE_SAMPLE_FAST': '1'})):
            roots[side] = record(env)
            t0 = self.arm_times(roots[side])
            self.anatomy_log(roots[side], t0, {2: (1.0, 1.2), 3: (1.0, 1.2)}, 2.0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            reader.main(['--cohort-of', 'off=%s' % docs['off'], '--cohort-of', 'on=%s' % docs['on'],
                         'off=%s' % roots['off'], 'on=%s' % roots['on']])
        self.assertIn('== D pair (decode-step builder: keep D only if dev_sample falls by >= 0.1 ms', buf.getvalue())
        self.assertIn('  rank2 off={"n":2,"p50":1.0,"p90":1.2} on={"n":2,"p50":1.0,"p90":1.2} drop_ms=0.0 '
                      'falls_by_0.1=False', buf.getvalue())
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            reader.main(['--cohort-of', 'off=%s' % docs['off'], '--cohort-of', 'on=%s' % docs['bad'],
                         'off=%s' % roots['off'], 'on=%s' % roots['on']])

    def sample_fast_case(self):
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
            body = dict(schema='s5-decode-step/1', rank=rank, **{self.OVERFLOW_KEY: tie},
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
        self.assertEqual((n1['3']['fast'], n1['3']['stock'], n1['3']['overflow_rows']), (8096, 16, 1))
        self.assertEqual((n1['2']['steps'], n1['2']['overflow_rows'], n0['2']['overflow_rows']), (8112, 0, 0))
        self.assertEqual(n1['2']['fast_share'], round(8096 / 8112, 4))
        sp1 = r['arms']['sp.g2-live-1']['sample_fast']['2']
        self.assertEqual((sp1['fast'], sp1['stock_by_reason']), (0, {'not_gated': 1024}))
        self.assertEqual(r['arms']['sp.g2-live-0']['sample_fast'], {})     # before D's first line: none
        self.assertFalse([n for n in r['compose_notes'] if n.startswith('sample-fast-')])
        text = reader.render([r])
        self.assertIn('  n3.g2-live-0  sample_fast rank2 steps=5024 fast=5000 overflow_rows=0 stock_fallback=24 '
                      'share=0.9952 stock_by_reason={"k_mixed":24} lines=2', text)
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


def time_packet(root, speed=2.0, targets=None, per_agent=6, seeds=(1, 2), cap_hit=False, drop=None):
    """A packet dir with the four replay journals: every agent's turns 10 s apart on stock, ``speed`` x faster on n3;
    with ``targets`` each arm's record carries replay_summary.fixed_work (the turns run to per_agent either way)."""
    root.mkdir()
    for (index, sp_name, n3_name), seed in zip(reader.PAIRS, seeds):
        for name, step in ((sp_name, 10.0), (n3_name, 10.0 / speed)):
            rows = []
            for agent in range(2):
                prompt = 1000
                for k in range(per_agent):
                    if drop == (name, agent, k):
                        continue
                    t = k * step + agent * step / 10
                    rows.append(dict(agent=agent, session=1, kind='fresh' if k == 0 else 'append', t=t, ttft=step / 4,
                                     seconds=step * 0.9, prompt_tokens=prompt, completion_tokens=50, reply_target=50 + k,
                                     arrivals_ms=[int((t + step / 4) * 1000)], complete=True))
                    prompt += 500
            head = dict(header=True, label=name, replay_start_s=1.0, replay_config=dict(DURATION=480.0, SEED=seed))
            (root / ('arm-%s.json.turns.jsonl' % name)).write_text(''.join(json.dumps(r) + '\n' for r in [head] + rows))
            summary = {}
            if targets is not None:
                summary['fixed_work'] = dict(targets=list(targets), cap_s=640.0, cap_hit=cap_hit, all_reached=not cap_hit,
                                             end_s=50.0, reached_s=[40.0, 50.0])
            (root / ('arm-%s.json' % name)).write_text(json.dumps(dict(label=name, replay_summary=summary)))
    return root


class TimeToTurns(unittest.TestCase):
    """Brief 12: R_t (the long reader's time to the same turns) per pair, per block (16 streams) and per cohort."""

    def loaded(self, root):
        return {n: json.loads((root / ('arm-%s.json' % n)).read_text()) for n in reader.SP_ARMS + reader.N3_ARMS}

    def test_engine_twice_as_fast_reads_two(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = time_packet(Path(tmp) / 'p')
            got = reader.time_to_turns(root, self.loaded(root))
        self.assertEqual(got['notes'], [])
        self.assertAlmostEqual(got['block']['whole']['R'], 2.0, places=4)
        self.assertEqual(got['block']['coverage'], 1.0)
        self.assertFalse(got['block']['fixed_work'])
        self.assertEqual([got['pairs'][i]['seed'] for i in ('0', '1')], [1, 2])

    def test_fixed_work_caps_k_at_the_target(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = time_packet(Path(tmp) / 'p', speed=2.0, targets=[3, 5])
            got = reader.time_to_turns(root, self.loaded(root))
            pr = got['pairs']['0']
            self.assertTrue(got['block']['fixed_work'])
            self.assertEqual((pr['matched_turns'], pr['stock_turns'], pr['coverage']), (8, 8, 1.0))
            # stock: agent 0 ends turn 3 at 2*10 + 9 = 29, agent 1 turn 5 at 40 + 1 + 9 = 50; n3 at half the step
            self.assertAlmostEqual(pr['whole']['sp_s'], 29.0 + 50.0, places=3)
            self.assertAlmostEqual(pr['whole']['n3_s'], 14.5 + 25.0, places=3)
            root2 = time_packet(Path(tmp) / 'q', targets=[3, 5], cap_hit=True)
            self.assertIn('fixed-work-cap-hit-sp.g2-live-0', reader.time_to_turns(root2, self.loaded(root2))['notes'])
            root3 = time_packet(Path(tmp) / 'r', targets=[3, 5], drop=('n3.g2-live-1', 1, 2))
            got3 = reader.time_to_turns(root3, self.loaded(root3))
            self.assertIn('fixed-work-short-1:1', got3['notes'])
            self.assertLess(got3['block']['coverage'], 1.0)

    def test_a_missing_journal_or_a_seed_mismatch_gives_no_block(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root = time_packet(Path(tmp) / 'p')
            (root / 'arm-n3.g2-live-1.json.turns.jsonl').rename(root / 'moved')
            got = reader.time_to_turns(root, self.loaded(root))
            self.assertIsNone(got['block'])
            self.assertIsNone(got['pairs']['1'])
            root2 = time_packet(Path(tmp) / 'q')
            path = root2 / 'arm-n3.g2-live-0.json.turns.jsonl'
            lines = path.read_text().splitlines()
            head = json.loads(lines[0]); head['replay_config']['SEED'] = 2
            path.write_text('\n'.join([json.dumps(head)] + lines[1:]) + '\n')
            got2 = reader.time_to_turns(root2, self.loaded(root2))
            self.assertIn('time-pair-config-differs-0', got2['notes'])
            self.assertIsNone(got2['block'])

    def test_cohort_line(self):
        blocks = [dict(label='b%d' % i, time_to_turns=dict(block=dict(whole=dict(R=r), from_180=dict(R=r - 0.02),
                                                                      coverage=1.0, diverged_streams=0, fixed_work=True)))
                  for i, r in ((1, 1.10), (2, 1.12), (3, 1.08), (4, 1.11))]
        c = reader.time_cohort(blocks + [dict(label='b5', time_to_turns=dict(block=None))])
        self.assertAlmostEqual(c['whole']['G_t'], (1.10 * 1.12 * 1.08 * 1.11) ** 0.25, places=5)
        self.assertEqual(c['whole']['blocks'], 4)
        self.assertEqual(c['blocks_without_R_t'], ['b5'])
        logs = [math.log(x) for x in (1.10, 1.12, 1.08, 1.11)]
        mean = sum(logs) / 4
        self.assertEqual(c['whole']['sd_ln_R_t'], round(math.sqrt(sum((v - mean) ** 2 for v in logs) / 3), 4))

    def test_k_not_capped_is_caught(self):
        """Mutant: the target cap dropped, a fixed-work pair reads the whole prefix (the review's 480 s statistic)."""
        import tempfile
        source = inspect.getsource(reader.pair_time)
        old = "            a['K'] = min(a['K'], n)\n"
        self.assertEqual(source.count(old), 1)
        space = dict(reader.__dict__)
        exec(compile(source.replace(old, ''), 'pair_time_mutant', 'exec'), space)
        with tempfile.TemporaryDirectory() as tmp:
            root = time_packet(Path(tmp) / 'p', targets=[3, 5])
            _h, sp = reader.lr.load(root / 'arm-sp.g2-live-0.json.turns.jsonl')
            _h, n3 = reader.lr.load(root / 'arm-n3.g2-live-0.json.turns.jsonl')
        good = reader.pair_time(sp, n3, math.inf, [3, 5])
        bad = space['pair_time'](sp, n3, math.inf, [3, 5])
        self.assertNotEqual((good['matched_turns'], good['whole']), (bad['matched_turns'], bad['whole']))


if __name__ == '__main__':
    unittest.main()
