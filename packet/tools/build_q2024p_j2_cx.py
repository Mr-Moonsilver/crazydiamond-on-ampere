#!/usr/bin/env python3
"""Compose one block of a registered q2024p-j2 composition cohort (lead 12 Nov 2026, branch claude/n3-compose).

A cohort is four blocks cx<tag>-b1 .. cx<tag>-b4 (tools/q2024p_j2_cx_render.py), each an ordinary four-arm j2 packet
(sp.g2-live-0, n3.g2-live-0, n3.g2-live-1, sp.g2-live-1 at 480 s, j2's seeds: the V1 cohort's shape), all from ONE
source commit and ONE flag set given on the command line; a block differs from its siblings only in its name.

Method: the night-pack tool (tools/build_q2024p_j2_np.py, through it the ing, yr, yrs and host-gap batch tools) with
  * the flag set: the night pack's five flags and the ingress verify settings, plus
    FLASH_S5_APPEND_NO_BLOCK_STOP (lever A), FLASH_S5_DECODE_GROUPS (chunk|off|off-small; harness/n3_decode_groups) and
    FLASH_S5_DECODE_LIVENESS (the engine's decode liveness observer the arm guard reads); all in the settings line and
    asserted by the declaration and the draft;
  * the bundle's n3j2_packet.configure wrapped by n3_decode_groups (marked tail), the draft exporting
    FLASH_S5_DECODE_GROUPS before its first python, so every process the draft starts reads the same table;
  * the arm liveness guard (harness/decode_liveness_guard: ends an arm only on a still-stuck decoder) bound in the
    bundle's n3j2_box with its status-row line, and the arm gate closed by ENGINE_CRASHED (the dmp-b review fix);
  * the engine modules the composition moved (stage_overlap, native_overlap_control) bound as well, and the CPU suites
    of every lever (test_decode_merge_probe stays a hub suite: its bundle test needs the dmp table and budget).
  * FLASH_S5_CONTROL_INWAIT (Codex, a15bed1d: FULL-entry identity observed beside decode, the duplicate barrier
    skipped) in the settings line and asserted; its CPU suite test_control_inwait runs in the frozen CPU with its frozen
    off-path parent (46d28946's native_auto_arm.py, sha-pinned by the suite) copied into the bundle and named by
    CONTROL_INWAIT_PARENT_SOURCE (the container has no git).
Refused flag sets: settle with the yield window (settle sits where the window hands off), a decode-group mode other than
chunk without the liveness observer (the guard would be blind), verify settings without ingress, and CONTROL_INWAIT with
any of settle, the small-piece batch, ingress, a decode-group mode other than chunk or the liveness observer (none of
these pairs is tested together; test_control_inwait covers A/Y1/Y2, test_n3_compose_pairs A x Y2 on the whole window).
No box command, GPU, VM, deletion or push.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


np = _load('build_q2024p_j2_np', 'tools/build_q2024p_j2_np.py')
dmp = _load('build_q2024p_j2_dmp', 'tools/build_q2024p_j2_dmp.py')
ing, yr, yrs, hgb = np.ing, np.yr, np.yrs, np.hgb

PARENT_NAME = hgb.PARENT_NAME
FAMILY = 'q2024p-j2-cx'
# flag -> (VARIANTS key, allowed values, default); the order is the settings line's
FLAGS = (
    ('FLASH_S5_SMALL_PIECE_BATCH', 'small_piece_batch', ('0', '1'), '0'),
    ('FLASH_S5_YIELD_REENTRY_SETTLE', 'yield_reentry_settle', ('0', '1'), '0'),
    ('FLASH_S5_YIELD_NO_APPEND_OWNER', 'no_append_owner', ('0', '1'), '0'),
    ('FLASH_S5_YIELD_WINDOW', 'yield_window', ('0', '1'), '0'),
    ('FLASH_S5_INGRESS_INCREMENTAL', 'ingress_incremental', ('0', '1'), '0'),
    ('FLASH_S5_INGRESS_VERIFY_EVERY', 'verify_every', None, None),          # positive int; only with ingress
    ('FLASH_S5_INGRESS_VERIFY_ASYNC', 'verify_async', ('0', '1'), None),    # only with ingress
    ('FLASH_S5_APPEND_NO_BLOCK_STOP', 'append_no_block_stop', ('0', '1'), '0'),
    ('FLASH_S5_DECODE_GROUPS', 'decode_groups', ('chunk', 'off', 'off-small'), 'chunk'),
    ('FLASH_S5_DECODE_LIVENESS', 'decode_liveness', ('0', '1'), '0'),
    ('FLASH_S5_CONTROL_INWAIT', 'control_inwait', ('0', '1'), '0'),
)
EXTRA = ('append_no_block_stop', 'decode_groups', 'decode_liveness', 'control_inwait')   # added on top of np's queues
# CONTROL_INWAIT=1 is refused with each of these (flag, value that is NOT refused): untested together
CONTROL_INWAIT_UNTESTED = (('FLASH_S5_YIELD_REENTRY_SETTLE', '0'), ('FLASH_S5_SMALL_PIECE_BATCH', '0'),
                           ('FLASH_S5_INGRESS_INCREMENTAL', '0'), ('FLASH_S5_DECODE_GROUPS', 'chunk'),
                           ('FLASH_S5_DECODE_LIVENESS', '0'))
CONTROL_INWAIT_PARENT = ('46d28946', 'patches/s5_engine/native_auto_arm.py',
                         '3d5c813856f0c7acd68225a2df69c157d7d2aca2b653ea19ee0586ea4e1f9672')
CONTROL_INWAIT_PARENT_FILE = 'control_inwait_parent_native_auto_arm.py.src'   # bundle root = /src/harness; not importable
CONTROL_INWAIT_ENV = ("'-e','PYTHONDONTWRITEBYTECODE=1',",
                      "'-e','PYTHONDONTWRITEBYTECODE=1','-e','CONTROL_INWAIT_PARENT_SOURCE=/src/harness/%s',"
                      % CONTROL_INWAIT_PARENT_FILE)
BLOCKS = ('b1', 'b2', 'b3', 'b4')
DRY = 'dry'
BOUND_MODULES = ing.BOUND_MODULES + ('stage_overlap', 'native_overlap_control')
ENGINE_MOVED = {'e3/' + n + '.py' for n in BOUND_MODULES}
NEW_OVERLAY_FILES = np.NEW_OVERLAY_FILES
SUITES = np.SUITES + ('test_append_no_block_stop', 'test_append_no_block_stop_stack', 'test_n3_decode_groups',
                      'test_dmp_reader', 'test_q2024p_j2_cx', 'test_n3_compose_pairs', 'test_control_inwait')
TOOLS = np.TOOLS + ('tools/build_q2024p_j2_dmp.py', 'tools/q2024p_j2_dmp_reader.py', 'tools/build_q2024p_j2_cx.py', 'tools/q2024p_j2_cx_finish.py', 'tools/q2024p_j2_cx_render.py',
                    'tools/q2024p_j2_cx_reader.py')
CX_CONTAINER = "'q2024p-j2-cx-frozen-cpu-'"
PACKET_TAIL = ('\n# composition branch (q2024p-j2-cx): FLASH_S5_DECODE_GROUPS sets the n3 selection\'s decode_merge\n'
               '# (harness/n3_decode_groups; chunk, the default, is j2\'s table unchanged). The draft exports it.\n'
               'import n3_decode_groups as _n3dg\n'
               'configure = _n3dg.wrap_env(configure)\n')
DRAFT_EXPORT = ('export FLASH_S5_DECODE_GROUPS=%s  # composition branch: n3j2_packet.configure reads it in every python '
                'started below\n')
VARIANTS = np.VARIANTS          # filled by configure(); np's helpers read this dict


def parse_flags(pairs):
    """KEY=VALUE pairs over the defaults; refuses unknown keys, bad values and the refused combinations."""
    names = {f[0]: f for f in FLAGS}
    flags = {f[0]: f[3] for f in FLAGS}
    for pair in pairs:
        key, sep, value = pair.partition('=')
        if not sep or key not in names:
            raise SystemExit('unknown flag %r (known: %s)' % (pair, ' '.join(names)))
        allowed = names[key][2]
        if allowed is None:
            if not re.fullmatch(r'[1-9][0-9]{0,4}', value):
                raise SystemExit('%s must be a positive integer' % key)
        elif value not in allowed:
            raise SystemExit('%s must be one of %s' % (key, '|'.join(allowed)))
        flags[key] = value
    ingress = flags['FLASH_S5_INGRESS_INCREMENTAL'] == '1'
    for key in ('FLASH_S5_INGRESS_VERIFY_EVERY', 'FLASH_S5_INGRESS_VERIFY_ASYNC'):
        if flags[key] is not None and not ingress:
            raise SystemExit('%s needs FLASH_S5_INGRESS_INCREMENTAL=1' % key)
        if flags[key] is None and ingress:
            raise SystemExit('FLASH_S5_INGRESS_INCREMENTAL=1 needs %s set explicitly (np-ing: 1, np-all16: 16)' % key)
    if flags['FLASH_S5_YIELD_REENTRY_SETTLE'] == '1' and flags['FLASH_S5_YIELD_WINDOW'] == '1':
        raise SystemExit('refused: FLASH_S5_YIELD_REENTRY_SETTLE=1 with FLASH_S5_YIELD_WINDOW=1 (settle sits where the '
                         'yield window hands off; keep it out of a Y2 build)')
    if flags['FLASH_S5_DECODE_GROUPS'] != 'chunk' and flags['FLASH_S5_DECODE_LIVENESS'] != '1':
        raise SystemExit('refused: FLASH_S5_DECODE_GROUPS=%s needs FLASH_S5_DECODE_LIVENESS=1 (the arm guard reads it)'
                         % flags['FLASH_S5_DECODE_GROUPS'])
    if flags['FLASH_S5_CONTROL_INWAIT'] == '1':
        for key, safe in CONTROL_INWAIT_UNTESTED:
            if flags[key] != safe:
                raise SystemExit('refused: FLASH_S5_CONTROL_INWAIT=1 with %s=%s (untested together; only lever A, Y1 '
                                 'and Y2 are composed with it)' % (key, flags[key]))
    return flags


def variant_record(flags):
    return {f[1]: flags[f[0]] for f in FLAGS}


def name_of(variant, tag=None):
    tag = CONFIG['tag'] if tag is None else tag
    assert variant in BLOCKS + (DRY,), variant
    assert re.fullmatch(r'[a-z0-9]{0,8}', tag), tag
    return PARENT_NAME + '-cx' + tag + '-' + variant


CONFIG = dict(tag='')


def configure(flags, tag):
    """Point np's (and through it ing's, yr's, yrs's and hgb's) helpers at this cohort."""
    CONFIG['tag'] = tag
    VARIANTS.clear()
    VARIANTS.update({v: variant_record(flags) for v in BLOCKS + (DRY,)})
    ing.BOUND_MODULES, ing.ENGINE_MOVED = BOUND_MODULES, ENGINE_MOVED
    np.name_of = name_of
    np.SUITES, np.TOOLS, np.NP_CONTAINER = SUITES, TOOLS, CX_CONTAINER
    np.queues = queues
    hgb.BOX_IMPORT_NEW, hgb.BOX_ROW_NEW = dmp.BOX_IMPORT_NEW, dmp.BOX_ROW_NEW        # + the arm liveness guard
    np.bind()


_np_queues = np.queues


def queues(variant):
    """np's triple for the variant's five flags (+ verify), then lever A, decode groups and liveness after the ingress
    setting with their assertions; the draft also exports the decode-group mode and closes its arm gate on a crash."""
    v = VARIANTS[variant]
    setting = '%s=%s ' % (ing.ING_ENV, v['ingress_incremental'])
    check = "assert env.get('%s') == '%s', 'ingress setting differs'\n" % (ing.ING_ENV, v['ingress_incremental'])
    if v['ingress_incremental'] == '1':
        setting = '%s=%s ' % (np.VERIFY_ASYNC_ENV, v['verify_async'])
        check = "assert env.get('%s') == '%s', 'ingress verify mode differs'\n" % (np.VERIFY_ASYNC_ENV, v['verify_async'])
    extra = [(f[0], v[f[1]]) for f in FLAGS if f[1] in EXTRA]
    result = {}
    for rel, text in _np_queues(variant).items():
        if rel.endswith('_settings.sh'):
            text = hgb.once(text, setting, setting + ''.join('%s=%s ' % kv for kv in extra))
        else:
            text = hgb.once(text, check, check + ''.join(
                "assert env.get('%s') == '%s', 'composition setting differs: %s'\n" % (k, x, k) for k, x in extra))
        if rel.endswith('.draft.sh'):
            text = hgb.once(text, dmp.ARM_GATE, dmp.ARM_GATE_NEW.replace('decode-merge probe (dmp-b)', 'dmp-b review fix'))
            head, sep, _ = text.partition(dmp.DRAFT_ANCHOR)
            code = [line for line in head.splitlines() if not line.lstrip().startswith('#')]
            assert sep and not any(m in line for line in code for m in ('n3j2_packet', 'n3j2_arms', 'python')), \
                'table use before the export'
            text = hgb.once(text, dmp.DRAFT_ANCHOR, dmp.DRAFT_ANCHOR + DRAFT_EXPORT % v['decode_groups'])
        result[rel] = text
    return result


def copy_delta(b, commit, bundle):
    copied = np.copy_delta(b, commit, bundle)
    for name in ('harness/n3_decode_groups.py', 'harness/decode_liveness_guard.py', 'harness/test_n3_decode_groups.py',
                 'harness/test_append_no_block_stop.py', 'harness/test_append_no_block_stop_stack.py',
                 'harness/test_q2024p_j2_cx.py', 'harness/test_n3_compose_pairs.py',
                 'harness/test_control_inwait.py', 'harness/control_inwait_fixture_leaves.json',
                 'patches/s5_engine/s5_startup.py', 'patches/s5_engine/executor_hooks.py',
                 'patches/s5_engine/stage_overlap.py',
                 'patches/s5_engine/native_overlap_control.py', 'patches/s5_engine/scheduler_hooks.py'):
        assert name in copied, name
    return copied


def control_inwait_parent(b, bundle):
    """test_control_inwait's frozen off-path parent, by bytes from git (its sha checked here and again by the suite)."""
    commit, name, digest = CONTROL_INWAIT_PARENT
    target = bundle / CONTROL_INWAIT_PARENT_FILE
    assert not target.exists(), target
    target.write_bytes(b.git(ROOT, 'show', commit + ':' + name))
    assert b.sha(target) == digest, 'control-inwait parent sha differs'
    return dict(status='A', bundle=CONTROL_INWAIT_PARENT_FILE, sha256=digest, source=commit + ':' + name)


def frozen_cpu_tool(bundle):
    """yrs's frozen CPU tool, the container also naming the control-inwait parent source."""
    yrs.frozen_cpu_tool(bundle)
    path = bundle / 'tools/n3j2_frozen_cpu.py'
    path.write_text(hgb.once(path.read_text(), *CONTROL_INWAIT_ENV))


def prepare(commit, variant, out, flags, tag=''):
    """np.prepare's steps in its order, with this cohort's queues, bundle edits, delta check and record."""
    configure(flags, tag)
    b = hgb.helper()
    name = name_of(variant)
    commit = b.git(ROOT, 'rev-parse', commit + '^{commit}').decode().strip()
    assert len(commit) == 40
    assert not out.exists() and not out.is_symlink(), 'fresh output required'
    out.mkdir(parents=True)
    bundle = out / (name + '-bundle-pending')
    shutil.copytree(hgb.PARENT, bundle)
    for p in (bundle, *bundle.rglob('*')):
        p.chmod(0o755 if p.is_dir() else 0o644)
    copied = copy_delta(b, commit, bundle)
    copied['control_inwait_parent'] = control_inwait_parent(b, bundle)
    box = bundle / 'n3j2_box.py'
    box.write_text(hgb.once(hgb.once(box.read_text(), hgb.BOX_IMPORT, hgb.BOX_IMPORT_NEW), hgb.BOX_ROW, hgb.BOX_ROW_NEW))
    packet = bundle / 'n3j2_packet.py'
    packet.write_text(packet.read_text() + PACKET_TAIL)
    for rel, text in queues(variant).items():
        (bundle / rel).write_text(text)
        subprocess.run(['bash', '-n', str(bundle / rel)], check=True)
    proof = hgb.regenerate(b, bundle, out)
    assert set(NEW_OVERLAY_FILES) <= set(proof['overlay_changed']), 'new overlay files not in the regenerated overlay'
    frozen_cpu_tool(bundle)
    hgb.queue_adapter(bundle, variant)
    for p in bundle.rglob('*.py'):
        compile(p.read_bytes(), str(p), 'exec')
    manifest = b.manifest(bundle)
    final = out / (name + '-bundle-' + manifest[:12])
    bundle.rename(final)
    runner = out / (name + '-owner-run-' + manifest[:12] + '.py')
    runner.write_text(hgb.runner_text(b, final, manifest, proof, commit, variant))
    compile(runner.read_bytes(), str(runner), 'exec')
    for p in (final, *final.rglob('*')):
        p.chmod(0o555 if p.is_dir() else 0o444)
    runner.chmod(0o444)
    record = dict(proof, schema='q2024p-j2-cx-build/1', name=name, variant=variant, family=FAMILY, tag=tag,
                  flags=dict(flags), diagnostic='always-on', liveness_guard='bound (decode_liveness_guard)',
                  new_overlay_files=sorted(NEW_OVERLAY_FILES), source_commit=commit, base_commit=hgb.BASE_COMMIT,
                  bundle=str(final), manifest=manifest, runner=str(runner), runner_sha256=b.sha(runner),
                  runner_size=runner.stat().st_size, held=True, hardware_qualified=False, exploratory=True,
                  parent_manifest=hgb.PARENT_SHA, parent_runner_sha256=hgb.PARENT_RUNNER_SHA,
                  parent_overlay=hgb.PARENT_OVERLAY, source_changes=copied, cpu_suites=SUITES,
                  hgb_modules=BOUND_MODULES, box_action=False, gpu_action=False)
    b.dump(out / 'build.json', record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--variant', choices=BLOCKS + (DRY,), required=True)
    parser.add_argument('--tag', default='')
    parser.add_argument('--set', dest='flags', action='append', default=[], metavar='FLAG=VALUE')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source_commit, args.variant, args.out.resolve(), parse_flags(args.flags), args.tag)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_changes'}, sort_keys=True))
