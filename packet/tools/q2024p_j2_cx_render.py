#!/usr/bin/env python3
"""Render a registered q2024p-j2 composition cohort (four blocks cx<tag>-b1 .. b4) or the dry packet (cx-dry) from ONE
source commit and ONE flag set (lead 12 Nov 2026, branch claude/n3-compose). CPU and hub only.

  python3 -I -B tools/q2024p_j2_cx_render.py --source-commit <sha> --tag <t> --set FLAG=VALUE ... --out <fresh dir>
  python3 -I -B tools/q2024p_j2_cx_render.py --source-commit <sha> --dry --out <fresh dir>

Per packet, in parallel (one process each, niced): tools/build_q2024p_j2_cx.py, the bundle's frozen CPU self-run
(tools/n3j2_frozen_cpu.py; container q2024p-j2-cx-frozen-cpu-<manifest12>, kept), the bundle's precheck, then
tools/q2024p_j2_cx_finish.py (negatives, transport, compile, DELIVERY-SHA256SUMS). Logs: <out>/build-<v>.pipeline.log
and .rc. Then, for a cohort: every block's bundle and runner are proved equal to b1's after the name substitution
(the only other differences are the manifest-derived names, sums and records, listed), and <out>/cohort.json and
<out>/COHORT.md hold the table (label, QNAME, flags, seq path and sha256, DELIVERY-SHA256SUMS path and sha256,
nominal/hard minutes) and the reader call.
QNAMEs: none of the rendered names may be a '-'-prefix of another, or of / by a name in --known (the np family and
V1 cohort names by default); a clash refuses before any build. No box command, GPU, VM, deletion or push.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('build_q2024p_j2_cx', ROOT / 'tools/build_q2024p_j2_cx.py')
cx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cx)

KNOWN = tuple('q18a-parity-n3s24q2024p-j2-' + s for s in (
    'np-off', 'np-y12', 'np-ing', 'np-all', 'np-off-n2', 'np-y12-n2', 'np-set', 'np-hgb', 'np-off2', 'np-y12x',
    'np-all16', 'v1b1', 'v1b2', 'v1b3', 'v1b4', 'dmp-off', 'dmp-offsmall', 'dmp-off-b', 'dmp-offsmall-b'))
# the parent j2 QNAME (q18a-parity-n3s24q2024p-j2) is a '-'-prefix of every derived name, as for every j2 family;
# only a re-run of the parent itself would see these directories under its stale guard
PARENT_QNAME = 'q18a-parity-n3s24q2024p-j2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clashes(names):
    """Pairs (a, b) with a + '-' a prefix of b (the outer chain's stale guard globs <QNAME>-*)."""
    return sorted((a, b) for a in names for b in names if a != b and b.startswith(a + '-'))


def pipeline(out, variant, commit, tag, pairs, nice):
    d = out / ('build-' + variant)
    log = out / ('build-' + variant + '.pipeline.log')

    def step(name, cmd, stdout, stderr, cwd=ROOT):
        with log.open('a') as handle:
            handle.write('STEP %s %s\n' % (name, time.strftime('%H:%M:%S')))
        with open(stdout, 'x') as so, open(stderr, 'x') as se:
            rc = subprocess.run(['nice', '-n', str(nice)] + cmd, stdout=so, stderr=se, cwd=cwd).returncode
        with log.open('a') as handle:
            handle.write('%s rc=%d %s\n' % (name, rc, time.strftime('%H:%M:%S')))
        if rc:
            raise RuntimeError('%s %s rc=%d (see %s)' % (variant, name, rc, stderr))

    try:
        build_cmd = [sys.executable, '-I', '-B', str(ROOT / 'tools/build_q2024p_j2_cx.py'), '--source-commit', commit,
                     '--variant', variant, '--tag', tag, '--out', str(d)]
        for p in pairs:
            build_cmd += ['--set', p]
        step('build', build_cmd, out / ('build-' + variant + '.build.out'), out / ('build-' + variant + '.build.err'))
        b = json.loads((d / 'build.json').read_text())
        cmd = ['python3', '-B', b['bundle'] + '/tools/n3j2_frozen_cpu.py', '--bundle', b['bundle'], '--runner', b['runner'],
               '--source-commit', b['source_commit'], '--out', str(d / 'frozen-cpu')]
        for s in b['cpu_suites']:
            cmd += ['--suite', s]
        with (d / 'frozen-cpu-command.json').open('x') as handle:
            json.dump(cmd, handle, indent=2)
        step('frozen-cpu', cmd, d / 'frozen-cpu.out', d / 'frozen-cpu.err')
        step('precheck', ['python3', '-B', b['bundle'] + '/tools/n3j2_precheck.py', '--build', str(d / 'build.json'),
                          '--cpu', str(d / 'frozen-cpu/result.json'), '--out', str(d / 'precheck')],
             d / 'precheck.out', d / 'precheck.err')
        step('finish', [sys.executable, '-I', '-B', str(ROOT / 'tools/q2024p_j2_cx_finish.py'), '--build', str(d)],
             d / 'finish.out', d / 'finish.err')
        rc = 0
    except Exception as exc:                # recorded; the other packets continue
        with log.open('a') as handle:
            handle.write('FAILED %s\n' % exc)
        rc = 1
    (out / ('build-' + variant + '.rc')).write_text('%d\n' % rc)
    return variant, rc


def name_only(out, variants):
    """Every file of each block's bundle and runner equals b1's after replacing the block's name (and its manifest
    prefix) by b1's; the files left different are the manifest- and sum-bearing ones, listed."""
    first = json.loads((out / ('build-' + variants[0]) / 'build.json').read_text())
    report = {}
    for v in variants[1:]:
        other = json.loads((out / ('build-' + v) / 'build.json').read_text())
        subs = [(other['name'].encode(), first['name'].encode()), (other['manifest'].encode(), first['manifest'].encode()),
                (other['manifest'][:12].encode(), first['manifest'][:12].encode())]
        a_root, b_root = Path(first['bundle']), Path(other['bundle'])
        a_files = sorted(p.relative_to(a_root).as_posix() for p in a_root.rglob('*') if p.is_file())
        b_files = sorted(p.relative_to(b_root).as_posix() for p in b_root.rglob('*') if p.is_file())
        assert a_files == b_files, (v, sorted(set(a_files) ^ set(b_files)))
        same, renamed, differ = 0, 0, []
        for rel in a_files:
            x, y = (a_root / rel).read_bytes(), (b_root / rel).read_bytes()
            if x == y:
                same += 1
                continue
            for old, new in subs:
                y = y.replace(old, new)
            if x == y:
                renamed += 1
            else:
                differ.append(rel)
        rx, ry = Path(first['runner']).read_bytes(), Path(other['runner']).read_bytes()
        for old, new in subs:
            ry = ry.replace(old, new)
        report[v] = dict(identical=same, equal_after_name=renamed, differ=differ, runner_equal_after_name=rx == ry,
                         flags_equal=first['flags'] == other['flags'], source_equal=first['source_commit'] == other['source_commit'],
                         overlay_equal=first['overlay_manifest'] == other['overlay_manifest'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--tag', default='')
    parser.add_argument('--set', dest='flags', action='append', default=[], metavar='FLAG=VALUE')
    parser.add_argument('--dry', action='store_true', help='render only cx<tag>-dry (all flags at default required)')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--known', action='append', default=None, help='QNAMEs already in use (default: np family, V1, dmp)')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args(argv)
    flags = cx.parse_flags(args.flags)
    if args.dry and flags != cx.parse_flags([]):
        parser.error('--dry renders the all-default packet only')
    variants = (cx.DRY,) if args.dry else cx.BLOCKS
    names = {v: cx.name_of(v, args.tag) for v in variants}
    known = tuple(args.known) if args.known is not None else KNOWN
    found = clashes(list(names.values()) + [k for k in known if k not in names.values()])
    found = [c for c in found if c[0] in names.values() or c[1] in names.values()]
    taken = sorted(set(names.values()) & set(known))
    if found or taken:
        parser.error('QNAME clash: %s %s' % (found, taken))
    out = args.out.resolve()
    assert not out.exists() and not out.is_symlink(), 'fresh output required'
    commit = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', args.source_commit + '^{commit}'], check=True,
                            capture_output=True, text=True).stdout.strip()
    out.mkdir(parents=True)
    nice = max(0, 10 - os.nice(0))
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rcs = dict(pool.map(lambda v: pipeline(out, v, commit, args.tag, args.flags, nice), variants))
    minutes = round((time.time() - started) / 60, 1)
    rows = {}
    for v in variants:
        d = out / ('build-' + v)
        if rcs[v]:
            rows[v] = dict(qname=names[v], rc=rcs[v])
            continue
        delivery = json.loads((d / 'delivery.json').read_text())
        seq = delivery['owner_command'].split()[-1]
        rows[v] = dict(qname=names[v], rc=0, seq=seq, seq_sha256=sha(seq), delivery_sums=str(d / 'DELIVERY-SHA256SUMS'),
                       delivery_sums_sha256=sha(d / 'DELIVERY-SHA256SUMS'), owner_command=delivery['owner_command'],
                       manifest=delivery['manifest'], overlay_manifest=delivery['overlay_manifest'],
                       runner=delivery['runner'], runner_sha256=delivery['runner_sha256'], cpu_tests=delivery['cpu_tests'],
                       negative_cases=delivery['negative_cases'], nominal_minutes=delivery['nominal_minutes'],
                       hard_minutes=delivery['hard_minutes'])
    ok = all(rc == 0 for rc in rcs.values())
    proof = name_only(out, list(variants)) if ok and len(variants) > 1 else {}
    reader = ('python3 -I -B tools/q2024p_j2_cx_reader.py --cohort %s %s [--json-out <fresh.json>]' % (
        out / 'cohort.json', ' '.join('%s=<FETCH>/%s' % (v, names[v]) for v in variants)))
    doc = dict(schema='q2024p-j2-cx-cohort/1', exploratory=True, kind='dry' if args.dry else 'cohort', tag=args.tag,
               source_commit=commit, flags=flags, set=list(args.flags), blocks=rows, ok=ok, wall_minutes=minutes,
               name_only=proof, known_names=list(known), parent_qname_prefix=PARENT_QNAME,
               arms='sp.g2-live-0 n3.g2-live-0 n3.g2-live-1 sp.g2-live-1 at 480 s (j2 seeds)', reader=reader,
               box_action=False, gpu_action=False, pushed=False)
    with (out / 'cohort.json').open('x') as handle:
        json.dump(doc, handle, indent=2, sort_keys=True)
        handle.write('\n')
    lines = ['# %s %s from %s' % ('dry packet' if args.dry else 'cohort', 'cx' + args.tag, commit[:12]), '',
             'flags: ' + ' '.join('%s=%s' % kv for kv in sorted(flags.items()) if kv[1] is not None), '',
             '| label | QNAME | seq | seq sha256 | DELIVERY-SHA256SUMS | sums sha256 | min nominal/hard |', '|---|---|---|---|---|---|---|']
    for v, r in rows.items():
        lines.append('| %s | `%s` | `%s` | `%s` | `%s` | `%s` | %s/%s |' % (
            v, r['qname'], r.get('seq'), r.get('seq_sha256'), r.get('delivery_sums'), r.get('delivery_sums_sha256'),
            r.get('nominal_minutes'), r.get('hard_minutes')) if not r['rc'] else '| %s | `%s` | FAILED rc=%s |||||' % (
            v, r['qname'], r['rc']))
    lines += ['', 'reader: `%s`' % reader, '', 'render wall: %s min' % minutes]
    (out / 'COHORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(ok=ok, out=str(out), minutes=minutes, rcs=rcs, name_only=proof), sort_keys=True))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
