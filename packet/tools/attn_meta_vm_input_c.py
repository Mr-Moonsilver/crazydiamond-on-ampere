#!/usr/bin/env python3
"""Build, stage, run and fetch the attn-meta VM <vm-id> run c (one card, CUDA, no model, no engine).

Supersedes tools/attn_meta_vm_input_b.py by name (it built and ran attn-meta-vm-b and is unchanged). Changes: the probe is
probe c (tools/attn_meta_vm_probe_c.py: probe b plus the per-arm Selector phase; it imports probes a and b unchanged, so all
three are shipped), the module patches/s5_engine/attn_meta_select.py is shipped and bound in ``engine_bytes``, the wrapper is
tools/attn_meta_vm_run_gpu_c.py, names must start with attn-meta-vm-c. ``check-bind`` compares a bundle's engine directory
(e.g. BUNDLE/native30_overlay/e3) with every engine_bytes row.

    python3 tools/attn_meta_vm_input_c.py build --name attn-meta-vm-c --work DIR --overlay-qsa FILE
    python3 tools/attn_meta_vm_input_c.py run   --input DIR/attn-meta-vm-c-input --record DIR [--lock-wait S]
    python3 tools/attn_meta_vm_input_c.py check-bind --input DIR/attn-meta-vm-c-input --engine-dir DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VM = '<service-user>@<encoder-host>'
SSH = ['ssh', '-F', '/dev/null', '-i', str(Path.home() / '.ssh/<ssh-key>'),
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=30', VM]
GPU_LOCK = '/home/<service-user>/.flashnext-vm-gpu.lock'
FILES = {
    'w/tools/attn_meta_witness.py': 'tools/attn_meta_witness.py',
    'w/tools/attn_meta_cases.py': 'tools/attn_meta_cases.py',
    'w/tools/attn_meta_vm_probe.py': 'tools/attn_meta_vm_probe.py',
    'w/tools/attn_meta_vm_probe_b.py': 'tools/attn_meta_vm_probe_b.py',
    'w/tools/attn_meta_vm_probe_c.py': 'tools/attn_meta_vm_probe_c.py',
    'w/patches/s5_engine/attn_meta_batch.py': 'patches/s5_engine/attn_meta_batch.py',
    'w/patches/s5_engine/attn_meta_select.py': 'patches/s5_engine/attn_meta_select.py',
    'w/patches/s5_engine/lane_chunk.py': 'patches/s5_engine/lane_chunk.py',
    'w/patches/s5_engine/e3_types.py': 'patches/s5_engine/e3_types.py',
}
SCOPE = dict(
    exercised=['real GDN / PLE / QSA ring / QSA compressed / FA metadata builders on CUDA, box KV-group layout',
               'real Qwen4ExpModelState.prepare_attn (stock) vs attn_meta_batch.install on the same class',
               'a captured CUDA graph per fixture reading every GDN persistent buffer at its address, replayed '
               'after every step of a 48-step decode plan (block crossings, admissions, finishes, padding)',
               'CUDA profiler device records per prepare_attn call; runtime launch/sync calls inside a '
               'record_function step window (profiler stop flush outside); a real sync injected in the window '
               '(sync-mutant control) must be counted',
               'the per-arm Selector (attn_meta_select) over the batch: switched off -> on -> off between steps of the '
               '48-step plan at a synced point, equality vs stock and the captured-graph read after every step; '
               'Selector-off device records and in-window runtime calls equal stock',
               'host µs per call, per builder and per tiny device op on the GPU path',
               'native lane private clones (lane_chunk.private_attention_groups) on CUDA: storage and bytes'],
    not_exercised=['the model forward and GDN numerics (no model on one 3090): greedy tokens NOT COVERED',
                   'TP/PP, the engine, the scheduler, the native lane runtime (only its clone function)',
                   'the typed control path (client, driver admission, worker RPC): CPU-tested, harness/test_attn_meta_control.py',
                   'box numbers: one RTX 3090 and the VM CPU, not the box'])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def manifest(d):
    rows = sorted('%s  %s' % (sha(p), p.relative_to(d).as_posix()) for p in d.rglob('*')
                  if p.is_file() and p.name != 'SHA256SUMS')
    (d / 'SHA256SUMS').write_text('\n'.join(rows) + '\n')


def build(name, work, overlay_qsa):
    if not name.startswith('attn-meta-vm-c'):
        raise SystemExit('name must start with attn-meta-vm-c')
    out = work / (name + '-input')
    if out.exists():
        raise SystemExit('refused: exists: %s' % out)
    status = subprocess.check_output(['git', 'status', '--porcelain', '--', *FILES.values(),
                                      'tools/attn_meta_vm_run_gpu_c.py'], cwd=ROOT, text=True)
    if status.strip():
        raise SystemExit('refused: uncommitted sources:\n' + status)
    out.mkdir(parents=True)
    for dst, src in FILES.items():
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / src, out / dst)
    shutil.copyfile((ROOT / 'reference/model-config.json').resolve(), out / 'w/model-config.json')
    shutil.copyfile(overlay_qsa, out / 'w/overlay-qsa.py')
    text = (ROOT / 'tools/attn_meta_vm_run_gpu_c.py').read_text()
    assert text.count("NAME = 'flashnext-@NAME@'") == 1
    (out / 'run_gpu.py').write_text(text.replace('@NAME@', name))
    engine = {dst.split('/', 1)[1]: sha(ROOT / src) for dst, src in FILES.items() if dst.startswith('w/patches/')}
    prov = dict(schema='attn-meta-vm-input-c/1', name=name, engine_bytes=engine,
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                sources={src: sha(ROOT / src) for src in [*FILES.values(), 'tools/attn_meta_vm_run_gpu_c.py']},
                overlay_qsa=str(overlay_qsa), overlay_qsa_sha256=sha(overlay_qsa),
                model_config_sha256=sha(out / 'w/model-config.json'), scope=SCOPE,
                builder_sha256=sha(Path(__file__)), built_unix=int(time.time()), gpu_run=False, box_action=False)
    (out / 'provenance.json').write_text(json.dumps(prov, indent=1, sort_keys=True) + '\n')
    manifest(out)
    for p in [out, *out.rglob('*')]:
        p.chmod(0o555 if p.is_dir() else 0o444)
    res = dict(verdict='PASS', path=str(out), manifest=sha(out / 'SHA256SUMS'), commit=prov['commit'])
    print(json.dumps(res, sort_keys=True))
    return res


def vm(command, **kw):
    return subprocess.run([*SSH, command], **kw)


def stage(inp):
    name = inp.name.removesuffix('-input')
    msha = sha(inp / 'SHA256SUMS')
    owned = name + '-' + msha[:12] + '-owner'
    remote = '/opt/flashnext/gpu/' + owned
    script = ('set -e; D=%s; test ! -e $D; tar xf - -C /opt/flashnext/gpu; chown -R root:root $D; chmod -R a-w $D; '
              'chmod 500 $D; cd $D; test "$(sha256sum SHA256SUMS | cut -d" " -f1)" = %s; '
              'sha256sum -c SHA256SUMS --quiet; echo VM_SUMS_OK' % (remote, msha))
    tar = subprocess.Popen(['tar', 'cf', '-', '-C', str(inp.parent), inp.name,
                            '--transform', 's|^' + inp.name + '|' + owned + '|'], stdout=subprocess.PIPE)
    done = vm('sudo -n bash -c ' + shlex.quote(script), stdin=tar.stdout)
    tar.stdout.close()
    if tar.wait() or done.returncode:
        raise RuntimeError('stage failed rc=%s' % done.returncode)
    return remote


def fetch(run_dir, record):
    record.mkdir(parents=True, exist_ok=False)
    parent, name = run_dir.rsplit('/', 1)
    pull = subprocess.Popen([*SSH, 'sudo -n tar cf - -C %s %s' % (parent, name)], stdout=subprocess.PIPE)
    untar = subprocess.run(['tar', 'xf', '-', '--strip-components=1', '--no-same-owner', '-C', str(record)],
                           stdin=pull.stdout)
    pull.stdout.close()
    if pull.wait() or untar.returncode:
        raise RuntimeError('fetch failed')


def run(inp, record, lock_wait):
    if record.exists():
        raise SystemExit('refused: record exists: %s' % record)
    name = inp.name.removesuffix('-input')
    remote = stage(inp)
    run_dir = '/opt/flashnext/runs/' + name
    t0 = time.time()
    cmd = 'flock -w %d %s sudo -n python3 -B %s/run_gpu.py --input %s --out %s' % (
        lock_wait, GPU_LOCK, remote, remote, run_dir)
    done = vm(cmd)
    rec = dict(vm_rc=done.returncode, run_dir=run_dir, remote_input=remote, wall_s=round(time.time() - t0, 1))
    print(json.dumps(rec), flush=True)
    fetch(run_dir, record)
    rec['record'] = str(record)
    (record.parent / (record.name + '-run.json')).write_text(json.dumps(rec, indent=1, sort_keys=True) + '\n')
    print(json.dumps(rec), flush=True)
    return done.returncode


def check_bind(inp, engine_dir):
    """Compare a bundle's engine directory (e.g. .../s5_engine) with this input's engine bytes; PASS / FAIL."""
    prov = json.loads((inp / 'provenance.json').read_text())
    rows = []
    for rel, digest in sorted(prov['engine_bytes'].items()):
        p = engine_dir / Path(rel).name
        got = sha(p) if p.is_file() else None
        rows.append(dict(file=rel, vm_b=digest, bundle=got, same=got == digest))
    res = dict(verdict='PASS' if all(r['same'] for r in rows) else 'FAIL', input=str(inp),
               input_manifest=sha(inp / 'SHA256SUMS'), engine_dir=str(engine_dir), rows=rows)
    print(json.dumps(res, indent=1, sort_keys=True))
    return 0 if res['verdict'] == 'PASS' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--name', required=True)
    b.add_argument('--work', type=Path, required=True)
    b.add_argument('--overlay-qsa', type=Path, required=True)
    r = sub.add_parser('run')
    r.add_argument('--input', type=Path, required=True)
    r.add_argument('--record', type=Path, required=True)
    r.add_argument('--lock-wait', type=int, default=300)
    c = sub.add_parser('check-bind')
    c.add_argument('--input', type=Path, required=True)
    c.add_argument('--engine-dir', type=Path, required=True)
    a = p.parse_args()
    if a.cmd == 'check-bind':
        return check_bind(a.input.resolve(), a.engine_dir.resolve())
    if a.cmd == 'build':
        build(a.name, a.work.resolve(), a.overlay_qsa.resolve())
        return 0
    return run(a.input.resolve(), a.record.resolve(), a.lock_wait)


if __name__ == '__main__':
    sys.exit(main())
