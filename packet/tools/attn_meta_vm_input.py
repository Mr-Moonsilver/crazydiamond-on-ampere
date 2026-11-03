#!/usr/bin/env python3
"""Build, stage, run and fetch the attn-meta Phase 2 VM <vm-id> run (one card, CUDA, no model, no engine).

    python3 tools/attn_meta_vm_input.py build --name attn-meta-vm-a --work DIR --overlay-qsa FILE
    python3 tools/attn_meta_vm_input.py run   --input DIR/attn-meta-vm-a-input --record DIR [--lock-wait S]

``build`` writes a fresh, read-only input directory: run_gpu.py (tools/attn_meta_vm_run_gpu.py with its container
name), w/tools/{attn_meta_witness,attn_meta_cases,attn_meta_vm_probe}.py, w/patches/s5_engine/{attn_meta_batch,
lane_chunk,e3_types}.py, w/model-config.json, w/overlay-qsa.py, provenance.json (commit, sources, scope), SHA256SUMS.
The hub self-run of the same probe (``tools/attn_meta_selfrun2.py --mode probe``) runs the same w/ files on CPU.

``run`` stages over ssh as the other VM inputs do (tar, root-owned, read-only, manifest re-verified on the VM:
tools/step_a_vm_input.py stage()), runs run_gpu.py under the shared GPU flock, and fetches the whole run dir.
It never retries, never removes anything on the VM, and refuses an existing record directory.
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
    'w/patches/s5_engine/attn_meta_batch.py': 'patches/s5_engine/attn_meta_batch.py',
    'w/patches/s5_engine/lane_chunk.py': 'patches/s5_engine/lane_chunk.py',
    'w/patches/s5_engine/e3_types.py': 'patches/s5_engine/e3_types.py',
}
SCOPE = dict(
    exercised=['real GDN / PLE / QSA ring / QSA compressed / FA metadata builders on CUDA, box KV-group layout',
               'real Qwen4ExpModelState.prepare_attn (stock) vs attn_meta_batch.install on the same class',
               'a captured CUDA graph per fixture reading every GDN persistent buffer at its address, replayed '
               'after every step of a 48-step decode plan (block crossings, admissions, finishes, padding)',
               'CUDA profiler device records and runtime launch/sync calls per prepare_attn call',
               'host µs per call, per builder and per tiny device op on the GPU path',
               'native lane private clones (lane_chunk.private_attention_groups) on CUDA: storage and bytes'],
    not_exercised=['the model forward and GDN numerics (no model on one 3090): greedy tokens NOT COVERED',
                   'TP/PP, the engine, the scheduler, the native lane runtime (only its clone function)',
                   'box numbers: one RTX 3090 and the VM CPU, not the box'])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def manifest(d):
    rows = sorted('%s  %s' % (sha(p), p.relative_to(d).as_posix()) for p in d.rglob('*')
                  if p.is_file() and p.name != 'SHA256SUMS')
    (d / 'SHA256SUMS').write_text('\n'.join(rows) + '\n')


def build(name, work, overlay_qsa):
    if not name.startswith('attn-meta-vm-'):
        raise SystemExit('name must start with attn-meta-vm-')
    out = work / (name + '-input')
    if out.exists():
        raise SystemExit('refused: exists: %s' % out)
    status = subprocess.check_output(['git', 'status', '--porcelain', '--', *FILES.values(),
                                      'tools/attn_meta_vm_run_gpu.py'], cwd=ROOT, text=True)
    if status.strip():
        raise SystemExit('refused: uncommitted sources:\n' + status)
    out.mkdir(parents=True)
    for dst, src in FILES.items():
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / src, out / dst)
    shutil.copyfile((ROOT / 'reference/model-config.json').resolve(), out / 'w/model-config.json')
    shutil.copyfile(overlay_qsa, out / 'w/overlay-qsa.py')
    text = (ROOT / 'tools/attn_meta_vm_run_gpu.py').read_text()
    assert text.count("NAME = 'flashnext-@NAME@'") == 1
    (out / 'run_gpu.py').write_text(text.replace('@NAME@', name))
    prov = dict(schema='attn-meta-vm-input/1', name=name,
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                sources={src: sha(ROOT / src) for src in [*FILES.values(), 'tools/attn_meta_vm_run_gpu.py']},
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
    a = p.parse_args()
    if a.cmd == 'build':
        build(a.name, a.work.resolve(), a.overlay_qsa.resolve())
        return 0
    return run(a.input.resolve(), a.record.resolve(), a.lock_wait)


if __name__ == '__main__':
    sys.exit(main())
