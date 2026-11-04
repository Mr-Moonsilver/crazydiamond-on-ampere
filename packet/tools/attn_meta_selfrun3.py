#!/usr/bin/env python3
"""Hub self-run of the attn-meta tests and probe b's CPU pass, inside the pinned image, CPU only.

Supersedes tools/attn_meta_selfrun2.py by name (that one ran t1..t3 and p1 and is unchanged). Same container contract:
no GPU (no device requests, NVIDIA_VISIBLE_DEVICES=none), no network, read-only root, --cpus 4, nice 19 inside,
inputs mounted read-only, one fresh output directory. Differences, all from the lead's resume-2 rules (3 Nov):
  - it WAITS (polls every 20 s, up to --wait-max s, then refuses) until no running container's name contains any of
    alpha-next-c1, append-yield, returning-append, and no pinned-image container runs (mine included);
  - each container step is capped at --cap s (default 600, the lead's 10 min): stopped at the cap, rc kept.

    --mode tests-b   harness/test_attn_meta_probe_b.py, then harness/test_attn_meta_overlay.py (unittest)
    --mode probe-b   tools/attn_meta_vm_probe_b.py --device cpu (probe b's test set on CPU; opcount = probe a's census)
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
ROOT = Path(__file__).resolve().parents[1]
CHAIN = re.compile(r'alpha-next-c1|append-yield|returning-append')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def busy():
    rows = subprocess.check_output(['sudo', '-n', 'docker', 'ps', '--format', '{{.Names}} {{.Image}}'], text=True)
    # Lead 3 Nov resume 2: wait while a container whose NAME contains alpha-next-c1 / append-yield /
    # returning-append runs; also never two pinned-image containers at once.
    out = []
    for r in rows.splitlines():
        name = r.split(' ', 1)[0]
        if CHAIN.search(name) or 'ad067db5' in r or 'qwen-flash-next' in r:
            out.append(r)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--overlay-qsa', type=Path, required=True)
    p.add_argument('--mode', choices=('tests-b', 'probe-b'), required=True)
    p.add_argument('--wait-max', type=int, default=3600)
    p.add_argument('--cap', type=int, default=600)
    a = p.parse_args()
    model_cfg = (ROOT / 'reference/model-config.json').resolve()
    overlay = a.overlay_qsa.resolve()
    out = a.out.resolve()
    waited, t_wait = [], time.monotonic()
    while True:
        others = busy()
        if not others:
            break
        if not waited or waited[-1][1] != others:
            waited.append([time.strftime('%H:%M:%S'), others])
            print('waiting:', others, flush=True)
        if time.monotonic() - t_wait > a.wait_max:
            raise SystemExit(f'refused: busy after {a.wait_max} s: {others}')
        time.sleep(20)
    wait_s = round(time.monotonic() - t_wait, 1)
    out.mkdir(parents=True)
    out.chmod(0o777)
    load0 = Path('/proc/loadavg').read_text().split()[0]
    cmd = ['sudo', '-n', 'docker', 'run', '--rm=false', '--pull=never', '--name', a.name, '--runtime', 'runc',
           '--network', 'none', '--cpus', '4', '--memory', '6g', '--memory-swap', '6g', '--read-only',
           '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g', '--ulimit', 'core=0']
    for s in ('NVIDIA_VISIBLE_DEVICES=none', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
              'HF_HUB_OFFLINE=1', 'TRANSFORMERS_OFFLINE=1', 'VLLM_QSA_KV_OFFLOAD=1', 'HOME=/tmp',
              'ATTN_META_MODEL_CONFIG=/w/model-config.json', 'ATTN_META_OVERLAY_QSA=/w/overlay-qsa.py'):
        cmd += ['-e', s]
    cmd += ['-v', f'{ROOT / "tools"}:/w/tools:ro', '-v', f'{ROOT / "patches/s5_engine"}:/w/patches/s5_engine:ro',
            '-v', f'{ROOT / "harness/test_attn_meta_overlay.py"}:/w/harness/test_attn_meta_overlay.py:ro',
            '-v', f'{ROOT / "harness/test_attn_meta_probe_b.py"}:/w/harness/test_attn_meta_probe_b.py:ro',
            '-v', f'{ROOT / "results/vm<vm-id>/attn-meta-vm-a"}:/w/results/vm<vm-id>/attn-meta-vm-a:ro',
            '-v', f'{ROOT / "results/hub/attn-meta"}:/w/results/hub/attn-meta:ro',
            '-v', f'{model_cfg}:/w/model-config.json:ro', '-v', f'{overlay}:/w/overlay-qsa.py:ro', '-v', f'{out}:/out']
    if a.mode == 'tests-b':
        script = ('cd /w/harness && { nice -n 19 python3 -B -m unittest -v test_attn_meta_probe_b '
                  'test_attn_meta_overlay; } > /out/tests.log 2>&1; '
                  't=$?; echo "rc=$t" >> /out/tests.log; '
                  'printf \'{"verdict": "%s", "tests_rc": %s}\\n\' "$([ $t = 0 ] && echo PASS || echo FAIL)" $t '
                  '> /out/witness.json; [ $t = 0 ]')
    else:
        script = ('cd /w && nice -n 19 python3 -B tools/attn_meta_vm_probe_b.py --device cpu --out /out '
                  '--model-config /w/model-config.json --overlay-qsa /w/overlay-qsa.py > /out/probe.log 2>&1; '
                  't=$?; echo "rc=$t" >> /out/probe.log; exit $t')
    cmd += ['--entrypoint', 'sh', IMAGE, '-c', script]
    started = time.monotonic()
    capped = False
    with (out / 'container.log').open('x') as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=a.cap)
        except subprocess.TimeoutExpired:
            capped = True
            subprocess.call(['sudo', '-n', 'docker', 'stop', '--time', '10', a.name])
            rc = proc.wait(timeout=60)
    inspect = json.loads(subprocess.check_output(['sudo', '-n', 'docker', 'inspect', a.name], text=True))[0]
    devices = inspect['HostConfig'].get('DeviceRequests') or []
    subprocess.call(['sudo', '-n', 'docker', 'rm', a.name], stdout=subprocess.DEVNULL)
    w = out / ('witness.json' if a.mode == 'tests-b' else 'probe.json')
    verdict = json.loads(w.read_text()).get('verdict') if w.exists() else None
    files = sorted(str(f.relative_to(ROOT)) for d in ('tools', 'patches/s5_engine') for f in (ROOT / d).glob('attn_meta*.py'))
    files += ['harness/test_attn_meta_overlay.py', 'harness/test_attn_meta_probe_b.py',
              'patches/s5_engine/lane_chunk.py', 'patches/s5_engine/e3_types.py']
    result = dict(schema='attn-meta-selfrun3/1', mode=a.mode, rc=rc, inner_verdict=verdict, capped=capped,
                  cap_s=a.cap, wait_s=wait_s, waited=waited,
                  verdict='INCONCLUSIVE' if capped else 'PASS' if rc == 0 and verdict == 'PASS' else 'FAIL' if rc == 1 or verdict == 'FAIL' else 'INCONCLUSIVE',
                  image=IMAGE, gpu_device_requests=len(devices), network=inspect['HostConfig'].get('NetworkMode'),
                  sources={f: sha(ROOT / f) for f in files}, model_config_sha256=sha(model_cfg),
                  overlay_qsa=str(overlay), overlay_qsa_sha256=sha(overlay), container=a.name,
                  load_start=load0, load_end=Path('/proc/loadavg').read_text().split()[0],
                  wall_s=round(time.monotonic() - started, 1), command=cmd, box_action=False, gpu_execution=False)
    assert not devices and not inspect['HostConfig'].get('Devices')
    (out / 'result.json').write_text(json.dumps(result, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: result[k] for k in ('verdict', 'rc', 'inner_verdict', 'wall_s', 'load_start', 'load_end')}))
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
