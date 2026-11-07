#!/usr/bin/env python3
"""lane-o3: e3_lane_o3_probe's cases in one CPU container over the built bundle (the a1-trunk-1g runner's shape); one record per
case and a result. No GPU, no network, nothing on the box.

result.json verdict: PASS when every case reads PASS (a mutant case passes when its invariant is reported violated); rc follows it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from native37_robust_selfrun import IMAGE, overlay_mounts  # noqa: E402

CASES = ('on', 'off', 'uninstalled', 'mutant-no-classify', 'mutant-no-seam',
         'lag-on', 'lag-off', 'lag-k2', 'mutant-lag-no-settle', 'mutant-lag-park')   # lane-o3-fix: the worker-lag cases


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--probe', type=Path, help='iteration only: this probe file over the bundle\'s copy (recorded)')
    a = p.parse_args()
    bundle, out = a.bundle.resolve(), a.out.resolve()
    out.mkdir()
    out.chmod(0o777)
    lines = ['set -u', 'cd /source/harness']
    for case in CASES:
        lines.append('python3 -B e3_lane_o3_probe.py --case %s --out /out/%s.json > /out/%s.log 2>&1; echo $? > /out/%s.rc'
                     % (case, case, case, case))
    (out / 'inner.sh').write_text('\n'.join(lines) + '\n')
    c = ['sudo', '-n', 'docker', 'run', '--pull=never', '--name', a.name, '--runtime', 'runc', '--network', 'none',
         '--cpus', '4', '--memory', '6g', '--memory-swap', '6g', '--read-only',
         '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g', '--ulimit', 'core=0',
         '-e', 'VLLM_CACHE_ROOT=/tmp/vllm-cache', '-e', 'NVIDIA_VISIBLE_DEVICES=none', '-e', 'CUDA_VISIBLE_DEVICES=',
         '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'PYTHONPATH=/source/harness:/source/harness/e3_builder']
    mounts = dict(overlay_mounts(bundle / 'native30_overlay'))
    mounts.update({bundle: '/source/harness', bundle / 'e3_builder': '/source/patches/s5_engine', bundle / 'native35_tools': '/source/tools',
                   bundle / 'sparse_stage': '/source/patches/sparse_stage',
                   (HERE.parent / 'reference/vllm-src').resolve(): '/source/reference/vllm-src'})
    if a.probe is not None:
        mounts[a.probe.resolve()] = '/source/harness/e3_lane_o3_probe.py'
    for source, target in mounts.items():
        c += ['-v', '%s:%s:ro' % (source, target)]
    c += ['-v', '%s:/out' % out, '--entrypoint', 'bash', IMAGE, '/out/inner.sh']
    started = time.monotonic()
    with (out / 'container.log').open('x') as f:
        rc = subprocess.call(c, stdout=f, stderr=subprocess.STDOUT)
    inspected = json.loads(subprocess.check_output(['sudo', '-n', 'docker', 'inspect', a.name], text=True))[0]['HostConfig']
    no_gpu = not inspected.get('DeviceRequests') and not inspected.get('Devices') and inspected['NetworkMode'] == 'none'
    rows = {}
    for case in CASES:
        path = out / (case + '.json')
        record = json.loads(path.read_text()) if path.exists() else {}
        rows[case] = dict(verdict=record.get('verdict'), violations=record.get('violations'),
                          expect_violation=record.get('expect_violation'), checks=len(record.get('checks') or ()),
                          error=record.get('error') or record.get('error_type'), wall_s=record.get('wall_s'),
                          sha256=sha(path) if path.exists() else None)
    ok = rc == 0 and no_gpu and all(row['verdict'] == 'PASS' for row in rows.values())
    result = dict(schema='lane-o3-cpu-result/1', verdict='PASS' if ok else 'FAIL', container_rc=rc, no_gpu_no_network=no_gpu,
                  cases=rows, manifest=sha(bundle / 'SHA256SUMS'), image=IMAGE, wall_s=round(time.monotonic() - started, 1),
                  gpu_execution=False, box_evidence=False,
                  probe_override=(sha(a.probe) if a.probe is not None else None))
    (out / 'result.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(verdict=result['verdict'], cases={k: v['verdict'] for k, v in rows.items()})))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
