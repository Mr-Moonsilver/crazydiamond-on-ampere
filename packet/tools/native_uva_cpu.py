#!/usr/bin/env python3
"""Retained, CPU-only pinned-image suites. Every output name must be fresh."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
SUITES = ('test_native_uva_retirement', 'test_native_uva_legacy', 'test_native_uva_pinned', 'test_native_uva_package', 'test_e3_lane_chunk', 'test_e3_lane_dispatch', 'test_prepare_ahead', 'test_parity_ahead', 'test_inwait_slices', 'test_native35_dispatch_lifetime', 'test_e3_native21_rpc_stream')

INNER = '''import json,sys,unittest
from pathlib import Path
import torch
assert not torch.cuda.is_initialized()
r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName(sys.argv[1]))
assert not torch.cuda.is_initialized()
Path(sys.argv[2]).write_text(json.dumps(dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skips=len(r.skipped),ok=r.wasSuccessful(),cuda_initialized=False)))
raise SystemExit(0 if r.wasSuccessful() and not r.skipped else 1)
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for directory in ('patches/s5_engine', 'harness', 'tools')
            for p in sorted((ROOT / directory).glob('*.py'))} | {'harness/native_uva_fixture_leaves.json': sha(ROOT / 'harness/native_uva_fixture_leaves.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    out.chmod(0o777)
    model_config = (ROOT / 'reference/model-config.json').resolve(strict=True)
    config_sha = sha(model_config)
    frozen = sources()
    (out / 'source-shas.json').write_text(json.dumps(frozen, indent=2) + '\n')
    (out / 'suite.py').write_text(INNER)
    (out / 'run.py').write_text('''import json,subprocess,sys
from pathlib import Path
rows=[]
for suite in sys.argv[1:]:
 with (Path('/out')/(suite+'.log')).open('x') as log:
  rc=subprocess.call([sys.executable,'-B','/out/suite.py',suite,'/out/'+suite+'.json'],stdout=log,stderr=subprocess.STDOUT)
 rows.append(dict(suite=suite,rc=rc))
Path('/out/returns.json').write_text(json.dumps(rows))
raise SystemExit(any(r['rc'] for r in rows))
''')
    command = ['sudo', '-n', 'docker', 'run', '--pull=never', '--name', args.name,
               '--runtime', 'runc', '--network', 'none', '--cpus', '4', '--memory', '6g',
               '--memory-swap', '6g', '--read-only', '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=1g',
               '--ulimit', 'core=0', '-e', 'CUDA_VISIBLE_DEVICES=', '-e', 'NVIDIA_VISIBLE_DEVICES=none',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'VLLM_CACHE_ROOT=/tmp/vllm-cache',
               '-e', 'PYTHONPATH=/source/patches/s5_engine:/source/harness',
               '-v', str(ROOT) + ':/source:ro',
               '-v', str((ROOT / 'reference/vllm-src').resolve()) + ':/source/reference/vllm-src:ro',
               '-v', str(model_config) + ':/source/reference/model-config.json:ro',
               '-v', str(out) + ':/out', '-w', '/source/harness', '--entrypoint', 'python3',
               IMAGE, '-B', '/out/run.py', *SUITES]
    started = time.monotonic()
    with (out / 'container.log').open('x') as log:
        rc = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT)
    inspect = json.loads(subprocess.check_output(['sudo', '-n', 'docker', 'inspect', args.name], text=True))[0]
    config = inspect['HostConfig']
    no_device = not config.get('DeviceRequests') and not config.get('Devices') and config['Runtime'] == 'runc'
    rows = []
    for suite in SUITES:
        path = out / (suite + '.json')
        facts = json.loads(path.read_text()) if path.exists() else {'ok': False}
        rows.append(dict(suite=suite, **facts))
    same = frozen == sources()
    ok = (rc == 0 and no_device and config['NetworkMode'] == 'none' and same and sha(model_config) == config_sha
          and all(r.get('ok') and r.get('skips') == 0 and not r.get('cuda_initialized') for r in rows))
    record = dict(schema='legacy-native-uva-cpu/1', verdict='PASS' if ok else 'FAIL',
                  container_rc=rc, image=inspect['Image'], container=args.name, command=command,
                  network=config['NetworkMode'], gpu_execution=False, no_device=no_device,
                  model_config_sha256=config_sha,
                  source_bytes_unchanged=same and sha(model_config) == config_sha, source_manifest_sha256=sha(out / 'source-shas.json'),
                  tests=sum(r.get('tests', 0) for r in rows), skips=sum(r.get('skips', 0) for r in rows),
                  suites=rows, wall_s=round(time.monotonic() - started, 3), box_action=False,
                  artifacts_preserved=True)
    (out / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('verdict', 'tests', 'skips', 'wall_s')}))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
