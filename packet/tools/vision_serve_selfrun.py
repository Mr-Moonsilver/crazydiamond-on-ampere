#!/usr/bin/env python3
"""CPU-only whole-protocol replay of the frozen vision serve, reusing serve-c shims.

No builder, VM, box, GPU or production control. Exact served clients run against
schema-valid loopback HTTP and fake systemctl/docker/NVML under the retained R.
This proves owner/client protocol only; all model/device values are stubs.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SELF = ROOT / 'deploy/final-engine-serve/selfrun'
IMAGE = 'sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
OVERLAY_NAME = 'overlay-v16s5-q18a-parity-n3s24q2024p-j2-cxfx42-b1'
TAG = 'fe-20261115c'  # set from --tag in main (the packet's SERVE_TAG; a new tag = a fresh key)


def free_port(port):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('0.0.0.0', port))
    return port


def write(path, text, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)


def sums(d, names):
    import hashlib
    write(d / 'SHA256SUMS', ''.join('%s  %s\n' % (hashlib.sha256((d / n).read_bytes()).hexdigest(), n) for n in names))


def layout(out, a):
    R = out / 'root'
    prod = R / 'data/vllm/flash-next-tp4/prod'
    prod.mkdir(parents=True)
    shutil.copy2(a.fetch_b1 / 'private/launcher.found', prod / 'run-flash-prod.sh')
    (prod / 'run-flash-prod.sh').chmod(0o755)
    subprocess.run(['cp', '-a', str(a.bundle / 'native30_overlay'), str(prod / OVERLAY_NAME)], check=True)
    v16 = prod / 'overlay-v16'
    for n in ('qsa.py', 'ops_qsa.py', 'core/block_pool.py', 'core/single_type_kv_cache_manager.py'):
        write(v16 / n, '# self-run stand-in for the box overlay-v16 %s (never mounted into a real engine)\n' % n)
    sums(v16 / 'core', ['block_pool.py', 'single_type_kv_cache_manager.py'])
    sums(v16, ['qsa.py', 'ops_qsa.py'])
    write(R / 'data/vllm/flash-next-tp4/models/intel-w4a16/config.json', '{"model_type": "selfrun-stub"}\n')
    write(R / 'etc/default/vllm-flash', (a.fetch_b1 / 'private/defaults.found').read_text())
    (R / 'tmp').mkdir()
    shutil.copy2(ROOT / 'deploy/setcfg.py', R / 'tmp/setcfg.py')
    (R / 'root').mkdir(mode=0o700)
    (R / 'home/<service-user>/s0').mkdir(parents=True)
    SR = out / 'shimstate'
    sd = SR / 'systemd'
    sd.mkdir()
    write(sd / 'vllm-qwen38.enabled', 'enabled\n')
    write(sd / 'vllm-qwen38.active', 'inactive\n')
    write(sd / 'vllm-flash.enabled', 'disabled\n')
    # the image's own env = the recorded cxfx42 container env minus every key the run sets (the S5 launcher's -e words, the
    # b1 settings' EXTRA_ENV words, the --env-file key): docker inspect shows image env + run env, the readback compares both
    launcher = (a.bundle / 'native30_overlay/run-flash-s5-test.sh').read_text()
    run_keys = set(re.findall(r'-e ([A-Z][A-Z0-9_]*)=', launcher)) | {'VLLM_API_KEY'}
    for line in (a.fetch_b1 / 'private/defaults.next.generated').read_text().splitlines():
        if line.startswith('EXTRA_ENV='):
            run_keys |= {w.split('=', 1)[0] for w in line.split('=', 1)[1].strip('"').split()}
    rec = json.loads((out / 'packet/g0-container-env.json').read_text())
    image_env = {k: ('x' * v['redacted_length'] if isinstance(v, dict) else v) for k, v in rec.items() if k not in run_keys}
    write(SR / 'image-env.json', json.dumps(image_env, sort_keys=True, indent=1) + '\n')
    return R, SR


class Run:
    def __init__(self, out, env, SR):
        self.out, self.env, self.SR, self.steps = out, env, SR, []
        (out / 'steps').mkdir()

    def seen(self, t0, t1):
        p = self.SR / 'requests.jsonl'
        rows = [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
        acts, paths = [], {}
        for r in rows:
            if t0 <= r['t'] <= t1:
                if r['kind'] in ('control', 'worker'):
                    name = r['action'] if r['kind'] == 'control' else 'worker:%s' % r.get('worker_action')
                    if not acts or acts[-1][0] != name:
                        acts.append([name, 0])
                    acts[-1][1] += 1
                else:
                    k = '%s %s' % (r['path'], r['code'])
                    paths[k] = paths.get(k, 0) + 1
        return ['%s x%d' % (k, n) if n > 1 else k for k, n in acts], paths

    def step(self, name, argv, expect, rc_ok=(0,), extra_env=None, timeout=1800, check=None):
        env = dict(self.env, **(extra_env or {}))
        log = self.out / 'steps' / ('%02d-%s.log' % (len(self.steps) + 1, name))
        t0 = time.time()
        with open(log, 'w') as h:
            p = subprocess.run(argv, env=env, stdout=h, stderr=subprocess.STDOUT, timeout=timeout, cwd=self.out)
        t1 = time.time()
        text = log.read_text(errors='replace')
        hits = [ln for ln in text.splitlines() if re.search(expect, ln)]
        extra = check() if check else (True, '')
        ok = p.returncode in rc_ok and bool(hits) and extra[0]
        acts, paths = self.seen(t0, t1)
        rec = dict(step=name, argv=argv, env=sorted((extra_env or {}).items()), rc=p.returncode, expect=expect,
                   verdict=hits[-1] if hits else (text.strip().splitlines() or [''])[-1], check=extra[1], ok=ok,
                   seconds=round(t1 - t0, 1), log=str(log), stub_controls=acts, stub_http=paths)
        self.steps.append(rec)
        print('SELFRUN %-22s %s rc=%d %s%s' % (name, 'PASS' if ok else 'FAIL', p.returncode, rec['verdict'][:200],
                                               (' [%s]' % extra[1]) if extra[1] else ''), flush=True)
        return rec


def http(url, key=None, body=None, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 headers=dict({'Content-Type': 'application/json'}, **({'Authorization': 'Bearer ' + key} if key else {})))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except OSError:
        return 0, b''


def main(argv=None):
    global TAG, OVERLAY_NAME
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--g0', type=Path, required=True)
    p.add_argument('--fetch-b1', type=Path, required=True)
    p.add_argument('--port', type=int, default=18731)
    p.add_argument('--tag', default=TAG)
    a = p.parse_args(argv)
    TAG = a.tag
    OVERLAY_NAME = json.loads((a.packet / 'vision-delta.json').read_text())['overlay_name']
    out = a.out.resolve()
    out.mkdir(parents=True)
    port = free_port(a.port)
    PK, R, SR = out / 'packet', out / 'root', out / 'shimstate'
    SR.mkdir()
    shims = SELF / 'shims'
    hub_key_dir = out / 'hub' / ('serve-' + TAG)
    env = dict(os.environ, PATH='%s:%s' % (shims, os.environ['PATH']), R=str(R), SR=str(SR), STUB_ENGINE=str(SELF / 'stub_engine.py'),
               STUB_IMAGE=IMAGE, STUB_RECORDED=str(a.fetch_b1 / 'startup-grade.json'),
               STUB_RECORDED_ARM=str(a.fetch_b1 / 'arm-n3.g2-live-0.json'), STUB_OVERLAY=str(a.bundle / 'native30_overlay'), SERVE_PORT=str(port),
               SERVE_OWNER_UID=str(os.getuid()), SERVE_OWNER=os.environ.get('USER', '<service-user>'), HEALTH_DEADLINE_S='90',
               PVE=str(shims / 'pve'), HUBD=str(hub_key_dir), STUB_IMAGE_ENV=str(SR / 'image-env.json'), PYTHONDONTWRITEBYTECODE='1')
    # Real recorded identity shape, with explicitly synthetic MM byte fields.
    recorded=json.loads((a.fetch_b1 / 'startup-grade.json').read_text())
    def add_mm_fixture(value):
        if isinstance(value,dict):
            inv=value.get('native_inventory')
            if isinstance(inv,dict):
                rank=inv['worker_rank']
                inv['native_mm']=dict(schema='native-mm-inventory/1',tower_registered_cuda_bytes=0,
                    gpu_encoder_cache_present=rank<2,cache_budget_tokens=8192,cache_budget_bytes=40*2**20,
                    cache_feature_bytes=0,cache_storage_bytes=0,encoder_inputs_buffer_bytes=5*2**20 if rank<2 else 0,
                    active_splice_device_bytes=0,basis='CPU protocol fixture only; NOT measured bytes')
            for child in value.values():add_mm_fixture(child)
        elif isinstance(value,list):
            for child in value:add_mm_fixture(child)
    add_mm_fixture(recorded)
    fixture=SR/'recorded-mm.json';fixture.write_text(json.dumps(recorded))
    env['STUB_RECORDED']=str(fixture)
    run = Run(out, env, SR)
    url = 'http://127.0.0.1:%d' % port
    eng_out, stock_out = R / ('data/serve-%s-engine' % TAG), R / ('data/serve-%s-stock' % TAG)
    found_launcher = (a.fetch_b1 / 'private/launcher.found').read_bytes()
    found_defaults = (a.fetch_b1 / 'private/defaults.found').read_bytes()

    def found_back():
        lf = (R / 'data/vllm/flash-next-tp4/prod/run-flash-prod.sh').read_bytes() == found_launcher
        df = (R / 'etc/default/vllm-flash').read_bytes() == found_defaults
        nx = not (R / 'etc/default/vllm-flash.next').exists()
        return lf and df and nx, 'found launcher %s, found defaults %s, no .next %s' % (lf, df, nx)

    shutil.copytree(a.packet, PK)
    run.step('frozen-packet', ['bash', '-c', 'cd "$1" && sha256sum --strict -c PACKET-SHA256SUMS', 'verify-frozen', str(PK)],
             r'vision-delta.json: OK', extra_env={}, timeout=60)
    layout(out, a)
    run.step('mint-key', ['bash', str(PK / 'serve-mint-key.sh')], r'^SERVE_KEY_MINTED len=64 ',
             check=lambda: ((R / ('root/.flashnext-serve-%s/vllm-api.env' % TAG)).read_bytes() == (hub_key_dir / 'vllm-api.env').read_bytes()
                            and oct((hub_key_dir / 'vllm-api.env').stat().st_mode & 0o777) == '0o600', 'box copy == hub copy, 0600'))
    run.step('mint-key-again', ['bash', str(PK / 'serve-mint-key.sh')], r'^hub key dir exists', rc_ok=(2,))
    key = (hub_key_dir / 'vllm-api.env').read_text().split('=', 1)[1].strip()
    # the guard refuses while production runs: no change made (found files intact, no container, OUT only has the guard log)
    (SR / 'systemd/vllm-qwen38.active').write_text('active\n')
    run.step('launch-engine-prod-up', ['bash', str(PK / 'serve-launch.sh'), 'engine'], r'^SERVE_LAUNCH FAILED at units$', rc_ok=(1,),
             extra_env=dict(OUT=str(R / ('data/serve-%s-engine-refused' % TAG))), check=found_back)
    (SR / 'systemd/vllm-qwen38.active').write_text('inactive\n')
    run.step('launch-engine', ['bash', str(PK / 'serve-launch.sh'), 'engine'], r'^SERVE_UP .* arm=engine .*auth key .*SERVE_SMOKE PASS$',
             check=lambda: ((lambda d: (d.get('auth') == 'key' and (d.get('noauth_probe') or {}).get('code') == 401,
                                        'readback auth=%s noauth %s' % (d.get('auth'), d.get('noauth_probe'))))(
                 json.loads((eng_out / 'readback.json').read_text()))))

    def contract(arm, want_noauth):
        c1, b1 = http(url + '/v1/models', key)
        c0, _ = http(url + '/v1/models')
        ids = [m['id'] for m in json.loads(b1).get('data', [])] if c1 == 200 else []
        c2, b2 = http(url + '/v1/chat/completions', key, dict(model='Qwen3.8-Flash-Next', messages=[dict(role='user', content='hi')], max_tokens=4))
        ok = c1 == 200 and 'Qwen3.8-Flash-Next' in ids and c0 == want_noauth and c2 == 200
        line = 'CONTRACT %s %s models-with-bearer %d ids %s no-bearer %d chat %d' % ('PASS' if ok else 'FAIL', arm, c1, ','.join(ids), c0, c2)
        (out / 'steps' / ('contract-%s.txt' % arm)).write_text(line + '\n')
        return ok, line

    contract('engine', 401)
    run.step('contract-engine', [sys.executable, '-c', 'print(open(%r).read().strip())' % str(out / 'steps/contract-engine.txt')], r'^CONTRACT PASS')
    # VRAM floor: GPU 2 below 40 MiB free for ~12 s (>= two 5 s floor samples): the recorder latches floor-breach.json
    (SR / 'gpu-override').write_text('22850,22740,24550,22850\n')
    time.sleep(12)
    (SR / 'gpu-override').write_text('')
    breach = eng_out / 'floor-breach.json'
    run.step('floor-breach', [sys.executable, '-c', 'import json,sys;d=json.load(open(sys.argv[1]));print("FLOOR_BREACH", d.get("status"), d.get("low"), "floor", d.get("floor_mib"), "action", d.get("action"))',
                              str(breach)], r'^FLOOR_BREACH ')
    # two recorder ticks (60 s apart; the first one at start)
    t_end = time.time() + 150
    while time.time() < t_end:
        rows = (eng_out / 'recorder.jsonl').read_text().splitlines() if (eng_out / 'recorder.jsonl').exists() else []
        if sum('"tick"' in x for x in rows) >= 2:
            break
        time.sleep(5)
    # the ticks after the breach: the recorder survived it, and the tick after it carries the floor counts (breached, low samples)
    run.step('recorder-ticks', [sys.executable, '-c', 'import json,sys;a=[json.loads(x) for x in open(sys.argv[1])];r=[x for x in a if "tick" in x];'
                                'f=[x for x in a if x.get("kind")=="floor"];'
                                'print("RECORDER_TICKS", len(r), "floor_rows", len(f), "lane_epoch", (r[-1].get("lane") or {}).get("engine_epoch"), "floor", r[-1]["vram_floor"])',
                                str(eng_out / 'recorder.jsonl')], r'^RECORDER_TICKS [2-9] floor_rows 1 lane_epoch [0-9]+ floor .*\'breached\': True')
    # drain with one request in flight (20 s hold): the drain must see running 1, then quiet x3, then stop to READY
    held = threading.Thread(target=http, args=(url + '/v1/chat/completions', key, dict(model='Qwen3.8-Flash-Next', max_tokens=4,
                            messages=[dict(role='user', content='held')], metadata=dict(stub_hold_s=20))), kwargs=dict(timeout=120))
    held.start()
    time.sleep(1)
    run.step('drain-engine', ['bash', str(PK / 'serve-drain.sh'), str(eng_out), '120'], r'^SERVE_STOP_STATE READY ',
             check=lambda: ((lambda t: ('running 1' in t and 'SERVE_DRAIN drained' in t, 'drain saw running 1 then drained'))(
                 (run.out / 'steps' / ('%02d-drain-engine.log' % (len(run.steps) + 1))).read_text())))
    held.join()
    run.step('stop-engine-again', ['bash', str(PK / 'serve-stop.sh'), str(eng_out)], r'^SERVE_STOP_STATE READY ', check=found_back)
    # RULE-TEST-SERVE-KEYED-V2: a keyless stock serve is refused before any change (no key file -> auth step, found files intact)
    run.step('launch-stock-keyless', ['bash', str(PK / 'serve-launch.sh'), 'stock'], r'^SERVE_LAUNCH FAILED at auth$', rc_ok=(1,),
             extra_env=dict(OUT=str(R / ('data/serve-%s-stock-keyless' % TAG)), AUTH_ENV=str(R / ('root/.flashnext-serve-%s/absent.env' % TAG))),
             check=found_back)
    run.step('launch-stock', ['bash', str(PK / 'serve-launch.sh'), 'stock'], r'^SERVE_UP .* arm=stock .*auth key .*SERVE_SMOKE PASS$',
             check=lambda: ((lambda d: (d.get('auth') == 'key' and (d.get('noauth_probe') or {}).get('code') == 401,
                                        'readback auth=%s noauth %s' % (d.get('auth'), d.get('noauth_probe'))))(
                 json.loads((stock_out / 'readback.json').read_text()))))
    contract('stock', 401)
    run.step('contract-stock', [sys.executable, '-c', 'print(open(%r).read().strip())' % str(out / 'steps/contract-stock.txt')], r'^CONTRACT PASS')
    run.step('stop-stock', ['bash', str(PK / 'serve-stop.sh'), str(stock_out)], r'^SERVE_STOP_STATE READY ', check=found_back)
    passed = sum(s['ok'] for s in run.steps)
    verdict = 'PASS' if passed == len(run.steps) else 'FAIL'
    (out / 'selfrun.json').write_text(json.dumps(dict(schema='final-engine-serve-selfrun/1', verdict=verdict, port=port, root=str(R),
                                                      packet_sums=subprocess.run(['sha256sum', str(PK / 'PACKET-SHA256SUMS')], capture_output=True,
                                                                                 text=True).stdout.split()[0] if (PK / 'PACKET-SHA256SUMS').exists() else None,
                                                      steps=run.steps), indent=1) + '\n')
    print('FINAL_ENGINE_SERVE_SELFRUN %s %d/%d out %s' % (verdict, passed, len(run.steps), out))
    return 0 if verdict == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
