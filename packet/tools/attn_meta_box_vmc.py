#!/usr/bin/env python3
"""attn-meta run c on the 4x box instead of VM <vm-id> (lead 3 Nov 16:49: the 3090 serves <caller-a> all night). One card, CUDA, no model,
no engine. The lead runs it on the box between queue runs. Stdlib only; needs nothing of this tree on the box.

Supersedes the VM side of tools/attn_meta_vm_input_c.py by name (that tool and its wrapper tools/attn_meta_vm_run_gpu_c.py are
unchanged). The input is that tool's own, built on the hub, unchanged:
    python3 tools/attn_meta_vm_input_c.py build --name attn-meta-vm-c --work DIR --overlay-qsa HELD/native30_overlay/qsa.py
    python3 tools/attn_meta_vm_input_c.py check-bind --input DIR/attn-meta-vm-c-input --engine-dir HELD/e3_builder
(e3_builder holds the source bytes engine_bytes pins; native30_overlay/e3 holds packaged modules, two of which differ by design.)

The run is the wrapper the input carries (INPUT/run_gpu.py, sha-verified), not a copy: its text is exec'd here with the exact-once
swaps in WRAPPER_SWAPS, so probe c (/w/tools/attn_meta_vm_probe_c.py), the mounts, limits, memory cap and stop (3072 / 2816 MiB),
the 600 s deadline measured on the monotonic clock, and the whole PASS / FAIL / INCONCLUSIVE rule with rc 0 / 1 / 2 are the wrapper's.
What differs, and only this:
  card      docker --gpus device=N in place of --runtime nvidia and NVIDIA_VISIBLE_DEVICES=0 (the box launcher's --gpus form); the
            one-3090 assert becomes the named card; the same free-memory assert;
  image     the conf's IMAGE, whose local Id must equal it (--pull=never stays);
  lane      a flock in OUT_ROOT in place of /opt/flashnext/.job.lock, and the engine and GPU refusal below in place of 'VM lane occupied';
  memory    own_mib is the larger of the wrapper's PID sum and the card's rise over the baseline taken before the start (a read-only
            query of the named card), so an unattributable PID in the container cannot read as no memory; compute-app rows with a
            non-integer memory field count as 0 instead of stopping the run. The cleanup check also requires the card back within
            CARD_RELEASE_SLACK_MIB of its baseline, polled for up to CARD_RELEASE_WAIT_S;
  schema    attn-meta-box-vmc-owner/1, box_action true.
No gate of the wrapper is removed; both memory changes only add evidence or a condition.

Refused, rc 3, nothing started, no output directory created: an incomplete or unknown conf; input or bundle sums or manifests that do
not verify; engine bytes or overlay qsa not the bundle's; the image missing; the unit vllm-flash active; a running container whose
name matches the engine pattern or that has GPU device requests or the nvidia runtime; any compute process on any card; the named
card above GPU_IDLE_MAX_MIB; the container name or the output directory already used; a failing docker, nvidia-smi or systemctl query.

It starts no engine: no systemctl action (is-active only), no docker start or restart, nothing under /tmp/q*.sh, /tmp/stall2.sh,
/tmp/setcfg.py or the engine's directories, no restore, no production. The only container it stops or removes is its own, by exact
name; the wrapper retains the exited container, and removing it is the separate ``cleanup`` command, the lead's call. It reads no
auth file and passes no secret.

    python3 -B attn_meta_box_vmc.py check   --conf CONF     preflight only (read-only), rc 0 or 3
    python3 -B attn_meta_box_vmc.py run     --conf CONF     preflight, then the run; the wrapper's rc 0 / 1 / 2, or 3 refused
    python3 -B attn_meta_box_vmc.py cleanup --conf CONF     docker rm of its own exited container, after result.json exists
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SCHEMA = 'attn-meta-box-vmc-owner/1'
REFUSED_RC = 3
REQUIRED = ('INPUT_DIR', 'INPUT_MANIFEST', 'BUNDLE_DIR', 'BUNDLE_MANIFEST', 'GPUS', 'OUT_ROOT', 'IMAGE')
DEFAULTS = dict(ENGINE_UNIT='vllm-flash', ENGINE_NAME_RE=r'vllm|qwen|flash|engine', GPU_IDLE_MAX_MIB='512')
BIND_DIR = 'e3_builder'
OVERLAY_QSA = 'native30_overlay/qsa.py'
NAME_PREFIX = 'attn-meta-vm-c'
CARD_RELEASE_SLACK_MIB = 64
CARD_RELEASE_WAIT_S = 20
QUERY_TIMEOUT_S = 15


class Refused(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def query(args):
    """A read-only query; a missing program or a timeout refuses (fail closed)."""
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=QUERY_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Refused('query-failed:%s:%s' % (args[0], type(exc).__name__))


def read_conf(path):
    conf = {}
    try:
        text = Path(path).read_text()
    except OSError as exc:
        raise Refused('conf-unreadable:%s' % exc)
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        key, value = key.strip(), value.strip()
        if not sep or key not in REQUIRED + tuple(DEFAULTS):
            raise Refused('conf-unknown-line:%d:%s' % (number, key))
        if key in conf:
            raise Refused('conf-duplicate-key:' + key)
        conf[key] = value
    missing = [k for k in REQUIRED if not conf.get(k) or '@' in conf[k]]
    if missing:
        raise Refused('conf-incomplete:' + ','.join(missing))
    for key in ('INPUT_DIR', 'BUNDLE_DIR', 'OUT_ROOT'):
        if not conf[key].startswith('/'):
            raise Refused('conf-path-not-absolute:' + key)
    for key in ('INPUT_MANIFEST', 'BUNDLE_MANIFEST'):
        if not re.fullmatch(r'[0-9a-f]{64}', conf[key]):
            raise Refused('conf-not-a-sha256:' + key)
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', conf['IMAGE']):
        raise Refused('conf-image-not-a-digest')
    gpus = [g.strip() for g in conf['GPUS'].split(',') if g.strip()]
    if len(gpus) != 1 or not gpus[0].isdigit():
        raise Refused('conf-gpus-must-name-one-card:probe-c-runs-on-one-card:' + conf['GPUS'])
    conf['GPU'] = gpus[0]
    merged = dict(DEFAULTS, **conf)
    if not merged['GPU_IDLE_MAX_MIB'].isdigit():
        raise Refused('conf-not-a-number:GPU_IDLE_MAX_MIB')
    try:
        re.compile(merged['ENGINE_NAME_RE'])
    except re.error:
        raise Refused('conf-bad-pattern:ENGINE_NAME_RE')
    return merged


def verify_sums(root):
    """The wrapper's own input rule, before any of the input's text is exec'd."""
    sums = root / 'SHA256SUMS'
    if not sums.is_file():
        raise Refused('sums-missing:' + str(root))
    try:
        rows = {n: d for d, n in (x.split(maxsplit=1) for x in sums.read_text().splitlines())}
    except ValueError:
        raise Refused('sums-unreadable:' + str(root))
    every = list(root.rglob('*'))
    if any(p.is_symlink() for p in every):
        raise Refused('sums-symlink-in-input')
    if {p.relative_to(root).as_posix() for p in every if p.is_file()} != set(rows) | {'SHA256SUMS'}:
        raise Refused('sums-file-set-differs')
    bad = sorted(n for n, d in rows.items() if sha(root / n) != d)
    if bad:
        raise Refused('sums-mismatch:' + ','.join(bad))


def check_input(conf):
    root = Path(conf['INPUT_DIR'])
    if not root.is_dir():
        raise Refused('input-missing:' + str(root))
    if not root.name.startswith(NAME_PREFIX) or not root.name.endswith('-input'):
        raise Refused('input-name:' + root.name)
    verify_sums(root)
    if sha(root / 'SHA256SUMS') != conf['INPUT_MANIFEST']:
        raise Refused('input-manifest-differs')
    prov = json.loads((root / 'provenance.json').read_text())
    if prov.get('schema') != 'attn-meta-vm-input-c/1':
        raise Refused('input-schema:%s' % prov.get('schema'))
    return root, prov


def check_bundle(conf, prov, root):
    bundle = Path(conf['BUNDLE_DIR'])
    if not (bundle / 'SHA256SUMS').is_file():
        raise Refused('bundle-missing:' + str(bundle))
    if sha(bundle / 'SHA256SUMS') != conf['BUNDLE_MANIFEST']:
        raise Refused('bundle-manifest-differs')
    rows = {n: d for d, n in (x.split(maxsplit=1) for x in (bundle / 'SHA256SUMS').read_text().splitlines())}
    binds = []
    for rel, digest in sorted(prov['engine_bytes'].items()):
        name = BIND_DIR + '/' + Path(rel).name
        path = bundle / name
        got = sha(path) if path.is_file() else None
        if got != digest or rows.get(name) != digest:
            raise Refused('engine-bytes-not-the-bundles:' + name)
        binds.append(dict(file=name, sha256=digest))
    qsa = bundle / OVERLAY_QSA
    if not qsa.is_file() or sha(qsa) != sha(root / 'w/overlay-qsa.py') or rows.get(OVERLAY_QSA) != sha(qsa):
        raise Refused('overlay-qsa-not-the-bundles')
    return binds


def csv_rows(text):
    return [[v.strip() for v in row] for row in csv.reader(io.StringIO(text)) if row]


def card_memory(gpu):
    done = query(['nvidia-smi', '--id=' + gpu, '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'])
    rows = csv_rows(done.stdout) if done.returncode == 0 else []
    if len(rows) != 1 or len(rows[0]) != 2 or not all(v.isdigit() for v in rows[0]):
        raise Refused('nvidia-smi-card-query:' + gpu)
    return int(rows[0][0]), int(rows[0][1])


def engine_holders(conf, own):
    """Names of what holds or may hold the GPUs: the unit, running containers by name or by GPU access, compute processes."""
    held = []
    unit = query(['systemctl', 'is-active', conf['ENGINE_UNIT']]).stdout.strip()
    if unit not in ('inactive', 'failed', 'unknown'):
        held.append('unit:%s:%s' % (conf['ENGINE_UNIT'], unit or 'no-state'))
    ps = query(['docker', 'ps', '--format', '{{.ID}} {{.Names}}'])
    if ps.returncode:
        raise Refused('docker-ps-failed')
    pattern = re.compile(conf['ENGINE_NAME_RE'])
    for line in ps.stdout.splitlines():
        cid, _, name = line.strip().partition(' ')
        if not cid:
            continue
        if name == own:
            held.append('own-name-running:' + name)
            continue
        if pattern.search(name):
            held.append('container-name:' + name)
            continue
        info = query(['docker', 'inspect', '--format', '{{.HostConfig.Runtime}} {{json .HostConfig.DeviceRequests}}', cid])
        if info.returncode:
            raise Refused('docker-inspect-failed:' + name)
        runtime, _, requests = info.stdout.strip().partition(' ')
        if runtime == 'nvidia' or requests.strip() not in ('null', '[]', ''):
            held.append('container-gpu-access:' + name)
    apps = query(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'])
    if apps.returncode:
        raise Refused('nvidia-smi-compute-apps-failed')
    held += ['compute-process:' + ' '.join(row) for row in csv_rows(apps.stdout)]
    return held


def preflight(conf):
    root, prov = check_input(conf)
    binds = check_bundle(conf, prov, root)
    name = root.name[:-len('-input')]
    if prov.get('name') != name:
        raise Refused('input-name-differs-from-provenance')
    own = 'flashnext-' + name
    out_root = Path(conf['OUT_ROOT'])
    if not out_root.is_dir():
        raise Refused('out-root-missing:' + str(out_root))
    out = out_root / name
    if out.exists() or out.is_symlink():
        raise Refused('output-exists:' + str(out))
    image = query(['docker', 'image', 'inspect', conf['IMAGE'], '--format', '{{.Id}}'])
    if image.returncode or image.stdout.strip() != conf['IMAGE']:
        raise Refused('image-missing:' + conf['IMAGE'])
    used = query(['docker', 'ps', '-a', '--format', '{{.Names}}'])
    if used.returncode:
        raise Refused('docker-ps-failed')
    if own in used.stdout.splitlines():
        raise Refused('container-name-already-used:' + own)
    holders = engine_holders(conf, own)
    if holders:
        raise Refused('gpu-held:' + ';'.join(holders))
    mib, total = card_memory(conf['GPU'])
    if mib > int(conf['GPU_IDLE_MAX_MIB']):
        raise Refused('card-not-idle:%s:%d MiB' % (conf['GPU'], mib))
    return dict(root=root, out=out, own=own, name=name, baseline_mib=mib, card_total_mib=total, binds=binds)


# --- the wrapper, exec'd from the input with exact-once swaps -------------------------------------------------------------------

WRAPPER_SWAPS = (
    ("    lock = open('/opt/flashnext/.job.lock', 'a')       # existing lock retained; never unlinked\n"
     "    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n",
     "    lock = BOX['lock']()\n"),
    ("    running = command(['docker', 'ps', '--format', '{{.Names}}'])\n"
     "    assert running.returncode == 0 and not any(x.startswith('flashnext-') for x in running.stdout.splitlines()), \\\n"
     "        'VM lane occupied'\n",
     "    assert not BOX['holders'](), 'GPU lane occupied'\n"),
    ("    dev = command(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'])\n",
     "    dev = command(['nvidia-smi', '--id=' + BOX['gpu'], '--query-gpu=name,memory.total,memory.used', "
     "'--format=csv,noheader,nounits'])\n"),
    ("    assert len(devices) == 1 and '3090' in devices[0][0]\n", "    assert len(devices) == 1\n"),
    ("                    own_mib = sum(x['used_mib'] for x in own)\n",
     "                    own_mib = max(sum(x['used_mib'] for x in own), BOX['card_rise']())\n"
     "                    own = own or ([dict(pid=None, used_mib=own_mib, name='card-rise')] if own_mib else [])\n"),
    ("    cleanup = final.returncode == 0 and not any(x['pid'] in seen for x in parse_apps(final.stdout))\n",
     "    cleanup = final.returncode == 0 and not any(x['pid'] in seen for x in parse_apps(final.stdout)) and BOX['released']()\n"),
    ("record = dict(schema='attn-meta-vm-owner-c/1',", "record = dict(schema=BOX['schema'],"),
    ("box_action=False, engine_constructed=False,", "box_action=True, engine_constructed=False,"),
)
RUNTIME_ARGS = ['--runtime', 'nvidia']
VM_DEVICE_ARGS = ['--env', 'NVIDIA_VISIBLE_DEVICES=0']
PROBE = '/w/tools/attn_meta_vm_probe_c.py'


def swap_wrapper(text):
    for old, new in WRAPPER_SWAPS:
        if text.count(old) != 1:
            raise Refused('wrapper-text-moved:' + old.strip()[:60])
        text = text.replace(old, new)
    return text


def replace_once(args, old, new):
    hits = [i for i in range(len(args) - len(old) + 1) if args[i:i + len(old)] == old]
    if len(hits) != 1:
        raise Refused('wrapper-docker-args-moved:' + ' '.join(old))
    return args[:hits[0]] + new + args[hits[0] + len(old):]


def parse_apps_box(text):
    """The wrapper's rows; a non-integer memory field (a container's view of the driver) counts 0, the row stays."""
    rows = []
    for fields in csv.reader(io.StringIO(text)):
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError('compute-app schema')
        pid, used, name = (v.strip() for v in fields)
        rows.append(dict(pid=int(pid), used_mib=int(used) if used.isdigit() else 0, name=name))
    return rows


def load_wrapper(root, conf, plan):
    text = swap_wrapper((root / 'run_gpu.py').read_text())
    gpu, baseline = conf['GPU'], plan['baseline_mib']

    def lock():
        handle = open(Path(conf['OUT_ROOT']) / '.attn-meta-box-vmc.lock', 'a')    # retained; never unlinked
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle

    def card_rise():
        return max(0, card_memory(gpu)[0] - baseline)

    def released():
        import time
        end = time.monotonic() + CARD_RELEASE_WAIT_S
        while True:
            try:
                if card_rise() <= CARD_RELEASE_SLACK_MIB:
                    return True
            except Refused:              # the query failed: not shown released, the wrapper's missing own_gpu_cleanup (INC)
                pass
            if time.monotonic() >= end:
                return False
            time.sleep(1)

    box = dict(lock=lock, holders=lambda: engine_holders(conf, plan['own']), gpu=gpu, card_rise=card_rise, released=released,
               schema=SCHEMA)
    ns = dict(__name__='attn_meta_box_vmc_wrapper', __file__=str(root / 'run_gpu.py'), BOX=box)
    exec(compile(text, str(root / 'run_gpu.py') + ':box-swapped', 'exec'), ns)
    if ns['NAME'] != plan['own']:
        raise Refused('wrapper-container-name-differs:' + ns['NAME'])
    ns['IMAGE'] = conf['IMAGE']
    ns['parse_apps'] = parse_apps_box
    vm_command = ns['docker_command']

    def docker_command(wrapper_root, out):
        args = vm_command(wrapper_root, out)
        args = replace_once(args, RUNTIME_ARGS, [])
        args = replace_once(args, VM_DEVICE_ARGS, ['--gpus', 'device=' + gpu])
        if args.count(PROBE) != 1 or args.count(plan['own']) != 1:
            raise Refused('wrapper-probe-or-name-moved')
        return args

    ns['docker_command'] = docker_command
    return ns


def run(conf):
    plan = preflight(conf)
    root, out = plan['root'], plan['out']
    ns = load_wrapper(root, conf, plan)
    sys.argv = ['run_gpu.py', '--input', str(root), '--out', str(out)]
    try:
        rc = ns['main']()
    except AssertionError as exc:
        if (out / 'result.json').exists() or out.exists():
            raise
        raise Refused('wrapper-assert:%s' % exc)
    with (out / 'box-run.json').open('x') as handle:
        json.dump(dict(schema='attn-meta-box-vmc/1', rc=rc, gpu=conf['GPU'], card_baseline_mib=plan['baseline_mib'],
                       card_total_mib=plan['card_total_mib'], bundle=conf['BUNDLE_DIR'], bundle_manifest=conf['BUNDLE_MANIFEST'],
                       input=str(root), input_manifest=conf['INPUT_MANIFEST'], engine_bytes_bound=plan['binds'], image=conf['IMAGE'],
                       tool_sha256=sha(__file__), container=plan['own'],
                       scope_note="run on the box card, not VM <vm-id>: the input provenance's scope text is the VM's and is "
                                  'unchanged; card and CPU numbers here are the box\'s; still no model, no engine, no greedy tokens'),
                  handle, indent=2, sort_keys=True)
        handle.write('\n')
    return rc


def cleanup(conf):
    """The lead's call: docker rm (no force) of this run's own exited container, by exact name, once the record exists."""
    root = Path(conf['INPUT_DIR'])
    name = root.name[:-len('-input')]
    own = 'flashnext-' + name
    out = Path(conf['OUT_ROOT']) / name
    if not (out / 'result.json').is_file() or not (out / 'container-state.json').is_file():
        raise Refused('cleanup-no-record:' + str(out))
    state = query(['docker', 'inspect', '--format', '{{.Name}} {{.State.Running}}', own])
    if state.returncode or state.stdout.strip() != '/%s false' % own:
        raise Refused('cleanup-not-an-exited-own-container:' + own)
    if query(['docker', 'rm', own]).returncode:
        raise Refused('cleanup-rm-failed:' + own)
    print(json.dumps(dict(removed=own)))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('action', choices=('check', 'run', 'cleanup'))
    p.add_argument('--conf', required=True)
    a = p.parse_args(argv)
    try:
        conf = read_conf(a.conf)
        if a.action == 'check':
            plan = preflight(conf)
            print(json.dumps(dict(verdict='CLEAR', container=plan['own'], out=str(plan['out']), gpu=conf['GPU'],
                                  card_baseline_mib=plan['baseline_mib'], card_total_mib=plan['card_total_mib'],
                                  tool_sha256=sha(__file__)), sort_keys=True))
            return 0
        return run(conf) if a.action == 'run' else cleanup(conf)
    except Refused as exc:
        print(json.dumps(dict(verdict='REFUSED', reason=str(exc))), file=sys.stderr)
        return REFUSED_RC


if __name__ == '__main__':
    sys.exit(main())
