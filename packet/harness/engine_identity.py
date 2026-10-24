#!/usr/bin/env python3
"""Engine identity capture/check. Standard library only; no engine control.

capture OUT --request REQUEST [--container qwen-flash-prod] [--defaults PATH]
check BEFORE REQUEST [AFTER]
Exit: 0 verified, 1 identity mismatch, 2 unavailable/invalid evidence.
See identity.md for the exact request schema, exclusions and redaction limits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

SCHEMA = 'flash-engine-identity/v1'
REQUEST = 'flash-engine-request/v1'
HERE = Path(__file__).resolve().parent
HEX = re.compile(r'^[a-f0-9]{64}$')
NAME = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
SECRET = re.compile(r'api[_-]?key|secret|password|passwd|credential|authorization|cookie|private|access[_-]?key|(^|[_-])(key|token|pwd)([_-]|$)', re.I)
PUBLIC = set('PATH HOME USER LANG LC_ALL TZ HOSTNAME PYTHONPATH PYTHON_VERSION PYTHON_PIP_VERSION PYTHON_SETUPTOOLS_VERSION LD_LIBRARY_PATH LD_PRELOAD '
             'MODEL_DIR MODEL_PROFILE OVERLAY IMAGE GPU_ORDER PP_LAYER_PARTITION MAX_MODEL_LEN MAX_SEQS '
             'GPU_KV_BYTES HOST_KV_GIB_PER_RANK PREFILL_CHUNK PREFIX_CACHING MTP_TOKENS CORE_OVERLAY '
             'VISION BREAKABLE_CG MPS_PIPE PORT TP PP EAGER DISABLE_CUSTOM_ALL_REDUCE SPARSE_PREFILL '
             'OMP_NUM_THREADS HF_HUB_OFFLINE HF_HUB_DISABLE_TELEMETRY'.split())
PUBLIC_PREFIX = ('VLLM_', 'QSA_', 'FLASH_', 'CUDA_', 'NCCL_', 'PYTORCH_', 'TORCH_', 'TRITON_', 'NVIDIA_')
PUBLIC_FLAGS = set('model served-model-name tensor-parallel-size pipeline-parallel-size dtype kv-cache-dtype '
                   'kv-cache-memory-bytes max-model-len max-num-seqs max-num-batched-tokens gpu-memory-utilization '
                   'mamba-cache-mode reasoning-parser tool-call-parser max-logprobs host port prefix-match-unit '
                   'revision tokenizer-revision tokenizer load-format quantization distributed-executor-backend '
                   'max-num-partial-prefills max-long-partial-prefills long-prefill-token-threshold'.split())
JSON_FLAGS = {'compilation-config', 'engram-config', 'speculative-config', 'limit-mm-per-prompt', 'mm-processor-kwargs'}
PUBLIC_BOOLEANS = set('enable-expert-parallel language-model-only enable-prefix-caching no-enable-prefix-caching '
                      'enable-chunked-prefill enable-auto-tool-choice enforce-eager disable-custom-all-reduce '
                      'trust-remote-code disable-log-requests'.split())


class EvidenceError(Exception):
    pass


def need(ok):
    if not ok:
        raise EvidenceError('invalid or incomplete evidence')


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def redacted(v, unclassified=False):
    out = {'redacted_length': len(v.encode('utf-8'))}
    if unclassified:
        out['unclassified'] = True
    return out


def is_redacted(v):
    return isinstance(v, dict) and set(v) == {'redacted_length'} and type(v['redacted_length']) is int and v['redacted_length'] >= 0


def clean_json(v):
    if isinstance(v, dict):
        return {k: redacted(str(x)) if SECRET.search(k) else clean_json(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean_json(x) for x in v]
    if isinstance(v, str) and ('://' in v or '\n' in v):
        return redacted(v)
    return v


def clean_value(k, v):
    if is_redacted(v):
        return v
    need(isinstance(v, str))
    if SECRET.search(k):
        return redacted(v)
    if k == 'EXTRA_ENV':
        words = shlex.split(v)
        need(all('=' in w for w in words))
        return clean_env(words)
    if k in {'EXTRA_ARGS', 'VISION_ARGS'}:
        return clean_args(shlex.split(v))
    if k == 'COMPILATION_CONFIG':
        return clean_json(json.loads(v))
    if (k in PUBLIC or k.startswith(PUBLIC_PREFIX)) and not any(s in v for s in ('://', '\n', '\r', 'Bearer ', 'hf_')):
        return v
    return redacted(v, unclassified=not any(s in v for s in ('://', '\n', '\r', 'Bearer ', 'hf_')))


def clean_env(env):
    if isinstance(env, dict):
        pairs = list(env.items())
    else:
        need(isinstance(env, list) and all(isinstance(x, str) and '=' in x for x in env))
        pairs = [s.split('=', 1) for s in env]
    out = {}
    for k, v in pairs:
        need(isinstance(k, str) and NAME.fullmatch(k) and k not in out)
        out[k] = clean_value(k, v)
    return dict(sorted(out.items()))


def clean_args(args):
    need(isinstance(args, list))
    result, flag = [], None
    for i, arg in enumerate(args):
        if is_redacted(arg):
            result.append(arg)
            continue
        need(isinstance(arg, str))
        if arg.startswith('--'):
            opt, eq, val = arg.partition('=')
            flag = opt[2:]
            if eq:
                result.append({'option': opt, 'value': clean_arg_value(flag, val)})
                flag = None
            else:
                result.append(opt)
        elif flag is None and (arg.startswith('/') or arg in {'serve', 'vllm', 'python', 'python3', '-m',
                                                              'vllm.entrypoints.openai.api_server', 'vllm.entrypoints.cli.main'}):
            result.append(arg)  # executable/module/model positional arguments
        else:
            result.append(clean_arg_value(flag, arg))
    return result


def clean_arg_value(flag, value):
    if flag and not SECRET.search(flag) and flag in PUBLIC_FLAGS:
        return redacted(value) if any(x in value for x in ('://', '\n', 'Bearer ')) else value
    if flag in JSON_FLAGS:
        return clean_json(json.loads(value))
    return redacted(value, unclassified=not bool(flag and SECRET.search(flag)))


def defaults_snapshot(raw):
    """Supported EnvironmentFile subset: KEY=value, single/double quotes, no continuations.

    No sourcing, interpolation or shell execution. Reject unsupported syntax instead
    of pretending shlex is the complete systemd EnvironmentFile parser.
    """
    values, lines = {}, []
    for line in raw.decode('utf-8').splitlines():
        s = line.strip()
        if not s or s.startswith(('#', ';')):
            lines.append({'ignored_line_bytes': len(line.encode())})
            continue
        key, sep, val = s.partition('=')
        key, val = key.strip(), val.strip()
        need(sep and NAME.fullmatch(key) and key not in values and not val.endswith('\\'))
        if val.startswith("'"):
            need(val.endswith("'") and "'" not in val[1:-1])
            val = val[1:-1]
        elif val.startswith('"'):
            need(val.endswith('"'))
            inner = val[1:-1]
            need(not re.search(r'(?<!\\)"', inner))
            val = re.sub(r'\\([\\"$`])', r'\1', inner)
        else:
            need(not any(c in val for c in ('"', "'", '\\')))
        values[key] = clean_value(key, val)
        lines.append({'key': key, 'value': values[key]})
    need(bool(values))
    return {'values': dict(sorted(values.items())), 'lines': lines}


def relative(p):
    return isinstance(p, str) and p not in ('', '.') and not p.startswith('/') and all(x not in ('', '.', '..') for x in p.split('/'))


def validate_leaf(c):
    need(isinstance(c, dict))
    if c.get('kind') == 'file':
        need(set(c) == {'kind', 'bytes', 'sha256'} and type(c['bytes']) is int and c['bytes'] >= 0 and bool(HEX.fullmatch(c['sha256'])))
    elif c.get('kind') == 'symlink':
        need(set(c) == {'kind', 'target', 'referent'} and isinstance(c['target'], str))
        validate_leaf(c['referent'])
        need(c['referent']['kind'] == 'file')
    else:
        need(c == {'kind': 'directory'} or c == {'kind': 'absent'})


def validate_content(c):
    if c.get('kind') != 'directory':
        validate_leaf(c)
        need(c['kind'] in ('file', 'symlink'))
        return
    need(set(c) == {'kind', 'entries', 'exclusions'})
    seen = set()
    for ex in c['exclusions']:
        need(set(ex) == {'path', 'reason'} and relative(ex['path']) and ex['path'] not in seen and isinstance(ex['reason'], str) and bool(ex['reason'].strip()))
        seen.add(ex['path'])
    paths = set()
    for entry in c['entries']:
        p = entry['path']
        need(relative(p) and p not in paths and not any(p == e or p.startswith(e + '/') for e in seen))
        paths.add(p)
        validate_leaf({k: v for k, v in entry.items() if k != 'path'})
        need(entry['kind'] != 'absent')


def validate_identity(d):
    need(set(d) == {'image', 'command', 'env', 'defaults', 'mounts'})
    need(set(d['image']) == {'id', 'repo_digests'} and re.fullmatch(r'sha256:[a-f0-9]{64}', d['image']['id']))
    need(isinstance(d['image']['repo_digests'], list) and all(isinstance(s, str) and re.fullmatch(r'.+@sha256:[a-f0-9]{64}', s) for s in d['image']['repo_digests']))
    need(set(d['command']) == {'path', 'args'} and isinstance(d['command']['path'], str) and d['command']['path'] and isinstance(d['command']['args'], list) and d['command']['args'])
    need(isinstance(d['env'], dict) and bool(d['env']) and isinstance(d['defaults'], dict) and bool(d['defaults']))
    need(isinstance(d['mounts'], list) and bool(d['mounts']))
    destinations = set()
    for m in d['mounts']:
        need(set(m) == {'type', 'source', 'destination', 'rw', 'propagation', 'content', 'watch'})
        need(m['type'] in ('bind', 'volume', 'tmpfs') and isinstance(m['source'], str) and isinstance(m['destination'], str) and m['destination'].startswith('/') and m['destination'] not in destinations)
        need(type(m['rw']) is bool and isinstance(m['propagation'], str))
        destinations.add(m['destination'])
        validate_content(m['content'])
        watched = set()
        for w in m['watch']:
            need(relative(w['path']) and w['path'] not in watched and m['content']['kind'] == 'directory')
            watched.add(w['path'])
            validate_leaf({k: v for k, v in w.items() if k != 'path'})


def load_request(path):
    d = json.loads(Path(path).read_text())
    need(set(d) == {'schema', 'identity'} and d['schema'] == REQUEST)
    validate_identity(d['identity'])
    # Request uses the normalized/redacted representation (same schema as capture).
    # Never log its contents, even on a mismatch or malformed input.
    return d


def run(args, timeout, data=None):
    try:
        p = subprocess.run(args, input=data, capture_output=True, text=True, timeout=timeout)
        need(p.returncode == 0)
        return json.loads(p.stdout)
    except Exception:
        # stderr, stdout, argv, exceptions can contain secrets. Not diagnostics.
        raise EvidenceError('docker command failed or returned invalid JSON') from None


def inspect(container, timeout):
    d = run(['docker', 'inspect', container], timeout)
    need(isinstance(d, list) and len(d) == 1)
    d = d[0]
    s = d['State']
    need(s['Running'] is True and s.get('Paused') is False and s.get('Restarting') is False and s['Status'] == 'running')
    need(re.fullmatch(r'[a-f0-9]{64}', d['Id']) and isinstance(s['StartedAt'], str) and s['StartedAt'] and type(s['Pid']) is int and s['Pid'] > 0 and type(d['RestartCount']) is int)
    return d


def execution(d):
    return {'id': d['Id'], 'started_at': d['State']['StartedAt'], 'pid': d['State']['Pid'], 'restart_count': d['RestartCount']}


def stable_inspect(d):
    # Healthcheck logs and other daemon bookkeeping change normally. Compare the
    # actual identity fields, using raw values in RAM to catch secret changes too.
    return (execution(d), d['Image'], d['Path'], d['Args'], d['Config']['Env'], descriptors(d))


def descriptors(d):
    mounts = []
    for m in d['Mounts']:
        mounts.append({'type': m['Type'], 'source': m.get('Source', ''), 'destination': m['Destination'], 'rw': m['RW'], 'propagation': m.get('Propagation', '')})
    return sorted(mounts, key=lambda m: m['destination'])


def capture(a, request):
    start = time.time_ns()
    before = inspect(a.container, a.timeout)
    raw_defaults = Path(a.defaults).read_bytes()
    snap = defaults_snapshot(raw_defaults)
    img = run(['docker', 'image', 'inspect', before['Image']], a.timeout)
    need(isinstance(img, list) and len(img) == 1 and img[0]['Id'] == before['Image'])
    expected = {m['destination']: m for m in request['identity']['mounts']}
    mounts = descriptors(before)
    need(len({m['destination'] for m in mounts}) == len(mounts))
    queries = []
    for m in mounts:
        want = expected.get(m['destination'], {})
        queries.append({'destination': m['destination'], 'exclusions': want.get('content', {}).get('exclusions', []), 'watch': [w['path'] for w in want.get('watch', [])]})
    probe = run(['docker', 'exec', '-i', before['Id'], 'python3', '-c', (HERE / 'identity_probe.py').read_text()], a.timeout,
                canonical({'mounts': queries, 'max_bytes': a.max_bytes, 'max_files': a.max_files}))
    need(set(probe) == {'mounts', 'bytes_read', 'files_read'} and len(probe['mounts']) == len(mounts))
    for m, observed in zip(mounts, probe['mounts']):
        need(set(observed) == {'destination', 'content', 'watch'} and observed['destination'] == m['destination'])
        m.update(content=observed['content'], watch=observed['watch'])
    after = inspect(a.container, a.timeout)
    # Raw equality stays in RAM: detects even same-length secret changes while
    # capturing, without persisting raw secret values or their hashes.
    need(stable_inspect(before) == stable_inspect(after) and raw_defaults == Path(a.defaults).read_bytes())
    ident = {'image': {'id': before['Image'], 'repo_digests': sorted(img[0].get('RepoDigests') or [])},
             'command': {'path': before['Path'], 'args': clean_args(before['Args'])},
             'env': clean_env(before['Config']['Env']), 'defaults': snap['values'], 'mounts': mounts}
    validate_identity(ident)
    return {'schema': SCHEMA, 'complete': True, 'capture_id': uuid.uuid4().hex, 'begin_ns': start, 'end_ns': time.time_ns(),
            'requested_sha256': digest(request), 'execution': execution(before), 'identity': ident,
            'defaults_path': str(Path(a.defaults).absolute()), 'defaults_snapshot': snap['lines'],
            'io': {'bytes_read': probe['bytes_read'], 'files_read': probe['files_read']},
            'limits': {'max_bytes': a.max_bytes, 'max_files': a.max_files},
            'redaction': 'Unknown or sensitive values are lengths only; same-length secret changes across captures are not attested.'}


def validate_capture(d):
    need(d.get('schema') == SCHEMA and d.get('complete') is True)
    validate_identity(d['identity'])
    need(re.fullmatch(r'[a-f0-9]{32}', d['capture_id']) and HEX.fullmatch(d['requested_sha256']))
    need(type(d['begin_ns']) is int and type(d['end_ns']) is int and 0 < d['begin_ns'] <= d['end_ns'])
    e = d['execution']
    need(set(e) == {'id', 'started_at', 'pid', 'restart_count'} and re.fullmatch(r'[a-f0-9]{64}', e['id']))
    need(isinstance(e['started_at'], str) and e['started_at'] and type(e['pid']) is int and e['pid'] > 0 and type(e['restart_count']) is int and e['restart_count'] >= 0)
    need(isinstance(d['defaults_snapshot'], list) and bool(d['defaults_snapshot']) and isinstance(d['defaults_path'], str))


def compare(before, request, after=None):
    validate_capture(before)
    want = request['identity']
    differences = []
    for label, d in [('before', before)] + ([('after', after)] if after is not None else []):
        validate_capture(d)
        if d['requested_sha256'] != digest(request):
            differences.append(label + ': request/policy changed')
        for k in want:
            if canonical(d['identity'][k]) != canonical(want[k]):
                differences.append(label + ': ' + k + ' differs from request')
    if after is not None:
        need(before['capture_id'] != after['capture_id'] and before['end_ns'] <= after['begin_ns'])
        for k in ('execution', 'identity', 'defaults_path', 'defaults_snapshot'):
            if canonical(before[k]) != canonical(after[k]):
                differences.append('between captures: ' + k + ' changed')
    def unclassified(x):
        if isinstance(x, dict):
            return x.get('unclassified') is True or any(unclassified(v) for v in x.values())
        return isinstance(x, list) and any(unclassified(v) for v in x)
    unknown = unclassified(before['identity']) or (after is not None and unclassified(after['identity']))
    verdict = 'FAIL' if differences else 'INCONCLUSIVE' if unknown else 'PASS'
    if unknown:
        differences.append('unclassified values: review public env/argument allowlist; length-only unknowns cannot PASS')
    return {'verdict': verdict, 'differences': differences,
            'scope': 'Exact declared configuration and non-excluded content; redacted values compared by length only.'}


def write(path, d):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        json.dump(d, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    c = sub.add_parser('capture')
    c.add_argument('out'); c.add_argument('--request', required=True)
    c.add_argument('--container', default='qwen-flash-prod')
    c.add_argument('--defaults', default='/etc/default/vllm-flash')
    c.add_argument('--timeout', type=int, default=300)
    c.add_argument('--max-bytes', type=int, default=512 * 1024**3)
    c.add_argument('--max-files', type=int, default=100000)
    v = sub.add_parser('check'); v.add_argument('before'); v.add_argument('request'); v.add_argument('after', nargs='?')
    a = parser.parse_args()
    artifact = None
    try:
        request = load_request(a.request)
        if a.action == 'capture':
            need(a.timeout > 0 and a.max_bytes > 0 and a.max_files > 0 and not Path(a.out).exists())
            artifact = capture(a, request)
            result = compare(artifact, request)
            artifact['check'] = result
        else:
            result = compare(json.loads(Path(a.before).read_text()), request,
                             json.loads(Path(a.after).read_text()) if a.after else None)
        rc = {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['verdict']]
    except Exception:
        # Never echo raw commands, file contents, docker stderr or parser errors.
        result = {'verdict': 'INCONCLUSIVE', 'differences': ['identity capture/check unavailable or malformed evidence']}
        rc = 2
        artifact = {'schema': SCHEMA, 'complete': False, 'check': result}
    if a.action == 'capture':
        try:
            write(a.out, artifact)
        except Exception:
            result = {'verdict': 'INCONCLUSIVE', 'differences': ['cannot create exclusive artifact']}
            rc = 2
    print('identity ' + canonical(result))
    return rc


if __name__ == '__main__':
    sys.exit(main())
