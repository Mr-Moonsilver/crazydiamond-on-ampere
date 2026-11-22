#!/usr/bin/env python3
"""Final-engine serve: effective-settings readback (stdlib only; values that look secret are shown as lengths).

  serve_readback.py found       --defaults F --params P          the found /etc/default/vllm-flash is the pinned deployment file
  serve_readback.py rendered    --arm A --defaults NEXT --found F --params P --overlay O --auth K
        the rendered defaults differ from the found file in exactly the declared keys (arm engine: the cxfx42 S5 keys plus
        the serve keys; arm stock: MAX_MODEL_LEN, MAX_SEQS, QSA_ROWCACHE_SLOTS, SERVE_AUTH_ENV_FILE only)
  serve_readback.py engine-args --log L --params P               the engine's own echo: max_model_len, max_num_seqs, host 0.0.0.0, no speculation
  serve_readback.py container   --arm A --container C --params P --g0-env G --overlay O --out J
        `docker inspect`: image = the pin; env QSA_ROWCACHE_SLOTS/REQS; engine: every FLASH_S5_*/FLASH_*/VLLM_*/QSA_* env equal to the
        G0 candidate's container env (G0-manifest effective.container_env) except the declared serve keys; VLLM_API_KEY present
        (length 64); stock: no FLASH_S5_* env; the overlay mounts.  BOTH arms (c packet, RULE-TEST-SERVE-KEYED-V2): VLLM_API_KEY
        length 64 in the container and one unauthenticated GET /v1/models answered 401, recorded as auth=key (else
        'keyless-test-serve').  Writes J; last line SERVE_READBACK PASS|FAIL.
"""
import argparse
import json
import re
import shlex
import subprocess
import sys

SECRET = re.compile(r'(KEY|TOKEN|SECRET|PASSWORD|AUTH_ENV|BUILD_URL)', re.I)
# container env keys the serve declares different from the G0 candidate (both read from the params)
SERVE_ENV = ('QSA_ROWCACHE_SLOTS', 'QSA_ROWCACHE_REQS')
# never compared by value (process/session specifics)
VOLATILE = ('HOSTNAME', 'HOME', 'PATH', 'NVIDIA_REQUIRE_CUDA')



def _packet_error(exc):
    import sys, traceback
    frame=traceback.extract_tb(exc.__traceback__)[-1]
    print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {str(exc)[:256]} at {frame.filename}:{frame.lineno}', file=sys.stderr)

def params(path):
    out = {}
    for line in open(path).read().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1)
            out[k.strip()] = v.strip()
    return out


def parse_defaults(path):
    vals = {}
    for line in open(path).read().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1)
            vals[k.strip()] = ' '.join(shlex.split(v)) if v else ''
    return vals


def show(k, v):
    return '<len %d>' % len(v) if SECRET.search(k) and v else v


def sha(path):
    import hashlib
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def cmd_found(a):
    p = params(a.params)
    got = sha(a.defaults)
    vals = parse_defaults(a.defaults)
    print('found defaults sha256 %s (pinned %s)' % (got[:16], p['FOUND_DEFAULTS_SHA'][:16]))
    for k in ('OVERLAY', 'QSA_ROWCACHE_SLOTS', 'MAX_MODEL_LEN', 'MAX_SEQS', 'EXTRA_ARGS', 'EXTRA_ENV', 'GPU_KV_BYTES', 'MTP_TOKENS'):
        print('found %s=%s' % (k, show(k, vals.get(k))))
    bad = got != p['FOUND_DEFAULTS_SHA'] or 'VLLM_API_KEY' in vals or 'VLLM_API_KEY' in vals.get('EXTRA_ENV', '')
    print('SERVE_FOUND %s' % ('FAIL' if bad else 'PASS'))
    return 1 if bad else 0


def cmd_rendered(a):
    p = params(a.params)
    new, old = parse_defaults(a.defaults), parse_defaults(a.found)
    changed = sorted(k for k in set(new) | set(old) if new.get(k) != old.get(k))
    want = dict(MAX_MODEL_LEN=p['MAX_MODEL_LEN'], MAX_SEQS=p['MAX_SEQS'])
    bad = []
    if a.arm == 'engine':
        want.update(EXTRA_ARGS='--prefix-match-unit 176 --enable-prompt-tokens-details', FLASH_S5_AUTH_ENV_FILE=a.auth, OVERLAY=a.overlay,
                    S5='1', CORE_OVERLAY='1', PREFILL_CHUNK='1024', QSA_ROWCACHE_SLOTS=p['ENGINE_ROWCACHE_SLOTS'], PP_LAYER_PARTITION='24,24')
        want.update(VISION='1', VISION_ARGS='--enable-mm-embeds --limit-mm-per-prompt {"image":0,"video":0} --mm-processor-cache-gb 0 --skip-mm-profiling')
        env = new.get('EXTRA_ENV', '').split()
        for flag in ('FLASH_S5_G4_HEADROOM=1', 'FLASH_S5_NATIVE_MM=1', 'FLASH_S5_OVERLAP=1', 'FLASH_S5_AUTO_ARM=1', 'FLASH_S5_YIELD_WINDOW=1', 'FLASH_S5_YIELD_WINDOW_FIX=1',
                     'FLASH_S5_DECODE_OVERLAP_SMS=42', 'FLASH_S5_DECODE_GROUPS=chunk', 'FLASH_S5_O3=1', 'FLASH_S5_DECODE_SMS=24'):
            print('EXTRA_ENV has %s: %s' % (flag, flag in env))
            if flag not in env:
                bad.append('extra-env:' + flag)
        allowed = None          # the engine's S5 keys replace many found keys; pinned values above, the rest by the settings file
    else:
        # c packet (RULE-TEST-SERVE-KEYED-V2): the keyed stock launcher reads the serve key file named here
        want.update(QSA_ROWCACHE_SLOTS=p['STOCK_ROWCACHE_SLOTS'], SERVE_AUTH_ENV_FILE=a.auth)
        allowed = {'MAX_MODEL_LEN', 'MAX_SEQS', 'QSA_ROWCACHE_SLOTS', 'SERVE_AUTH_ENV_FILE'}
        for k in ('OVERLAY', 'EXTRA_ENV', 'EXTRA_ARGS', 'CORE_OVERLAY', 'PREFILL_CHUNK', 'GPU_KV_BYTES', 'HOST_KV_GIB_PER_RANK', 'GPU_ORDER'):
            print('stock %s=%s (found, unchanged: %s)' % (k, show(k, new.get(k)), new.get(k) == old.get(k)))
    for k, v in sorted(want.items()):
        print('serve %s=%s' % (k, show(k, new.get(k))))
        if new.get(k) != v:
            bad.append(k)
    print('changed keys vs found: %s' % ','.join(changed))
    if allowed is not None and set(changed) - allowed:
        bad.append('undeclared-change:' + ','.join(sorted(set(changed) - allowed)))
    if '--host' in new.get('EXTRA_ARGS', '') or 'VLLM_API_KEY' in new or 'VLLM_API_KEY' in new.get('EXTRA_ENV', ''):
        bad.append('host-or-key-in-defaults')
    print('SERVE_RENDERED %s %s' % ('FAIL' if bad else 'PASS', ','.join(bad)))
    return 1 if bad else 0


def engine_args(text):
    def first(rx):
        m = re.search(rx, text)
        return m.group(1) if m else None
    return dict(max_model_len=first(r"max_model_len'?[=:] ?(\d+)"), max_num_seqs=first(r"max_num_seqs'?[=:] ?(\d+)"),
                host=first(r"'host': '([^']*)'"), speculative=first(r"speculative_config=(\w+)"))


def cmd_engine_args(a):
    p = params(a.params)
    got = engine_args(open(a.log, errors='replace').read())
    for k, v in got.items():
        print('engine %s=%s' % (k, v))
    ok = (got['max_model_len'] == p['MAX_MODEL_LEN'] and got['max_num_seqs'] == p['MAX_SEQS'] and got['host'] == '127.0.0.1'
          and got['speculative'] == 'None')
    print('engine max_model_len=%s max_num_seqs=%s host=%s speculative=%s' % tuple(got[k] for k in ('max_model_len', 'max_num_seqs', 'host', 'speculative')))
    return 0 if ok else 1


def inspect(container):
    out = subprocess.run(['docker', 'inspect', container], capture_output=True, text=True, timeout=60)
    if out.returncode:
        raise SystemExit('SERVE_READBACK FAIL docker-inspect rc=%d' % out.returncode)
    return json.loads(out.stdout)[0]


def noauth_probe(url):
    """GET <url>/v1/models with NO Authorization header: {path, code} (0 = no answer).  The keyed serve must say 401."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(urllib.request.Request(url + '/v1/models'), timeout=20) as r:
            code = r.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as _packet_exc:  # noqa: BLE001 - no answer is not a 401
        _packet_error(_packet_exc)
        code = 0
    return dict(path='/v1/models', code=code)


def auth_diffs(env, probe):
    """Both arms (RULE-TEST-SERVE-KEYED-V2): the container holds a 64-character key and the unauthenticated probe got 401."""
    out = []
    if len(env.get('VLLM_API_KEY', '')) != 64:
        out.append('keyless-test-serve:VLLM_API_KEY len %d' % len(env.get('VLLM_API_KEY', '')))
    if probe.get('code') != 401:
        out.append('keyless-test-serve:noauth %s %s' % (probe.get('path'), probe.get('code')))
    return out


def cmd_container(a):
    p = params(a.params)
    d = inspect(a.container)
    env = dict(e.split('=', 1) for e in (d.get('Config') or {}).get('Env') or [] if '=' in e)
    args = (d.get('Config') or {}).get('Cmd') or d.get('Args') or []
    mounts = sorted('%s:%s' % (m.get('Source'), m.get('Destination')) for m in d.get('Mounts') or [])
    image = d.get('Image')
    diffs = []
    if image != p['IMAGE_ID']:
        diffs.append('image:%s' % image)
    rc = p['ENGINE_ROWCACHE_SLOTS'] if a.arm == 'engine' else p['STOCK_ROWCACHE_SLOTS']
    if env.get('QSA_ROWCACHE_SLOTS') != rc:
        diffs.append('QSA_ROWCACHE_SLOTS:%s' % env.get('QSA_ROWCACHE_SLOTS'))
    if env.get('QSA_ROWCACHE_REQS') != p['MAX_SEQS']:
        diffs.append('QSA_ROWCACHE_REQS:%s' % env.get('QSA_ROWCACHE_REQS'))
    if a.arm == 'engine':
        g0 = json.load(open(a.g0_env))
        for k in sorted(set(g0) | set(env)):
            if k in SERVE_ENV or k in VOLATILE:
                continue
            ref, got = g0.get(k), env.get(k)
            if type(ref) is dict and 'redacted_length' in ref:      # G0 kept a length only: compare lengths
                if got is None or len(got) != ref['redacted_length']:
                    diffs.append('%s:len %s!=%s' % (k, None if got is None else len(got), ref['redacted_length']))
            elif ref != got:
                diffs.append('%s:%s->%s' % (k, show(k, ref) if ref is not None else None, show(k, got) if got is not None else None))
        if not any(m.startswith(a.overlay + '/') or m.startswith(a.overlay + ':') for m in mounts):
            diffs.append('overlay-not-mounted')
    else:
        s5 = sorted(k for k in env if k.startswith('FLASH_S5_'))
        if s5:
            diffs.append('stock-has-FLASH_S5:%s' % ','.join(s5[:4]))
    if a.arm == 'engine':
        target='/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py'
        model_mounts=[m for m in d.get('Mounts',[]) if m.get('Destination')==target]
        if len(model_mounts)!=1 or model_mounts[0].get('Source')!=a.overlay+'/e3_sources/model.py':
            diffs.append('native-model-mount-shadowed')
    probe = noauth_probe(a.url)          # RULE-TEST-SERVE-KEYED-V2: one unauthenticated probe, before any test traffic
    diffs += auth_diffs(env, probe)
    words = [str(x) for x in args]
    def argv(flag):
        vals = [words[i + 1] for i, w in enumerate(words[:-1]) if w == flag]
        return vals[-1] if vals else None
    if argv('--max-model-len') != p['MAX_MODEL_LEN'] or argv('--max-num-seqs') != p['MAX_SEQS'] or argv('--host') != '127.0.0.1' or argv('--port') != '8001':
        diffs.append('argv:max_model_len=%s,max_num_seqs=%s,host=%s' % (argv('--max-model-len'), argv('--max-num-seqs'), argv('--host')))
    if a.arm == 'engine':
        try:
            limits=json.loads(argv('--limit-mm-per-prompt'))
        except (TypeError,ValueError):
            limits=None
        if ('--language-model-only' in words or '--enable-mm-embeds' not in words
                or '--skip-mm-profiling' not in words or limits != {'image':0,'video':0}
                or argv('--mm-processor-cache-gb') != '0'):
            diffs.append('vision-embedding-only-argv-differs')
    if '--speculative-config' in words or '--api-key' in words:
        diffs.append('argv:speculative-or-api-key')
    record = dict(schema='serve-readback/1', arm=a.arm, image=image, container_id=d.get('Id'), started=(d.get('State') or {}).get('StartedAt'),
                  env={k: show(k, v) for k, v in sorted(env.items())}, argv=[show('', w) if not SECRET.search(w) else '<len %d>' % len(w) for w in words],
                  mounts=mounts, auth='key' if not auth_diffs(env, probe) else 'NOT-KEYED', noauth_probe=probe,
                  diffs=diffs, verdict='FAIL' if diffs else 'PASS')
    with open(a.out, 'x') as handle:
        json.dump(record, handle, sort_keys=True, indent=1)
    print('readback env %d keys, argv %d words, mounts %d' % (len(env), len(words), len(mounts)))
    print('SERVE_READBACK %s arm=%s auth=%s noauth=%s image=%s rowcache=%s reqs=%s %s' % (
        record['verdict'], a.arm, record['auth'], probe.get('code'), (image or '')[:19], env.get('QSA_ROWCACHE_SLOTS'),
        env.get('QSA_ROWCACHE_REQS'), ';'.join(diffs)[:400]))
    return 1 if diffs else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('found'); f.add_argument('--defaults', required=True); f.add_argument('--params', required=True)
    r = sub.add_parser('rendered')
    for k in ('--arm', '--defaults', '--found', '--params', '--overlay', '--auth'):
        r.add_argument(k, required=True)
    e = sub.add_parser('engine-args'); e.add_argument('--log', required=True); e.add_argument('--params', required=True)
    c = sub.add_parser('container')
    for k in ('--arm', '--container', '--params', '--g0-env', '--overlay', '--out', '--url'):
        c.add_argument(k, required=True)
    a = p.parse_args(argv)
    return dict(found=cmd_found, rendered=cmd_rendered, container=cmd_container)[a.cmd](a) if a.cmd != 'engine-args' else cmd_engine_args(a)


if __name__ == '__main__':
    raise SystemExit(main())
