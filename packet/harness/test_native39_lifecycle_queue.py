"""Native39 lifecycle queue: the exact derivation from the trunk texts (held-b) and the native35-L block, the
lifecycle window at real wall clock, and the queue's own lifecycle bytes executed under bash against a loopback
engine leaf.  CPU only, no box action.

Connected compositions (the queue's bytes with the real checks, clients, tools and grades):
  Tail      gates (startup, row1) -> instruments (per-rank maps) -> native lifetime -> typed FULL hygiene ->
            cross-session leak -> pressure ready -> partial-tail leak -> pressure completion -> pressure lifetime ->
            instruments end -> the on_exit native35 lifetime grade: the queue segment with the real instruments
            check and session client, the real e3_auth_exec bearer hook, the real leaktest (8 x 3) against a
            chat leaf, the real pressure_wait, pp_release_grade (leak and lifetime) and native35_lifetime_grade;
            fake native lifetime client, pressure2 and partial-tail driver (each makes its tool's first engine
            call through the real auth hook) and a fake docker CLI.
  Workload  the real e3_q18a_lifecycle CLI with the queue's argv on the trunk W2 manifest: it validates the
            manifest and reaches its first control call; its fresh prompt is the one the w2-length guard checks.
  PoolTelemetry  the trunk engine's own pool telemetry bytes (core/block_pool.py, core/scheduler.py, sha-pinned)
            under each declared env, graded by the real pressure-lifetime oracle: INCONCLUSIVE without
            FLASH_POOL_STATS, PASS with 200, the pool's lookups/evictions identical either way.
  Settings  the lifecycle settings sourced through the real setcfg algorithm, the S5 launcher render and the
            declaration projection (test_native39_trunk_queue.Launcher.declared), then the lifecycle declare
            script's own PY_DECL and the preflight's env assertion lines on that identity.
Fixture leaves: native39_lifecycle_fixture_leaves.json.
"""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import native35p_gate
import native39_lifecycle_gate as gate
import native39_trunk_gate
import test_native39_trunk_queue as trunkq  # helpers and leaves only; its TestCases are collected in their own module

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / 'runs'
LIFECYCLE = RUNS / 'q18a_native39_lifecycle.draft.sh'
SETTINGS = RUNS / 'q18a_native39_lifecycle_settings.sh'
DECLARE = RUNS / 'q18a_native39_lifecycle_declare.sh'
# The parent texts: the trunk held-b texts (claude/native39-trunk 0b41fed) and native35-L's draft as held in bundle
# q18a-native35-l-bundle-1cdf24641e15 and run on the box (native35-L run 1, 31 Oct 2026).
PARENT_SHA = {'q18a_native39_trunk.draft.sh': 'e6c54accdb54090659ddc86b2615e520e2d1aa81d238dd78900da7675360013f',
              'q18a_native39_trunk_settings.sh': '8f0d5afae84cee747c4e7e6b051b492aa63634dfe8b6ec93992b2643c3dff59e',
              'q18a_native39_trunk_declare.sh': 'f756563dfae0514b9f2babd4a961afc5984cb1214d6194b57da346301abe7215',
              'q18a_native35_l.draft.sh': 'c4703d52ecd29a599bea356d7c7aceae5870677d52792dbe18a486e1ea6c1a86'}
MANIFEST = HERE / 'workloads/q18a-w2-overlap-native39.json'
KEY = 'fixture-key-not-a-secret'
QUALIFIED = {'startup': 'PASS', 'row1': 'PASS'}
SLOT_RECEIPTS = []
QUEUE_RECEIPTS = []  # one row per connected case, appended only after its assertions (the bundle probe asserts the set)


def record_case(case, **fields):
    QUEUE_RECEIPTS.append(dict(case=case, verdict='PASS', **fields))


def drafts_tool():
    spec = importlib.util.spec_from_file_location('native39_lifecycle_queue_drafts',
                                                  ROOT / 'tools/native39_lifecycle_queue_drafts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Fixture leaf lifecycle-docker-cli: `docker inspect --format {{.Id}}`, `docker exec <id> cat /proc/<pid>/maps` (the
# trunk's fixture leaf queue-docker-exec-maps, same semantics) and a one-shot `docker logs --timestamps <container>`
# that stamps the retained log so far: FAKE/engine.raw.<n> on the n-th call (fixture leaf lifecycle-engine-log).
FAKE_DOCKER = r'''
import datetime, json, os, sys
from pathlib import Path
fake = Path(os.environ['FAKE']); argv = sys.argv[1:]
cid = (fake / 'current-id').read_text().strip()
if argv[0] == 'inspect' and argv[1:3] == ['--format', '{{.Id}}'] and argv[3] == 'qwen-flash-prod':
    print(cid); sys.exit(0)
if argv[0] == 'exec':
    p = fake / 'maps' / argv[-1].split('/')[2]
    if argv[1] != cid or argv[2:4] != ['cat', argv[-1]] or not p.exists(): sys.exit(1)
    sys.stdout.write(p.read_text()); sys.exit(0)
if argv == ['logs', '--timestamps', 'qwen-flash-prod']:
    calls = (fake / 'calls.jsonl').read_text() if (fake / 'calls.jsonl').exists() else ''
    n = calls.count('"docker-logs"')
    with open(fake / 'calls.jsonl', 'a') as f: f.write(json.dumps({'name': 'docker-logs', 'argv': argv}) + '\n')
    for line in (fake / ('engine.raw.%d' % n)).read_text().splitlines(True):
        t = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f') + '000Z '
        sys.stdout.write(t + line)
    sys.exit(0)
sys.exit(2)
'''
# Fixture leaf lifecycle-phase-executables: e3_q18a_lifecycle (record shape of e3_q18a_lifecycle.run), pressure2
# (progress and record shapes of pressure2.py, STOP honoured) and leak_partial_tail (record shape of
# leak_partial_tail.py: 10 x 4 sessions, partial hits, cancellations, one container identity).  pressure2 and
# leak_partial_tail make their tool's first engine call (/tokenize) through the real e3_auth_exec bearer hook.
FAKE_TOOL = r'''
import json, os, sys, time, urllib.request
from pathlib import Path
fake = Path(os.environ['FAKE']); name = Path(__file__).stem
keys = ('PRESSURE_READY', 'PRESSURE_MAX_TOKENS', 'MODE', 'GEN', 'TIMEOUT', 'CANCEL_FRAC', 'NEED_PARTIAL', 'UNIQUE_PREFIX')
with open(fake / 'calls.jsonl', 'a') as f:
    f.write(json.dumps({'name': name, 'argv': sys.argv[1:], 'env': {k: os.environ[k] for k in keys if k in os.environ}}) + '\n')
def tokenize(url, model):
    request = urllib.request.Request(url + '/tokenize', data=json.dumps({'model': model, 'prompt': 'fixture'}).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=10) as reply: reply.read()
if name == 'e3_q18a_lifecycle':
    def arg(key): return sys.argv[sys.argv.index(key) + 1]
    record = {'schema': 's5-native-lifetime/1', 'run_id': arg('--run-id'), 'engine_epoch': int(arg('--epoch')),
              'verdict': 'PASS', 'failures': [], 'missing': [], 'cancel_grade': {'verdict': 'PASS'}, 'reuse': {'complete': True}}
    with open(arg('--out'), 'x') as f: json.dump(record, f)
    print('NATIVE_LIFETIME PASS'); sys.exit(0)
if name == 'pressure2':
    url, model, n, pt, secs, out, stop = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6], sys.argv[7]
    tokenize(url, model)
    ready, t0, cap = os.environ['PRESSURE_READY'], time.time(), int(os.environ['PRESSURE_MAX_TOKENS'])
    first, last, counts, issued = [t0] * n, [t0] * n, [0] * n, [1] * n
    end = time.monotonic() + secs
    while time.monotonic() < end and not os.path.exists(stop):
        now = time.time(); counts = [c + 1 for c in counts]; issued = [c + 1 for c in issued]; last = [now] * n
        with open(ready + '.tmp', 'w') as f:
            json.dump({'first_tokens': first, 'last_tokens': last, 'issued': issued, 'completed_requests': counts,
                       'started_wall': t0, 'wall': now, 'max_tokens': cap}, f)
        os.replace(ready + '.tmp', ready); time.sleep(0.2)
    with open(out, 'w') as f:
        json.dump({'started_wall': t0, 'ended_wall': time.time(), 'first_tokens': first, 'last_tokens': last, 'issued': issued,
                   'streams': n, 'prompt': pt, 'max_tokens': cap, 'completed_requests': counts, 'preemptions': 0, 'errors': []}, f)
    sys.exit(0)
if name == 'leak_partial_tail':
    url, model, n, rounds, b, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    tokenize(url, model)
    results = [{'round': r, 'session': s, 'code': 'CODE-%d-%d' % (r, s), 'locker': 'LOCK-%d-%d' % (r, s), 'gen1': 'filler',
                'answer': 'CODE-%d-%d' % (r, s), 'cold': 'CODE-%d-%d' % (r, s), 'cached': b,
                'answer3': 'LOCK-%d-%d' % (r, s), 'cold3': 'LOCK-%d-%d' % (r, s), 'cached3': b + 176}
               for r in range(rounds) for s in range(n)]
    identity = (fake / 'current-id').read_text().strip() + ' 2026-11-01T00:00:00.000000000Z'
    run = {'verdict': 'PASS', 'counts': {'OK': 2 * n * rounds}, 'fails': [], 'inconclusive': [],
           'cancelled': int(n * rounds * float(os.environ['CANCEL_FRAC']))}
    with open(out, 'w') as f:
        json.dump({'MODE': os.environ['MODE'], 'B': b, 'N': n, 'ROUNDS': rounds, 'GEN': int(os.environ['GEN']),
                   'identity_start': identity, 'identity_end': identity, 'run': run, 'results': results, 'errors': []}, f)
    sys.exit(0)
sys.exit(2)
'''
# The box engine's core overlay (the trunk runner's EXPECTED_SOURCES rows; the lifecycle bundle's overlay is the
# trunk's byte for byte): the bundle's own native30_overlay when this suite runs from a bundle (the self-run), else
# the frozen trunk held-b overlay on the hub.
TRUNK_CORE_PINS = {'block_pool.py': '540bb7bec14b0dfc6106e3a02d1853fa842f951b70ccab79c77bb9e729f3fe39',
                   'scheduler.py': '33fcd566b649619776488bb59266d6620cf6b347fc750e020c3525a02c823dc9'}
TRUNK_HELD_OVERLAY = Path('<hub-dir>/agents/flash-next/tmp/native39-trunk-held-b/'
                          'q18a-native39-trunk-bundle-756f826c221c/native30_overlay')
TELEMETRY_METHODS = ('_flash_count', '_flash_pool_snapshot', 'flash_pool_step', 'get_cached_block',
                     '_maybe_evict_cached_block', '_remove_cached_block_hashes')


def trunk_core():
    base = next(b for b in (HERE / 'native30_overlay', TRUNK_HELD_OVERLAY) if (b / 'core/block_pool.py').is_file())
    texts = {}
    for name, pin in TRUNK_CORE_PINS.items():
        data = (base / 'core' / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == pin, name
        texts[name] = data.decode()
    return texts


def _bare(node):
    """A function node without annotations (they name vLLM types; evaluation order only, no behaviour)."""
    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    node.returns = None
    return node


class _Capture:
    """vLLM's init_logger(__name__) as the telemetry calls it: logger.info(fmt, *args)."""

    def __init__(self, sink, where):
        self.sink, self.where = sink, where

    def info(self, fmt, *args):
        self.sink.append('(EngineCore_DP0 pid=212) INFO 11-01 00:00:00 [%s] %s\n' % (self.where, fmt % args))


class _Block:
    """Fixture leaf lifecycle-pool-core: the KVCacheBlock fields the telemetry reads."""

    def __init__(self, block_id):
        self.block_id, self.ref_cnt, self.block_hash, self.is_null = block_id, 0, None, block_id == 0

    def reset_hash(self):
        self.block_hash = None


class _HashMap:
    """Fixture leaf lifecycle-pool-core: BlockHashToBlockMap (get_one_block, pop(key, block_id))."""

    def __init__(self):
        self.rows = {}

    def get_one_block(self, key):
        blocks = self.rows.get(key) or {}
        return next(iter(blocks.values()), None)

    def pop(self, key, block_id):
        return (self.rows.get(key) or {}).pop(block_id, None)


def telemetry_run(env):
    """The trunk engine's pool telemetry executed from its own bytes under ``env`` (the declared EXTRA_ENV as the
    container sees it): BlockPool.__init__'s FLASH_POOL_STATS statements and the six telemetry/bookkeeping methods
    from core/block_pool.py, the Scheduler.__init__ layout block from core/scheduler.py, and one
    ``flash_pool_step()`` per scheduler step as Scheduler.schedule calls it.  Returns the startup, before-pressure and
    after-pressure engine log lines (fixture leaf lifecycle-pool-core)."""
    import ast
    import types
    sys.path.insert(0, str(ROOT / 'patches/s5_engine'))
    import telemetry_hooks
    core = trunk_core()
    pool_src, sched_src = core['block_pool.py'], core['scheduler.py']
    # The overlay bytes are telemetry_hooks' output: its layout and step insertions appear verbatim.
    assert sched_src.count(telemetry_hooks.LAYOUT) == 1 and sched_src.count(telemetry_hooks.STEP) == 1
    tree = ast.parse(pool_src)
    pool_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BlockPool')
    methods = {f.name: f for f in pool_cls.body if isinstance(f, ast.FunctionDef)}
    init = methods['__init__'].body
    first = next(i for i, s in enumerate(init) if 'self.num_gpu_blocks =' in ast.get_source_segment(pool_src, s))
    flash_init = [s for s in init[:first] if '_flash_pool' in ast.get_source_segment(pool_src, s)]
    assert len(flash_init) == 4, [ast.get_source_segment(pool_src, s) for s in flash_init]
    init_fn = ast.FunctionDef(name='_flash_init', args=ast.arguments(posonlyargs=[], args=[ast.arg('self')], kwonlyargs=[],
                              kw_defaults=[], defaults=[]), body=flash_init, decorator_list=[], returns=None, type_params=[])
    sched_cls = next(n for n in ast.parse(sched_src).body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    sched_methods = {f.name: f for f in sched_cls.body if isinstance(f, ast.FunctionDef)}
    layout = [n for n in ast.walk(sched_methods['__init__']) if isinstance(n, ast.If)
              and ast.unparse(n.test) == 'self.kv_cache_manager.block_pool._flash_pool_stats_every']
    steps = [n for n in ast.walk(sched_methods['schedule']) if isinstance(n, ast.Expr)
             and ast.unparse(n) == 'self.kv_cache_manager.block_pool.flash_pool_step()']
    assert len(layout) == 1 and len(steps) == 1
    layout_fn = ast.FunctionDef(name='_layout', args=init_fn.args, body=layout, decorator_list=[], returns=None,
                                type_params=[])
    lines = []
    pool_ns = dict(_flash_os=types.SimpleNamespace(environ=dict(env)), _flash_json=json,
                   logger=_Capture(lines, 'block_pool.py'), get_group_id=lambda key: key[1],
                   make_block_hash_with_group_id=lambda block_hash, gid: (block_hash, gid))
    module = ast.Module(body=[_bare(init_fn)] + [_bare(methods[m]) for m in TELEMETRY_METHODS], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), 'core/block_pool.py', 'exec'), pool_ns)
    sched_ns = dict(logger=_Capture(lines, 'scheduler.py'))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[layout_fn], type_ignores=[])), 'core/scheduler.py', 'exec'),
         sched_ns)

    class Pool:
        def __init__(self, n):
            self.num_gpu_blocks, self.metrics_collector = n, None
            self.blocks = [_Block(i) for i in range(n)]
            self.cached_block_hash_to_block, self.cached_block_hashes_by_block = _HashMap(), {}
            self.removed = []
            self._flash_init()

        def get_num_free_blocks(self):
            return sum(1 for b in self.blocks if not b.is_null and b.ref_cnt == 0)

        def _emit_block_removed_events(self, hashes):
            self.removed.extend(hashes)

        def cache(self, block, block_hash, gids):
            keys = [(block_hash, gid) for gid in gids]
            block.block_hash = keys[0]
            self.cached_block_hashes_by_block.setdefault(block.block_id, set()).update(keys[1:])
            for key in keys:
                self.cached_block_hash_to_block.rows.setdefault(key, {})[block.block_id] = block

    for name in ('_flash_init',) + TELEMETRY_METHODS:
        setattr(Pool, name, pool_ns[name])
    groups = [('FullAttentionSpec', ['model.layers.3.self_attn.attn']), ('MambaSpec', ['model.layers.0.linear_attn']),
              ('MambaSpec', ['model.layers.1.linear_attn'])]
    pool = Pool(64)
    sched = types.SimpleNamespace(
        kv_cache_manager=types.SimpleNamespace(block_pool=pool),
        kv_cache_config=types.SimpleNamespace(num_blocks=64, kv_cache_groups=[
            types.SimpleNamespace(kv_cache_spec=type(kind, (), {'block_size': 176})(), layer_names=layers)
            for kind, layers in groups]))
    sched_ns['_layout'](sched)
    startup = list(lines)
    behaviour = dict(lookups=[], evicted=[])
    for i in range(1, 41):  # 40 free cached blocks, each holding one hash per group
        pool.cache(pool.blocks[i], 'h%d' % i, [0, 1, 2])
    for step in range(200):  # serving: prefix lookups, one flash_pool_step per scheduler step
        found = pool.get_cached_block('h%d' % (1 + step % 50), [0, 1, 2])
        behaviour['lookups'].append(None if found is None else [b.block_id for b in found])
        pool.flash_pool_step()
    before = list(lines)
    for step in range(400):  # pressure: cached blocks evicted for reuse
        if step % 10 == 0 and step // 10 + 1 <= 40:
            behaviour['evicted'].append(pool._maybe_evict_cached_block(pool.blocks[step // 10 + 1]))
        pool.flash_pool_step()
    behaviour.update(free=pool.get_num_free_blocks(), removed=list(pool.removed),
                     hashes={str(k): sorted(v) for k, v in pool.cached_block_hash_to_block.rows.items() if v},
                     block_hashes=[b.block_hash for b in pool.blocks])
    return startup, before, list(lines), behaviour


def lifetime_grade(before, after):
    """The real pp_release_grade lifetime CLI with the queue's argv shape over a bracketing pressure record."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / 'before.log').write_text(''.join(before))
        (d / 'after.log').write_text(''.join(after))
        (d / 'pressure.json').write_text(json.dumps({'streams': 2, 'first_tokens': [1.0, 1.0], 'last_tokens': [9.0, 9.0],
                                                     'completed_requests': [3, 3], 'errors': [], 'preemptions': 0}))
        (d / 'interval.json').write_text(json.dumps([2.0, 8.0]))
        done = subprocess.run([sys.executable, str(HERE / 'pp_release_grade.py'), 'lifetime', str(d / 'before.log'),
                               str(d / 'after.log'), str(d / 'pressure.json'), str(d / 'interval.json'), '0'],
                              capture_output=True, text=True)
        return done.returncode, json.loads(done.stdout)


SECRET = re.compile(r'Your private session code is (\S+)\.')
CRASH_LINE = '(Worker_PP1_TP0 pid=310) ERROR 11-01 00:00:00 torch.distributed.DistBackendError: NCCL error\n'
ARTS = ('gates.json', 'instruments-lifecycle.json', 'instruments-end.json', 'lifecycle.json', 'hygiene-full.json',
        'pressure.json', 'pressure.completion.json', 'pressure.interval.json', 'partial-leak.json', 'partial-leak.grade.json',
        'pressure.coverage.json', 'native35-lifetime-grade.json')


class LifecycleLeaf:
    """Loopback engine leaf (fixture leaf lifecycle-engine-leaf): /health, /metrics, /tokenize; the native session
    endpoint over the trunk's NativeSession (fixture leaf queue-native-session); and a non-streamed
    /v1/chat/completions that answers each leaktest session with its own private code
    (fixture leaf lifecycle-chat-answer; with leak=True one answer names another session's code)."""

    def __init__(self, session, leak=False):
        self.session, self.leak = session, leak
        self.lock = threading.Lock()
        self.auth, self.chats, self.tokenized, self.seen, self.leaked = set(), 0, 0, [], None
        leaf = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body=b'', kind='application/json'):
                self.send_response(code)
                self.send_header('Content-Type', kind)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == '/health':
                    self.reply(200)
                elif self.path == '/metrics':
                    self.reply(200, b'vllm:num_requests_running{engine="0",model_name="m"} 0.0\n', 'text/plain')
                else:
                    self.reply(404)

            def do_POST(self):
                with leaf.lock:
                    leaf.auth.add(self.headers.get('Authorization'))
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/tokenize':
                    with leaf.lock:
                        leaf.tokenized += 1
                    self.reply(200, json.dumps({'count': 8, 'max_model_len': 262144, 'tokens': [0] * 8}).encode())
                elif self.path == '/v1/flash_s5_native_overlap':
                    code, reply = leaf.session.act(data)
                    self.reply(code, json.dumps(reply).encode())
                elif self.path == '/v1/chat/completions':
                    self.reply(200, json.dumps(leaf.answer(data)).encode())
                else:
                    self.reply(404)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.server.request_queue_size = 64
        self.url = 'http://127.0.0.1:%d' % self.server.server_address[1]

    def answer(self, data):
        own = SECRET.search(next(m['content'] for m in data['messages'] if m['role'] == 'user')).group(1)
        with self.lock:
            self.chats += 1
            if own not in self.seen:
                self.seen.append(own)
            text = own
            if self.leak and self.leaked is None and len(self.seen) > 1:
                text = self.leaked = next(s for s in self.seen if s != own)
        return {'id': 'chatcmpl-%d' % self.chats, 'object': 'chat.completion',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 26000, 'completion_tokens': 4, 'prompt_tokens_details': {'cached_tokens': 20240}}}

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class Tail(unittest.TestCase):
    """From the lifecycle preamble to the end of the lifecycle region, then the ending's stop_pressure and the
    on_exit native35 lifetime grade lines: the queue's own bytes under bash with the real step/record/stop_pressure
    and health_ok, against the loopback leaf."""

    def run_tail(self, verdicts=QUALIFIED, crashed=0, mapped_rank=None, stuck=False, leak=False, crash_stream=False,
                 telemetry=True):
        # The engine as declared: the lifecycle settings (FLASH_POOL_STATS=200) or, for the negative, the trunk's
        # (no FLASH_POOL_STATS), through the real launcher render; identity-start captures exactly that identity when
        # it passes (fixture leaf lifecycle-identity-capture), and the engine's telemetry is the trunk overlay's own
        # code under that env (telemetry_run).
        declared, _ = trunkq.Launcher.declared(self, SETTINGS if telemetry else trunkq.SETTINGS)
        startup, before, after, _ = telemetry_run(declared['identity']['env'])
        draft = LIFECYCLE.read_text()
        drafts = drafts_tool()
        start, end = draft.index(drafts.LIFECYCLE_PREAMBLE), draft.index(drafts.ENDING)
        region = draft[start:end]
        self.assertEqual(draft.count(drafts.L_GRADE), 1)
        health = [line for line in draft.splitlines(True) if line.startswith('health_ok() {')]
        self.assertEqual(len(health), 1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake, h, q, bindir = d / 'fake', d / 'h', d / 'q', d / 'bin'
            for p in (fake, h, q, bindir, fake / 'maps', h / 'workloads'):
                p.mkdir()
            session = trunkq.NativeSession()
            session.stuck = stuck
            with LifecycleLeaf(session, leak=leak) as leaf:
                for name in ('e3_auth_exec.py', 'leaktest.py', 'pressure_wait.py', 'pp_release_grade.py',
                             'native35_lifetime_grade.py', 'e3_native39_trunk_checks.py'):
                    shutil.copyfile(HERE / name, h / name)  # the real tools, run from the harness dir as on the box
                for name in ('e3_q18a_lifecycle', 'pressure2', 'leak_partial_tail'):
                    (h / (name + '.py')).write_text(FAKE_TOOL)
                shutil.copyfile(MANIFEST, h / 'workloads' / MANIFEST.name)
                (fake / 'current-id').write_text(trunkq.NEW_ID)
                served = trunkq.BIND_LINE + ''.join(startup) + trunkq.SERVED_LINE
                (fake / 'engine.raw.0').write_text(served + ''.join(before[len(startup):]))
                (fake / 'engine.raw.1').write_text(served + ''.join(after[len(startup):]))
                (q / 'engine-stream.log').write_text(trunkq.BIND_LINE + ''.join(startup) + (CRASH_LINE if crash_stream else ''))
                (q / 'engine.identity-start.json').write_text(json.dumps({'identity': declared['identity']}))
                for rank in range(4):
                    text = trunkq.OBSERVER_MAPS if rank == mapped_rank else trunkq.CLEAN_MAPS
                    (fake / 'maps' / str(1001 + rank)).write_text(text)
                (bindir / 'docker').write_text('#!' + sys.executable + '\n' + FAKE_DOCKER)
                (bindir / 'docker').chmod(0o755)
                script = ('set -uo pipefail\nFAILS=0; INCS=0; ENGINE_CRASHED=%d; PRESSURE_PID=\'\'; PRESSURE_STOP=\'\'\n'
                          'declare -A PHASE_VERDICT=()\n' % crashed
                          + trunkq.functions(draft, ('step', 'record', 'stop_pressure')) + health[0]
                          + ''.join('PHASE_VERDICT[%s]=%s\n' % item for item in verdicts.items())
                          + region + 'stop_pressure || true\n' + drafts.L_GRADE
                          + 'for k in "${!PHASE_VERDICT[@]}"; do echo "PV $k=${PHASE_VERDICT[$k]}"; done\n'
                            'echo "TOTAL FAILS=$FAILS INCS=$INCS"\n')
                done = subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=400, cwd=str(d),
                                      env=trunkq.env(PATH=str(bindir) + os.pathsep + os.environ['PATH'], FAKE=str(fake),
                                                     H=str(h), Q=str(q), PROBE_URL=leaf.url, MODEL='m',
                                                     CONTAINER='qwen-flash-prod', VLLM_API_KEY=KEY,
                                                     RUN_ID='q18a-native39-lifecycle-cpu-1', EPOCH='7',
                                                     PYTHONPATH=os.pathsep.join((str(h), str(HERE),
                                                                                 str(ROOT / 'patches/s5_engine')))))
            rows = [json.loads(line) for line in (fake / 'calls.jsonl').read_text().splitlines()] \
                if (fake / 'calls.jsonl').exists() else []
            arts = {name: json.loads((q / name).read_text()) for name in ARTS if (q / name).exists()}
            leak_log = (q / 'cross-session-leak.log').read_text() if (q / 'cross-session-leak.log').exists() else None
            paths = dict(h=str(h), q=str(q), pool_telemetry=(q / 'pool-telemetry.log').read_text())
            return done, trunkq.phases(done.stdout), rows, arts, leaf, session, leak_log, paths

    def connected(self, done, pv, rows, arts, leaf, session, leak_log, paths):
        """The connected block's common shape: every phase before pressure-lifetime PASS, the exact tool calls."""
        passed = ('instruments-lifecycle', 'lifecycle', 'hygiene-full', 'cross-session-leak', 'pressure-ready',
                  'partial-leak-driver', 'partial-leak', 'pressure-completed', 'instruments-end')
        self.assertEqual({k: pv.get(k) for k in passed}, dict.fromkeys(passed, 'PASS'), done.stdout)
        for line in ('STEP lifetime-gates PASS', 'STEP pressure-driver PASS'):
            self.assertIn(line, done.stdout)
        self.assertEqual(arts['pressure.coverage.json']['fails'], [])
        # Gates read startup and row1 only; no rule4 continuation.
        gates = arts['gates.json']
        self.assertEqual((gates['outcomes'], gates['rule4_control_continuation']), (QUALIFIED, False))
        # Observer and offers off on the same four rank processes before and after the block.
        for name in ('instruments-lifecycle', 'instruments-end'):
            ranks = arts[name + '.json']['ranks']
            self.assertEqual({r['observer_mapped'] for r in ranks.values()}, {False}, name)
            self.assertEqual(sorted(r['pid'] for r in ranks.values()), [1001, 1002, 1003, 1004])
        self.assertTrue(arts['instruments-end.json']['same_processes'])
        # Session actions in order: instruments barrier, typed FULL hygiene, end barrier.
        self.assertEqual(session.calls, ['pause', 'resume', 'prepare', 'resume', 'pause', 'resume'])
        hygiene = arts['hygiene-full.json']
        self.assertEqual((hygiene['prepare']['route'], hygiene['prepare']['typed_acks'], hygiene['resume']['resumed']),
                         ('full', True, True))
        # The exact tool argv and environment the queue gives each phase executable.
        h, q = paths['h'], paths['q']
        self.assertEqual([r['name'] for r in rows], ['e3_q18a_lifecycle', 'docker-logs', 'pressure2', 'leak_partial_tail',
                                                     'docker-logs'])
        life = rows[0]['argv']
        self.assertEqual(life, ['--url', leaf.url, '--model', 'm', '--run-id', 'q18a-native39-lifecycle-cpu-1', '--epoch', '7',
                                '--workload', h + '/workloads/q18a-w2-overlap-native39.json', '--out', q + '/lifecycle.json'])
        self.assertEqual(rows[2]['argv'], [leaf.url, 'm', '2', '200000', '1800', q + '/pressure.json', q + '/pressure.stop'])
        self.assertEqual(rows[2]['env'], {'PRESSURE_READY': q + '/pressure.ready.json', 'PRESSURE_MAX_TOKENS': '128'})
        self.assertEqual(rows[3]['argv'], [leaf.url, 'm', '10', '4', '7568', q + '/partial-leak.json'])
        self.assertEqual(rows[3]['env'], {'MODE': 'inside', 'GEN': '200', 'TIMEOUT': '900', 'CANCEL_FRAC': '0.3',
                                          'NEED_PARTIAL': '1', 'UNIQUE_PREFIX': '1'})
        # Real leaktest: 8 sessions x 3 turns, every answer its own code, bearer from the auth hook on every call.
        self.assertEqual(leaf.chats, 24)
        self.assertRegex(leak_log, r'(?m)^sessions 8 rounds 3 bad 0 in ')
        self.assertEqual(leaf.auth, {'Bearer ' + KEY})
        self.assertEqual(leaf.tokenized, 2)
        # Real partial-leak grade over the driver record; real pressure wait and bracket.
        self.assertEqual(arts['partial-leak.grade.json']['verdict'], 'PASS')
        self.assertEqual((arts['pressure.completion.json']['rc'], arts['pressure.completion.json']['satisfied']), (0, True))
        interval = arts['pressure.interval.json']
        self.assertLess(interval[0], interval[1])
        self.assertLessEqual(max(arts['pressure.json']['first_tokens']), interval[0])
        self.assertGreaterEqual(min(arts['pressure.json']['last_tokens']), interval[1])

    def test_connected_lifecycle_block_with_the_declared_pool_telemetry_passes_pressure_lifetime(self):
        out = self.run_tail()
        done, pv, rows, arts, leaf, session, leak_log, paths = out
        self.assertEqual(done.returncode, 0, done.stderr)
        self.connected(*out)
        # The declared FLASH_POOL_STATS=200, as identity-start captures it, and the startup layout line the trunk
        # overlay's own code logs under it: the pool-telemetry record passes.
        self.assertEqual(pv['pool-telemetry'], 'PASS')
        self.assertEqual(json.loads(paths['pool_telemetry'].splitlines()[-1]),
                         {'flash_pool_stats': '200', 'layout_lines': 1, 'malformed': 0,
                          'kinds': ['FullAttentionSpec', 'MambaSpec']})
        # The unchanged pressure-lifetime oracle reads the state-group layout and the evictions between the two logs.
        coverage = arts['pressure.coverage.json']
        self.assertEqual((pv['pressure-lifetime'], coverage['verdict'], coverage['missing']), ('PASS', 'PASS', []))
        self.assertEqual((coverage['state_evictions'], coverage['evictions']), (80, 120))
        grade = arts['native35-lifetime-grade.json']
        self.assertEqual((grade['verdict'], grade['failures'], grade['missing']), ('PASS', [], []))
        self.assertEqual(pv['native35-lifetime'], 'PASS')
        self.assertIn('TOTAL FAILS=0 INCS=0', done.stdout)
        record_case('tail-lifecycle-connected-pool-telemetry-on', calls=[r['name'] for r in rows], session_calls=session.calls,
                    leak_requests=leaf.chats, pool_telemetry='PASS', pressure_lifetime='PASS',
                    state_evictions=coverage['state_evictions'], lifetime_grade='PASS', lifetime_missing=grade['missing'])

    def test_connected_lifecycle_block_without_pool_telemetry_stays_inconclusive(self):
        # Negative: the trunk engine as declared (no FLASH_POOL_STATS).  Its block pool logs no FLASH_PP_LAYOUT /
        # FLASH_POOL_STATS line, so the unchanged oracle cannot see a state-group eviction and the grade cannot pass;
        # the pool-telemetry record names the engine env.
        out = self.run_tail(telemetry=False)
        done, pv, rows, arts, leaf, session, leak_log, paths = out
        self.assertEqual(done.returncode, 0, done.stderr)
        self.connected(*out)
        self.assertEqual(pv['pool-telemetry'], 'FAIL')
        self.assertIn('engine env FLASH_POOL_STATS None, declared 200', paths['pool_telemetry'])
        self.assertEqual((pv['pressure-lifetime'], pv['native35-lifetime']), ('INCONCLUSIVE', 'INCONCLUSIVE'))
        self.assertEqual(arts['pressure.coverage.json']['missing'],
                         ['no state-group layout evidence', 'no state-group eviction observed during lifetime interval'])
        grade = arts['native35-lifetime-grade.json']
        self.assertEqual((grade['verdict'], grade['failures'], grade['missing']),
                         ('INCONCLUSIVE', [], ['pressure-lifetime:INCONCLUSIVE']))
        self.assertIn('TOTAL FAILS=1 INCS=2', done.stdout)
        record_case('tail-lifecycle-connected-pool-telemetry-absent', calls=[r['name'] for r in rows],
                    session_calls=session.calls, leak_requests=leaf.chats, pool_telemetry='FAIL',
                    pressure_lifetime='INCONCLUSIVE', lifetime_grade='INCONCLUSIVE', lifetime_missing=grade['missing'])

    def test_row1_not_pass_withholds_instruments_and_the_block(self):
        done, pv, rows, arts, leaf, session, leak_log, paths = self.run_tail(dict(QUALIFIED, row1='INCONCLUSIVE'))
        self.assertIn('STEP lifetime-gates INCONCLUSIVE', done.stdout)
        self.assertEqual(pv['lifecycle'], 'INCONCLUSIVE')
        self.assertNotIn('instruments-lifecycle', pv)
        self.assertNotIn('instruments-end', pv)
        self.assertEqual((rows, session.calls, leaf.chats, leaf.tokenized), ([], [], 0, 0))
        grade = arts['native35-lifetime-grade.json']
        self.assertEqual((grade['verdict'], grade['failures']), ('INCONCLUSIVE', []))
        self.assertIn('native-lifetime:record-missing', grade['missing'])
        record_case('tail-row1-not-pass-withholds', calls=[], session_calls=[], lifetime_grade='INCONCLUSIVE')

    def test_engine_crash_withholds_the_block_and_the_crash_stream_fails_the_grade(self):
        done, pv, rows, arts, leaf, session, leak_log, paths = self.run_tail(crashed=1, crash_stream=True)
        self.assertIn('STEP lifetime-gates PASS', done.stdout)
        self.assertEqual(pv['lifecycle'], 'INCONCLUSIVE')
        self.assertNotIn('instruments-lifecycle', pv)
        self.assertEqual((rows, session.calls, leaf.chats), ([], [], 0))
        grade = arts['native35-lifetime-grade.json']
        self.assertEqual((grade['verdict'], grade['failures']), ('FAIL', ['engine-crash:engine-stream.log']))
        self.assertEqual(pv['native35-lifetime'], 'FAIL')
        record_case('tail-engine-crash-withholds', calls=[], lifetime_grade='FAIL')

    def test_observer_mapped_on_one_rank_withholds_the_block(self):
        done, pv, rows, arts, leaf, session, leak_log, paths = self.run_tail(mapped_rank=1)
        self.assertEqual(pv['instruments-lifecycle'], 'FAIL')
        self.assertTrue(arts['instruments-lifecycle.json']['ranks']['1']['observer_mapped'])
        self.assertEqual(pv['lifecycle'], 'INCONCLUSIVE')
        self.assertIn('observer and offers off not proven (instruments-lifecycle)', done.stdout)
        self.assertNotIn('instruments-end', pv)
        self.assertEqual((rows, session.calls, leaf.chats), ([], ['pause', 'resume'], 0))
        record_case('tail-observer-mapped-withholds', calls=[], instruments='FAIL', session_calls=session.calls)

    def test_refused_full_hygiene_stops_before_leak_and_pressure(self):
        done, pv, rows, arts, leaf, session, leak_log, paths = self.run_tail(stuck=True)
        self.assertEqual((pv['instruments-lifecycle'], pv['lifecycle'], pv['hygiene-full']), ('PASS', 'PASS', 'FAIL'))
        self.assertEqual([r['name'] for r in rows], ['e3_q18a_lifecycle'])
        self.assertEqual((leaf.chats, leaf.tokenized, leak_log), (0, 0, None))
        self.assertNotIn('STEP pressure-driver', done.stdout)
        for name in ('cross-session-leak', 'pressure-ready', 'partial-leak', 'pressure-lifetime'):
            self.assertNotIn(name, pv)
        self.assertEqual(pv['instruments-end'], 'PASS')
        grade = arts['native35-lifetime-grade.json']
        self.assertEqual(grade['verdict'], 'INCONCLUSIVE')
        self.assertLessEqual({'hygiene-full:typed-full-route', 'cross-session-leak:log-missing'}, set(grade['missing']))
        record_case('tail-hygiene-refused-stops-leak-and-pressure', calls=[r['name'] for r in rows], hygiene='FAIL')

    def test_a_real_cross_session_leak_fails_the_leak_and_the_grade(self):
        done, pv, rows, arts, leaf, session, leak_log, paths = self.run_tail(leak=True)
        self.assertIsNotNone(leaf.leaked)
        self.assertRegex(leak_log, r'(?m)^s\d+ r\d+ LEAK ')
        self.assertNotRegex(leak_log, r'(?m)^sessions 8 rounds 3 bad 0 in ')
        self.assertEqual(pv['cross-session-leak'], 'FAIL')
        # The block continues to the pressure phases as written; the grade fails on the leaked session.
        self.assertEqual([r['name'] for r in rows], ['e3_q18a_lifecycle', 'docker-logs', 'pressure2', 'leak_partial_tail',
                                                     'docker-logs'])
        grade = arts['native35-lifetime-grade.json']
        self.assertEqual((grade['verdict'], grade['failures']), ('FAIL', ['cross-session-leak:bad-session']))
        self.assertEqual(pv['native35-lifetime'], 'FAIL')
        record_case('tail-real-leak-detected-fails', leak='FAIL', lifetime_grade='FAIL')


class PoolTelemetry(unittest.TestCase):
    """The trunk engine's own pool telemetry bytes under the env each settings file declares, graded by the unchanged
    pressure-lifetime oracle (pp_release_grade lifetime, the queue's argv shape)."""

    def test_the_oracle_stays_inconclusive_without_the_env_and_passes_with_it_and_the_pool_behaves_the_same(self):
        lifecycle, _ = trunkq.Launcher.declared(self, SETTINGS)
        trunk, _ = trunkq.Launcher.declared(self, trunkq.SETTINGS)
        on = telemetry_run(lifecycle['identity']['env'])
        off = telemetry_run(trunk['identity']['env'])
        # Without the env (the trunk engine): the block pool logs nothing and the oracle stays INCONCLUSIVE.
        self.assertEqual(off[:3], ([], [], []))
        rc, coverage = lifetime_grade(off[1], off[2])
        self.assertEqual((rc, coverage['verdict'], coverage['missing']),
                         (2, 'INCONCLUSIVE', ['no state-group layout evidence',
                                              'no state-group eviction observed during lifetime interval']))
        # With FLASH_POOL_STATS=200: one startup layout line, one counter line every 200 scheduler steps.
        startup, before, after, _ = on
        kinds = [line.split('] ', 1)[1].split(' ', 1)[0] for line in after]
        self.assertEqual((len(startup), len(before), kinds),
                         (1, 2, ['FLASH_PP_LAYOUT', 'FLASH_POOL_STATS', 'FLASH_POOL_STATS', 'FLASH_POOL_STATS']))
        steps = [json.loads(line.split('FLASH_POOL_STATS ', 1)[1])['step'] for line in after[1:]]
        self.assertEqual(steps, [200, 400, 600])
        rc, coverage = lifetime_grade(before, after)
        self.assertEqual((rc, coverage['verdict'], coverage['missing'], coverage['state_evictions'], coverage['evictions']),
                         (0, 'PASS', [], 80, 120))
        # Read-only: the same lookups, evictions, removal events, free count and hash maps with and without it.
        self.assertEqual(on[3], off[3])
        self.assertEqual(len(on[3]['evicted']), 40)
        record_case('pool-telemetry-trunk-code-off-inc-on-pass', off_verdict='INCONCLUSIVE', on_verdict='PASS',
                    state_evictions=coverage['state_evictions'], behaviour_identical=True,
                    core_pins=dict(TRUNK_CORE_PINS))


class Workload(unittest.TestCase):
    """The lifecycle's W2 manifest through the real lifecycle CLI and the w2-length guard's prompt."""

    def test_the_lifecycle_cli_accepts_the_trunk_manifest_and_its_fresh_prompt_is_the_guarded_one(self):
        import e3_overlap_performance as shared
        import e3_native39_w2_length as length
        data = json.loads(MANIFEST.read_text())
        problems, cfg = shared.validate_manifest(data)  # e3_q18a_lifecycle.main passes the whole manifest
        self.assertEqual(problems, [])
        guard_cfg = shared.validate_manifest({k: v for k, v in data.items() if k != 'length_declaration'})[1]
        self.assertEqual(shared._prefill_prompts(cfg)['fresh'], shared._prefill_prompts(guard_cfg)['fresh'])
        fresh = length.band(length.model_tokens(shared._prefill_prompts(cfg)['fresh']))
        self.assertEqual((fresh['tokens'], fresh['plans']), (197270, True))
        self.assertEqual(data['length_declaration']['fresh']['tokens'], 197270)
        # The caps the lifecycle derivation reads (native-lifetime row).
        self.assertEqual((cfg['timeouts']['read_s'], cfg['timeouts']['drain_s']), (180.0, 120.0))
        self.assertIn('180 + drain_s 120 + reuse first advance read_s 180', dict(
            (n, why) for n, _, why in gate.LIFECYCLE_DERIVATION)['native-lifetime'])
        # The real CLI, with the queue's own argv lines, against a refused loopback port
        # (fixture leaf lifecycle-refused-port): the manifest validates and the first control call is attempted (the
        # refusal is the transport, not the inputs).
        draft = LIFECYCLE.read_text()
        argv = trunkq.segment(draft, '  python3 "$H/e3_q18a_lifecycle.py"', '--out "$Q/lifecycle.json" > "$Q/lifecycle.log" 2>&1\n')
        with tempfile.TemporaryDirectory() as d:
            import socket
            s = socket.socket()
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
            s.close()
            q = Path(d)
            done = subprocess.run(['bash', '-c', argv + 'echo "RC=$?"'], capture_output=True, text=True, timeout=120,
                                  env=trunkq.env(H=str(HERE), Q=str(q), PROBE_URL='http://127.0.0.1:%d' % port, MODEL='m',
                                                 RUN_ID='q18a-native39-lifecycle-cpu-2', EPOCH='7', VLLM_API_KEY=KEY,
                                                 PYTHONPATH=os.pathsep.join((str(HERE), str(ROOT / 'patches/s5_engine')))))
            self.assertIn('RC=2', done.stdout, done.stderr)
            record = json.loads((q / 'lifecycle.json').read_text())
        self.assertEqual(record['verdict'], 'INCONCLUSIVE')
        self.assertNotIn('lifetime-inputs', record['missing'])
        self.assertTrue(any(m.startswith('native-lifetime-refused:') for m in record['missing']), record['missing'])
        record_case('workload-lifecycle-cli-native39-manifest', fresh_tokens=fresh['tokens'], first_call_reached=True)


class Settings(unittest.TestCase):
    """Sourced lifecycle settings -> real setcfg -> real S5 launcher render -> declared identity -> PY_DECL, over the
    trunk's fixture leaf queue-declare-fixture-tree."""

    def py_decl(self, path):
        code = trunkq.segment(DECLARE.read_text(), "python3 - \"$R/startup.json\" \"$Q18A_TARGET_LABEL\" <<'PY_DECL'",
                              '\nPY_DECL\n')
        body = code.split('\n', 1)[1].rsplit('\nPY_DECL\n', 1)[0]
        return subprocess.run([sys.executable, '-I', '-c', body, str(path), 's5-overlap-80-80'], capture_output=True, text=True)

    def test_lifecycle_settings_declare_the_trunk_engine_plus_pool_telemetry(self):
        declared, path = trunkq.Launcher.declared(self, SETTINGS)
        env = declared['identity']['env']
        self.assertEqual({k: env.get(k) for k in ('FLASH_S5_REPLAY_OBSERVER', 'FLASH_S5_ENGINE_OFFERS', 'FLASH_NATIVE30_LOCALIZE',
                                                  'FLASH_POOL_STATS')},
                         {'FLASH_S5_REPLAY_OBSERVER': '0', 'FLASH_S5_ENGINE_OFFERS': '0', 'FLASH_NATIVE30_LOCALIZE': '0',
                          'FLASH_POOL_STATS': '200'})
        args = declared['identity']['command']['args']
        hosts = [args[j + 1] for j, x in enumerate(args[:-1]) if x == '--host']
        self.assertEqual(hosts, ['0.0.0.0', '127.0.0.1'])
        done = self.py_decl(path)
        self.assertEqual(done.returncode, 0, done.stderr)
        # Against the trunk engine: the declared env differs by FLASH_POOL_STATS=200 only; the args are identical.
        trunk, trunk_path = trunkq.Launcher.declared(self, trunkq.SETTINGS)
        tenv = trunk['identity']['env']
        delta = {k: (tenv.get(k), env.get(k)) for k in set(tenv) | set(env) if tenv.get(k) != env.get(k)}
        self.assertEqual(delta, {'FLASH_POOL_STATS': (None, '200')})
        self.assertEqual(trunk['identity']['command']['args'], args)
        # The lifecycle declaration refuses the trunk engine (no pool telemetry) and P-c's (observer and offers on).
        refused = self.py_decl(trunk_path)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('pool telemetry (FLASH_POOL_STATS=200) must be on', refused.stderr)
        _, pc_path = trunkq.Launcher.declared(self, RUNS / 'q18a_native35_p_c_settings.sh')
        refused = self.py_decl(pc_path)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn('observer must be off', refused.stderr)
        # The preflight's own env assertion lines (lifecycle draft, PY_PREFLIGHT) over both declared envs.
        draft, drafts = LIFECYCLE.read_text(), drafts_tool()
        body = trunkq.segment(draft, "<<'PY_PREFLIGHT' > \"$Q/preflight.log\" 2>&1\n", '\nPY_PREFLIGHT\n')
        first = "assert env.get('FLASH_S5_REPLAY_OBSERVER') == '0'\n"
        self.assertEqual((body.count(first), body.count(drafts.LIFECYCLE_PREFLIGHT_OFFERS)), (1, 1))
        lines = body[body.index(first):body.index(drafts.LIFECYCLE_PREFLIGHT_OFFERS) + len(drafts.LIFECYCLE_PREFLIGHT_OFFERS)]
        self.assertTrue(lines.endswith("assert env.get('FLASH_POOL_STATS') == '200'\n"))
        exec(lines, {'env': env})
        exec(lines[:lines.rindex('assert ')], {'env': tenv})
        with self.assertRaises(AssertionError):
            exec(lines, {'env': tenv})
        record_case('settings-trunk-engine-plus-pool-telemetry', hosts=hosts, env_delta_vs_trunk={'FLASH_POOL_STATS': '200'},
                    trunk_refused=True, p_c_refused=True, preflight_env_lines=True)


class Derivation(unittest.TestCase):
    def test_lifecycle_texts_are_the_declared_derivation_of_the_parent_texts(self):
        for name, digest in PARENT_SHA.items():
            self.assertEqual(hashlib.sha256((RUNS / name).read_bytes()).hexdigest(), digest, name)
        for path, text in drafts_tool().generate().items():
            self.assertEqual(path.read_text(), text, path.name)

    def test_the_lifecycle_draft_is_the_trunk_prefix_the_l_block_and_the_trunk_ending(self):
        # Independent of the generator's order: undo each declared change and compare to the parents.
        drafts = drafts_tool()
        trunk, life = (RUNS / 'q18a_native39_trunk.draft.sh').read_text(), LIFECYCLE.read_text()
        l_text = (RUNS / 'q18a_native35_l.draft.sh').read_text()
        prefix, rest = life.split(drafts.LIFECYCLE_PREAMBLE)
        region, ending = rest[:rest.index(drafts.ENDING)], rest[rest.index(drafts.ENDING):]
        self.assertEqual(ending, drafts.ENDING)
        self.assertTrue(trunk.endswith(drafts.ENDING))
        undo = prefix.replace('q18a-native39-lifecycle', 'q18a-native39-trunk').replace('q18a_native39_lifecycle',
                                                                                        'q18a_native39_trunk')
        for old, new in ((drafts.TRUNK_TIMEOUT, drafts.LIFECYCLE_TIMEOUT), (drafts.TRUNK_SLOT_NOTE, drafts.LIFECYCLE_SLOT_NOTE),
                         (drafts.TRUNK_SLOT, drafts.LIFECYCLE_SLOT), (drafts.TRUNK_INTERFACES, drafts.LIFECYCLE_INTERFACES),
                         (drafts.TRUNK_DIAGNOSTIC, drafts.L_GRADE), (drafts.TRUNK_TITLE, drafts.LIFECYCLE_TITLE),
                         (drafts.TRUNK_PREFLIGHT_OFFERS, drafts.LIFECYCLE_PREFLIGHT_OFFERS)):
            self.assertEqual(undo.count(new), 1, new[:60])
            undo = undo.replace(new, old)
        self.assertEqual(undo.count(drafts.LIFECYCLE_HEADER), 1)
        trunk_prefix = trunk[:trunk.index(drafts.TRUNK_CUT_BEGIN)]
        header = [line for line in trunk_prefix.splitlines(True) if line.startswith(drafts.TRUNK_HEADER_PREFIX)]
        self.assertEqual(undo.replace(drafts.LIFECYCLE_HEADER, header[0] + drafts.TRUNK_HEADER_NEXT), trunk_prefix)
        # The region: the pool-telemetry record, L's gates and block with the five declared changes, then the end
        # instruments receipt.
        block = l_text[l_text.index(drafts.L_BLOCK_BEGIN):l_text.index(drafts.ENDING)]
        self.assertTrue(region.startswith(drafts.POOL_TELEMETRY) and region.endswith(drafts.INSTRUMENTS_AFTER))
        undo = region[len(drafts.POOL_TELEMETRY):-len(drafts.INSTRUMENTS_AFTER)]
        for old, new in ((drafts.L_GATES_STEP, drafts.L_GATES_STEP + drafts.INSTRUMENTS_BEFORE),
                         (drafts.L_NOT_RUN, drafts.LIFECYCLE_NOT_RUN), (drafts.L_CONDITION, drafts.LIFECYCLE_CONDITION),
                         (drafts.L_WORKLOAD, drafts.LIFECYCLE_WORKLOAD), (drafts.L_GATES_LOOP, drafts.LIFECYCLE_GATES_LOOP)):
            self.assertEqual(undo.count(new), 1, new[:60])
            undo = undo.replace(new, old)
        self.assertEqual(undo, block)
        # The instruments lines are the trunk's own, but for the receipt names.
        for line in ('  python3 "$H/e3_native39_trunk_checks.py" instruments --url "$PROBE_URL" --run-id "$RUN_ID" --epoch "$EPOCH" \\\n',
                     "  _cid=$(docker inspect --format '{{.Id}}' \"$CONTAINER\" 2>/dev/null) || _cid=''\n"):
            self.assertIn(line, trunk)
            self.assertEqual(life.count(line), 2 if 'instruments --url' in line else 1)
        # Settings and declaration: the trunk's texts with the declared swaps (comment, EXTRA_ENV, PY_DECL, pins).
        for suffix, swaps in (('_settings.sh', ((drafts.TRUNK_SETTINGS_NOTE, drafts.LIFECYCLE_SETTINGS_NOTE),
                                                (drafts.TRUNK_EXTRA_ENV, drafts.LIFECYCLE_EXTRA_ENV))),
                              ('_declare.sh', ((drafts.TRUNK_DECL_NATIVE, drafts.LIFECYCLE_DECL_NATIVE),
                                               (drafts.TRUNK_DECL_OFFERS, drafts.LIFECYCLE_DECL_OFFERS)))):
            undo = (RUNS / ('q18a_native39_lifecycle' + suffix)).read_text()
            undo = undo.replace('q18a-native39-lifecycle', 'q18a-native39-trunk').replace('q18a_native39_lifecycle',
                                                                                        'q18a_native39_trunk')
            for old, new in swaps:
                self.assertEqual(undo.count(new), 1, new[:60])
                undo = undo.replace(new, old)
            self.assertEqual(undo, (RUNS / ('q18a_native39_trunk' + suffix)).read_text(), suffix)

    def test_phase_names_are_the_trunk_startup_row1_and_the_l_block(self):
        def names(text):
            return set(re.findall(r'(?:^|[;\s])(?:record|step) "?([A-Za-z0-9_.$\-{}]+)"? ', text, re.M))
        trunk, life = names((RUNS / 'q18a_native39_trunk.draft.sh').read_text()), names(LIFECYCLE.read_text())
        self.assertEqual(life - trunk, {'pool-telemetry', 'lifetime-gates', 'lifecycle', 'hygiene-full', 'cross-session-leak', 'pressure-ready',
                                        'partial-leak-driver', 'partial-leak', 'pressure-completed', 'pressure-lifetime',
                                        'instruments-lifecycle', 'native35-lifetime'})
        self.assertEqual(trunk - life, {'native35-diagnostic', 'overlap-gates', 'w1_controls', 'w1_plain', 'w1-plain-idle',
                                        'w2-idle-precondition', 'trunk-summary', 'instruments-w1', 'ingress-w1',
                                        'w1-plain-full-idle', 'w1-plain-route', 'ingress-plain', 'w2-ingress-precondition',
                                        'w2-idle-round-$_round', 'instruments-round-$_round', 'ingress-round-$_round',
                                        'overlap-round-$_round', 'memory-round-$_round', 'overlap', 'native35-performance',
                                        'returns', 'row4-capacity-continuation', 'overlap-engine', 'native30-receipt-tail'})

    def test_every_leaf_this_suite_names_is_declared_with_its_test(self):
        declared = {l['name']: l for l in json.loads((HERE / 'native39_lifecycle_fixture_leaves.json').read_text())['leaves']}
        named = set(re.findall(r'[Ff]ixture leaf ((?:lifecycle|queue)-[a-z0-9-]+)', Path(__file__).read_text()))
        self.assertEqual(named, set(declared))
        for name, leaf in declared.items():
            self.assertTrue(leaf['replaces'] and leaf['equivalence'] and leaf['cannot_prove'], name)
            self.assertIs(leaf['box_evidence'], False)
            tests = re.findall(r'test_native39_lifecycle_queue\.(\w+)', leaf['test'])
            self.assertTrue(tests, name)
            for test in tests:
                self.assertIn(test, globals(), name)

    def test_scripts_parse_and_name_only_the_lifecycle(self):
        for path in (LIFECYCLE, SETTINGS, DECLARE):
            done = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            text = path.read_text()
            self.assertNotIn('q18a-native39-trunk', text)
            self.assertNotIn('q18a_native39_trunk', text)
            self.assertNotIn('workloads/q18a-w2-overlap.json', text)
        text = LIFECYCLE.read_text()
        self.assertIn('--kill-after=900s 9600s env', text)
        for gone in ('qualification_phase rule4', 'ROW4_DECLARED_NOT_RUN', 'e3_native39_w1.py "$PROBE_URL"',
                     'e3_native39_round.py" --url', 'w1_trace.py" "$PROBE_URL"', 'overlap-round-'):
            self.assertNotIn(gone, text)


class Window(unittest.TestCase):
    def test_bounds_unchanged_and_the_derived_lifecycle_budget_fits(self):
        self.assertEqual((gate.NOT_BEFORE, gate.HARD_END), (native35p_gate.NOT_BEFORE, native35p_gate.HARD_END))
        self.assertEqual((gate.NOT_BEFORE, gate.HARD_END), (native39_trunk_gate.NOT_BEFORE, native39_trunk_gate.HARD_END))
        self.assertEqual(gate.HARD_END, datetime.datetime(2026, 11, 2, 6, tzinfo=datetime.timezone.utc).timestamp())
        self.assertEqual((gate.LIFECYCLE_ACTIVE_S, gate.LIFECYCLE_CLEANUP_MARGIN_S, gate.LIFECYCLE_ACTIVE_AND_CLEANUP),
                         (9600, 900, 10500))
        # Startup and row1 rows are P's and the trunk's, by name and value.
        p_rows = dict((n, s) for n, s, _ in native35p_gate.ACTIVE_DERIVATION)
        t_rows = dict((n, s) for n, s, _ in native39_trunk_gate.TRUNK_DERIVATION)
        for name, seconds, _ in gate.STARTUP_DERIVATION:
            self.assertEqual(seconds, p_rows.get(name, t_rows.get(name)), name)
        self.assertEqual([n for n, _, _ in gate.STARTUP_DERIVATION],
                         ['restart-to-health', 'speculation-and-sampler', 'identity-start', 'startup-ledger-headroom-plan',
                          'bind-receipt', 'w2-length-guard', 'row1'])
        self.assertEqual(sum(s for _, s, _ in gate.ACTIVE_DERIVATION), 9547)
        self.assertEqual(sum(s for _, s, _ in gate.CLEANUP_DERIVATION), 866)
        self.assertEqual(gate.CLEANUP_DERIVATION[:len(native35p_gate.CLEANUP_DERIVATION)], native35p_gate.CLEANUP_DERIVATION)
        self.assertFalse(gate.lifecycle_window_ok(gate.NOT_BEFORE - 1))
        self.assertTrue(gate.lifecycle_window_ok(gate.HARD_END - 10500))
        self.assertFalse(gate.lifecycle_window_ok(gate.HARD_END - 10500 + 1))
        self.assertEqual(gate.LATEST_ENTRY, gate.HARD_END - 10500)
        self.assertEqual(gate.LATEST_ENTRY_UTC, '2026-11-02T03:05:00Z')

    def test_lifecycle_draft_window_slot_executes_at_real_wall_clock(self):
        source = LIFECYCLE.read_text()
        actual = trunkq.segment(source, "python3 - <<'PY_SLOT'", '\nPY_SLOT\n')
        self.assertIn('from native39_lifecycle_gate import lifecycle_window_ok as window_ok', actual)
        done = subprocess.run(['bash', '-c', 'fail_early() { return 2; }\n' + actual], env=trunkq.env(HARNESS_DIR=str(HERE)),
                              capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stderr)
        SLOT_RECEIPTS.append(dict(draft=LIFECYCLE.name, verdict='PASS', rc=0,
                                  slot_sha256=hashlib.sha256(actual.encode()).hexdigest(), real_wall_clock=True,
                                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), engine_contact=False,
                                  active_s=gate.LIFECYCLE_ACTIVE_S, cleanup_margin_s=gate.LIFECYCLE_CLEANUP_MARGIN_S,
                                  latest_entry_utc=gate.LATEST_ENTRY_UTC))


if __name__ == '__main__':
    unittest.main()
