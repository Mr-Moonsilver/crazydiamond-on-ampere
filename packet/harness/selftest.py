#!/usr/bin/env python3
"""Offline self-test of the harness verdict logic: every known false-pass case must fail. No inference, no SSH.

Covers canary_grade.py (review fixture: foreign text + correct final line + rc=124; missing output; rc=1; wrong note;
malformed output; duplicate ids), leakgrade.py (foreign text in a retry; retries that did not recreate the original
cache path -- the J-inside case; recreated path all right / all wrong; missing results; engine restart) and
stall_trace2.py --recompute (good run; missing newcomer token; stopped stream; incomplete run).
usage: selftest.py [REVIEW_FIXTURE_DIR]   exit 0 only if every case behaves as required
"""
import atexit, hashlib, json, os, re, shutil, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_bound  # noqa: E402  declared clock-bound exclusions

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import canary_grade, leakgrade  # noqa: E402

results = []


def check(name, got, want):
    ok = got == want
    results.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")


unittest_observations = []


def audit_suite(name, process):
    """A skipped unittest is missing coverage, not a green child exit."""
    output = (process.stdout or '') + (process.stderr or '')
    summaries = re.findall(r'^(?:OK|FAILED)\s*\(([^\n]*)\)\s*$', output, re.MULTILINE)
    skipped = sum(int(count) for summary in summaries
                  for count in re.findall(r'\bskipped=(\d+)\b', summary))
    ran = re.findall(r'^Ran (\d+) tests? in ', output, re.MULTILINE)
    row = dict(suite=name, returncode=process.returncode, skipped=skipped,
               tests_run=int(ran[-1]) if ran else None,
               stdout_sha256=hashlib.sha256((process.stdout or '').encode()).hexdigest(),
               stderr_sha256=hashlib.sha256((process.stderr or '').encode()).hexdigest())
    unittest_observations.append(row)
    print('selftest suite: ' + json.dumps(row, sort_keys=True))
    if skipped:
        print('selftest missing coverage: ' + name + ' skipped=' + str(skipped))
    return process.returncode or (1 if skipped else 0)


def canary_dir(sessions):
    d = tempfile.mkdtemp(prefix='canary-fixture-')
    atexit.register(shutil.rmtree, d, True)
    with open(os.path.join(d, 'expected.txt'), 'w') as f:
        for i, (code, note, out) in enumerate(sessions, 1):
            f.write(f'1 {i} {code} {note} /synthetic/{i}\n')
            if out is not None:
                open(os.path.join(d, f'r1-s{i}.out'), 'w').write(out)
    return d


# --- canary_grade ---
review = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    '~/agents/cde/investigations/flash-current-review-20261024T0226Z/offline-checks/canary-fixture')
if os.path.isdir(review):
    check('canary: review fixture (foreign + rc=124)', canary_grade.grade(review)['verdict'], 'FAIL')
good = ('AMBER-1111', 'note-aaaa', 'CODE=AMBER-1111 NOTE=note-aaaa\nrc=0\n')
other = ('COBALT-2222', 'note-bbbb', 'CODE=COBALT-2222 NOTE=note-bbbb\nrc=0\n')
check('canary: clean pair', canary_grade.grade(canary_dir([good, other]))['verdict'], 'PASS')
check('canary: missing output', canary_grade.grade(canary_dir([good, other[:2] + (None,)]))['verdict'], 'FAIL')
check('canary: rc=1', canary_grade.grade(canary_dir([good, other[:2] + ('CODE=COBALT-2222 NOTE=note-bbbb\nrc=1\n',)]))['verdict'], 'FAIL')
check('canary: wrong note', canary_grade.grade(canary_dir([good, other[:2] + ('CODE=COBALT-2222 NOTE=made-up\nrc=0\n',)]))['verdict'],
      'INCONCLUSIVE')
check('canary: malformed output', canary_grade.grade(canary_dir([good, other[:2] + ('garbage\nrc=0\n',)]))['verdict'], 'INCONCLUSIVE')
check('canary: foreign note only in early text', canary_grade.grade(canary_dir(
    [good, other[:2] + ('I see note-aaaa here\nCODE=COBALT-2222 NOTE=note-bbbb\nrc=0\n',)]))['verdict'], 'FAIL')
check('canary: duplicate codes', canary_grade.grade(canary_dir([good, ('AMBER-1111', 'note-bbbb', 'CODE=AMBER-1111 NOTE=note-bbbb\nrc=0\n')]))['verdict'],
      'FAIL')

def transcripts_for(d, sessions, stale=False):
    # a Claude Code projects dir with one transcript per synthetic workdir: (code, tools)
    t = tempfile.mkdtemp(prefix='canary-transcripts-')
    atexit.register(shutil.rmtree, t, True)
    for i, (code, tools) in enumerate(sessions, 1):
        p = os.path.join(t, f'-synthetic-{i}'); os.makedirs(p)
        f = os.path.join(p, 's.jsonl')
        with open(f, 'w') as fh:
            content = [{'type': 'tool_use', 'name': n} for n in tools] + [{'type': 'text', 'text': f'CODE={code}'}]
            fh.write(json.dumps({'type': 'assistant', 'message': {'id': f'm{i}', 'content': content,
                                                                  'usage': {'cache_read_input_tokens': 0}}}) + '\n')
        if stale:
            old = os.path.getmtime(os.path.join(d, 'expected.txt')) - 3600
            os.utime(f, (old, old))
    return t


TOOLS = ['Read', 'Bash', 'Bash']
d = canary_dir([good, other])
check('canary: transcripts present', canary_grade.grade(d, transcripts_for(d, [(good[0], TOOLS), (other[0], TOOLS)]))['verdict'], 'PASS')
d = canary_dir([good, other])
empty_transcripts = tempfile.mkdtemp()
atexit.register(shutil.rmtree, empty_transcripts, True)
check('canary: requested transcript missing (review fixture)', canary_grade.grade(d, empty_transcripts)['verdict'], 'FAIL')
d = canary_dir([good, other])
check('canary: transcript older than this run', canary_grade.grade(
    d, transcripts_for(d, [(good[0], TOOLS), (other[0], TOOLS)], stale=True))['verdict'], 'FAIL')
d = canary_dir([good, other])
check('canary: tool chain skipped', canary_grade.grade(d, transcripts_for(d, [(good[0], TOOLS), (other[0], ['Bash'])]))['verdict'],
      'INCONCLUSIVE')

# --- leakgrade ---
own, foreign = 'JUNIPER-5856', ['AMBER-1111', 'COBALT-2222']
check('leak: first verdict foreign', leakgrade.first_verdict(' AMBER-1111.', own, foreign, ' JUNIPER-5856'), 'LEAK')
check('leak: foreign code in the cold control (review fixture)', leakgrade.first_verdict(' JUNIPER-5856', own, foreign, ' AMBER-1111'),
      'LEAK')
check('leak: foreign id in request-1 output', leakgrade.foreign_in('... AMBER-1111 ...', foreign), ['AMBER-1111'])
check('leak: first verdict suspect', leakgrade.first_verdict(' Question:', own, foreign, ' JUNIPER-5856'), 'SUSPECT')
check('leak: review counterexample (foreign in retries)', leakgrade.settle(
    [{'warm': 'AMBER-1111', 'warm_cached': 0, 'cold': own}, {'warm': own, 'warm_cached': 8096, 'cold': own},
     {'warm': 'COBALT-2222', 'warm_cached': 8096, 'cold': own}], own, foreign, 4048), 'LEAK')
check('leak: J-inside (retries at 0/8096/8096, original 4048)', leakgrade.settle(
    [{'warm': own, 'warm_cached': 0, 'cold': own}, {'warm': own, 'warm_cached': 8096, 'cold': own},
     {'warm': own, 'warm_cached': 8096, 'cold': own}], own, foreign, 4048), 'INCONCLUSIVE')
check('leak: recreated path, all right', leakgrade.settle(
    [{'warm': own, 'warm_cached': 4048, 'cold': own}] * 3, own, foreign, 4048), 'NOISE')
check('leak: recreated path, all wrong', leakgrade.settle(
    [{'warm': 'x', 'warm_cached': 4048, 'cold': own}] * 3, own, foreign, 4048), 'COPYFAIL')
check('leak: one recreated retry only', leakgrade.settle(
    [{'warm': own, 'warm_cached': 4048, 'cold': own}, {'warm': own, 'warm_cached': 0, 'cold': own}], own, foreign, 4048),
      'INCONCLUSIVE')
ok_case = {'first': 'OK', 'settled': None, 'partial': True}
check('leak: run with a missing answer', leakgrade.run_verdict([ok_case] * 9, 10, [])['verdict'], 'FAIL')
check('leak: run with a worker error', leakgrade.run_verdict([ok_case] * 10, 10, ['r0 s1: URLError'])['verdict'], 'FAIL')
check('leak: run with an engine restart', leakgrade.run_verdict([ok_case] * 10, 10, [], restarted=True)['verdict'], 'FAIL')
check('leak: run with an unsettled suspect', leakgrade.run_verdict(
    [ok_case] * 9 + [{'first': 'SUSPECT', 'settled': 'INCONCLUSIVE', 'partial': True}], 10, [])['verdict'], 'INCONCLUSIVE')
check('leak: run without partial-hit coverage', leakgrade.run_verdict(
    [{'first': 'OK', 'settled': None, 'partial': False}] * 10, 10, [])['verdict'], 'INCONCLUSIVE')
check('leak: clean run', leakgrade.run_verdict([ok_case] * 10, 10, [])['verdict'], 'PASS')


# --- stall_trace2 --recompute ---
def raw_run(newcomer=True, stop_stream=False, complete=True, small=None):
    ev = []
    for i in range(2):
        last = 30.0 if (stop_stream and i == 0) else 60.0
        ev.append([[t / 10, int(t - 9)] for t in range(10, int(last * 10)) if t % 2 == 0])
    marks = {'b0': 5.0, 'b1': 15.0, 'p0': 20.0, 'p1': 40.0}
    if complete:
        marks['a1'] = 50.0
    return {'raw': {'label': 'fixture', 'streams': 2, 'stream_prompt': 1000, 'long_prompt': 20000, 'wall_anchor': [0, 0],
                    'errors': [], 'marks': marks, 'events': ev, 'newcomer_token': newcomer, 'engine_newcomer': {},
                    'small_during': small}}


for name, kw, want in [('good run', {}, 0), ('missing newcomer token', {'newcomer': False}, 1),
                       ('stopped stream', {'stop_stream': True}, 1), ('incomplete run', {'complete': False}, 1),
                       ('small arrival answered', {'small': {'tokens': 2048, 'ttft_s': 1.2}}, 0),
                       ('small arrival without a token', {'small': {'tokens': 2048}}, 1)]:
    fd, f = tempfile.mkstemp(suffix='.json')
    with os.fdopen(fd, 'w') as fh:
        json.dump(raw_run(**kw), fh)
    rc = subprocess.run([sys.executable, os.path.join(HERE, 'stall_trace2.py'), '--recompute', f],
                        capture_output=True, text=True).returncode
    check(f'stall_trace2 --recompute: {name}', rc, want)

# --- cow_invariants (release-gate byte phase) ---
import cow_invariants  # noqa: E402
inv = "{'src_rechecks': 50, 'src_mutated': %d, 'src_reallocated': 3, 'controls': 20, 'controls_changed': %d, " \
      "'src_changed_by_copy': 0, 'prefix_unknown': 0}"
ranks = ['Worker_PP0_TP0', 'Worker_PP0_TP1', 'Worker_PP1_TP0', 'Worker_PP1_TP1']
line = '(%s_EP0 pid=1) INFO QSA host-page CoW verify: 10 copies, after 0 (x6 layers); invariants %s'
check('invariants: clean 4 ranks', cow_invariants.check([line % (r, inv % (0, 0)) for r in ranks])[0], True)
check('invariants: source mutated, destinations equal (review fixture)',
      cow_invariants.check([line % (r, inv % (1 if r == ranks[2] else 0, 0)) for r in ranks])[0], False)
check('invariants: control block changed', cow_invariants.check([line % (r, inv % (0, 1 if r == ranks[0] else 0)) for r in ranks])[0], False)
check('invariants: only 3 ranks', cow_invariants.check([line % (r, inv % (0, 0)) for r in ranks[:3]])[0], False)

# --- e2_cow_count (matched CoW frequency gate on the FLASH_E2_COW per-frame counter; cow_invariants does not gate it) ---
import e2_cow_count  # noqa: E402
cow_line = '(%s_EP0 pid=1) INFO FLASH_E2_COW seq=%d pairs=%d views=%d cumulative_pairs=%d cumulative_views=%d'


def cow_log(deltas, layers=6, ranks_=ranks, start_seq=1, base=0):
    out = []
    for r in ranks_:
        cp, cv = base, base * layers
        for i, p in enumerate(deltas):
            cp += p; cv += p * layers
            out.append(cow_line % (r, start_seq + i, p, p * layers, cp, cv))
    return out


clean = cow_log([0, 3, 0, 1, 0])
check('e2 cow: clean 4 ranks PASS', e2_cow_count.phase(clean)[0], True)
check('e2 cow: inferred layers 6', e2_cow_count.phase(clean)[1][ranks[0]]['layers'], 6)
check('e2 cow: given layers mismatch FAIL', e2_cow_count.phase(clean, layers=4)[0], False)
check('e2 cow: only 3 ranks FAIL', e2_cow_count.phase(cow_log([1, 0], ranks_=ranks[:3]))[0], False)
check('e2 cow: seq gap FAIL', e2_cow_count.phase([l for l in clean if ' seq=3 ' not in l])[0], False)
check('e2 cow: cumulative inconsistent FAIL',
      e2_cow_count.phase([l.replace('cumulative_pairs=4', 'cumulative_pairs=5') for l in clean])[0], False)
check('e2 cow: views not pairs x L FAIL', e2_cow_count.phase([l.replace('pairs=3 views=18', 'pairs=3 views=17') for l in clean])[0], False)
check('e2 cow: no rank marker FAIL', e2_cow_count.phase([l.replace('Worker_', 'W_') for l in clean])[0], False)
check('e2 cow: mid-life phase (seq 40, cumulative carried) PASS', e2_cow_count.phase(cow_log([2, 0, 1], start_seq=40, base=17))[0], True)
check('e2 cow: compare equal -> 0', e2_cow_count.compare(clean, cow_log([1, 0, 3, 0, 0]))[0], 0)
check('e2 cow: compare extra copies on -> 2 INSPECT', e2_cow_count.compare(clean, cow_log([0, 3, 0, 2, 0]))[0], 2)
check('e2 cow: compare expected pairs mismatch -> 2', e2_cow_count.compare(clean, clean, expected_pairs=5)[0], 2)
check('e2 cow: compare expected pairs match -> 0', e2_cow_count.compare(clean, clean, expected_pairs=4)[0], 0)
check('e2 cow: compare with a broken side -> 1 FAIL', e2_cow_count.compare(clean, cow_log([1], ranks_=ranks[:2]))[0], 1)

# --- pressure_wait (v3 lifetime phase: bounded wait for completed pressure requests after the leak) ---
import pressure_wait  # noqa: E402


def pw_run(states, bound=100.0, leak_end=1000.0):
    """states: successive progress-file contents (None = unreadable); a fake clock advances 10 s per poll."""
    fd, f = tempfile.mkstemp(suffix='.json'); os.close(fd)
    it = iter(states); now = [0.0]

    def sleep(s):
        now[0] += 10.0
        try:
            d = next(it)
        except StopIteration:
            return
        if d is None:
            open(f, 'w').write('{')
        else:
            json.dump(d, open(f, 'w'))

    open(f, 'w').write('{')  # unreadable until the first poll
    return pressure_wait.wait(f, leak_end, bound, 1, sleep=sleep, clock=lambda: now[0])


prog = lambda c, last: {'completed_requests': c, 'last_tokens': last, 'issued': [1, 1], 'max_tokens': 128}  # noqa: E731
check('pressure_wait: both streams completed after the leak end -> 0', pw_run([prog([1, 2], [1001.0, 1005.0])])[0], 0)
check('pressure_wait: one stream not completed within the bound -> 2', pw_run([prog([1, 0], [1001.0, 1005.0])] * 20)[0], 2)
check('pressure_wait: last token before the leak end -> 2', pw_run([prog([1, 1], [999.0, 1005.0])] * 20)[0], 2)
check('pressure_wait: completes on a later poll -> 0', pw_run([prog([0, 0], [900.0, 900.0]), prog([1, 1], [1001.0, 1002.0])])[0], 0)
check('pressure_wait: progress never readable -> 1', pw_run([None] * 20)[0], 1)
check('pressure_wait: stream count mismatch is not satisfied', pressure_wait.satisfied({'completed_requests': [1], 'last_tokens': [1001.0, 1002.0], 'issued': [1, 1]}, 1000.0), False)


# --- vision_test against a mocked engine ---
def mock_engine(answers):
    import http.server, threading
    seq = iter(answers)

    class Hd(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            body = json.dumps({'choices': [{'message': {'content': next(seq)}}],
                               'usage': {'prompt_tokens': 600, 'prompt_tokens_details': {'cached_tokens': 176}}}).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Hd)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


assets = os.path.join(HERE, 'vision_assets')
vman = json.load(open(os.path.join(assets, 'manifest.json')))
for name, mode, answers, want in [
        ('followup right', 'followup', [vman['table.png']['a'], vman['table.png']['followup_a']], 0),
        ('followup wrong number (review fixture)', 'followup', [vman['table.png']['a'], '999999'], 1),
        ('uuid stale reuse (review fixture)', 'uuid', [vman['isolation'][0]['code']] * 2, 1),
        ('uuid distinct answers', 'uuid', [vman['isolation'][0]['code'], vman['isolation'][1]['code']], 0)]:
    srv = mock_engine(answers)
    rc = subprocess.run([sys.executable, os.path.join(HERE, 'vision_test.py'), f'http://127.0.0.1:{srv.server_port}', 'm', assets,
                         mode, tempfile.mkstemp(suffix='.json')[1]], capture_output=True, text=True).returncode
    srv.shutdown()
    check(f'vision {mode}: {name}', rc, want)

# --- cow_invariants schema (24 Oct review fixture: counters missing -> must not pass) ---
check('invariants: mutation counters missing (review fixture)',
      cow_invariants.check([line % (r, "{'src_rechecks': 1, 'controls': 1}") for r in ranks])[0], False)
check('invariants: counter not an integer',
      cow_invariants.check([line % (r, inv.replace("'prefix_unknown': 0", "'prefix_unknown': None") % (0, 0)) for r in ranks])[0], False)

# --- release_combine completeness (24 Oct review fixture: only `bytes PASS` + empty files was approved) ---
GATES = ['harness', 'identity', 'cow-hook', 'inside', 'after', 'steer-agent-31', 'steer-agent-32', 'steer-agent-33',
         'steer-std-41', 'steer-std-42', 'coverage', 'bytes', 'errors', 'bench']
MANIFEST = ('## mounted overlay files\naaaa /o/qsa.py\nbbbb /o/ops_qsa.py\n## settings\nOVERLAY=/o\nQSA_COW_VERIFY=%s\n'
            '## harness\nhhhh selftest\nweights digest 0123456789abcdef\n')


def release_fixture(v_rows, p_rows, steps, end=(True, True), same_id=False, profile=None):
    d = tempfile.mkdtemp(prefix='release-fixture-')
    atexit.register(shutil.rmtree, d, True)
    for ph, rows, verify, e, rid in (('V', v_rows, '1', end[0], 'run-v'), ('P', p_rows, '-', end[1], 'run-v' if same_id else 'run-p')):
        os.makedirs(f'{d}/{ph}')
        open(f'{d}/{ph}/verdicts.txt', 'w').write(''.join(f'{k} {v}\n' for k, v in rows) + (f'END {rid}\n' if e else ''))
        open(f'{d}/{ph}/manifest-start.txt', 'w').write(MANIFEST % verify)
    open(f'{d}/steps.txt', 'w').write(''.join(f'{k} {v}\n' for k, v in steps))
    args = [sys.executable, os.path.join(HERE, 'release_combine.py'), f'{d}/V', f'{d}/P', f'{d}/steps.txt', f'{d}/out.txt']
    return subprocess.run(args + ([profile] if profile else []), capture_output=True, text=True).returncode


good_v = [(g, 'NOT_RUN' if g == 'bench' else 'PASS') for g in GATES]
good_p = [(g, 'NOT_RUN' if g == 'bytes' else 'PASS') for g in GATES]
good_steps = [(x, 'PASS') for x in ('restart-verify', 'restart-async', 'inside-gen200-cancel', 'long-fullpool')]
check('release: complete matched run', release_fixture(good_v, good_p, good_steps), 0)
check('release: only bytes PASS + empty files (review fixture)', release_fixture([('bytes', 'PASS')], [], []), 2)
check('release: a gate missing in one phase', release_fixture(good_v, good_p[:-1], good_steps), 2)
check('release: duplicate gate', release_fixture(good_v + [('inside', 'PASS')], good_p, good_steps), 1)
check('release: no END marker (run cut short)', release_fixture(good_v, good_p, good_steps, end=(True, False)), 2)
check('release: both phases same run id', release_fixture(good_v, good_p, good_steps, same_id=True), 1)
check('release: required step missing', release_fixture(good_v, good_p, good_steps[:2]), 2)
check('release: vision profile without vision steps', release_fixture(good_v, good_p, good_steps, profile='vision'), 2)


# --- retain_ab --inproc against a mocked engine (24 Oct review fixture: zero compared tokens passed) ---
def mock_lp_engine(first_tokens, cached_fn):
    import http.server, threading, itertools
    seq = itertools.cycle(first_tokens)
    state = {'n': 0}

    class Hd(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            state['n'] += 1
            first = next(seq)
            toks = [first] + ['t'] * 63
            out = {'choices': [{'logprobs': {'content': [{'token': t, 'logprob': -0.5} for t in toks]}}],
                   'usage': {'prompt_tokens': 1000 + len(body['messages'][0]['content']) // 10,
                             'prompt_tokens_details': {'cached_tokens': cached_fn(state['n'], body)}}}
            b = json.dumps(out).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(b)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Hd)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def cached_ok(n, body):  # requests 1-6 are A and B (fresh), 7-9 are C (extend B)
    return 0 if n <= 6 else 1000 + 67000 * 0  + len(body['messages'][0]['content']) // 10 - 700


import retain_ab  # noqa: E402
retain_ab.cases = lambda: {'A': 'a' * 100, 'B': 'b' * 1000, 'C': 'b' * 1000 + 'c' * 1000}
for name, firsts, cfn, want in [('identical outputs, caches as required', ['x'], cached_ok, 0),
                                 ('every run diverges at token 0 (review fixture)', ['x', 'y', 'z'], cached_ok, 2),
                                 ('salt ignored: off runs served from cache', ['x'], lambda n, b: 500, 2)]:
    srv = mock_lp_engine(firsts, cfn)
    sw = tempfile.mkstemp()[1]
    rc = retain_ab.inproc(f'http://127.0.0.1:{srv.server_port}', 'm', sw, tempfile.mkstemp(suffix='.json')[1])
    srv.shutdown()
    check(f'retain_ab inproc: {name}', rc, want)

# --- teacher_forced against a mocked engine: identical distributions pass, a shifted top-1 fails, cache hit inconclusive ---
import time


def mock_tf_engine(shift_after, cached=0, mode='top1'):
    import http.server, threading
    state = {'n': 0}

    class Hd(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                out = {'tokens': list(range(len(body['prompt'].split()) * 2))}
            else:
                if body.get('max_tokens') != 1:  # a load stream: 0.3 s per request, back to back
                    time.sleep(0.3)
                    out = {'choices': [{'text': 'x'}], 'usage': {'completion_tokens': 400}}
                else:
                    state['n'] += 1
                    k = sum(body['prompt'][-5:]) % 7
                    top = {f'token_id:{j}': -0.1 * (j + 1) for j in range(20)}
                    if state['n'] > shift_after and mode == 'top1':  # later conditions: a different top-1 at large Δ
                        top['token_id:5'] = -0.01
                    if state['n'] > shift_after and mode == 'runner-up':  # same winner, a top-5 competitor moves 2.0
                        top['token_id:2'] = -2.3
                    usage = {'prompt_tokens': len(body['prompt'])}
                    if cached is not None:
                        usage['prompt_tokens_details'] = {'cached_tokens': cached}
                    out = {'choices': [{'logprobs': {'tokens': [max(top, key=top.get)], 'top_logprobs': [top]}}], 'usage': usage}
            b = json.dumps(out).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(b)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Hd)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


env = dict(os.environ, PROBES='10', CTX='200', TAIL='40', LOAD='1', MIN_PROBES='10', TF_SETTLE='0')
tf_out = {}
for name, shift, cached, mode, want in [('identical everywhere', 10**9, 0, 'top1', 0),
                                        ('concurrent shifts top-1', 30, 0, 'top1', 1),
                                        ('probes served from cache', 10**9, 128, 'top1', 2),
                                        ('no cache evidence (cached missing)', 10**9, None, 'top1', 2),
                                        ('winner kept, top-5 competitor moves 2.0', 30, 0, 'runner-up', 1)]:
    srv = mock_tf_engine(shift, cached, mode)
    tf_out[name] = tempfile.mkstemp(suffix='.json')[1]
    rc = subprocess.run([sys.executable, os.path.join(HERE, 'teacher_forced.py'), f'http://127.0.0.1:{srv.server_port}', 'm',
                         tf_out[name]], capture_output=True, text=True, env=env).returncode
    srv.shutdown()
    check(f'teacher_forced: {name}', rc, want)
# --regrade re-validates the evidence: a PASS artifact with one probe made cached / short must become INCONCLUSIVE
d = json.load(open(tf_out['identical everywhere']))
regrade = lambda dd: subprocess.run([sys.executable, os.path.join(HERE, 'teacher_forced.py'), '--regrade',
                                     (lambda f: (json.dump(dd, open(f, 'w')), f)[1])(tempfile.mkstemp(suffix='.json')[1])],
                                    capture_output=True, text=True).returncode
check('teacher_forced regrade: valid artifact', regrade(d), 0)
import copy
d2 = copy.deepcopy(d)
d['conditions']['concurrent']['0'].update(cached=100, prompt_tokens=1)
check('teacher_forced regrade: cached + wrong length', regrade(d), 2)
for c in d2['conditions'].values():
    for k in [k for k in c if k != '0']:
        del c[k]
check('teacher_forced regrade: probes dropped from every condition', regrade(d2), 2)
d3 = copy.deepcopy(d2); d3['load_log'] = []
check('teacher_forced regrade: no load evidence', regrade(d3), 2)

# --- cov_gate: per-worker coverage (Codex round 3: a clean last line from one worker hid a mismatch / a silent rank) ---
def cov(kind, text, *base):
    return subprocess.run([sys.executable, os.path.join(HERE, 'cov_gate.py'), kind, '4', *base], input=text, capture_output=True,
                          text=True).returncode
HL = lambda w, h, f, v, x: f"(Worker_PP{w // 2}_TP{w % 2} pid=1) INFO QSA host meta coverage: host {h} fallback {f} verified {v} mismatches {x}\n"
hok = ''.join(HL(w, 500, 0, 500, 0) for w in range(4))
check('cov_gate hostmeta: 4 clean workers', cov('hostmeta', hok), 0)
check('cov_gate hostmeta: earlier mismatch, clean last lines', cov('hostmeta', HL(0, 500, 0, 500, 1) + hok), 1)
check('cov_gate hostmeta: a worker silent', cov('hostmeta', ''.join(HL(w, 500, 0, 500, 0) for w in range(3))), 1)
check('cov_gate hostmeta: MISMATCH line after the last report', cov('hostmeta', hok + "(Worker_PP0_TP0 pid=1) ERROR QSA host meta MISMATCH\n"), 1)
RL = lambda w, sn, c: f"(Worker_PP{w // 2}_TP{w % 2} pid=1) INFO QSA rowcache mixed steps: {sn} seen, {c} with cached decode rows\n"
check('cov_gate rowmix: used on every worker', cov('rowmix', ''.join(RL(w, 100, 90) for w in range(4))), 0)
check('cov_gate rowmix: never used', cov('rowmix', ''.join(RL(w, 100, 0) for w in range(4))), 1)
# Codex round 4: with a baseline the counters must advance during the test; the S3 retained-view gates
_bf = tempfile.mkstemp(suffix='.log')[1]
open(_bf, 'w').write(''.join(RL(w, 100, 90) for w in range(4)))
check('cov_gate rowmix: counters unchanged since the baseline', cov('rowmix', ''.join(RL(w, 100, 90) for w in range(4)), _bf), 1)
check('cov_gate rowmix: counters advanced since the baseline', cov('rowmix', ''.join(RL(w, 150, 120) for w in range(4)), _bf), 0)
ACT = ''.join(f"(Worker_PP{w // 2}_TP{w % 2} pid=1) INFO QSA retained history active: 24 pages per layer\n" for w in range(4))
VL = lambda w, n, b: f"(Worker_PP{w // 2}_TP{w % 2} pid=1) INFO QSA stage timing (5.0 s, per call): pages<=max: n=3 ; verify n={n} bad={b} | dropped 0\n"
check('cov_gate retain: activation lines only (review fixture)', cov('retain', ACT), 1)
check('cov_gate retain: activation then disabled (review fixture)', cov('retain', ACT + ''.join(VL(w, 40, 0) for w in range(4)) + "(Worker_PP0_TP0 pid=1) WARNING QSA retained history disabled: buffer allocation failed\n"), 1)
check('cov_gate retain: verified on every worker', cov('retain', ACT + ''.join(VL(w, 40, 0) for w in range(4))), 0)
check('cov_gate retain: one mismatch', cov('retain', ACT + ''.join(VL(w, 40, int(w == 2)) for w in range(4))), 1)
UL = lambda w: f"(Worker_PP{w // 2}_TP{w % 2} pid=1) INFO QSA stage timing (5.0 s, per call): pages<=max: n=3 rows=9 ret=12 fill=0.10 meta_host=0.01 ms | dropped 0\n"
check('cov_gate retain-used: retained reads on every worker', cov('retain-used', ACT + ''.join(UL(w) for w in range(4))), 0)
check('cov_gate retain-used: active but never read', cov('retain-used', ACT), 1)

# --- decode_ab against a mocked engine (Codex round 4: content-dependent tokens, per-session distributions, streamed
# arrivals): identical decodes pass; a shifted logprob, a foreign winner, a swapped session fail; NaN scores, a
# queued-only run (no token arrives during the prefill) and an early near-tie divergence are INCONCLUSIVE ---
def mock_decode_engine(fault):
    import http.server, threading, zlib
    busy = {'prefill': False, 'done': threading.Event()}

    class Hd(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                ws = body['prompt'].split()
                return self.send({'tokens': [zlib.crc32(w.encode()) % 100000 for w in ws for _ in (0, 1)]})
            if body['max_tokens'] == 1:  # the long prefill
                busy['prefill'] = True; time.sleep(3.0); busy['prefill'] = False; busy['done'].set()
                return self.send({'choices': [{'text': 'x'}], 'usage': {'prompt_tokens': len(body['prompt']),
                                                                         'prompt_tokens_details': {'cached_tokens': 0}}})
            self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.end_headers()
            n, sess = body['max_tokens'], sum(body['prompt'][:5]) % 997
            tied = False  # neartie: every position is a near-tie; the run swaps winners at its first token under the prefill
            for j in range(n):
                mixed = busy['prefill']
                if fault == 'queued' and mixed:
                    busy['done'].wait(); mixed = False
                time.sleep(0.03 if mixed else 0.002)
                s2 = (sess + 1) % 997 if fault == 'swapped' and mixed else sess
                tok, lp, alt, alp = f'token_id:{s2 * 50 + j}', -0.1, 'token_id:99999', -1.5
                if fault == 'foreign' and mixed:
                    tok, lp, alt, alp = 'token_id:77777', -50.0, f'token_id:{sess * 50 + j}', -99.0
                if fault == 'shift' and mixed and j > 10:
                    lp = -2.5; alp = -3.9
                if fault == 'nan' and mixed:
                    lp = float('nan')
                if fault == 'neartie':
                    a1, a2 = f'token_id:{s2 * 50 + j}', f'token_id:{s2 * 50 + 49}'
                    swap = mixed and not tied
                    tied = tied or swap
                    tok, alt = (a2, a1) if swap else (a1, a2)
                    lp, alp = -0.6, -0.8
                top = {tok: lp, alt: alp}
                if fault == 'eosmask':  # min_tokens masks the stop token, the raw logprobs still rank it first
                    top['token_id:248046'] = -0.01
                ch = {'choices': [{'logprobs': {'tokens': [tok], 'token_logprobs': [lp], 'top_logprobs': [top]}}]}
                self.wfile.write(b'data: ' + json.dumps(ch).encode() + b'\n\n'); self.wfile.flush()
            u = {'choices': [], 'usage': {'prompt_tokens': len(body['prompt']), 'prompt_tokens_details': {'cached_tokens': 0}}}
            self.wfile.write(b'data: ' + json.dumps(u).encode() + b'\n\ndata: [DONE]\n\n')

        def send(self, out):
            b = json.dumps(out).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(b)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Hd)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


for name, fault, want in [('identical decodes', None, 0), ('logprob shifted during the prefill', 'shift', 1),
                          ('foreign winner, nothing in common (zero prefix)', 'foreign', 1),
                          ('another session\'s tokens during the prefill', 'swapped', 1),
                          ('NaN logprobs during the prefill', 'nan', 2),
                          ('queued-only: no token arrives during the prefill', 'queued', 2),
                          ('near-tie divergence at the first token under the prefill (no mixed coverage)', 'neartie', 2),
                          ('a masked stop token outranks the emitted token (min_tokens)', 'eosmask', 0)]:
    srv = mock_decode_engine(fault)
    dab_env = dict(os.environ, N='3', LEN='48', LONG='20000', PLEN='800', MIN_MIXED='16', PRE_LEAD='0.1', PRE_TAIL='0.1')
    dab_out = tempfile.mkstemp(suffix='.json')[1]
    rc = subprocess.run([sys.executable, os.path.join(HERE, 'decode_ab.py'), f'http://127.0.0.1:{srv.server_port}', 'm',
                         dab_out], capture_output=True, text=True, env=dab_env).returncode
    srv.shutdown()
    check(f'decode_ab: {name}', rc, want)
    if fault in (None, 'shift', 'queued'):  # --rescore of the saved runs reaches the same verdict without an engine
        rc2 = subprocess.run([sys.executable, os.path.join(HERE, 'decode_ab.py'), '--rescore', dab_out,
                              tempfile.mkstemp(suffix='.json')[1]], capture_output=True, text=True, env=dab_env).returncode
        check(f'decode_ab rescore: {name}', rc2, want)

# --- quality_ab against a mocked engine: top-20 maps are a function of the prompt (deterministic engine); "shift"
# moves the winner on every 8th prompt (candidate worse than the noise floor -> FAIL), "noisy" on every 12th in one
# reference run (noise agreement < 0.97 -> INCONCLUSIVE), "drop" returns 19 entries once (incomplete -> INCONCLUSIVE) ---
def mock_quality_engine(fault):
    import http.server, threading, zlib

    class Hq(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path == '/tokenize':
                return self.send({'tokens': [zlib.crc32(w.encode()) % 5000 for w in body['prompt'].split()]})
            pr = body['prompt']
            h = zlib.crc32(json.dumps(pr[-8:]).encode())
            # the corpus is a word chain w(n) -> w((7n + 3) % 401): the mock "model" predicts the true next token
            true_next = QCHAIN.get(pr[-1], 0)
            toks = [true_next] + [t for t in ((h + 13 * j) % 5000 for j in range(40)) if t != true_next][:19]
            lps = [-0.2 - 0.5 * j for j in range(20)]
            if (fault == 'shift' and h % 8 == 0) or (fault == 'noisy' and h % 12 == 0):
                lps[0], lps[1] = lps[1] - 0.1, lps[0]
            top = {f'token_id:{t}': v for t, v in zip(toks, lps)}
            if fault == 'drop' and h % 50 == 1:
                top.pop(next(iter(top)))
            w = max(top, key=top.get)
            self.send({'choices': [{'logprobs': {'tokens': [w], 'top_logprobs': [top]}}],
                       'usage': {'prompt_tokens': len(pr), 'prompt_tokens_details': {'cached_tokens': 0}}})

        def send(self, out):
            b = json.dumps(out).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(b)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Hq)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


import zlib  # noqa: E402
qd = tempfile.mkdtemp()
atexit.register(shutil.rmtree, qd, True)
qcorpus = os.path.join(qd, 'corpus.txt')
QTOK = {n: zlib.crc32(f'w{n}'.encode()) % 5000 for n in range(401)}
QCHAIN = {QTOK[n]: QTOK[(7 * n + 3) % 401] for n in range(401)}
qn, qwords = 1, []
for _ in range(10800):
    qwords.append(f'w{qn}'); qn = (7 * qn + 3) % 401
open(qcorpus, 'w').write('\n'.join(' '.join(qwords[i:i + 12]) for i in range(0, len(qwords), 12)))
qenv = dict(os.environ, PROBES='200', CTXS='64,256', SEED='5')
qruns = {}
for fault in (None, 'shift', 'noisy', 'drop'):
    srv = mock_quality_engine(fault)
    qruns[fault] = os.path.join(qd, f'{fault}.json')
    subprocess.run([sys.executable, os.path.join(HERE, 'quality_ab.py'), 'run', f'http://127.0.0.1:{srv.server_port}', 'm',
                    qcorpus, qruns[fault]], capture_output=True, text=True, env=qenv)
    srv.shutdown()
for name, (a, b, c), want in [('identical engines', (None, None, None), 0),
                              ('candidate moves the winner on 1/8 of the probes', (None, None, 'shift'), 1),
                              ('reference noisier than 0.97 agreement', (None, 'noisy', None), 2),
                              ('incomplete candidate run', (None, None, 'drop'), 2)]:
    rc = subprocess.run([sys.executable, os.path.join(HERE, 'quality_ab.py'), 'compare', qruns[a], qruns[b], qruns[c],
                         os.path.join(qd, 'cmp.json')], capture_output=True, text=True).returncode
    check(f'quality_ab: {name}', rc, want)

# --- sf_starve against a mocked engine: the long prompt returns under short load (PASS); it waits until the shorts stop
# (starved: FAIL at MAX_S); no short completes while it waits (load not sustained: INCONCLUSIVE) ---
def mock_starve_engine(mode):
    import http.server, threading
    st = {'long': False}

    class Hd(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if body['max_tokens'] == 1:
                st['long'] = True
                time.sleep(30 if mode == 'starve' else 2.0)
                out = {'choices': [{'text': 'x'}], 'usage': {'prompt_tokens': len(body['prompt'])}}
            else:
                time.sleep(5.0 if mode == 'stall' and st['long'] else 0.1)
                out = {'choices': [{'text': 'y'}]}
            b = json.dumps(out).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(b)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Hd)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


for name, mode, want in [('long prompt completes under short load', None, 0), ('long prompt starved', 'starve', 1),
                         ('shorts stall while the long waits', 'stall', 2)]:
    srv = mock_starve_engine(mode)
    rc = subprocess.run([sys.executable, os.path.join(HERE, 'sf_starve.py'), f'http://127.0.0.1:{srv.server_port}', 'm',
                         tempfile.mkstemp(suffix='.json')[1]], capture_output=True, text=True,
                        env=dict(os.environ, K='3', SHORT='64', LONG='2000', WARM='0.5', MAX_S='4', MIN_SHORTS='5')).returncode
    srv.shutdown()
    check(f'sf_starve: {name}', rc, want)

# --- S4: anchored model continuation + pure-Python ownership/completion protocol ---
# No torch/GPU/network in this sub-suite; fake tensor arithmetic is labelled in its output.
resume = subprocess.run([sys.executable, os.path.join(HERE, 'test_resume_prefill.py')],
                        capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
if resume.returncode:
    print(resume.stdout + resume.stderr)
check('resumable prefill: generated control flow + offline protocol (fake tensors)', audit_suite('test_resume_prefill.py', resume), 0)

# --- S5: offline production-shape plan and fail-closed evidence grading ---
s5 = subprocess.run([sys.executable, os.path.join(HERE, 'test_s5_segment.py')],
                    capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
if s5.returncode:
    print(s5.stdout + s5.stderr)
check('S5 segment: CPU tests (GPU path not run)', audit_suite('test_s5_segment.py', s5), 0)
clock_test = subprocess.run([sys.executable, os.path.join(HERE, 'test_s5_clock.py')],
                            capture_output=True, text=True)
if clock_test.returncode:
    print(clock_test.stdout + clock_test.stderr)
check('S5 clock: multi-window, coarse ordering, build and SM-probe rules (CPU)', audit_suite('test_s5_clock.py', clock_test), 0)
runtime_test = subprocess.run([sys.executable, os.path.join(HERE, 'test_s5_runtime.py')],
                              capture_output=True, text=True)
if runtime_test.returncode:
    print(runtime_test.stdout + runtime_test.stderr)
check('S5 runtime: progress, alarms and hard watchdog (CPU only)', audit_suite('test_s5_runtime.py', runtime_test), 0)
from test_s5_segment import fixture as s5_fixture
from s5_segment import combine as s5_combine
check('S5 verdict GO: both pairs', s5_combine([s5_fixture(), s5_fixture('1,3')])['verdict'], 'GO')
s5_bad = s5_fixture('1,3')
for s5_rank in s5_bad['ranks']:
    for s5_mode in ('green24', 'green34', 'green44'):
        s5_rank['modes'][s5_mode]['prefill_steps_s'] = 30
        s5_rank['modes'][s5_mode]['prefill_steps_s_upper'] = 30
check('S5 verdict NO-GO: prefill floor', s5_combine([s5_fixture(), s5_bad])['verdict'], 'NO-GO')
check('S5 verdict INCONCLUSIVE: missing pair', s5_combine([s5_fixture()])['verdict'], 'INCONCLUSIVE')

# --- Full engine identity: mocked docker, actual destination-side hashing; no GPU ---
sys.path.insert(0, HERE)
from step_b_clock_bound import MODULES as STEP_B_CLOCK_BOUND_MODULES  # noqa: E402
from alpha_next_robust_swap import MODULES as ROBUST_SWAP_MODULES  # noqa: E402
# stage1-step: c1's tree-identity ids are superseded on this tree by stage1_step_robust_swap, which needs the base-tree receipt (tools/stage1_step_base_proof.py, the
# chain's or one per run) and runs the default-off proof itself. The helper is imported, not defined here: tools/strict_parallel.py evaluates the command expression below.
from stage1_step_robust_swap import MODULES as STAGE1_STEP_SWAP_MODULES, selftest_base_receipt as stage1_step_base_receipt  # noqa: E402
# stage1-step (attn-meta's rule, lead 3 Nov 13:3x): c1's tree-identity ids are superseded on this tree by stage1_step_robust_swap, whose --self-proof runs the base-tree proof
# (tools/stage1_step_base_proof.py: c1's two modules at fbb9123e in a detached checkout) and the default-off proof in the same process; a literal command, so tools/strict_parallel.py can read it.
from attn_meta_robust_swap import MODULES as ATTN_META_SWAP_MODULES  # noqa: E402
# attn-meta (lead 3 Nov 13:3x): c1's tree-identity ids are superseded on this tree by attn_meta_robust_swap, which needs the base-tree
# receipt of this run (tools/attn_meta_base_proof.py: c1's two modules at fbb9123e in a detached checkout) and runs the default-off
# proof itself; written once per selftest run, before the first such module.
ATTN_META_BASE_RECEIPT = []


def attn_meta_base_receipt():
    if not ATTN_META_BASE_RECEIPT:
        d = tempfile.mkdtemp(prefix='attn-meta-base-proof-')
        atexit.register(shutil.rmtree, d, True)
        path = os.path.join(d, 'base-proof.json')
        proof = subprocess.run([sys.executable, '-B', os.path.join(HERE, '..', 'tools', 'attn_meta_base_proof.py'), '--out', path],
                               capture_output=True, text=True)
        print('selftest attn-meta base proof: rc %d' % proof.returncode)
        ATTN_META_BASE_RECEIPT.append(path)
    return ATTN_META_BASE_RECEIPT[0]
for suite, label in [('test_s5_arena_direct.py', 'S5 arena direct: pinned arithmetic, bucket bytes, fragmentation and no extra window (CPU)'),
                     ('test_s5_dma_window.py', 'S5 repeated DMA: every layer/chunk, bounded CPU copies, strict oracle, solo/overlap accounting (CPU)'),
                     ('test_e2_arena.py', 'E2 single arena: owner epochs, bounded fills, completion and retirement (CPU only)'),
                     ('test_e2_mirror.py', 'E2 mirror: allocation generations, exact-source tables/CoW and fenced RPC ownership (CPU only)'),
                     ('test_e2_arena_dma.py', 'E2 dense arena DMA: real backend call ordering, leases and unchanged sparse fallback (CPU only)'),
                     ('test_e2_runner_lifecycle.py', 'E2 runner seam: actual CoW filter, immutable batch binding, zero-token/dummy RPC and drain (CPU only)'),
                     ('test_e2_startup.py', 'E2 startup: loaded six-layer discovery, baseline arena geometry and config rejection (CPU only)'),
                     ('test_e2_exactness.py', 'E2 exactness: same-input bucket/output byte oracle, immutable failure and strict coverage grader (CPU only)'),
                     ('test_e2_package.py', 'E2 package: pinned-source transforms, manifest coverage, default-off mounts and import identity (CPU only)'),
                     ('test_e2_state_grade.py', 'E2 owner gate: startup geometry and drained-state ledger grader on log lines (CPU only)'),
                     ('test_e2_drain_freshness.py', 'E2 drained proof freshness: final STATE rpc_sequence vs latest completed COW seq per worker, stale -> INCONCLUSIVE (CPU only)'),
                     ('test_e2_ledger_grade.py', 'E2 owner gate: declared-reserve memory contract on ledger records and sampled peaks (CPU only)'),
                     ('test_e2_memory_attribution.py', 'E2 memory diagnostic: timestamp pairing, skew limits, missing evidence and bounded sampler (CPU only)'),
                     ('test_e2_diagnostic_continue.py', 'E2 operator diagnostic continuation: reserve FAIL retained, other faults stop, real queue final FAIL (CPU only)'),
                     ('test_exposure_compare.py', 'E2 owner gate: realized exposure of the matched steps next to the matched retry verdict (CPU only)'),
                     ('test_q16_attribution.py', 'E2 owner gate v6: hard memory failures exit before exposure, retry-only failures attribute only with matched exposure (extracted queue fragment, CPU only)'),
                     ('test_partial_tail_cov.py', 'E2 lifetime coverage: constructed partial-tail eligibility, eviction vs exercise classification, cancellation, bounded budget and FAIL/INCONCLUSIVE propagation (CPU only)'),
                     ('test_e2_serial.py', 'E2 serial CPU-source DMA adapter and unchanged attention arithmetic (CPU only)'),
                     ('test_s5_marlin_partition.py', 'S5 Marlin DP/Stream-K positional partition (CPU, not GPU attribution)'),
                     ('test_s5_retention.py', 'S5 E0 retained histories: incarnation, dirty DMA, publication and retirement (CPU)'),
                     ('test_s5_collector.py', 'S5 E0 collector: 12 fake-worker ordering, ownership and shutdown cases (CPU)'),
                     ('test_s5_transport_source.py', 'S5 E0 exact-source async-output FIFO and overflow-marker hazards (CPU)'),
                     ('test_s5_reply_transport.py', 'S5 E3 owned receiver: FIFO, bounded credits, delayed overflow, quarantine, CPU process (offline)'),
                     ('test_e3_wire_v1.py', 'E3 wire-v1 schema and identities (CPU only)'),
                     ('test_e3_admission_native.py', 'E3 admission and bounded engine prototype (CPU only, no native bridge qualification)'),
                     ('test_e3_frontier_retirement.py', 'E3 allocation-event frontier retirement: recycled pages, atomic refusals and prefix-hit preservation (CPU only)'),
                     ('test_e3_admission_metadata.py', 'E3 production admission metadata: scoped allocator capture, CoW endpoint retention, PLE ngram context, boundary refusal (CPU only, fakes)'),
                     ('test_e3_r4_worker_bridge.py', 'E3 typed worker bridge: open/segment/bind/receipt/release/cancel over the real dispatcher, terminal handback and stock broadcast fidelity (CPU only, fakes)'),
                     ('test_e3_lane_state.py', 'E3 execution state: transaction/lane identities, completion proofs, lane lifecycle and pinned runner seams (CPU only)'),
                     ('test_e3_async_handoffs.py', 'E3 async handoffs: parked transaction, owned replies, completion/publication split, lifetime leases, stop paths and pinned C seams (CPU only)'),
                     ('test_n3_fused_phases.py', 'N3 passive fused phase events: bounded sampling, actual generated score/top-k CPU equality, real fused operation copy/consumer lifetimes unchanged, lazy read, incomplete/failure telemetry and atomic integrity reader (CPU ports; n3_fused_phases fixture leaves)'),
                     ('test_n3_next_layer_prefetch.py', 'N3 bounded immutable-prefix next-QSA lookahead: real operation, four-worker fused path, default-off equality, whole-descriptor order, cancel/drain and stale ownership; CPU leaves in n3_next_layer_prefetch_fixture_leaves'),
                     ('test_n3_producer_witness.py', 'N3 producer witness: real two-layer dependency order, per-bucket provenance and exposed copy wait, refreshed history, bounded atomic rows and six-layer event budget (CPU stream ports; n3_producer_witness fixture leaves)'),
                     ('test_n3_indexer_lifetime.py', 'N3 indexer slice lifetime: pinned actual host functions, CPU score/top-k ports, single-live-slice and bit-equal outputs across 1024/1352 rows and 97k/133k/201k histories (CUDA proof deferred; n3_indexer_lifetime fixture leaves)'),
                     ('test_e3_package.py', 'E3 package: A+B+C composition, default-off mounts, manifest coverage, source drift and import identity (CPU only)'),
                     ('test_e3_native_storage_replay.py', 'E3 retained decode storage: native graph replay events and exact-command all-rank transition retries (CPU)'),
                     ('test_e3_native_overlap.py', 'E3 native-control completion: named engine utility dispatch, typed all-rank acks, worker census/drain, journaled overlap evidence and qualification delegation (CPU only, fakes)'),
                     ('test_e3_graph_routing.py', 'E3 graph routing: fixed-split graph keys, capture/replay/transition guards, budgets and pinned seam transforms (CPU only)'),
                     ('test_e3_lane_dispatch.py', 'E3 lane dispatcher: typed segment kinds to bounded host operations, pause/cursor, PP send/receive, STATE_WRITE event, cancel/drain/release (CPU only, fakes)'),
    ('test_e3_native_submission_credit.py', 'E3 native submission credit: real dispatcher/bridge op identity under delayed selection/attention queries, unchanged pending retry, quarantine and pinned accepted-op bound (CPU only)'),
                     ('test_e3_native_device_path_probe.py', 'E3 native device proof receipt/memory/owner guards (CPU only; actual GPU proof owner-run)'),
                     ('test_e3_native_sparse_waves_device_probe.py', 'E3 native sparse waves device-proof receipts, cancellation and owner bounds (CPU only; GPU proof owner-run)'),
                     ('test_e3_native_expanded_reports.py', 'E3 expanded native continuation reports: lossless pages, unchanged aggregate capacity and exact source joins (CPU)'),
                     ('test_e3_report_validation_cost.py', 'E3 report validation: exact legacy acceptance/refusal equivalence, accepted-set construction once (CPU)'),
                     ('test_e3_staged_chunk_plan.py', 'Native15 staged admission: unchanged minimum64/cap/alignment, complete short-tail plan before allocation, all-rank closure and source measurement (CPU fixtures)'),
                     ('test_e3_staged_device_probe.py', 'Native15 device boundary receipts: unchanged stock minimum64, real operation body at balanced query shapes, raw source/output coverage and owner bounds (CPU guards only)'),
                     ('test_e3_control_observation.py', 'Native15 observability: helper guard shape before failure, exclusive coordinator prepare/poll/validate/serialize counters, snapshot deltas without readiness authority (CPU)'),
                     ('test_e3_transitions_matrix_probe.py', 'Native15 connected matrix receipt guards: numeric comparison mutation, complete raw row keys, fixture limitations and explicit box HOLD (CPU)'),
                     ('test_e3_transitions_lifecycle_probe.py', 'Native15 connected lifecycle receipt guards: actual typed wire records, closed/nonterminal rows, state-write identity, cache/reuse coverage and scoped handoff (CPU synthetic receipt guards only)'),
                     ('test_e3_lane_chunk.py', 'E3 lane chunk: worker-side request registration/update, lane tables and CPU slot mappings, private align context, metadata and forward context (CPU only, fakes)'),
                     ('test_native_uva_retirement.py', 'Native UVA descriptor isolation: real pinned pools and writers, legacy nt/na adoption, pending decode readers, scalar capture views and retirement before reuse/growth (CPU launch leaves)'),
                     ('test_native_uva_legacy.py', 'Legacy nt/na real adopted PREPARE and STATE_FINALIZE through isolated UVA descriptor slots (CPU launch leaves)'),
                     ('test_native_uva_package.py', 'Native UVA is generated, relatively imported and runtime-measured without changing launcher names (CPU)'),
                     ('test_e3_qsa_driver.py', 'E3 QSA layer driver: additive attention pre/post on the D2-composed QSA source, producer -> one-bucket generator -> tail phases, pause/pending (CPU only)'),
                     ('test_e3_native_startup.py', 'E3 worker-native startup: split bank + lane communicators + PLE scratch composition, graph builder, transition RPC (CPU only, fakes)'),
                     ('test_e3_native_imports.py', 'Native8 engine import ownership: E2/E3 package rewriting, actual pure provider path, preserved GPU-state guard and isolated exact namespace with no builder path (CPU source only; real image proof separately required)'),
                     ('test_e3_row1_exactness.py', 'E3-owned row1 backend: frozen E2 counterexample, seven-method source provenance and explicit worker provider binding (CPU)'),
                     ('test_e3_stock_api.py', 'Native7 stock API contract: exhaustive source roles, exact native6 frontend counterexample, signature refusals and pristine-vs-derived pins (AST only; actual real-class imports separately required)'),
                     ('test_e3_executor_threads.py', 'Native6 process-owned executor: actual pinned ready-response/RPC methods on IO/monitor threads, FIFO serialization, shared futures, fork refusal and conservative close (CPU MQ leaves)'),
                     ('test_e3_stock_rpc_credits.py', 'Native16 stock batch queue: claim earlier ordinary RPC futures without consuming native credits, unchanged capacity and cached failures (CPU MQ/output leaves)'),
                     ('test_e3_native_selection_asymmetry.py', 'Native17 known TP selection-copy ordering counterexample: unmodified source stack over explicitly synthetic CPU tensor/event/model leaves, not box evidence'),
                     ('test_e3_native_schedule_profile.py', 'Native18 canonical selection boundary, queried TP profile agreement, exact retries and coordinated refusal/drain over explicit CPU device/model leaves; not box evidence'),
                     ('test_e3_stage_local.py', 'C0 stage-local progress: real driver, bridge, dispatcher, BoundedSender and ReplyTransport on a per-rank ring; switch on vs off per-rank device ledger and persistent state under adversarial stage timing, non-target ranks read and do not answer, stock ring reserve, local release with the PP slot retained until the typed receipt, stop at every tick, TP peer replay, stale and duplicate events, in-wait permit per stage (CPU only; leaves in stage_fixture_leaves.json)'),
                     ('test_e3_stage_local_control.py', 'C0 per-arm switch: native_stage_local over the real router, session, control client and policy; paused idle FULL only; every other setting refuses it (CPU only)'),
                     ('test_e3_stage_local_qsa.py', 'C0 over the real generated QSA source: same counters as the switch-off path, profile and tail stage-targeted, stop at every tick, TP tail replayed to the pending peer before any cancel (CPU only; leaves in stage_fixture_leaves.json)'),
                     ('test_e3_native_reply_hol.py', 'Held-f native reply head-of-line: over the generated WorkerProc reply path and the real ReplyTransport, a native reply on the output rank waits behind pending async decode output (repro); with burst, flash_s5_submit/poll replies leave directly with their transport sequence and are accepted only as the exact head of that rank\'s direct FIFO (reordered, duplicate, unregistered or untyped poisons every pending call), ineligible methods refused at both ends, serialized response writes, the worker progress rows and the bounded in-wait loop (CPU; leaves declared in the module docstring)'),
                     ('test_e3_tp_finalize_skew.py', 'TP completion skew at STATE_FINALIZE through the real driver/bridge/dispatcher/QSA composition: commit beside state_finalize is one continuation, and no completion-dependent action (release, drain, handback, transition/open, later stage submit) runs before both ranks\' own state events; a delayed peer never repeats or enters the TP tail early; every phase pair at every (kind, stage) and every layer/bucket/paused/profile/tail difference still refuses; a random sweep against the parent full-cursor key (CPU only; leaves declared in the module docstring)'),
                     ('test_e3_native21_rpc_stream.py', 'Native21 RPC-thread QSA profile query against a stream-checking device leaf; stream-bound operations still refuse (CPU only)'),
                     ('test_e3_native_pp_stop_order.py', 'Native19 deferred PP receive and typed stop obligation: matching-send-gated events, partial acceptance, clean stage-zero refusal/abort (CPU leaves; not box evidence)'),
                     ('test_e3_native_profile_observability.py', 'Native19 receipt-only peer profiles, worker host query time and actual stock-step bypass counts (CPU leaves)'),
                     ('test_e3_native_tp_tail_skew.py', 'Native19 TP tail barrier and exact stop retry: both peers clear the final event before collective entry (CPU model/communicator leaves, not hardware liveness)'),
                     ('test_e3_native_speculation_gate.py', 'Native19 startup log gate: actual speculative config must be None, with missing or enabled evidence refused (CPU)'),
                     ('test_e3_native26_host_observer.py', 'Native26 host Docker log follow, one-second disk samples, bounded shutdown markers and queue lifecycle (CPU fake Docker only)'),
                     ('test_native34_queue.py', 'Native34 fresh weekend window and protected queue/restore byte equality (CPU only)'),
                     ('test_native34_storage.py', 'Native34 host storage aliases and bounded allocator incarnation coverage (CPU only)'),
                     ('test_native34_control.py', 'Native34 I/S/O checkpoint identity, completion and exact-cohort guards (CPU only)'),
                     ('test_native34_indexer.py', 'Native34 actual pinned indexer host launch and operand range negatives (CPU only)'),
                     ('test_native34_readback.py', 'Native34 bounded indirect readback, real bracket order and host-intent negatives (CPU only)'),
                     ('test_native34_runtime.py', 'Native34 actual stream providers, capture host body, discriminator and cleanup (CPU only)'),
                     ('test_native34_frontend.py', 'Native34 exact queue CLI validation and dual-log diagnostic grading (CPU only)'),
                     ('test_native35_queue.py', 'Native35 R/O/L queue derivation from frozen O3/I4/native28 texts, real-clock slots and unchanged weekend bounds (CPU only)'),
                     ('test_native35_readback.py', 'Native35 arm-O evidence readback: current-stream copies before launch, no drain, no fence; I bracket unchanged (CPU only)'),
                     ('test_native35_grade.py', 'Native35 native-private writer catalogue, startup ledger, lifetime consolidation and exact R/O queue argv (CPU only)'),
                     ('test_native35_dispatch_lifetime.py', 'Native35 metadata-generation lease through the real LaneDispatcher and LaneChunkFactory: release, cancel/drain/reuse, premature reuse and quarantine fail-closed (CPU only)'),
                     ('test_native35_performance.py', 'Native35 P speed arm: exact W1/W2 round argv to the first send, real package/gates/session identity, two-round salt namespace over a loopback APIServer leaf, aggregate grade over actual driver packets (CPU only)'),
                     ('test_native35_performance_queue.py', 'Native35 P queue derivation from frozen O/native28 texts, declared speed window at real clock, measurement tail with the real aggregate grade and PY_FINAL measured_performance (CPU only)'),
                     ('test_native35_l2.py', 'Native35 L2 row4: client window at the accepted resume receipt (claude/native35-l2), parent run source relation, box-shaped timing with the real window/open waits, exact L2 row4 argv (CPU only)'),
                     ('test_native35_p_b.py', 'Native35 P-b: P-a under fresh names plus the exact L2 row4 qualification phase, pinned L2 runner bytes, exact P-b row4 argv to the first send with the L2 runner (CPU only)'),
                     ('test_native35_p_c.py', 'Native35 P-c: P-b under fresh names with FLASH_NATIVE30_LOCALIZE=0, native35-diagnostic NOT_RUN, the post-row4 note and the row4 runner swap as the only deltas; real cleanup graders over a LOCALIZE=0 log with the P-b and emitter counterfactuals (CPU only)'),
                     ('test_native35_p_c_window.py', 'Native35 P-c row4: L2 runner plus the decode stop at the 20-token window, real window/open waits; observer_end spool over real loopback HTTP through the real driver and runner (>112 MiB collected, >512 MiB refused with bytes and sha); observer census reasons; generated decode abort with a native chunk in flight; exact P-c row4 argv (CPU only)'),
                     ('test_native40_box_queue.py', 'Native40 short box queue, exact plain-client CLI, unarmed timeout receipts, invalid recovery rejection and executable window (CPU leaves; real Core gate separately)'),
                     ('test_native41_autoarm.py', 'Native41 opt-in automatic FCFS prefill policy: startup ordering, default-off inertness, request-bound intents, manual/pause exclusion, epoch-bound lease release and weak ownership (CPU; connected plain-request cycles in pinned container; native41 fixture leaves)'),
                     ('test_native41_box_queue.py', 'Native41 auto-arm queue: the native router forwards release_auto_arm (and no other new action) only with FLASH_S5_AUTO_ARM=1 (exact 409 off), the exact arm CLI over loopback HTTP with a status-only receipt poller into the real endpoint/session/driver/policy of a startup-bound stack (stock completes with stock receipts, auto sends only status/pause/release_auto_arm/resume), client-basis grade with the engine basis declared unavailable, every body command and the native41 unarmed clock-proof command through their own argparse, and every python3 call of the queue texts that names a script (draft, declare, settings, qlib identity functions, setcfg) expanded by the real bash against the real parser of that script, with the n43-f missing --container-id as a negative control, derivation from the native40 texts and one phase budget at real clock (CPU; real automatic cycle through the same receipt reader in e3_native41_receipts_probe; native41 fixture leaves)'),
                     ('test_step_a_queue.py', 'Step A queue, fused against stepped on the composed engine: native_qsa_fused forwarded by the real router and session only with both settings, 409 otherwise, 400 when running; startup refusals; both selections and bands through the exact arm CLI over the real control plane; the per-arm and overall grade over real progress emitters (PASS/FAIL/INC, HARNESS or ENGINE, engagement both ways, memory, three report sections); the derived arm loop under real bash with the real death watch and grade; every command through its real parser; everything kept is alpha-perf line for line; the 9,000 s budget and the window at real clock.'),
                     ('test_step_b_queue.py', 'Step B queue, group sizes against size 1 with fused selected in every arm: native_layer_group forwarded by the real router and session only with the setting, 409 otherwise, the arm control, engagement, torn-line rule and three report sections.'),
                     ('test_e3_native_layer_groups.py', 'Step B layer groups: consecutive layers of one stage as one bounded worker command over the real driver, worker bridge, dispatcher, QSA driver and operation with declared device leaves; run-ahead, cancel, failure, ledger, control, startup guard, mutants.'),
                     ('test_step_a_torn_lines.py', 'Step A torn progress line rule, registered before any run: real-format lines through the real assemble and progress_receipts.'),
                     ('test_stage1_handler_anatomy.py', 'Stage-1 stock-handler anatomy (FLASH_S5_HANDLER_ANATOMY, default off): per-RPC seams, CPU and device times over the real executor_hooks worker functions and the generated WorkerProc reply path with V1/V2 runner leaves; off installs and imports nothing and progress lines are byte-identical to step-b off and on; bounds; failure path; mutants (stage1_anatomy_fixture_leaves.json).'),
                     ('test_stage1_anatomy_vm_input.py', 'Stage-1 anatomy VM probe helpers on the CPU rig: decode RPCs and releases through the generated WorkerProc reply path with progress meta, replies read back, final flush, probe.json summary; probe patch anchors on the B5 probe.'),
                     ('test_stage1_anatomy_report.py', 'Stage-1 anatomy report: handler lines joined to arms and chunks by sequence, hand-computed seam shares and the pre-registered decision readout, slower TP rank and carrier, torn/short/gap lines counted, real recorder lines read back.'),
                     ('test_step_a_report.py', 'Step A post-run report: owner time, main-thread occupancy, decode delay and the service model from real progress lines; fused/stepped/g<n> arms group apart.'),
                     ('test_e3_native_qsa_fused.py', 'Step A fused dense QSA layer: one bounded worker command per layer over the real driver, worker bridge, dispatcher, QSA driver and operation with declared CPU leaves: peer skew, cancel, the collective ledger, refusals, the per-arm selection and its latch, mutants.'),
                     ('test_alpha_c0_queue.py', 'C0 queue: the typed serial or local selection through the exact arm CLI over the real control plane, the grade over real emitters, numerics, the arm loop under real bash (CPU only)'),
                     ('test_alpha_next_queue.py', 'alpha-next queue q18a-alpha-next: the declared arm table (g2 ABBA C0 on/off over 3 band 1 rounds, band 2, a C0-on g4 read-only tail) through the exact arm CLI over the real control plane, greedy standing streams, the grade over real emitters, numerics envelope, STOP_RACE on verbatim stop lines, the arm loop under real bash, relative budget only (CPU only; leaves in alpha_next_queue_fixture_leaves.json)'),
                     ('test_alpha_next_c1_queue.py', 'alpha-next-c1 queue q18a-alpha-next-c1: the 20-arm C1 off/on/defer table (g2 ABBA over 2 band 1 rounds with the defer arms beside their on arms, band 2 with its defer arms, a C1-on g4 read-only tail; 9,750 s budget) with C0 local and defer_bind explicit on every arm, the acknowledged defer_bind refused when it differs, the overlap observation read (c1_overlap_read, joined by sequence), the arm forwarding overlap, the grade with C1 regroup, engagement, one command in flight per stage, overlap signatures, OOM, tears, memory-peak and stage margin against the matched C1-off arm, texts derived from alpha-next, the arm loop under real bash, the post-run report, relative budget only (CPU only; leaves in alpha_next_c1_queue_fixture_leaves.json)'),
                     ('test_alpha_next_stopfix_matrix.py', 'alpha-next stop-fix matrix: fix 2 (partial acceptance replayed, then cancel) on fused, grouped g2/g4 and C0 staged drivers, each with a witness mutant that brings back the partial-PREPARE refusal with ownership kept; fix 1 with the fused latch; the item-1 stop receipts (journal row, status counter, flash-s5 stop line, stop_bound_exhausted before the refusal) (CPU; leaves in alpha_next_stopfix_fixture_leaves.json)'),
                     ('test_alpha_next_probe_launch.py', 'alpha-next probe launcher (g4 engine self-run, HARNESS): every engine self-run probe case runs through it; a real ReceiptWriter with only its write seam taken, records read back with the box grader parser (receipt_lines) and handed to the probe capture once per handler, framing defects counted, an overlay without the writer installs nothing (CPU; leaf probe-launch-receipt-sink in alpha_next_seams_fixture_leaves.json)'),
                     ('test_alpha_next_wall_guard.py', 'alpha-next negative control guard (g5 negative observation hung, HARNESS): a hanging suite is stopped at its wall limit and recorded TIMEOUT with its rc (also when the stop does not stop); the judge fails closed: rc 137 with empty suites, a timeout, a missing suite or skips present is INC, never OBSERVED and never an expected FAIL (CPU, subprocesses)'),
                     ('test_alpha_next_peak_receipt.py', 'alpha-next per-chunk peak-allocated receipt (REPORT option 2): peak reset at open and one flash-s5 memory-peak line at release under the memory-receipt flag only, per-chunk values, stage-timer lower-bound flag, failure counted never raised, invisible to inherited memory graders (CPU; leaf peak-cuda in alpha_next_stopfix_fixture_leaves.json)'),
                     ('test_alpha_next_stage1_slow_rank.py', 'alpha-next 4C.2: stage-1 anatomy report slower TP rank per chunk (summed stock-handler time over sequences both ranks logged, carrying seam) and its roll-up per arm and per band; per-sequence majority vs per-chunk, tie, one-rank and unassigned sequences (CPU, synthetic rows)'),
                     ('test_alpha_next_path_qualification.py', 'alpha-next 4C.5 fused and grouped path qualification: the inherited tests for auto admission, engagement, numerical contract and partial acceptance loaded by id; fused partial LAYER acceptance with the accepted peer\'s buckets in flight (replay, no early release, close); grouped run-ahead bound with a peer held, then a clean stop; reuse of every rank\'s bridge and dispatcher after the drained cancel with a fresh fused or grouped open (CPU; Step A/B leaves)'),
                     ('test_alpha_next_c0_ring.py', 'alpha-next C0 ring reading: time unable to submit (episodes, blocked and max ns) and the limiting readers beside headroom_refused, the stage line field under the C0 grader prefix, and the reading never deciding (CPU; stage fixture leaves)'),
                     ('test_alpha_c0_report.py', 'C0 post-run report over real emitter lines (CPU only)'),
                     ('test_e3_stage_overlap.py', 'C1 two chunks in flight on the per-rank ring (stage_net): per-rank device ledger and PP order equal to serial under fair and adversarial timing, successor admission and eligibility, expected lanes, stop at every tick and at each internal boundary of the pair (partial-stop target filter), stale and duplicate replies (CPU only; leaves in stage_overlap_fixture_leaves.json)'),
                     ('test_c1_rebind_witness.py', 'C1 bind-gate witness (review 2 Nov, section 2) and the deferred PP rebind (FLASH_S5_STAGE_OVERLAP_DEFER_BIND): stage 1 late device events of chunk n held incomplete on the real dispatcher; off BLOCKED, on OVERLAPS with the PP_SEND waiting for its own proof; exact peer identity, staggered and late consumers, bounded slots and tables, stop at each successor command, partial PREPARE/LAYER replay to its own targets, refused successor open; FLASH_S5_STAGE_OVERLAP_OBSERVE: per-command device-completion bounds from ingested reports, txn per command, PP_CONSUMER reply marker, the per-successor OVERLAP/NONE read (c1_overlap_read) on and off, its soundness on hand-built interleavings, zero added commands or event queries (CPU only; leaves in c1_rebind_fixture_leaves.json)'),
                     ('test_c1_rebind_on.py', 'C1 connected suite (test_e3_stage_overlap) rerun with the deferred PP rebind selected, plus every early successor deferring its bind (CPU only; leaves in stage_overlap_fixture_leaves.json)'),
                     ('test_e3_stage_overlap_control.py', 'C1 per-arm field: native_stage_local {local, overlap} over the real router, session, control client and policy; the overlap field refused before sending without FLASH_S5_STAGE_OVERLAP=1, a C0-shaped command unchanged (CPU only)'),
                     ('test_alpha_c1_box.py', 'C1 grade over the real overlap composition: sixteen connected arms, regroup, engagement with early open, one command in flight per stage, memory read, OOM, tears (CPU only; leaves in stage_overlap_fixture_leaves.json)'),
                     ('test_alpha_next_c1_composed_fields.py', 'alpha-next-c1 composition fixes in stage_overlap.py: two chunks in flight on C1 ring fixture with fused, g2 and g4 groups and lean replies latched by the real dispatchers; each frame keeps its own composed fields, successors open lean on a lean arm, the fused stall watchdog with two frames, one peak-allocated receipt per chunk per rank; mutants reverting each fix caught (CPU only; leaves in alpha_next_c1_engine_fixture_leaves.json)'),
                     ('test_alpha_next_c1_defer_control.py', 'alpha-next-c1 per-arm deferred bind: the typed defer_bind field of native_stage_local selects the C1 rebind installed by FLASH_S5_STAGE_OVERLAP_DEFER_BIND, only paused idle FULL (refused mid-chunk, with a retained row, never accepted while a PP ticket is live), refused where not installed, without overlap or not a bool, acknowledged through the router, session and client; s5_startup refuses DEFER_BIND/OBSERVE without FLASH_S5_STAGE_OVERLAP; off-on-off-on in one ring incarnation reads BLOCKED/OVERLAPS/BLOCKED/OVERLAPS with ledgers and state equal serial; five mutants caught (CPU only; leaves in alpha_next_c1_engine_fixture_leaves.json)'),
                     ('test_alpha_next_c1_engine_selfrun_b.py', 'alpha-next-c1 engine self-run b (HARNESS, chain a1): C1 pairs and early open judged on the full case records each part wrote, not on case_row summaries (the original judge fails on the same files, b holds; missing or differing files fail closed); run() is the original with one exact-once swap in a copied namespace; gates_b differs from gates by the four selfrun calls (CPU only)'),
                     ('test_alpha_next_c1_gates_c.py', 'alpha-next-c1 gates c (lead rule 5, after chain a1\'s overlap): gates_b plus wait_clear before every container step; the block waits while a returning-append-* container is up or docker ps fails, other containers do not block, its lines carry no rc field; nothing else differs (CPU only, fake docker ps)'),
                     ('test_alpha_next_c1b_queue_names.py', 'alpha-next-c1b QueueNames (lead 3 Nov 13:08, after the box slot lost to an inherited q18a rule): the runner\'s QNAME (build CANDIDATE q18a-alpha-next-c1b) through every [[ $QNAME =~ ]] rule in the c1b runs/ texts in bash, the declaration executed to its last name rule offline, the runner phase names, the builder\'s swapped candidate rule; q18b and dotted names refused (CPU only; also imported by test_alpha_next_c1b_queue)'),
                     ('test_alpha_next_c1b_gates_b.py', 'alpha-next-c1b gates b (chain b and later; supersedes tools/alpha_next_c1b_gates.sh by name after the 3 Nov container overlap): the same chain with wait_clear blocking while any container whose name contains returning-append or append-yield is up (e.g. flashnext-alpha-c0-append-yield-...), not only names starting with them; nothing else differs (CPU only, fake docker ps)'),
                     ('test_alpha_next_c1b_gates.py', 'alpha-next-c1b gates (chain a over bundle-carried engine bytes): tools/alpha_next_c1b_gates.sh line by line against alpha-next-c1 gates_c with the declared swaps, the step order (vm-carry right after the build), wait_clear blocking on returning-append-* and append-yield-* containers or a failing docker ps and on nothing else (CPU only, fake docker ps)'),
                     ('test_alpha_next_c1_negative_guard.py', 'alpha-next-c1 negative control bounded and fail-closed (alpha-next shared wall guard): a hanging suite through the real guarded container call stopped at its wall limit, TIMEOUT judged INC; rc 137 with suites {}, a missing suite, skips, no guard record or an unreadable container record INC, never OBSERVED; negative() end to end writes INC and exits 4; the chain stops INC (CPU only; leaf negative-guard-fakes)'),
                     ('test_alpha_next_c1_rebind_default_off.py', 'alpha-next-c1 default-off proof (tools/alpha_next_c1_rebind_default_off.py): with the rebind switches unset the engine projects to the pre-rebind tree (AST equal, every other engine file identical) and the field is refused; receipt written once; each projection, the byte compare and the refusal step shown live (CPU only)'),
                     ('test_alpha_next_c1b_queue.py', 'alpha-next-c1b queue q18a-alpha-next-c1b: the fixed-schedule numerics table S1 S2 L1 L2 D S3 (six solo band 1 arms: serial x3, C0 local x2, C1 defer; serial selected per arm by native_stage_local local false), the numerics probe (temperature 0, 256 tokens, top-20 logprobs, fresh only) and its verdict declared before the run (bit-equal if S1 = S2, else the S1-S2 bound; outside FAIL ENGINE, drift or missing INC), the C1-aware open-rows rule (e3_c1_open_rows, check 4 INC ENGINE) and ring increments in c1\'s grade, texts derived from alpha-next-c1\'s, the arm loop under real bash, every python call parsed, relative budget 7,800 s only (CPU only)'),
                     ('test_e3_c1_open_rows.py', 'alpha-next-c1b (a): progress-line-open-rows-carried (supersedes progress-line-with-open-rows on C1 overlap arms): each of the four checks and its named reason (check 4 the ENGINE reason), line order, the stage-overlap index; ring_increments from the running headroom_blocked_ns total (CPU only)'),
                     ('test_alpha_next_c1_send_gate.py', 'alpha-next-c1b (b): the deferred PP_SEND gate in stage_overlap: unreachable through the real tick and in-wait paths (bind-due runs first on every tick; the in-wait path excludes PP_SEND), each declining case driven explicitly, a call-site scan; the gate kept as a backstop and its contract held with the bind-due faulted; send-gate-removed is an equivalent mutant on the real paths and caught by the contract tests (CPU only)'),
                     ('test_native43_engine_death_h.py', 'Health-scoped engine death: a broadcast-block stall before the first APIServer health 200 is a startup read, recorded, not a death (returning-append 14904f18, byte for byte; lead 3 Nov 16:49) (CPU only; leaves in native43_engine_death_h_fixture_leaves.json)'),
                     ('test_alpha_next_stop_race_h.py', 'Stop race over the health-scoped scan, importing only native43_engine_death_h (returning-append 14904f18, byte for byte) (CPU only)'),
                     ('test_hub_slot.py', 'Hub test slot and the dry-run record: at most N commands of one class, released on any exit; require-dryrun refuses a missing, failed, stale or dirty record (CPU only)'),
                     ('test_stage1_step_box.py', "q18a-stage1-step box: the arm table's selection through the real router, the acknowledgement recorded, the grade over this table with c1b's C1-aware rules bound, the gate both ways (selection, counters, worker rows) with mutants, the grader rules on a synthetic jump-on flush (CPU only; leaves in stage1_step_fixture_leaves.json)"),
                     ('test_stage1_step_robust_swap.py', 'stage1-step superseded map: exactly the four c1 tree-identity ids, each found once, a fifth refused (mutant); the base-tree receipt and the default-off proof carry the claim (missing INC, not holding FAIL) (CPU only)'),
                     ('test_stage1_step_queue_names.py', 'stage1-step queue names: every bash QNAME rule run in bash with the real QNAME, the runner names, the declaration executed up to its last rule (CPU only)'),
                     ('test_stage1_step_queue.py', 'q18a-stage1-step queue: texts derived from c1 (c1 unchanged), the switch, names, relative budget (18 arms), the arm loop under real bash over the real grade, every python3 call parsed by its real parser (CPU only; leaves in stage1_step_fixture_leaves.json)'),
                     ('test_alpha_next_c1_defer_control_b.py', 'alpha-next-c1b (b): test_alpha_next_c1_defer_control.Mutants superseded by name: the five mutants rebuilt with explicit super(OverlapSegmentDriver, self), each scenario passing unmutated and failing under its mutant by assertion only (no TypeError); the old construction\'s TypeError recorded (CPU only)'),
                     ('test_native43_engine_death_h.py', 'Health-scoped engine death: a broadcast-block stall before the first APIServer health 200 is a startup read, recorded, not a death (returning-append 14904f18, byte for byte; lead 3 Nov 16:49) (CPU only)'),
                     ('test_alpha_next_stop_race_h.py', 'Stop race over the health-scoped scan, importing only native43_engine_death_h (returning-append 14904f18, byte for byte) (CPU only)'),
                     ('test_attn_meta_box_b.py', 'q18a-attn-meta box on the c1b base: the open-rows rule and ring read bound for one grade over this table, overlap arms only, check 4 INC ENGINE, the amendment A grade through it (CPU only)'),
                     ('test_attn_meta_engine_selfrun_c.py', "attn-meta chain c (lead 3 Nov ~20:00): the negative's argument assembly with the real types through both phases and the pin writer, the old _b assembly as its mutant; the name wait with no clearance file, the carry check both ways, the continuation's refusals (CPU only)"),
                     ('test_attn_meta_box_vmc.py','attn-meta vm-c wrapper for the 4x box (lead 3 Nov 16:49): refusals, the input sums, the unchanged probe c, exact-name cleanup; docker and nvidia-smi stubbed (CPU only)'),
                     ('test_attn_meta_robust_swap.py', "attn-meta superseded map (lead 3 Nov 13:3x, option A): exactly the four c1 tree-identity ids, each found once, a fifth refused (mutant); the base-tree receipt and the default-off proof carry the claim (missing INC, not holding FAIL); c1's two modules minus the four run green (CPU only)"),
                     ('test_attn_meta_default_off.py', 'attn-meta default-off proof (tools/attn_meta_default_off.py): with FLASH_ATTN_META_GDN_BATCH unset the engine projects byte-identically onto c1 fbb9123e (declared rows, each with why), switch-unset refusal tests in a child; every check live (CPU only)'),
                     ('test_attn_meta_control.py', 'attn-meta per-arm switch (decision A): attn_meta_select/read through the client, the engine driver (admitted only paused idle FULL: mid-decode, native-active, split, admission, policy cycle and recording refused with typed reasons) and the worker on four real bridge ranks; the qualification worker adapter route (exact payloads, unknown action and switch-off refused); startup hook order; mutants (CPU only; leaves in attn_meta_fixture_leaves.json)'),
                     ('test_attn_meta_c1_g1.py', 'attn-meta G1: C1 defer composes at layer group size 1 (two chunks in flight, defer on and off, equal to serial); a grouped run at size 1 is seen (mutant) (CPU only; stage_overlap fixture leaves)'),
                     ('test_small_piece_batch.py', 'host-gap batch (FLASH_S5_SMALL_PIECE_BATCH, default off): two LAYER commands of a <=176-row fused piece as one worker submission over the C1 ring fixture; equal device ledgers and state to the unbatched driver (full and lean replies), ring-message yield (a stock RPC waits for one command), live run-ahead stop, TP peers one command apart as the frozen partial retry, stop replay, typed failure of command 1 with command 0 kept, prediction checked; mutants (CPU only; leaves in the suite docstring)'),
                     ('test_small_piece_batch_reader.py', 'host-gap batch pair: the marked status-row persistence helper (block kept on, row untouched off) and the exploratory off/on reader over synthetic records: real stage-local receipt line, arm-window <=176-row piece stage times, status-block deltas, token gaps, notes never verdicts, fresh JSON only (CPU only, no engine)'),
                     ('test_control_o3_inwait.py', 'kept-SPLIT control-inwait: bounded applied-output prefix, real Core/four-worker FIFO, later decode still queued, fallback drain, exact cohort/ledger/first output, A/Y1/Y2 and Y2 window, a15 OFF trace and mutants; CPU only'),
                     ('test_control_inwait.py', 'control-inwait: ordered identity beside decode; applied-output cut, exact cohort and typed stream fence; frozen off path, A/Y1/Y2 matrix, four-worker Core/FIFO, refusal and guard mutants; CPU only'),
                     ('test_yield_reentry_settle.py', 'yield-reentry (lead 12 Nov): the always-on observational diagnostic of refused yield re-entry proofs (stage:field counts, one numbers-only line, retries-until-armed histogram; inert against the same run without it, flag off and on; no extra control) and FLASH_S5_YIELD_REENTRY_SETTLE (default off: the K2 first-decode wait before the re-entry, unpaused and holding, bounded, retry bound and stock fallback unchanged); mutants (CPU only; test_append_yield rig, stock handback leaves)'),
                     ('test_q2024p_j2_yrs_reader.py', 'yield-reentry pair: the exploratory yrs off/on reader over synthetic records (real policy yrs_record shape, real V1 log line formats): status-block deltas, proof-miss lines and FULL handoff per path inside the arm window, notes never verdicts, fresh JSON only (CPU only, no engine)'),
                     ('test_yield_no_append_owner.py', 'yield-rules Y1 (lead 12 Nov): FLASH_S5_YIELD_NO_APPEND_OWNER (default off): no junction yield and no candidate lookup while the lane owner has at most max_new rows left; counted once per junction with a request waiting, one receipt, bounded; a fault takes today\'s offer; off equals no hook; boundary, dedupe, waiter and hook-order mutants (CPU only; test_append_yield rig)'),
                     ('test_q2024p_j2_yr_reader.py', 'yield-rules pair: the exploratory yr reader over synthetic records (V1 native-gap field names): yields by owner class (owner-new and rows-left), follow-on yields in lane order, appends per window, Y1 receipts and block, turn join for append pre-open and TTFT, notes never verdicts (CPU only, no engine)'),
                     ('test_yield_window.py', 'yield-rules Y2 (lead 12 Nov): FLASH_S5_YIELD_WINDOW (default off): a yield window; after a served handback (FULL or O3 kept split) the next waiting eligible append (today\'s candidate gates per candidate) is armed by the re-entry instead of the owner\'s resume; PER_BULK counts windows; the owner re-enters when none waits, at 4 appends, after an unserved handback or when the staged append left; a geometry-tail append\'s FULL handback runs first (counted); abort inside a window, O3 path, off never calls the window; bound, served, per-bulk, tail and ordering mutants (CPU only; test_append_yield and test_lane_o3 rigs)'),
                     ('test_ingress_incremental.py', 'ingress-incremental (lead 12 Nov): FLASH_S5_INGRESS_INCREMENTAL (default off): the exact append splice (cached ids up to the last special join inside the common text prefix, the rest tokenised), equal to the full call over replay-shaped conversations; every fallback; cache bounds, LRU, candidates per key; 8 threads; the API binding against native_auto_arm.manual_control (stock sp arm holds the lease: the stock call and object); install off at 0; the log-only engine common-prefix counter and its driver hook; the key fix (lead 13 Nov): the per-request length bound (truncation, max_length) is not in the key, turns differing only in max_length splice, every bound equals stock, a result at its bound is never a splice source, other arguments keep separate entries (CPU only, fake HF-semantics tokenizer; the real HF differential is test_ingress_incremental_hf, pinned container)'),
                     ('test_q2024p_j2_np_reader.py', 'night-pack family reader (lead 12 Nov): per arm n3 absolute turns/h then x sp, native share, agent-seconds (TTFT-s, decode-s, output tokens); the lever counters of the one-commit family (re-entry with the proof-miss append fields, Y2 not-served split / next-left / gap by path, sent -> admit split at the API arrival stamp, verify and timing, reply-aligned prefix match and divergence windows, batch); identity notes against the variant flags, never verdicts (CPU only, synthetic records, no engine)'),
                     ('test_append_no_block_stop.py', 'append-pieces lever A (lead 12 Nov): FLASH_S5_APPEND_NO_BLOCK_STOP (default off): an append (<= 8192 prefill rows past its cached start) runs a native piece through the Mamba block boundary with stock\'s own next piece; the last hash boundary, a junction ahead, the native end, the cap and chunk bound stay stops; every plan equals the seam model (V1 records reproduced off); bulk turns unchanged; commit withholds the stale block from cache_blocks before it runs; counters, one numbers-only fallback line; off is stock\'s split; guard mutant (CPU only; geometry-tail rig over stock\'s split text; the real stock KV manager is test_append_no_block_stop_stack, pinned container)'),
                     ('test_n3_decode_groups.py', 'composition branch (lead 12 Nov): FLASH_S5_DECODE_GROUPS=chunk|off|off-small (default chunk) on an ordinary four-arm j2 packet: chunk is j2\'s table object unchanged, off / off-small move only the n3 rows\' decode_merge (labels, seeds, order equal; back to chunk restores), unknown values refuse; decode groups x yield window pinned on the real _decode_merge_wanted (merge off at each window handback, on for window appends); bundle tail and fresh pythons per mode (CPU only, no engine)'),
                     ('test_q2024p_j2_busyfar_reader.py', 'busy/far decode-forward reader (lead 13 Nov, Codex 15:36 recipe): atomic receipts by writer pid with torn / missing-part / conflicting-duplicate / receipt-seq loss counts, execute-sample pairing by the ordered stock-RPC seq (several nonempty, unknown emptiness, seq gap unpaired), the [L,U] bracket from the proven get_output row, far only 200 ms clear of every expanded envelope, inside >= 0.8 D in the core, ambiguous excluded, INSUFFICIENT_SAME_ARM_SUPPORT, stage = max of TP peers by seq; mutants: neighbour-line join, far without the guard, sum instead of max, zero-filled nulls; decode-timeline stamp: own completion U on every rank (no borrowed U), outside-sample far controls armed by neighbouring certified chunks, sample/txn/anchor_sample session keys, metadata strata and unsupported cells, d-row/handler mismatch, stamp loss facts, exposed-wait critical-path bins with cadence; mutants: borrowed U, arm by wall stamp, line-only key, sum over peers, sum over stages, zero-filled null idle (CPU only, synthetic records, no engine)'),
                     ('test_q2024p_j2_cx.py', 'composition cohort tooling (lead 12 Nov): flag sets (defaults, each flag alone, refused pairs: settle with Y2, decode groups without liveness, verify without ingress, control-inwait with settle/batch/ingress/decode groups/liveness; control-inwait with A, Y1, Y2 accepted), the control-inwait frozen parent pin and container env, four block names and the dry name per tag with no dash-prefix clash, the queue edits (settings, asserts, draft export, arm gate on ENGINE_CRASHED), the cohort reader\'s pooled n3/sp ratio per block and geometric mean, lever A append_pieces delta, lever D steps, fast, exact-overflow rows (the retired counter, either field name, in-flight copies skipped) and stock fallback per arm and rank from stamped sample-fast lines, dev_sample per sampling rank per clean anatomy window and per arm in native context with a no-clean-row note, the D off/on pair adoption line (drop of the first clean window p50 against 0.1 ms) with a second flag difference refused, the control-inwait delta per transition class on every arm beside the auto-arm delta (off: block absent) with absent/present notes, the kept-SPLIT classes cut counters and the lane-o3 line (scope 2, brief 8) with install and cut notes, control-o3-inwait refused without control-inwait or without AUTO_ARM=1/O3=1 in the settings, its frozen a15bed1d parent pin and env, lever D accepted only for the reviewed-fix tree (2eba0dd0 an ancestor and the overflow and watermark markers in the source and overlay decode_step.py; each marker removed refused, None blocks it again), the decode-timeline flag (settings line, asserts, new bound overlay module, its frozen-CPU suite; a parent without handler anatomy, auto-arm, burst progress or one fused period in 1..4096 refused at render time), the liveness guard observer declaration (brief 11: observer off reads NOT_APPLICABLE and is not missing, observer on still reads a blind guard INC, a liveness block seen with the observer off is INC, the poll watch kept either way; the builder binds the packet FLASH_S5_DECODE_LIVENESS and the bundle box is checked against its settings; five mutants), the work-in-window reading (brief 11 B.2: prefill rows from prompt growth prorated to the first token, open turns by chunk share, work = c rows + tokens with the cost ratio named, R_w per block and G_w with a pinned or record turn work, the opposite-side flag and the gap; four mutants: cached tokens for rows, no proration, arrivals shifted by the send, open turns counted done), and identity notes (CPU only, synthetic records, no engine)'),
                     ('test_replay_reply_text.py', 'replay determinism (brief 11 B.1): REPLAY_REPLY_TEXT=model (default) sends the earlier driver request bodies byte for byte (sha pinned from 347e459c over a fake server) with no new key; seeded makes every request a function of the seed alone under any reply content and completion timing, keeps the other draws and the forced decode length, records the mode per turn and in the summary, refuses unknown modes before the agents start; four mutants (CPU only, fake server and clock)'),
                     ('test_quiet_split_box.py', 'quiet-SPLIT control (brief 11 C): 8, 4 and 1 live decoders on the split route with decode_merge=always and no native work, 20..30 s each; the declaration guard sends no control on another declaration, every runner guard (route, generation, transition, admission, live count, native admitted, chunk, auto-arm idle, manual lease, merge selection, decoders advancing, merge steps) has a negative that restores FULL, mutants keep the negatives live, and the real SSE reader closes a stream blocked between chunks (CPU only, fake engine and loopback server)'),
                     ('test_n3_compose_pairs.py', 'composition branch (lead 12 Nov): yield rules Y1 x Y2 together on the yield-window rig (real auto-arm policy, engine driver and scheduler hooks over the CPU worker stack): owner rows left past max_new, the window runs exactly as with Y1 off (counters, event kinds, receipts); within max_new, Y1 refuses the yield and no window opens; two Y1 mutants caught. Y2 x lever A (13 Nov): the same rig with stock\'s Mamba split, the window append fused through its block stop with A on, window counters, kinds and receipts as with A off, call counters on lever A\'s split; four lever A mutants caught. CONTROL_INWAIT x Y2 x lever A (brief 6, 13 Nov): the whole window with CI off/on and A off/on, window counters, kinds, receipts and append-yield counters equal, pieces by A only, CI re-entries identity-only (no barrier) and armed through the in-wait path; a decoder leaving after the certificate invalidates it, a yield revoked after it never switches; three CI mutants caught. D x Y2 and D x lever A (13 Nov): the same flows with decode_step.EngineSide stepped on the real policy before every tick, results and tick counts unchanged, no shared state written per step, D\'s gate on, the anatomy window never eligible; inverted-gate and policy-write mutants caught. D x CONTROL_INWAIT (brief 7): the CI window flows and both CI races with EngineSide stepped, D fast on the decode steps beside the identity observation, the anatomy window never eligible there, views, controls, CI counters and ticks unchanged. DECODE_ANATOMY x CONTROL_O3_INWAIT (brief 8): six kept-SPLIT cases of test_control_o3_inwait (entry with later decode queued, changed-cohort fallback drain, kept yield, Y2 window, revoked yield, eight A/Y1/Y2 vectors) pass unchanged with EngineSide stepped (anatomy on, D off), no state written, no anatomy window in a scope-2 phase; an always-eligible anatomy mutant caught. D x CONTROL_O3_INWAIT (brief 9, reviewed fix 2eba0dd0): the same six cases with D and the anatomy on, the sample tag equal to the gate on every scope-2 step and D gated on beside the cut, nothing else changed; a gate-off-in-scope-2 mutant caught; DECODE_TIMELINE x DECODE_ANATOMY on the generated WorkerProc and real handler recorder, every rank: both tags popped, replies unchanged, no second row beside the clean window, one d row with the same metadata and its own completion; a second-row mutant caught (CPU only)'),
                     ('test_decode_timeline.py', 'decode timeline: default-off AST; generated WorkerProc and real recorders; metadata, event lifetime, bounds, sink faults and guard mutants; CPU only'),
                     ('test_decode_step.py', 'decode-step (lead 13 Nov): FLASH_S5_DECODE_SAMPLE_FAST (lever D, default off) and FLASH_S5_DECODE_ANATOMY (default off): off never imports decode_step and leaves every RPC untouched (fresh process); the wire tag on the real owned collective_rpc; the per-arm gate (native arm only, stock sp excluded); the 20 s clean decode window (no native progress, policy inactive), its budget, window cap, stamp bound and line size; worker handler-anatomy rows of the window kept out of the progress trace; every ineligible sampler step is the stock call unchanged; the image sampler signature and call site; startup refusals and overlay declarations; trace-leak and unpopped-tag mutants; the periodic rank-tagged sample-fast line (compose line); no install on a sampler whose _sample_random is not the one Sampler defines, or that carries watermarking state (GPUWatermarkSampler, owner review 13 Nov), one refusal line, two mutants caught (CPU only, fake torch; the sampling math is test_decode_step_torch, pinned container)'),
                     ('test_attn_meta_box.py', "q18a-attn-meta box: the amendment A table (2x2 at G1 interleaved, MP band 2, P2/MP2 G2 tail, decode-only baselines), the select inside c1's release before release_auto_arm, the end read, the attn-meta rules (leak FAIL ENGINE, not engaged INC, order, startup lines) over c1's real grade, the three decode retentions and summary (a)/(b); mutants (CPU only; leaves in attn_meta_fixture_leaves.json)"),
                     ('test_attn_meta_queue.py', "q18a-attn-meta queue: texts derived from c1's (c1 unchanged), the declaration's QNAME test executed with the runner's real QNAME, names resolve, relative budget (17 arms), the derived arm loop under real bash with the real grade and read (CPU only; leaves in attn_meta_fixture_leaves.json)"),
                     ('test_attn_meta_vm_input_b.py', 'attn-meta VM input b: provenance with engine bytes, check-bind against a bundle, name rule (CPU only)'),
                     ('test_attn_meta_vm_input_c.py', 'attn-meta VM input c: probe c and attn_meta_select in the input, engine-bytes binding, name rule (CPU only)'),
                     ('test_attn_meta_gates.py', "q18a-attn-meta gate chain (tools/attn_meta_gates.sh): c1's steps in c1's order with the base-tree and default-off proofs, the superseded runner with the receipt, every container step after wait_clear (the lead's names and the clearance file), the block run under bash with docker ps replaced (CPU only, no docker)"),
                     ('test_attn_meta_pack.py', "q18a-attn-meta chain tools: parent pin against c1's delivery b, the freeze (off is c1, five mutants refused), the engine self-run derived from c1's by exact swaps, the probe derivation (530), the declared names of build, spec, review pins, tamper, clock scan (PASS over the tree), precheck conf, staging check and queue pins (CPU only, git)"),
                     ('test_alpha_c1_report.py', 'C1 post-run report over real emitter lines (CPU only)'),
                     ('test_alpha_c1_queue.py', 'C1 queue: texts derived from C0 l, the sixteen-arm loop under real bash over the real C1 grade, every python3 call parsed, the 9,000 s budget against the measured box walls (CPU only; leaves in alpha_c1_fixture_leaves.json)'),
                     ('test_alpha_perf_queue.py', 'Alpha-perf queue on the composed engine with the dense policy: both bands through the exact arm CLI over the real control plane with the declared standing budget, the per-arm and per-band grade over real progress emitters (explicit PASS/FAIL on crash-free completion and 0 disagree, speed never gated), the derived arm loop under real bash with the real death watch and grade (band manifest, env through the watch, budget guard, dead engine), every command through its real parser, the band 2 manifest and length declaration, the 2.5 h budget and the window at real clock.'),
                     ('test_native41_progress_queue.py', 'Held-f native progress queue: the native router forwards native_progress only with FLASH_S5_AUTO_ARM=1 and FLASH_S5_NATIVE_PROGRESS=burst (held-e 409 otherwise) and the engine only when burst is installed and paused idle FULL, the exact arm CLI for current|burst x live|solo over loopback HTTP into the real endpoint/session/driver/policy/NativeProgress (status, pause, native_progress, release_auto_arm, resume only; solo cancel at the first observed native chunk; /metrics preemption reads), the receipts grade over lines from the real engine and worker ProgressTrace emitters in docker log form with each completeness defect INCONCLUSIVE, every queue python call against its real parser, derivation from the held-e texts and one phase budget at real clock (CPU; native chunk receipts on the connected Core in e3_native41_progress_probe; native41 progress fixture leaves)'),
                     ('test_e3_native_anatomy.py', 'Held-g R1/R5 anatomy: DeviceSpans per lane command (exact busy/idle/QSA sums, refusal reasons and held time, pool and slot reuse, failed read counted, detail cadence, release never raises) and StepAnatomy step classes on a fake clock; the real ReplyTransport idle wait; the idle predicate names pinned against the vllm source; spans on the real driver/lane/dispatcher/bridge credit composition (classes, 41 credit refusals, one anchor sync after every event completed, lane unchanged) (CPU; leaves in native41_held_g_fixture_leaves.json)'),
                     ('test_e3_native_lean_replies.py', 'Held-g L4 lean replies on the r4 connected stack: full and lean drive the same chunk to the same cumulative reports, census and cache calls under seeded out-of-order completions; the validating constructor gives the same run; journal report keys pinned; driver refusals of replayed, foreign, stale, unaccepted and full-transaction lean reports; stream gap/replay/reorder/substitution; release forgeries and unsent completions refused; chain and release-check mutants (CPU; leaves in native41_held_g_fixture_leaves.json)'),
                     ('test_native41_tp_skew.py', 'Native41 held-e auto-arm-only per-chunk allocator receipts on each worker over the connected SkewNetwork composition: open and released rows per rank with FLASH_S5_AUTO_ARM=1, no query and the same RPC method sequence with it off, uninitialized or failing CUDA counted and never raised (CPU; native41 fixture leaves; the TP skew tests are test_e3_tp_finalize_skew)'),
                     ('test_native41_unarmed_box.py', 'Native41 unarmed-hold clock proof across the CT/container identity split: native40 refuses the CT view exactly as on the box (client-engine-clock-unproved, HARNESS), native41 proves the domain from container init time namespace, zero offsets, kernel boot_id and a bracketed engine sample over the real stack; refusals for offsets, identities, stale samples and docker failure (CPU; native41 clock fixture leaves)'),
                     ('test_native40_unarmed_hold.py', 'Native40 unarmed stock-prefill physical split exclusion, weak exact request identity and 30s FULL return across disarm; no native-work interruption (CPU only; native40 fixture leaves)'),
                     ('test_native39_trunk.py', 'Native39 trunk robustness (Codex native37 P1/P2): generated positive-frontier and running-prefill seams with exact revert, physical split holds even without an admission intent and never on the native one-token handback, a held head no longer strands the retained row, weak refusal memo with free-seam eviction, typed full-prefix-hit/resumed-with-output refusals and furthest-boundary basis, bounded idle hold returning FULL with a typed split-hold-timeout, narrowed and window-correlated prefill_end fallback (CPU only; R/W run in the pinned-image self-run; native39 fixture leaves)'),
                     ('test_native43_dense_boundary.py', 'Native43 explicit dense policy: real generated transport, bounds, source holds, trace cap, four-rank certified cutoff/cancel/FULL (CPU leaves)'),
                     ('test_native43_boundary_queue.py', 'Native43 paired diagnostic: exact drafts, real CLI/tokenize seam, bracket across clock views, strict boundary evidence and real window budgets (CPU leaves)'),
                     ('test_native39_trunk_queue.py', 'Native39 trunk queue: exact derivation from the P-c texts, trunk window at real clock; restart/health/bind receipt under bash with the real host observer (bind before the first 200 request PASS, after FAIL, absent or another container INC, health deadline stops); native W1 then the real plain 24 Oct client, idle check and W2 rounds with the real grade and trunk summary; W2 length guard against /tokenize; sourced trunk settings through the real setcfg and S5 launcher render to a loopback --host; checks over retained box records (CPU only; native39 queue-* fixture leaves)'),
                     ('test_native39_lifecycle_queue.py', 'Native39 lifecycle queue: exact derivation from the trunk held-b texts and the native35-L block, lifecycle window (9600 + 900 s, last entry 03:05Z) at real clock; the lifecycle region under bash with the real instruments check, session client, auth hook, leaktest 8 x 3, pressure wait and pp_release/native35 lifetime grades (pressure-lifetime INC without pool telemetry, row1 or crash or mapped observer withholds, refused FULL hygiene stops, a real leak fails); the lifecycle CLI on the trunk W2 manifest; sourced lifecycle settings = the trunk engine with observer and offers off (CPU only; native39 lifecycle fixture leaves)'),
                     ('test_native43_lifecycle_queue.py', 'Native43 lifecycle queue (short scope): exact derivation from the box-run native39 lifecycle texts with the dense policy and no trace/stop, short window (5000 + 900 s, last entry 04:21:40Z) at real clock; the short lifecycle region under bash with the real instruments check, session client and native43/native35 lifetime grades (reuse-incomplete INC, row1 or crash or mapped observer withholds, refused FULL hygiene INC, no leak or pressure tool reached); the lifecycle CLI on the trunk W2 manifest; sourced settings through the real launcher render and the native43 engine policy parser (CPU only; native43 lifecycle fixture leaves)'),
                     ('test_native43_engine_death.py', 'Native43 le-fix engine death: scan of retained host streams against the queue stop intent (worker fatal record, executor/worker teardown or broadcast stall before the orderly trigger DEAD; orderly shutdowns ALIVE) over the l-e and earlier box logs, and the bounded watch over a real child process, GPU and container fakes (CPU only; native43 engine-death fixture leaves)'),
                     ('test_native43_le_fix_queue_deltas.py', 'Native43 le-fix queue deltas D1-D7 over the l-e texts: each applied once, bash syntax, only the declared lines change; host_crash_detected and engine_watched under bash with the real death tool (CPU only; native43 le-fix queue fixture leaves)'),
                     ('test_n3_gap_receipts.py', 'N3 bounded monotonic gap receipts: real generated Core admission, four-worker control and commit with identical work under a failed sink, policy pause/drain, actual atomic writer and integrity reader; CPU leaves declared'),
                     ('test_native43_le_fatal_log.py', 'Native43 le-fix worker-fatal log: the parent logs the fatal record before executor shutdown, on the actual generated WorkerProc RPC path in a spawned worker, including the l-e Row1OracleError chain (CPU only; native43 le-fix fatal-log fixture leaves)'),
                     ('test_native43_le_bounded_enqueue.py', 'Native43 le-fix bounded broadcast enqueue: the generated stock and owned collective_rpc writes raise the stored WorkerRPCError once the executor is marked failed (pre-fix blocks), live writes complete and keep the long-wait warning (CPU only; vLLM real MessageQueue in the pinned-image probe; native43 bounded-enqueue fixture leaves)'),
                     ('test_native43_le_bounded_enqueue_stepped.py', 'Alpha-next PLAN 5b: load-robust twin of the frozen le-chain bound (swapped in by alpha_next_robust_swap): a SteppedRing releases each bounded slice, the stored WorkerRPCError is raised within the slice in progress at the hand-off (no new slice after it, exact in slices, no clock), nothing written, one teardown (CPU only; alpha_next_robust fixture leaves)'),
                     ('test_native43_row1_dense.py', 'Native43 l-f row1 under the dense policy: the declared expected-route table, the real generated flash_e2_arena_attention entered on its own dense branch with os.environ unchanged, the generated sparse helper passing dense against it at 25 pages, and the pre-fix route_mismatch (CPU only; native43 row1-dense fixture leaves)'),
                     ('test_native43_row1_grade.py', 'Native43 l-f policy-aware row1 grade: the base grade with only the cartesian route rule replaced under the dense policy (dense and beyond-arena coverage, sparse FAIL), current policy identical to the base, and the qualification wrapper binding/restoring _grade_row1 (CPU only; native43 row1-dense fixture leaves)'),
                     ('test_native43_lf_queue.py', 'Native43 l-f queue: exact derivation from the box-run l-e texts (le-fix deltas, row1 policy client, names), row1 under bash with the --qsa-policy dense argv and the engine watch, the connected l-f lifecycle tail healthy and on the l-e teardown line, and the l-f window slot at real wall clock (CPU only; le-fix queue and native43 lifecycle fixture leaves)'),
                     ('test_native38_plain_bind.py', 'Native38 plain-client startup FULL fence, later native prepare, restart/health proof latch and generated non-output PP error over independent fatal pipe, real reply transport (CPU only; native38 fixture leaves)'),
                     ('test_native37_robust.py', 'Native37 robustness: staged-plan search receipt, typed admission refusals held on an armed split route through the real session/control/driver, prefill_end typed over an empty evidence window (counterfactual: raised unchanged), and T3 W2 harness FULL return after a 500, a typed refusal or a dead peer with the lane error and first_error kept (CPU only; T1/T2 run in the pinned-image self-run)'),
                     ('test_native35_p_c_continuation.py', 'Native35 P-c declared row4 capacity continuation: real census/driver/qualification grading over source packets (kernel and correlation overflow continue; replay bound, non-capacity reasons, raw FAIL and report errors withhold), the >512 MiB refusal over real loopback HTTP, each precondition, the queue gates and both W2 rounds under bash with the real classifier and W2 gate, receipt re-proof (CPU only)'),
                     ('test_native33.py', 'Native33 rank1 launch fences, PP exclusion and real pinned graph host capture/pointer conformance (CPU only)'),
                     ('test_native32.py', 'Native32 real Bridge receipt receiver conformance, frozen native31 negative and scope preservation (CPU only)'),
                     ('test_native31.py', 'Native31 TP canonical selection, real sparse wave boundary and certified prompt client error (CPU only)'),
                     ('test_native30_e.py', 'Native30-e exact native28 preconditioning, receipt activation/tail and coverage grading (CPU only)'),
                     ('test_native30_cli.py', 'Native30 exact short-CLI host validation to first HTTP send (CPU only)'),
                     ('test_native30_localize.py', 'Native30 per-rank flushed labels, scoped sync, PP deadlock guard, short queue restore and classification (CPU fixtures only)'),
                     ('test_native30_host.py', 'Native30-d passive host indices, expanded block bounds, absolute UVA rotations, incomplete event coverage and bounded output (CPU only)'),
                     ('test_native30_c.py', 'Native30-c measured snapshot gate, dual-log fault receipts and operator window (CPU only)'),
                     ('test_native30_identity.py', 'Native30 identity real capture/check/probe, old timeout refusal, content mismatch and cleanup budget (CPU fixtures only)'),
                     ('test_e3_native28_host_guard.py', 'Native28 root-space final evidence and prompt queue trap, declared first-advance read bound (CPU only)'),
                     ('test_e3_native28_rule4_continuation.py', 'Native28 control hard-cap continuation preserves raw INC and refuses split, structural and other gates (CPU only)'),
                     ('test_e3_native_startup_binding.py', 'Native generated-package startup: actual closure/relay/transport/admission/driver binding, namespace identity and pinned auth with no mocked ready/capability flag (CPU filesystem/OS fixtures)'),
                     ('test_e3_model_hooks.py', 'E3 lane forward: additive prefix/layer/suffix on the pinned model, exact inverse, segment driver scopes and pause (CPU only)'),
                     ('test_e3_native_observation.py', 'Native shared observer lifecycle: real ledgers, dynamic source pins, role drains and retained window before FULL handback (CPU only)'),
                     ('test_e3_native_qualification.py', 'Native qualification worker identity, source scope, teacher and observer ownership over actual modules (CPU only)'),
                     ('test_e3_native_qualification_driver.py', 'Native qualification phase producer: immutable plans, actual control ownership, row1 source scope, retained teacher artifact and short-window proof (CPU loopback/fakes)'),
                     ('test_e3_q18a_guards.py', 'Native raw qualification graders: producer-shaped row1, unchanged rule4, paired raw graph proof, immutable identities and refusal precedence (CPU only)'),
                     ('test_e3_native_measurement_clients.py', 'Native measurement clients: actual session/ordered worker actions, unpadded batch census, typed cancellation reports, unchanged W1 requests, fixed row1 prefix/tail and source-joined window refusal (CPU loopback/fakes)'),
                     ('test_e3_native_performance_queue.py', 'Fresh native performance queue: actual interface names and phase contracts, W1/W2 locked by qualification, unchanged hygiene shapes, reserve failure and exact owned restore (fake OS/phase executables only)'),
                     ('test_e3_native_teacher_sequence.py', 'Native teacher lifecycle: actual native receipt bindings, physical rows unchanged, initial typed FULL certificate, bounded worker transport and no probe action relabelling (CPU only)'),
                     ('test_e3_native_qualification_frontend.py', 'Native qualification API frontend: exact commands, authenticated loopback and actual teacher plan ownership (CPU only)'),
                     ('test_e3_native_api_qualification_composition.py', 'Actual API NativeControlClient owns the input processor, distinct from EngineCore utility ownership (CPU only)'),
                     ('test_e3_native_row1_grade.py', 'E3 native row1 raw grader: direct producer records, exact layer/route coverage, identity and release census (CPU only)'),
                     ('test_e3_native_daytime_rehearsal.py', 'E3 daytime integration: full-prompt source-store/HTTP/grader lifecycle and exact held native12 queue refusal/restore (CPU fixtures only)'),
                     ('test_e3_native_source_oracle.py', 'E3 native source ownership and serial QSA byte/output oracle: first-PREPARE binding, pinned host layout, source holds and cancellation (CPU only)'),
    ('test_e3_native_measurement.py', 'E3 source-joined native measurement: stage-local expanded ops, actual queried completions, raw journal loss retained and exact session window binding (CPU only)'),
                     ('test_e3_lane_private.py', 'E3 lane-private bindings: owned QSA top-k/UID views, extra workspace lane, interleaved decoder access, exact restoration and refusals (CPU only, fakes)'),
                     ('test_e3_lane_comm.py', 'E3 prefill-lane communicators: PyNccl-only TP scope, tagged PP channel credits/header/release order (CPU only, faked NCCL/torch)'),
                     ('test_e3_lane_comm_probe.py', 'E3 lane-comm probe: frozen verdict, import isolation and unarmed pair-1,3 queue (CPU only)'),
                     ('test_e3_native_report_read.py', 'Actual bounded native JSON read: capacity vs bytes, successful parse/digest, failure freshness and thread-local identity (CPU ctypes only)'),
                     ('test_e3_activity_accounting.py', 'Deferred CUPTI joins and conservation (CPU only)'),
                     ('test_vmid_job9b.py', 'VM9b CUPTI thread identity and wrapper conservation (CPU only)'),
                     ('test_e3_engine_offers.py', 'Engine offer ledger: pinned leaves and separate composition (CPU only)'),
                     ('test_vmid_job13.py', 'VM13 native prefill probe: grading wiring, negative control, declared vocabulary and queue text (CPU only, never run)'),
                     ('test_vmid_job13b.py', 'VM13b graph helper ABI: actual replay call sites use owner, raw graph reserved for serialization, retained progress (CPU only)'),
                     ('test_vmid_job13c.py', 'VM13c pinned metadata/event API conservation, query failures/rogue work refused, scoped negative controls (CPU only)'),
                     ('test_vmid_job13d.py', 'VM13d actual private initialization and exact pending query: source constructor, shared accounting and hardware-probe collection guards (CPU only)'),
                     ('test_vmid_job13e.py', 'VM13e real owner-start lifecycle and standalone bundle finalizer: bounded fresh rerun, positive/private/graph guards unchanged (CPU only)'),
                     ('test_vmid_job13f.py', 'VM13f exact header-backed library APIs: no GPU joins, no nonzero returns, raw retained hardware regrade and fresh queue (CPU only)'),
                     ('test_vmid_job13g.py', 'VM13g model owner shells: wrapped backbone discovery, native startup/private/forward ownership and unchanged scope gates (CPU imported-class doubles; actual vLLM classes require VM)'),
                     ('test_vmid_job13h.py', 'VM13h lazy Mamba startup: stock descriptor validation, private/ordinary context isolation and required hardware proof (CPU fixtures; actual context kernels require VM)'),
    ('test_vmid_job14.py', 'VM<vm-id> native source/oracle: real holds, same-storage UVA, actual epoch rebind, cancellation, exact packed union and bounded queue (CPU only)'),
    ('test_vmid_job14b.py', 'VM<vm-id> native oracle setup: actual operation/backend constructor, source registration and scoped abort, correct exactness module and fresh bounded rerun (CPU only)'),
                     ('test_e3_native_graph_placement.py', 'Native raw graph + two-partition accounting: same bound window, per-node replay multiplicity, eager conservation and raw FAIL precedence (CPU only)'),
                     ('test_e3_native_placement_evidence.py', 'Native worker packet proof: actual bounded report reads, paired raw graph regrade, exact retained/live generation distinction (CPU ctypes and fakes)'),
                     ('test_e3_native_placement.py', 'Native overlap accounting: exact-id decode/prefill partition union, independent regrade, raw-loss retention and receipt refusals (CPU only)'),
                     ('test_e3_native_prefill_offers.py', 'Native prefill offers: source-pinned lane vocabulary, auxiliary owner scopes, copy-only role, completion credit and fail-closed seams (CPU only)'),
                     ('test_e3_native_accounting_source_closure.py', 'E3 native accounting source closure: exact pending event queries, private UID/lock leaves and original-operation failure ownership (CPU only)'),
                     ('test_e3_teacher12_queue.py', 'Teacher12 diagnostic never runs W1; scope and cleanup (CPU only)'),
                     ('test_vmid_job10b.py', 'VM10b pinned NumPy host mirrors in teacher collector (CPU only)'),
                     ('test_vmid_job11.py', 'VM11 native-window-bound engine ledger composition (CPU only)'),
                     ('test_vmid_job10.py', 'VM10 teacher sampler readback and digest qualification (CPU only)'),
                     ('test_vmid_job9.py', 'VM9 bridge grading and ownership (CPU only)'),
                     ('test_vmid_job8.py', 'VM8 positive eager activity accounting and negative controls (CPU only)'),
                     ('test_vmid_job7b.py', 'VM7b late allocator release and native-crash veto (CPU only)'),
                     ('test_vmid_job7.py', 'VM7 real-owner capture repair, frozen negative controls, exact native node join and pinned header compile (CPU only)'),
                     ('test_e3_capture_streams.py', 'Pinned split capture side-stream ownership, sharing, restoration, graph leases and source pins (CPU only)'),
                     ('test_e3_green_native.py', 'E3 green-context split, split decode bank capture/replay/transition and worker/cudagraph seams (CPU only, faked driver/torch)'),
                     ('test_e3_prefill_streams.py', 'E3 prefill side-stream owners: green-role isolation, exact owner restoration and allocator lifetime (CPU)'),
                     ('test_vmid_job7f.py', 'VM<vm-id> prefill owners and decode canonical node/context/multiplicity proof (CPU)'),
                     ('test_vmid_job7c.py', 'VM<vm-id> prefill owner qualification: raw eager/graph census, negative controls and retained context cleanup (CPU)'),
                     ('test_e3_green_probe.py', 'E3 green probe: frozen verdict rules, reviewed sm_86 build command and unarmed queue mounts/syntax (CPU only)'),
                     ('test_e3_native_integration.py', 'E3 native integration: generated worker/engine startup, typed RPC driving, completion/commit ordering, stop paths and record status (CPU only, faked device/transport)'),
                     ('test_e3_worker_operations.py', 'E3 D2 worker operations: bounded QSA buckets, generated call sites executed, delayed events, cancel/drain and stale-epoch refusal (CPU only, fake CUDA)'),
                     ('test_e3_sparse_history_dma.py', 'E3 sparse transport: fixed-capacity union, deferred selection and lease-bound compact DMA (CPU)'),
                     ('test_e3_stopfix_partial_segment.py', 'Stop with one TP peer accepted and the other pending: the exact retained command is replayed to its targets under the stop bound, then cancelled, on the real driver, bridges and dispatchers (CPU leaves declared, no box evidence)'),
                     ('test_e3_stopfix_refused_open.py', 'Stop after a refused continuation open: the retained row closes first, the chunk retires as never submitted, on the real driver, bridges and dispatchers (CPU leaves declared, no box evidence)'),
                     ('test_alpha_m3_receipts.py', 'Alpha M3 receipts: indexer score sizing against the reference, real log formats through the phase-delta tool, queue deltas (CPU)'),
                     ('test_alpha_next_receipts.py', 'Alpha-next producer-side log fix: one atomic os.write per receipt record (<=3968 bytes) on a private dup(1), pid/seq/sha identity, fail-visible receipt_path, every emit site through the one writer, receipt_lines grader (missing_seq, bad_sha, incomplete, unsafe_path), frozen parsers still read every part, 4 writers + noise on one pipe with 0 bad (CPU; leaves in alpha_next_receipts_fixture_leaves.json)'),
                     ('test_alpha_next_receipts_mutants.py', 'Alpha-next receipt writer mutants: two writes per record, budget above PIPE_BUF, part budget left at 12000, seq not advanced on k=v records, sha over the part, silent short write each fail test_alpha_next_receipts; the unmutated copy passes (CPU)'),
                     ('test_e3_sparse_copy_waves.py', 'E3 sparse descriptor overflow: bounded copy waves, queried credit, original attention boundaries and unpublished cancellation (CPU)'),
                     ('test_vmid_job12.py', 'VM<vm-id> native sparse operation: real generated union and attention, delayed selection and cancellation (CPU)'),
                     ('test_vmid_job12b.py', 'VM<vm-id> sparse qualification: inspectable pinned generated Triton module and bounded exception evidence (CPU)'),
                     ('test_vmid_job12c.py', 'VM<vm-id> sparse qualification: preserved pinned JIT decorators and actual wrapper grid-launch protocol (CPU)'),
                     ('test_e3_teacher_decode.py', 'E3 teacher forcing: private fixed cohort, pinned sampler, unchanged settings and temporary ownership (CPU only)'),
                     ('test_e3_teacher_decode_rule4.py', 'E3 teacher forcing rule v4: frozen control-envelope allowances, symmetric metrics, tie order, calibration pinning and immutable evidence (CPU only)'),
                     ('test_e3_route_invariant.py', 'E3 route invariants: deduplicated prompt bytes, mutation failures and native graph definition serialization (CPU only)'),
                     ('test_e3_launch_diagnostic.py', 'E3 launch diagnostic: exact config fields, aliases, per-node counts and private pointer redaction (CPU only)'),
                     ('test_e3_probe_session.py', 'E3 W1 controller composition: real client validator, awaited pause, all-rank receipts, stale proof refusal (CPU only)'),
                     ('test_e3_probe_rpc.py', 'E3 diagnostic HTTP transport: scoped auth, loopback, fixed paths, bounded response, no retries/redirects (CPU only)'),
                     ('test_e3_probe_control.py', 'E3 diagnostic control: actual awaited pause/ordered RPC, all-rank barriers, scoped auth, source inverse and fail-closed transitions (CPU only)'),
                     ('test_e3_decode_probe.py', 'E3 decode-only diagnostic probe: flag conflict before allocation, evidence gate, typed no-prefill routing/invalidation, split bank pool/bytes, drained transitions/quarantine, shutdown order and default-off bytes (CPU only, fakes)'),
                     ('test_e3_finalize_package.py', 'E3 native final package: immutable source closure, exact library dependency/source hashes, no GPU qualification (CPU only)'),
                     ('test_e3_native_readiness.py', 'Native implementation/build closure: immutable source and VM build pins, actual finalizer, payload mutation refusals, no environment bypass or hardware qualification (CPU only)'),
                     ('test_e3_private_auth.py', 'E3 private auth: existing-key file, no value-bearing argv, strict immutable private file and source declaration (CPU only)'),
                     ('test_e3_q17b_clients.py', 'Q17b composed clients: one controller owner, observer placement and stopped-before-W1 (CPU only)'),
                     ('test_e3_q17b_full_queue.py', 'Q17b full queue: declared startup, actual clients aggregate and owned cleanup (CPU fake commands)'),
                     ('test_e3_overlap_performance.py', 'Q18a real overlap: frozen target/denominator policy, accepted segment proofs, serialized refusal, invalid-interval retention and reviewed readiness (CPU fake SSE/controller)'),
                     ('test_e3_overlap_performance_queue.py', 'Q18a overlap queue: armable refusal, all-gate gating, retained FAIL/INCONCLUSIVE, crash/stop cleanup, exact restore and no production start (CPU fake commands)'),
                     ('test_e3_startup_grade.py', 'E3 startup/capture grade: frozen FAIL-over-INCONCLUSIVE verdict, requested 44 recorded apart from the measured split, no replay or speed claim (CPU only)'),
                     ('test_e3_startup_queue.py', 'E3 startup/capture queue: offline fake-command lifecycle, owned cleanup on every failure, no production start and no printed auth (CPU only)'),
                     ('test_e3_graph_audit.py', 'E3 raw retained-graph audit: exact descriptor inventory, expected green owners, bounded native node walk, FAIL-over-unknown verdicts and stub-CUDA compile (CPU only)'),
                     ('test_vmid_job1.py', 'VM<vm-id> tiny graph diagnostic: exact multiplicity, resource ownership and fail-closed receipts (CPU only)'),
                     ('test_vmid_job2.py', 'VM<vm-id> mixed nodes/capacity: exact counts, copy directions, explicit child limitation and pinned header syntax (CPU only)'),
                     ('test_vmid_job3.py', 'VM<vm-id> sparse DMA: compact-row mapping, four-page pieces, whole-plan fragmentation refusal and exact byte coverage (CPU only)'),
                     ('test_vmid_job4.py', 'VM<vm-id> actual QSA sparse DMA: pinned gather-only substitution, exact outputs and cancellation/reuse ownership (CPU only)'),
                     ('test_vmid_job5.py', 'VM<vm-id> bank-scale observer: seven graph inventories, exact per-node windows and explicit capacity-loss failures (CPU only)'),
                     ('test_vmid_job5b.py', 'VM<vm-id> bank census progress: real phase with exclusive writer, all 30 snapshots, refusal without overwrite (CPU fakes)'),
                     ('test_vmid_job6.py', 'VM<vm-id> context isolation: real native node census, immutable replay subwindows, negative guard and bounded queue (CPU only)'),
                     ('test_e3_observer_runtime.py', 'E3 replay observer integration: pre-context start, actual worker drain, one scoped window and stopped-before-performance (CPU only)'),
                     ('test_e3_replay_observer.py', 'E3 actual replay context observer: bounded CUPTI activity collection, real green-context binding, clone/update mapping and FAIL-over-missing grading (CPU only, fake CUPTI)'),
                     ('test_s5_oracle_queue.py', 'S5 source oracle: strict subset controls, per-op diagnostics, cuBLAS binding (CPU)'),
                     ('test_s5_substitution.py', 'S5 exact host/resident trajectory oracle and payload-free differences (CPU)'),
                     ('test_s5_tail.py', 'S5 published tickets, workload-tail evidence and threaded eager-forward race (CPU)'),
                     ('test_s5_exit_cleanup.py', 'S5 pinned cleanup ordering and qualified measurement salvage (CPU)'),
                     ('test_s5_resident.py', 'S5 resident: source attribution, memory ledger, PLE buffer ownership (CPU)'),
                     ('test_s5_value.py', 'S5 value: matched solo shares, soft targets, spike counts, no-op scope'),
                     ('test_s5_variants.py', 'S5 ladder: worst-rank frontier, fenced controller and queue scope (CPU)'),
                     ('test_s5_onepair.py', 'S5 one-pair diagnostics: strict ranks, node audit, isolated queue (CPU)'),
                     ('test_morning_summary.py', 'morning report: scoped verdicts, per-stream metrics, cleanup and no payloads (CPU)'),
                     ('test_s5_followup.py', 'S5 follow-up: strict liveness, hang phases and queue admission (CPU)'),
                     ('test_vision_plan.py', 'remote vision: bytes, topology budget and CPU-only probe planning'),
                     ('test_cost_budget.py', 'cost budget: proxy fit decisions and generated scheduler hook (CPU)'),
                     ('test_sparse_stage.py', 'S6 sorted compaction, staged reduction oracle and verification coverage (CPU)'),
                     ('test_identity.py', 'engine identity: mocked Docker and content/provenance checks'),
                     ('test_identity_declare.py', 'source-only identity declaration: inert launcher and exact setcfg'),
                     ('test_round7_review.py', 'round 7 review: offline q8/decode_ab counterexamples'),
                     ('test_agent_replay.py', 'agent_replay v2: cut-off streams, full reuse'),
                     ('test_e3_w1_decode_driver.py', 'E3 W1 decode-only driver: live-cohort window after controller receipt, early-EOS invalidation, leave-before-drain, retained invalid repeats (loopback fakes)'),
                     ('test_e3_w1_decode_driver_ordered.py', 'Alpha-next PLAN 5a: load-robust twin of the frozen out-of-order open (swapped in by alpha_next_robust_swap): gated fake server opens streams 1, 2, 0 on the driver\'s own opened events; rc 0, advance order [1, 2, 0], all three opened (loopback fakes; alpha_next_robust fixture leaves)'),
                     ('test_e3_decode_correctness.py', 'E3 W1 short live-cohort correctness: full/split/full exact token oracle, paused-route lifecycle and control-vs-split verdicts (loopback fakes)'),
                     ('test_prefix_capacity.py', 'prefix capacity: source shapes and conditional LRU model'),
                     ('test_pp_grouping.py', 'PP grouping, pool stats, warm/cold gates: offline checks')]:
    # Step B / C0: a module holding a declared clock-bound frozen window test runs through step_b_clock_bound, which leaves out
    # only those ids (CLOCK_BOUND_FROZEN, each with its reason; a superset of C0's clock_bound list); the test files are unchanged.
    # Alpha-next PLAN 5: a module holding a declared load-flaky frozen method runs through alpha_next_robust_swap, which leaves
    # out only that id (ROBUST_SWAP, with its reason and its twin; it also applies CLOCK_BOUND_FROZEN); the twin is its own row.
    command = ([sys.executable, os.path.join(HERE, 'stage1_step_robust_swap.py'), '--base-receipt', stage1_step_base_receipt(), suite] if suite[:-3] in STAGE1_STEP_SWAP_MODULES
               else [sys.executable, os.path.join(HERE, 'attn_meta_robust_swap.py'), '--base-receipt', attn_meta_base_receipt(), suite]
               if suite[:-3] in ATTN_META_SWAP_MODULES
               else [sys.executable, os.path.join(HERE, 'alpha_next_robust_swap.py'), suite] if suite[:-3] in ROBUST_SWAP_MODULES
               else [sys.executable, os.path.join(HERE, 'step_b_clock_bound.py'), suite] if suite[:-3] in STEP_B_CLOCK_BOUND_MODULES
               else [sys.executable, os.path.join(HERE, suite)])
    identity_test = subprocess.run(command, capture_output=True, text=True,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    if identity_test.returncode:
        print(identity_test.stdout + identity_test.stderr)
    check(label, audit_suite(suite, identity_test), 0)

pp_release_test = subprocess.run([sys.executable, os.path.join(HERE, 'test_pp_release.py')],
                                 capture_output=True, text=True)
if pp_release_test.returncode:
    print(pp_release_test.stdout + pp_release_test.stderr)
check('PP release: strict coverage, handshake and canary regressions (offline)', audit_suite('test_pp_release.py', pp_release_test), 0)
canary_relay_test = subprocess.run([sys.executable, os.path.join(HERE, 'test_canary_relay.py')],
                                   capture_output=True, text=True)
if canary_relay_test.returncode:
    print(canary_relay_test.stdout + canary_relay_test.stderr)
check('Canary relay: unbuffered SSE, unchanged requests, terminal counting (loopback fake upstream)', audit_suite('test_canary_relay.py', canary_relay_test), 0)
print('selftest coverage: ' + json.dumps(dict(suites=len(unittest_observations), skipped=sum(r['skipped'] for r in unittest_observations)), sort_keys=True))
print(f"selftest: {sum(results)}/{len(results)} cases as required")
sys.exit(0 if all(results) else 1)
