#!/usr/bin/env python3
"""Offline self-test of the harness verdict logic: every known false-pass case must fail. No inference, no SSH.

Covers canary_grade.py (review fixture: foreign text + correct final line + rc=124; missing output; rc=1; wrong note;
malformed output; duplicate ids), leakgrade.py (foreign text in a retry; retries that did not recreate the original
cache path -- the J-inside case; recreated path all right / all wrong; missing results; engine restart) and
stall_trace2.py --recompute (good run; missing newcomer token; stopped stream; incomplete run).
usage: selftest.py [REVIEW_FIXTURE_DIR]   exit 0 only if every case behaves as required
"""
import hashlib, json, os, re, subprocess, sys, tempfile

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
check('canary: requested transcript missing (review fixture)', canary_grade.grade(d, tempfile.mkdtemp())['verdict'], 'FAIL')
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
                     ('test_e3_qsa_driver.py', 'E3 QSA layer driver: additive attention pre/post on the D2-composed QSA source, producer -> one-bucket generator -> tail phases, pause/pending (CPU only)'),
                     ('test_e3_native_startup.py', 'E3 worker-native startup: split bank + lane communicators + PLE scratch composition, graph builder, transition RPC (CPU only, fakes)'),
                     ('test_e3_native_imports.py', 'Native8 engine import ownership: E2/E3 package rewriting, actual pure provider path, preserved GPU-state guard and isolated exact namespace with no builder path (CPU source only; real image proof separately required)'),
                     ('test_e3_row1_exactness.py', 'E3-owned row1 backend: frozen E2 counterexample, seven-method source provenance and explicit worker provider binding (CPU)'),
                     ('test_e3_stock_api.py', 'Native7 stock API contract: exhaustive source roles, exact native6 frontend counterexample, signature refusals and pristine-vs-derived pins (AST only; actual real-class imports separately required)'),
                     ('test_e3_executor_threads.py', 'Native6 process-owned executor: actual pinned ready-response/RPC methods on IO/monitor threads, FIFO serialization, shared futures, fork refusal and conservative close (CPU MQ leaves)'),
                     ('test_e3_stock_rpc_credits.py', 'Native16 stock batch queue: claim earlier ordinary RPC futures without consuming native credits, unchanged capacity and cached failures (CPU MQ/output leaves)'),
                     ('test_e3_native_selection_asymmetry.py', 'Native17 known TP selection-copy ordering counterexample: unmodified source stack over explicitly synthetic CPU tensor/event/model leaves, not box evidence'),
                     ('test_e3_native_schedule_profile.py', 'Native18 canonical selection boundary, queried TP profile agreement, exact retries and coordinated refusal/drain over explicit CPU device/model leaves; not box evidence'),
                     ('test_e3_native_pp_stop_order.py', 'Native19 deferred PP receive and typed stop obligation: matching-send-gated events, partial acceptance, clean stage-zero refusal/abort (CPU leaves; not box evidence)'),
                     ('test_e3_native_profile_observability.py', 'Native19 receipt-only peer profiles, worker host query time and actual stock-step bypass counts (CPU leaves)'),
                     ('test_e3_native_speculation_gate.py', 'Native19 startup log gate: actual speculative config must be None, with missing or enabled evidence refused (CPU)'),
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
                     ('test_e3_decode_correctness.py', 'E3 W1 short live-cohort correctness: full/split/full exact token oracle, paused-route lifecycle and control-vs-split verdicts (loopback fakes)'),
                     ('test_prefix_capacity.py', 'prefix capacity: source shapes and conditional LRU model'),
                     ('test_pp_grouping.py', 'PP grouping, pool stats, warm/cold gates: offline checks')]:
    identity_test = subprocess.run([sys.executable, os.path.join(HERE, suite)], capture_output=True, text=True,
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
