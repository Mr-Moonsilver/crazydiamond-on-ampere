#!/usr/bin/env python3
"""Offline self-test of the harness verdict logic: every known false-pass case must fail. No inference, no SSH.

Covers canary_grade.py (review fixture: foreign text + correct final line + rc=124; missing output; rc=1; wrong note;
malformed output; duplicate ids), leakgrade.py (foreign text in a retry; retries that did not recreate the original
cache path -- the J-inside case; recreated path all right / all wrong; missing results; engine restart) and
stall_trace2.py --recompute (good run; missing newcomer token; stopped stream; incomplete run).
usage: selftest.py [REVIEW_FIXTURE_DIR]   exit 0 only if every case behaves as required
"""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import canary_grade, leakgrade  # noqa: E402

results = []


def check(name, got, want):
    ok = got == want
    results.append(ok)
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")


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
def raw_run(newcomer=True, stop_stream=False, complete=True):
    ev = []
    for i in range(2):
        last = 30.0 if (stop_stream and i == 0) else 60.0
        ev.append([[t / 10, int(t - 9)] for t in range(10, int(last * 10)) if t % 2 == 0])
    marks = {'b0': 5.0, 'b1': 15.0, 'p0': 20.0, 'p1': 40.0}
    if complete:
        marks['a1'] = 50.0
    return {'raw': {'label': 'fixture', 'streams': 2, 'stream_prompt': 1000, 'long_prompt': 20000, 'wall_anchor': [0, 0],
                    'errors': [], 'marks': marks, 'events': ev, 'newcomer_token': newcomer, 'engine_newcomer': {}}}


for name, kw, want in [('good run', {}, 0), ('missing newcomer token', {'newcomer': False}, 1),
                       ('stopped stream', {'stop_stream': True}, 1), ('incomplete run', {'complete': False}, 1)]:
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

print(f"selftest: {sum(results)}/{len(results)} cases as required")
sys.exit(0 if all(results) else 1)
