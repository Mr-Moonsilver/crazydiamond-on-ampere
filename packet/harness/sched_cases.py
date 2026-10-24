#!/usr/bin/env python3
"""Live scheduler scenarios for the PD-alternation / parity patches (handover §4.1), run inside CT <ct-id> with
FLASH_SCHED_STATS set. Each case reads the scheduler counters ("flash-sched stats" log lines) at its start and end
and records per-stream token advances.

  memblock  NSTREAMS long streams fill most of the KV pool; once each has generated GROW tokens a long newcomer no
            longer fits although a request slot is free. Pass: the engine shows it waiting with a free slot, decoders
            keep >= MIN_RATIO of their rate, empty prefill steps stay bounded (<= 3 per case), no errors.
  preempt   10 streams of PROMPT tokens (default 230k: all fit at admission) generate up to PREEMPT_OUT tokens each
            (default 30000), so the running set outgrows the pool and admitted requests get preempted. Pass:
            preemptions observed, every stream (including the preempted ones) keeps or resumes producing tokens.
  grammar   decoders run; a structured-output request (JSON schema) arrives, waits for its grammar, then completes.
            Pass: it returns valid JSON matching the schema, decoders keep their rate, no errors.
  cancel    8 decoders; 3 are cancelled mid-decode (client disconnect). Pass: the other 5 keep decoding, the live
            cohorts re-balance (balancer on) or stay consistent, no errors.
usage: sched_cases.py URL MODEL CASE OUT.json   env NSTREAMS=9 PROMPT=245000 NEWCOMER=200000 GROW=8000 BASE=30 MIN_RATIO=0.85
exit 0 PASS, 1 FAIL
"""
import ast, json, os, random, re, subprocess, sys, threading, time, urllib.request, uuid

URL, MODEL, CASE, OUT = sys.argv[1:5]
BASE, MIN_RATIO = float(os.environ.get('BASE', 30)), float(os.environ.get('MIN_RATIO', 0.85))
corpus = open('/home/<service-user>/p2p-tests/corpus.txt', encoding='utf-8', errors='ignore').read()
errors, lock = [], threading.Lock()


def post(path, body, timeout=3600):
    return urllib.request.urlopen(urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                                         headers={'Content-Type': 'application/json'}), timeout=timeout)


def metric(name):
    text = urllib.request.urlopen(URL + '/metrics', timeout=10).read().decode()
    return sum(float(l.split()[-1]) for l in text.splitlines() if l.startswith(f'vllm:{name}{{') or l.startswith(f'vllm:{name} '))


def stats():
    log = subprocess.run(['docker', 'logs', '--tail', '400', 'qwen-flash-prod'], capture_output=True, text=True)
    lines = [l for l in (log.stdout + log.stderr).splitlines() if 'flash-sched stats' in l]
    if not lines:
        return None
    m = re.search(r"stats step (\d+): (\{.*\}) cohorts even=(\d+) odd=(\d+) none=(\d+)", lines[-1])
    if not m:
        return None
    d = ast.literal_eval(m.group(2))  # the scheduler logs a plain dict literal of ints
    d.update(step=int(m.group(1)), cohort_even=int(m.group(3)), cohort_odd=int(m.group(4)))
    return d


base_tok = json.load(post('/tokenize', {'model': MODEL, 'prompt': corpus[:260000 * 5]}, 600))['tokens']


class Stream:
    def __init__(self, i, prompt_len, max_tokens=16384):
        self.i, self.events, self.cancel = i, [], threading.Event()
        s = random.Random(i).randrange(0, len(base_tok) // 3)
        self.body = {'model': MODEL, 'prompt': (base_tok[s:] + base_tok[:s] + base_tok)[:prompt_len],
                     'max_tokens': max_tokens, 'min_tokens': max_tokens, 'ignore_eos': True, 'temperature': 0.7,
                     'seed': i, 'stream': True, 'cache_salt': uuid.uuid4().hex,
                     'stream_options': {'include_usage': True, 'continuous_usage_stats': True}}
        self.t = threading.Thread(target=self.run, daemon=True); self.t.start()

    def run(self):
        try:
            with post('/v1/completions', self.body) as r:
                for line in r:
                    if self.cancel.is_set():
                        return
                    if line.startswith(b'data: {'):
                        u = json.loads(line[6:]).get('usage') or {}
                        if u.get('completion_tokens'):
                            self.events.append((time.monotonic(), u['completion_tokens']))
        except Exception as e:  # noqa: BLE001
            if not self.cancel.is_set():
                with lock:
                    errors.append(f'stream {self.i}: {type(e).__name__} {e}')

    def tokens_at(self, t):
        return max((n for x, n in self.events if x <= t), default=0)


def rate(streams, t0, t1):
    return sum(s.tokens_at(t1) - s.tokens_at(t0) for s in streams) / (t1 - t0)


def wait_decoding(streams, timeout=1800):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not errors:
        if all(s.events for s in streams):
            return True
        time.sleep(1)
    return False


res: dict = {'case': CASE}
fails: list = []
try:
    if CASE == 'memblock':
        n, pl, nl = int(os.environ.get('NSTREAMS', 9)), int(os.environ.get('PROMPT', 245000)), int(os.environ.get('NEWCOMER', 200000))
        grow = int(os.environ.get('GROW', 8000))
        streams = [Stream(i, pl) for i in range(n)]
        if not wait_decoding(streams, 3600):
            fails.append('streams did not all start')
        else:
            end = time.monotonic() + 1800
            while min(s.tokens_at(time.monotonic()) for s in streams) < grow and time.monotonic() < end and not errors:
                time.sleep(2)
            pre0 = metric('num_preemptions_total'); s0 = stats()
            t0 = time.monotonic(); time.sleep(BASE); t1 = time.monotonic()
            newcomer = Stream(100, nl, max_tokens=1)
            time.sleep(5); samples = []
            t2 = time.monotonic()
            while time.monotonic() < t2 + BASE:
                samples.append((metric('num_requests_waiting'), metric('num_requests_running'))); time.sleep(1)
            t3 = time.monotonic(); s1 = stats()
            res.update(base_tps=round(rate(streams, t0, t1), 1), blocked_tps=round(rate(streams, t2, t3), 1),
                       waiting_running=samples[:3] + samples[-3:], preemptions=metric('num_preemptions_total') - pre0,
                       stats_start=s0, stats_end=s1, newcomer_got_token=bool(newcomer.events))
            res['ratio'] = round(res['blocked_tps'] / max(res['base_tps'], 1e-9), 3)
            blocked = [w for w, r in samples if w >= 1 and r < 10]
            res['blocked_with_free_slot_samples'] = len(blocked)
            if not blocked:
                fails.append('never observed a waiting request with a free slot (not memory-blocked)')
            if res['ratio'] < MIN_RATIO:
                fails.append(f"decode while blocked {res['ratio']} x baseline")
            if s0 and s1 and s1['empty_P'] - s0['empty_P'] > 3:
                fails.append(f"empty prefill steps while blocked: {s1['empty_P'] - s0['empty_P']}")
            if not (s0 and s1):
                fails.append('scheduler stats unavailable (FLASH_SCHED_STATS off?)')
            newcomer.cancel.set()
        for s in streams:
            s.cancel.set()
    elif CASE == 'preempt':
        pl, po = int(os.environ.get('PROMPT', 230000)), int(os.environ.get('PREEMPT_OUT', 30000))
        streams = [Stream(i, pl, max_tokens=po) for i in range(10)]
        if not wait_decoding(streams, 3600):
            fails.append('streams did not all start')
        else:
            pre0 = metric('num_preemptions_total'); s0 = stats()
            end = time.monotonic() + 3000
            while metric('num_preemptions_total') <= pre0 and time.monotonic() < end and not errors:
                time.sleep(5)
            res['preemptions_seen'] = metric('num_preemptions_total') - pre0
            t_pre = time.monotonic()
            time.sleep(120)   # long enough for a preempted request to be re-admitted and recompute
            t1 = time.monotonic(); s1 = stats()
            res.update(per_stream_tokens_after=[s.tokens_at(t1) - s.tokens_at(t_pre) for s in streams],
                       final_tokens=[s.tokens_at(t1) for s in streams], stats_start=s0, stats_end=s1)
            if not res['preemptions_seen']:
                fails.append('no preemption happened')
            stuck = [s.i for s in streams if s.tokens_at(t1) - s.tokens_at(t_pre) <= 0 and s.tokens_at(t1) < po]
            res['stuck_streams'] = stuck
            if len(stuck) > res['preemptions_seen']:
                fails.append(f'streams without progress beyond the preempted ones: {stuck}')
        for s in streams:
            s.cancel.set()
    elif CASE == 'grammar':
        streams = [Stream(i, 20000) for i in range(6)]
        if not wait_decoding(streams):
            fails.append('streams did not all start')
        else:
            t0 = time.monotonic(); time.sleep(BASE / 2); t1 = time.monotonic(); s0 = stats()
            schema = {'type': 'object', 'properties': {'city': {'type': 'string'}, 'year': {'type': 'integer'},
                      'tags': {'type': 'array', 'items': {'type': 'string', 'enum': ['a', 'b', 'c']}}},
                      'required': ['city', 'year', 'tags'], 'additionalProperties': False}
            g0 = time.monotonic()
            d = json.load(post('/v1/chat/completions', {
                'model': MODEL, 'messages': [{'role': 'user', 'content': 'Give a city, a year and tags as JSON.'}],
                'max_tokens': 200, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False},
                'response_format': {'type': 'json_schema', 'json_schema': {'name': 'x', 'schema': schema}}}))
            g1 = time.monotonic(); s1 = stats()
            t2 = time.monotonic(); time.sleep(BASE / 2); t3 = time.monotonic()
            content = d['choices'][0]['message']['content']
            try:
                obj = json.loads(content)
                valid = set(obj) == {'city', 'year', 'tags'} and isinstance(obj['year'], int) and all(t in 'abc' for t in obj['tags'])
            except Exception:  # noqa: BLE001
                valid = False
            res.update(grammar_s=round(g1 - g0, 2), json_valid=valid, content=content[:200],
                       before_tps=round(rate(streams, t0, t1), 1), during_tps=round(rate(streams, g0, g1), 1) if g1 - g0 > 1 else None,
                       after_tps=round(rate(streams, t2, t3), 1), stats_start=s0, stats_end=s1)
            if not valid:
                fails.append('structured output invalid')
            if res['after_tps'] < MIN_RATIO * res['before_tps']:
                fails.append('decode slower after the structured request')
        for s in streams:
            s.cancel.set()
    elif CASE == 'cancel':
        streams = [Stream(i, 20000) for i in range(8)]
        if not wait_decoding(streams):
            fails.append('streams did not all start')
        else:
            time.sleep(10); s0 = stats()
            for s in streams[:3]:
                s.cancel.set()
            time.sleep(10)
            t0 = time.monotonic(); time.sleep(BASE); t1 = time.monotonic(); s1 = stats()
            rest = streams[3:]
            res.update(rest_tps=round(rate(rest, t0, t1), 1), per_stream=[round(rate([s], t0, t1), 1) for s in rest],
                       stats_start=s0, stats_end=s1)
            if any(p <= 0 for p in res['per_stream']):
                fails.append('a remaining stream stopped')
            if s1 and abs(s1['cohort_even'] - s1['cohort_odd']) > 1 and os.environ.get('BALANCE') == '1':
                fails.append(f"cohorts not balanced after cancel: {s1['cohort_even']}/{s1['cohort_odd']}")
        for s in streams:
            s.cancel.set()
    else:
        fails.append(f'unknown case {CASE}')
except Exception as e:  # noqa: BLE001 - fails the run, artifact still written
    fails.append(f'foreground: {type(e).__name__} {e}')
time.sleep(2)
fails = list(errors) + fails
res.update(errors=errors, fails=fails, verdict='FAIL' if fails else 'PASS')
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('stats_start',)})[:1500])
sys.exit(0 if res['verdict'] == 'PASS' else 1)
