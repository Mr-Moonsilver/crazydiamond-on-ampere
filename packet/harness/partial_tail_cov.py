#!/usr/bin/env python3
"""Partial-tail coverage diagnostic, with unchanged leakgrade correctness/80% oracle.

A fixed one-token refresh is attempted before each graded consumer. It is NOT
proof of registration, physical eviction or CoW. Cache-hit observations describe
lookup outcomes; only a real engine run establishes coverage. Lower-pressure
qualification and original stress retain separate verdicts, including all misses.
No payloads or synthetic identifiers are persisted. Cancellation counts mean
client disconnects submitted, not proof of a device-side cancellation boundary.
The external queue timeout remains the hard process bound (HTTP timeouts bound
individual I/O, not a slow-trickling response's total duration).

CLI: run URL MODEL SESSIONS ROUNDS [B] OUT.json
     grade WORKLOAD.json PRESSURE.json INTERVAL.json OUT.json
     combine QUAL_GRADE.json STRESS_LEAK.json STRESS_LIFETIME.json OUT.json
"""
import http.client
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leakgrade  # noqa: E402

TOOL = 'partial_tail_cov'
VERSION = 1
MIN_PARTIAL = 0.8      # unchanged leakgrade default; re-exported by grade_artifact
NAMES = ['AMBER', 'COBALT', 'JUNIPER', 'OSPREY', 'TUNDRA', 'VIOLET', 'MAGPIE', 'QUARTZ', 'SIENNA', 'FALCON',
         'HAZEL', 'RIVET']
QUESTION_TEXT = ("\n\nQuestion: Earlier in this document an archive note gave a vault code. What exactly was that "
                 "vault code? Answer: The vault code is")
QUESTION2_TEXT = ("\n\nQuestion: Earlier in this document a registry note gave a locker number. What exactly was that "
                  "locker number? Answer: The locker number is")
DEFAULT_CORPUS = '/home/<service-user>/p2p-tests/corpus.txt'


class BudgetExhausted(RuntimeError):
    """The bounded workload deadline was reached before a request could be issued."""


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_bool(name, default):
    return os.environ.get(name, str(default)).lower() in ('1', 'true', 'yes')


class HttpEngine:
    """Minimal client for the engine endpoints the workload uses; stateless and thread-safe."""

    def __init__(self, url, model, timeout=900):
        self.url, self.model, self.timeout = url.rstrip('/'), model, float(timeout)
        self.deadline = None

    def set_deadline(self, deadline):
        self.deadline = deadline

    def bounded_timeout(self, timeout=None):
        value = min(self.timeout, timeout or self.timeout)
        if self.deadline is not None:
            left = self.deadline - time.monotonic()
            if left <= 0:
                raise BudgetExhausted('deadline reached')
            value = min(value, left)
        return value

    # -- plumbing -------------------------------------------------------------------------------------------------
    def _post(self, path, body, timeout=None):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.bounded_timeout(timeout)) as response:
            return json.load(response)

    def tokenize(self, text):
        return self._post('/tokenize', {'model': self.model, 'prompt': text,
                                        'add_special_tokens': False})['tokens']

    def detokenize(self, tokens):
        return self._post('/detokenize', {'model': self.model, 'tokens': tokens})['prompt']

    def complete(self, prompt, max_tokens, salt=None, ignore_eos=False, timeout=None):
        body = {'model': self.model, 'prompt': prompt, 'max_tokens': int(max_tokens), 'temperature': 0,
                'return_token_ids': True}
        if ignore_eos:
            body.update(ignore_eos=True, min_tokens=int(max_tokens))
        if salt:
            body['cache_salt'] = salt
        d = self._post('/v1/completions', body, timeout=timeout() if callable(timeout) else timeout)
        choice = d['choices'][0]
        text = choice.get('text') or ''
        ids = choice.get('token_ids')
        if not isinstance(ids, list) or not all(type(x) is int for x in ids):
            raise ValueError('exact generated token IDs required')
        details = (d.get('usage') or {}).get('prompt_tokens_details') or {}
        return {'text': text, 'token_ids': list(ids), 'cached': details.get('cached_tokens')}

    def abort_after(self, prompt, seconds, salt=None):
        # streamed request dropped after `seconds`, without waiting for headers (same shape as leak_partial_tail.py)
        u = urllib.parse.urlparse(self.url)
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=self.bounded_timeout())
        body = {'model': self.model, 'prompt': prompt, 'max_tokens': 12, 'temperature': 0, 'stream': True}
        if salt:
            body['cache_salt'] = salt
        conn.request('POST', '/v1/completions', json.dumps(body), {'Content-Type': 'application/json'})
        time.sleep(seconds)
        conn.close()

    def identity(self):
        container = os.environ.get('ENGINE_CONTAINER', 'qwen-flash-prod')
        try:
            return subprocess.run(['docker', 'inspect', container, '--format',
                                   '{{.Id}} {{.State.StartedAt}}'],
                                  capture_output=True, text=True, timeout=30).stdout.strip() or None
        except Exception:  # noqa: BLE001 - identity unknown is recorded as None and fails the run verdict
            return None


# -- pure classification ------------------------------------------------------------------------------------------

def classify_cached(cached, block):
    """What the lookup showed: a partial boundary, a full-block boundary, or nothing."""
    if cached is None:
        return 'unknown'
    if cached <= 0:
        return 'none'
    return 'partial' if cached % block != 0 else 'full'


def next_full_after(boundary, block):
    return (boundary // block + 1) * block if boundary else block


def construction_events(prompt_len, generated, unit, block):
    """Eligibility facts fixed by the workload shape (before any eviction can intervene)."""
    boundary = prompt_len // unit * unit
    if boundary == 0:
        return {'boundary': 0, 'expected_partial': False, 'promoted': False, 'reason': 'no hash boundary'}
    if boundary % block == 0:
        return {'boundary': boundary, 'expected_partial': False, 'promoted': False, 'reason': 'boundary is block-aligned'}
    promoted = prompt_len + generated >= next_full_after(boundary, block)
    return {'boundary': boundary, 'expected_partial': True, 'promoted': promoted,
            'reason': 'generation crossed the full block boundary; the partial entry is promoted away' if promoted
            else ''}


def transition(expected_boundary, producer, refresh_cached, cached, block):
    """Observed refresh/consumer hit sequence, not physical eviction attribution."""
    if not producer['expected_partial']:
        return 'ineligible_boundary' if 'aligned' in producer['reason'] else 'ineligible_no_boundary'
    if producer['promoted']:
        return 'ineligible_promotion'
    observed = classify_cached(refresh_cached, block)
    got = classify_cached(cached, block)
    if got == 'partial':
        if expected_boundary is not None and cached == expected_boundary:
            return 'partial_at_expected'
        return 'partial_other_boundary'
    if got == 'full':
        return {'partial': 'full_page_only_after_partial_refresh',
                'full': 'full_page_only_after_full_refresh',
                'none': 'full_page_only_after_zero_refresh'}.get(observed, 'full_page_only_unknown')
    if got == 'none':
        return {'partial': 'zero_hit_after_partial_refresh',
                'full': 'zero_hit_after_full_refresh',
                'none': 'zero_hit_after_zero_refresh'}.get(observed, 'no_hit_unknown')
    return 'unknown_hit'


def summarize_cases(cases, spec):
    classes = {}
    transitions = {}
    hits = {'answer': set(), 'answer3': set()}
    for c in cases:
        classes[c.get('cached_class', 'unknown')] = classes.get(c.get('cached_class', 'unknown'), 0) + 1
        transitions[c.get('transition', 'unknown')] = transitions.get(c.get('transition', 'unknown'), 0) + 1
        if c.get('cached') is not None:
            hits.setdefault(c.get('kind'), set()).add(c['cached'])
    construction = {'ineligible_boundary': sum(1 for c in cases if c.get('transition') in
                                               ('ineligible_boundary', 'ineligible_no_boundary')),
                    'ineligible_promotion': sum(1 for c in cases if c.get('transition') == 'ineligible_promotion')}
    partial = sum(1 for c in cases if c.get('partial'))
    target = sum(1 for c in cases if c.get('partial') and
                 c['cached'] >= c['producer_boundary'] and
                 c['cached'] // spec['block'] == c['producer_boundary'] // spec['block'])
    return {'attempted_graded': len(cases),
            'expected_graded': spec['n'] * spec['rounds'] * 2,
            'eligible_cases': sum(1 for c in cases if c.get('eligible')),
            'partial_hits': partial, 'target_partial_hits': target,
            'partial_ratio': (partial / len(cases)) if cases else 0.0,
            'class_counts': classes,
            'transition_counts': transitions,
            'construction': construction,
            'cache_hit_values': {k: sorted(v) for k, v in hits.items()},
            'min_partial': MIN_PARTIAL}


# -- workload construction ----------------------------------------------------------------------------------------

def _codes_for_round(rnd, n, every_code, lockers):
    codes = []
    while len(codes) < n:
        code = f'{rnd.choice(NAMES)}-{rnd.randint(1000, 9999)}'
        if code in codes or any(code in cs for cs in every_code):
            continue
        while True:
            locker = f'L{rnd.randrange(16 ** 6):06x}'
            if locker not in lockers.values():
                lockers[code] = locker
                break
        codes.append(code)
    return codes


def build_inside(engine, corpus, shared, rd, i, code, locker, b, block, fact, fact2, unit=176):
    """Same shape as leak_partial_tail.py MODE=inside: unique body [BLOCK, B), facts inside the copied part."""
    off = random.Random(rd * 1000 + i).randrange(0, len(corpus) - 200000)
    filler = engine.tokenize(corpus[off:off + (b - block) * 8])
    fact_ids = engine.tokenize(f"\n\nArchive note: the vault code for this document is {code}. "
                               "(End of archive note.)\n\n")
    fact2_ids = engine.tokenize(f"\n\nRegistry note: the locker number for this document is {locker}. "
                                "(End of registry note.)\n\n")
    a, z = fact - block, fact2 - block
    body = (filler[:a] + fact_ids + filler[a:z] + fact2_ids + filler[z:])[:b - block]
    text = engine.detokenize(body)
    if len(body) != b - block or code not in text or locker not in text:
        raise ValueError('tokenizer round-trip did not preserve the fact at the requested boundary')
    tail = engine.tokenize('\n\nNow continue the text:\n')
    if not 0 < len(tail) < unit - 8:
        raise ValueError('short producer tail must fit generation margin')
    return list(shared[:block]) + body + tail


def retries_for(engine, prompt, hit, timeout, n=3):
    """Recreate the original cache path exactly as leak_partial_tail.py does; payloads stay in memory."""
    out = []
    for _ in range(n):
        salt = uuid.uuid4().hex
        if hit:
            engine.complete(prompt[:hit + 1], 1, salt=salt, ignore_eos=True, timeout=timeout() if callable(timeout) else timeout)
        warm = engine.complete(prompt, 12, salt=salt, ignore_eos=True, timeout=timeout() if callable(timeout) else timeout)
        cold = engine.complete(prompt, 12, salt=uuid.uuid4().hex, ignore_eos=True, timeout=timeout() if callable(timeout) else timeout)['text']
        out.append({'warm': warm['text'], 'warm_cached': warm.get('cached'), 'cold': cold})
    return out


def _grade_case(answer, cold, cached, own, foreign, prompt, expected_boundary, producer, refresh_cached,
                block, engine, timeout, kind, rd, i):
    first = leakgrade.first_verdict(answer, own, foreign, cold)
    settled = None
    if first == 'SUSPECT':
        settled = leakgrade.settle(retries_for(engine, prompt, cached or 0, timeout), own, foreign, cached)
    return {'round': rd, 'session': i, 'kind': kind,
            'first': first, 'settled': settled,
            'partial': bool(cached) and cached % block != 0,
            'cached': cached, 'expected_boundary': expected_boundary,
            'cached_class': classify_cached(cached, block),
            'refresh_cached': refresh_cached,
            'refresh_class': classify_cached(refresh_cached, block),
            'eligible': producer['expected_partial'] and not producer['promoted'],
            'producer_promoted': producer['promoted'],
            'producer_boundary': producer['boundary'],
            'transition': transition(expected_boundary, producer, refresh_cached, cached, block),
            'producer_len': producer['prompt_len'], 'producer_generated': producer['generated'],
            'consumer_len': len(prompt), 'answer_len': len(answer or ''), 'cold_len': len(cold or '')}


def run_workload(engine, corpus, spec, clock=time.monotonic, sleep=time.sleep):
    """Construct the cases and measure every graded lookup.  No payload is written to the artifact."""
    validate_spec(spec)
    deadline = clock() + spec['budget_s']
    if hasattr(engine, 'set_deadline'):
        engine.set_deadline(deadline)
    n, rounds, b = spec['n'], spec['rounds'], spec['b']
    block, gen = spec['block'], spec['gen']
    question = engine.tokenize(QUESTION_TEXT)
    question2 = engine.tokenize(QUESTION2_TEXT)
    corner = engine.tokenize('\n\n')
    started = time.time()
    lock = threading.Lock()
    results, errors, cancelled = [], [], []
    cancel_attempts = []
    every_code, lockers = [], {}
    plans = []
    # All identifiers are known before any generation (also catches future-round leaks).
    for rd in range(rounds):
        rnd = random.Random(spec['seed'] * 100003 + rd)
        shared_text = f'[round {rd} seed {spec["seed"]} cov {rnd.getrandbits(64):016x}]\n'
        off = rnd.randrange(0, len(corpus) - 200000)
        codes = _codes_for_round(rnd, n, every_code, lockers)
        every_code.append(codes)
        plans.append((shared_text, off, codes))
    run_nonce = uuid.uuid4().hex
    exhausted = threading.Event()
    wall_hits: dict = {'start': None, 'end': None}

    def remaining():
        left = deadline - clock()
        if left <= 0:
            raise BudgetExhausted(f'bounded workload budget {spec["budget_s"]}s exhausted')
        return max(0.05, min(spec['request_timeout'], left))

    def session(rd, i, shared, code):
        try:
            delay = spec['stagger'] * i
            if delay:
                sleep(delay)
            ns = f'{spec["cache_ns"]}-{rd}-{i}' if spec['cache_ns'] else f'cov-{run_nonce}-{rd}-{i}'
            # deterministic per-session budget (never the shared RNG under threads)
            budget = gen + random.Random(rd * 48611 + i).randrange(0, gen)
            p1 = build_inside(engine, corpus, shared, rd, i, code, lockers[code], b, block,
                              spec['fact'], spec['fact2'], spec['unit'])
            with lock:
                if wall_hits['start'] is None:
                    wall_hits['start'] = time.time()
            producer1 = engine.complete(p1, budget, salt=ns, ignore_eos=True, timeout=remaining())
            ids1, gen1 = producer1['token_ids'], producer1['text']
            # R1: fixed refresh attempt; registration/survival must be measured.
            r1 = engine.complete(p1, 1, salt=ns, ignore_eos=True, timeout=remaining()) \
                if spec['refresh'] else {'cached': None}
            p2 = p1 + list(ids1) + question
            cancelled_here = False
            if random.Random(rd * 7919 + i).random() < spec['cancel_frac']:
                with lock:
                    cancel_attempts.append((rd, i))
                engine.abort_after(p2, random.Random(rd * 104729 + i).uniform(0, 0.3), salt=ns)
                cancelled_here = True
            a2 = engine.complete(p2, 12, salt=ns, ignore_eos=True, timeout=remaining())
            # R2: fixed refresh at p2's boundary, with no best-of retries.
            r2 = engine.complete(p2, 1, salt=ns, ignore_eos=True, timeout=remaining()) \
                if spec['refresh'] else {'cached': None}
            p3 = p2 + list(a2['token_ids']) + corner + list(ids1[:200]) + question2
            a3 = engine.complete(p3, 12, salt=ns, ignore_eos=True, timeout=remaining())
            # Cold controls AFTER both graded requests: they must not evict the refreshed entry before it is read.
            cold2 = engine.complete(p2, 12, salt=f'cold-{ns}-{rd}-{i}', ignore_eos=True,
                                    timeout=remaining())['text']
            cold3 = engine.complete(p3, 12, salt=f'cold3-{ns}-{rd}-{i}', ignore_eos=True,
                                    timeout=remaining())['text']
            flat_codes = [c for cs in every_code for c in cs]
            foreign_codes = [c for c in flat_codes if c != code]
            foreign_lockers = [lockers[c] for c in flat_codes if lockers[c] != lockers[code]]
            c2 = _case(rd, i, 'answer', code, foreign_codes, a2['text'], cold2, a2.get('cached'),
                       p2, p1, len(ids1), r1.get('cached'), block, spec, engine, remaining)
            c3 = _case(rd, i, 'answer3', lockers[code], foreign_lockers, a3['text'], cold3,
                       a3.get('cached'), p3, p2, len(a2['token_ids']), r2.get('cached'), block, spec,
                       engine, remaining)
            # request 1's own continuation is generated text too: a foreign identifier there is a leak
            foreign = [x for x in flat_codes + [lockers[c] for c in flat_codes] if x not in (code, lockers[code])]
            bad = leakgrade.foreign_in(gen1, foreign)
            with lock:
                if bad:
                    errors.append(f'r{rd} s{i}: LEAK in request 1 output')
                if cancelled_here:
                    cancelled.append((rd, i))
                results.append({'round': rd, 'session': i, 'p1_len': len(p1), 'gen1_len': len(ids1),
                                'p2_len': len(p2), 'p3_len': len(p3), 'cases': [c2, c3]})
        except BudgetExhausted as e:  # noqa: PERF203 - one audit trail for a bounded phase
            exhausted.set()
            with lock:
                errors.append(f'r{rd} s{i}: budget exhausted')
        except Exception as e:  # noqa: BLE001 - counted as a failure, never hidden
            with lock:
                errors.append(f'r{rd} s{i}: {type(e).__name__}')

    identity_start = engine.identity()
    for rd, (shared_text, off, codes) in enumerate(plans):
        try:
            remaining()
        except BudgetExhausted:
            exhausted.set(); errors.append('budget exhausted before round'); break
        shared = engine.tokenize(shared_text + corpus[off:off + b * 6])[:b]
        if len(shared) != b:
            raise ValueError('shared prefix did not tokenize to B tokens')
        if spec['serial']:
            for i, code in enumerate(codes):
                session(rd, i, shared, code)
        else:
            threads = [threading.Thread(target=session, args=(rd, i, shared, code), daemon=True)
                       for i, code in enumerate(codes)]
            [t.start() for t in threads]
            [t.join() for t in threads]
    wall_hits['end'] = time.time()
    identity_end = engine.identity()
    cases = [c for r in sorted(results, key=lambda x: (x['round'], x['session'])) for c in r['cases']]
    expected = n * rounds * 2
    run = leakgrade.run_verdict(cases, expected, errors,
                                restarted=identity_start != identity_end or identity_start is None,
                                need_partial=spec['need_partial'], min_partial=MIN_PARTIAL)
    if spec['cancel_frac'] and not cancelled:
        run['inconclusive'].append('no request was cancelled')
        if run['verdict'] == 'PASS':
            run['verdict'] = 'INCONCLUSIVE'
    if exhausted.is_set() and not any(e.startswith(('r',)) and 'budget' in e for e in errors):
        errors.append('bounded workload budget exhausted')
    artifact = {'tool': TOOL, 'version': VERSION, 'mode': 'inside', 'spec': spec,
                'identity_start': identity_start, 'identity_end': identity_end,
                'work_start_wall': wall_hits['start'], 'work_end_wall': wall_hits['end'],
                'wall_seconds': round(wall_hits['end'] - started, 3) if wall_hits['end'] else None,
                'budget': {'sanctioned_seconds': spec['budget_s'], 'exhausted': exhausted.is_set()},
                'results': results, 'cases': cases, 'errors': errors,
                'cancelled': len(cancelled), 'cancellation_attempts': len(cancel_attempts),
                'run': run,
                'coverage': summarize_cases(cases, spec)}
    return artifact


def _case(rd, i, kind, own, foreign, answer, cold, cached, prompt, producer_prompt, produced, refresh_cached,
          block, spec, engine, remaining):
    producer = construction_events(len(producer_prompt), produced, spec['unit'], block)
    producer.update(prompt_len=len(producer_prompt), generated=produced)
    return _grade_case(answer, cold, cached, own, foreign, prompt, producer['boundary'], producer,
                       refresh_cached, block, engine, remaining, kind, rd, i)


# -- grading -------------------------------------------------------------------------------------------------------

def finite_number(x):
    return type(x) in (int, float) and math.isfinite(x)


def validate_spec(spec):
    for key in ('n', 'rounds', 'b', 'unit', 'block', 'gen'):
        if type(spec[key]) is not int or spec[key] <= 0:
            raise ValueError('positive integer required: ' + key)
    if spec['n'] > 100 or spec['rounds'] > 100:
        raise ValueError('workload exceeds bounded session/round limit')
    for key in ('budget_s', 'request_timeout'):
        if not finite_number(spec[key]) or spec[key] <= 0:
            raise ValueError('positive finite duration required')
    if not finite_number(spec['cancel_frac']) or not 0 <= spec['cancel_frac'] <= 1:
        raise ValueError('invalid cancellation fraction')
    if not finite_number(spec['stagger']) or spec['stagger'] < 0:
        raise ValueError('invalid stagger')
    if not (spec['need_partial'] is True and spec['refresh'] is True and spec['cold'] is True):
        raise ValueError('cold, refresh and partial oracle must remain enabled')
    if not (spec['block'] < spec['fact'] < spec['fact2'] < spec['b']):
        raise ValueError('facts must lie in the intended partial page')
    if spec['fact'] // spec['block'] != spec['b'] // spec['block']:
        raise ValueError('facts outside partial source page')
    if spec['b'] % spec['unit'] or spec['b'] % spec['block'] == 0:
        raise ValueError('invalid partial hash boundary')


def validated_cases(cases, spec):
    expected_keys = {(r, i, k) for r in range(spec['rounds']) for i in range(spec['n'])
                     for k in ('answer', 'answer3')}
    seen = set()
    for c in cases:
        key = (c['round'], c['session'], c['kind'])
        if key not in expected_keys or key in seen:
            raise ValueError('duplicate/foreign case key')
        seen.add(key)
        if c['first'] not in ('OK', 'MISS', 'LEAK', 'SUSPECT'):
            raise ValueError('invalid first verdict')
        if c['first'] != 'SUSPECT' and c.get('settled') is not None:
            raise ValueError('settlement overrides non-suspect verdict')
        if c['first'] == 'SUSPECT' and c.get('settled') not in ('LEAK', 'COPYFAIL', 'NOISE', 'INCONCLUSIVE'):
            raise ValueError('unsettled suspect')
        for k in ('producer_len', 'producer_generated', 'consumer_len'):
            if type(c[k]) is not int or c[k] <= 0:
                raise ValueError('invalid case geometry')
        for k in ('cached', 'refresh_cached'):
            x = c[k]
            limit = c['consumer_len'] if k == 'cached' else c['producer_len']
            if x is not None and (type(x) is not int or not 0 <= x <= limit):
                raise ValueError('invalid cached count')
        producer = construction_events(c['producer_len'], c['producer_generated'], spec['unit'], spec['block'])
        computed = dict(partial=bool(c['cached']) and c['cached'] % spec['block'] != 0,
                        cached_class=classify_cached(c['cached'], spec['block']),
                        refresh_class=classify_cached(c['refresh_cached'], spec['block']),
                        eligible=producer['expected_partial'] and not producer['promoted'],
                        producer_promoted=producer['promoted'], producer_boundary=producer['boundary'],
                        expected_boundary=producer['boundary'],
                        transition=transition(producer['boundary'], producer, c['refresh_cached'], c['cached'], spec['block']))
        if any(c.get(k) != v for k, v in computed.items()):
            raise ValueError('case coverage fields disagree with raw geometry')
    return cases


def grade_artifact(artifact, pressure=None, interval=None):
    """Re-derive the unchanged oracle verdict and add the coverage/bracket evidence.  Fail closed on bad input."""
    fails, missing = [], []
    try:
        if not isinstance(artifact, dict):
            raise TypeError('artifact is not an object')
        spec, cases = artifact['spec'], artifact['cases']
        validate_spec(spec)
        validated_cases(cases, spec)
        for key in ('n', 'rounds', 'b', 'unit', 'block', 'gen', 'need_partial', 'refresh', 'budget_s'):
            spec[key]
        if artifact.get('tool') != TOOL or artifact.get('version') != VERSION:
            raise ValueError('not a partial_tail_cov artifact')
        if spec['need_partial'] is not True:
            fails.append('NEED_PARTIAL must stay true for the partial-tail gate')
        if abs(spec.get('min_partial', MIN_PARTIAL) - MIN_PARTIAL) > 1e-12:
            fails.append(f'min_partial changed from {MIN_PARTIAL}')
        if spec.get('refresh') is not True:
            fails.append('REFRESH=1 is required: eligibility must be constructed, not assumed')
        if spec['b'] % spec['block'] == 0:
            fails.append('B is block-aligned: this workload can never register a partial tail')
        if spec['b'] % spec['unit'] != 0:
            fails.append('B is not a multiple of the hash unit')
        expected = spec['n'] * spec['rounds'] * 2
        oracle_cases = [{'first': c['first'], 'settled': c.get('settled'),
                         'partial': bool(c.get('partial'))} for c in cases]
        restarted = (not artifact.get('identity_start')
                     or artifact.get('identity_start') != artifact.get('identity_end'))
        run = leakgrade.run_verdict(oracle_cases, expected, artifact.get('errors') or [], restarted=restarted,
                                    need_partial=True, min_partial=MIN_PARTIAL)
        if spec.get('cancel_frac') and not artifact.get('cancelled'):
            run['inconclusive'].append('no request was cancelled')
            if run['verdict'] == 'PASS':
                run['verdict'] = 'INCONCLUSIVE'
        saved = (artifact.get('run') or {}).get('verdict')
        if saved != run['verdict']:
            fails.append(f'saved run verdict {saved!r} disagrees with the cases ({run["verdict"]})')
        coverage = summarize_cases(cases, spec)
        construction = coverage['construction']
        if coverage['target_partial_hits'] < MIN_PARTIAL * expected:
            missing.append('insufficient hits in the intended partial source page')
        if construction['ineligible_boundary']:
            fails.append(f'{construction["ineligible_boundary"]} cases had no partial boundary (workload defect)')
        if construction['ineligible_promotion']:
            fails.append(f'{construction["ineligible_promotion"]} producers crossed the full block and promoted '
                         'the partial entry away (workload defect)')
        for key in ('cancelled', 'cancellation_attempts'):
            if type(artifact.get(key)) is not int or not 0 <= artifact[key] <= spec['n'] * spec['rounds']:
                fails.append('invalid cancellation count')
        if artifact.get('cancelled', 0) > artifact.get('cancellation_attempts', 0):
            fails.append('completed disconnects exceed attempts')
        if artifact.get('budget', {}).get('exhausted'):
            missing.append('bounded workload budget exhausted before every answer was graded')
        fails += run['fails']
        missing += run['inconclusive']
        firsts: list = []
        lasts: list = []
        completed: list = []
        if pressure is None:
            missing.append('pressure evidence missing')
        else:
            if pressure.get('errors'):
                fails.append('pressure stream error')
            streams = pressure.get('streams', 0)
            firsts, lasts = pressure.get('first_tokens', []), pressure.get('last_tokens', [])
            completed = pressure.get('completed_requests', [])
            if type(streams) is not int or streams != 2 or len(firsts) != streams or len(lasts) != streams:
                fails.append('pressure stream evidence missing')
            elif not all(finite_number(x) and finite_number(y) for x, y in zip(firsts, lasts)):
                fails.append('pressure stream start/end tokens missing')
            if len(completed) != streams or not all(type(x) is int and x >= 1 for x in completed):
                missing.append('no completed pressure request on every stream')
            if (not isinstance(interval, list) or len(interval) != 2 or
                not all(finite_number(x) for x in interval) or interval[0] >= interval[1]):
                missing.append('workload interval missing')
            elif streams > 0 and len(firsts) == len(lasts) == streams and all(finite_number(x) for x in firsts + lasts):
                if not (finite_number(artifact.get('work_start_wall')) and finite_number(artifact.get('work_end_wall'))
                        and interval[0] <= artifact['work_start_wall'] <= artifact['work_end_wall'] <= interval[1]):
                    fails.append('interval does not contain workload')
                if not all(a <= interval[0] and b >= interval[1] for a, b in zip(firsts, lasts)):
                    missing.append('pressure tokens did not bracket the workload interval on every stream')
        evidence = {'attempts': coverage['attempted_graded'], 'expected': coverage['expected_graded'],
                    'eligible_cases': coverage['eligible_cases'], 'partial_hits': coverage['partial_hits'],
                    'target_partial_hits': coverage['target_partial_hits'],
                    'achieved_partial_coverage': coverage['partial_ratio'],
                    'class_counts': coverage['class_counts'],
                    'transition_counts': coverage['transition_counts'],
                    'cache_hit_values': coverage['cache_hit_values'],
                    'cancellation_scope': 'client disconnect submissions only; device-side cancellation unproven',
                    'pressure_scope': 'token bracketing and completions; not continuous device occupancy proof',
                    'eligibility_scope': 'constructed shape, not physical registration proof',
                    'cancellations_attempted': artifact.get('cancellation_attempts'),
                    'cancelled': artifact.get('cancelled'), 'errors': len(artifact.get('errors') or []),
                    'pressure_overlap': None if interval is None else {
                        'work_start_wall': interval[0], 'work_end_wall': interval[1],
                        'first_tokens': firsts, 'last_tokens': lasts,
                        'completed_requests': completed},
                    'budget': artifact.get('budget')}
        verdict = 'FAIL' if fails else ('INCONCLUSIVE' if missing else 'PASS')
        return {'tool': TOOL, 'verdict': verdict, 'fails': fails, 'missing': missing,
                'oracle': run, 'coverage': coverage, 'evidence': evidence}
    except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError) as e:
        return {'tool': TOOL, 'verdict': 'FAIL', 'fails': [f'malformed/missing artifact: {type(e).__name__}'],
                'missing': [], 'oracle': None, 'coverage': None, 'evidence': None}


def combine(qualification, stress_leak, stress_lifetime):
    """Preserve every subphase verdict; qualifying at lower load cannot clear stress."""
    try:
        values = [qualification['verdict'], stress_leak['run']['verdict'], stress_lifetime['verdict']]
        if any(v not in ('PASS', 'FAIL', 'INCONCLUSIVE') for v in values):
            raise ValueError('unknown verdict')
        for record in (qualification, stress_leak['run'], stress_lifetime):
            if record.get('fails') and record['verdict'] != 'FAIL':
                raise ValueError('contradictory failure evidence')
            if (record.get('missing') or record.get('inconclusive')) and record['verdict'] == 'PASS':
                raise ValueError('contradictory coverage evidence')
        verdict = 'FAIL' if 'FAIL' in values else ('INCONCLUSIVE' if 'INCONCLUSIVE' in values else 'PASS')
        return dict(verdict=verdict, qualification=values[0], stress_partial_coverage=values[1],
                    stress_lifetime=values[2], fails=['subphase FAIL'] if verdict == 'FAIL' else [],
                    missing=['subphase INCONCLUSIVE'] if 'INCONCLUSIVE' in values else [])
    except (AttributeError, KeyError, TypeError, ValueError):
        return dict(verdict='FAIL', fails=['malformed/contradictory subphase grade'], missing=[],
                    stress_partial_coverage=None)


# -- CLI -----------------------------------------------------------------------------------------------------------

def _spec_from_env():
    spec = {'n': None, 'rounds': None, 'b': None, 'unit': _env_int('UNIT', 176), 'block': _env_int('BLOCK', 4048),
            'gen': _env_int('GEN', 200), 'fact': _env_int('FACT', 5000), 'fact2': _env_int('FACT2', 6800),
            'seed': _env_int('SEED', 1), 'cancel_frac': _env_float('CANCEL_FRAC', 0.0),
            'need_partial': _env_bool('NEED_PARTIAL', True), 'refresh': _env_bool('REFRESH', True),
            'budget_s': _env_float('BUDGET_S', 900), 'request_timeout': _env_float('REQUEST_TIMEOUT', 900),
            'stagger': _env_float('SESSION_STAGGER', 0.0), 'serial': _env_bool('SERIAL', False),
            'cold': _env_bool('COLD', True), 'cache_ns': os.environ.get('CACHE_NS', ''),
            'corpus': os.environ.get('CORPUS', DEFAULT_CORPUS), 'min_partial': MIN_PARTIAL}
    if os.environ.get('MODE', 'inside') != 'inside':
        raise SystemExit('MODE=inside is the only coverage-aware mode')
    return spec


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 1
    cmd = argv[1]
    if cmd == 'run':
        args = argv[2:]
        if len(args) == 5:
            url, model, n, rounds, out = args
            b = 43 * 176
        elif len(args) == 6:
            url, model, n, rounds, b, out = args
        else:
            print('usage: partial_tail_cov.py run URL MODEL SESSIONS ROUNDS [B] OUT.json', file=sys.stderr)
            return 1
        n, rounds, b = int(n), int(rounds), int(b)
        spec = _spec_from_env()
        spec.update(n=n, rounds=rounds, b=b)
        if not os.path.isfile(spec['corpus']):
            print(f'corpus missing: {spec["corpus"]}', file=sys.stderr)
            return 1
        corpus = open(spec['corpus'], encoding='utf-8', errors='ignore').read()
        engine = HttpEngine(url, model, timeout=spec['request_timeout'])
        artifact = run_workload(engine, corpus, spec)
        with open(out, 'w') as f:
            json.dump(artifact, f, sort_keys=True)
        run = artifact['run']
        print(f"{TOOL} n={n} rounds={rounds} B={b}: {run['verdict']} answers {len(artifact['cases'])}/"
              f"{n * rounds * 2} {run['counts']} errors {len(artifact['errors'])}"
              f" cancelled {artifact['cancelled']} partial {artifact['coverage']['partial_hits']}"
              f" transitions {artifact['coverage']['transition_counts']}")
        for e in artifact['errors']:
            print('ERROR', e)
        return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[run['verdict']]
    if cmd == 'grade':
        if len(argv) != 6:
            print('usage: partial_tail_cov.py grade WORKLOAD.json PRESSURE.json INTERVAL.json OUT.json',
                  file=sys.stderr)
            return 1
        try:
            artifact = json.load(open(argv[2]))
            pressure = json.load(open(argv[3])) if argv[3] != '-' else None
            interval = json.load(open(argv[4]))
        except (OSError, ValueError) as e:
            result = {'tool': TOOL, 'verdict': 'FAIL', 'fails': [f'unreadable evidence: {type(e).__name__}'],
                      'missing': [], 'oracle': None, 'coverage': None, 'evidence': None}
        else:
            result = grade_artifact(artifact, pressure, interval)
        with open(argv[5], 'w') as f:
            json.dump(result, f, sort_keys=True)
        ev = result.get('evidence') or {}
        print(f"{TOOL}-grade {result['verdict']} fails={result['fails']} missing={result['missing']} "
              f"attempts={ev.get('attempts')} eligible={ev.get('eligible_cases')} "
              f"partial={ev.get('partial_hits')} ratio={ev.get('achieved_partial_coverage')} "
              f"cancelled={ev.get('cancelled')} transitions={ev.get('transition_counts')}")
        return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['verdict']]
    if cmd == 'combine':
        if len(argv) != 6:
            print('usage: partial_tail_cov.py combine QUAL_GRADE.json STRESS_LEAK.json STRESS_LIFETIME.json OUT.json',
                  file=sys.stderr)
            return 1
        try:
            qual = json.load(open(argv[2]))
            stress = json.load(open(argv[3]))
            lifetime = json.load(open(argv[4]))
            result = combine(qual, stress, lifetime)
        except (OSError, ValueError) as e:
            result = {'verdict': 'FAIL', 'fails': [f'unreadable evidence: {type(e).__name__}'], 'missing': [],
                      'stress_partial_coverage': None}
        with open(argv[5], 'w') as f:
            json.dump(result, f, sort_keys=True)
        print(f"{TOOL}-combine {result['verdict']} fails={result['fails']} missing={result['missing']} "
              f"stress_partial_coverage={result.get('stress_partial_coverage')}")
        return {'PASS': 0, 'FAIL': 1, 'INCONCLUSIVE': 2}[result['verdict']]
    print(f'unknown command {cmd}', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
