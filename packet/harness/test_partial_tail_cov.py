#!/usr/bin/env python3
"""Offline, deterministic tests for harness/partial_tail_cov.py (no engine, no GPU, no network).

The fake engine models the vLLM prefix cache closely enough to distinguish the four outcomes the E2 lifetime phase
must tell apart:
  * eligible partial hit       (cached = the prompt's own hash boundary, not block-aligned),
  * full-page-only miss        (the partial source block was evicted, the earlier full block survived),
  * full eviction              (no usable prefix at all),
  * missing partial-tail exercise (the boundary is block-aligned or the producer's generation promoted it away).
It also models cancellation, engine identity, and an optional policy hook that evicts the partial source at a chosen
transition.  A separate slow-clock fixture proves the bounded budget fails closed.  What this CANNOT prove is that
the real engine registers a partial entry or that a real workload reaches the coverage threshold; only a measured
GPU run does that.
"""
import json
import os
import random
import re
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leakgrade  # noqa: E402
import partial_tail_cov as cov  # noqa: E402

UNIT, BLOCK = 41, 401
B = 43 * UNIT          # 1763, a hash boundary, not block-aligned (1763 % 97 == 17)
FACT, FACT2 = 1650, 1690
CORPUS = 'w ' * 150000
RESULTS = []


def check(name, got, want):
    RESULTS.append((name, got == want, got, want))


class FakeEngine:
    """Deterministic word-level tokenizer + prefix cache with partial-tail entries and promotions."""

    def __init__(self, unit=UNIT, block=BLOCK, answer_mode='own', policy=None, gen_foreign=None,
                 foreign_literal=None):
        self.unit, self.block = unit, block
        self.words, self.vocab = {}, {}
        self.cache, self.lock = {}, threading.Lock()
        self.answer_mode = answer_mode
        self.policy = policy
        self.gen_foreign = gen_foreign
        self.foreign_literal = foreign_literal
        self.aborts = 0
        self.calls = []
        self.identity_value = 'fake-container 2026-10-25T00:00:00Z'

    # -- tokenizer ------------------------------------------------------------------------------------------------
    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab) + 1
            self.words[self.vocab[w]] = w
        return self.vocab[w]

    def tokenize(self, text):
        return [self._id(w) for w in re.findall(r'\S+', text)]

    def detokenize(self, ids):
        return ' '.join(self.words[i] for i in ids)

    # -- engine surface -------------------------------------------------------------------------------------------
    def identity(self):
        return self.identity_value

    def abort_after(self, prompt, seconds, salt=None):
        with self.lock:
            self.aborts += 1

    def next_full(self, boundary):
        return cov.next_full_after(boundary, self.block)

    def complete(self, prompt, max_tokens, salt=None, ignore_eos=False, timeout=None):
        n = len(prompt)
        text = self.detokenize(prompt)
        boundary = n // self.unit * self.unit
        with self.lock:
            live = self.cache.setdefault(salt or '', set())
            hit = max((x for x in live if x <= boundary), default=0)
            kind = 'producer' if max_tokens > 12 else ('refresh' if max_tokens == 1 else 'consumer')
            # register the prompt's full blocks and its partial tail (or the promotion target)
            for fb in range(self.block, n + 1, self.block):
                live.add(fb)
            promoted = False
            if boundary and boundary % self.block:
                if n + max_tokens >= self.next_full(boundary):
                    live.discard(boundary)
                    live.add(self.next_full(boundary))
                    promoted = True
                else:
                    live.add(boundary)
            elif boundary:
                live.add(boundary)
            if self.policy:
                self.policy(self, kind, salt, n, boundary, hit, live)
            self.calls.append({'kind': kind, 'salt': salt, 'n': n, 'hit': hit, 'boundary': boundary,
                               'promoted': promoted})
        return {'text': self._text(kind, text, max_tokens), 'token_ids': [self._id('gen')] * max_tokens,
                'cached': hit}

    def _text(self, kind, prompt_text, max_tokens):
        if kind == 'producer':
            if self.gen_foreign:
                return f'{self.gen_foreign} ' * max_tokens
            return 'gen ' * max_tokens
        lockers = re.findall(r'L[0-9a-f]{6}', prompt_text)
        codes = re.findall(r'[A-Z]+-[0-9]{4}', prompt_text)
        own = (lockers[0] if 'The locker number is' in prompt_text else codes[0]) if (lockers or codes) else 'NONE-0000'
        if self.answer_mode == 'foreign':
            return f'The answer is {self.foreign_literal}.'
        if self.answer_mode == 'wrong':
            return 'I do not know.'
        return f'The answer is {own}.'


def make_spec(**over):
    spec = {'n': 2, 'rounds': 1, 'b': B, 'unit': UNIT, 'block': BLOCK, 'gen': 60, 'fact': FACT, 'fact2': FACT2,
            'seed': 1, 'cancel_frac': 1.0, 'need_partial': True, 'refresh': True, 'budget_s': 120,
            'request_timeout': 120, 'stagger': 0.0, 'serial': True, 'cold': True, 'cache_ns': 'test',
            'corpus': 'fake', 'min_partial': cov.MIN_PARTIAL}
    spec.update(over)
    return spec


def codes_for(seed, n):
    rnd = random.Random(seed * 100003)
    rnd.getrandbits(64)                       # shared nonce
    rnd.randrange(0, len(CORPUS) - 200000)    # shared corpus offset
    return cov._codes_for_round(rnd, n, [], {})


def pressure_for(artifact, streams=2, completed=None, first=None, last=None, errors=None):
    start = artifact['work_start_wall'] or time.time()
    end = artifact['work_end_wall'] or start + 1
    return {'streams': streams,
            'first_tokens': first if first is not None else [start - 1] * streams,
            'last_tokens': last if last is not None else [end + 1] * streams,
            'completed_requests': completed if completed is not None else [3] * streams,
            'errors': errors or []}


def grade_run(artifact, **pressure):
    return cov.grade_artifact(artifact, pressure_for(artifact, **pressure),
                              [artifact['work_start_wall'], artifact['work_end_wall']])


# -- deterministic fixtures: eligibility transitions ---------------------------------------------------------------

def test_hit_evict_recompute_hit():
    fake = FakeEngine()
    p = [fake._id('w')] * B
    producer = fake.complete(p, 60, salt='s')
    check('fixture producer registered the partial boundary', producer['cached'], 0)
    first = fake.complete(p, 12, salt='s')
    check('fixture eligible partial hit', first['cached'], B)
    check('fixture hit is a partial boundary', first['cached'] % BLOCK != 0, True)
    with fake.lock:
        fake.cache['s'].discard(B)                       # evict only the partial source block
    lost = fake.complete(p, 12, salt='s')
    check('fixture eviction falls back to the full block', lost['cached'], 1604)
    check('fixture full-page-only is not a partial hit', lost['cached'] % BLOCK == 0, True)
    fake.complete(p, 1, salt='s')                        # refresh recomputes and re-registers
    again = fake.complete(p, 12, salt='s')
    check('fixture refresh restores the eligible partial hit', again['cached'], B)


def test_promotion_is_not_eviction():
    fake = FakeEngine()
    prompt = [fake._id('w')] * B
    fake.complete(prompt, 300, salt='s')                 # B + 300 >= next full block
    got = fake.complete(prompt, 12, salt='s')
    check('fixture promoted producer leaves a full-block hit', got['cached'] % BLOCK == 0, True)
    check('fixture promotion target is the next full block', fake.next_full(B), 2005)


def test_transition_labels():
    producer_partial = {'boundary': B, 'expected_partial': True, 'promoted': False, 'reason': ''}
    producer_full = {'boundary': B, 'expected_partial': True, 'promoted': True, 'reason': 'x'}
    aligned = {'boundary': B, 'expected_partial': False, 'promoted': False, 'reason': 'boundary is block-aligned'}
    check('transition eligible partial', cov.transition(B, producer_partial, B, B, BLOCK), 'partial_at_expected')
    check('transition lost after refresh (full page only)',
          cov.transition(B, producer_partial, B, BLOCK, BLOCK), 'full_page_only_after_partial_refresh')
    check('transition lost before refresh (full page only)',
          cov.transition(B, producer_partial, BLOCK, BLOCK, BLOCK), 'full_page_only_after_full_refresh')
    check('transition full eviction after refresh',
          cov.transition(B, producer_partial, B, 0, BLOCK), 'zero_hit_after_partial_refresh')
    check('transition promotion', cov.transition(B, producer_full, B, BLOCK, BLOCK), 'ineligible_promotion')
    check('transition aligned boundary', cov.transition(B, aligned, B, B, BLOCK), 'ineligible_boundary')


# -- driver + oracle coverage -------------------------------------------------------------------------------------

def test_driver_pass_and_evidence():
    fake = FakeEngine()
    art = cov.run_workload(fake, CORPUS, make_spec())
    check('driver grades every answer', len(art['cases']), 4)
    check('driver run PASS', art['run']['verdict'], 'PASS')
    check('driver every case eligible', art['coverage']['eligible_cases'], 4)
    check('driver every case partial', art['coverage']['partial_hits'], 4)
    check('driver transition all partial', art['coverage']['transition_counts'], {'partial_at_expected': 4})
    check('driver cancellation exercised', art['cancelled'] > 0, True)
    check('driver cancellation attempts reported', art['cancellation_attempts'], art['cancelled'])
    check('driver no errors', art['errors'], [])
    check('driver grade PASS', grade_run(art)['verdict'], 'PASS')
    check('driver artifact has no prompt or answer payload',
          any('answer' in c and not isinstance(c['answer'], str) for c in art['cases']), False)
    check('driver artifact only stores lengths and counts',
          set(art['results'][0]) >= {'p1_len', 'p2_len', 'p3_len', 'gen1_len', 'cases'}, True)


def test_interleaving_under_pressure():
    fake = FakeEngine()
    stop = threading.Event()
    pressure_calls = []

    def pressure():
        while not stop.is_set():
            fake.complete([fake._id('p')] * (BLOCK * 2), 12, salt='pressure')
            pressure_calls.append(1)

    t = threading.Thread(target=pressure, daemon=True)
    t.start()
    try:
        art = cov.run_workload(fake, CORPUS, make_spec(n=2, rounds=1))
    finally:
        stop.set()
        t.join(timeout=5)
    check('interleaved pressure issued requests', len(pressure_calls) > 0, True)
    check('interleaved workload still complete', len(art['cases']), 4)
    check('interleaved workload PASS', art['run']['verdict'], 'PASS')
    check('interleaved grade PASS', grade_run(art)['verdict'], 'PASS')


def evict_after(kind_to_evict, full_clear=False):
    def policy(fake, kind, salt, n, boundary, hit, live):
        if kind == kind_to_evict:
            if full_clear:
                live.clear()
            else:
                for x in [x for x in live if x % fake.block]:   # every partial source, not just this prompt's
                    live.discard(x)
    return policy


def test_driver_classifies_eviction_not_exercise():
    art = cov.run_workload(FakeEngine(policy=evict_after('refresh')), CORPUS, make_spec())
    counts = art['coverage']['transition_counts']
    fp = sum(v for k, v in counts.items() if k.startswith('full_page_only_after'))
    check('evict-after-refresh is classified as full-page-only', fp, 4)
    check('evict-after-refresh lowers coverage', art['coverage']['partial_hits'], 0)
    check('evict-after-refresh is INCONCLUSIVE (not FAIL)', art['run']['verdict'], 'INCONCLUSIVE')
    check('evict-after-refresh grade INCONCLUSIVE', grade_run(art)['verdict'], 'INCONCLUSIVE')
    full = cov.run_workload(FakeEngine(policy=evict_after('refresh', full_clear=True)), CORPUS, make_spec())
    nh = sum(v for k, v in full['coverage']['transition_counts'].items() if k.startswith('zero_hit_after'))
    check('full eviction classified separately', nh, 4)
    check('full eviction reports cached=0', set(full['coverage']['cache_hit_values']['answer']), {0})
    check('full eviction reports cached3=0', set(full['coverage']['cache_hit_values']['answer3']), {0})


def test_driver_refresh_recovers_producer_eviction():
    art = cov.run_workload(FakeEngine(policy=evict_after('producer')), CORPUS, make_spec())
    check('producer eviction recovered by refresh: full coverage', art['coverage']['partial_hits'], 4)
    check('producer eviction visible in refresh observation',
          {c['refresh_class'] for c in art['cases'] if c['kind'] == 'answer'}, {'full'})
    check('producer eviction recovered at the expected boundary',
          {c['cached'] for c in art['cases'] if c['kind'] == 'answer'}, {B})
    check('producer eviction recovered: PASS', art['run']['verdict'], 'PASS')


def test_driver_promotion_is_a_construction_failure():
    art = cov.run_workload(FakeEngine(), CORPUS, make_spec(gen=260))
    check('promotion counted as construction failure',
          art['coverage']['construction']['ineligible_promotion'] > 0, True)
    check('promotion grade FAIL', grade_run(art)['verdict'], 'FAIL')
    check('promotion failure names the workload defect',
          any('promoted' in f for f in grade_run(art)['fails']), True)


def test_driver_leak_propagates():
    codes = codes_for(1, 2)
    fake = FakeEngine(answer_mode='foreign', foreign_literal=codes[1])
    art = cov.run_workload(fake, CORPUS, make_spec())
    check('driver LEAK run FAIL', art['run']['verdict'], 'FAIL')
    check('driver LEAK grade FAIL', grade_run(art)['verdict'], 'FAIL')
    check('driver LEAK counted once', art['run']['counts'].get('LEAK'), 1)


def test_driver_gen1_foreign_identifier_is_an_error():
    codes = codes_for(1, 2)
    fake = FakeEngine(gen_foreign=codes[1])
    art = cov.run_workload(fake, CORPUS, make_spec())
    check('request-1 foreign identifier is an error',
          any('LEAK in request 1' in e for e in art['errors']), True)
    check('request-1 foreign identifier fails the run', art['run']['verdict'], 'FAIL')


def test_driver_miss_is_not_retried_as_leak():
    art = cov.run_workload(FakeEngine(answer_mode='wrong'), CORPUS, make_spec())
    check('wrong warm and cold is MISS', art['run']['counts'].get('MISS'), 4)
    check('MISS above threshold is INCONCLUSIVE', art['run']['verdict'], 'INCONCLUSIVE')
    check('MISS cases keep first verdict and no settlement',
          all(c['first'] == 'MISS' and c['settled'] is None for c in art['cases']), True)


def test_real_geometry_margin():
    # the production geometry: B=43*176=7568, block 4048, GEN 200 gives p2+12 < 8096 (no promotion)
    spec = make_spec(n=2, rounds=1, b=7568, unit=176, block=4048, gen=200, fact=5000, fact2=6800,
                     serial=True, cache_ns='qualification-test')
    art = cov.run_workload(FakeEngine(unit=176, block=4048), CORPUS, spec)
    check('real geometry all cases eligible', art['coverage']['eligible_cases'], 4)
    check('real geometry no construction failure', art['coverage']['construction'],
          {'ineligible_boundary': 0, 'ineligible_promotion': 0})
    check('real geometry hits the expected boundary',
          {c['cached'] for c in art['cases'] if c['kind'] == 'answer'}, {7568})
    check('real geometry PASS', art['run']['verdict'], 'PASS')
    # GEN 400 crosses the next full block; the driver must classify and fail the workload, not call it eviction
    bad = cov.run_workload(FakeEngine(unit=176, block=4048), CORPUS,
                           make_spec(n=2, rounds=1, b=7568, unit=176, block=4048, gen=400, fact=5000, fact2=6800,
                                     serial=True, cache_ns='qualification-test'))
    check('real geometry GEN 400 is a construction failure',
          bad['coverage']['construction']['ineligible_promotion'] +
          bad['coverage']['construction']['ineligible_boundary'] > 0, True)
    check('real geometry GEN 400 grade FAIL',
          grade_run(bad)['verdict'], 'FAIL')


def test_bounded_budget_fails_closed():
    clock = [0.0]

    class SlowFake(FakeEngine):
        def complete(self, prompt, max_tokens, salt=None, ignore_eos=False, timeout=None):
            clock[0] += 10.0
            return super().complete(prompt, max_tokens, salt=salt, ignore_eos=ignore_eos, timeout=timeout)

    started = time.monotonic()
    art = cov.run_workload(SlowFake(), CORPUS, make_spec(budget_s=25, serial=True),
                           clock=lambda: clock[0], sleep=lambda s: None)
    elapsed = time.monotonic() - started
    check('bounded budget is marked exhausted', art['budget']['exhausted'], True)
    check('bounded budget leaves ungraded answers', len(art['cases']) < 4, True)
    check('bounded budget run FAIL', art['run']['verdict'], 'FAIL')
    check('bounded budget grade FAIL', grade_run(art)['verdict'], 'FAIL')
    check('bounded budget returns promptly', elapsed < 5, True)


# -- grader: thresholds, tampering, pressure, cancellation ---------------------------------------------------------

def synthetic_artifact(total=100, partial=80, cancel_frac=0.3, cancelled=1, promotions=0, leak=0,
                       need_partial=True, refresh=True, b=B, identity=True):
    cases = []
    spec = make_spec(n=total // 2, rounds=1, cancel_frac=cancel_frac, need_partial=need_partial,
                     refresh=refresh, b=b)
    for i in range(total):
        generated = 300 if i < promotions else 60
        producer = cov.construction_events(B + 4, generated, UNIT, BLOCK)
        cached = B if i < partial else BLOCK
        cases.append(dict(round=0, session=i // 2, kind='answer' if i % 2 == 0 else 'answer3',
            first='LEAK' if i < leak else 'OK', settled=None, partial=i < partial,
            cached=cached, cached_class=cov.classify_cached(cached, BLOCK),
            refresh_cached=B, refresh_class='partial', producer_len=B + 4,
            producer_generated=generated, consumer_len=B + 400,
            producer_boundary=producer['boundary'], expected_boundary=producer['boundary'],
            eligible=producer['expected_partial'] and not producer['promoted'],
            producer_promoted=producer['promoted'],
            transition=cov.transition(producer['boundary'], producer, B, cached, BLOCK)))
    run = leakgrade.run_verdict(cases, total, [], need_partial=need_partial, min_partial=cov.MIN_PARTIAL)
    if cancel_frac and not cancelled:
        run['inconclusive'].append('no request was cancelled')
        if run['verdict'] == 'PASS':
            run['verdict'] = 'INCONCLUSIVE'
    return {'tool': cov.TOOL, 'version': cov.VERSION, 'mode': 'inside', 'spec': spec,
            'identity_start': 'x' if identity else None, 'identity_end': 'x',
            'work_start_wall': 1000.0, 'work_end_wall': 1100.0,
            'budget': {'sanctioned_seconds': 900, 'exhausted': False},
            'results': [], 'cases': cases, 'errors': [],
            'cancelled': cancelled, 'cancellation_attempts': cancelled,
            'run': run, 'coverage': cov.summarize_cases(cases, spec)}


def test_thresholds_unchanged():
    art = synthetic_artifact(partial=80)
    check('80/100 partial PASS', cov.grade_artifact(art, pressure_for(art), [1000.0, 1100.0])['verdict'], 'PASS')
    art = synthetic_artifact(partial=79)
    check('79/100 partial INCONCLUSIVE',
          cov.grade_artifact(art, pressure_for(art), [1000.0, 1100.0])['verdict'], 'INCONCLUSIVE')
    check('threshold is reported, not lowered',
          cov.grade_artifact(art, pressure_for(art), [1000.0, 1100.0])['coverage']['min_partial'], 0.8)
    art = synthetic_artifact(partial=79, leak=1)
    check('LEAK dominates the coverage shortfall',
          cov.grade_artifact(art, pressure_for(art), [1000.0, 1100.0])['verdict'], 'FAIL')


def test_grader_recomputes_and_rejects_tampering():
    art = synthetic_artifact(partial=79)
    art['run'] = {'verdict': 'PASS', 'fails': [], 'inconclusive': []}
    out = cov.grade_artifact(art, pressure_for(art), [1000.0, 1100.0])
    check('tampered saved PASS is rejected', out['verdict'], 'FAIL')
    check('tampered PASS names the disagreement', any('disagrees' in f for f in out['fails']), True)
    art = synthetic_artifact(partial=100, need_partial=False)
    check('NEED_PARTIAL=false is rejected',
          cov.grade_artifact(art, pressure_for(art), [1000.0, 1100.0])['verdict'], 'FAIL')
    art = synthetic_artifact(partial=100, refresh=False)
    check('REFRESH=0 is rejected', cov.grade_artifact(art, pressure_for(art), [1000.0, 1100.0])['verdict'], 'FAIL')
    art = synthetic_artifact(partial=100, b=BLOCK * 2)
    check('block-aligned B is rejected', cov.grade_artifact(art, pressure_for(art), [1000.0, 1100.0])['verdict'],
          'FAIL')


def test_grader_pressure_and_cancellation():
    art = synthetic_artifact(partial=100)
    check('pressure missing is INCONCLUSIVE', cov.grade_artifact(art, None, [1000.0, 1100.0])['verdict'],
          'INCONCLUSIVE')
    check('pressure error is FAIL', cov.grade_artifact(art, pressure_for(art, errors=['boom']),
                                                       [1000.0, 1100.0])['verdict'], 'FAIL')
    check('pressure stream with no completion is INCONCLUSIVE',
          cov.grade_artifact(art, pressure_for(art, completed=[3, 0]), [1000.0, 1100.0])['verdict'],
          'INCONCLUSIVE')
    check('pressure not bracketing the interval is INCONCLUSIVE',
          cov.grade_artifact(art, pressure_for(art, last=[900.0, 900.0]), [1000.0, 1100.0])['verdict'],
          'INCONCLUSIVE')
    check('pressure stream count mismatch is FAIL',
          cov.grade_artifact(art, pressure_for(art, first=[1000.0]), [1000.0, 1100.0])['verdict'], 'FAIL')
    art = synthetic_artifact(partial=100, cancelled=0)
    check('configured cancellation not exercised is INCONCLUSIVE',
          cov.grade_artifact(art, pressure_for(art), [1000.0, 1100.0])['verdict'], 'INCONCLUSIVE')
    check('runtime restart is FAIL',
          cov.grade_artifact(synthetic_artifact(partial=100, identity=False), pressure_for(art),
                             [1000.0, 1100.0])['verdict'], 'FAIL')


def test_malformed_artifacts_fail_closed():
    check('None artifact FAIL', cov.grade_artifact(None)['verdict'], 'FAIL')
    check('empty artifact FAIL', cov.grade_artifact({})['verdict'], 'FAIL')
    bad = synthetic_artifact()
    del bad['cases']
    check('artifact without cases FAIL', cov.grade_artifact(bad)['verdict'], 'FAIL')
    bad = synthetic_artifact()
    bad['tool'] = 'other'
    check('foreign artifact FAIL', cov.grade_artifact(bad)['verdict'], 'FAIL')


# -- combine: declared qualification vs stress subphase ------------------------------------------------------------

def test_combine_contract():
    ok_qual = {'verdict': 'PASS', 'fails': [], 'missing': []}
    ok_leak = {'run': {'verdict': 'PASS', 'fails': [], 'inconclusive': []}}
    ok_life = {'verdict': 'PASS'}
    check('combine all PASS', cov.combine(ok_qual, ok_leak, ok_life)['verdict'], 'PASS')
    check('qualification INCONCLUSIVE gates', cov.combine({'verdict': 'INCONCLUSIVE'}, ok_leak, ok_life)['verdict'],
          'INCONCLUSIVE')
    check('qualification FAIL gates', cov.combine({'verdict': 'FAIL'}, ok_leak, ok_life)['verdict'], 'FAIL')
    cov_leak = {'run': {'verdict': 'INCONCLUSIVE', 'fails': [],
                        'inconclusive': ['partial-page hits on 15 of 80 answers']}}
    out = cov.combine(ok_qual, cov_leak, ok_life)
    check('stress partial coverage stays INCONCLUSIVE', out['verdict'], 'INCONCLUSIVE')
    check('stress partial coverage is explicit', out['stress_partial_coverage'],
          'INCONCLUSIVE')
    hard = {'run': {'verdict': 'INCONCLUSIVE', 'fails': [], 'inconclusive': ['3 unsettled suspect answers']}}
    check('stress non-coverage INCONCLUSIVE gates', cov.combine(ok_qual, hard, ok_life)['verdict'], 'INCONCLUSIVE')
    leak = {'run': {'verdict': 'FAIL', 'fails': ['LEAK 1 COPYFAIL 0'], 'inconclusive': []}}
    check('stress LEAK gates', cov.combine(ok_qual, leak, ok_life)['verdict'], 'FAIL')
    check('stress lifetime INCONCLUSIVE gates',
          cov.combine(ok_qual, ok_leak, {'verdict': 'INCONCLUSIVE'})['verdict'], 'INCONCLUSIVE')
    check('stress lifetime FAIL gates', cov.combine(ok_qual, ok_leak, {'verdict': 'FAIL'})['verdict'], 'FAIL')
    check('malformed qualification FAIL', cov.combine(None, ok_leak, ok_life)['verdict'], 'FAIL')
    check('malformed stress leak FAIL', cov.combine(ok_qual, None, ok_life)['verdict'], 'FAIL')
    check('malformed stress lifetime FAIL', cov.combine(ok_qual, ok_leak, None)['verdict'], 'FAIL')


# -- CLI ----------------------------------------------------------------------------------------------------------

def test_cli_exit_codes():
    with tempfile.TemporaryDirectory() as td:
        art = synthetic_artifact(partial=100)
        p = os.path.join(td, 'w.json')
        pr = os.path.join(td, 'p.json')
        iv = os.path.join(td, 'i.json')
        out = os.path.join(td, 'g.json')
        json.dump(art, open(p, 'w'))
        json.dump(pressure_for(art), open(pr, 'w'))
        json.dump([1000.0, 1100.0], open(iv, 'w'))
        rc = cov.main(['x', 'grade', p, pr, iv, out])
        check('grade CLI PASS rc', rc, 0)
        check('grade CLI writes evidence', json.load(open(out))['verdict'], 'PASS')
        check('grade CLI missing file is FAIL', cov.main(['x', 'grade', '/nonexistent', pr, iv, out]), 1)
        q = os.path.join(td, 'q.json')
        s = os.path.join(td, 's.json')
        l = os.path.join(td, 'l.json')
        c = os.path.join(td, 'c.json')
        json.dump({'verdict': 'PASS'}, open(q, 'w'))
        json.dump({'run': {'verdict': 'PASS', 'fails': [], 'inconclusive': []}}, open(s, 'w'))
        json.dump({'verdict': 'PASS'}, open(l, 'w'))
        check('combine CLI PASS rc', cov.main(['x', 'combine', q, s, l, c]), 0)
        json.dump({'verdict': 'FAIL'}, open(q, 'w'))
        check('combine CLI FAIL rc', cov.main(['x', 'combine', q, s, l, c]), 1)


def test_review_regressions():
    import copy
    import ast
    import subprocess
    from pathlib import Path
    import partial_tail_legacy_metadata as legacy
    art = synthetic_artifact(partial=100)
    def verdict(a):
        return cov.grade_artifact(a, pressure_for(art), [1000.0, 1100.0])['verdict']
    changes = [
        ('negative cache', lambda a: a['cases'][0].update(cached=-1)),
        ('bool cache', lambda a: a['cases'][0].update(cached=True)),
        ('fabricated partial flag', lambda a: a['cases'][0].update(cached=BLOCK)),
        ('duplicate case', lambda a: a['cases'].__setitem__(1, copy.deepcopy(a['cases'][0]))),
        ('unknown first verdict', lambda a: a['cases'][0].update(first='BANANA')),
        ('masked leak', lambda a: a['cases'][0].update(first='LEAK', settled='NOISE')),
        ('zero workload', lambda a: a['spec'].update(n=0)),
        ('bool count', lambda a: a['spec'].update(n=True)),
        ('NaN budget', lambda a: a['spec'].update(budget_s=float('nan'))),
        ('cold disabled', lambda a: a['spec'].update(cold=False)),
        ('bad cancellation count', lambda a: a.update(cancelled=5000)),
    ]
    for label, change in changes:
        a = copy.deepcopy(art); change(a)
        check('review rejects ' + label, verdict(a), 'FAIL')
    for label, interval in [('reverse', [1100.0,1000.0]), ('NaN', [float('nan'),1100.0]),
                            ('excludes work', [1050.0,1100.0])]:
        check('interval ' + label + ' cannot PASS',
              cov.grade_artifact(art, pressure_for(art), interval)['verdict'] != 'PASS', True)
    check('NaN pressure cannot PASS', cov.grade_artifact(art,
          pressure_for(art, first=[float('nan'),999.0]), [1000.0,1100.0])['verdict'] != 'PASS', True)
    check('one pressure stream cannot qualify', cov.grade_artifact(art,
          pressure_for(art, streams=1), [1000.0,1100.0])['verdict'], 'FAIL')
    for v in ('BANANA', None):
        check('unknown stress verdict rejected', cov.combine({'verdict':'PASS'}, {'run':{'verdict':v}},
                                                            {'verdict':'PASS'})['verdict'], 'FAIL')
    check('contradictory qualification rejected', cov.combine({'verdict':'PASS','fails':['x']},
          {'run':{'verdict':'PASS'}}, {'verdict':'PASS'})['verdict'], 'FAIL')
    fake = FakeEngine()
    generated = cov.run_workload(fake, CORPUS, make_spec())
    serialized = json.dumps(generated)
    check('no synthetic code identifiers persisted', re.search(r'[A-Z]+-[0-9]{4}', serialized) is None, True)
    check('no locker identifiers persisted', re.search(r'L[0-9a-f]{6}', serialized) is None, True)
    class BadEngine(FakeEngine):
        def complete(self, *a, **kw):
            raise ValueError('PRIVATE-PAYLOAD-DO-NOT-PERSIST')
    bad = cov.run_workload(BadEngine(), CORPUS, make_spec())
    check('exception payload is redacted', 'PRIVATE-PAYLOAD' in json.dumps(bad), False)
    # Future-round foreign identities must be known before the first output.
    rnd = random.Random(1 * 100003 + 1)
    rnd.getrandbits(64); rnd.randrange(0, len(CORPUS) - 200000)
    later = cov._codes_for_round(rnd, 2, [], {})[0]
    future = cov.run_workload(FakeEngine(gen_foreign=later), CORPUS, make_spec(rounds=2))
    check('future-round identity detected in first-round producer',
          any(e.startswith('r0 ') and 'LEAK' in e for e in future['errors']), True)
    early = copy.deepcopy(art)
    for c in early['cases']:
        c['cached'] = B - UNIT
        c['transition'] = 'partial_other_boundary'
    check('earlier partial boundary cannot qualify intended page', verdict(early), 'INCONCLUSIVE')
    check('generic partial oracle still unchanged',
          cov.grade_artifact(early, pressure_for(art), [1000.0,1100.0])['oracle']['verdict'], 'PASS')
    # Actual legacy source changes only sinks, never requests or oracle expressions.
    source = Path(__file__).with_name('leak_partial_tail.py').read_text()
    parsed = ast.parse(source); transformed = legacy.transformed(source)
    class Reverse(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id == '_quiet': n.id = 'print'
            if n.id == '_metadata_dump':
                return ast.Attribute(value=ast.Name(id='json',ctx=ast.Load()), attr='dump',ctx=ast.Load())
            return n
    check('legacy AST inverse is exact', ast.dump(Reverse().visit(transformed)), ast.dump(parsed))
    raw = dict(MODE='inside',B=7568,N=1,ROUNDS=1,GEN=200,identity_start='x',identity_end='x',
        run=dict(verdict='FAIL',counts={'LEAK':1},cancelled=1,fails=['PRIVATE-PAYLOAD'],inconclusive=[]),
        errors=['PRIVATE-PAYLOAD'],results=[dict(round=0,session=0,cached=7568,cached3=7744,
            p2_len=7800,p1_ids=[1]*7569,gen1_ids=[2]*200,p3=[3]*8000,
            answer='PRIVATE-PAYLOAD',cold='PRIVATE-PAYLOAD',code='PRIVATE-PAYLOAD')])
    meta = legacy.metadata(raw)
    check('legacy payload/exception stripped before persistence', 'PRIVATE-PAYLOAD' in json.dumps(meta), False)
    check('legacy failure retained', meta['run']['verdict'], 'FAIL')
    check('legacy error count retained', meta['error_count'], 1)
    q = Path(__file__).parent/'runs/q16_e2_serial_v7.draft.sh'
    text = q.read_text()
    check('queue pin matches current source', '2630f86c961c3c5e' in text, True)
    check('current-run drain bound retained', 'LOGSTART' in text, True)
    check('reserve diagnostic FAIL retained', 'DIAGNOSTIC_RESERVE_BEGIN' in text, True)
    body = text[text.index('# One ordinary feature-OFF test engine'):]
    check('no unrelated phase in coverage body', 'restart shipping' in body or 'restart dense' in body, False)
    # Execute the exact queue entry block with shell stubs: final scan happens
    # even when workload failed/inconclusive, before ownership cleanup by trap.
    for work_rc, end_rc in [(0,0),(1,0),(2,0),(0,1)]:
        script = ('set -u\nE2_CONTROL=()\nrestart(){ echo restart; }\n'
                  'run(){ echo workload; return '+str(work_rc)+'; }\n'
                  'phase_end(){ echo phase_end; return '+str(end_rc)+'; }\n' + body)
        result = subprocess.run(['bash','-c',script], capture_output=True, text=True)
        check('phase end preserved for workload rc '+str(work_rc)+' end '+str(end_rc),
              'phase_end' in result.stdout, True)
        check('entry outcome rc '+str(work_rc)+' end '+str(end_rc), result.returncode,
              0 if work_rc == end_rc == 0 else 1)


def main():
    tests = [test_hit_evict_recompute_hit, test_promotion_is_not_eviction, test_transition_labels,
             test_driver_pass_and_evidence, test_interleaving_under_pressure,
             test_driver_classifies_eviction_not_exercise, test_driver_refresh_recovers_producer_eviction,
             test_driver_promotion_is_a_construction_failure, test_driver_leak_propagates,
             test_driver_gen1_foreign_identifier_is_an_error, test_driver_miss_is_not_retried_as_leak,
             test_real_geometry_margin, test_bounded_budget_fails_closed, test_thresholds_unchanged,
             test_grader_recomputes_and_rejects_tampering, test_grader_pressure_and_cancellation,
             test_malformed_artifacts_fail_closed, test_combine_contract, test_cli_exit_codes, test_review_regressions]
    for t in tests:
        t()
    bad = [(n, g, w) for n, ok, g, w in RESULTS if not ok]
    for n, g, w in bad:
        print(f'FAIL {n}: got {g!r} want {w!r}')
    print(f'partial_tail_cov: {len(RESULTS) - len(bad)}/{len(RESULTS)} cases as required')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
