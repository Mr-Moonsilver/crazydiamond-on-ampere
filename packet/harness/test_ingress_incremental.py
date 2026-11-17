"""ingress-incremental (lead, 12 Nov 2026): FLASH_S5_INGRESS_INCREMENTAL and the append common-prefix counter. CPU only.

Hub suite: a pure-Python tokenizer with the HF added-token semantics (special tokens extracted first, each other segment
tokenised on its own) and a position-dependent segment encoding (three-character pieces counted from the segment
start), so a cut anywhere but a join token changes the ids. The real HF tokenizer differential (trained BPE, NFC, byte
level) is test_ingress_incremental_hf (pinned container only: the hub has no tokenizers).

Covers: the splice equals the full call over replay-shaped conversations (many turns, tool outputs, unicode, the reply
re-rendered without the generation prompt's think block); every fallback; cache bounds and LRU eviction; clear on
selection; 8 conversations from 8 threads; the binding against the engine's own manual_control rule; the renderer
wrapper (stock object returned untouched when not selected); install with the flag at 0 and 1; the engine-side prefix
counter (hash bisection, token refinement, buckets, through-reply counts, eviction, errors) and its driver hook.
No GPU, CUDA or box claim.
"""
import array
import ast
import asyncio
from pathlib import Path
import random
import sys
import threading
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'patches/s5_engine')]
import ingress_incremental as ii  # noqa: E402
import agent_replay as ar  # noqa: E402

KW = dict(truncation=True, max_length=262145, add_special_tokens=False)


class Added:
    def __init__(self, content, special=True, normalized=False, lstrip=False, rstrip=False, single_word=False):
        self.content, self.special, self.normalized = content, special, normalized
        self.lstrip, self.rstrip, self.single_word = lstrip, rstrip, single_word


class FakeTok:
    """Added tokens split first (leftmost, longest); a segment is cut into 3-char pieces from its own start."""
    SPECIAL = {'<|im_start|>': 1, '<|im_end|>': 2, '<|endoftext|>': 3}
    PLAIN = {'<think>': 4, '</think>': 5}

    def __init__(self, extra_empty=None, fast=True, added=None):
        self.added = dict(self.SPECIAL, **self.PLAIN) if added is None else added
        self.added_tokens_decoder = {tid: Added(c, special=c in self.SPECIAL) for c, tid in self.added.items()}
        self.backend_tokenizer = object() if fast else None
        self.extra_empty = extra_empty
        self.calls = []
        self.fail_on = None

    def segment(self, s):
        return [100 + _stable(s[i:i + 3]) for i in range(0, len(s), 3)]

    def encode(self, text):
        out, i, seg = [], 0, []
        names = sorted(self.added, key=len, reverse=True)
        while i < len(text):
            for name in names:
                if text.startswith(name, i):
                    if seg:
                        out.extend(self.segment(''.join(seg)))
                        seg = []
                    out.append(self.added[name])
                    i += len(name)
                    break
            else:
                seg.append(text[i])
                i += 1
        if seg:
            out.extend(self.segment(''.join(seg)))
        return out

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        self.calls.append(len(text))
        if self.fail_on is not None and self.fail_on in text:
            raise RuntimeError('tokenizer failure')
        ids = self.encode(text)
        if add_special_tokens or (self.extra_empty and not text):
            ids = [9] + ids
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {'input_ids': ids}


def _stable(s):
    h = 0
    for ch in s:
        h = (h * 1000003 + ord(ch)) % 1000000007
    return h % 50000


def render(messages, gen=True):
    """The Qwen3-style shape: history assistant turns without think; the generation prompt with the empty think block."""
    out = []
    for m in messages:
        out.append('<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n')
    if gen:
        out.append('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    return ''.join(out)


UNI = ['naïve', '数据', '😀', 'é', ' ', '\t', '\r\n', '<think>', '</think>', '<|im_end', 'ﬁ']


def conversation(seed, turns, unicode=False, scale=40):
    rng = random.Random(seed)

    def words(n):
        w = ar.text(rng, n).split(' ')
        if unicode:
            for _ in range(max(1, len(w) // 8)):
                w.insert(rng.randrange(len(w) + 1), rng.choice(UNI))
        return ' '.join(w)
    messages = [{'role': 'system', 'content': 'You are coding agent %d, session 0.' % seed},
                {'role': 'user', 'content': 'Repository context:\n' + words(scale * 20)}]
    for _ in range(turns):
        yield render(messages)
        messages.append({'role': 'assistant', 'content': words(scale)})
        messages.append({'role': 'user', 'content': 'Tool output:\n' + words(scale * 3)})


class SpliceExact(unittest.TestCase):
    def test_replay_shaped_conversations_equal_full(self):
        tok = FakeTok()
        sp = ii.Splicer(tok, verify_every=0)
        n = 0
        for seed in range(6):
            for text in conversation(seed, 8, unicode=seed % 2 == 1):
                got = sp.tokenize(text, dict(KW), lambda: tok(text, **KW)['input_ids'])
                self.assertEqual(got, tok.encode(text))
                n += 1
        snap = sp.snapshot()
        self.assertEqual(snap['counts']['spliced'], 6 * 7)
        self.assertEqual(snap['counts']['full'], 6)
        self.assertEqual(snap['fallbacks'], {'no-entry': 6})
        self.assertEqual(n, 48)

    def test_cut_is_after_the_last_join_inside_the_common_prefix(self):
        tok = FakeTok()
        sp = ii.Splicer(tok, verify_every=0)
        a = render([{'role': 'user', 'content': 'abc def'}])
        sp.tokenize(a, dict(KW), lambda: tok(a, **KW)['input_ids'])
        b = render([{'role': 'user', 'content': 'abc def'}, {'role': 'assistant', 'content': 'xyz'},
                    {'role': 'user', 'content': 'more'}])
        tok.calls.clear()
        got = sp.tokenize(b, dict(KW), lambda: self.fail('full called'))
        self.assertEqual(got, tok.encode(b))
        cut = a.index('<|im_start|>assistant') + len('<|im_start|>')
        self.assertEqual(tok.calls, [len(b) - cut])      # only the text after the join is tokenised

    def test_a_cut_off_a_join_would_differ(self):
        """The control: the fake tokenizer is position-dependent inside a segment, so only join cuts are exact."""
        tok = FakeTok()
        text = render([{'role': 'user', 'content': 'abcdefgh'}])
        cut = text.index('cd')
        self.assertNotEqual(tok.encode(text[:cut]) + tok.encode(text[cut:]), tok.encode(text))

    def test_identical_retry_and_shorter_text(self):
        tok = FakeTok()
        sp = ii.Splicer(tok, verify_every=0)
        texts = list(conversation(3, 3))
        for t in texts + [texts[-1], texts[0]]:
            self.assertEqual(sp.tokenize(t, dict(KW), lambda: tok(t, **KW)['input_ids']), tok.encode(t))


class Fallbacks(unittest.TestCase):
    def run_two(self, tok, a, b, kwargs=None, sp=None):
        sp = sp or ii.Splicer(tok, verify_every=0)
        kw = dict(KW) if kwargs is None else kwargs
        sp.tokenize(a, kw, lambda: tok(a, **kw)['input_ids'])
        got = sp.tokenize(b, kw, lambda: tok(b, **kw)['input_ids'])
        self.assertEqual(got, tok(b, **kw)['input_ids'])
        return sp

    def test_no_entry_and_no_join(self):
        tok = FakeTok()
        a = 'plain text without a join ' * 200          # longer than KEY_CHARS: one key, no join to cut at
        b = a + 'longer'
        sp = self.run_two(tok, a, b)
        self.assertEqual(sp.fallbacks['no-entry'], 1)
        # same key (opening text), no join in the cached text: full call
        self.assertEqual(sp.fallbacks['no-join'], 1)

    def test_diverged_before_the_first_join(self):
        tok = FakeTok()
        a = 'X' * 10 + '<|im_end|>' + 'tail'
        b = 'X' * 9 + 'Y<|im_end|>tail'
        sp = ii.Splicer(tok, verify_every=0)
        sp.key = lambda text, kwargs: 'same'       # force one entry: correctness must rest on the text only
        self.run_two(tok, a, b, sp=sp)
        self.assertEqual(sp.fallbacks['no-join'], 1)

    def test_other_encode_arguments(self):
        tok = FakeTok()
        texts = list(conversation(1, 2))
        for kw in (dict(KW, add_special_tokens=True), dict(KW, return_offsets_mapping=True)):
            sp = ii.Splicer(tok, verify_every=0)
            for t in texts:
                sp.tokenize(t, kw, lambda: tok(t, add_special_tokens=kw['add_special_tokens'])['input_ids'])
            self.assertEqual(sp.fallbacks, {'encode-arguments': 2})
            self.assertEqual(sp.snapshot()['entries'], 0)

    def test_over_truncation_length(self):
        tok = FakeTok()
        a, b = list(conversation(2, 2))
        n = len(tok.encode(b))
        kw = dict(KW, max_length=n - 1)
        sp = self.run_two(tok, a, b, kwargs=kw)
        self.assertEqual(sp.fallbacks['over-truncation-length'], 1)

    def test_disabled_tokenizers(self):
        cases = {'not-a-fast-tokenizer': FakeTok(fast=False), 'no-join-token': FakeTok(added={'<think>': 4}),
                 'empty-encode-adds-tokens': FakeTok(extra_empty=True)}
        for reason, tok in cases.items():
            sp = ii.Splicer(tok, verify_every=0)
            self.assertEqual(sp.disabled, reason)
            for t in conversation(4, 2):
                self.assertEqual(sp.tokenize(t, dict(KW), lambda: tok(t, **KW)['input_ids']), tok(t, **KW)['input_ids'])
            self.assertEqual(sp.fallbacks, {reason: 2})

    def test_join_contained_in_another_added_token_is_not_a_join(self):
        tok = FakeTok(added={'<|im_start|>': 1, '<|im_end|>': 2, '<|im_end|>x': 6})
        sp = ii.Splicer(tok, verify_every=0)
        self.assertEqual(sorted(sp.joins), ['<|im_start|>'])          # <|im_end|> is inside <|im_end|>x

    def test_splice_error_takes_the_full_call(self):
        tok = FakeTok()
        a, b = list(conversation(5, 2))
        sp = ii.Splicer(tok, verify_every=0)
        sp.tokenize(a, dict(KW), lambda: tok(a, **KW)['input_ids'])
        tok.fail_on = 'Tool output'
        full = FakeTok()
        got = sp.tokenize(b, dict(KW), lambda: full(b, **KW)['input_ids'])
        self.assertEqual(got, full.encode(b))
        self.assertEqual(sp.fallbacks, {'no-entry': 1, 'splice-error:RuntimeError': 1})

    def test_stock_error_propagates_unchanged(self):
        tok = FakeTok()
        sp = ii.Splicer(tok, verify_every=0)

        def boom():
            raise ValueError('stock refusal')
        with self.assertRaisesRegex(ValueError, 'stock refusal'):
            sp.tokenize('<|im_start|>x', dict(KW), boom)

    def test_verify_every_and_mismatch(self):
        tok = FakeTok()
        texts = list(conversation(6, 6))
        sp = ii.Splicer(tok, verify_every=2)
        for t in texts:
            sp.tokenize(t, dict(KW), lambda: tok(t, **KW)['input_ids'])
        self.assertEqual(sp.counts['verified'], 2)
        self.assertNotIn('verify_mismatch', sp.counts)
        # a stock path that disagrees: the stock ids are returned and counted
        sp = ii.Splicer(tok, verify_every=1)
        sp.tokenize(texts[0], dict(KW), lambda: tok(texts[0], **KW)['input_ids'])
        other = tok.encode(texts[1]) + [77]
        self.assertEqual(sp.tokenize(texts[1], dict(KW), lambda: list(other)), other)
        self.assertEqual(sp.counts['verify_mismatch'], 1)
        self.assertEqual(sp.snapshot()['entries'], 1)

    def test_store_refused_drops_the_entry(self):
        tok = FakeTok()
        sp = ii.Splicer(tok, verify_every=0)
        a = render([{'role': 'user', 'content': 'q'}])
        sp.tokenize(a, dict(KW), lambda: tok(a, **KW)['input_ids'])
        self.assertEqual(sp.snapshot()['entries'], 1)
        eid = next(iter(sp.cache))                  # a store whose ids miss the text's joins replaces nothing
        sp._store(sp.key(a, dict(KW)), a, [2, 2, 2, 2, 2, 2, 2], None, replace=eid)
        self.assertEqual(sp.counts['store_refused'], 1)
        self.assertEqual(sp.snapshot()['entries'], 0)


class LeftTok(FakeTok):
    """truncation_side='left': a truncated result keeps the END of the ids (not a prefix of the full ids)."""

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None):
        ids = FakeTok.__call__(self, text, add_special_tokens=add_special_tokens)['input_ids']
        if truncation and max_length is not None and len(ids) > max_length:
            ids = ids[len(ids) - max_length:]
        return {'input_ids': ids}


def budget_kw(tok, text, rng):
    """vLLM's encode arguments for one turn: max_length = max_model_len - max_tokens + 1, a fresh reply budget per turn."""
    return dict(truncation=True, max_length=len(tok.encode(text)) + 1 + rng.randrange(1, 30000), add_special_tokens=False)


class KeyFix(unittest.TestCase):
    """Ingress key fix (lead 13 Nov): the per-request length bound is not in the key; it is applied to every result."""

    def test_turns_differing_only_in_max_length_splice(self):
        rng = random.Random(5)
        tok = FakeTok()
        sp = ii.Splicer(tok, verify_every=0)
        lengths = set()
        for seed in range(4):
            for text in conversation(seed, 6):
                kw = budget_kw(tok, text, rng)
                lengths.add(kw['max_length'])
                got = sp.tokenize(text, dict(kw), lambda: tok(text, **kw)['input_ids'])
                self.assertEqual(got, tok(text, **kw)['input_ids'])
                self.assertEqual(got, tok.encode(text))
        self.assertEqual(len(lengths), 24)                       # no two turns share a bound
        snap = sp.snapshot()
        self.assertEqual((snap['counts']['spliced'], snap['counts']['full']), (4 * 5, 4))   # every non-first turn
        self.assertEqual(snap['fallbacks'], {'no-entry': 4})

    def test_every_max_length_equals_stock(self):
        tok = FakeTok()
        a, b = list(conversation(7, 2))
        n = len(tok.encode(b))
        for bound in (1, len(tok.encode(a)), n - 2, n - 1, n, n + 1, n + 2, 262145):
            for truncation in (True, None, False):
                with self.subTest(bound=bound, truncation=truncation):
                    kw = dict(truncation=truncation, max_length=bound, add_special_tokens=False)
                    sp = ii.Splicer(tok, verify_every=0)
                    warm = dict(KW)
                    sp.tokenize(a, warm, lambda: tok(a, **warm)['input_ids'])
                    stock = tok(b, **kw)['input_ids']
                    got = sp.tokenize(b, dict(kw), lambda: tok(b, **kw)['input_ids'])
                    self.assertEqual(got, stock)
                    bounded = truncation is not False
                    if bounded and n > bound:
                        self.assertEqual(sp.fallbacks['over-truncation-length'], 1)
                        self.assertEqual(sp.counts['spliced'], 0)
                    else:
                        self.assertEqual(sp.counts['spliced'], 1)

    def test_a_result_at_its_bound_is_never_a_splice_source(self):
        for tok in (FakeTok(), LeftTok()):
            a, b = list(conversation(8, 2))
            for cut in (0, 1, 5):
                with self.subTest(tok=type(tok).__name__, cut=cut):
                    sp = ii.Splicer(tok, verify_every=0)
                    kw = dict(KW, max_length=len(tok.encode(a)) - cut)     # stock truncates a (or meets its bound)
                    self.assertEqual(sp.tokenize(a, dict(kw), lambda: tok(a, **kw)['input_ids']), tok(a, **kw)['input_ids'])
                    self.assertEqual((sp.counts['store_length_bound'], sp.snapshot()['entries']), (1, 0))
                    got = sp.tokenize(b, dict(KW), lambda: tok(b, **KW)['input_ids'])
                    self.assertEqual(got, tok.encode(b))
                    self.assertEqual((sp.counts['spliced'], sp.fallbacks['no-entry']), (0, 2))

    def test_other_arguments_still_separate_entries(self):
        tok = FakeTok()
        sp = ii.Splicer(tok, verify_every=0)
        text = next(conversation(9, 1))
        base = sp.key(text, dict(KW))
        for kw in (dict(KW, max_length=7), dict(KW, truncation=False), dict(add_special_tokens=False),
                   dict(KW, truncation=None)):
            self.assertEqual(sp.key(text, kw), base, kw)          # the bound only: one conversation, one key
        self.assertNotEqual(sp.key(text, dict(KW, add_special_tokens=True)), base)
        self.assertNotEqual(sp.key(next(conversation(10, 1)), dict(KW)), base)   # another conversation
        # stock-only arguments never store, so the next turn finds no entry of theirs
        a, b = list(conversation(11, 2))
        for kw in (dict(KW, add_special_tokens=True), dict(KW, return_offsets_mapping=True),
                   dict(truncation=True, add_special_tokens=False)):       # truncation to model_max_length
            with self.subTest(kw=kw):
                sp = ii.Splicer(tok, verify_every=0)
                sp.tokenize(a, kw, lambda: tok(a, add_special_tokens=kw['add_special_tokens'])['input_ids'])
                self.assertEqual(sp.fallbacks['encode-arguments'], 1)
                got = sp.tokenize(b, dict(KW), lambda: tok(b, **KW)['input_ids'])
                self.assertEqual(got, tok.encode(b))
                self.assertEqual((sp.counts['spliced'], sp.fallbacks['no-entry']), (0, 1))

    def test_inline_and_async_verify_with_per_turn_bounds(self):
        rng = random.Random(6)
        for verify_async in (False, True):
            tok = FakeTok()
            sp = ii.Splicer(tok, verify_every=1, verify_async=verify_async)
            for text in conversation(12, 5):
                kw = budget_kw(tok, text, rng)
                self.assertEqual(sp.tokenize(text, dict(kw), lambda: tok(text, **kw)['input_ids']), tok.encode(text))
            self.assertTrue(sp.drain())
            snap = sp.snapshot()
            self.assertEqual((snap['counts']['spliced'], snap['counts'].get('verified'), snap['counts'].get('verify_mismatch', 0),
                              snap['disabled']), (4, 4, 0, None), verify_async)


class Bounds(unittest.TestCase):
    def test_lru_entries(self):
        tok = FakeTok()
        sp = ii.Splicer(tok, verify_every=0, max_entries=3)
        firsts = [next(conversation(s, 1)) for s in range(5)]
        for t in firsts:
            sp.tokenize(t, dict(KW), lambda: tok(t, **KW)['input_ids'])
        self.assertEqual(sp.snapshot()['entries'], 3)
        self.assertEqual(sp.counts['evicted'], 2)
        keys = [e[0] for e in sp.cache.values()]
        self.assertEqual(keys, [sp.key(t, dict(KW)) for t in firsts[2:]])
        # a hit replaces its entry at the end
        sp.tokenize(firsts[2], dict(KW), lambda: self.fail('full'))
        self.assertEqual([e[0] for e in sp.cache.values()][-1], sp.key(firsts[2], dict(KW)))
        self.assertEqual(sp.snapshot()['entries'], 3)

    def test_shared_system_prompt_keeps_candidates(self):
        tok = FakeTok()
        sp = ii.Splicer(tok, verify_every=0)
        convs = []
        for seed in range(3):
            msgs = [{'role': 'system', 'content': 'shared'}, {'role': 'user', 'content': 'task %d ' % seed + 'x' * seed}]
            convs.append([render(msgs), render(msgs + [{'role': 'assistant', 'content': 'r%d' % seed},
                                                       {'role': 'user', 'content': 'tool %d' % seed}])])
        self.assertEqual(len({sp.key(c[0], dict(KW)) for c in convs}), 1)
        for c in convs:
            sp.tokenize(c[0], dict(KW), lambda: tok(c[0], **KW)['input_ids'])
        self.assertEqual(sp.counts['spliced'], 2)          # later first turns cut after the shared system message
        for c in convs:
            self.assertEqual(sp.tokenize(c[1], dict(KW), lambda: tok(c[1], **KW)['input_ids']), tok.encode(c[1]))
        self.assertEqual(sp.counts['spliced'], 5)
        for i in range(5):                                # PER_KEY bounds one key's candidates
            t = render([{'role': 'system', 'content': 'shared'}, {'role': 'user', 'content': 'other %d' % i}])
            sp.tokenize(t, dict(KW), lambda: tok(t, **KW)['input_ids'])
        self.assertEqual(len(sp.by_key[sp.key(convs[0][0], dict(KW))]), ii.PER_KEY)

    def test_key_is_stable_for_short_first_messages(self):
        tok = FakeTok()
        texts = list(conversation(2, 4, scale=2))
        self.assertLess(len(texts[0]), ii.KEY_CHARS)
        sp = ii.Splicer(tok, verify_every=0)
        self.assertEqual(len({sp.key(t, dict(KW)) for t in texts}), 1)

    def test_byte_bound_and_accounting(self):
        tok = FakeTok()
        t = next(conversation(1, 1))
        one = len(t) * 4 + len(tok.encode(t)) * 4 + 16 * t.count('<|im_')
        sp = ii.Splicer(tok, verify_every=0, max_bytes=one * 2 + 10)
        for s in range(4):
            x = next(conversation(s, 1))
            sp.tokenize(x, dict(KW), lambda: tok(x, **KW)['input_ids'])
        snap = sp.snapshot()
        self.assertLessEqual(snap['bytes'], snap['max_bytes'])
        self.assertEqual(snap['bytes'], sum(sp.sizes.values()))
        self.assertEqual(set(sp.sizes), set(sp.cache))
        sp.clear()
        self.assertEqual((sp.snapshot()['entries'], sp.snapshot()['bytes'], sp.sizes), (0, 0, {}))


class Concurrency(unittest.TestCase):
    def test_eight_conversations_eight_threads(self):
        tok = FakeTok()
        sp = ii.Splicer(tok, verify_every=3, max_entries=8)
        convs = [list(conversation(s, 10, unicode=s % 2 == 0, scale=20)) for s in range(8)]
        bad, barrier = [], threading.Barrier(8)

        def run(i):
            barrier.wait()
            for t in convs[i]:
                if sp.tokenize(t, dict(KW), lambda: tok(t, **KW)['input_ids']) != tok.encode(t):
                    bad.append(i)
        threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(bad, [])
        snap = sp.snapshot()
        self.assertEqual(snap['entries'], 8)
        self.assertEqual(snap['counts']['spliced'] + snap['counts']['full'], 80)
        self.assertEqual(snap['counts']['full'], 8)
        self.assertNotIn('verify_mismatch', snap['counts'])


class BindingMirrorsEngine(unittest.TestCase):
    """The API mirror against native_auto_arm.AutoArmPolicy.manual_control's own lease rule."""

    def engine(self):
        import native_auto_arm as auto
        d = types.SimpleNamespace(engine_epoch=7, active=False, _split_ready=False, _engine_paused=lambda: True,
                                  admission_intent=lambda: None, _clock_record=lambda n: {}, _monotonic_ns=lambda: 0)
        policy = types.SimpleNamespace(_internal=False, active=False, manual_owned=False, driver=d,
                                       engine=types.SimpleNamespace(batch_queue=None))
        return auto, policy

    # API action -> the engine commands it sends through manual_control (None: stock resume_generation, no command)
    SENDS = {'pause': ['pause', 'barrier', 'status'], 'status': ['status'], 'resume': None, 'identity': ['identity'],
             'native_progress': ['native_progress'], 'native_stage_local': ['native_stage_local'],
             'native_append_yield': ['native_append_yield'], 'release_auto_arm': ['release_auto_arm']}

    def drive(self, actions):
        auto, policy = self.engine()
        b = ii.Binding()
        for action in actions:
            for cmd in self.SENDS[action] or ():
                command = {'action': cmd} if cmd != 'release_auto_arm' else {'action': cmd, 'engine_epoch': 7}
                auto.AutoArmPolicy.manual_control(policy, command, lambda c: {'ok': True})
            result = {'released': True} if action == 'release_auto_arm' else {}
            if action in ('status', 'pause'):
                result['auto_arm'] = {'manual_owned': policy.manual_owned}
            b.observe(action, result)
            self.assertEqual(b.selected, not policy.manual_owned, (actions, action))
        return b

    def test_n3_arm_is_selected(self):
        b = self.drive(['status', 'pause', 'native_stage_local', 'native_append_yield', 'native_progress',
                        'release_auto_arm', 'resume', 'status', 'status'])
        self.assertTrue(b.selected)

    def test_sp_arm_holds_the_lease(self):
        b = self.drive(['status', 'pause', 'native_progress', 'release_auto_arm', 'resume', 'pause', 'resume', 'status'])
        self.assertFalse(b.selected)

    def test_identity_and_next_arm(self):
        b = self.drive(['pause', 'release_auto_arm', 'resume', 'identity', 'status', 'pause', 'release_auto_arm'])
        self.assertTrue(b.selected)

    def test_unreleased_reply_and_odd_results(self):
        b = ii.Binding()
        b.observe('release_auto_arm', {'released': False})
        b.observe('status', {'auto_arm': {'manual_owned': 'no'}})
        b.observe('status', None)
        self.assertFalse(b.selected)
        b.observe('status', {'auto_arm': {'manual_owned': False}})
        self.assertTrue(b.selected)
        b.observe('anything_else', {})
        self.assertFalse(b.selected)
        self.assertEqual(b.snapshot()['flips'], {'on:status': 1, 'off:anything_else': 1})

    def test_select_clears_the_cache(self):
        cleared = []
        b = ii.Binding(on_select=lambda: cleared.append(1))
        b.observe('release_auto_arm', {'released': True})
        b.observe('status', {'auto_arm': {'manual_owned': False}})     # already on: no second clear
        b.observe('pause', {})
        b.observe('release_auto_arm', {'released': True})
        self.assertEqual(len(cleared), 2)


class FakeRenderer:
    def __init__(self, tok):
        self.tok, self.stock_calls = tok, []
        self._executor = None

    def get_tokenizer(self):
        return self.tok

    def _wants_offsets(self, prompt, params):
        return params.offsets

    @staticmethod
    def _build_tokens_prompt(token_ids, prompt, *, offset_mapping=None):
        return dict(prompt_token_ids=list(token_ids), **prompt)

    def _tokenize_prompt(self, prompt, params):
        self.stock_calls.append(prompt)
        enc = self.tok(prompt['prompt'], **params.get_encode_kwargs())
        return self._build_tokens_prompt(enc['input_ids'], prompt)


class Params:
    def __init__(self, offsets=False, **kw):
        self.offsets, self.kw = offsets, dict(KW, **kw)

    def get_encode_kwargs(self):
        return dict(self.kw)


class Wrapper(unittest.TestCase):
    def test_not_selected_is_the_stock_call_and_object(self):
        r = FakeRenderer(FakeTok())
        ing = ii.Ingress(r)
        sentinel = object()
        r._tokenize_prompt = lambda prompt, params: sentinel
        ing.original = r._tokenize_prompt
        p = {'prompt': 'x'}
        self.assertIs(ing.tokenize_prompt(p, Params()), sentinel)
        self.assertEqual(ing.splicer.snapshot()['counts'], {})

    def test_selected_splices_and_shapes(self):
        tok = FakeTok()
        r = FakeRenderer(tok)
        ing = ii.Ingress(r)
        ing.binding.observe('release_auto_arm', {'released': True})
        texts = list(conversation(9, 4))
        for i, t in enumerate(texts):
            out = ing.tokenize_prompt({'prompt': t}, Params())
            self.assertEqual(out, {'prompt_token_ids': tok.encode(t), 'prompt': t})
        self.assertEqual(len(r.stock_calls), 1)                 # the first turn only
        self.assertEqual(ing.counts['spliced_prompts'], 3)
        snap = ing.snapshot()
        self.assertEqual((snap['process'], snap['binding']['selected']), ('api', True))

    def test_selected_but_stock_shapes(self):
        tok = FakeTok()
        r = FakeRenderer(tok)
        ing = ii.Ingress(r)
        ing.binding.observe('release_auto_arm', {'released': True})
        for prompt, params in (({'prompt': '<|im_start|>a'}, Params(offsets=True)),
                               ({'prompt': '<|im_start|>a', 'multi_modal_data': {'image': 1}}, Params()),
                               ({'prompt_token_ids': [1]}, Params())):
            if 'prompt' in prompt:
                ing.tokenize_prompt(prompt, params)
            else:
                with self.assertRaises(KeyError):
                    ing.tokenize_prompt(prompt, params)
        self.assertEqual(ing.counts['stock_shape'], 3)
        self.assertEqual(ing.splicer.snapshot()['counts'], {})


class Install(unittest.TestCase):
    def test_flag_unset_or_zero_installs_nothing(self):
        r = FakeRenderer(FakeTok())
        before = r._tokenize_prompt
        for env in ({}, {ii.ENV: '0'}, {ii.ENV: 'yes'}):
            self.assertIsNone(ii.install(types.SimpleNamespace(renderer=r), env))
        self.assertEqual(r._tokenize_prompt, before)
        self.assertFalse(hasattr(r, '_flash_s5_ingress'))

    def test_flag_one_wraps_once(self):
        made = []
        fake = types.ModuleType('vllm.utils.async_utils')

        def make_async(fn, executor=None):
            made.append((fn, executor))

            async def run(*a, **k):
                return fn(*a, **k)
            return run
        fake.make_async = make_async
        saved = {k: sys.modules.get(k) for k in ('vllm', 'vllm.utils', 'vllm.utils.async_utils')}
        sys.modules.update({'vllm': types.ModuleType('vllm'), 'vllm.utils': types.ModuleType('vllm.utils'),
                            'vllm.utils.async_utils': fake})
        try:
            tok = FakeTok()
            r = FakeRenderer(tok)
            engine = types.SimpleNamespace(renderer=r)
            ing = ii.install(engine, {ii.ENV: '1'})
            self.assertIs(ii.install(engine, {ii.ENV: '1'}), ing)
            self.assertEqual(len(made), 1)
            ing.binding.observe('release_auto_arm', {'released': True})
            t = next(conversation(1, 1))
            out = asyncio.run(r._tokenize_prompt_async({'prompt': t}, Params()))
            self.assertEqual(out['prompt_token_ids'], tok.encode(t))
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v


class Req:
    """A vLLM Request's fields the counter reads; block hashes chained over full blocks of prompt + output."""
    def __init__(self, salt, prompt, output=(), unit=4):
        self.cache_salt, self.prompt_token_ids, self.unit = salt, list(prompt), unit
        self.num_prompt_tokens = len(prompt)
        self.output_token_ids = []
        self.block_hashes = []
        self.extend(output)

    def extend(self, output):
        self.output_token_ids.extend(output)
        allt = self.prompt_token_ids + self.output_token_ids
        prev = self.block_hashes[-1] if self.block_hashes else 'none:' + str(self.cache_salt)
        for i in range(len(self.block_hashes) * self.unit, len(allt) - self.unit + 1, self.unit):
            prev = prev + '|' + ','.join(map(str, allt[i:i + self.unit]))
            self.block_hashes.append(prev)


def plain_lcp(a, b):
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k


class Counter(unittest.TestCase):
    def test_lcp_matches_a_plain_scan_near_the_end_and_blocks_deeper(self):
        rng = random.Random(3)
        c = ii.PrefixCounter(4)
        exact_seen = deep_seen = 0
        for _ in range(400):
            n = rng.randint(0, 60)
            prompt = [rng.randint(0, 3) for _ in range(n)]
            out = [rng.randint(0, 3) for _ in range(rng.randint(0, 12))]
            old = Req('s', prompt)
            old.extend(out)                                   # outputs arrive after admission
            whole = prompt + out
            new = Req('s', whole[:rng.randint(0, len(whole))] + [rng.randint(0, 3) for _ in range(rng.randint(0, 20))])
            p = len(prompt)
            prev = (p, prompt[max(0, p - 8):], old.output_token_ids, old.block_hashes)
            k, exact = c.lcp(new.prompt_token_ids, new.block_hashes, prev)
            want = plain_lcp(new.prompt_token_ids, whole)
            if exact:
                self.assertEqual(k, want)
                exact_seen += 1
            else:
                self.assertTrue(k <= want < k + 4 and k < p - 8, (k, want, p))
                deep_seen += 1
        self.assertGreater(exact_seen, 100)
        self.assertGreater(deep_seen, 20)

    def test_buckets_and_through_reply(self):
        c = ii.PrefixCounter(4)
        base = list(range(10, 50))                       # 40 prompt tokens
        reply = [70, 71, 72, 2]

        def turn(salt, prompt, output=()):
            r = Req(salt, prompt)
            c.record(r)                                  # admitted before it generates
            r.extend(output)
            return r
        turn('a', base, reply)
        turn('a', base[:-4] + reply[:2] + [99])                       # think block dropped: -4
        turn('a', base[:-4] + [5])                                    # previous prompt 39 tokens, lcp 36: -3
        turn('b', base, reply)
        turn('b', base + reply + [1, 2])                              # through the whole reply
        turn('c', base, reply)
        turn('c', base + reply[:3])                                   # all but the stop token
        turn('d', base, reply)
        turn('d', [0] + base)                                         # diverged at token 0: block resolved
        c.record(Req(None, base))
        snap = c.snapshot()
        self.assertEqual(snap['counts'], {'first': 4, 'compared': 5, 'through_reply': 1, 'through_reply_but_last': 2,
                                          'block_resolved': 1, 'no_salt': 1})
        self.assertEqual(snap['delta_vs_prev_prompt_end'], {'-7..-1': 2, 'whole_reply': 1, 'reply_but_last': 1,
                                                            '-63..-8': 1})
        self.assertEqual(snap['sums']['lcp'], 36 + 36 + 44 + 43 + 0)
        self.assertLessEqual(max(len(v[1]) for v in c.prev.values()), 8)  # only the prompt tail is copied

    def test_eviction_errors_and_no_unit(self):
        c = ii.PrefixCounter(4, max_entries=2)
        for s in 'abc':
            c.record(Req(s, [1, 2, 3, 4]))
        self.assertEqual((c.counts['evicted'], list(c.prev)), (1, ['b', 'c']))
        c.record(Req('a', [1, 2, 3, 4]))                 # night-pack fix 4: evicted, not counted as first again
        self.assertEqual((c.counts['first'], c.counts['after_eviction']), (3, 1))
        self.assertEqual(ii.COUNTER_ENTRIES, 32)
        c.record(types.SimpleNamespace(cache_salt='c'))
        self.assertEqual(c.counts['errors'], 1)
        z = ii.PrefixCounter(0)
        z.record(Req('a', [1, 2]))
        self.assertEqual(dict(z.counts), {'no_hash_unit': 1})


class DriverHook(unittest.TestCase):
    def test_note_only_while_released_and_status_block(self):
        import engine_driver as ed
        engine = types.SimpleNamespace(scheduler=types.SimpleNamespace(hash_block_size=4))
        d = types.SimpleNamespace(auto_arm=types.SimpleNamespace(manual_owned=True),
                                  controller=types.SimpleNamespace(native_admission_record=lambda: {'admitted': 0}))
        note = ed.NativeSegmentDriver.note_ingress_prefix
        note(d, engine, Req('a', [1, 2, 3, 4, 5]))
        self.assertFalse(hasattr(d, '_ingress_prefix'))
        self.assertNotIn('ingress_prefix', ed.NativeSegmentDriver._native_admission_record(d))
        d.auto_arm.manual_owned = False
        note(d, engine, Req('a', [1, 2, 3, 4, 5]))
        note(d, engine, Req('a', [1, 2, 3, 4, 6]))
        block = ed.NativeSegmentDriver._native_admission_record(d)['ingress_prefix']
        self.assertEqual((block['unit'], block['counts']), (4, {'first': 1, 'compared': 1, 'through_reply_but_last': 1}))
        none = types.SimpleNamespace(controller=None)
        note(none, engine, Req('a', [1]))
        self.assertFalse(hasattr(none, '_ingress_prefix'))

    def test_executor_hook_calls_and_swallows(self):
        import executor_hooks as eh
        seen = []
        engine = types.SimpleNamespace(_flash_s5_driver=types.SimpleNamespace(
            note_ingress_prefix=lambda e, r: seen.append(r) or (_ for _ in ()).throw(RuntimeError('x'))))
        req = types.SimpleNamespace(request_id='r', num_prompt_tokens=1)
        eh.engine_request_admitted(engine, req)
        self.assertEqual(seen, [req])
        eh.engine_request_admitted(types.SimpleNamespace(), req)


class Sources(unittest.TestCase):
    def test_router_installs_only_under_the_flag(self):
        src = (ROOT / 'patches/s5_engine/probe_control.py').read_text()
        tree = ast.parse(src)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'attach_native_router')
        text = ast.get_source_segment(src, fn)
        self.assertIn("ingress = None", text)
        self.assertIn("if env.get('FLASH_S5_INGRESS_INCREMENTAL', '0') == '1':\n"
                      "                    ingress = state['ingress'] = _install_ingress(engine, env)", text)
        self.assertEqual(text.count('ingress.'), 0, 'night-pack fix 2: every ingress call goes through a guard')
        self.assertEqual(text.count('_observe_ingress(ingress, action, result)'), 1)
        self.assertEqual(text.count('if ingress is not None:'), 1)
        self.assertNotIn('import install', text)

    def test_overlay_declares_the_module(self):
        import make_s5_overlap as g
        import decode_probe
        self.assertIn('ingress_incremental', g.E3_MODULES)
        self.assertIn('ingress_incremental', g.LAUNCHER_LIST_EXEMPT)
        self.assertIn('ingress_incremental', decode_probe.GENERATED_SOURCES)


# -- night-pack fixes 2-4 (lead, 12 Nov 2026) ---------------------------------------------------------------------
class InstallSafety(unittest.TestCase):
    """fix 2: an exception in install, observe or snapshot never reaches the native command path."""

    def test_install_error_is_reported_never_raised(self):
        import probe_control as pc
        with mock.patch.object(ii, 'install', side_effect=RuntimeError('renderer shape')):
            got = pc._install_ingress(types.SimpleNamespace(renderer=object()), {ii.ENV: '1'})
        self.assertEqual(got.snapshot(), {'disabled': 'install-error:RuntimeError'})
        result = {'native_admission': {}}
        pc._observe_ingress(got, 'status', result)
        self.assertEqual(result['native_admission']['ingress_api'], {'disabled': 'install-error:RuntimeError'})
        pc._observe_ingress(got, 'release_auto_arm', {'released': True})       # no raise, nothing selected
        with mock.patch.dict(sys.modules, {'ingress_incremental': None}):           # the import itself fails
            self.assertIn(pc._install_ingress(object(), {ii.ENV: '1'}).snapshot()['disabled'],
                          ('install-error:ImportError', 'install-error:ModuleNotFoundError'))
        none = pc._install_ingress(types.SimpleNamespace(renderer=None), {ii.ENV: '1'})
        self.assertEqual(none.snapshot(), {'disabled': 'install-error:no-renderer'})

    def test_partial_install_restores_the_stock_renderer(self):
        r = FakeRenderer(FakeTok())
        r._tokenize_prompt_async = 'stock-async'
        before = r._tokenize_prompt
        with fake_async(), mock.patch.object(ii.Ingress, 'async_wrapper', side_effect=TypeError('x')):
            with self.assertRaises(TypeError):
                ii.install(types.SimpleNamespace(renderer=r), {ii.ENV: '1'})
        self.assertEqual((r._tokenize_prompt, r._tokenize_prompt_async), (before, 'stock-async'))
        self.assertFalse(hasattr(r, '_flash_s5_ingress'))

    def test_observe_and_snapshot_faults_deselect_and_report(self):
        import probe_control as pc
        r = FakeRenderer(FakeTok())
        ing = ii.Ingress(r)
        ing.binding.observe('release_auto_arm', {'released': True})
        with mock.patch.object(ing.binding, 'observe', side_effect=KeyError('x')):
            pc._observe_ingress(ing, 'pause', {})
        self.assertFalse(ing.binding.selected)
        self.assertEqual(ing.splicer.disabled, 'observe-error:KeyError')
        result = {'native_admission': {}}
        with mock.patch.object(ing, 'snapshot', side_effect=ValueError('y')):
            pc._observe_ingress(ing, 'status', result)
        self.assertEqual(result['native_admission']['ingress_api'], {'disabled': 'snapshot-error:ValueError'})
        odd = {'native_admission': 'not-a-dict'}
        pc._observe_ingress(ing, 'status', odd)                 # untouched, no raise
        self.assertEqual(odd, {'native_admission': 'not-a-dict'})

    def test_bad_verify_setting_is_an_install_error(self):
        import probe_control as pc
        r = FakeRenderer(FakeTok())
        with fake_async():
            for env in ({ii.VERIFY_EVERY_ENV: 'all'}, {ii.VERIFY_EVERY_ENV: '-1'}, {ii.VERIFY_ASYNC_ENV: 'yes'}):
                got = pc._install_ingress(types.SimpleNamespace(renderer=r), dict(env, **{ii.ENV: '1'}))
                self.assertEqual(got.snapshot(), {'disabled': 'install-error:ValueError'}, env)
                self.assertFalse(hasattr(r, '_flash_s5_ingress'))


def fake_async():
    """vllm.utils.async_utils.make_async stand-in (the hub has no vllm)."""
    fake = types.ModuleType('vllm.utils.async_utils')

    def make_async(fn, executor=None):
        async def run(*a, **k):
            return fn(*a, **k)
        return run
    fake.make_async = make_async
    return mock.patch.dict(sys.modules, {'vllm': types.ModuleType('vllm'), 'vllm.utils': types.ModuleType('vllm.utils'),
                                         'vllm.utils.async_utils': fake})


class Verification(unittest.TestCase):
    """fix 3: the settable rate, the off-path verifier (spliced ids returned at once), disable on the first mismatch."""

    def test_settings(self):
        self.assertEqual(ii.verify_settings({}), (16, False))
        self.assertEqual(ii.verify_settings({ii.VERIFY_EVERY_ENV: '1', ii.VERIFY_ASYNC_ENV: '1'}), (1, True))
        self.assertEqual(ii.verify_settings({ii.VERIFY_EVERY_ENV: '0'}), (0, False))
        for bad in ({ii.VERIFY_EVERY_ENV: 'x'}, {ii.VERIFY_EVERY_ENV: '-2'}, {ii.VERIFY_ASYNC_ENV: '2'}):
            with self.assertRaises(ValueError):
                ii.verify_settings(bad)
        ing = ii.Ingress(FakeRenderer(FakeTok()), {ii.VERIFY_EVERY_ENV: '1', ii.VERIFY_ASYNC_ENV: '1'})
        self.assertEqual((ing.splicer.verify_every, ing.splicer.verify_async), (1, True))

    def test_verify_all_off_path_matches_and_never_calls_stock_inline(self):
        tok = FakeTok()
        sp = ii.Splicer(tok, verify_every=1, verify_async=True)
        inline = []
        for seed in range(3):
            for t in conversation(30 + seed, 5, unicode=seed == 1):
                got = sp.tokenize(t, dict(KW), lambda: inline.append(1) or tok(t, **KW)['input_ids'])
                self.assertEqual(got, tok.encode(t))
        self.assertTrue(sp.drain())
        snap = sp.snapshot()
        self.assertEqual(len(inline), 3, 'only the three first turns take the stock call inline')
        self.assertEqual((snap['counts']['spliced'], snap['counts']['verify_submitted'], snap['counts']['verified']),
                         (12, 12, 12))
        self.assertEqual((snap['verify']['mismatches'], snap['disabled'], snap['verify']['pending']), ([], None, 0))
        self.assertNotIn('verify_mismatch', snap['counts'])

    def test_first_mismatch_disables_and_logs_index_and_position_only(self):
        tok = FakeTok()
        sp = ii.Splicer(tok, verify_every=1, verify_async=True)
        texts = list(conversation(41, 4))
        sp.tokenize(texts[0], dict(KW), lambda: tok(texts[0], **KW)['input_ids'])
        real_submit = sp._submit

        def wrong_splice(conversation, text, kwargs, packed):     # as if the splice had produced a wrong id at 5
            bad = array.array('I', packed)
            bad[5] = 7
            return real_submit(conversation, text, kwargs, bad)
        sp._submit = wrong_splice
        with self.assertLogs('vllm.flash_s5.ingress', level='WARNING') as logs:
            got = sp.tokenize(texts[1], dict(KW), lambda: tok(texts[1], **KW)['input_ids'])
            self.assertEqual(got, tok.encode(texts[1]), 'the spliced ids are returned at once (off-path verify)')
            self.assertTrue(sp.drain())
        self.assertEqual(sp.disabled, 'verify-mismatch')
        snap = sp.snapshot()
        self.assertEqual((snap['counts']['verify_mismatch'], snap['verify']['mismatches'], snap['entries']), (1, [[0, 5]], 0))
        self.assertEqual(logs.output, ['WARNING:vllm.flash_s5.ingress:flash-s5 ingress-verify-mismatch conversation=0 '
                                       'position=5'])
        calls = []
        sp.tokenize(texts[2], dict(KW), lambda: calls.append(1) or tok(texts[2], **KW)['input_ids'])
        self.assertEqual((calls, sp.fallbacks['verify-mismatch']), ([1], 1), 'disabled: the stock call from now on')

    def test_queue_bound_counts_unverified(self):
        sp = ii.Splicer(FakeTok(), verify_every=1, verify_async=True)
        with mock.patch.object(ii, 'VERIFY_QUEUE', 0):
            sp._submit(0, 'x', dict(KW), array.array('I'))
        self.assertEqual((sp.counts['verify_dropped'], sp.counts['verify_submitted']), (1, 0))

    def test_async_wrapper_times_submit_to_done_by_class(self):
        tok = FakeTok()
        r = FakeRenderer(tok)
        with fake_async():
            ing = ii.install(types.SimpleNamespace(renderer=r), {ii.ENV: '1'})
        t = next(conversation(2, 1))
        asyncio.run(r._tokenize_prompt_async({'prompt': t}, Params()))
        ing.binding.observe('release_auto_arm', {'released': True})
        asyncio.run(r._tokenize_prompt_async({'prompt': t}, Params()))
        timing = ing.snapshot()['timing']
        for key in ('async_stock_n', 'run_stock_n', 'async_n3_n', 'run_n3_n'):
            self.assertEqual(timing[key], 1, key)
        self.assertGreaterEqual(timing['async_n3_us'], timing['run_n3_us'])


class CounterWindows(unittest.TestCase):
    """fix 4: the id window around the first divergence (first appends only) and the reply-aligned match."""

    def turn(self, c, salt, prompt, output=()):
        r = Req(salt, prompt)
        c.record(r)
        r.extend(output)
        return r

    def test_reply_aligned_classes_and_offsets(self):
        c = ii.PrefixCounter(4)
        base, reply = list(range(10, 50)), [70, 71, 72, 2]
        self.turn(c, 'a', base, reply)
        self.turn(c, 'a', base + reply + [5])                        # whole reply at offset 0
        self.turn(c, 'b', base, reply)
        self.turn(c, 'b', base[:-4] + reply + [5])                  # think block dropped: the reply at offset -4
        self.turn(c, 'c', base, reply)
        self.turn(c, 'c', base + reply[:3] + [9])                   # all but the stop token
        self.turn(c, 'd', base, reply)
        self.turn(c, 'd', base + [70, 99])                          # into the reply
        self.turn(c, 'e', base, reply)
        self.turn(c, 'e', base + [1, 2, 3])                         # the reply re-tokenised away
        snap = c.snapshot()
        self.assertEqual(snap['reply_aligned'], {'whole': 2, 'but_last': 1, 'into': 1, 'none': 1})
        self.assertEqual(snap['reply_offset'], {'0': 3, '-4': 1})
        self.assertEqual((snap['sums']['reply_len'], snap['sums']['reply_run']), (20, 4 + 4 + 3 + 1))

    def test_divergence_windows_first_appends_only(self):
        c = ii.PrefixCounter(4)
        base, reply = list(range(10, 50)), [70, 71, 72, 2]
        for i in range(ii.WINDOW_APPENDS + 3):
            salt = 's%d' % i
            self.turn(c, salt, base, reply)
            self.turn(c, salt, base[:-4] + [99, 98])
        windows = c.snapshot()['divergence_windows']
        self.assertEqual(len(windows), ii.WINDOW_APPENDS)
        w = windows[0]
        self.assertEqual((w['k'], w['d'], w['exact'], w['prev_prompt'], w['prev_output']), (36, -4, True, 40, 4))
        self.assertEqual(w['start'], 36 - ii.WINDOW_HALF)
        self.assertEqual(w['new'], base[30:36] + [99, 98])
        self.assertEqual(w['old'], [None, None] + base[32:40] + reply[:2], 'ids outside the held 2-block tail: None')

    def test_deep_divergence_window_marks_unheld_ids(self):
        c = ii.PrefixCounter(4)
        base = list(range(10, 50))
        self.turn(c, 'a', base, [1])
        self.turn(c, 'a', [0] + base)
        w = c.snapshot()['divergence_windows'][0]
        self.assertEqual((w['k'], w['exact']), (0, False))
        self.assertEqual(w['old'], [None] * ii.WINDOW_HALF)


class AdmittedStamps(unittest.TestCase):
    """fix 3: request_admitted carries the API arrival stamp and the admission wall clock (log-only)."""

    def test_fields_reach_the_gap_row(self):
        import executor_hooks as eh
        seen = []
        engine = types.SimpleNamespace(_flash_s5_driver=types.SimpleNamespace(
            _gap_event=lambda event, **fields: seen.append((event, fields))))
        eh.engine_request_admitted(engine, types.SimpleNamespace(request_id='r', num_prompt_tokens=3, arrival_time=12.5))
        (event, fields), = seen
        self.assertEqual((event, fields['arrival_s'], fields['prompt_tokens']), ('request_admitted', 12.5, 3))
        self.assertEqual(type(fields['admit_wall_s']), float)
        seen.clear()
        eh.engine_request_admitted(engine, types.SimpleNamespace(request_id='r', num_prompt_tokens=3))
        self.assertIsNone(seen[0][1]['arrival_s'])


if __name__ == '__main__':
    unittest.main()
