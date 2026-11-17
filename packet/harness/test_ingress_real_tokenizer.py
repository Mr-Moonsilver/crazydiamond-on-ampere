"""ingress-incremental against the engine model's own tokenizer and chat template (night-pack fix 1, lead 12 Nov 2026).

Pinned container only, with the real files mounted read-only (tokenizer.json vocab 248,044 + 33 added,
tokenizer_config.json, chat_template.jinja; sha256 verified against CT <ct-id> by the lead: BOX-SHA256SUMS beside them).
The directory is FLASH_S5_REAL_TOKENIZER_DIR (default /real); a missing directory FAILS (never skips): this suite is run
on purpose, outside the frozen bundle runs (the files are not part of any bundle).

The tokenizer is loaded the way the API server loads it (vLLM get_tokenizer on the directory). Covers:
  * the join-token property probe passes (the flag would disable itself otherwise and a launch would measure nothing);
  * the real template with enable_thinking False (the replay's chat_template_kwargs) keeps the empty think block in
    re-rendered history (preserve_thinking undefined = true), so a previous prompt is an exact text prefix of the next;
    the splice cut lands right after the previous generation prompt's <|im_start|>;
  * every turn of replay-shaped conversations through the installed wrapper (vLLM's make_async and
    BaseRenderer._build_tokens_prompt) equals the stock call, with verify-all on the off-path verifier: verified ==
    spliced, no mismatch, never disabled; 8 conversations from 8 threads;
  * power: a cut one character before the join changes the ids on this tokenizer;
  * the prefix counter on real ids: a canonical reply is matched whole, at offset 0 from the previous prompt end.
No GPU, CUDA or box claim; replies here are canonical encodings (the model's own generations may not be).
"""
import os
from pathlib import Path
import random
import sys
import threading
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'patches/s5_engine')]
import ingress_incremental as ii  # noqa: E402
import test_ingress_incremental_hf as hf  # noqa: E402  (Renderer, Params, via_async, words; its BPE is lazy)

REAL = Path(os.environ.get('FLASH_S5_REAL_TOKENIZER_DIR', '/real'))
KW = hf.KW
_TOK = []


def tokenizer():
    if not _TOK:
        if not (REAL / 'tokenizer.json').is_file() or not (REAL / 'chat_template.jinja').is_file():
            raise AssertionError('real tokenizer files absent at %s' % REAL)
        from vllm.tokenizers import get_tokenizer
        _TOK.append(get_tokenizer(str(REAL)))
    return _TOK[0]


def conversation(seed, turns, unicode, scale=300):
    rng = random.Random(seed)
    tok = tokenizer()
    messages = [{'role': 'system', 'content': 'You are coding agent %d, session 0.' % seed},
                {'role': 'user', 'content': 'Repository context:\n' + hf.words(rng, scale * 10, unicode)}]
    for _ in range(turns):
        yield tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        messages.append({'role': 'assistant', 'content': hf.words(rng, rng.randint(scale // 3, scale), unicode)})
        messages.append({'role': 'user', 'content': 'Tool output:\n' + hf.words(rng, rng.randint(scale, scale * 4), unicode)})


def installed(env=None):
    tok = tokenizer()
    r = hf.Renderer(tok)
    env = dict({ii.ENV: '1', ii.VERIFY_EVERY_ENV: '1', ii.VERIFY_ASYNC_ENV: '1'}, **(env or {}))
    ing = ii.install(type('E', (), {'renderer': r})(), env)
    ing.binding.observe('release_auto_arm', {'released': True})
    return tok, r, ing


class Probe(unittest.TestCase):
    def test_join_probe_passes(self):
        sp = ii.Splicer(tokenizer(), verify_every=0)
        self.assertIsNone(sp.disabled)
        self.assertTrue({'<|im_start|>', '<|im_end|>', '<|endoftext|>'} <= set(sp.joins))
        for plain in ('<think>', '</think>', '<tool_call>'):
            self.assertNotIn(plain, sp.joins)
        self.assertEqual(len(tokenizer()), 248077)


class Template(unittest.TestCase):
    def test_history_keeps_the_empty_think_block(self):
        tok = tokenizer()
        p1, p2 = list(conversation(3, 2, unicode=False, scale=40))
        self.assertTrue(p1.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'))
        self.assertTrue(p2.startswith(p1), 'the previous prompt is an exact text prefix of the next one')
        self.assertNotIn('Reasoning effort', p1, 'enable_thinking False: no reasoning instructions')
        sp = ii.Splicer(tok, verify_every=0)
        ids = tok(p1, **KW)['input_ids']
        sp.tokenize(p1, dict(KW), lambda: ids)
        entry = next(iter(sp.cache.values()))[1:]
        cut, _k, _i = sp.cut(entry, p2)
        self.assertEqual(cut, p1.rindex('<|im_start|>') + len('<|im_start|>'))


class Differential(unittest.TestCase):
    def test_every_turn_equals_the_stock_call_verify_all(self):
        tok, r, ing = installed()
        turns = 0
        for seed in range(8):
            for text in conversation(seed, 6, unicode=seed % 2 == 1):
                got = hf.via_async(r, ({'prompt': text}, hf.Params()))
                self.assertEqual(got['prompt_token_ids'], tok(text, **KW)['input_ids'], (seed, turns))
                turns += 1
        self.assertTrue(ing.splicer.drain(timeout=60))
        snap = ing.snapshot()
        c = snap['counts']
        self.assertEqual((turns, c['spliced'], c['full'], c['verified']), (48, 40, 8, 40))
        self.assertEqual((snap['disabled'], snap['verify']['mismatches']), (None, []))
        self.assertNotIn('verify_mismatch', c)
        self.assertGreater(snap['timing']['async_n3_n'], 0)

    def test_eight_threads(self):
        tok, r, ing = installed()
        convs = [list(conversation(20 + s, 5, unicode=s % 2 == 0, scale=150)) for s in range(8)]
        bad = []

        def run(i):
            for t in convs[i]:
                if ing.tokenize_prompt({'prompt': t}, hf.Params())['prompt_token_ids'] != tok(t, **KW)['input_ids']:
                    bad.append(i)
        threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertTrue(ing.splicer.drain(timeout=60))
        self.assertEqual(bad, [])
        snap = ing.snapshot()
        self.assertEqual((snap['counts']['spliced'], snap['counts']['verified'], snap['disabled']), (32, 32, None))

    def test_a_cut_before_the_join_differs(self):
        """Power: the same splice with the cut one character early changes the ids on this tokenizer."""
        tok = tokenizer()
        differ = 0
        for seed in range(4):
            texts = list(conversation(50 + seed, 3, unicode=seed % 2 == 1, scale=100))
            for a, b in zip(texts, texts[1:]):
                cut = a.rindex('<|im_start|>') + len('<|im_start|>') - 1
                head = tok(a[:cut], **KW)['input_ids']
                differ += head + tok(b[cut:], **KW)['input_ids'] != tok(b, **KW)['input_ids']
        self.assertGreater(differ, 0)


class Counter(unittest.TestCase):
    def test_canonical_reply_matches_whole_at_offset_zero(self):
        tok = tokenizer()
        rng = random.Random(9)
        messages = [{'role': 'system', 'content': 'You are coding agent 0, session 0.'},
                    {'role': 'user', 'content': 'Repository context:\n' + hf.words(rng, 2000, False)}]
        c = ii.PrefixCounter(176)
        im_end = tok.convert_tokens_to_ids('<|im_end|>')
        for _ in range(3):
            prompt = tok(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False), **KW)['input_ids']
            reply = hf.words(rng, 200, False)
            req = Req176(prompt)
            c.record(req)
            req.extend(tok(reply, **KW)['input_ids'] + [im_end])
            messages += [{'role': 'assistant', 'content': reply}, {'role': 'user', 'content': 'Tool output:\nok'}]
        snap = c.snapshot()
        self.assertEqual(snap['reply_aligned'], {'whole': 2})
        self.assertEqual(snap['reply_offset'], {'0': 2})
        self.assertEqual(snap['counts'].get('through_reply'), 2)


class Req176:
    """A vLLM Request's fields the counter reads; chained block hashes over full 176-token blocks."""

    def __init__(self, prompt):
        self.cache_salt, self.prompt_token_ids = 'real', list(prompt)
        self.num_prompt_tokens = len(prompt)
        self.output_token_ids, self.block_hashes = [], []
        self._hash()

    def extend(self, output):
        self.output_token_ids.extend(output)
        self._hash()

    def _hash(self):
        allt = self.prompt_token_ids + self.output_token_ids
        prev = self.block_hashes[-1] if self.block_hashes else 'none'
        for i in range(len(self.block_hashes) * 176, len(allt) - 175, 176):
            prev = hash((prev, tuple(allt[i:i + 176])))
            self.block_hashes.append(prev)


if __name__ == '__main__':
    unittest.main()
