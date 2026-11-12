"""ingress-incremental: the real HF tokenizer differential. Pinned container only (needs tokenizers, transformers, vllm).

A byte-level BPE is trained in the test on replay text (agent_replay.text) plus unicode, with the NFC normalizer and the
Qwen added tokens (<|im_start|>, <|im_end|>, <|endoftext|> special; <think>, </think>, <tool_call> plain). Conversations
are rendered by transformers apply_chat_template with a Qwen3-style template (enable_thinking False: the generation
prompt carries the empty think block, history assistant turns do not). Every turn of every conversation: the splice
through ingress_incremental.Ingress on a renderer shaped like vLLM's (its real BaseRenderer._build_tokens_prompt, real
make_async) equals the stock call (transformers __call__ with HfRenderer's encode arguments), including tool outputs
holding special-token text, CRLF, combining marks, emoji and replies holding a think tag. Also: 8 conversations from 8
threads through the async wrapper; truncation fallbacks. No GPU, CUDA or box claim.
"""
import asyncio
from pathlib import Path
import random
import sys
import threading
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'patches/s5_engine')]
import agent_replay as ar  # noqa: E402
import ingress_incremental as ii  # noqa: E402

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers, AddedToken  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402
from vllm.renderers.base import BaseRenderer  # noqa: E402

TEMPLATE = (
    "{%- if messages[0].role == 'system' %}{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}{%- endif %}"
    "{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}"
    "{%- for message in messages[::-1] %}{%- set index = (messages|length - 1) - loop.index0 %}"
    "{%- if ns.multi_step_tool and message.role == 'user' and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}"
    "{%- set ns.multi_step_tool = false %}{%- set ns.last_query_index = index %}{%- endif %}{%- endfor %}"
    "{%- for message in messages %}{%- set content = message.content if message.content is string else '' %}"
    "{%- if (message.role == 'user') or (message.role == 'system' and not loop.first) %}"
    "{{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}"
    "{%- elif message.role == 'assistant' %}{%- set reasoning_content = '' %}"
    "{%- if '</think>' in content %}{%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}"
    "{%- set content = content.split('</think>')[-1].lstrip('\\n') %}{%- endif %}"
    "{%- if loop.index0 > ns.last_query_index %}{%- if loop.last or (not loop.last and reasoning_content) %}"
    "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}"
    "{%- else %}{{- '<|im_start|>' + message.role + '\\n' + content }}{%- endif %}"
    "{%- else %}{{- '<|im_start|>' + message.role + '\\n' + content }}{%- endif %}{{- '<|im_end|>\\n' }}{%- endif %}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}{%- endif %}")
UNI = ['naïve', 'café', 'Straße', '数据', '函数', '😀', '👍🏽', 'e\u0301', '\u00a0', '\t', '  ', '\r\n', '\n\n', 'ﬁ', 'Ⅻ',
       '"q"', "it's", '->', '```', '<|im_end', '<|im_start|>', '<|im_end|>', '<think>', '</think>', '<tool_call>', '\u200b']
KW = dict(truncation=True, max_length=262145, add_special_tokens=False)
_TOK = []


def tokenizer():
    if _TOK:
        return _TOK[0]
    rng = random.Random(7)
    corpus = [ar.text(rng, 400) for _ in range(60)] + [' '.join(UNI) * 3, 'Tool output:\nRepository context:\n']
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=3000, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                       show_progress=False))
    tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in ('<|endoftext|>', '<|im_start|>', '<|im_end|>')])
    tok.add_tokens([AddedToken(t, special=False, normalized=False) for t in ('<think>', '</think>', '<tool_call>')])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok)
    fast.chat_template = TEMPLATE
    _TOK.append(fast)
    return fast


def words(rng, n, unicode):
    w = ar.text(rng, n).split(' ')
    if unicode:
        for _ in range(max(1, len(w) // 10)):
            w.insert(rng.randrange(len(w) + 1), rng.choice(UNI))
    return ' '.join(w)


def conversation(seed, turns, unicode, scale=300):
    rng = random.Random(seed)
    messages = [{'role': 'system', 'content': 'You are coding agent %d, session 0.' % seed},
                {'role': 'user', 'content': 'Repository context:\n' + words(rng, scale * 10, unicode)}]
    tok = tokenizer()
    for _ in range(turns):
        yield tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        reply = words(rng, rng.randint(scale // 3, scale), unicode)
        messages.append({'role': 'assistant', 'content': reply})
        messages.append({'role': 'user', 'content': 'Tool output:\n' + words(rng, rng.randint(scale, scale * 4), unicode)})


class Renderer:
    """The parts of vLLM's BaseRenderer the wrapper touches; _build_tokens_prompt is vLLM's own."""
    _build_tokens_prompt = staticmethod(BaseRenderer._build_tokens_prompt)

    def __init__(self, tok):
        from concurrent.futures import ThreadPoolExecutor
        self.tokenizer = tok
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.stock = 0

    def get_tokenizer(self):
        return self.tokenizer

    def _wants_offsets(self, prompt, params):
        return False

    def _tokenize_prompt(self, prompt, params):     # HfRenderer's: tokenizer(prompt, **encode kwargs)
        self.stock += 1
        enc = self.tokenizer(prompt['prompt'], **params.get_encode_kwargs())
        return self._build_tokens_prompt(enc['input_ids'], prompt)


class Params:
    def __init__(self, **kw):
        self.kw = dict(KW, **kw)

    def get_encode_kwargs(self):
        return dict(self.kw)


def via_async(r, args):
    """vLLM's make_async wrapper needs a running loop at call time (it returns a future)."""
    async def call():
        return await r._tokenize_prompt_async(*args)
    return asyncio.run(call())


def installed():
    tok = tokenizer()
    r = Renderer(tok)
    ing = ii.install(type('E', (), {'renderer': r})(), {ii.ENV: '1'})
    ing.splicer.verify_every = 0
    ing.binding.observe('release_auto_arm', {'released': True})
    return tok, r, ing


class Differential(unittest.TestCase):
    def test_every_turn_equals_the_stock_call(self):
        tok, r, ing = installed()
        self.assertEqual(sorted(ing.splicer.joins), ['<|endoftext|>', '<|im_end|>', '<|im_start|>'])
        turns = 0
        for seed in range(8):
            for text in conversation(seed, 7, unicode=seed % 2 == 1):
                got = via_async(r, ({'prompt': text}, Params()))
                self.assertEqual(got['prompt_token_ids'], tok(text, **KW)['input_ids'], (seed, turns))
                self.assertEqual(got['prompt'], text)
                turns += 1
        snap = ing.snapshot()
        self.assertEqual((turns, snap['counts']['spliced'], snap['counts']['full']), (56, 48, 8))
        self.assertEqual(r.stock, 8)

    def test_reply_rerender_drops_the_think_block(self):
        """Under this hub-written template the new prompt diverges 4 tokens before the previous prompt's end. The engine's
        own template keeps the empty think block in history (test_ingress_real_tokenizer, night-pack fix 1)."""
        tok = tokenizer()
        texts = list(conversation(3, 2, unicode=False))
        a, b = (tok(t, **KW)['input_ids'] for t in texts)
        k = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
        self.assertEqual(texts[0][-len('<think>\n\n</think>\n\n'):], '<think>\n\n</think>\n\n')
        self.assertEqual(len(a) - k, len(tok('<think>\n\n</think>\n\n', **KW)['input_ids']))

    def test_truncation_length_falls_back(self):
        tok, r, ing = installed()
        texts = list(conversation(5, 2, unicode=True))
        n = len(tok(texts[1], **KW)['input_ids'])
        for t in texts:
            got = via_async(r, ({'prompt': t}, Params(max_length=n - 1)))
            self.assertEqual(got['prompt_token_ids'], tok(t, **dict(KW, max_length=n - 1))['input_ids'])
        self.assertEqual(ing.splicer.fallbacks['over-truncation-length'], 1)

    def test_eight_threads(self):
        tok, r, ing = installed()
        ing.splicer.verify_every = 4
        convs = [list(conversation(20 + s, 6, unicode=s % 2 == 0, scale=150)) for s in range(8)]
        bad = []

        def run(i):
            for t in convs[i]:
                if ing.tokenize_prompt({'prompt': t}, Params())['prompt_token_ids'] != tok(t, **KW)['input_ids']:
                    bad.append(i)
        threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(bad, [])
        snap = ing.snapshot()
        self.assertEqual((snap['counts']['spliced'], snap['counts']['full'], snap['entries']), (40, 8, 8))
        self.assertEqual(snap['counts']['verified'], 10)
        self.assertNotIn('verify_mismatch', snap['counts'])

    def test_not_selected_is_stock(self):
        tok, r, ing = installed()
        ing.binding.observe('pause', {})
        for t in conversation(1, 3, unicode=False):
            via_async(r, ({'prompt': t}, Params()))
        self.assertEqual(r.stock, 3)
        self.assertEqual(ing.splicer.snapshot()['counts'], {})


if __name__ == '__main__':
    unittest.main()
