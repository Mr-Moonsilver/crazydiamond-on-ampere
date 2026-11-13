#!/usr/bin/env python3
"""decode-step lever D (FLASH_S5_DECODE_SAMPLE_FAST): the sampling math on CPU torch. Pinned container only (needs torch).

The reference is the stock FlashInfer full-vocab path's semantics, written in torch over all columns:
  * ``top_k_mask_logits``: keep ``val >= pivot``, pivot the K-th largest (flashinfer/data/include/flashinfer/topk.cuh,
    RadixTopKMaskLogitsKernel_MultiCTA: ``(val >= pivot) ? val : neg_inf``);
  * ``torch.softmax`` over the full row;
  * top-p as a threshold: keep every probability >= q, q the largest value whose kept mass reaches p (ties together);
  * a token-keyed sampler: one exponential draw per (seed, row, vocab id), the token the argmin of draw / probability.
D (``decode_step.FastSampler.sample_top``) gets the same keyed sampler over its WIDTH columns, keyed by their vocab ids.
With the same seeds it must return the same token on every row of every trial:
  * bf16-rounded logits (natural ties), forced ties at the pivot, -inf stop tokens (the min_tokens bias);
  * K in {1, 5, 20, 64, 128}; top_p in {None, 0.5, 0.95, 0.999}.
Also tested: no install on the image's GPUWatermarkSampler (its own watermarked ``_sample_random``; owner review
13 Nov) while the plain V2 Sampler installs, with a mutant that drops the check; the stock call for a mixed-K batch, the tie-overflow counter (more than WIDTH - K ties at the pivot), its
lagged non-blocking read, and four source-text mutants that must each break identity: an exact-K mask (ties dropped),
no tie slack (width K), the column index instead of the vocab id, and the mixed-K check removed.
On the box the RNG differs: FlashInfer draws its own, and flag-off is unseeded, so no two flag-off runs agree either.
What this suite proves is the restriction to WIDTH columns, not FlashInfer's kernels. No GPU, CUDA or box claim.
"""
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'patches/s5_engine')]
import decode_step as ds  # noqa: E402

VOCAB = 70000          # >= decode_step.MIN_VOCAB
ROWS = 8
TRIALS = 120


def keyed_noise(seed, rows, vocab):
    g = torch.Generator().manual_seed(seed)
    return torch.empty(rows, vocab, dtype=torch.float64).exponential_(generator=g)


def top_p_keep(probs, top_p):
    """Threshold top-p over one row set (any column order): every prob >= q, q the largest reaching mass p."""
    if top_p is None:
        return probs > 0
    keep = torch.zeros_like(probs, dtype=torch.bool)
    for r in range(probs.shape[0]):
        p = float(top_p[r])
        values = torch.sort(probs[r], descending=True).values
        mass = torch.cumsum(values.double(), 0)
        index = int(torch.searchsorted(mass, torch.tensor([p], dtype=torch.float64)).clamp(max=len(values) - 1))
        q = values[index]
        keep[r] = probs[r] >= q
    return keep


def race(probs, keep, noise_rows):
    score = noise_rows / probs.double()
    score = score.masked_fill(~keep | (probs <= 0), float('inf'))
    return score.argmin(dim=-1)


def reference(logits, k, top_p, noise):
    pivot = torch.topk(logits, k, dim=-1).values[:, -1:]
    masked = logits.masked_fill(logits < pivot, float('-inf'))
    probs = torch.softmax(masked, dim=-1)
    return race(probs, top_p_keep(probs, top_p), noise)


def kernels(noise):
    def top_p_sample(probs, top_p, ids):
        return race(probs, top_p_keep(probs, top_p), noise.gather(1, ids))

    def sample(probs, ids):
        return race(probs, top_p_keep(probs, None), noise.gather(1, ids))
    return top_p_sample, sample


def logits_case(rng, k, ties):
    x = torch.from_numpy(rng.standard_normal((ROWS, VOCAB)).astype(np.float32) * 3.0)
    x = x.to(torch.bfloat16).to(torch.float32) / 0.7             # bf16 LM-head output, then temperature 0.7
    stop = torch.from_numpy(rng.choice(VOCAB, size=4, replace=False))
    x[:, stop] = float('-inf')                                   # min_tokens: stop tokens masked
    if ties:
        for r in range(ROWS):
            pivot = torch.topk(x[r], k).values[-1]
            spots = torch.from_numpy(rng.choice(VOCAB, size=ties, replace=False))
            x[r, spots] = pivot                                  # forced ties at the K-th value
    return x


class States:
    def __init__(self, ks):
        self.top_k = types.SimpleNamespace(np=np.array(ks, dtype=np.int32))


class Sampler:
    def __init__(self, ks):
        self.sampling_states = States(ks)
        self.stock_calls = 0

    def _sample_random(self, logits, expanded, idx_np, pos, top_k, top_p, use_flashinfer):
        self.stock_calls += 1
        return 'stock', logits


def load_mutant(old, new):
    source = (ROOT / 'patches/s5_engine/decode_step.py').read_text()
    assert source.count(old) == 1, old
    module = types.ModuleType('decode_step_mutant')
    exec(compile(source.replace(old, new), 'decode_step_mutant.py', 'exec'), module.__dict__)
    return module


def trials(module, *, count=TRIALS, seed=1311):
    """Per trial: (reference tokens, D tokens) over random K, top_p, ties."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(count):
        k = int(rng.choice([1, 5, 20, 64, 128]))
        p = [None, 0.5, 0.95, 0.999][t % 4]
        ties = int(rng.choice([0, 0, 3, 40]))
        logits = logits_case(rng, k, ties)
        top_p = None if p is None else torch.full((ROWS,), p, dtype=torch.float32)
        noise = keyed_noise(10_000 + t, ROWS, VOCAB)
        sampler = Sampler([k] * ROWS)
        top_p_sample, sample = kernels(noise)
        fast = module.FastSampler(sampler, torch=torch, top_p_sample=top_p_sample, sample=sample)
        fast.gated = True
        got, kept = sampler._sample_random(logits, None, np.arange(ROWS), None, torch.full((ROWS,), k), top_p, True)
        assert kept is logits
        out.append((reference(logits, k, top_p, noise), got, sampler.stock_calls, fast))
    return out


class Identity(unittest.TestCase):
    def test_token_identical_to_the_full_vocab_path(self):
        results = trials(ds)
        for ref, got, stock, fast in results:
            self.assertEqual(stock, 0)
            self.assertEqual(got.dtype, torch.int64)
            self.assertTrue(torch.equal(ref, got), (ref, got))
            self.assertEqual(int(fast.tie_overflow), 0)
        self.assertEqual(sum(f.counters['fast'] for *_, f in results), TRIALS)

    def test_mixed_k_is_the_stock_call(self):
        sampler = Sampler([20] * 7 + [40])
        fast = ds.FastSampler(sampler, torch=torch, top_p_sample=lambda *a: 1 / 0, sample=lambda *a: 1 / 0)
        fast.gated = True
        logits = torch.zeros(ROWS, VOCAB)
        self.assertEqual(sampler._sample_random(logits, None, np.arange(ROWS), None, 'k', None, True)[0], 'stock')
        self.assertEqual(fast.counters['stock_k_mixed'], 1)

    def test_tie_overflow_is_counted_and_read_lagged(self):
        sampler = Sampler([20] * ROWS)
        noise = keyed_noise(5, ROWS, VOCAB)
        fast = ds.FastSampler(sampler, torch=torch, top_p_sample=kernels(noise)[0], sample=kernels(noise)[1])
        fast.gated = True
        logits = torch.randn(ROWS, VOCAB)
        logits[0, :400] = 50.0                                   # 400 ties at the top: more than WIDTH - K
        top_p = torch.full((ROWS,), 0.95)
        sampler._sample_random(logits, None, np.arange(ROWS), None, 'k', top_p, True)
        self.assertEqual(int(fast.tie_overflow), 1)
        self.assertIsNone(fast.overflow_seen())                  # the first read only starts the copy
        sampler._sample_random(logits, None, np.arange(ROWS), None, 'k', top_p, True)
        self.assertEqual(fast.overflow_seen(), 1)                # lagged by one log point
        self.assertEqual(int(fast.tie_overflow), 2)

    def test_logits_and_kept_tensor_untouched(self):
        sampler = Sampler([20] * ROWS)
        noise = keyed_noise(6, ROWS, VOCAB)
        fast = ds.FastSampler(sampler, torch=torch, top_p_sample=kernels(noise)[0], sample=kernels(noise)[1])
        fast.gated = True
        logits = torch.randn(ROWS, VOCAB)
        before = logits.clone()
        _, kept = sampler._sample_random(logits, None, np.arange(ROWS), None, 'k', torch.full((ROWS,), 0.9), True)
        self.assertIs(kept, logits)
        self.assertTrue(torch.equal(logits, before))


class ImageSamplers(unittest.TestCase):
    """Owner review (13 Nov): the image's own sampler classes. GPUWatermarkSampler (built when watermark_config is
    set) runs its own _sample_random, so WorkerSide must not install D on it; the plain V2 Sampler must still install.
    Instances are made without __init__ (no CUDA): only the class layout and the install gate are under test."""

    def side_on(self, module, cls):
        sampler = object.__new__(cls)
        sampler.sampling_states = States([20] * ROWS)
        side = module.WorkerSide({module.SAMPLE_FAST_ENV: '1'})
        side.pop(types.SimpleNamespace(rank=0, worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(sampler=sampler))), 'sample_tokens', {module.WIRE_KEY: [1, 0, 1]})
        return side, sampler

    def classes(self):
        from vllm.v1.watermarking.gpu_sampler import GPUWatermarkSampler
        from vllm.v1.worker.gpu.sample.sampler import Sampler as V2Sampler
        return V2Sampler, GPUWatermarkSampler

    def test_watermark_sampler_refused_plain_sampler_installed(self):
        V2Sampler, GPUWatermarkSampler = self.classes()
        self.assertEqual(ds.sample_random_owner(object.__new__(V2Sampler)), 'Sampler')
        self.assertEqual(ds.sample_random_owner(object.__new__(GPUWatermarkSampler)), 'GPUWatermarkSampler')
        side, sampler = self.side_on(ds, GPUWatermarkSampler)
        self.assertIsNone(side.fast)
        self.assertEqual(side.counters['sampler_override'], 1)
        self.assertNotIn('_sample_random', vars(sampler))           # its own watermarked draw, unwrapped
        side, sampler = self.side_on(ds, V2Sampler)
        self.assertIsNotNone(side.fast)
        self.assertEqual(sampler._sample_random, side.fast.sample_random)

    def test_owner_check_removed_installs_on_the_watermark_sampler(self):
        module = load_mutant("                if owner != 'Sampler' or getattr(sampler, 'watermarking', None) is not None:\n",
                             "                if False:\n")
        side, sampler = self.side_on(module, self.classes()[1])
        self.assertIsNotNone(side.fast, 'the mutant must install over the watermarked draw')
        self.assertIn('_sample_random', vars(sampler))


class Mutants(unittest.TestCase):
    def broken(self, old, new, **kw):
        module = load_mutant(old, new)
        results = trials(module, count=40, **kw)
        return any(not torch.equal(ref, got) for ref, got, *_ in results)

    def test_exact_k_mask_drops_ties(self):
        self.assertTrue(self.broken(
            "        kept = values.masked_fill(values < pivot, float('-inf'))\n",
            "        kept = values.masked_fill(torch.arange(values.shape[1]).unsqueeze(0) >= k, float('-inf'))\n"))

    def test_no_tie_slack(self):
        self.assertTrue(self.broken("values, ids = torch.topk(logits, self.width, dim=-1",
                                    "values, ids = torch.topk(logits, k, dim=-1"))

    def test_column_index_instead_of_vocab_id(self):
        self.assertTrue(self.broken(
            "        return ids.gather(1, local.view(-1, 1).to(torch.int64)).view(-1).to(torch.int64)\n",
            "        return local.view(-1).to(torch.int64)\n"))

    def test_mixed_k_check_removed(self):
        module = load_mutant("        if not bool((ks == k).all()):\n", "        if False:\n")
        sampler = Sampler([20] * 7 + [40])
        noise = keyed_noise(7, ROWS, VOCAB)
        fast = module.FastSampler(sampler, torch=torch, top_p_sample=kernels(noise)[0], sample=kernels(noise)[1])
        fast.gated = True
        sampler._sample_random(torch.randn(ROWS, VOCAB), None, np.arange(ROWS), None, 'k', None, True)
        self.assertEqual(sampler.stock_calls, 0, 'the mutant samples a mixed batch with row 0 K')


if __name__ == '__main__':
    unittest.main()
