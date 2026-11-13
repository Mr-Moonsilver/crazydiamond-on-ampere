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
Also tested: the stock call for a mixed-K batch, and four source-text mutants that must each break identity: an exact-K
mask (ties dropped), no tie slack (width K), the column index instead of the vocab id, and the mixed-K check removed.
Overflow (lever D reviewed-fix, Codex review 13 Nov): a row whose stock kept set is wider than WIDTH (the reviewer's
repro: 400 equal highest logits, top_k 20, top_p 0.95) takes the exact overflow route before its draw. Its
distribution over the whole vocab equals the full-vocab reference (allclose), on mixed batches too. Its draws cover the
whole tie set; the diagnostic counter is read only from a retired copy. Two more mutants: overflow ignored, and overflow
routed only after the draw (the next step).
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
            self.assertEqual(int(fast.overflow_rows), 0)
        self.assertEqual(sum(f.counters['fast'] for *_, f in results), TRIALS)

    def test_mixed_k_is_the_stock_call(self):
        sampler = Sampler([20] * 7 + [40])
        fast = ds.FastSampler(sampler, torch=torch, top_p_sample=lambda *a: 1 / 0, sample=lambda *a: 1 / 0)
        fast.gated = True
        logits = torch.zeros(ROWS, VOCAB)
        self.assertEqual(sampler._sample_random(logits, None, np.arange(ROWS), None, 'k', None, True)[0], 'stock')
        self.assertEqual(fast.counters['stock_k_mixed'], 1)

    def test_overflow_read_only_from_a_retired_copy(self):
        sampler = Sampler([20] * ROWS)
        noise = keyed_noise(5, ROWS, VOCAB)
        fast = ds.FastSampler(sampler, torch=torch, top_p_sample=kernels(noise)[0], sample=kernels(noise)[1])
        fast.gated = True
        logits = torch.randn(ROWS, VOCAB)
        logits[0, :400] = 50.0                                   # 400 ties at the top: more than WIDTH
        top_p = torch.full((ROWS,), 0.95)
        sampler._sample_random(logits, None, np.arange(ROWS), None, 'k', top_p, True)
        self.assertEqual(int(fast.overflow_rows), 1)
        self.assertIsNone(fast.overflow_seen())                  # no copy has retired yet: unavailable
        sampler._sample_random(logits, None, np.arange(ROWS), None, 'k', top_p, True)
        self.assertEqual(fast.overflow_seen(), 1)                # the CPU copy retired at once (synchronous)

        class Pending:
            def __init__(self):
                self.done = False

            def query(self):
                return self.done
        event = Pending()
        fast._copy_event = event                                 # a CUDA copy still in flight
        self.assertIsNone(fast.overflow_seen())
        self.assertIs(fast._copy_event, event)                   # no new copy over the pending one
        event.done = True
        self.assertEqual(fast.overflow_seen(), 2)                # the copy the second read started (rows then: 2)

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


def reference_dense(logits, k, top_p):
    """The full-vocab reference's distribution: >= pivot mask, full softmax, threshold top-p, renormalised."""
    pivot = torch.topk(logits, k, dim=-1).values[:, -1:]
    probs = torch.softmax(logits.masked_fill(logits < pivot, float('-inf')).double(), dim=-1)
    kept = probs * top_p_keep(probs, top_p)
    return kept / kept.sum(dim=-1, keepdim=True)


def overflow_dense(fast, logits, k, top_p):
    """D's overflow route as a distribution over the vocab (its weights spread back: ties share the group equally)."""
    values, ids = torch.topk(logits, fast.width, dim=-1, largest=True, sorted=True)
    pivot = values[:, k - 1:k]
    tie_mask = logits == pivot
    count = tie_mask.sum(dim=-1, keepdim=True).to(torch.int32)
    weights = fast.overflow_weights(values, pivot, count, k, top_p).double()
    dense = torch.zeros(logits.shape, dtype=torch.float64)
    dense.scatter_add_(1, ids[:, :k], weights[:, :k])
    dense += tie_mask.double() * (weights[:, k:] / count.double())
    return dense / dense.sum(dim=-1, keepdim=True)


def overflow_case(rng, k, ties, above, vocab=VOCAB, rows=ROWS):
    x = torch.from_numpy(rng.standard_normal((rows, vocab)).astype(np.float32) * 3.0)
    x = x.to(torch.bfloat16).to(torch.float32) / 0.7
    for r in range(rows):
        level = float(x[r].max()) + 1.0
        spots = torch.from_numpy(rng.choice(vocab, size=ties + above, replace=False))
        x[r, spots[:ties]] = level                               # ties at the K-th value, more than WIDTH - above
        x[r, spots[ties:]] = level + torch.from_numpy(rng.uniform(0.05, 4.0, size=above).astype(np.float32))
    return x


class Overflow(unittest.TestCase):
    def fast(self, ks, uniform=None):
        sampler = Sampler(ks)
        noise = keyed_noise(3, len(ks), VOCAB)
        fast = ds.FastSampler(sampler, torch=torch, top_p_sample=kernels(noise)[0], sample=kernels(noise)[1],
                              uniform=uniform)
        fast.gated = True
        return sampler, fast

    def test_reviewer_repro_support_is_the_stock_support(self):
        """400 equal highest logits, top_k 20, top_p 0.95 (the review's tie-overflow.json): stock keeps 400."""
        vocab = 65536
        logits = torch.full((1, vocab), -5.0)
        logits[0, :400] = 10.0
        sampler = Sampler([20])
        g = torch.Generator().manual_seed(1311)
        fast = ds.FastSampler(sampler, torch=torch, top_p_sample=lambda p, t, i: p.argmax(-1),
                              sample=lambda p, i: p.argmax(-1), uniform=lambda n: torch.rand(n, 2, generator=g))
        fast.gated = True
        tokens = [int(sampler._sample_random(logits, None, np.arange(1), None, 'k', torch.full((1,), 0.95), True)[0][0])
                  for _ in range(4000)]
        self.assertEqual(sampler.stock_calls, 0)
        self.assertEqual(int(fast.overflow_rows), 4000)
        self.assertTrue(set(tokens) <= set(range(400)))
        self.assertGreater(len(set(tokens)), 380)                # the whole tie set, not the 256 the view holds
        self.assertGreater(len([t for t in tokens if t >= 256]), 4000 * 144 / 400 * 0.8)
        dense = overflow_dense(fast, logits, 20, torch.full((1,), 0.95))
        self.assertEqual(int((dense > 0).sum()), 400)
        self.assertTrue(torch.allclose(dense, reference_dense(logits, 20, torch.full((1,), 0.95)), atol=1e-9))

    def test_overflow_distribution_equals_the_full_vocab_reference(self):
        rng = np.random.default_rng(77)
        for k, ties, above in ((20, 400, 0), (20, 300, 7), (5, 260, 3), (64, 500, 40), (128, 1000, 100), (1, 257, 0)):
            for p in (None, 0.5, 0.95, 0.999):
                logits = overflow_case(rng, k, ties, above)
                top_p = None if p is None else torch.full((ROWS,), p, dtype=torch.float32)
                _, fast = self.fast([k] * ROWS)
                got, want = overflow_dense(fast, logits, k, top_p), reference_dense(logits, k, top_p)
                self.assertTrue(torch.allclose(got, want, atol=1e-6), (k, ties, above, p, float((got - want).abs().max())))
                self.assertTrue(torch.equal(got > 0, want > 0), (k, ties, above, p))

    def test_overflow_draws_follow_the_distribution(self):
        rng = np.random.default_rng(5)
        logits = overflow_case(rng, 20, 300, 6, vocab=4096, rows=1)
        top_p = torch.full((1,), 0.95)
        g = torch.Generator().manual_seed(9)
        sampler, fast = self.fast([20], uniform=lambda n: torch.rand(n, 2, generator=g))
        fast.min_vocab = 4096
        want = reference_dense(logits, 20, top_p)[0]
        draws = 6000
        counts = torch.zeros_like(want)
        for _ in range(draws):
            counts[int(sampler._sample_random(logits, None, np.arange(1), None, 'k', top_p, True)[0][0])] += 1
        self.assertEqual(float(counts[want == 0].sum()), 0.0)    # never outside the stock support
        tie_set = logits[0] == torch.topk(logits[0], 20).values[-1]
        # the tie group and every token above it: frequency within 5 sigma of the stock probability
        for mass, seen in [(float(want[tie_set].sum()), float(counts[tie_set].sum()))] + \
                [(float(want[i]), float(counts[i])) for i in torch.nonzero((want > 0) & ~tie_set).view(-1).tolist()]:
            sigma = (draws * mass * (1 - mass)) ** 0.5
            self.assertLess(abs(seen - draws * mass), 5 * sigma + 1, (mass, seen))

    def test_mixed_batch_overflow_rows_exact_the_rest_unchanged(self):
        rng = np.random.default_rng(21)
        logits = logits_case(rng, 20, 0)
        logits[[1, 5]] = overflow_case(rng, 20, 350, 4, rows=2)
        top_p = torch.full((ROWS,), 0.95)
        noise = keyed_noise(44, ROWS, VOCAB)
        sampler = Sampler([20] * ROWS)
        g = torch.Generator().manual_seed(2)
        fast = ds.FastSampler(sampler, torch=torch, top_p_sample=kernels(noise)[0], sample=kernels(noise)[1],
                              uniform=lambda n: torch.rand(n, 2, generator=g))
        fast.gated = True
        got, _ = sampler._sample_random(logits, None, np.arange(ROWS), None, 'k', top_p, True)
        want = reference(logits, 20, top_p, noise)
        normal = [r for r in range(ROWS) if r not in (1, 5)]
        self.assertTrue(torch.equal(got[normal], want[normal]))  # the view rows: token-identical as before
        self.assertEqual(int(fast.overflow_rows), 2)
        dense = reference_dense(logits, 20, top_p)
        self.assertTrue(all(float(dense[r, int(got[r])]) > 0 for r in (1, 5)))


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
            "        fast = ids.gather(1, local.view(-1, 1).to(torch.int64)).view(-1).to(torch.int64)\n",
            "        fast = local.view(-1).to(torch.int64)\n"))

    def overflow_probe(self, old, new):
        """(distinct first draws over 200 fresh samplers, distinct draws over 1,500 later steps) on the review's repro.
        The stock support is 0..399; the view holds 256 of them and the fake kernel always takes its first column, so a
        draw served from the view is one fixed token."""
        module = load_mutant(old, new)
        logits = torch.full((1, 4096), -5.0)                 # min_vocab lowered: the probe's cost, not its shape
        logits[0, :400] = 10.0
        g = torch.Generator().manual_seed(1311)

        def fresh():
            sampler = Sampler([20])
            fast = module.FastSampler(sampler, torch=torch, top_p_sample=lambda p, t, i: p.argmax(-1),
                                      sample=lambda p, i: p.argmax(-1), uniform=lambda n: torch.rand(n, 2, generator=g),
                                      min_vocab=4096)
            fast.gated = True
            return lambda: int(sampler._sample_random(logits, None, np.arange(1), None, 'k',
                                                      torch.full((1,), 0.95), True)[0][0])
        first = {fresh()() for _ in range(200)}
        draw = fresh()
        draw()
        later = {draw() for _ in range(1500)}
        return len(first), len(later)

    LINE = "        return torch.where(overflow.view(-1), exact, fast)\n"

    def test_control_the_fixed_module(self):
        first, later = self.overflow_probe(self.LINE, self.LINE)
        self.assertGreater(first, 100)
        self.assertGreater(later, 350)

    def test_overflow_ignored_caught(self):
        first, later = self.overflow_probe(self.LINE, "        return fast\n")
        self.assertEqual((first, later), (1, 1))

    def test_overflow_routed_after_the_draw_caught(self):
        """The route decided from the previous step's overflow (a lagged flag): every first overflowing draw is served
        from the truncated view, although later steps are routed."""
        first, later = self.overflow_probe(self.LINE, "        late = getattr(self, '_late', None)\n"
                                                      "        self._late = overflow\n"
                                                      "        return fast if late is None else "
                                                      "torch.where(late.view(-1), exact, fast)\n")
        self.assertEqual(first, 1)
        self.assertGreater(later, 350)

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
