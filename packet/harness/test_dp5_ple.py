"""Decode-phase dp5b: the private PLE prefetch buffer against the real stock _lookup, with the real torch (pinned image), CPU tensors; the kernel itself is stubbed.
The 2048 stock case is the box crash of 4 Nov 21:0x CET; the 1024 cases hold what every run so far did."""
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'patches/s5_engine'), str(ROOT / 'harness')]
import model_hooks as mh
from vllm.models.qwen4_exp.nvidia import ngram_embedding as ne

HEADS, DIM = 8, 160


class Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def run(weight, ids, output, dim, start, end, BLOCK_D):
            self.calls.append((grid, tuple(output.shape), output.dtype))
        return run


def embedding(stock_rows):
    weight = torch.zeros((4, DIM), dtype=torch.bfloat16)
    e = types.SimpleNamespace(supports_prefetch=True, etp_data_parallel_size=1, embedding_dim=DIM, weight=weight, _uva_weight=weight, _block_d=256,
                              shard_indices=types.SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=4),
                              _prefetch_stream=types.SimpleNamespace(cuda_stream=7),
                              _prefetch_buffer=torch.empty((stock_rows, HEADS, DIM), dtype=torch.bfloat16))
    return e


def prefetch(e, rows):
    """What start_prefetch does to the buffer at etp 1: slice to the rows, then the stock check."""
    ids = torch.zeros((rows, HEADS), dtype=torch.int64)
    return ne.Qwen4ExpPLEPinnedHostEmbedding._lookup(e, ids, output=e._prefetch_buffer[: ids.shape[0]])


class Prefetch(unittest.TestCase):
    def setUp(self):
        self.kernel = Kernel()
        patcher = mock.patch.object(ne, '_lookup_ple_embedding_from_pinned_kernel', self.kernel)
        patcher.start()
        self.addCleanup(patcher.stop)

    def scratch(self, e, rows):
        return mh.PlePrivateScratch(e, rows, torch_module=torch, device='cpu', stream=types.SimpleNamespace(cuda_stream=9))

    def test_the_box_crash_on_stock_at_2048(self):
        with self.assertRaisesRegex(ValueError, 'PLE prefetch output must match'):
            prefetch(embedding(1024), 2048)

    def test_private_at_2048_passes_shape_dtype_device(self):
        e = embedding(1024)
        scratch = self.scratch(e, 2048)
        self.assertEqual(tuple(scratch.buffer.shape), (2048, HEADS, DIM))
        self.assertEqual((scratch.buffer.dtype, scratch.buffer.device), (e.weight.dtype, e.weight.device))
        self.assertEqual(scratch.bytes, 2048 * HEADS * DIM * 2)
        stock = e._prefetch_buffer
        with scratch.scope():
            out = prefetch(e, 2048)
            self.assertIs(e._prefetch_buffer, scratch.buffer)
            self.assertEqual(tuple(out.shape), (2048, HEADS, DIM))
            prefetch(e, 2000)
            prefetch(e, 148)
        self.assertIs(e._prefetch_buffer, stock)
        self.assertEqual([c[1][0] for c in self.kernel.calls], [2048, 2000, 148])

    def test_stock_at_1024_passes(self):
        out = prefetch(embedding(1024), 1024)
        self.assertEqual(tuple(out.shape), (1024, HEADS, DIM))

    def test_private_at_1024_passes(self):
        e = embedding(1024)
        with self.scratch(e, 1024).scope():
            self.assertEqual(tuple(prefetch(e, 1024).shape), (1024, HEADS, DIM))


class Real(unittest.TestCase):
    def test_the_pinned_source_holds_it_where_the_hook_looks(self):
        pinned = Path(ne.__file__).parent
        self.assertIn('self.ple_embedding = Qwen4ExpNGramEmbedding(', (pinned / 'ple_layer.py').read_text())
        self.assertIn('self.ngram_embedding = embedding_cls(', (pinned / 'ngram_embedding.py').read_text())

    def test_the_real_layer_shape_resolves_only_above_1024(self):
        e = embedding(1024)
        ple = types.SimpleNamespace(ple_embedding=types.SimpleNamespace(ngram_embedding=e))
        model = types.SimpleNamespace(layers=[types.SimpleNamespace(ple=ple)])
        self.assertIsNone(mh.ple_embedding(model, 0))
        self.assertIs(mh.ple_embedding(model, 0, 2048), e)


if __name__ == '__main__':
    unittest.main()
