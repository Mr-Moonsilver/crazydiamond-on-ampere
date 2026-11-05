"""ple-private: the private PLE buffer selected per arm at 1024, and the witness, against the real stock _lookup with the real torch (pinned image), CPU tensors; the kernel is stubbed
to a deterministic fill. The real nested attribute path, not the old fake's. Nothing here races on a device: the witness tests write another lane's rows by hand."""
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
CLS = ne.Qwen4ExpPLEPinnedHostEmbedding


class Kernel:
    def __getitem__(self, grid):
        def run(weight, ids, output, dim, start, end, BLOCK_D):
            output.view(-1, dim)[:] = (ids % 251 + 1).to(output.dtype).unsqueeze(1)
        return run


def embedding(rows=1024):
    weight = torch.zeros((4, DIM), dtype=torch.bfloat16)
    e = types.SimpleNamespace(supports_prefetch=True, etp_data_parallel_size=1, tp_size=1, embedding_dim=DIM, weight=weight, _uva_weight=weight, _block_d=256,
                              shard_indices=types.SimpleNamespace(org_vocab_start_index=0, org_vocab_end_index=4),
                              _prefetch_stream=types.SimpleNamespace(cuda_stream=7),
                              _prefetch_buffer=torch.empty((rows, HEADS, DIM), dtype=torch.bfloat16))
    e._lookup = types.MethodType(CLS._lookup, e)
    e._reduce_etp_embeddings = types.MethodType(CLS._reduce_etp_embeddings, e)
    return e


def model_of(e):
    layer = types.SimpleNamespace(ple=types.SimpleNamespace(ple_embedding=types.SimpleNamespace(ngram_embedding=e)))
    return types.SimpleNamespace(layers=[types.SimpleNamespace(ple=None), layer])


class Base(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(ne, '_lookup_ple_embedding_from_pinned_kernel', Kernel())
        patcher.start()
        self.addCleanup(patcher.stop)
        for state, values in ((mh.PLE_PRIVATE, dict(selected=False, scopes=0)), (mh.PLE_WITNESS, dict.fromkeys(mh.PLE_WITNESS, 0))):
            saved = dict(state)
            self.addCleanup(state.update, saved)
            state.update(values)
        mh.PLE_WITNESS['selected'] = False


class Private(Base):
    def scratch(self, e, rows=1024, selectable=True):
        return mh.PlePrivateScratch(e, rows, torch_module=torch, device='cpu', stream=types.SimpleNamespace(cuda_stream=9), selectable=selectable)

    def test_the_pinned_source_holds_it_on_the_nested_path(self):
        self.assertIn('self.ple_embedding = Qwen4ExpNGramEmbedding(', (Path(ne.__file__).parent / 'ple_layer.py').read_text())

    def test_the_flag_finds_the_real_path_at_1024(self):
        e = embedding()
        self.assertIsNone(mh.ple_embedding(model_of(e), 1))
        self.assertIs(mh.ple_embedding(model_of(e), 1, 1024, True), e)

    def test_at_1024_the_lever_selects_it(self):
        e, stock = embedding(), None
        scratch, model = self.scratch(e), model_of(e)
        self.assertFalse(scratch.engaged(model, 1))
        self.assertEqual(mh.PLE_PRIVATE['scopes'], 0)
        mh.select_ple_private(True)
        self.assertTrue(scratch.engaged(model, 1))
        self.assertFalse(scratch.engaged(model, 0))
        self.assertEqual(mh.PLE_PRIVATE['scopes'], 1)

    def test_private_at_1024_passes_the_stock_check_and_leaves_stock_untouched(self):
        e = embedding()
        stock = e._prefetch_buffer
        stock.fill_(5)
        scratch = self.scratch(e)
        self.assertEqual((scratch.buffer.dtype, scratch.buffer.device, tuple(scratch.buffer.shape)), (e.weight.dtype, e.weight.device, (1024, HEADS, DIM)))
        with scratch.scope():
            e._lookup(torch.ones((1024, HEADS), dtype=torch.int64), output=e._prefetch_buffer[:1024])
        self.assertIs(e._prefetch_buffer, stock)
        self.assertTrue((stock == 5).all())

    def test_above_1024_it_is_always_engaged_no_lever(self):
        e = embedding()
        scratch, model = self.scratch(e, 2048), model_of(e)
        self.assertFalse(scratch.selectable)
        self.assertTrue(scratch.engaged(model, 1))
        self.assertEqual(mh.PLE_PRIVATE['scopes'], 1, "compose-5: the box's only sign of it at 2048, and compose-4's smoke and grade read it")


class Witness(Base):
    def make(self):
        e = embedding()
        native = types.SimpleNamespace(cuda_stream=9)
        decode = e._prefetch_stream
        witness = mh.PleWitness(e, native, torch_module=torch, device='cpu').install()
        mh.select_ple_witness(True)
        return e, native, decode, witness

    def run_lane(self, e, stream, rows, clobber=None):
        e._prefetch_stream = stream
        ids = torch.arange(rows * HEADS, dtype=torch.int64).view(rows, HEADS) + rows
        e._lookup(ids, output=e._prefetch_buffer[:rows])
        if clobber:
            clobber()
        return e._reduce_etp_embeddings(e._prefetch_buffer[:rows])

    def test_undisturbed_counts_checks_and_no_unequal(self):
        e, native, decode, _ = self.make()
        self.run_lane(e, native, 1024)
        self.run_lane(e, decode, 8)
        s = mh.PLE_WITNESS
        self.assertEqual((s['native_checks'], s['native_unequal'], s['decode_checks'], s['decode_unequal'], s['missing_ids'], s['errors']), (1, 0, 1, 0, 0, 0))

    def test_decode_writing_inside_native_window_is_seen_on_native(self):
        e, native, decode, _ = self.make()
        out = self.run_lane(e, native, 1024, clobber=lambda: e._prefetch_buffer[:2].fill_(99))
        s = mh.PLE_WITNESS
        self.assertEqual((s['native_unequal'], s['native_rows'], s['decode_unequal']), (1, 2, 0))
        self.assertEqual(tuple(out.shape), (1024, HEADS, DIM))

    def test_native_writing_inside_decode_window_is_seen_on_decode(self):
        e, native, decode, _ = self.make()
        self.run_lane(e, decode, 8, clobber=lambda: e._prefetch_buffer[:8].fill_(99))
        s = mh.PLE_WITNESS
        self.assertEqual((s['decode_unequal'], s['decode_rows'], s['native_unequal']), (1, 8, 0))

    def test_a_stale_row_past_the_first_8_is_outside_the_witness(self):
        e, native, decode, _ = self.make()
        self.run_lane(e, native, 1024, clobber=lambda: e._prefetch_buffer[8:9].fill_(99))
        self.assertEqual(mh.PLE_WITNESS['native_unequal'], 0)

    def test_off_it_does_nothing(self):
        e, native, decode, _ = self.make()
        mh.select_ple_witness(False)
        self.run_lane(e, native, 1024, clobber=lambda: e._prefetch_buffer[:2].fill_(99))
        s = mh.PLE_WITNESS
        self.assertEqual((s['native_checks'], s['native_unequal'], s['missing_ids']), (0, 0, 0))

    def test_a_failure_is_counted_not_raised(self):
        e, native, decode, witness = self.make()
        e._prefetch_stream = native
        ids = torch.ones((8, HEADS), dtype=torch.int64)
        e._lookup(ids, output=e._prefetch_buffer[:8])
        witness._stock_lookup = mock.Mock(side_effect=RuntimeError('boom'))
        e._reduce_etp_embeddings(e._prefetch_buffer[:8])
        self.assertEqual(mh.PLE_WITNESS['errors'], 1)


if __name__ == '__main__':
    unittest.main()
