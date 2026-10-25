"""E2 adapter CPU tests. Fake DMA completion is not a CUDA visibility proof."""
import ast
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'patches/s5_engine'))
from retention_contract import ContractError, Pending
from serial_dma import Binding, SerialDMA, TensorFacts
import resident_attention
from test_s5_retention import fixture, LAYERS
from test_sparse_stage import old_plan, staged


class Buffer:
    def __init__(self, shape, device, address):
        self.facts = TensorFacts(device, shape, (shape[2]*512, shape[2]*512, 512, 1),
                                 'torch.bfloat16', 2, device == 'cpu', True, address)
        self.data = bytearray([255])*self.facts.nbytes


class Event:
    def __init__(self, action=lambda: None):
        self.action, self.waited = action, False


class Backend:
    device = 'cuda:0'
    def __init__(self):
        self.submissions = []; self.fail_submit = self.fail_wait = False
        self.before_wait = lambda: None
    def facts(self, buffer): return buffer.facts
    def copy_rows(self, binding, source_row, destination_row, count):
        if self.fail_submit: raise RuntimeError('synthetic submit failure')
        self.submissions.append((source_row, destination_row, count))
        def copy():
            binding.resident.data[destination_row*1024:(destination_row+count)*1024] = (
                binding.host.data[source_row*1024:(source_row+count)*1024])
        return Event(copy)
    def wait(self, event):
        self.before_wait()
        if self.fail_wait: raise RuntimeError('synthetic completion failure')
        event.action(); event.waited = True


def dma_fixture(**kwargs):
    h, tables = fixture(**kwargs)
    backend = Backend(); bindings = {}
    for index, layer in enumerate(LAYERS):
        host = Buffer((32,1,4,512), 'cpu', 1_000_000*(index+1))
        dst = Buffer((h.bank.capacity_pages,1,4,512), backend.device, 1_000_000*(index+1))
        for page in range(32):
            host.data[page*4096:(page+1)*4096] = bytes([page])*4096
        bindings[layer] = Binding(host, dst)
    return h, tables, backend, bindings


class SerialTests(unittest.TestCase):
    def test_fragmented_bounded_dma_and_invalid_tail_untouched(self):
        h, _, b, bindings = dma_fixture(end=26, ids=[1,2,3,4,5,19,20])
        adapter = SerialDMA(h.bank, bindings, b)
        observed = []
        def pending():
            observed.append(sum(w.ticket.kind == 'copy' for w in h.pending.values()))
            with self.assertRaises(Pending): h.attend(LAYERS[0])
        b.before_wait = pending
        result = adapter.fill(h, LAYERS[0])
        self.assertEqual(result, {'copies': 3, 'bytes': 26*1024})
        self.assertEqual(observed, [1,1,1])
        self.assertEqual(b.submissions, [(4,0,16),(20,16,4),(76,20,6)])
        dst = bindings[LAYERS[0]].resident.data
        self.assertEqual(dst[20*1024:24*1024], bytes([19])*4096)
        self.assertEqual(dst[24*1024:26*1024], bytes([20])*2048)
        self.assertEqual(dst[26*1024:], bytes([255])*(len(dst)-26*1024))
        self.assertEqual(adapter.fill(h, LAYERS[0]), {'copies':0,'bytes':0})
        self.assertEqual(adapter.bytes_completed, 26*1024)

    def test_producer_event_required_and_six_layers_independent(self):
        h, tables, b, bindings = dma_fixture()
        adapter = SerialDMA(h.bank, bindings, b); layer = LAYERS[0]
        h.pool.page(tables[layer][1]).valid = 1
        t = h.store(layer,1,2)
        with self.assertRaises(Pending): adapter.fill(h,layer)
        e = Event(); adapter.producer_complete(h,t,e)
        self.assertTrue(e.waited)
        adapter.fill(h,layer)
        self.assertTrue(h.plan(LAYERS[1]))
        consumer, tensor = adapter.begin_attention(h,layer)
        self.assertIs(tensor,bindings[layer].resident)
        self.assertNotIn(layer,h.attended)
        adapter.attention_complete(h,consumer,Event())
        self.assertIn(layer,h.attended)

    def test_physical_index_bounds_checked_before_first_transfer(self):
        h, _, b, bindings = dma_fixture(end=8,ids=[1,32])
        adapter = SerialDMA(h.bank,bindings,b)
        with self.assertRaises(ContractError): adapter.fill(h,LAYERS[0])
        self.assertFalse(b.submissions); self.assertFalse(h.pending)

    def test_uva_unpinned_wrong_stride_dtype_or_device_rejected(self):
        mutations = [dict(device='cuda:0'), dict(pinned=False), dict(contiguous=False),
                     dict(dtype='torch.float16'), dict(strides=(1,1,1,1)), dict(shape=(32,2,4,512))]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                h,_,b,bindings = dma_fixture()
                host = bindings[LAYERS[0]].host
                host.facts = replace(host.facts,**mutation)
                with self.assertRaises(ContractError): SerialDMA(h.bank,bindings,b)
        h,_,b,bindings = dma_fixture()
        dst=bindings[LAYERS[0]].resident; dst.facts=replace(dst.facts,device='cuda:1')
        with self.assertRaises(ContractError): SerialDMA(h.bank,bindings,b)

    def test_capacity_missing_layer_and_alias_rejected(self):
        for kind in ('capacity','missing','alias'):
            h,_,b,bindings=dma_fixture()
            if kind == 'missing': bindings.pop(LAYERS[1])
            elif kind == 'alias': bindings[LAYERS[1]]=bindings[LAYERS[0]]
            else:
                dst=bindings[LAYERS[0]].resident; dst.facts=replace(dst.facts,shape=(7,1,4,512))
            with self.assertRaises(ContractError): SerialDMA(h.bank,bindings,b)

    def test_resize_or_storage_change_rejected_after_binding(self):
        h,_,b,bindings=dma_fixture(); adapter=SerialDMA(h.bank,bindings,b)
        host=bindings[LAYERS[0]].host; host.facts=replace(host.facts,address=99)
        with self.assertRaises(ContractError): adapter.fill(h,LAYERS[0])
        self.assertFalse(b.submissions)

    def test_submit_and_wait_errors_quarantine_without_freeing(self):
        for which in ('fail_submit','fail_wait'):
            h,_,b,bindings=dma_fixture(); adapter=SerialDMA(h.bank,bindings,b)
            setattr(b,which,True)
            with self.assertRaises(RuntimeError): adapter.fill(h,LAYERS[0])
            self.assertTrue(h.quarantined); self.assertTrue(h.pending); self.assertTrue(h.held)
            self.assertEqual(adapter.bytes_completed,0)
            with self.assertRaises(Pending): adapter.close()
            self.assertFalse(h.shutdown()['clean'])

    def test_cancel_drains_existing_attention_then_close(self):
        h,_,b,bindings=dma_fixture(); adapter=SerialDMA(h.bank,bindings,b)
        adapter.fill(h,LAYERS[0]); ticket,_=adapter.begin_attention(h,LAYERS[0])
        h.cancel()
        with self.assertRaises(Pending): h.retire()
        adapter.attention_complete(h,ticket,Event())
        with self.assertRaises(ContractError): adapter.fill(h,LAYERS[1])
        h.retire(); adapter.close(); self.assertFalse(adapter.bindings)

    def test_cancel_drains_existing_producer_without_publishing(self):
        h,tables,b,bindings=dma_fixture(); adapter=SerialDMA(h.bank,bindings,b)
        layer=LAYERS[0]; h.pool.page(tables[layer][1]).valid=1
        ticket=h.store(layer,1,2); h.cancel()
        adapter.producer_complete(h,ticket,Event())
        self.assertEqual(h.published,0)
        h.retire(); adapter.close()

    def test_source_mutation_during_copy_quarantines_without_readiness(self):
        h,tables,b,bindings=dma_fixture(); adapter=SerialDMA(h.bank,bindings,b)
        def mutate(): h.pool.page(tables[LAYERS[0]][0]).version += 1
        b.before_wait=mutate
        with self.assertRaises(ContractError): adapter.fill(h,LAYERS[0])
        self.assertTrue(h.quarantined); self.assertTrue(h.pending)
        self.assertFalse(h.copied); self.assertEqual(adapter.bytes_completed,0)

    def test_foreign_bank_and_duplicate_completion_rejected(self):
        h,_,b,bindings=dma_fixture(); adapter=SerialDMA(h.bank,bindings,b)
        other=fixture()[0]
        with self.assertRaises(ContractError): adapter.fill(other,LAYERS[0])
        adapter.fill(h,LAYERS[0]); ticket,_=adapter.begin_attention(h,LAYERS[0])
        adapter.attention_complete(h,ticket,Event())
        with self.assertRaises(ContractError): adapter.attention_complete(h,ticket,Event())


class AttentionTests(unittest.TestCase):
    def source(self): return (ROOT/'patches/retain/ops_qsa.py.v8.1').read_text()

    def test_stock_source_and_kernel_are_untouched(self):
        source=self.source(); generated=resident_attention.generate(source)
        self.assertTrue(generated.startswith(source+'\n\n'))
        stock=resident_attention.function(source,resident_attention.NAME)
        self.assertEqual(stock,resident_attention.function(generated,resident_attention.NAME))
        candidate=resident_attention.function(generated,'flash_e2_resident_attention')
        # Everything from the original num_rows through all sparse remapping,
        # arena gathers and attention accumulator updates is the original text.
        anchor='    num_rows = rows.numel()\n'
        self.assertEqual(candidate[candidate.index(anchor):],stock[stock.index(anchor):])
        ast.parse(generated)
        self.assertIn('kv_cache = resident',candidate)
        self.assertIn('torch.arange(max_logical_page + 1',candidate)

    def test_source_pin_rejects_drift(self):
        with self.assertRaises(ValueError): resident_attention.generate(self.source()+'\n')

    def test_fragmented_source_substitution_preserves_sparse_dense_empty_arithmetic(self):
        rng=np.random.default_rng(121)
        page,arena,pages=8,3,7; n=page*pages
        physical=np.array([11,3,7,9,1,5,10])
        host=rng.normal(size=(12,page,8)).astype('f')
        resident=host[physical].copy()
        q=rng.normal(size=(3,8)).astype('f'); rows=np.array([2,0])
        for count in (0,1,23,24,25,44,45,56):
            ids=np.full((3,65),-1,np.int32)
            for r in rows:
                ids[r,:count]=rng.choice(n,count,replace=False); ids[r,-1]=count
            baseline=old_plan(ids,rows,n,page,arena)
            logical_host=host[physical].reshape(n,8)
            actual=staged(baseline,ids,rows,resident.reshape(n,8),q,page)
            expected=staged(baseline,ids,rows,logical_host,q,page)
            self.assertEqual(actual.tobytes(),expected.tobytes())
            if count==n: self.assertEqual(baseline[2],[(0,3),(3,6),(6,7)])
        # Actual production dense bucket boundaries; never one 50-page bucket.
        self.assertEqual([(lo,min(lo+23,50)) for lo in range(0,50,23)],[(0,23),(23,46),(46,50)])


if __name__ == '__main__': unittest.main()
