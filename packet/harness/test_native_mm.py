"""CPU ownership/wire proof. Four workers are real bridges with fake CUDA/PP leaves."""
from dataclasses import replace
import contextlib
import hashlib
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'patches/s5_engine'))
import native_mm as mm
import native_mm_overlay
from e3_types import TransactionId
from worker_bridge import admission_to_chunk, WorkerBridgeError
from test_e3_r4_worker_bridge import Stack
from test_e3_admission_native import Request, Scheduler, Manager, make_controller


def payload(tx, epoch=1, rows=(1, 3)):
    data = b'\x80\x3f' * (len(rows)*mm.WIDTH)
    return mm.MMChunk(tx, epoch, rows, data, hashlib.sha256(data).hexdigest())


def cached_request():
    request = Request(tokens=12)
    request.has_encoder_inputs = True
    place = types.SimpleNamespace(offset=1, length=4, is_embed=None)
    tensor, grid = object(), object()
    item = {'image_embeds': types.SimpleNamespace(data=tensor), 'image_grid_thw': types.SimpleNamespace(data=grid)}
    feature = types.SimpleNamespace(modality='image', data=item, mm_position=place, identifier='fixture-image')
    request.mm_features = [feature]
    signature = (request.request_id, id(feature), id(item), id(tensor), id(grid), feature.identifier, 1, 4, id(None))
    request._flash_s5_mm_plan = mm.ImagePlan(signature, (1,2,3,4), b'\x80\x3f'*(4*mm.WIDTH))
    return request


class NativeMM(unittest.TestCase):
    def setUp(self):
        self.flag = patch.dict(os.environ, FLASH_S5_NATIVE_MM='1'); self.flag.start(); self.addCleanup(self.flag.stop)
        self.tx = TransactionId(1,'mm-test',1,0,8,1)

    def test_typed_bytes_rows_epoch_and_generation_refuse(self):
        value = payload(self.tx)
        self.assertNotIn('mm-test', repr(value))
        with self.assertRaises(mm.NativeMMError): replace(value, data=value.data[:-1]+b'x')
        with self.assertRaises(mm.NativeMMError): replace(value, rows=(1,1))
        with self.assertRaises(mm.NativeMMError): value.validate(replace(self.tx, request_generation=2),1)
        with self.assertRaises(mm.NativeMMError): value.validate(self.tx,2)
        with self.assertRaises(mm.NativeMMError): replace(value, rows=(1,8))

    def test_snapshot_slices_noncontiguous_rows_and_cached_prefix(self):
        plan = mm.ImagePlan((), (1,3,8), b'\x80\x3f'*(3*mm.WIDTH))
        self.assertEqual(plan.chunk(self.tx,1).rows,(1,3))
        later = replace(self.tx,chunk_start=7,chunk_end=11,txn_seq=2)
        self.assertEqual(plan.chunk(later,2).rows,(1,))
        self.assertIsNone(plan.chunk(replace(later,chunk_start=9),2))

    def test_cached_request_metadata_change_refuses(self):
        request = cached_request()
        self.assertTrue(mm.eligible(request))
        request.mm_features[0].mm_position.offset = 2
        self.assertFalse(mm.eligible(request))
        self.assertIn('changed', request._flash_s5_mm_refusal)

    def test_default_off_keeps_encoder_refusal_and_budget(self):
        with patch.dict(os.environ,FLASH_S5_NATIVE_MM='0'):
            request = cached_request()
            self.assertFalse(mm.eligible(request))
            self.assertEqual(mm.cap_encoder_budget(200000,200000),(200000,200000))
            stack = Stack()
            record = replace(stack.record,mm=payload(stack.record.transaction,stack.record.state_epoch))
            with self.assertRaisesRegex(WorkerBridgeError,'off'): admission_to_chunk(record)

    def test_bound_budget_and_native_admission_preserve_payload(self):
        self.assertEqual(mm.cap_encoder_budget(200000,200000),(8192,8192))
        request = cached_request(); scheduler=Scheduler(Manager(),[request])
        controller=make_controller(scheduler,chunk=4,ranks=(0,1,2,3))
        record=controller.admit(scheduler)
        self.assertIsNotNone(record.mm)
        self.assertEqual(record.mm.rows,(1,2,3))
        chunk=admission_to_chunk(record)
        self.assertIs(chunk.mm,record.mm)
        self.assertEqual(chunk.mm.transaction,chunk.transaction)

    def test_budget_transform_is_pinned_and_default_off(self):
        source='class Budget:\n    def initialize(self, encoder_compute_budget, encoder_cache_size):\n'+native_mm_overlay.OLD
        with self.assertRaisesRegex(ValueError,'pinned'):native_mm_overlay.budget_source(source)
        with patch.object(native_mm_overlay,'BUDGET_SHA',hashlib.sha256(source.encode()).hexdigest()):
            changed=native_mm_overlay.budget_source(source)
            namespace={};exec(changed,namespace)
            budget=namespace['Budget']()
            with patch.dict(os.environ,FLASH_S5_NATIVE_MM='0'):
                budget.initialize(200000,200000)
            self.assertEqual((budget.encoder_compute_budget,budget.encoder_cache_size),(200000,200000))
            with patch.dict(sys.modules,{'vllm.v1.flash_e3.native_mm':mm}):
                budget.initialize(200000,200000)
            self.assertEqual((budget.encoder_compute_budget,budget.encoder_cache_size),(8192,8192))

    def test_prefix_lease_requires_queried_completion(self):
        lease=mm.SpliceLease(payload(self.tx)); lease.keepalive.append(object())
        with self.assertRaises(mm.NativeMMError): lease.release()
        event=types.SimpleNamespace(complete=False); event.query=lambda:event.complete
        lease.bind(event)
        with self.assertRaises(mm.NativeMMError):lease.release()
        self.assertEqual(len(lease.keepalive),1)
        event.complete=True;lease.release();self.assertEqual(lease.keepalive,[])

    def test_private_feature_upload_refuses_shared_storage(self):
        tensor=types.SimpleNamespace(shape=(1,2560),contiguous=lambda:object())
        feature=types.SimpleNamespace(modality='image',data={'image_embeds':types.SimpleNamespace(data=tensor),'image_grid_thw':object()},
            mm_position=types.SimpleNamespace(get_num_embeds=lambda:1))
        result=types.SimpleNamespace(nbytes=5120,untyped_storage=lambda:types.SimpleNamespace(nbytes=lambda:5120))
        with patch.object(mm,'image_tensor',return_value=tensor):
            self.assertIs(mm.stage_image(feature,'cuda',lambda value,device:result),result)
            result.untyped_storage=lambda:types.SimpleNamespace(nbytes=lambda:40960)
            with self.assertRaisesRegex(mm.NativeMMError,'shared storage'):
                mm.stage_image(feature,'cuda',lambda value,device:result)

    def test_inventory_unwraps_hidden_tower_and_counts_unique_storage(self):
        def tensor(size,device='cuda',ptr=1):
            return types.SimpleNamespace(nbytes=size,device=types.SimpleNamespace(type=device),
                untyped_storage=lambda:types.SimpleNamespace(data_ptr=lambda:ptr,nbytes=lambda:size))
        actual=types.SimpleNamespace(parameters=lambda:[tensor(123)],buffers=lambda:[tensor(42,'meta')])
        proxy=types.SimpleNamespace(module=actual,parameters=lambda:[],buffers=lambda:[])
        cached=tensor(5120,ptr=7)
        runner=types.SimpleNamespace(model=types.SimpleNamespace(visual=proxy),
            encoder_cache=types.SimpleNamespace(encoder_outputs={'a':cached,'b':cached}),
            model_state=types.SimpleNamespace(encoder_runner=types.SimpleNamespace(inputs_embeds=tensor(99))))
        record=mm.inventory(runner)
        self.assertEqual(record['tower_registered_cuda_bytes'],123)
        self.assertEqual(record['cache_feature_bytes'],10240)
        self.assertEqual(record['cache_storage_bytes'],5120)
        self.assertEqual(record['encoder_inputs_buffer_bytes'],99)

    def stack(self):
        stack=Stack()
        stack.record=replace(stack.record,mm=payload(stack.record.transaction,stack.record.state_epoch))
        stack.controller.parked.admission=stack.record
        leases=[]
        for worker in stack.workers.values():
            if worker.stage != 0: continue
            lane=worker.lane; old=lane.begin
            @contextlib.contextmanager
            def scope(*,lane=lane):
                lane.mm_scope=True
                try:yield
                finally:lane.mm_scope=False
            lane._scopes=scope
            def begin(intermediate_tensors=None,inputs_embeds=None,*,worker=worker,old=old):
                self.assertEqual(inputs_embeds,'MM_CPU_SPLICE')
                record=old(intermediate_tensors)
                record.event=worker.torch.cuda.Event(enable_timing=False)
                record.event.record(worker.split.prefill_stream)
                return record
            lane.begin=begin
        def splice(lane,admission,torch):
            self.assertTrue(lane.mm_scope)  # embedding must retain the native TP communicator.
            admission.mm.validate(admission.transaction,admission.state_epoch)
            lane._native_mm_lease=mm.SpliceLease(admission.mm)
            lane._native_mm_lease.keepalive.append(object())
            leases.append(lane._native_mm_lease)
            return 'MM_CPU_SPLICE'
        return stack,leases,splice

    def test_real_four_worker_plan_does_not_retire_before_completion(self):
        stack,leases,splice=self.stack()
        with patch.object(mm,'prefix_inputs',side_effect=splice):
            stack.drive_to_finalize()
            self.assertEqual(len(leases),2)  # PP1 carries HC; no image device allocation.
            self.assertTrue(all(lease.keepalive for lease in leases))
            stack.advance();self.assertEqual(stack.controller.commits,0)
            stack.complete_everything()
            stack.drive(lambda: stack.controller.commits == 1)
        self.assertTrue(all(lease.keepalive==[] for lease in leases))
        self.assertTrue(stack.all_dispatchers_reset())

    def test_cancel_after_prefix_retires_without_publishing(self):
        stack,leases,splice=self.stack()
        with patch.object(mm,'prefix_inputs',side_effect=splice):
            stack.drive_to_finalize(); stack.complete_everything();stack.advance()
            self.assertTrue(stack.driver.stop('cancel'))
        self.assertEqual(stack.controller.commits,0)
        self.assertEqual(stack.cache_calls(),[])
        self.assertTrue(all(lease.keepalive==[] for lease in leases))

    def test_pending_cancel_keeps_the_image_sources(self):
        stack,leases,splice=self.stack()
        with patch.object(mm,'prefix_inputs',side_effect=splice):
            stack.drive_to_finalize()
            stack.driver.stop('cancel')
            self.assertIsNotNone(stack.driver.tx)
            self.assertTrue(all(lease.keepalive for lease in leases))
            stack.complete_everything()
            for _ in range(32):
                if stack.driver.tx is None:break
                stack.driver.pump_stop(1)
        self.assertIsNone(stack.driver.tx)
        self.assertTrue(all(lease.keepalive==[] for lease in leases))

if __name__ == '__main__': unittest.main()
