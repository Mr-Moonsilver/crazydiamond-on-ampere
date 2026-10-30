"""Host-only alias/lifetime receipts; fake CPU-backed Tensor protocol is a leaf."""
import ctypes
import gc
import unittest
import weakref

from native34_storage import AllocationWindow,compact,tensor_view,overlap,MAX_HISTORY,compare_manifests


class Storage:
    def __init__(self,n=256):self.buffer=ctypes.create_string_buffer(n);self._cdata=id(self)
    def data_ptr(self):return ctypes.addressof(self.buffer)
    def nbytes(self):return len(self.buffer)


class View:
    dtype='int32';device='cuda:0'
    def __init__(self,storage,offset=0,shape=(8,),stride=(1,)):
        self.storage=storage;self.offset=offset;self.shape=shape;self.steps=stride
    def untyped_storage(self):return self.storage
    def data_ptr(self):return 0 if any(n==0 for n in self.shape) else self.storage.data_ptr()+self.offset*4
    def element_size(self):return 4
    def storage_offset(self):return self.offset
    def stride(self):return self.steps
    def item(self):raise AssertionError('no-device-readback')
    def cpu(self):raise AssertionError('no-device-readback')
    def tolist(self):raise AssertionError('no-device-readback')


def snapshot(base,events=()):
    return dict(segments=[dict(device=0,stream=11,segment_pool_id=(0,7),
        blocks=[dict(address=base,size=256,state='active_allocated')])],
        device_traces=[list(events)])


class StorageReceipts(unittest.TestCase):
    def test_different_pointers_can_overlap_same_allocation(self):
        s=Storage();a=tensor_view(View(s));b=tensor_view(View(s,4));c=tensor_view(View(s,16))
        self.assertNotEqual(a['data_ptr'],b['data_ptr']);self.assertTrue(overlap(a,b))
        self.assertFalse(overlap(a,c));self.assertEqual(a['allocation_base'],b['allocation_base'])
        r=compact(dict(a=View(s),b=View(s,4),c=View(s,16)))
        self.assertEqual(r['overlapping_ranges'],[['a','b']])
        compared=compare_manifests(compact(dict(captured=View(s))),compact(dict(live=View(s,4))))
        self.assertTrue(compared['complete']);self.assertEqual(len(compared['overlaps']),1)
        self.assertFalse(compare_manifests(r,r,limit=0)['complete'])

    def test_bounds_violation_is_receipt_and_empty_has_no_access(self):
        s=Storage();bad=tensor_view(View(s,63,shape=(8,)))
        self.assertFalse(bad['view_within_storage'])
        empty=tensor_view(View(s,shape=(0,)))
        self.assertTrue(empty['view_within_storage'])
        self.assertFalse(overlap(empty,tensor_view(View(s))))
        reverse=tensor_view(View(s,7,stride=(-1,)))
        self.assertEqual(reverse['view_start'],s.data_ptr())
        self.assertEqual(reverse['view_end'],s.data_ptr()+32)

    def test_metadata_does_not_keep_tensor_or_storage_alive(self):
        def collect():
            s=Storage();v=View(s)
            return compact(dict(v=v)),weakref.ref(v),weakref.ref(s)
        result,tensor,storage=collect();gc.collect()
        self.assertIsNone(tensor());self.assertIsNone(storage())
        self.assertFalse(result['tensor_retention']);self.assertFalse(result['device_readback'])

    def test_same_address_new_allocation_has_new_incarnation(self):
        s=Storage();v=tensor_view(View(s));base=s.data_ptr();w=AllocationWindow()
        w.start(snapshot(base));first=w.identify(v)
        self.assertEqual(first['pre_window_creation'],'unknown')
        trace=[dict(action='free_requested',addr=base),dict(action='free_completed',addr=base),
               dict(action='alloc',addr=base,size=256,stream=12,pool_id=(0,7))]
        w.update(snapshot(base,trace));second=w.identify(v)
        self.assertNotEqual(first['incarnation'],second['incarnation'])
        self.assertTrue(second['creation_observed']);self.assertEqual(second['stream'],12)

    def test_unretired_free_and_history_loss_are_not_safe(self):
        s=Storage();base=s.data_ptr();v=tensor_view(View(s));w=AllocationWindow();w.start(snapshot(base))
        w.update(snapshot(base,[dict(action='free_requested',addr=base)]))
        self.assertEqual(w.identify(v)['state'],'active_awaiting_free')
        w.update(snapshot(base,[dict(action='snapshot')]*MAX_HISTORY))
        self.assertFalse(w.identify(v)['lifecycle_complete'])
        external=tensor_view(View(Storage()));self.assertFalse(w.identify(external)['available'])


if __name__=='__main__':unittest.main()
