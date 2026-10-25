"""CPU state-machine evidence only; injected completion is not a CUDA fence."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'patches/s5_engine'))
from retention_contract import Pool,RequestKey,ContractError,Pending
from arena_contract import Arena,Owner

LAYERS=tuple('qsa'+str(i) for i in range(6))

def setup(pages=50,tokens=None):
    p=Pool(1,4);a=Arena(p,LAYERS);q=RequestKey(1,'slot0',1);a.register(q)
    tokens=tokens or pages*4
    keys=tuple(p.allocate(LAYERS[0],i+1,min(4,tokens-4*i)) for i in range(pages))
    o=Owner(q,0,LAYERS[0]);a.acquire(o,keys,tokens)
    return p,a,q,o,keys

def drain(a):
    spans=[]
    while a.next_page<len(a.sources):
        b,t=a.begin_fill();spans.append((b.lo,b.hi,b.epoch))
        a.complete_fill(t);a.complete_attention(a.begin_attention(b))
    a.release();return spans

class ArenaContractTests(unittest.TestCase):
    def test_50_pages_23_23_4_and_no_persistent_destination(self):
        p,a,q,o,keys=setup();spans=drain(a)
        self.assertEqual(spans,[(0,23,1),(23,46,2),(46,50,3)])
        self.assertEqual(p.held_count(),0);self.assertEqual(a.incremental_kv_bytes,0)
        self.assertEqual(a.storage_bytes,23*4*1024)
        self.assertEqual(a.counters()['valid_arena_pages'],0)
        a.acquire(replace(o,chunk=1),keys,200)
        self.assertEqual(drain(a),[(0,23,4),(23,46,5),(46,50,6)]) # unchanged sources still ALL recopied
        a.retire(q);self.assertEqual(a.counters()['registered_requests'],0)

    def test_copy_piece_plan_fragmentation_and_partial_last_page(self):
        p,a,q,o,keys=setup(pages=5,tokens=18)
        b,t=a.begin_fill();self.assertEqual([s.rows for s in b.sources],[4,4,4,4,2])
        self.assertEqual([(x.source_page,x.arena_page,x.pages) for x in b.pieces],[(1,0,4),(5,4,1)])
        a.complete_fill(t);a.complete_attention(a.begin_attention(b));a.release()
        a.acquire(replace(o,chunk=1),[keys[3],keys[1],keys[0],keys[2],keys[4]],18)
        b,t=a.begin_fill();self.assertEqual([(x.source_page,x.arena_page,x.pages) for x in b.pieces],[(4,0,1),(2,1,1),(1,2,1),(3,3,1),(5,4,1)])

    def test_one_arena_across_layers_and_no_reuse_before_completion(self):
        p,a,q,o,keys=setup(pages=1)
        other=p.allocate(LAYERS[1],1,4)
        with self.assertRaises(Pending):a.acquire(replace(o,layer=LAYERS[1]),[other],4)
        b,t=a.begin_fill()
        for fn in (a.begin_fill,a.release,lambda:a.begin_attention(b),lambda:p.free(keys[0])):
            with self.assertRaises(Pending):fn()
        a.complete_fill(t);c=a.begin_attention(b)
        for fn in (a.begin_fill,a.release):
            with self.assertRaises(Pending):fn()
        a.complete_attention(c);a.release()
        a.acquire(replace(o,layer=LAYERS[1]),[other],4)
        b2,t2=a.begin_fill();self.assertGreater(b2.epoch,b.epoch)
        with self.assertRaises(ContractError):a.complete_fill(t)
        a.complete_fill(t2)
        with self.assertRaises(ContractError):a.begin_attention(b)
        a.complete_attention(a.begin_attention(b2));a.release()

    def test_out_of_order_wrong_owner_duplicate_and_forged_completion(self):
        p,a,q,o,keys=setup(pages=1);b,t=a.begin_fill()
        for bad in (replace(t,epoch=t.epoch+1),replace(t,owner=replace(o,chunk=2)),replace(t,serial=99)):
            with self.assertRaises(ContractError):a.complete_fill(bad)
        self.assertEqual(a.state,'filling');a.complete_fill(t)
        with self.assertRaises(ContractError):a.complete_fill(t)
        c=a.begin_attention(b);a.complete_attention(c)
        with self.assertRaises(ContractError):a.complete_attention(c)
        a.release()
        with self.assertRaises(ContractError):a.acquire(o,keys,4)

    def test_cancel_in_each_phase_holds_until_pending_completes(self):
        for phase in ('owned','filling','ready','consuming'):
            with self.subTest(phase=phase):
                p,a,q,o,keys=setup(pages=1);ticket=None
                if phase!='owned':b,ticket=a.begin_fill()
                if phase in ('ready','consuming'):a.complete_fill(ticket)
                if phase=='consuming':ticket=a.begin_attention(b)
                a.cancel(q)
                with self.assertRaises(ContractError):a.begin_fill()
                if phase in ('filling','consuming'):
                    with self.assertRaises(Pending):a.release()
                    with self.assertRaises(Pending):a.retire(q)
                    (a.complete_fill if phase=='filling' else a.complete_attention)(ticket)
                a.release();a.retire(q);p.free(keys[0])
                self.assertEqual(a.counters()['source_leases'],0)

    def test_shutdown_pending_and_failed_fences_never_fake_clean(self):
        for kind in ('copy','consume'):
            p,a,q,o,keys=setup(pages=1);b,t=a.begin_fill()
            if kind=='consume':a.complete_fill(t);t=a.begin_attention(b)
            r=a.shutdown();self.assertEqual(r['active_owners'],1);self.assertEqual(p.held_count(),1)
            (a.complete_fill if kind=='copy' else a.complete_attention)(t)
            r=a.shutdown();self.assertEqual(r['active_owners'],0);self.assertEqual(r['registered_requests'],0)
            p,a,q,o,keys=setup(pages=1);b,t=a.begin_fill()
            if kind=='consume':a.complete_fill(t);t=a.begin_attention(b)
            (a.complete_fill if kind=='copy' else a.complete_attention)(t,False)
            r=a.shutdown();self.assertTrue(r['quarantined']);self.assertEqual(r['source_leases'],1)
            with self.assertRaises(Pending):a.release()
            with self.assertRaises(Pending):p.free(keys[0])

    def test_producer_pending_and_source_versions_are_checked(self):
        p=Pool(1,4);a=Arena(p,LAYERS);q=RequestKey(1,'slot0',1);a.register(q)
        k=p.allocate(LAYERS[0],1,2);o=Owner(q,0,LAYERS[0])
        with self.assertRaises(Pending):a.acquire(o,[k],3)
        p.page(k).valid=3;p.page(k).writer='pending'
        with self.assertRaises(Pending):a.acquire(o,[k],3)
        p.page(k).writer=None;a.acquire(o,[k],3)
        b,t=a.begin_fill();p.page(k).version+=1 # simulates a broken engine lease, never legitimate
        with self.assertRaises(ContractError):a.complete_fill(t)
        self.assertTrue(a.counters()['quarantined']);self.assertEqual(a.counters()['pending_dma'],1)

    def test_request_slot_reuse_and_physical_reincarnation_reject_old_events(self):
        p,a,q,o,keys=setup(pages=1);b,t=a.begin_fill();a.complete_fill(t)
        a.complete_attention(a.begin_attention(b));a.release();a.retire(q)
        with self.assertRaises(ContractError):a.register(q)
        newq=replace(q,generation=2);a.register(newq);p.free(keys[0]);newkey=p.allocate(LAYERS[0],1,4)
        with self.assertRaises(ContractError):a.acquire(Owner(newq,0,LAYERS[0]),keys,4)
        a.acquire(Owner(newq,0,LAYERS[0]),[newkey],4);nb,nt=a.begin_fill()
        with self.assertRaises(ContractError):a.complete_fill(t)
        self.assertNotEqual(nb.sources[0].page,keys[0]);self.assertEqual(a.pending,nt)

    def test_cow_rebind_is_new_owner_not_stale_arena_contents(self):
        p,a,q,o,keys=setup(pages=1);drain(a)
        p.page(keys[0]).readonly=True
        cow=p.allocate(LAYERS[0],2,4);p.page(cow).version=1
        a.acquire(replace(o,chunk=1),[cow],4);b,t=a.begin_fill()
        self.assertEqual(b.sources[0].page,cow);self.assertEqual(b.sources[0].version,1)
        self.assertEqual(p.page(keys[0]).holders,set());self.assertEqual(p.page(cow).holders,{replace(o,chunk=1)})

    def test_bad_acquisition_is_atomic_and_source_table_is_frozen(self):
        p,a,q,o,keys=setup(pages=1);drain(a)
        for wrong in ([keys[0],keys[0]],[replace(keys[0],layer=LAYERS[1])]):
            with self.assertRaises(ContractError):a.acquire(replace(o,chunk=1),wrong,4*len(wrong))
            self.assertEqual(a.state,'idle');self.assertEqual(p.held_count(),0)
        mutable=list(keys);a.acquire(replace(o,chunk=1),mutable,4);mutable.clear()
        b,t=a.begin_fill();self.assertEqual(len(b.sources),1)

if __name__=='__main__':unittest.main()
