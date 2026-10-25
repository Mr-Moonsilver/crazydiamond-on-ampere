"""CPU-only E0 main-QSA history contract; injected completions are not GPU evidence."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'patches/s5_engine'))
from retention_contract import (Pool, RequestKey, RetentionBank, Publication,
                                ContractError, Pending)

LAYERS = tuple('qsa'+str(i) for i in range(6))


def fixture(rank=0, committed=0, end=6, ids=None, capacity=8):
    pool = Pool(1, page_tokens=4)
    bank = RetentionBank(pool, rank, LAYERS, capacity)
    h = bank.open(RequestKey(1, 'synthetic', 1), committed)
    ids = ids or list(range(1, (end+3)//4+1))
    tables = {layer: [pool.allocate(layer, p, min(4, end-i*4)) for i, p in enumerate(ids)] for layer in LAYERS}
    h.reserve(end, tables)
    return h, tables


def finish(h):
    for layer in h.bank.layers:
        for plan in h.plan(layer): h.complete(h.copy(plan))
        h.complete(h.attend(layer))
    h.complete(h.finalize())


def quartet(end=6):
    hs = [fixture(rank, end=end)[0] for rank in range(4)]
    return hs, Publication()


class RetentionContract(unittest.TestCase):
    def test_six_independent_histories_and_fixed_capacity_ledger(self):
        p = Pool(1)
        bank = RetentionBank(p, 0, LAYERS, 50)
        self.assertEqual(bank.reserved_bytes, 1243545600)
        self.assertEqual(RetentionBank(p, 1, LAYERS, 65).reserved_bytes, 1616609280)
        with self.assertRaises(ValueError): RetentionBank(p, 0, LAYERS[:5])
        h, _ = fixture()
        layer = LAYERS[0]
        for plan in h.plan(layer): h.complete(h.copy(plan))
        self.assertEqual(h.plan(layer), ())
        self.assertTrue(all(h.plan(other) for other in LAYERS[1:]))

    def test_fresh_source_write_is_not_readable_on_submission(self):
        p = Pool(1, 4); bank = RetentionBank(p, 0, LAYERS)
        h = bank.open(RequestKey(1, 'fresh', 1))
        tables = {layer: [p.allocate(layer, 1)] for layer in LAYERS}
        h.reserve(3, tables)
        with self.assertRaises(Pending): h.plan(LAYERS[0])
        t = h.store(LAYERS[0], 0, 3)
        self.assertEqual(p.page(tables[LAYERS[0]][0]).valid, 0)
        with self.assertRaises(Pending): h.plan(LAYERS[0])
        h.complete(t)
        plan, = h.plan(LAYERS[0]); self.assertEqual(plan.bytes, 3*1024)
        self.assertEqual(plan.sources[0].version, 1)
        self.assertEqual(h.published, 0)

    def test_copy_completion_required_before_attention_and_one_dma_per_rank(self):
        h, _ = fixture(); a, b = LAYERS[:2]
        t = h.copy(h.plan(a)[0])
        with self.assertRaises(Pending): h.attend(a)
        with self.assertRaises(Pending): h.copy(h.plan(b)[0])
        self.assertNotIn((a, 0), h.copied)
        h.complete(t); self.assertFalse(h.plan(a))
        h.complete(h.attend(a))
        h.complete(h.copy(h.plan(b)[0]))

    def test_fragmented_coalescing_four_page_cap_and_valid_tail_only(self):
        h, _ = fixture(end=26, ids=[1,2,3,4,5,19,20])
        plans = h.plan(LAYERS[0])
        self.assertEqual([len(p.sources) for p in plans], [4,1,2])
        self.assertEqual([s.rows for s in plans[-1].sources], [4,2])
        self.assertEqual(sum(p.bytes for p in plans), 26*1024)
        # Apply the plan to fake host/resident bytes; uninitialized tail sentinel
        # must never be copied, regardless of physical fragmentation.
        page_bytes = 4*1024
        host = {s.page.physical: bytes([s.page.physical])*page_bytes for p in plans for s in p.sources}
        dst = bytearray([255])*(7*page_bytes)
        for plan in plans:
            for offset, snap in enumerate(plan.sources):
                lo = (plan.logical_start+offset)*page_bytes
                dst[lo:lo+snap.rows*1024] = host[snap.page.physical][:snap.rows*1024]
        self.assertEqual(dst[26*1024:], bytes([255])*(2*1024))
        self.assertEqual(dst[5*page_bytes:6*page_bytes], bytes([19])*page_bytes)

    def test_readonly_prefix_needs_cow_before_tail_write(self):
        h, tables = fixture(committed=4)
        key = tables[LAYERS[0]][1]; h.pool.page(key).readonly = True
        # New chunk target 6, actual host tail still only one valid row.
        h.pool.page(key).valid = 1
        with self.assertRaises(ContractError): h.store(LAYERS[0], 1, 2)
        self.assertEqual(h.pool.page(key).version, 0)

    def test_shared_page_pin_prevents_write_and_free(self):
        h, tables = fixture(); key = tables[LAYERS[0]][1]
        h.pool.page(key).valid = 1
        other = RequestKey(1, 'other', 1); h.pool.pin([key], other)
        with self.assertRaises(ContractError): h.store(LAYERS[0], 1, 2)
        with self.assertRaises(Pending): h.pool.free(key)
        h.pool.unpin([key], other)
        h.complete(h.store(LAYERS[0], 1, 2))

    def test_partial_tail_is_refreshed_on_next_chunk_not_trusted_forever(self):
        hs, pub = quartet(); [finish(h) for h in hs]; pub.commit(hs)
        h = hs[0]; tables = {k:list(v) for k,v in h.tables.items()}
        h.reserve(8, tables)
        for layer in LAYERS:
            h.complete(h.store(layer, 1, 4))
            plans = h.plan(layer)
            self.assertEqual(len(plans), 1); self.assertEqual(plans[0].logical_start, 1)
            self.assertEqual(plans[0].bytes, 4*1024) # conservative dirty page, not claimed minimum
            self.assertEqual(plans[0].sources[0].version, 1)

    def test_mapping_is_snapshotted_and_bad_last_layer_has_no_partial_pin(self):
        h, tables = fixture()
        before = h.tables[LAYERS[0]]; tables[LAYERS[0]].reverse()
        self.assertEqual(h.tables[LAYERS[0]], before)
        p = Pool(1,4); bank = RetentionBank(p,0,LAYERS); x = bank.open(RequestKey(1,'bad',1))
        rows = {layer:[p.allocate(layer,1,3)] for layer in LAYERS}
        rows[LAYERS[-1]] = [replace(rows[LAYERS[-1]][0],incarnation=99)]
        with self.assertRaises(ContractError): x.reserve(3,rows)
        self.assertEqual(p.held_count(),0); self.assertIsNone(x.end)

    def test_cow_mapping_refreshes_new_page_and_releases_old_only_after_commit(self):
        hs,pub = quartet(); [finish(h) for h in hs]; pub.commit(hs)
        h=hs[0]; layer=LAYERS[0]; old=h.tables[layer][0]
        new=h.pool.allocate(layer,20,4,readonly=True)
        tables={k:list(v) for k,v in h.tables.items()}; tables[layer][0]=new
        for row in tables.values(): h.pool.page(row[1]).valid=4 # externally completed append for this fixture
        h.reserve(8,tables)
        self.assertNotIn(h.request,h.pool.page(old).holders)
        self.assertEqual(h.plan(layer)[0].sources[0].page,new)
        self.assertNotIn((layer,0),h.copied)

    def test_aba_physical_reuse_and_request_generation_rejected(self):
        h,tables=fixture(); key=tables[LAYERS[0]][0]
        h.cancel(); h.retire(); h.pool.free(key)
        new=h.pool.allocate(key.layer,key.physical,4)
        self.assertNotEqual(new,key)
        with self.assertRaises(ContractError): h.pool.page(key)
        with self.assertRaises(ContractError): h.bank.open(h.request)
        next_h=h.bank.open(replace(h.request,generation=2))
        self.assertEqual(next_h.published,0)
        with self.assertRaises(ContractError): next_h.reserve(6,tables)

    def test_stale_plan_duplicate_completion_and_writer_race(self):
        h,tables=fixture(); layer=LAYERS[0]
        plan=h.plan(layer)[0]
        h.pool.page(tables[layer][0]).version += 1 # external adapter violation; plan must revalidate
        with self.assertRaises(ContractError): h.copy(plan)
        self.assertEqual(h.pending,{})
        t=h.copy(h.plan(layer)[0]); h.complete(t)
        with self.assertRaises(ContractError): h.complete(t)
        with self.assertRaises(ContractError): h.complete(replace(t,request=replace(t.request,generation=2)))

    def test_no_write_while_dma_or_attention_reads_and_no_double_attention(self):
        h,tables=fixture(); layer=LAYERS[0]
        t=h.copy(h.plan(layer)[0])
        with self.assertRaises(Pending): h.store(layer,1,2)
        h.complete(t); a=h.attend(layer)
        with self.assertRaises(Pending): h.store(layer,1,2)
        with self.assertRaises(ContractError): h.attend(layer)
        h.complete(a)
        with self.assertRaises(ContractError): h.attend(layer)

    def test_publication_waits_for_four_workers_and_state_finalize(self):
        hs,pub=quartet(); [finish(h) for h in hs[:3]]
        with self.assertRaises(Pending): pub.commit(hs)
        self.assertEqual([h.published for h in hs],[0]*4)
        last=hs[-1]
        for layer in LAYERS:
            for plan in last.plan(layer): last.complete(last.copy(plan))
            last.complete(last.attend(layer))
        t=last.finalize()
        with self.assertRaises(Pending): pub.commit(hs)
        last.complete(t); pub.commit(hs)
        self.assertEqual([h.published for h in hs],[6]*4)
        with self.assertRaises(ContractError): pub.commit(hs)
        self.assertEqual([h.commit_count for h in hs],[1]*4)

    def test_shared_prefix_arrival_sees_only_previous_committed_frontier(self):
        hs,pub=quartet(); [finish(h) for h in hs]; pub.commit(hs)
        for h in hs:
            tables={k:list(v) for k,v in h.tables.items()}
            h.reserve(8,tables)
        self.assertEqual([h.published for h in hs],[6]*4)
        for h in hs:
            for layer in LAYERS:h.complete(h.store(layer,1,4))
            finish(h)
        self.assertEqual([h.published for h in hs],[6]*4)
        pub.commit(hs); self.assertEqual([h.published for h in hs],[8]*4)

    def test_mismatched_rank_frontier_has_no_partial_publication(self):
        hs,pub=quartet(); [finish(h) for h in hs]
        with self.assertRaises(ContractError): pub.commit(hs[:3]+[hs[0]])
        hs[3].request=replace(hs[3].request,generation=2)
        with self.assertRaises(ContractError): pub.commit(hs)
        self.assertEqual([h.published for h in hs],[0]*4)

    def test_cancel_disconnect_preempt_drain_then_retire_never_publish(self):
        for reason in ('cancel','disconnect','preempt'):
            with self.subTest(reason=reason):
                h,_=fixture(); t=h.copy(h.plan(LAYERS[0])[0]); h.cancel()
                with self.assertRaises(Pending): h.retire()
                with self.assertRaises(ContractError): h.copy(h.plan(LAYERS[1])[0])
                with self.assertRaises(Pending): h.bank.open(replace(h.request,generation=2))
                h.complete(t); h.retire()
                self.assertEqual(h.pool.held_count(),0); self.assertEqual(h.published,0)
                with self.assertRaises(ContractError): h.ready()
                self.assertFalse(h.copied)

    def test_cancel_during_producer_and_attention_retain_real_pins(self):
        h,tables=fixture(); layer=LAYERS[0]
        h.pool.page(tables[layer][1]).valid=1
        t=h.store(layer,1,2); h.cancel()
        with self.assertRaises(Pending): h.retire()
        h.complete(t); h.retire(); self.assertEqual(h.pool.held_count(),0)
        h,_=fixture(); h.complete(h.copy(h.plan(layer)[0])); t=h.attend(layer); h.cancel()
        with self.assertRaises(Pending): h.retire()
        h.complete(t); h.retire(); self.assertEqual(h.pool.held_count(),0)

    def test_shutdown_pending_or_failed_event_quarantines_instead_of_freeing(self):
        h,_=fixture(); t=h.copy(h.plan(LAYERS[0])[0])
        r=h.shutdown(); self.assertFalse(r['clean']); self.assertGreater(r['held'],0)
        h.complete(t); self.assertTrue(h.shutdown()['clean'])
        h,_=fixture(); t=h.copy(h.plan(LAYERS[0])[0]); h.complete(t,success=False)
        r=h.shutdown(); self.assertTrue(r['quarantined']); self.assertFalse(r['clean'])
        with self.assertRaises(Pending): h.retire()
        with self.assertRaises(ContractError): h.complete(t)
        with self.assertRaises(Pending): h.bank.open(RequestKey(1,'next',1))

    def test_pool_reset_epoch_null_block_capacity_and_negative_index_guards(self):
        p=Pool(2,4); bank=RetentionBank(p,0,LAYERS)
        with self.assertRaises(ContractError):bank.open(RequestKey(1,'old-epoch',1))
        with self.assertRaises(ContractError):p.allocate(LAYERS[0],0)
        h,tables=fixture()
        with self.assertRaises(ContractError):h.store(LAYERS[0],-1,3)
        h.cancel(); h.retire(); fresh=h.bank.open(RequestKey(1,'new',1))
        with self.assertRaises(ContractError):fresh.reserve(33,tables)
        self.assertEqual(h.pool.held_count(),0)

    def test_bad_second_dma_source_leaves_entire_completion_unapplied(self):
        h,tables=fixture(); layer=LAYERS[0]; t=h.copy(h.plan(layer)[0])
        second=tables[layer][1]
        h.pool.page(second).version += 1 # injected adapter violation during DMA
        with self.assertRaises(ContractError):h.complete(t)
        self.assertEqual(h.copied,{})
        self.assertIn(t,h.pending)
        self.assertTrue(all(t in h.pool.page(k).readers for k in tables[layer]))
        h.cancel()
        with self.assertRaises(Pending):h.retire()

    def test_lost_old_lease_cannot_partially_pin_new_mapping(self):
        hs,pub=quartet(); [finish(h) for h in hs]; pub.commit(hs)
        h=hs[0]; layer=LAYERS[0]; old=h.tables[layer][0]
        new=h.pool.allocate(layer,20,4)
        tables={k:list(v) for k,v in h.tables.items()}; tables[layer][0]=new
        h.pool.page(old).holders.remove(h.request) # injected broken adapter
        with self.assertRaises(ContractError):h.reserve(8,tables)
        self.assertFalse(h.pool.page(new).holders); self.assertIsNone(h.end)

    def test_cancelled_peer_forbids_all_rank_commit(self):
        hs,pub=quartet(); [finish(h) for h in hs]; hs[-1].cancel()
        with self.assertRaises(ContractError):pub.commit(hs)
        self.assertEqual([h.published for h in hs],[0]*4)
        self.assertEqual(pub.records,[])

    def test_warm_prefix_first_fill_is_not_deferred_until_second_observation(self):
        h,_=fixture(committed=20,end=22,ids=[1,4,9,10,11,30])
        plans=h.plan(LAYERS[0])
        self.assertEqual(sum(p.bytes for p in plans),22*1024)
        for plan in plans:h.complete(h.copy(plan))
        h.complete(h.attend(LAYERS[0]))
        self.assertEqual(h.published,20)


if __name__=='__main__':unittest.main()
