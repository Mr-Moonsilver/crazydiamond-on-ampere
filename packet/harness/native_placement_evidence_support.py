"""CPU-only producer-shaped paired-window fixture. No CUDA/hardware evidence."""
import copy
import time
from collections import Counter
from unittest.mock import patch
import native_prefill_support as s
import test_e3_engine_offers as t
import engine_offers as eo
from native_prefill_offers import partition_view
from test_e3_native_report_read import Export, session


def packet(rank=0):
    b=s.FakeBridge(start_ns=time.monotonic_ns(),route_generation=7,epoch=3,generation=7,rank=rank)
    torch=s.make_torch();p=s.make_ledger(b,torch_module=torch,window_id="win-1",rank=rank,worker="w%d"%rank);s.declare(p)
    parts=s.build_lane(b,torch_module=torch);shadows=s.install_lane_seams(parts,p)
    events=s.start_role_owners(p,torch);s.submit_all(parts['dispatcher'])
    s.finish_role_owners(p,events=events[:1]);p.owner_finish('lane',events=events[:1])
    for shadow in reversed(shadows):shadow.close()
    ident={'main':{'context_id':40,'stream_id':61},'pp':{'context_id':40,'stream_id':62},'output_copy':{'context_id':40,'stream_id':62}}
    with patch.object(t,'GREEN',40),patch.object(t,'IDENT',ident),patch.object(t,'START',b.start_ns):
        binding=t.make_binding(engine_incarnation='engine:3',native_end_ns=None,router_census=None,rank=rank,worker='w%d'%rank,worker_incarnation='worker-inc-%d'%rank)
        owner=t.Owner();d=eo.EngineOfferLedger(b,worker='w%d'%rank,rank=rank,epoch=3,generation=7,route_generation=7,green_context_id=40,identities=ident,pp_stage=0,binding=binding,env={eo.ENABLE_ENV:'1'});d.activate(owner)
        t.open_positive_window(owner,d,include_eager=False)
    offer=[v for v in b.offers if d._meta.get(v['id'],{}).get('suboperation')=='graph_replay'][0]
    cid=b.activity('kernel',stream=s.DECODE,count=2,graph_id=100,offer=offer['id'],name='cudaGraphLaunch')
    row=copy.deepcopy(b.activities[-1]);row.update(kind='memcpy',copy_kind=3,count=1);b.activities.append(row)
    s.run_decode_offers(d,b)
    b.close_window();d.deactivate();d.finish_window({'total':1,'graphs':{'graph:0':1}})
    p.finish();pre=p.seal();whole=p.snapshot();decl=dict(d._meta)
    view=partition_view(whole['native'],whole['offers'],set(decl),allow_shared=True)
    dec=eo.grade_engine(view,[r for r in whole['offers'] if r['id'] in decl],decl,d.window())
    counts=Counter()
    for row in whole['native']['activities']:counts[row['kind']]+=row['count']
    expected=sum(row['count'] for row in whole['native']['activities'] if row['graph_id']==0 and row['context_id']==40)
    other=sum(row['count'] for row in whole['native']['activities'] if row['graph_id']==0 and row['context_id']!=40)
    with patch.object(t,'START',b.start_ns),patch.object(t,'END',b.end_ns):
     raw=t.raw_receipt(dict(engine_by_kind=dict(counts),joined_records=sum(counts.values())),expected=expected,other=other,verdict='INCONCLUSIVE',native_reasons=['unattributed_expected_context'])
    raw['caller_proof'].update(engine_incarnation='engine:3',worker_rank=rank,worker_incarnation='worker-inc-%d'%rank)
    
    packet = dict(receipt=raw, native_report=whole['native'],
        offer_report=whole['offers'], decode_declared=decl,
        decode_window=d.window(), engine_accounting=dec,
        native_prefill_accounting=pre, route='split', route_generation=7,
        rank=rank, engine_epoch=3, observer_stopped=True, state='closed')
    reader = session()
    reader._report_call(Export(whole['native']), (), max_bytes=64 * 1024**2)
    packet['native_report_read'] = reader.native_read_metadata()
    return packet
