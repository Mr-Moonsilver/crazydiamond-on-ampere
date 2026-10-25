"""CPU evidence only: actual refill control flow and fail-closed accounting."""
import copy
import sys
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
from s5_dma_window import WindowRefill,validate
from test_s5_resident import Resident
from s5_value import grade,monitor_lines


def evidence(steps=3):
    mode=dict(raw_prefill=[dict(step=s,submitted_s=s,observed_s=s+.9) for s in range(steps)],
              raw_decode=[dict(submitted_s=.2,observed_s=.3),dict(submitted_s=2.8,observed_s=2.9)])
    rows=[dict(step=s,layer=l,start_s=s+.01+l*.1,end_s=s+.05+l*.1,
               bytes=50*1024,chunks=13,gpu_copy_ms=2.) for s in range(steps) for l in range(6)]
    r=dict(mode='every_qsa_layer',layers=6,windows=1,chunk_cap_pages=4,max_outstanding=1,
           refills=rows,bytes=sum(x['bytes'] for x in rows),chunks=13*len(rows),gpu_copy_ms=2.*len(rows))
    r.update(validate(r,mode,50,1024));return r,mode


class DMAWindow(unittest.TestCase):
    def fixture(self):
        d=Resident().fixture()
        for r in d['ranks']:
            r['requested_split']=34;r['spec']=dict(layers=6,mix=['full_attention']*6)
            r['value_probe'].update(resident_kind='kv',resident_dma=True,dma_every_layer=True,source_trials=3)
            r['resident_ledger'].update(kind='kv',kv_source_bytes=50*1024,pages=50,page_bytes=1024)
            r['ple_prefetch_stats']=dict(calls=0,h2d_bytes=0,pack_seconds=0.)
            r['resident_substitution']=dict(passed=True,dma_layer_exact=dict(passed=True,checks=54,expected=54))
            dma,m=evidence();r['resident_dma']=dma
            r['modes']['overlap_resident']['raw_prefill']=m['raw_prefill']
            r['modes']['overlap_resident']['raw_decode']=[dict(submitted_s=.2+i*.26,observed_s=.3+i*.26) for i in range(11)]
            r['resident_dma'].update(validate(dma,r['modes']['overlap_resident'],50,1024))
            solo,sm=evidence();sm['raw_decode']=[];solo.update(validate(solo,sm,50,1024))
            r['resident_dma_solo']=solo;r['modes']['resident_prefill'].update(sm)
        return d

    def test_counts_every_layer_and_guard_and_reports_solo_cost(self):
        r,m=evidence();self.assertEqual(r['bytes'],18*50*1024)
        self.assertEqual(r['fully_inside_decode_chunks'],2)
        self.assertTrue(r['inside_decode_window'])
        d=self.fixture();g=grade(d);self.assertEqual(g['verdict'],'PASS')
        line='\n'.join(monitor_lines('dma-every-step34',g))
        for text in ('S5_DMA_REPEAT','bytes_per_chunk=307200','solo_bytes=921600','solo_prefill_vs_original_solo='):
            self.assertIn(text,line)

    def test_omitted_or_duplicate_or_untimed_dma_never_passes(self):
        for mutation in ('missing','duplicate','outside','wrong_bytes','wrong_sum','nan','order','step','empty'):
            r,m=evidence()
            if mutation=='missing':r['refills'].pop()
            elif mutation=='duplicate':r['refills'][-1]=dict(r['refills'][0])
            elif mutation=='outside':r['refills'][0]['end_s']=1.
            elif mutation=='wrong_bytes':r['refills'][0]['bytes']-=1
            elif mutation=='wrong_sum':r['bytes']-=1
            elif mutation=='nan':r['refills'][0]['gpu_copy_ms']=float('nan')
            elif mutation=='order':r['refills'][0]['end_s']=.2
            elif mutation=='step':m['raw_prefill'][0]['step']=1
            elif mutation=='empty':m['raw_prefill']=[];r['refills']=[]
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate(r,m,50,1024)
        r,m=evidence();m['raw_decode'][1]['submitted_s']=.8
        proof=validate(r,m,50,1024)
        self.assertGreater(proof['fully_inside_decode_refills'],0)
        self.assertFalse(proof['inside_decode_window'],'partial chunk cannot prove sustained inside-window refill')

    def test_missing_solo_or_layer_equality_or_refills_fail_closed(self):
        for mutation in ('solo','equality','shape','refills','false_claim'):
            d=self.fixture();r=d['ranks'][0]
            if mutation=='solo':del r['resident_dma_solo']
            elif mutation=='equality':r['resident_substitution']['dma_layer_exact']['checks']=53
            elif mutation=='shape':r['spec']['layers']=4
            elif mutation=='refills':r['resident_dma']['refills'].pop()
            else:r['resident_dma']['inside_decode_window']=False
            self.assertEqual(grade(d)['verdict'],'INCONCLUSIVE',mutation)
        d=self.fixture();del d['ranks'][0]['modes']
        d['ranks'][1]['resident_substitution']['dma_layer_exact']['passed']=False
        self.assertEqual(grade(d)['verdict'],'FAIL')

    def fake(self):
        events=[]
        class Tensor:
            def __init__(self,values,device='cpu'):
                self.values=list(values);self.shape=(len(values),1,1,2);self.device=NS(type=device)
            def is_pinned(self):return True
            def is_contiguous(self):return True
            def numel(self):return len(self.values)*2
            def element_size(self):return 2
            def __getitem__(self,index):
                ids=list(range(len(self.values)))[index]
                if isinstance(index,int):ids=[ids]
                parent=self
                class View(Tensor):
                    def copy_(v,source,non_blocking):
                        events.append(('copy',len(ids),source.device.type,non_blocking))
                        for i,x in zip(ids,source.values):parent.values[i]=x
                        return v
                return View([self.values[i] for i in ids],self.device.type)
        class Stream:
            def wait_stream(self,s):events.append(('wait-consumer',))
            def wait_event(self,e):events.append(('wait-copy',))
            def synchronize(self):events.append(('stream-fence',))
        class Event:
            def __init__(self,**kw):pass
            def record(self):events.append(('record',))
            def synchronize(self):events.append(('fence',))
            def elapsed_time(self,e):return 1.
        current=Stream()
        torch=NS(cuda=NS(Stream=Stream,Event=Event,current_stream=lambda:current,
                         stream=lambda s:nullcontext(),synchronize=lambda:events.append(('device-fence',))),
                 equal=lambda a,b:a==b,uint8='uint8')
        host=Tensor(range(10));resident=Tensor([-1]*10,'cuda');uva=Tensor(range(10),'cuda')
        pre=NS(spec=dict(mix=['full_attention']*6),host=host,kv=resident,pages=10)
        def qsa(x,i):
            events.append(('attend',i,pre.kv is resident))
            value=sum(pre.kv.values)+x+i
            return NS(view=lambda dtype:value)
        pre.qsa=qsa
        def forward(trace):
            return [pre.qsa(7,i) for i in range(6)]
        pre.forward=forward
        return torch,pre,resident,uva,events,forward,qsa

    def test_actual_wrapper_refills_six_layers_each_step_and_fences_consumer(self):
        torch,pre,resident,uva,events,forward,qsa=self.fake()
        with patch.dict(sys.modules,{'torch':torch}):
            w=WindowRefill(pre,uva,resident);w.verify_layers=True
            pre.forward(None);pre.forward(None)
            self.assertEqual(w.layer_checks,12)
            self.assertEqual(len(w.refills),12)
            self.assertEqual([(r['step'],r['layer']) for r in w.refills],[(s,l) for s in range(2) for l in range(6)])
            self.assertEqual(w.report()['bytes'],12*40)
            copies=[x for x in events if x[0]=='copy']
            self.assertEqual([x[1] for x in copies],[4,4,2]*12)
            self.assertTrue(all(x[2:] == ('cpu',True) for x in copies))
            self.assertEqual(sum(x[0]=='wait-consumer' for x in events),12)
            for i,x in enumerate(events):
                if x[0]=='copy':self.assertEqual(events[i+2],('fence',))
            n=len(copies);pre.kv=uva;pre.forward(None)
            self.assertEqual(sum(x[0]=='copy' for x in events),n,'host oracle must stay UVA')
            w.close();self.assertIs(pre.forward,forward);self.assertIs(pre.qsa,qsa)
            self.assertIsNone(w.pre);self.assertIsNone(w.end)

    def test_batched_control_queues_disjoint_pieces_then_one_terminal_fence(self):
        torch,pre,resident,uva,events,forward,qsa=self.fake()
        with patch.dict(sys.modules,{'torch':torch}):
            w=WindowRefill(pre,uva,resident,batched=True);w.verify_layers=True
            pre.forward(None);pre.forward(None)
            self.assertEqual(w.layer_checks,12)
            self.assertEqual(sum(e[0]=='copy' for e in events),36)
            self.assertEqual(sum(e[0]=='fence' for e in events),12)
            self.assertEqual(sum(e[0]=='wait-consumer' for e in events),12)
            # Nothing reads/overwrites the window before all pieces finish.
            for a,b in zip([i for i,e in enumerate(events) if e[0]=='wait-consumer'],
                           [i for i,e in enumerate(events) if e[0]=='wait-copy']):
                seq=events[a:b+1]
                self.assertEqual([e[1] for e in seq if e[0]=='copy'],[4,4,2])
                self.assertEqual([e[0] for e in seq][-3:],['record','fence','wait-copy'])
                self.assertFalse(any(e[0]=='attend' for e in seq))
            r=w.report();self.assertEqual(r['max_outstanding'],3)
            self.assertEqual(r['max_outstanding_bytes'],40)
            self.assertEqual(r['max_refills_in_flight'],1)
            self.assertEqual(r['timing_basis'],'whole_refill_stream_span')
            self.assertEqual(r['bytes'],480)
            w.close();self.assertIs(pre.forward,forward)

    def test_batched_evidence_bound_to_requested_policy_and_budget(self):
        d=self.fixture()
        for rank in d['ranks']:
            rank['value_probe']['dma_batched']=True
            for key in ('resident_dma','resident_dma_solo'):
                rank[key].update(batched=True,max_outstanding=13,max_outstanding_bytes=50*1024,
                                 max_refills_in_flight=1,timing_basis='whole_refill_stream_span')
        self.assertEqual(grade(d)['verdict'],'PASS')
        for key,value in [('max_outstanding',14),('max_outstanding_bytes',51*1024),
                          ('max_refills_in_flight',2),('timing_basis','sum_of_piece_stream_spans')]:
            bad=copy.deepcopy(d);bad['ranks'][0]['resident_dma'][key]=value
            self.assertEqual(grade(bad)['verdict'],'INCONCLUSIVE',key)
        bad=copy.deepcopy(d);bad['ranks'][0]['value_probe']['dma_batched']=False
        self.assertEqual(grade(bad)['verdict'],'INCONCLUSIVE')
        bad=copy.deepcopy(d);bad['ranks'][0]['resident_dma_solo']['batched']=False
        self.assertEqual(grade(bad)['verdict'],'INCONCLUSIVE')

    def test_layer_mismatch_is_hard_and_missing_pinned_source_rejected(self):
        torch,pre,resident,uva,events,forward,qsa=self.fake()
        class CorrectnessError(RuntimeError):pass
        with patch.dict(sys.modules,{'torch':torch,'s5_segment_gpu':NS(CorrectnessError=CorrectnessError)}):
            w=WindowRefill(pre,uva,resident);w.verify_layers=True;uva.values[0]+=1
            with self.assertRaises(CorrectnessError):pre.forward(None)
            self.assertIs(pre.kv,resident);w.close()
            pre.host.device.type='cuda'
            with self.assertRaises(ValueError):WindowRefill(pre,uva,resident)


if __name__=='__main__':unittest.main()
