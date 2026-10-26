"""CPU contracts for the probe-only sampler trace seam; no GPU claim."""
import asyncio
import copy
import hashlib
from pathlib import Path
import sys
import types
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import teacher_decode as td
import probe_control_hooks

MEASURED=list(range(td.ENTRY_STEPS,td.ENTRY_STEPS+td.MEASURED_STEPS))
PROMPTS={slot:100+slot for slot in range(8)}
COHORT_TRACES=[[20+stream]*64 for stream in range(8)]
COHORT=[td.digest(trace) for trace in COHORT_TRACES]
IDENTITY={'rank':2,'engine_epoch':71,'route':'full','route_generation':0,'storage_epoch':1}

def plan():
    return {'schema':td.SCHEMA,'requests':[
        {'request_id':f'teacher-{i}','prompt_sha256':td.digest([i,11,12]),
         'continuation':[i+20]*td.STEPS} for i in range(8)]}
class Params(types.SimpleNamespace):
    def _validate_trace_replay(self,model,speculative):
        if speculative is not None or any(t>=model.get_vocab_size() for t in self.trace_decode_token_ids):
            raise ValueError('private payload must not escape')
def request(i=0):
    return types.SimpleNamespace(request_id=f'cmpl-teacher-{i}-0',prompt_token_ids=[i,11,12],
        sampling_params=Params(trace_decode_token_ids=None,n=1,max_tokens=td.STEPS,min_tokens=td.STEPS,
        temperature=0.,ignore_eos=True,logprobs=20,stop=[],stop_token_ids=[]),lora_request=None,mm_features=None)
class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.events=[]
        def original(req):
            self.events.append('validated')
            if req.sampling_params.trace_decode_token_ids:raise ValueError('stock trace-mode rejection')
            return req
        async def original_async(req):return original(req)
        self.p=types.SimpleNamespace(process_inputs=original,process_inputs_async=original_async,
            model_config=types.SimpleNamespace(get_vocab_size=lambda:100),speculative_config=None)
        self.owner=td.FrontendPlan(self.p)
    def test_private_injection_preserves_every_other_setting(self):
        self.owner.configure(plan())
        for i in range(8):
            r=request(i); before=copy.deepcopy(vars(r.sampling_params))
            self.assertIs(self.p.process_inputs(r),r)
            self.assertEqual(r.sampling_params.trace_decode_token_ids,[20+i]*td.STEPS)
            after=vars(r.sampling_params).copy();after['trace_decode_token_ids']=None
            self.assertEqual(before,after)
            r.request_id+='-randomized'
        self.assertEqual(self.owner.status()['admitted'],8)
        self.assertNotIn('continuation',str(self.owner.status()))
        self.owner.configure(None)
        self.assertIsNone(self.p.process_inputs(request()).sampling_params.trace_decode_token_ids)
    def test_async_bound_method_is_not_a_bypass(self):
        self.owner.configure(plan())
        r=asyncio.run(self.p.process_inputs_async(request()))
        self.assertEqual(r.sampling_params.trace_decode_token_ids,[20]*td.STEPS)
        self.assertEqual(self.events,['validated'])
    def test_default_off_and_public_trace_rejection_unchanged(self):
        r=request();self.assertIs(self.p.process_inputs(r),r)
        r.sampling_params.trace_decode_token_ids=[20]*td.STEPS
        with self.assertRaisesRegex(ValueError,'stock trace-mode'):self.p.process_inputs(r)
    def test_wrong_prompt_foreign_id_duplicate_and_reconfiguration_refused(self):
        self.owner.configure(plan())
        r=request();r.prompt_token_ids=[0,11,13]
        with self.assertRaises(td.TeacherDecodeError):self.p.process_inputs(r)
        r=request();r.request_id='foreign'
        with self.assertRaises(td.TeacherDecodeError):self.p.process_inputs(r)
        self.p.process_inputs(request())
        with self.assertRaises(td.TeacherDecodeError):self.p.process_inputs(request())
        with self.assertRaises(td.TeacherDecodeError):self.owner.configure(plan())
        with self.assertRaises(td.TeacherDecodeError):self.owner.configure(None)
        self.assertEqual(self.owner.status()['pending'],7)
    def test_bounded_typed_plan_and_no_settings_normalization(self):
        for field,value in [('continuation',[1]*(td.STEPS+1)),('continuation',[True]*td.STEPS),
                            ('request_id','invalid/name'),('prompt_sha256','bad')]:
            p=plan();p['requests'][0][field]=value
            with self.assertRaises(td.TeacherDecodeError):td.validate_plan(p)
        self.owner.configure(plan())
        for key,value in [('max_tokens',255),('temperature',.7),('ignore_eos',False),('logprobs',19),('stop',['stop'])]:
            r=request();setattr(r.sampling_params,key,value)
            with self.assertRaises(td.TeacherDecodeError):self.p.process_inputs(r)
        self.p.speculative_config=object();r=request()
        with self.assertRaisesRegex(td.TeacherDecodeError,'compatibility-refused'):self.p.process_inputs(r)
        self.assertIsNone(r.sampling_params.trace_decode_token_ids)

def worker_runner(**overrides):
    runner=types.SimpleNamespace(
        is_last_pp_rank=True,_flash_s5_decode_probe=object(),
        req_states=types.SimpleNamespace(max_num_reqs=1024),
        sample=lambda hidden_states,input_batch,grammar_output:'sampled',
        batch_sharder=types.SimpleNamespace(shard_sampler_inputs=lambda batch,grammar: (batch,None,None,None)),
        cudagraph_manager=types.SimpleNamespace(dispatch=lambda *args,**kwargs: None),
        sampler=types.SimpleNamespace(trace_replay_state=None,logprobs_mode='raw_logprobs'))
    runner.__dict__.update(overrides)
    return runner
class WorkerTests(unittest.TestCase):
    def test_temporary_sampler_state_restores_original_path(self):
        r=worker_runner()
        original=(r.sample,r.batch_sharder.shard_sampler_inputs,r.cudagraph_manager.dispatch)
        made=[]
        def factory(states):made.append(states);return types.SimpleNamespace()
        out=td.configure_worker(r,True,factory=factory)
        self.assertEqual(out['table_bytes'],1024*(td.STEPS+1)*4)
        self.assertEqual((out['row_schema'],out['row_basis']),(td.ROW_SCHEMA,td.ROW_BASIS))
        self.assertIsNot(r.sample,original[0])
        self.assertIsNot(r.batch_sharder.shard_sampler_inputs,original[1])
        self.assertIsNot(r.cudagraph_manager.dispatch,original[2])
        td.configure_worker(r,True,factory=factory);self.assertEqual(len(made),1)
        td.configure_worker(r,False,factory=factory);self.assertIsNone(r.sampler.trace_replay_state)
        self.assertEqual((r.sample,r.batch_sharder.shard_sampler_inputs,r.cudagraph_manager.dispatch),original)
        self.assertFalse(hasattr(r,'_flash_s5_teacher_observer'))
        r.sampler.trace_replay_state=object()
        with self.assertRaises(td.TeacherDecodeError):td.configure_worker(r,True,factory=factory)
    def test_observer_restores_class_methods_without_instance_shadows(self):
        class Sharder:
            def shard_sampler_inputs(self,batch,grammar):return (batch,None,None,None)
        class Manager:
            def dispatch(self,*args,**kwargs):return None
        class Runner:
            is_last_pp_rank=True
            def __init__(self):
                self.req_states=types.SimpleNamespace(max_num_reqs=8)
                self.sampler=types.SimpleNamespace(trace_replay_state=None,
                                                   logprobs_mode='raw_logprobs')
                self.batch_sharder=Sharder();self.cudagraph_manager=Manager()
                self._flash_s5_decode_probe=object()
            def sample(self,hidden_states,input_batch,grammar_output):return 'sampled'
        r=Runner()
        td.configure_worker(r,True,factory=lambda states: types.SimpleNamespace())
        self.assertIn('sample',vars(r))
        self.assertIn('shard_sampler_inputs',vars(r.batch_sharder))
        self.assertIn('dispatch',vars(r.cudagraph_manager))
        td.configure_worker(r,False)
        self.assertNotIn('sample',vars(r))
        self.assertNotIn('shard_sampler_inputs',vars(r.batch_sharder))
        self.assertNotIn('dispatch',vars(r.cudagraph_manager))
        self.assertEqual(r.sample(None,None,None),'sampled')
    def test_observer_drift_is_refused_instead_of_restored_blindly(self):
        r=worker_runner()
        td.configure_worker(r,True,factory=lambda states: types.SimpleNamespace())
        r.sample=lambda *args:'foreign'
        with self.assertRaisesRegex(td.TeacherDecodeError,'observer-drift'):
            td.configure_worker(r,False)
    def test_nested_sampler_invocation_is_a_capture_defect(self):
        r=worker_runner();seen=[]
        def sample(hidden_states,input_batch,grammar_output):
            if not seen:
                seen.append(1)
                r.sample(hidden_states,input_batch,grammar_output)
            return 'sampled'
        r.sample=sample
        td.configure_worker(r,True,factory=lambda states: types.SimpleNamespace(
            call_depth=0,capture_defects=set(),capture_overflow=False,
            call_global=None,call_local=None,call_graph=None,call_graph_stale=False,
            last_graph=None,dispatch_seq=0,consumed_dispatch_seq=-1))
        self.assertEqual(r.sample(None,None,None),'sampled')
        state=r._flash_s5_teacher_trace
        self.assertIn('nested-sampler-call',state.capture_defects)
        self.assertTrue(state.capture_overflow)
        td.configure_worker(r,False)
        self.assertIs(r.sample,sample)
    def test_pp0_does_not_allocate_and_processed_logprobs_refused(self):
        def forbidden(states):raise AssertionError('allocation')
        r=worker_runner(is_last_pp_rank=False,sampler=None)
        out=td.configure_worker(r,True,factory=forbidden)
        self.assertFalse(out['sampler_owner'])
        self.assertIsNone(out['row_schema'])
        r=worker_runner()
        r.sampler=types.SimpleNamespace(trace_replay_state=None,logprobs_mode='processed_logprobs')
        with self.assertRaises(td.TeacherDecodeError):td.configure_worker(r,True,factory=forbidden)
    def test_pinned_sampler_order_and_request_sources(self):
        ref=ROOT/'reference/vllm-src'
        for path,sha in probe_control_hooks.PINS.items():
            self.assertEqual(hashlib.sha256((ref/path).read_bytes()).hexdigest(),sha,path)
        s=(ref/'v1/worker/gpu/sample/sampler.py').read_text()
        self.assertLess(s.index('self.trace_replay_state.apply_trace(sampled, idx_mapping)'),
                        s.index('logprobs_tensors = compute_topk_scores('))
        s=(ref/'v1/worker/gpu/sample/trace_replay.py').read_text()
        self.assertIn('total_len - prompt_len',s)
        self.assertIn('trace_token_ids.stride(0)',s)


class Positions:
    """Stand-in for the pinned positions buffer: positions[logits_indices]."""
    def __init__(self,values):self.values=list(values)
    def __getitem__(self,index):return [self.values[i] for i in index]

def batch_fixture(*,slots,prompts,position,logits_indices=None,cu=None,
                  positions_slots=None,padded_tokens=None,padded_reqs=None,
                  num_tokens=None,draft=0):
    n=len(slots)
    logits_indices=list(range(n)) if logits_indices is None else list(logits_indices)
    cu=list(range(n+1)) if cu is None else list(cu)
    table_slots=list(slots) if positions_slots is None else list(positions_slots)
    positions=Positions([prompts[slot]+position-1 for slot in table_slots])
    return types.SimpleNamespace(
        idx_mapping=list(slots),expanded_idx_mapping=list(slots),
        logits_indices=logits_indices,cu_num_logits=cu,
        expanded_local_pos=[0]*n,num_scheduled_tokens=[1]*n,is_prefilling_np=[False]*n,
        num_reqs=n,num_reqs_after_padding=n if padded_reqs is None else padded_reqs,
        num_tokens=n if num_tokens is None else num_tokens,
        num_tokens_after_padding=n if padded_tokens is None else padded_tokens,
        num_draft_tokens=draft,positions=positions)

class RowCaptureTests(unittest.TestCase):
    """capture_call is the sampler-invocation seam; exercise it without torch."""
    def capture(self,rank=2,*,slots=(0,1,2,3,4,5,6,7),position=300,prompts=None,
                traces=None,graph=True,graph_pad=None,pad=0,seq=1,calls_used=0,
                rows_used=0,identity=None,local_logits=None,local_cu=None,
                global_logits=None,graph_stale=False):
        prompts=dict(PROMPTS) if prompts is None else dict(prompts)
        traces=(dict((slot,(COHORT[slot],tuple(COHORT_TRACES[slot])))
                     for slot in range(8)) if traces is None else traces)
        global_slots=list(slots);n=len(global_slots)
        # apply_trace receives the sampler's OWN idx_mapping: in TP-sharded
        # mode that is the local sub-batch, not the global batch order.
        owned=[row for row,slot in enumerate(global_slots) if slot%2==rank%2]
        local_slots=[global_slots[row] for row in owned]
        local=batch_fixture(slots=local_slots,prompts=prompts,position=position,
                            logits_indices=owned if local_logits is None else local_logits,
                            cu=list(range(len(owned)+1)) if local_cu is None else local_cu,
                            positions_slots=global_slots,num_tokens=n,
                            padded_tokens=n+pad,padded_reqs=n+pad)
        glob=batch_fixture(slots=global_slots,prompts=prompts,position=position,
                           logits_indices=list(range(n)) if global_logits is None else global_logits,
                           padded_tokens=n+pad,padded_reqs=n+pad)
        graph_spec=None
        if graph:
            width=n+(pad if graph_pad is None else graph_pad)
            graph_spec={'cg_mode':'full','cg_mode_index':2,'num_tokens':width,'num_reqs':width,
                        'uniform_token_count':1,'max_query_len':1,'num_active_loras':0,
                        'num_ubatches':1}
        chosen=dict(IDENTITY,rank=rank) if identity is None else dict(identity)
        return td.capture_call(seq=seq,slots=local_slots,
                               total=[prompts[slot]+position for slot in local_slots],
                               prompt=[prompts[slot] for slot in local_slots],traces=traces,
                               local=local,global_batch=glob,graph=graph_spec,
                               graph_stale=graph_stale,identity=chosen,
                               calls_used=calls_used,rows_used=rows_used)
    def test_permuted_slot_and_logit_row_order_join_by_row(self):
        slots=[7,2,0,5,3,1,6,4]
        captured=self.capture(slots=slots,global_logits=[3,0,2,1,7,4,6,5])
        self.assertEqual(captured['defects'],set())
        call=captured['call']
        self.assertEqual(call['global_cohort_order'],[COHORT[slot] for slot in slots])
        self.assertEqual(call['global_snapshot']['logit_input_positions'],
                         [PROMPTS[slots[row]]+299 for row in (3,0,2,1,7,4,6,5)])
        # rank 2 owns even slots: global rows 1,2,6,7 in global batch order.
        self.assertEqual(call['local_snapshot']['idx_mapping'],[2,0,6,4])
        self.assertEqual(call['local_snapshot']['logits_indices'],[1,2,6,7])
        self.assertEqual(call['local_cohort_order'],[COHORT[slot] for slot in (2,0,6,4)])
        for row,item in enumerate(captured['rows']):
            slot=call['local_snapshot']['idx_mapping'][row]
            self.assertEqual(item['slot'],slot)
            self.assertEqual(item['trace_sha256'],COHORT[slot])
            self.assertEqual(item['logits_row'],row)
            self.assertEqual(item['input_position'],PROMPTS[slot]+299)
            self.assertEqual(item['total_len']-item['prompt_len'],300)
            self.assertEqual(captured['row_traces'][row],tuple(COHORT_TRACES[slot]))
    def test_logit_rows_are_not_assumed_to_follow_batch_rows(self):
        # Local selection order is arbitrary here: input_position must come
        # from positions[logits_indices[logits_row]], never from batch_row.
        slots=[7,2,0,5,3,1,6,4]
        local_logits=[2,0,3,1]
        captured=self.capture(slots=slots,local_logits=local_logits)
        self.assertEqual(captured['defects'],set())
        for row,item in enumerate(captured['rows']):
            self.assertEqual(item['logits_row'],row)
            self.assertEqual(item['input_position'],
                             PROMPTS[slots[local_logits[row]]]+299)
    def test_padding_is_recorded_and_descriptor_mismatch_is_a_defect(self):
        captured=self.capture(pad=8)
        self.assertEqual(captured['defects'],set())
        call=captured['call']
        self.assertEqual(call['global_snapshot']['num_tokens'],8)
        self.assertEqual(call['global_snapshot']['num_tokens_after_padding'],16)
        self.assertEqual(call['global_snapshot']['num_reqs_after_padding'],16)
        self.assertEqual(call['graph']['num_tokens'],16)
        self.assertTrue(call['descriptor_matches_batch'])
        self.assertEqual(len(captured['rows']),4)
        mismatch=self.capture(pad=8,graph_pad=0)
        self.assertIn('graph-descriptor-batch-mismatch',mismatch['defects'])
        self.assertFalse(mismatch['call']['descriptor_matches_batch'])
    def test_slot_reuse_and_retirement_do_not_alias_rows(self):
        traces={slot:(COHORT[slot],tuple(COHORT_TRACES[slot])) for slot in range(8)}
        first=self.capture(traces=traces,position=300)
        first_row=[row for row in first['rows'] if row['slot']==2][0]
        self.assertEqual(first_row['trace_sha256'],COHORT[2])
        replacement=[99,98,97]
        traces[2]=(td.digest(replacement),tuple(replacement))
        second=self.capture(traces=traces,position=301,seq=2)
        second_row=[row for row in second['rows'] if row['slot']==2][0]
        self.assertEqual(second_row['trace_sha256'],td.digest(replacement))
        self.assertNotEqual(second_row['trace_sha256'],first_row['trace_sha256'])
        self.assertEqual(second['call']['local_cohort_order'][1],td.digest(replacement))
        traces.pop(6)
        third=self.capture(traces=traces,position=302,seq=3)
        self.assertEqual([row['slot'] for row in third['rows']],[0,2,4])
        self.assertIsNone(third['call']['local_cohort_order'][3])
    def test_pp_ranks_own_disjoint_rows_of_one_logical_batch(self):
        two=self.capture(rank=2)
        three=self.capture(rank=3)
        self.assertEqual([row['slot'] for row in two['rows']],[0,2,4,6])
        self.assertEqual([row['slot'] for row in three['rows']],[1,3,5,7])
        self.assertEqual(two['call']['global_snapshot'],three['call']['global_snapshot'])
        self.assertNotEqual(two['call']['local_snapshot'],three['call']['local_snapshot'])
    def test_overflow_is_flagged_never_silently_truncated(self):
        calls=self.capture(calls_used=td.MAX_CALLS)
        self.assertTrue(calls['call_overflow'])
        self.assertNotIn('call',calls)
        rows=self.capture(rows_used=td.MAX_ROWS-3)
        self.assertTrue(rows['rows_overflow'])
        self.assertNotIn('rows',rows)
        prompts={slot:100+slot for slot in range(9)}
        wide=self.capture(slots=list(range(9)),prompts=prompts)
        self.assertTrue(wide['list_overflow'])
        payload=wide['call']['global_snapshot']['idx_mapping']
        self.assertEqual((payload['overflow'],payload['length']),(True,9))
        self.assertEqual(payload['sha256'],td.digest([str(slot) for slot in range(9)]))
    def test_missing_graph_or_identity_is_a_capture_defect(self):
        missing=self.capture(graph=False)
        self.assertIn('graph-descriptor-missing',missing['defects'])
        stale=self.capture(graph_stale=True)
        self.assertIn('graph-descriptor-stale',stale['defects'])
        no_rank=self.capture(identity={'engine_epoch':71,'route':'full',
                                       'route_generation':0,'storage_epoch':1})
        self.assertIn('rank-missing',no_rank['defects'])
        no_route=self.capture(identity={'rank':2,'engine_epoch':71,'route':'other',
                                        'route_generation':0,'storage_epoch':1})
        self.assertIn('route-missing',no_route['defects'])


def cohort_of(traces):
    return [td.digest(trace) for trace in traces]

def sampler_call_fixture(rank,seq,position,*,slots=(0,1,2,3,4,5,6,7),prompts=None,
                         route='full',generation=0,epoch=71,storage=1,graph_mode='full',
                         pad=0,traces=None,stream_of_slot=None,sharded=False,lag=None):
    """One rank's exact sampler metadata for one decode sampler call.

    sharded=False is the pinned default (no batch-sharded sampling): both
    sampler ranks own every row. lag[slot] admits a stream that many steps
    late, as full11's staggered first tokens show; only measured rows record.
    """
    prompts=dict(PROMPTS) if prompts is None else dict(prompts)
    traces=COHORT_TRACES if traces is None else traces
    digest_of={slot:td.digest(traces[slot]) for slot in range(len(traces))}
    stream_of_slot=({slot:slot for slot in range(len(traces))}
                    if stream_of_slot is None else dict(stream_of_slot))
    slots=list(slots);n=len(slots)
    lag={} if lag is None else dict(lag)
    at={slot:position-lag.get(slot,0) for slot in slots}
    glob=dict(idx_mapping=list(slots),expanded_idx_mapping=list(slots),
              logits_indices=list(range(n)),cu_num_logits=list(range(n+1)),
              expanded_local_pos=[0]*n,num_scheduled_tokens=[1]*n,
              is_prefilling_np=[False]*n,num_reqs=n,num_reqs_after_padding=n+pad,
              num_tokens=n,num_tokens_after_padding=n+pad,num_draft_tokens=0,
              logit_input_positions=[prompts[slot]+at[slot]-1 for slot in slots])
    owned=[row for row,slot in enumerate(slots) if not sharded or slot%2==rank%2]
    local_slots=[slots[row] for row in owned]
    local=dict(idx_mapping=local_slots,expanded_idx_mapping=list(local_slots),
               logits_indices=list(owned),cu_num_logits=list(range(len(owned)+1)),
               expanded_local_pos=[0]*len(owned),num_scheduled_tokens=[1]*n,
               is_prefilling_np=[False]*n,num_reqs=len(owned),
               num_reqs_after_padding=n+pad,num_tokens=n,num_tokens_after_padding=n+pad,
               num_draft_tokens=0,
               logit_input_positions=[prompts[slot]+at[slot]-1 for slot in local_slots])
    graph={'cg_mode':graph_mode,'cg_mode_index':{'none':0,'piecewise':1,'full':2}[graph_mode],
           'num_tokens':n+pad,'num_reqs':n+pad,'uniform_token_count':1,
           'max_query_len':1,'num_active_loras':0,'num_ubatches':1}
    call=dict(rank=rank,engine_epoch=epoch,route=route,route_generation=generation,
              storage_epoch=storage,sampler_call_seq=seq,graph=graph,
              descriptor_stale=False,descriptor_matches_batch=True,
              global_snapshot=glob,local_snapshot=local,
              global_cohort_order=[digest_of.get(slot) for slot in slots],
              local_cohort_order=[digest_of[slot] for slot in local_slots],
              rows=len(local_slots))
    records=[dict(rank=rank,stream=stream_of_slot[slot],position=at[slot],
                  sampler_call_seq=seq,trace_sha256=digest_of[slot],slot=slot,
                  batch_row=row,logits_row=row,prompt_len=prompts[slot],
                  total_len=prompts[slot]+at[slot],output_position=at[slot],
                  input_position=prompts[slot]+at[slot]-1,forced_token_exact=True)
             for row,slot in enumerate(local_slots) if at[slot] in MEASURED_SET]
    return call,records

MEASURED_SET=set(MEASURED)

def default_steps(extra=0):
    return [(seq,td.ENTRY_STEPS+seq-1) for seq in range(1,td.MEASURED_STEPS+1+extra)]

def swapped_steps(extra=0):
    """Sampler calls are numbered in order but one step delivers the next
    position and the following step the previous one: a one-position shift."""
    steps=default_steps(extra)
    (first_seq,first_position),(second_seq,second_position)=steps[4],steps[5]
    steps[4]=(first_seq,second_position)
    steps[5]=(second_seq,first_position)
    return steps

def row_mapping_fixture(name,route,*,generation,repeat,steps=None,
                        slots=(0,1,2,3,4,5,6,7),prompts=None,epoch=71,storage=1,
                        graph_mode='full',pad=0,traces=None,stream_of_slot=None,
                        sharded=False,lag=None):
    prompts=dict(PROMPTS) if prompts is None else dict(prompts)
    traces=COHORT_TRACES if traces is None else traces
    calls=[];records=[]
    extra=max((lag or {0:0}).values())
    for seq,position in (default_steps(extra) if steps is None else steps):
        for rank in (2,3):
            call,rows=sampler_call_fixture(rank,seq,position,slots=slots,prompts=prompts,
                route=route,generation=generation,epoch=epoch,storage=storage,
                graph_mode=graph_mode,pad=pad,traces=traces,stream_of_slot=stream_of_slot,
                sharded=sharded,lag=lag)
            if not rows:continue  # capture_call records no call without measured rows
            call['rows']=len(rows)
            calls.append(call)
            records.extend(dict(row,run=name,repeat=repeat) for row in rows)
    return dict(schema='s5-teacher-row-mapping/1',basis='sampler-invocation/1',
                calls=calls,records=records,defects=[],overflow=False,
                physical_layout_differs=False,rank_layout={})

def bind_numeric_fixture(e):
    """Fixture author explicitly supplies matching worker and SSE data."""
    from e3_teacher_decode_grade import logprob_digest
    for run in e['runs'].values():
        numeric={(r['stream'],r['position']):r for r in run['positions']}
        for r in numeric.values():
            r.setdefault('top_order', list(r['top']))
        for row in run['row_mapping']['records']:
            r=numeric[(row['stream'],row['position'])]
            row['logprob_sha256']=logprob_digest(row['trace_sha256'],row['position'],r['top'],r['top_order'])
    return e

_BASE_EVIDENCE=None
def _build_evidence():
    import e3_teacher_decode_grade as g
    prefixes=['a'*64]*8
    evidence=dict(schema=g.EVIDENCE_SCHEMA,rule=g.FROZEN_RULE,
                  baseline_sha256=g.BASELINE_SHA256,fixed_positions=list(MEASURED),
                  prefix_digests=prefixes,cohort_trace_digests=list(COHORT),
                  row_schema=g.ROW_SCHEMA,row_mapping_basis=g.ROW_BASIS,runs={})
    for repeat,name in enumerate(g.RUNS):
        route='split' if name=='split-B' else 'full'
        generation=1 if route=='split' else 0
        rows=[dict(stream=stream,position=position,worker_route=route,full_graph=True,
                   decode_only=True,forced_token_exact=True,route_generation=generation,
                   top={f'token_id:{token}':-token*.2 for token in range(20)})
              for stream in range(8) for position in MEASURED]
        evidence['runs'][name]=dict(
            route=route,lifecycle_verdict='PASS',prefix_digests=prefixes,positions=rows,
            repeat=repeat,engine_epoch=71,route_generation=generation,storage_epoch=1,
            row_mapping=row_mapping_fixture(name,route,generation=generation,repeat=repeat))
    return bind_numeric_fixture(evidence)

def base_evidence():
    global _BASE_EVIDENCE
    if _BASE_EVIDENCE is None:_BASE_EVIDENCE=_build_evidence()
    return copy.deepcopy(_BASE_EVIDENCE)


class ControllerCompositionTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_session_and_client_bind_plan_and_all_rank_ack(self):
        from test_e3_probe_session import CountedEngine
        from probe_control import ProbeController
        from probe_session import ProbeSession, receipt_core
        import e3_w1_decode_driver as driver
        class Engine(CountedEngine):
            async def collective_rpc(self,*a,**kw):
                c=kw['args'][0]
                if c['action']=='teacher_configure':
                    owner=c['enabled'] and c['engine_epoch']>0
                    return [dict(rank=r,ack='flash_s5_teacher_configure',engine_epoch=c['engine_epoch'],
                        enabled=c['enabled'],sampler_owner=r>=2,table_bytes=1028 if r>=2 else 0,
                        drained_streams=3,snapshot_max_bytes=c['snapshot_max_bytes'],
                        row_schema=td.ROW_SCHEMA if (owner and r>=2) else None,
                        row_basis=td.ROW_BASIS if (owner and r>=2) else None) for r in range(4)]
                return await super().collective_rpc(*a,**kw)
        engine=Engine()
        engine.input_processor=types.SimpleNamespace(process_inputs=lambda r:r)
        session=ProbeSession(ProbeController(engine,epoch=71))
        context=dict(run_id='teacher-fixture',repeat=0,cohort_id='cohort-0',streams=8,
                     command_seq=0,prior_receipt=None,engine_epoch=None,storage_epoch=None,
                     route='full',route_generation=None)
        command=driver._next_command(context,'prepare','full')
        command['teacher_forced']=plan()
        reply=await session.execute(command)
        check=dict(context,action='prepare',command_seq=command['command_seq'])
        ok,reasons,_,accepted=driver.validate_receipt(reply,command,check)
        self.assertTrue(ok,reasons)
        core=driver._receipt_core(reply)
        self.assertEqual(core,receipt_core(reply))
        self.assertEqual(core['teacher_forced']['plan_sha256'],td.digest(plan()))
        self.assertEqual(len(core['teacher_forced']['ranks']),4)
        self.assertNotIn('continuation',str(core))
        self.assertFalse(engine._flash_s5_admission_seal.sealed)

    async def test_incomplete_worker_ack_does_not_enable_admission(self):
        from test_e3_probe_session import CountedEngine
        from probe_control import ProbeController,ProbeControlError
        class Engine(CountedEngine):
            async def collective_rpc(self,*a,**kw):
                if kw['args'][0]['action']=='teacher_configure':return []
                return await super().collective_rpc(*a,**kw)
        engine=Engine();ctl=ProbeController(engine,epoch=71)
        with self.assertRaises(ProbeControlError):await ctl.configure_teacher(plan())
        self.assertTrue(ctl.ingress.sealed)
        self.assertFalse(hasattr(ctl,'_teacher_plan'))


class LogprobBindingTests(unittest.TestCase):
    def test_digest_matches_pinned_processor_and_completion_serialization(self):
        import ast, __future__, itertools
        from e3_teacher_decode_grade import logprob_digest
        pins={
          'logprobs.py':'3cc2ed6cdd30572d29f9b293ddbc14ad318c6b5d5dc7fdd36bee2f8af259b373',
          'entrypoints/openai/completion/serving.py':'75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527',
          'v1/engine/logprobs.py':'14bbe78d92a2c451d79b3d9dd9d5268bc57e16bad5079e44ce9ed2cd805e873e',
          'v1/outputs.py':'65ab7698bbdea1d53502ef8df3689bec860521ab6e2fab3c7bff63151be039b1',
          'v1/worker/gpu/sample/logprob.py':'f47479a818192acb6b297c79fa67a3b2f9820e3492a394d5adda777e9b7c832e'}
        for path,sha in pins.items():
            self.assertEqual(hashlib.sha256((ROOT/'reference/vllm-src'/path).read_bytes()).hexdigest(),sha)
        env={'itertools':itertools,'Logprob':types.SimpleNamespace,
             'FlatLogprobs':type('FlatLogprobs',(),{}),'CompletionLogProbs':types.SimpleNamespace}
        for path,name in [('logprobs.py','append_logprobs_for_next_position'),
                          ('entrypoints/openai/completion/serving.py','_create_completion_logprobs')]:
            tree=ast.parse((ROOT/'reference/vllm-src'/path).read_text())
            node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
            exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec',
                         flags=__future__.annotations.compiler_flag),env)
        service=types.SimpleNamespace(_get_decoded_token=lambda row,token,tokenizer,**kw:'token_id:'+str(token))
        for forced in (3,27):
            ids=[forced]+list(range(20));values=[-float(forced)]+[-float(i) for i in range(20)]
            values[-1]=float('-inf')
            rows=[]
            env['append_logprobs_for_next_position'](rows,ids,values,[None]*21,1,20)
            wire=env['_create_completion_logprobs'](service,[forced],rows,20,None,return_as_token_id=True)
            top=wire.top_logprobs[0]
            self.assertEqual(td.logprob_digest('a'*64,328,ids,values),
                             logprob_digest('a'*64,328,top,list(top)))

    def test_sampler_return_binding_reads_real_batch_rows_and_rejects_missing(self):
        class Tensor:
            def __init__(self,rows):self.rows=rows;self.shape=(len(rows),len(rows[0]))
            def cpu(self):return self
            def tolist(self):return self.rows
        ids=[list(range(21)),list(range(30,51))]
        values=[[-float(i) for i in range(21)],[-float(i)/2 for i in range(21)]]
        output=types.SimpleNamespace(logprob_token_ids=Tensor(ids),logprobs=Tensor(values))
        state=types.SimpleNamespace(positions=[dict(batch_row=1,trace_sha256='b'*64,position=328)],
                                    capture_defects=set(),capture_overflow=False)
        td.observe_logprobs(state,(types.SimpleNamespace(logprobs_tensors=output),None,None),0)
        self.assertEqual(state.positions[0]['logprob_sha256'],td.logprob_digest('b'*64,328,ids[1],values[1]))
        td.observe_logprobs(state,(types.SimpleNamespace(logprobs_tensors=None),None,None),0)
        self.assertTrue(state.capture_overflow)

class GradeTests(unittest.TestCase):
    def evidence(self):
        return base_evidence()

    def test_replicated_rank_coverage_and_rank_identity_are_required(self):
        from e3_teacher_decode_grade import grade
        for mutation in ('one_row', 'rank_identity'):
            e=self.deployment_evidence();mapping=e['runs']['full-A1']['row_mapping']
            if mutation=='one_row':
                mapping['records'].pop(0)
            else:
                for row in mapping['records']+mapping['calls']:
                    if row['rank']==3: row['rank']=7
            out=grade(e)
            self.assertEqual(out['verdict'],'INCONCLUSIVE')
            self.assertTrue(any('coverage incomplete' in m or 'identity invalid' in m for m in out['missing']))

    def test_call_gaps_from_partial_decode_batches_are_not_position_shifts(self):
        from e3_teacher_decode_grade import grade
        e=self.deployment_evidence(steps=[(seq*2,pos) for seq,pos in default_steps(max(self.LAG.values()))])
        self.assertEqual(grade(e)['verdict'],'PASS')

    def test_legacy_and_sharded_evidence_cannot_qualify_full12(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence();e['schema']='s5-teacher-decode/1'
        self.assertEqual(grade(e)['verdict'],'INCONCLUSIVE')
        e=self.evidence()
        e['runs']['full-A1']['row_mapping']=row_mapping_fixture('full-A1','full',generation=0,repeat=0,sharded=True)
        self.assertEqual(grade(e)['verdict'],'INCONCLUSIVE')

    def test_physical_forced_token_proof_cannot_disagree_with_numeric_row(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence();e['runs']['full-A1']['row_mapping']['records'][0]['forced_token_exact']=False
        out=grade(e)
        self.assertEqual(out['verdict'],'INCONCLUSIVE')
        self.assertTrue(any('forced-token proof differs' in m for m in out['missing']))

    def test_sse_topk_shift_fails_binding_even_when_forced_tokens_match(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence()
        row=e['runs']['full-A1']['positions'][0]
        row['top']=dict(row['top']);row['top']['token_id:0']=-.01
        out=grade(e)
        self.assertEqual(out['verdict'],'INCONCLUSIVE')
        self.assertTrue(any('sampler/SSE logprob binding differs' in m for m in out['missing']))
        self.assertEqual(out['comparisons'],{})

    def test_ordered_ties_survive_sorted_json_and_bad_order_refused(self):
        import json
        from e3_teacher_decode_grade import grade, RUNS
        e=self.evidence();e['top_order_basis']='response-insertion-order/1'
        for name in RUNS:
            for row in e['runs'][name]['positions']:
                keys=list(row['top']);row['top'][keys[0]]=row['top'][keys[1]]=0.
                row['top_order']=keys if name=='full-A1' else list(reversed(keys))
        bind_numeric_fixture(e)
        before=grade(e);after=grade(json.loads(json.dumps(e,sort_keys=True)))
        self.assertEqual(before,after)
        self.assertTrue(before['comparisons']['control_pre']['flips'])
        for bad in (None, [], ['token_id:1']*20):
            broken=copy.deepcopy(e);broken['runs']['split-B']['positions'][0]['top_order']=bad
            self.assertEqual(grade(broken)['verdict'],'INCONCLUSIVE')
            self.assertIn('logprob order missing for row binding',grade(broken)['missing'])

    def test_all_positions_stable_then_split_out_of_rule_and_control_noise(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence();self.assertEqual(grade(e)['verdict'],'PASS')
        e['runs']['split-B']['positions'][0]['top']['token_id:0']=-1.
        bind_numeric_fixture(e)
        self.assertEqual(grade(e)['verdict'],'FAIL')
        e['runs']['full-A2']['positions'][0]['top']['token_id:0']=-1.
        bind_numeric_fixture(e)
        r=grade(e);self.assertEqual(r['verdict'],'INCONCLUSIVE')
        self.assertFalse(r['comparisons']['split']['comparable'])
    def test_missing_prefill_or_false_route_cannot_supply_decode_coverage(self):
        from e3_teacher_decode_grade import grade
        for key,value in [('decode_only',False),('worker_route','full'),('forced_token_exact',None)]:
            e=self.evidence();e['runs']['split-B']['positions'][0][key]=value
            self.assertEqual(grade(e)['verdict'],'INCONCLUSIVE')
        e=self.evidence();e['runs']['split-B']['positions'].pop()
        self.assertEqual(grade(e)['verdict'],'INCONCLUSIVE')
    def test_explicit_forced_token_mismatch_is_failure(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence();e['runs']['split-B']['positions'][0]['forced_token_exact']=False
        for row in e['runs']['split-B']['row_mapping']['records']:
            if (row['stream'], row['position']) == (0, MEASURED[0]): row['forced_token_exact']=False
        self.assertEqual(grade(e)['verdict'],'FAIL')

    def test_rule_is_frozen_and_near_tie_flips_only(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence();e['runs']['split-B']['positions'][0]['top']['token_id:1']=.01
        bind_numeric_fixture(e)
        self.assertEqual(grade(e)['verdict'],'PASS')
        for name in e['runs']:
            e['runs'][name]['positions'][0]['top']['token_id:1']=-.4
        e['runs']['split-B']['positions'][0]['top']['token_id:0']=-.21
        e['runs']['split-B']['positions'][0]['top']['token_id:1']=-.19
        bind_numeric_fixture(e)
        self.assertEqual(grade(e)['verdict'],'FAIL')
        e=self.evidence();e['rule']={**e['rule'],'TOL_WIN':1.}
        self.assertEqual(grade(e)['verdict'],'INCONCLUSIVE')

    def test_row_mapping_proves_every_coordinate_and_reports_pp_layout(self):
        from e3_teacher_decode_grade import grade
        report=grade(self.evidence())
        self.assertEqual(report['verdict'],'PASS')
        mapping=report['row_mapping']
        self.assertEqual(mapping['defect_count'],0)
        self.assertEqual(mapping['records'],2*8*td.MEASURED_STEPS*len(report['row_mapping']['ranks']))
        self.assertFalse(mapping['physical_layout_differs'])
        for name in report['row_mapping']['ranks']:
            self.assertEqual(report['row_mapping']['ranks'][name],[2,3])

    def test_permuted_slot_order_and_padding_still_join(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence()
        e['runs']['full-A1']['row_mapping']=row_mapping_fixture(
            'full-A1','full',generation=0,repeat=0,slots=[7,2,0,5,3,1,6,4],pad=8)
        bind_numeric_fixture(e)
        self.assertEqual(grade(e)['verdict'],'PASS')

    def test_one_position_shift_is_a_collector_defect(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence()
        e['runs']['split-B']['row_mapping']=row_mapping_fixture(
            'split-B','split',generation=1,repeat=2,steps=swapped_steps())
        bind_numeric_fixture(e)
        verdict=grade(e)
        self.assertEqual(verdict['verdict'],'INCONCLUSIVE')
        self.assertTrue(any('one-position shift' in reason for reason in verdict['missing']))

    def test_foreign_row_is_a_collector_defect(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence()
        e['runs']['full-C']['row_mapping']['records'][0]['trace_sha256']='f'*64
        verdict=grade(e)
        self.assertEqual(verdict['verdict'],'INCONCLUSIVE')
        self.assertTrue(any('foreign row' in reason for reason in verdict['missing']))

    def test_descriptor_padding_mismatch_and_overflow_are_collector_defects(self):
        from e3_teacher_decode_grade import grade
        e=self.evidence()
        call=e['runs']['full-A2']['row_mapping']['calls'][0]
        call['graph']=dict(call['graph'],num_tokens=call['graph']['num_tokens']+1)
        verdict=grade(e)
        self.assertEqual(verdict['verdict'],'INCONCLUSIVE')
        self.assertTrue(any('descriptor/batch padding differs' in reason
                            for reason in verdict['missing']))
        e=self.evidence();e['runs']['full-A2']['row_mapping']['overflow']=True
        verdict=grade(e)
        self.assertEqual(verdict['verdict'],'INCONCLUSIVE')
        self.assertTrue(any('row mapping overflow' in reason for reason in verdict['missing']))

    # Review fixes (q17b Opus pass). Deployment shape per full11 receipts:
    # ranks 2 and 3 each carried all 2048 rows (no batch-sharded sampling) and
    # first tokens arrived 990-1829 ms apart (streams admitted in different steps).
    LAG={0:0,1:0,2:1,3:1,4:3,5:5,6:7,7:9}
    def deployment_evidence(self,steps=None):
        import e3_teacher_decode_grade as g
        e=self.evidence()
        for repeat,name in enumerate(g.RUNS):
            route='split' if name=='split-B' else 'full'
            e['runs'][name]['row_mapping']=row_mapping_fixture(
                name,route,generation=1 if route=='split' else 0,repeat=repeat,
                sharded=False,lag=self.LAG,steps=steps)
        return bind_numeric_fixture(e)

    def test_pinned_input_position_is_the_fed_token(self):
        ref=ROOT/'reference/vllm-src/v1/worker/gpu'
        # positions[] = num_computed_tokens; post_update grows total_len after
        # sampling, so a decode sampler call feeds the token at total_len-1.
        self.assertIn('pos = num_computed_tokens + block',(ref/'input_batch.py').read_text())
        self.assertIn('tl.store(total_len_ptr + req_state_idx, total_len + num_sampled)',
                      (ref/'input_batch.py').read_text())
        self.assertIn('post_update runs after sampling',
                      (ref/'sample/trace_replay.py').read_text())
        from e3_teacher_decode_grade import grade
        e=self.evidence()
        mapping=e['runs']['full-A2']['row_mapping']
        record=mapping['records'][0]
        call=[c for c in mapping['calls'] if (c['rank'],c['sampler_call_seq'])
              ==(record['rank'],record['sampler_call_seq'])][0]
        record['input_position']=record['total_len']
        call['local_snapshot']['logit_input_positions'][record['logits_row']]=record['total_len']
        call['global_snapshot']['logit_input_positions'][record['logits_row']]=record['total_len']
        verdict=grade(e)
        self.assertEqual(verdict['verdict'],'INCONCLUSIVE')
        self.assertTrue(any('derived position differs' in reason for reason in verdict['missing']))

    def test_unsharded_staggered_deployment_shape_joins_and_passes(self):
        from e3_teacher_decode_grade import grade
        report=grade(self.deployment_evidence())
        self.assertEqual(report['verdict'],'PASS',report['missing'])
        self.assertEqual(report['row_mapping']['records'],4*2*8*td.MEASURED_STEPS)
        self.assertFalse(report['row_mapping']['physical_layout_differs'])

    def test_shift_still_detected_under_staggered_admission(self):
        from e3_teacher_decode_grade import grade
        verdict=grade(self.deployment_evidence(steps=swapped_steps(max(self.LAG.values()))))
        self.assertEqual(verdict['verdict'],'INCONCLUSIVE')
        self.assertTrue(any('one-position shift' in reason for reason in verdict['missing']))

    def test_pp_rank_disagreement_gap_and_duplicate_are_collector_defects(self):
        from e3_teacher_decode_grade import grade
        def run_mapping(e):return e['runs']['full-A1']['row_mapping']
        def check(e,reason):
            verdict=grade(e)
            self.assertEqual(verdict['verdict'],'INCONCLUSIVE')
            self.assertTrue(any(reason in item for item in verdict['missing']),verdict['missing'])
        e=self.deployment_evidence()
        rank3=[r for r in run_mapping(e)['records'] if r['rank']==3]
        rank3[0]['forced_token_exact']=None
        check(e,'forced-token proof differs from physical row')
        e=self.deployment_evidence()
        call=[c for c in run_mapping(e)['calls'] if c['rank']==3][10]
        call['graph']['cg_mode_index']=9
        check(e,'pp ranks disagree on the logical row')
        e=self.deployment_evidence();mapping=run_mapping(e)
        gone=[c for c in mapping['calls'] if c['rank']==3][10]
        mapping['calls'].remove(gone)
        mapping['records']=[r for r in mapping['records'] if (r['rank'],r['sampler_call_seq'])
                            !=(3,gone['sampler_call_seq'])]
        check(e,'pp rank sampler-call gap')
        e=self.deployment_evidence();mapping=run_mapping(e)
        mapping['records'][6]=dict(mapping['records'][5])  # count stays at the bound
        check(e,'duplicate physical coordinate')


class ClientTests(unittest.TestCase):
    def test_real_instrument_cohort_survives_shared_result_assembly(self):
        from dataclasses import replace
        from unittest import mock
        import e3_decode_correctness as dc
        import e3_teacher_decode_client as client
        calls=[]
        def post(url,path,body):
            calls.append(path)
            return {'tokens':[len(calls)]*900}
        instrument=client.Instrument(ROOT/'harness/baselines/q15-teacher-forced-rule3.json',post=post)
        cfg=dc.config_from_env({'RUN_ID':'teacher-assembly-fixture','CONTROLLER_CMD':'false',
            'WORKLOAD':str(ROOT/'harness/workloads/q17b-teacher-decode-v3.json')},
            Path('unused-teacher-artifact.json'))
        cfg=replace(cfg,instrument=instrument)
        def measure(cfg,controller,ctx,index,label,route,cohort):
            return dc._not_run_record(index,label,route,'offline-no-engine')
        # Exercise the actual cohort, teacher finish, and shared artifact
        # assembly. Only the network/controller measurement is replaced.
        with mock.patch.object(dc,'measure_run',side_effect=measure):
            result=dc.run(cfg,'http://unused','fixture',object(),source={'verified':True})
        self.assertEqual(len(calls),16)
        self.assertEqual([r['stream'] for r in result['workload']['prompts']],list(range(8)))
        self.assertEqual(result['verdict'],'INCONCLUSIVE')
        self.assertIn('teacher_forced',result)
        self.assertTrue(all(r['not_run'] for r in result['runs']))
        bad=instrument.cohort('http://unused','fixture')
        bad[0].pop('stream')
        with mock.patch.object(instrument,'cohort',return_value=bad), \
             mock.patch.object(dc,'measure_run') as measure:
            with self.assertRaisesRegex(dc.ConfigError,'cohort artifact identities incomplete'):
                dc.run(cfg,'http://unused','fixture',object(),source={'verified':True})
        measure.assert_not_called()

    def instrument(self):
        import e3_teacher_decode_client as client
        i=client.Instrument.__new__(client.Instrument);i.snapshot_max_bytes=2*1024**3;i.plan_shas={}
        i.baseline={'sha256':__import__('e3_teacher_decode_grade').BASELINE_SHA256}
        i.prompts=[[n,11,12] for n in range(8)]
        i.traces=[[n+20]*client.TOTAL for n in range(8)]
        i.plan=None
        return i

    def fixture_runs(self,instrument,steps=None,sharded=False,lag=None):
        """Raw four-rank worker evidence shaped as the pause_end receipt returns it."""
        import e3_teacher_decode_client as client
        top={f'token_id:{token}':-token/10 for token in range(20)}
        runs=[]
        for repeat,name in enumerate(('full-A1','full-A2','split-B','full-C')):
            route='split' if name=='split-B' else 'full'
            generation=1 if route=='split' else 0
            mapping=row_mapping_fixture(name,route,generation=generation,repeat=repeat,
                                        steps=steps,traces=instrument.traces,
                                        sharded=sharded,lag=lag)
            from e3_teacher_decode_grade import logprob_digest
            for row in mapping['records']:
                row['logprob_sha256']=logprob_digest(row['trace_sha256'], row['position'], top, list(top))
            workers=[]
            for rank in range(4):
                owned_calls=[call for call in mapping['calls'] if call['rank']==rank]
                owned_rows=[{key:value for key,value in row.items()
                             if key not in ('rank','run','repeat','stream','output_position')}
                            for row in mapping['records'] if row['rank']==rank]
                workers.append(dict(
                    rank=rank,sampler_owner=rank>=2,engine_epoch=71,
                    route_generation=generation,route=route,storage_epoch=1,
                    overflow=False,call_overflow=False,capture_defects=[],drained_streams=3,
                    row_schema=client.ROW_SCHEMA if rank>=2 else None,
                    row_basis=client.ROW_BASIS if rank>=2 else None,
                    calls=owned_calls if rank>=2 else [],
                    positions=owned_rows if rank>=2 else []))
            streams=[dict(stream=stream,_tokens=trace,
                          teacher_top=[dict(position=position,top=top,top_order=list(top))
                                       for position in MEASURED])
                     for stream,trace in enumerate(instrument.traces)]
            instrument.plan_shas[repeat]='a'*64
            proof=dict(pending=0,admitted=8,plan_sha256='a'*64,ranks=workers)
            core=dict(repeat=repeat,engine_epoch=71,route_generation_after=generation,
                      teacher_forced=proof)
            runs.append(dict(label=name,route=route,complete=True,contract=False,
                streams=streams,
                calls=[dict(action='pause_end',lifecycle_ok=True,
                            receipt={'teacher_forced':proof},receipt_core=core,
                            receipt_sha256=client.digest(core))]))
        return runs

    @staticmethod
    def rebind(run):
        import e3_teacher_decode_client as client
        call=run['calls'][0]
        call['receipt_sha256']=client.digest(call['receipt_core'])

    def test_api_original_logprobs_are_bound_to_fixed_token_positions(self):
        import e3_teacher_decode_client as client
        i=self.instrument();state=types.SimpleNamespace(index=0,token_count=client.ENTRY,teacher_top=[])
        top={f'token_id:{n}':-n/10 for n in range(20)}
        event={'choices':[{'token_ids':[20], 'logprobs':{'tokens':['token_id:20'],'top_logprobs':[top]}}]}
        i.consume(state,event)
        self.assertEqual(state.teacher_top,[{'position':client.ENTRY,'top':top,'top_order':list(top)}])
        event['choices'][0]['token_ids']=[21]
        with self.assertRaises(ValueError):i.consume(state,event)
    def test_requests_bind_private_plan_without_changing_sampling(self):
        import e3_teacher_decode_client as client
        import e3_decode_correctness as dc
        i=self.instrument()
        workload=types.SimpleNamespace(generation_budget=client.TOTAL,temperature=0.,top_p=1.,sampling_seed=909)
        bodies=[dc.request_body('model',p,workload,'salt') for p in i.prompts]
        before=[{k:v for k,v in b.items()} for b in bodies]
        i.requests(bodies,[{'prompt':p} for p in i.prompts],3)
        self.assertEqual(len(td.validate_plan(i.plan)),8)
        for a,b in zip(before,bodies):
            self.assertEqual(a,{k:v for k,v in b.items() if k not in ('request_id','logprobs','return_tokens_as_token_ids')})
    def test_fixed_worker_proof_join_then_missing_position_cannot_pass(self):
        i=self.instrument();runs=self.fixture_runs(i)
        self.assertEqual(i.finish(runs)['grade']['verdict'],'PASS')
        proof=runs[2]['calls'][0]['receipt']['teacher_forced']
        proof['ranks'][2]['positions'].pop()
        self.rebind(runs[2])
        self.assertEqual(i.finish(runs)['grade']['verdict'],'INCONCLUSIVE')

    def test_foreign_row_and_overflow_in_worker_evidence_are_inconclusive(self):
        i=self.instrument()
        runs=self.fixture_runs(i)
        runs[0]['calls'][0]['receipt']['teacher_forced']['ranks'][3]['positions'][0]['trace_sha256']='f'*64
        self.rebind(runs[0])
        verdict=i.finish(runs)['grade']
        self.assertEqual(verdict['verdict'],'INCONCLUSIVE')
        self.assertTrue(any('row mapping collector defect' in reason for reason in verdict['missing']))
        runs=self.fixture_runs(i)
        runs[1]['calls'][0]['receipt']['teacher_forced']['ranks'][2]['overflow']=True
        self.rebind(runs[1])
        self.assertEqual(i.finish(runs)['grade']['verdict'],'INCONCLUSIVE')

    def test_one_position_shift_in_worker_calls_is_inconclusive(self):
        i=self.instrument();runs=self.fixture_runs(i,steps=swapped_steps())
        verdict=i.finish(runs)['grade']
        self.assertEqual(verdict['verdict'],'INCONCLUSIVE')
        self.assertTrue(any('one-position shift' in reason for reason in verdict['missing']))

    def test_unsharded_staggered_worker_evidence_passes_and_input_position_is_bound(self):
        i=self.instrument()
        runs=self.fixture_runs(i,sharded=False,lag=GradeTests.LAG)
        verdict=i.finish(runs)['grade']
        self.assertEqual(verdict['verdict'],'PASS',verdict['missing'])
        row=runs[3]['calls'][0]['receipt']['teacher_forced']['ranks'][2]['positions'][7]
        row['input_position']=row['total_len']
        self.rebind(runs[3])
        verdict=i.finish(runs)['grade']
        self.assertEqual(verdict['verdict'],'INCONCLUSIVE')
        self.assertTrue(any('input position differs' in reason for reason in verdict['missing']))

    def test_full_replicated_pause_proof_fits_and_payloads_are_refused(self):
        import json
        import e3_w1_decode_driver as driver
        i=self.instrument();run=self.fixture_runs(i,lag=GradeTests.LAG)[0]
        receipt=run['calls'][0]['receipt']
        proof=receipt['teacher_forced']
        self.assertEqual(len(proof['ranks'][2]['positions']),2048)
        self.assertLess(len(json.dumps(proof).encode()),4*1024*1024)
        self.assertEqual(driver._sanitize_receipt(receipt)['teacher_forced'],proof)
        proof['continuation']=[123]
        self.assertIn('error',driver._sanitize_receipt(receipt)['teacher_forced'])

    def test_w1_receipt_sanitizer_keeps_captured_sampler_proof(self):
        import e3_w1_decode_driver as driver
        captured=RowCaptureTests().capture()
        rank=dict(rank=2,sampler_owner=True,row_schema=td.ROW_SCHEMA,row_basis=td.ROW_BASIS,
                  calls=[captured['call']],positions=captured['rows'],overflow=False,
                  call_overflow=False,list_overflow=False,capture_defects=[],basis='b',
                  ack='flash_s5_teacher_evidence',engine_epoch=71,storage_epoch=1,
                  route_generation=0,route='full',drained_streams=3)
        proof=dict(schema=td.SCHEMA,enabled=True,plan_sha256='a'*64,admitted=8,pending=0,
                   steps=td.STEPS,entry_steps=td.ENTRY_STEPS,measured_steps=td.MEASURED_STEPS,
                   tail_steps=td.TAIL_STEPS,ranks=[rank])
        self.assertEqual(driver._sanitize_receipt({'teacher_forced':proof})['teacher_forced'],proof)


class ProbeEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_controller_accepts_bool_call_overflow_and_refuses_missing(self):
        from test_e3_probe_session import CountedEngine
        from probe_control import ProbeController,ProbeControlError
        rows=[dict(rank=r,ack='flash_s5_teacher_evidence',engine_epoch=71,sampler_owner=r>=2,
                   drained_streams=3,positions=[],calls=[],overflow=False,call_overflow=False,
                   capture_defects=[],row_schema=td.ROW_SCHEMA if r>=2 else None,
                   row_basis=td.ROW_BASIS if r>=2 else None) for r in range(4)]
        class Engine(CountedEngine):
            async def collective_rpc(self,*a,**kw):
                if kw['args'][0]['action']=='teacher_evidence':return copy.deepcopy(rows)
                return await super().collective_rpc(*a,**kw)
        ctl=ProbeController(Engine(),epoch=71);ctl.paused=True
        ctl._teacher_plan=types.SimpleNamespace(pending={},status=lambda:{'schema':td.SCHEMA})
        out=await ctl.teacher_evidence()
        self.assertEqual([row['rank'] for row in out['ranks']],[0,1,2,3])
        del rows[2]['call_overflow']
        with self.assertRaisesRegex(ProbeControlError,'teacher-worker-evidence-unproved'):
            await ctl.teacher_evidence()

if __name__=='__main__':unittest.main()
