"""Pinned ready-response/RPC bodies with actual E3 hooks; CPU MQ leaves only."""
import ast
from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
from queue import Queue,Empty
import sys,threading,time,types,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness')]
import async_handoff as ah
import executor_hooks as eh
from reply_transport import ReplyTransport,TransportError,NotReady,Backpressure
PINS={eh.CORE_PATH:eh.CORE_SHA256,eh.EXECUTOR_PATH:eh.EXECUTOR_SHA256,
 'v1/executor/abstract.py':'97791426b549e4447c967f8b8e668c2e376c6a75fa7091fd9df11bc9dafb65f7'}
def method(path,owner,name,globals=None,generate=False):
 s=(ROOT/'reference/vllm-src'/path).read_text();assert hashlib.sha256(s.encode()).hexdigest()==PINS[path]
 if generate:s=eh.generate_executor(s)
 c=next(n for n in ast.parse(s).body if isinstance(n,ast.ClassDef) and n.name==owner)
 f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name)
 tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[])
 scope={'_flash_s5_hooks':eh};scope.update(globals or {});exec(compile(ast.fix_missing_locations(tree),path,'exec'),scope);return scope[name]
class Reader:
 def __init__(self):self.q=Queue();self.readers=set();self.closed=0
 def dequeue(self,timeout):
  self.readers.add(threading.get_ident())
  try:return self.q.get(timeout=timeout)
  except Empty:raise TimeoutError from None
 def shutdown(self):self.closed+=1
class Broadcast:
 def __init__(self,e):
  self.e=e;self.current_idx=0;self.n_remote_reader=0;self.closed=0
  self.buffer=types.SimpleNamespace(max_chunks=10,max_chunk_bytes=1024**2,n_reader=4,get_metadata=lambda i:nullcontext(bytearray(5)))
  self.entered=threading.Event();self.release=threading.Event();self.delay=False;self.writes=[];self.active=0;self.max_active=0;self.guard=threading.Lock()
 def enqueue(self,item):
  with self.guard:self.active+=1;self.max_active=max(self.max_active,self.active)
  try:
   if self.delay and not self.entered.is_set():self.entered.set();assert self.release.wait(6)
   self.writes.append(item);name,args,kw,rank=item
   for r,q in enumerate(self.e.response_mqs):
    v=self.e.capabilities[r] if name=='supports_draft_weight_updates' else (r,name,args)
    if rank is None or rank==r:q.q.put((self.e.SUCCESS,v))
  finally:
   with self.guard:self.active-=1
 def shutdown(self):self.closed+=1
class Executor:
 SUCCESS=object()
 collective_rpc=method(eh.EXECUTOR_PATH,'MultiprocExecutor','collective_rpc',generate=True)
 supports_draft_weight_updates=method('v1/executor/abstract.py','Executor','supports_draft_weight_updates')
 _flash_s5_stock_shutdown=method(eh.EXECUTOR_PATH,'MultiprocExecutor','_flash_s5_stock_shutdown',{'logger':types.SimpleNamespace(debug=lambda *a:None,debug_once=lambda *a:None)},generate=True)
 def __init__(self):
  self.workers=[];self.response_mqs=[Reader() for _ in range(4)];self.rpc_broadcast_mq=Broadcast(self);self.futures_queue=[];self.is_failed=False;self.capabilities=[False]*4
 def _flash_s5_stock_collective_rpc(self,*a,**kw):raise AssertionError('stock reply reader forbidden')
def install():
 e=Executor();t=eh.install(e,config=ah.HandoffConfig.from_env({'FLASH_S5_OVERLAP':'1'}),success_status=e.SUCCESS);return e,t
def threaded(fn):
 v=[];err=[]
 def run():
  try:v.append(fn())
  except BaseException as e:err.append(e)
 t=threading.Thread(target=run);t.start();return t,v,err
def join(test,row,timeout=3):
 t,v,e=row;t.join(timeout);test.assertFalse(t.is_alive(),'CPU thread did not finish');return v,e
class ThreadedExecutorTests(unittest.TestCase):
 def test_actual_pinned_ready_response_socket_thread(self):
  e,t=install();self.addCleanup(t.close,1)
  make=method(eh.CORE_PATH,'EngineCoreProc','_make_ready_response',{'EngineCoreReadyResponse':types.SimpleNamespace,'VLLM_VERSION':'pinned-source-fixture'})
  cfg=types.SimpleNamespace(parallel_config=types.SimpleNamespace(world_size=4,data_parallel_size=1,tensor_parallel_size=2,pipeline_parallel_size=2,decode_context_parallel_size=1),scheduler_config=types.SimpleNamespace(max_num_seqs=10,max_num_batched_tokens=1024),model_config=types.SimpleNamespace(max_model_len=262144,dtype='torch.bfloat16',enable_sleep_mode=False),cache_config=types.SimpleNamespace(num_gpu_blocks=1340,block_size=4048,mamba_block_size=4,kv_cache_size_tokens=100,kv_cache_max_concurrency=10),instance_id='CPU',lora_config=None,weight_transfer_config=None)
  core=types.SimpleNamespace(vllm_config=cfg,frontend_stats_publish_address=None,engine_index=0,scheduler=types.SimpleNamespace(get_kv_event_publisher_config=lambda:None),model_executor=e)
  for caps,expected in (([False]*4,False),([True,True,True,False],False),([True]*4,True)):
   e.capabilities=caps;v,err=join(self,threaded(lambda:make(core)));self.assertEqual(err,[]);self.assertEqual(v[0].supports_draft_weight_updates,expected)
  self.assertTrue(all(len(q.readers)==1 for q in e.response_mqs));self.assertTrue(eh.before_shutdown(e)[0])
 def test_noncreator_rpc_and_monitor_close_real_hooks(self):
  e,t=install();self.addCleanup(t.close,1)
  v,err=join(self,threaded(lambda:e.collective_rpc('check_health',args=('cpu',),timeout=2)))
  self.assertEqual(err,[]);self.assertEqual(v[0],[(r,'check_health',('cpu',)) for r in range(4)])
  v,err=join(self,threaded(lambda:eh.before_shutdown(e)));self.assertEqual(err,[]);self.assertTrue(v[0][0]);self.assertFalse(eh.retain_mq(e,None))
  self.assertTrue(all(q._flash_s5_receiver_owner is None for q in e.response_mqs))
  with self.assertRaises(TransportError):e.collective_rpc('later')
  with self.assertRaises(eh.E3ContractError):eh.install(e,config=ah.HandoffConfig.from_env({'FLASH_S5_OVERLAP':'1'}),success_status=e.SUCCESS)
 def test_registration_broadcast_single_fifo_section(self):
  e,t=install();self.addCleanup(t.close,1);b=e.rpc_broadcast_mq;b.delay=True
  a=threaded(lambda:e.collective_rpc('first',args=('A',),timeout=2));self.assertTrue(b.entered.wait(1))
  c=threaded(lambda:e.collective_rpc('second',args=('B',),timeout=2));time.sleep(.02)
  self.assertEqual(t._next,1);self.assertEqual(b.writes,[]);b.release.set();va,ea=join(self,a);vc,ec=join(self,c)
  self.assertEqual(ea+ec,[]);self.assertEqual([x[0] for x in b.writes],['first','second'])
  self.assertEqual(va[0],[(r,'first',('A',)) for r in range(4)]);self.assertEqual(vc[0],[(r,'second',('B',)) for r in range(4)]);self.assertEqual(b.max_active,1);self.assertEqual(t._calls,{})
 def test_two_submitters_unique_rank_and_aggregators(self):
  e,t=install();self.addCleanup(t.close,1);bad=[]
  class Aggregate:
   def aggregate(self,v,output_rank):return output_rank,tuple(v)
  def run(label):
   for i in range(25):
    label=('check_health','profile','sleep','wake_up','start_weight_update','update_weights','finish_weight_update','checkpoint_prepare','reset_mm_cache','get_supported_tasks')[i%10]
    if e.collective_rpc(label,args=(i,),timeout=2,unique_reply_rank=i%4)!=(i%4,label,(i,)):bad.append(i)
    if e.collective_rpc(label,args=(i,),timeout=2,unique_reply_rank=2,kv_output_aggregator=Aggregate())!=(2,tuple((r,label,(i,)) for r in range(4))):bad.append(i)
  a=threaded(lambda:run('health'));b=threaded(lambda:run('profile'));_,ea=join(self,a,5);_,eb=join(self,b,5)
  self.assertEqual(ea+eb,[]);self.assertEqual(bad,[]);self.assertEqual(t._calls,{});self.assertEqual(e.rpc_broadcast_mq.max_active,1);self.assertTrue(all(len(q.readers)==1 for q in e.response_mqs))
 def test_shared_future_result_aggregated_consumed_once(self):
  e,t=install();self.addCleanup(t.close,1);f=e.collective_rpc('future',non_block=True,timeout=2);calls=[];original=f.aggregate
  def aggregate(x):calls.append(x);return original(x)
  f.aggregate=aggregate;a=threaded(lambda:f.result(2));b=threaded(lambda:f.result(2));va,ea=join(self,a);vb,eb=join(self,b)
  self.assertEqual(ea+eb,[]);self.assertEqual(va,vb);self.assertEqual(len(calls),1);self.assertTrue(f.done());self.assertEqual(t._calls,{})
  with self.assertRaises(NotReady):f.take()
 def test_close_pending_broadcast_retains_mqs_and_calls(self):
  e,t=install();b=e.rpc_broadcast_mq;b.delay=True;a=threaded(lambda:e.collective_rpc('slow',timeout=2));self.assertTrue(b.entered.wait(1))
  start=time.monotonic();report=t.close(.02);self.assertLess(time.monotonic()-start,.5);self.assertFalse(report['clean']);self.assertTrue(report['active_broadcast']);self.assertTrue(all(q._flash_s5_receiver_owner is t for q in e.response_mqs))
  b.release.set();_,err=join(self,a);self.assertEqual(len(err),1);self.assertIsInstance(err[0],TransportError);self.assertFalse(t.close(1)['clean']);self.assertEqual(len(t._calls),1)
 def test_precheck_refusal_before_registration_no_poison(self):
  e,t=install();self.addCleanup(t.close,1)
  def refuse():raise Backpressure('command_ring_full')
  with self.assertRaises(Backpressure):t.submit('native',lambda _:self.fail('must not send'),precheck=refuse)
  self.assertEqual(t._calls,{});self.assertEqual(t._next,0);self.assertIsNone(t._failure)
  self.assertEqual(e.collective_rpc('after',timeout=2),[(r,'after',()) for r in range(4)])
 def test_reentrant_submission_cannot_steal_fifo(self):
  e,t=install();self.addCleanup(t.close,1)
  def send(seq):
   with self.assertRaisesRegex(TransportError,'reentrant'):t.submit('nested',lambda _:None)
   for q in e.response_mqs:q.q.put((e.SUCCESS,seq))
  seq=t.submit('outer',send,timeout_s=2);end=time.monotonic()+2
  while not t.call_ready(seq) and time.monotonic()<end:t.poll(4)
  self.assertEqual(t.take(seq)[0],[0]*4);self.assertEqual(t._next,1)
 @unittest.skipUnless(hasattr(os,'fork'),'POSIX process-owner refusal')
 def test_forked_process_refused_before_any_parent_lock_or_mq(self):
  e,t=install();self.addCleanup(t.close,1);r,w=os.pipe();pid=os.fork()
  if pid==0:
   os.close(r);out=[]
   for fn in (lambda:t.poll(),lambda:t.take(0),lambda:t.submit('fork',lambda _:None),lambda:t.close(0),lambda:eh.before_shutdown(e),lambda:e.collective_rpc('fork',timeout=1)):
    try:fn();out.append(False)
    except TransportError:out.append(True)
   os.write(w,bytes(out));os._exit(0)
  os.close(w);data=os.read(r,6);os.close(r);_,status=os.waitpid(pid,0);self.assertEqual(status,0);self.assertEqual(data,b'\x01'*6);self.assertTrue(eh.before_shutdown(e)[0])
 def test_concurrent_monitor_and_finalizer_close_idempotent(self):
  e,t=install();a=threaded(lambda:eh.before_shutdown(e));b=threaded(lambda:eh.before_shutdown(e));va,ea=join(self,a);vb,eb=join(self,b)
  self.assertEqual(ea+eb,[]);self.assertTrue(va[0][0]);self.assertEqual(va,vb);self.assertTrue(eh.before_shutdown(e)[0])
 def test_native_actor_still_refuses_foreign_thread_mutation(self):
  d=ah.HandoffDispatcher(ah.HandoffConfig.from_env({'FLASH_S5_OVERLAP':'1'}))
  _,errors=join(self,threaded(d.assert_owner));self.assertEqual(len(errors),1);self.assertIsInstance(errors[0],ah.HandoffError)
 def stock_shutdown(self):
  logger=types.SimpleNamespace(debug=lambda *a:None,debug_once=lambda *a:None)
  return method(eh.EXECUTOR_PATH,'MultiprocExecutor','shutdown',{'logger':logger},generate=True)
 def test_actual_generated_shutdown_closes_clean_mqs_from_monitor_thread(self):
  e,t=install();readers=list(e.response_mqs);writer=e.rpc_broadcast_mq
  _,err=join(self,threaded(lambda:self.stock_shutdown()(e)))
  self.assertEqual(err,[]);self.assertTrue(e._flash_s5_close['clean']);self.assertEqual(writer.closed,1)
  self.assertTrue(all(q.closed==1 for q in readers));self.assertEqual(e.response_mqs,[])
 def test_actual_generated_shutdown_retains_both_writer_and_readers_when_unclean(self):
  e,t=install();writer=e.rpc_broadcast_mq;readers=list(e.response_mqs);writer.delay=True
  active=threaded(lambda:e.collective_rpc('slow',timeout=5));self.assertTrue(writer.entered.wait(1))
  _,err=join(self,threaded(lambda:self.stock_shutdown()(e)),4)
  self.assertEqual(err,[]);self.assertFalse(e._flash_s5_close['clean']);self.assertEqual(writer.closed,0)
  self.assertTrue(all(q.closed==0 and q._flash_s5_receiver_owner is t for q in readers))
  writer.release.set();_,err=join(self,active);self.assertEqual(len(err),1);self.assertIsInstance(err[0],TransportError)
  self.assertFalse(t.close(1)['clean'])
 def test_poll_and_ready_take_do_not_wait_for_another_broadcast(self):
  e,t=install();self.addCleanup(t.close,1);f=e.collective_rpc('ready',non_block=True,timeout=2)
  end=time.monotonic()+2
  while not f.done() and time.monotonic()<end:time.sleep(.001)
  self.assertTrue(f.done());b=e.rpc_broadcast_mq;b.delay=True
  slow=threaded(lambda:e.collective_rpc('slow',timeout=2));self.assertTrue(b.entered.wait(1))
  start=time.monotonic();self.assertEqual(t.poll(4),0);value=f.take();self.assertLess(time.monotonic()-start,.2)
  self.assertEqual(value,[(r,'ready',()) for r in range(4)])
  b.release.set();_,err=join(self,slow);self.assertEqual(err,[])
 def test_native_precheck_writer_contention_is_immediate_backpressure(self):
  e,t=install();self.addCleanup(t.close,1);b=e.rpc_broadcast_mq;b.delay=True
  slow=threaded(lambda:e.collective_rpc('slow',timeout=2));self.assertTrue(b.entered.wait(1))
  start=time.monotonic()
  with self.assertRaisesRegex(Backpressure,'serialization_busy'):t.submit('native',lambda _:None,precheck=lambda:None)
  self.assertLess(time.monotonic()-start,.2);self.assertEqual(t._next,1);self.assertEqual(len(t._calls),1);self.assertIsNone(t._failure)
  b.release.set();_,err=join(self,slow);self.assertEqual(err,[])
 def test_full_stock_shutdown_is_serialized_until_workers_terminate(self):
  e,t=install();readers=list(e.response_mqs);writer=e.rpc_broadcast_mq
  e.workers=[types.SimpleNamespace(proc=object(),death_writer=None,worker_response_mq=readers[0])]
  entered=threading.Event();release=threading.Event()
  def terminate(procs):entered.set();assert release.wait(3)
  e._ensure_worker_termination=terminate
  a=threaded(lambda:self.stock_shutdown()(e));self.assertTrue(entered.wait(1))
  b=threaded(lambda:self.stock_shutdown()(e));time.sleep(.02)
  self.assertEqual(writer.closed,0);self.assertTrue(all(q.closed==0 for q in readers))
  self.assertTrue(b[0].is_alive());release.set();_,ea=join(self,a);_,eb=join(self,b)
  self.assertEqual(ea+eb,[]);self.assertEqual(writer.closed,1);self.assertEqual(e.response_mqs,[])
if __name__=='__main__':unittest.main()
