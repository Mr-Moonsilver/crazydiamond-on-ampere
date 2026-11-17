"""Default-off fit census. Ordered all-rank samples, no extra graph or model work."""
import json
import logging
import os
import time

from graph_event_floor import Floor

ENV = 'FLASH_S5_FIT_MEMORY'
ROLES = ('e6', 'e4096', 'stock')
FLOOR = 40 * 2**20
PREFIX = 'flash-s5 fit-memory '
OUTSIDE = (1259,1371,1197,1309)


def role(env=None):
    value = (os.environ if env is None else env).get(ENV, '0')
    if value not in ('0', *ROLES):
        raise ValueError('invalid-fit-role')
    return None if value == '0' else value


def configuration(engine):
    cfg=getattr(engine,'vllm_config',None)
    value=lambda owner,key:getattr(getattr(cfg,owner,None),key,None)
    return dict(max_model_len=value('model_config','max_model_len'),
                max_num_seqs=value('scheduler_config','max_num_seqs'),
                kv_cache_memory_bytes=value('cache_config','kv_cache_memory_bytes'),
                tensor_parallel_size=value('parallel_config','tensor_parallel_size'),
                pipeline_parallel_size=value('parallel_config','pipeline_parallel_size'),
                rowcache_reqs=os.environ.get('QSA_ROWCACHE_REQS'),
                rowcache_slots=os.environ.get('QSA_ROWCACHE_SLOTS'))


def emit(row):
    from native_measurement import emit_receipt_lines
    text = PREFIX+json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)
    if len(text.encode()) > 3500:
        raise ValueError('fit-receipt-too-large')
    if emit_receipt_lines((text,),logging.getLogger(__name__)) is False:
        raise RuntimeError('fit-receipt-lost')


def worker_measure(worker, command, *, torch_module=None):
    selected = role()
    if selected is None or type(command) is not dict or set(command) != {'sequence','role'} or command['role'] != selected:
        raise ValueError('fit-worker-command-refused')
    if type(command['sequence']) is not int or command['sequence'] < 1:
        raise ValueError('fit-worker-sequence')
    rank = worker.rank
    if type(rank) is not int or rank not in range(4):
        raise ValueError('fit-worker-rank')
    torch = __import__('torch') if torch_module is None else torch_module
    device = worker.device
    startup = getattr(worker,'_flash_fit_startup',None)
    if startup is None:
        # Before any packet request: explicitly synchronized feature receipts.
        # There are no new graph captures, tensors, streams or CUDA events.
        from startup_vram_budget import StartupBudget, VramBudgetRefused
        limits={str(r):dict(total_delta_bytes=4*2**20,outside_delta_bytes=4*2**20,
                           outside_absolute_bytes=OUTSIDE[r]*2**20) for r in range(4)}
        budget=StartupBudget(torch,device.index if hasattr(device,'index') else device,rank,
                             feature='fit-memory-no-extra-captures',budget=limits,writer=lambda row:(emit(row), receipts.append(row)))
        receipts=[]
        try:
            budget.begin()
            result=budget.finish()
            startup=dict(verdict='PASS',feature='fit-memory-no-extra-captures',limits=limits[str(rank)],receipts=receipts,**result)
        except VramBudgetRefused as exc:
            logging.getLogger(__name__).exception('fit startup refused')
            startup=dict(verdict='HOLD',reason=str(exc))
        worker._flash_fit_startup=startup
    free,total=torch.cuda.mem_get_info(device)
    stats=torch.cuda.memory_stats(device)
    # Like native_overlap_worker.measure_memory: no cudaDeviceSynchronize here.
    # The engine issues and joins the SAME sequence on all four worker FIFOs.
    mm = {}
    if os.environ.get('FLASH_S5_NATIVE_MM', '0') == '1':
        from native_mm import inventory
        mm = inventory(worker.model_runner)
    host = {}
    with open('/proc/self/status') as status:
        for line in status:
            name, _, value = line.partition(':')
            if name in ('VmRSS', 'VmHWM', 'RssAnon', 'RssShmem'):
                host[name] = int(value.split()[0])*1024
    return dict(native_mm=mm,host_rss_bytes=host,rank=rank,pid=os.getpid(),role=selected,sequence=command['sequence'],
                native_present=getattr(worker.model_runner,'_flash_s5_native',None) is not None,
                observed_ns=time.monotonic_ns(),free_bytes=int(free),total_bytes=int(total),
                allocated_bytes=int(torch.cuda.memory_allocated(device)),
                reserved_bytes=int(torch.cuda.memory_reserved(device)),
                allocator_peak_allocated_bytes=int(stats.get('allocated_bytes.all.peak',torch.cuda.memory_allocated(device))),
                allocator_peak_reserved_bytes=int(stats.get('reserved_bytes.all.peak',torch.cuda.memory_reserved(device))),
                ooms=int(stats.get('num_ooms',0)),startup=startup)


class FitFloor(Floor):
    def __init__(self,engine,selected,*,writer=emit):
        driver=getattr(engine,'_flash_s5_driver',None)
        super().__init__(driver)
        if (selected=='stock') != (driver is None):
            raise ValueError('fit-stock-must-not-own-native-driver')
        self.engine,self.selected,self.writer=engine,selected,writer
        self.sequence=0
        self.owed=[[('startup',None)]]
        self.in_query=[]
        self.minima={}
        self.mm_peaks={}
        self.host_rss_peaks={}
        self.torch_peaks={}
        self.startup={}
        self.mm_inventory={}
        self.mm_peak_storage={}
        self.worker_host_peaks=self.host_rss_peaks
        self.allocator_peaks=self.torch_peaks
        self.ooms=0
        self.preemptions=0
        self.finishing=False
        self.coverage=dict(decode_rounds=0,decode_checked=0,chunks=0,chunk_checked=0,checks=0)
        self.pending_chunk=None
        self.first_admissions={}
        self.permit_samples={'native-arm':[],'chunk':[]}
        self.resident_peak=dict(tokens=0,members=[],samples=0,unknown=0,eight_at_190k_seen=False,eight_minimum_peak=0,
            basis='sum min(num_computed_tokens,num_tokens) over unfinished scheduler requests; includes allocated in-flight frontiers, excludes freed prefix cache and compression')

    def trip(self,reason):
        if self.hold is not None:return
        self.hold=str(reason)[:240]
        if self.driver is not None:
            self.driver.auto_arm.external_pause=True
        scheduler=self.engine.scheduler
        scheduler.set_pause_state(type(scheduler.pause_state).PAUSED_ALL)
        self.writer(dict(kind='hold',role=self.selected,reason=self.hold,rank_map=dict(self.minima)))

    def observe(self,rows):
        try:
            if type(rows) not in (tuple,list) or len(rows)!=4:raise ValueError('missing-fit-peer')
            free={}
            for row in rows:
                rank=row.get('rank')
                if (type(rank) is not int or rank not in range(4) or str(rank) in free
                        or row.get('sequence')!=self.sequence or row.get('role')!=self.selected
                        or type(row.get('free_bytes')) is not int or row['free_bytes']<0
                        or type(row.get('ooms')) is not int or row['ooms']<0
                        or type(row.get('startup')) is not dict
                        or row.get('native_present') is not (self.selected!='stock')):
                    raise ValueError('foreign-fit-peer-round-or-native-owner')
                key=str(rank);free[key]=row['free_bytes']
                self.minima[key]=min(self.minima.get(key,row['free_bytes']),row['free_bytes'])
                self.ooms=max(self.ooms,row['ooms'])
                self.startup[key]=row.get('startup',{})
                for field in ('host_rss_bytes',):
                    if field in row:
                        peak=self.host_rss_peaks.setdefault(key,{})
                        for name,value in row[field].items():peak[name]=max(peak.get(name,0),value)
                peak=self.torch_peaks.setdefault(key,{})
                for name in ('allocated_bytes','reserved_bytes','allocator_peak_allocated_bytes','allocator_peak_reserved_bytes'):
                    if name in row:peak[name]=max(peak.get(name,0),row[name])
                mm=row.get('native_mm')
                if mm is not None:
                    peak=self.mm_peaks.setdefault(key,{})
                    for name,value in mm.items():
                        if type(value) is int:peak[name]=max(peak.get(name,0),value)
                if os.environ.get('FLASH_S5_NATIVE_MM','0') == '1':
                    mm=row.get('native_mm')
                    if (type(mm) is not dict or mm.get('schema')!='native-mm-inventory/1'
                            or mm.get('tower_registered_cuda_bytes')!=0
                            or type(mm.get('cache_storage_bytes')) is not int
                            or mm['cache_storage_bytes']>40*2**20
                            or mm.get('cache_feature_bytes')!=mm['cache_storage_bytes']):
                        raise ValueError('MM-tower-or-private-cache-budget-unknown')
                    self.mm_inventory[key]=mm
                    self.mm_peak_storage[key]=max(self.mm_peak_storage.get(key,0),mm['cache_storage_bytes'])
            self.coverage['checks']+=1
            for kind,key in self.in_query:
                if kind=='decode':self.coverage['decode_checked']+=1
                if kind in ('chunk','stock-chunk'):self.coverage['chunk_checked']+=1
                if kind=='chunk':self.checked=(key,{int(r):v for r,v in free.items()})
            if any(v<FLOOR for v in free.values()):self.trip('physical-free-below-40MiB')
            if self.ooms:self.trip('allocator-OOM')
            if any(s.get('verdict')!='PASS' for s in self.startup.values()):self.trip('startup-peer-budget')
            self.writer(dict(kind='check',schema='fit-memory/1',role=self.selected,sequence=self.sequence,
                             contexts=[kind for kind,_ in self.in_query],free_bytes=free,
                             rank3_margin_bytes=free['3']-FLOOR,hold=self.hold))
        except (ValueError,KeyError,TypeError) as exc:
            self.trip(type(exc).__name__+':'+str(exc))

    def tick(self):
        if self.hold is not None:
            return self.stop_tick()
        if self.future is not None:
            if not self.future.done():
                if self.driver is not None:
                    # Keep ordinary queued frames and native progress live.
                    # Only startup or a two-boundary census backlog applies backpressure.
                    return self.in_query==[('startup',None)] or len(self.owed)>=2
                if self.engine.batch_queue and len(self.owed)<2:
                    return False
            future,self.future=self.future,None
            try:
                self.observe(future.result())
            except Exception as exc:
                logging.getLogger(__name__).exception('fit ordered census failed')
                self.trip(type(exc).__name__+':'+str(exc))
            self.in_query=[]
        if self.hold is not None:return self.stop_tick()
        if self.owed:
            self.in_query=self.owed.pop(0)
            self.sequence+=1
            self.future=self.engine.model_executor.collective_rpc('flash_s5_fit_memory',
                args=(dict(sequence=self.sequence,role=self.selected),),timeout=30,non_block=True)
            # Stock FutureWrapper only receives ordered replies in result().
            # There is no native command needing host progress on this route.
            if self.driver is None and not self.engine.batch_queue:
                future,self.future=self.future,None
                try:self.observe(future.result())
                except Exception as exc:
                    logging.getLogger(__name__).exception('fit stock census failed')
                    self.trip(type(exc).__name__+':'+str(exc))
                self.in_query=[]
                return self.hold is not None
            if self.driver is None:return False
            return self.in_query==[('startup',None)] or len(self.owed)>=2
        if self.finishing:return self.stop_tick()
        return False

    def measure_before(self,context,key=None):
        if len(self.permit_samples[context])>=16384:
            self.trip('permit-duration-capacity');return False
        start=time.monotonic_ns()
        result=False
        try:
            result=self._measure_before(context,key)
            return result
        finally:
            end=time.monotonic_ns();samples=self.permit_samples[context]
            samples.append(end-start)
            self.writer(dict(kind='permit',role=self.selected,context=context,
                start_ns=start,end_ns=end,duration_ns=end-start,passed=result,
                chunk=None if key is None else list(key[1:])))

    def _measure_before(self,context,key=None):
        """Join worker FIFO replies before committing an intent or allocation."""
        if self.hold is not None:return False
        try:
            if self.future is not None:
                future,self.future=self.future,None
                self.observe(future.result());self.in_query=[]
            while self.owed and self.hold is None:
                self.in_query=self.owed.pop(0);self.sequence+=1
                self.observe(self.engine.model_executor.collective_rpc('flash_s5_fit_memory',
                    args=(dict(sequence=self.sequence,role=self.selected),),timeout=30))
                self.in_query=[]
            if self.hold is not None:return False
            self.in_query=[(context,key)];self.sequence+=1
            self.observe(self.engine.model_executor.collective_rpc('flash_s5_fit_memory',
                args=(dict(sequence=self.sequence,role=self.selected),),timeout=30))
            self.in_query=[]
        except Exception as exc:
            logging.getLogger(__name__).exception('fit pre-commit census failed')
            self.in_query=[];self.trip(type(exc).__name__+':'+str(exc))
        return self.hold is None

    def before_arm(self):
        return self.measure_before('native-arm')

    def before(self,key):
        if self.hold is not None:return False
        self.pending_chunk=key;self.checked=None
        self.coverage['chunks']+=1
        return self.measure_before('chunk',key)

    def admitted(self,record):
        tx=record.transaction;key=(tx.request_uid,tx.chunk_start,tx.chunk_end)
        if self.checked is None or self.checked[0]!=key:raise ValueError('fit-chunk-without-exact-floor-permit')
        self.checked=None;self.pending_chunk=None
        if tx.request_uid not in self.first_admissions:
            if len(self.first_admissions)>=128:raise ValueError('fit-first-admission-budget')
            self.first_admissions[tx.request_uid]=dict(start=record.committed_end,prompt_tokens=record.prompt_end,
                transaction=tx.txn_seq,engine_epoch=self.driver.engine_epoch)

    def quiet_tick(self):
        # Every stock schedule is marked explicitly, including outside chunks.
        return None

    def sample_resident(self):
        members=[]
        for uid,request in getattr(self.engine.scheduler,'requests',{}).items():
            if str(getattr(getattr(request,'status',None),'name','')).startswith('FINISHED'):continue
            computed=getattr(request,'num_computed_tokens',None)
            total=getattr(request,'num_tokens',None)
            if type(computed) is not int or type(total) is not int or min(computed,total)<0:
                self.resident_peak['unknown']+=1;continue
            if computed:members.append(dict(uid=uid,tokens=min(computed,total)))
        tokens=sum(m['tokens'] for m in members)
        if len(members)>=8:
            eighth=sorted((m['tokens'] for m in members),reverse=True)[7]
            self.resident_peak['eight_minimum_peak']=max(self.resident_peak['eight_minimum_peak'],eighth)
            if eighth>=190000:self.resident_peak['eight_at_190k_seen']=True
        self.resident_peak['samples']+=1
        if tokens>self.resident_peak['tokens']:
            self.resident_peak.update(tokens=tokens,members=members,observed_ns=time.monotonic_ns())

    def permit_summary(self):
        result={}
        for kind,values in self.permit_samples.items():
            ordered=sorted(values);n=len(ordered)
            result[kind]=dict(count=n,p50_ms=(ordered[(n-1)//2]+ordered[n//2])/2e6 if n else None,
                max_ms=max(ordered)/1e6 if n else None,total_ms=sum(ordered)/1e6)
        return result

    def note_schedule(self,output):
        self.sample_resident()  # Catch new allocations before an immediately-ready output removes its request.
        tokens=getattr(output,'num_scheduled_tokens',None)
        if type(tokens) is not dict:raise ValueError('fit-scheduler-shape')
        contexts=[]
        if any(v==1 for v in tokens.values()):
            self.coverage['decode_rounds']+=1;contexts.append(('decode',None))
        if any(v>1 for v in tokens.values()):
            self.coverage['chunks']+=1;contexts.append(('stock-chunk',None))
        if contexts:self.owed.append(contexts)

    def stop_tick(self):
        if self.settled:return True
        # A pending all-rank measurement is read before any release/stop proof.
        if self.future is not None:
            if self.driver is not None and not self.future.done():return False
            future,self.future=self.future,None
            self.observe(future.result());self.in_query=[]
        if self.driver is not None:
            blocked=super().stop_tick()
            if self.settled:self.writer(dict(kind='settled',role=self.selected,all_rank_stop=True))
            return self.settled or blocked
        self.engine.scheduler.set_pause_state(type(self.engine.scheduler.pause_state).PAUSED_ALL)
        if self.engine.batch_queue:return False  # Stock Core drains its real queued outputs.
        # Ordered final reply on every rank proves every earlier stock command
        # returned; this is a normal FULL process, no native leases to fabricate.
        self.sequence+=1
        self.in_query=[('final',None)]
        rows=self.engine.model_executor.collective_rpc('flash_s5_fit_memory',
                args=(dict(sequence=self.sequence,role=self.selected),),timeout=30)
        self.observe(rows);self.in_query=[];self.settled=True
        self.writer(dict(kind='settled',role=self.selected,all_rank_stop=True))
        return True

    def snapshot(self):
        native=None
        if self.driver is not None:native=self.driver.controller.coverage.record()
        return dict(schema='fit-memory/1',role=self.selected,hold=self.hold,settled=self.settled,
                    engine_epoch=self.driver.engine_epoch if self.driver is not None else None,
                    configuration=configuration(self.engine),
                    active_requests=[str(k) for k,r in getattr(self.engine.scheduler,'requests',{}).items()
                        if not str(getattr(getattr(r,'status',None),'name','')).startswith('FINISHED')],
                    native_mm=dict(peers=self.mm_inventory,peak_cache_storage_bytes=self.mm_peak_storage),
                    worker_host_peaks_bytes=self.worker_host_peaks,allocator_peaks_bytes=self.allocator_peaks,
                    min_free_bytes=dict(self.minima),coverage=dict(self.coverage),
                    mm_peaks=self.mm_peaks,host_rss_peaks=self.host_rss_peaks,torch_peaks=self.torch_peaks,
                    active_request_ids=[str(uid) for uid,r in getattr(self.engine.scheduler,'requests',{}).items()
                        if not str(getattr(getattr(r,'status',None),'name','')).startswith('FINISHED')],
                    preemptions=self.preemptions,ooms=self.ooms,fatal=None,errors400=None,
                    native_coverage=native,first_admissions=dict(self.first_admissions),
                    native_admissions=self.driver.controller.admitted if self.driver is not None else 0,
                    permit_latency=self.permit_summary(),simultaneous_resident_token_peak=dict(self.resident_peak),
                    stop_proof=dict(route='full' if self.settled else None,all_rank_stop=self.settled,
                        admission_disarmed=self.settled and (self.driver is None or self.driver.admission_intent() is None),
                        output_queue_empty=not bool(self.engine.batch_queue)),
                    startup=dict(verdict='PASS' if len(self.startup)==4 and all(s.get('verdict')=='PASS' for s in self.startup.values()) else 'HOLD',
                                 ranks=sorted(int(r) for r in self.startup),peers=self.startup),
                    sampling='ordered-all-rank-FIFO-mem_get_info; no CUDA drain during workload')


def install(engine):
    selected=role()
    if selected is None:return
    floor=FitFloor(engine,selected)
    engine._flash_fit_memory=floor
    if floor.driver is not None:floor.driver.graph_event_floor=floor
    old=engine.scheduler._preempt_request
    def preempt(*args,**kwargs):
        floor.preemptions+=1
        floor.trip('scheduler-preemption')
        return old(*args,**kwargs)
    engine.scheduler._preempt_request=preempt


def before_step(engine):
    floor=getattr(engine,'_flash_fit_memory',None)
    if floor is None:return None
    floor.sample_resident()
    if floor.tick():
        if floor.driver is not None and not floor.settled:floor.driver.advance()
        return {},False
    return None


def note_schedule(engine,output):
    floor=getattr(engine,'_flash_fit_memory',None)
    if floor is not None:floor.note_schedule(output)


def pending(engine):
    floor=getattr(engine,'_flash_fit_memory',None)
    return floor is not None and not floor.settled and bool(floor.future or floor.owed or floor.finishing)


def control(engine,command):
    floor=getattr(engine,'_flash_fit_memory',None)
    if floor is None or type(command) is not dict or set(command)!={'action'}:
        raise ValueError('fit-control-not-installed-or-invalid')
    action=command['action']
    if action not in ('status','finish','stop'):raise ValueError('fit-control-action')
    if action=='stop':floor.trip('client-stop')
    if action in ('stop','finish'):floor.finishing=True
    return floor.snapshot()


def install_api(args,app):
    if role() is None:return
    import asyncio
    import hashlib
    import ipaddress
    import secrets
    from fastapi import APIRouter,Request
    from fastapi.responses import JSONResponse
    configured=getattr(args,'api_key',None) or [os.environ.get('VLLM_API_KEY')]
    if type(configured) is not list or not configured or any(type(k) is not str or not k for k in configured):
        raise ValueError('fit-existing-api-auth-required')
    hashes=[hashlib.sha256(k.encode()).digest() for k in configured]
    if os.environ.get('VLLM_SERVER_DEV_MODE','0') not in ('0',''):raise ValueError('fit-refuses-dev-router')
    router=APIRouter()
    async def endpoint(request):
        try:local=ipaddress.ip_address(request.client.host).is_loopback
        except ValueError:local=False
        scheme,_,token=request.headers.get('authorization','').partition(' ')
        digest=hashlib.sha256(token.encode()).digest()
        if not local or scheme.lower()!='bearer' or not any(secrets.compare_digest(digest,x) for x in hashes):
            return JSONResponse({'error':'fit-auth-required'},status_code=403)
        body=bytearray()
        async for piece in request.stream():
            body.extend(piece)
            if len(body)>256:return JSONResponse({'error':'fit-command-too-large'},status_code=400)
        command=json.loads(body)
        if type(command) is not dict or set(command)!={'action'} or command['action'] not in ('status','finish','stop'):
            return JSONResponse({'error':'fit-command-refused'},status_code=400)
        client=request.app.state.engine_client.engine_core
        if client.client_count!=1 or client.client_index!=0:raise ValueError('fit-single-owning-api-required')
        reply=await asyncio.wait_for(client.call_utility_async('flash_s5_fit_control',command),60)
        return JSONResponse(reply)
    endpoint.__annotations__['request']=Request
    router.add_api_route('/v1/flash_s5_fit_memory',endpoint,methods=['POST'])
    app.include_router(router)
