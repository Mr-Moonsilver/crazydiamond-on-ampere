"""Diagnostic-only all-rank pre-allocation floor; reuse the typed normal stop."""
from decode_graph_events import emit

FLOOR=40*2**20


class Floor:
    def __init__(self,driver):
        self.driver=driver;self.future=None;self.key=None;self.checked=None
        self.hold=None;self.stop_requested=False;self.settled=False;self.checks=0;self.leave_future=None;self.quiet_future=None;self.next_quiet=0.0

    def snapshot(self):
        return dict(floor_bytes=FLOOR,hold=self.hold,settled=self.settled,checks=self.checks,
                    pending=self.future is not None)

    def trip(self,reason):
        if self.hold is not None:return
        self.hold=str(reason)[:240]
        policy=self.driver.auto_arm
        policy.external_pause=True
        policy._pause(type(policy.scheduler.pause_state).PAUSED_ALL)
        emit(dict(kind='floor-hold',schema='decode-graph-events/1',reason=self.hold,floor_bytes=FLOOR))

    def observe(self,rows,context):
        try:
            if type(rows) not in (list,tuple) or len(rows)!=4:raise ValueError('missing-floor-peer')
            free={}
            for row in rows:
                rank=row.get('worker_rank');memory=row.get('memory')
                if (type(rank) is not int or rank not in range(4) or rank in free
                        or row.get('engine_epoch')!=self.driver.engine_epoch
                        or row.get('route_generation')!=self.driver.route_generation
                        or type(memory) is not dict or type(memory.get('free_bytes')) is not int
                        or memory['free_bytes']<0):raise ValueError('unknown-floor-peer-memory')
                free[rank]=memory['free_bytes']
            if any(value<FLOOR for value in free.values()):self.trip('physical-free-below-40MiB')
            self.checks+=1
            emit(dict(kind='floor-check',schema='decode-graph-events/1',context=context,
                      free_bytes={str(k):v for k,v in free.items()},floor_bytes=FLOOR,
                      rank3_margin_bytes=free[3]-FLOOR,hold=self.hold,
                      chunk=list(self.key[1:]) if context=='native-chunk-before-allocation' and self.key else None))
            return None if self.hold is not None else free
        except (ValueError,KeyError,TypeError) as exc:
            self.trip(type(exc).__name__+': '+str(exc));return None

    def before(self,key):
        if self.hold is not None:return False
        d=self.driver
        if self.future is None:
            command=dict(action='status',engine_epoch=d.engine_epoch,route_generation=d.route_generation)
            self.future=d.executor.collective_rpc('flash_s5_native_overlap',args=(command,),non_block=True)
            self.key=key
            return False
        if not self.future.done():return False
        future,self.future=self.future,None
        try:rows=d._validate_worker_rows(future.result(),'status')
        except Exception as exc:
            self.trip(type(exc).__name__+': '+str(exc));return False
        if key!=self.key:return False
        free=self.observe(rows,'native-chunk-before-allocation')
        self.checked=(key,free)
        return free is not None and self.hold is None

    def admitted(self,record):
        if self.checked is None:raise ValueError('native-chunk-without-floor-proof')
        key,free=self.checked;self.checked=None
        tx=record.transaction
        if key!=(tx.request_uid,tx.chunk_start,tx.chunk_end):
            raise ValueError('native-chunk-floor-proof-identity')
        emit(dict(kind='floor-chunk',schema='decode-graph-events/1',transaction=tx.txn_seq,
                  chunk=[tx.chunk_start,tx.chunk_end],free_bytes={str(k):v for k,v in free.items()},
                  floor_bytes=FLOOR,rank3_margin_bytes=free[3]-FLOOR))

    def quiet_tick(self):
        d=self.driver
        if self.quiet_future is not None:
            future,generation=self.quiet_future
            if not future.done():return
            self.quiet_future=None
            try:rows=d._validate_worker_rows(future.result(),'status')
            except Exception as exc:
                self.trip(type(exc).__name__+': '+str(exc));return
            # A route change makes an earlier sample stale, never a current permit.
            if generation==d.route_generation:self.observe(rows,'quiet-periodic')
        if (self.hold is None and d.auto_arm.manual_owned and not d.active
                and d._control is None and d._transition is None and d.clock()>=self.next_quiet):
            self.next_quiet=d.clock()+0.5
            command=dict(action='status',engine_epoch=d.engine_epoch,route_generation=d.route_generation)
            self.quiet_future=(d.executor.collective_rpc('flash_s5_native_overlap',args=(command,),non_block=True),d.route_generation)

    def stop_tick(self):
        """No worker decides alone; finish through existing PP/TP/row close proofs."""
        d=self.driver
        if self.settled:return False
        if self.quiet_future is not None:
            future,_=self.quiet_future
            if not future.done():return not bool(getattr(d.engine,'batch_queue',None))
            try:future.result()
            except Exception as exc:
                emit(dict(kind='floor-query-error',schema='decode-graph-events/1',error_type=type(exc).__name__,error=str(exc)[:240]))
            self.quiet_future=None
        if self.future is not None:
            if not self.future.done():
                return not bool(getattr(d.engine,'batch_queue',None))
            try:self.future.result()
            except Exception as exc:
                emit(dict(kind='floor-query-error',schema='decode-graph-events/1',error_type=type(exc).__name__,error=str(exc)[:240]))
            self.future=None
        if not self.stop_requested:
            d.stop('pause');self.stop_requested=True
        else:d.advance()
        if (not d.active and d._split_ready and d._retained is None and not d._inflight
                and d._control is None and not getattr(d.engine,'batch_queue',None)
                and getattr(d,'_control_route',None) is not None):
            # pause intentionally keeps a manual SPLIT lease; close it with the normal typed transition.
            if self.leave_future is None:
                self.leave_future=d.submit_control(dict(action='leave_split',engine_epoch=d.engine_epoch))
            return True
        if self.leave_future is not None:
            if not self.leave_future.done():return True
            self.leave_future.result();self.leave_future=None
        if (d.active or d._split_ready or d._retained is not None or d._inflight
                or d._control is not None or d._return_pending is not None
                or getattr(d.engine,'batch_queue',None)):
            return not bool(getattr(d.engine,'batch_queue',None))
        d._disarm_admission('graph-event-floor')
        policy=d.auto_arm
        if policy.active:policy._restore('graph-event-floor')
        policy.manual_owned=True
        self.settled=True
        emit(dict(kind='floor-settled',schema='decode-graph-events/1',hold=self.hold,
                  route='full',all_rank_stop=True,admission_disarmed=True))
        return False
