"""Deferred CUPTI correlation joins. No CUDA import, timestamp-nearest guesses,
loss resets, or exception for graph-ID-zero work. Engine offers are separate
from the immutable raw graph-only placement receipt.
"""
from collections import Counter
import re

SCHEMA='s5ro-correlation/2'
KINDS=('kernel','memcpy','memset')
# Native API rows are not the bounded engine receipt. Keep this separate from
# the graph-only report's8MiB and engine receipt's16MiB limits.
CORRELATION_REPORT_MAX_BYTES=64*1024*1024

# CUDA_ERROR_NOT_READY / cudaErrorNotReady (reference/cuda-13/cuda.h:2828,
# reference/cuda-runtime-13/driver_types.h:767). cuEventQuery/cudaEventQuery
# return it while captured work is pending; that is a documented host
# observation, not a failure and not completion (cuda.h:16048-16071,
# cuda_runtime_api.h near 2920). Pinned to the event-query family only.
CUDA_ERROR_NOT_READY=600
_HOST_QUERY_APIS=('cudaEventQuery','cuEventQuery')


def is_host_query(name):
    """True only for the exact event-query API (version suffix rules)."""
    if not isinstance(name,str):return False
    return any(re.fullmatch(re.escape(stem)+r'(?:_v\d+)?(?:_ptsz|_ptds)?',name)
               for stem in _HOST_QUERY_APIS)


def host_return_class(name,return_value):
    """Classify one host-API return: ``success`` / ``pending`` / ``refused``.

    Only cuEventQuery/cudaEventQuery may report the pending status. A pending
    query observes zero GPU work; it is never completion and never evidence
    for a missing event, source or graph. Every other nonzero return refuses.
    """
    if type(return_value) is not int:return 'refused'
    if return_value==0:return 'success'
    if return_value==CUDA_ERROR_NOT_READY and is_host_query(name):return 'pending'
    return 'refused'


def host_return_ok(name,return_value):
    """A zero-GPU-work host observation: success, or a pending event query."""
    return host_return_class(name,return_value) in ('success','pending')


def api_kind(name):
    # Exact API families, version/thread-default suffixes only. Unsupported
    # launches cannot be silently classified as host-only work.
    for stem,kind in (('cudaLaunchKernel','kernel'),('cuLaunchKernel','kernel'),
        ('cudaLaunchKernelExC','kernel'),('cuLaunchKernelEx','kernel'),
        ('cudaGraphLaunch','graph'),('cuGraphLaunch','graph'),
        ('cudaMemcpyAsync','memcpy'),('cuMemcpyHtoDAsync','memcpy'),
        ('cuMemcpyDtoHAsync','memcpy'),('cuMemcpyDtoDAsync','memcpy'),
        ('cuMemcpyAsync','memcpy'),('cudaMemsetAsync','memset'),
        ('cuMemsetD8Async','memset'),('cuMemsetD16Async','memset'),('cuMemsetD32Async','memset')):
        if re.fullmatch(re.escape(stem)+r'(?:_v\d+)?(?:_ptsz|_ptds)?',name):return kind
    # RouteController.drain (split_bank.py) creates, records, synchronizes and
    # releases a tail event on every consumer stream. Creation/destruction are
    # host control operations. Every such API must still return success and
    # have ZERO kernel/copy/memset activity; grade_engine enforces both.
    harmless=('cudaEventCreate','cudaEventCreateWithFlags','cudaEventDestroy',
        'cuEventCreate','cuEventDestroy',
        'cudaGetDevice','cudaSetDevice','cudaGetLastError','cudaPeekAtLastError',
        'cudaStreamSynchronize','cudaDeviceSynchronize','cudaEventRecord','cudaEventQuery',
        'cudaEventSynchronize','cudaStreamWaitEvent','cuStreamSynchronize','cuEventRecord',
        'cuEventQuery','cuEventSynchronize','cuStreamWaitEvent','cuCtxGetCurrent',
        'cuCtxSetCurrent','cuCtxGetDevice','cuCtxPushCurrent','cuCtxPopCurrent',
        'cuGetProcAddress','cuGetExportTable','cudaGetDriverEntryPointByVersion',
        'cuStreamGetCtx','cuStreamGetId','cuCtxGetId','cuDevicePrimaryCtxGetState',
        # Pinned CUDA13 metadata queries and event recording. These have no
        # kernel/copy/memset activity; unexpected activity still fails the join.
        'cuFuncGetAttribute','cuKernelGetFunction','cuKernelGetName',
        'cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags','cuPointerGetAttribute',
        'cudaDeviceGetAttribute','cudaEventRecordWithFlags','cudaStreamIsCapturing')
    if any(re.fullmatch(re.escape(x)+r'(?:_v\d+)?(?:_ptsz|_ptds)?',name) for x in harmless):return 'host'
    return None


def grade(report,offers,*,epoch,generation):
    f=[];m=[];families={};joins=[];wrappers=[]
    def require(ok,reason):
        if not ok:raise ValueError(reason)
    try:
        require(report.get('schema')==SCHEMA,'correlation_schema')
        require(report.get('complete') is True,'correlation_incomplete')
        require(report.get('thread_id_type') in ('system-gettid','default-pthread-low32'),'thread_key_basis')
        require(all(report.get(k)==0 for k in ('overflow','errors','open_scopes','boundary_records')),'correlation_loss_or_scope')
        om={}
        for o in offers:
            ident=o.get('id');require(type(ident) is int and ident>0 and ident not in om,'offer_identity')
            require(o.get('epoch')==epoch and o.get('generation')==generation,'offer_generation')
            require(o.get('closed') is True and type(o.get('thread')) is int and o['thread']>0,'offer_lifetime')
            require(type(o.get('family')) is str and o['family'],'offer_family')
            require(set(o.get('expected',{}))<=set(KINDS) and all(type(n) is int and n>=0 for n in o['expected'].values()),'offer_expected_counts')
            om[ident]=o
        keys={};systems={}
        for o in offers:
            require(type(o.get('system_thread')) is int and o['system_thread']>0,'system_thread_missing')
            require(keys.setdefault(o['thread'],o['system_thread'])==o['system_thread'] and systems.setdefault(o['system_thread'],o['thread'])==o['thread'],'thread_key_not_bijective')
            if report['thread_id_type']=='system-gettid':require(o['thread']==o['system_thread'],'system_thread_key_differs')
            if o.get('parent'):
                p=om.get(o['parent']);require(p is not None and p['thread']==o['thread'] and p['id']!=o['id'],'offer_parent')
        stacks={};opened=set();closed=set()
        for e in report['scopes']:
            o=om.get(e['id']);require(o is not None and o['thread']==e['thread'] and o['system_thread']==e.get('system_thread'),'native_scope_identity')
            stack=stacks.setdefault(e['thread'],[])
            if e['action']=='push':
                require(e['id'] not in opened and e['parent']==o.get('parent',0)==(stack[-1] if stack else 0),'native_scope_parent')
                opened.add(e['id']);stack.append(e['id'])
            elif e['action']=='pop':
                require(stack and stack[-1]==e['id'] and e['id'] not in closed,'native_scope_pop')
                stack.pop();closed.add(e['id'])
            else:raise ValueError('native_scope_action')
        require(opened==closed==set(om) and not any(stacks.values()),'native_scope_coverage')
        apis={}
        for a in report['apis']:
            c=a['correlation_id'];require(type(c) is int and c>0 and c not in apis,'duplicate_api')
            require(a['domain'] in ('runtime','driver') and 0<a['start']<=a['end'],'api_shape')
            apis[c]=a
        external={}
        for e in report['external']:
            c=e['correlation_id'];require(c not in external,'duplicate_external_join')
            require(e.get('kind')=='custom0','foreign_external_kind')
            external[c]=e['external_id']
            require(c in apis,'external_api_missing')
            require(e['external_id'] in om,'unknown_offer')
            require(apis[c]['thread']==om[e['external_id']]['thread'],'external_thread_mismatch')
        counts={i:Counter() for i in om};by_api=Counter();records=0
        for r in report['activities']:
            require(r['kind'] in KINDS and type(r['count']) is int and r['count']>0,'activity_shape')
            records+=r['count'];c=r['correlation_id'];a=apis.get(c);oid=external.get(c);o=om.get(oid)
            if a is None or o is None:f.append('unoffered_activity');continue
            if a['return_value']!=0:f.append('failed_api_has_activity')
            k=api_kind(a['name'])
            if k!=('graph' if r['graph_id'] else r['kind']):f.append('api_activity_kind')
            if bool(r['graph_id']) is not o.get('graph',False):f.append('offer_graph_scope')
            if r['context_id']!=o['context_id']:f.append('wrong_context')
            if not r['graph_id'] and r['stream_id']!=o['stream_id']:f.append('wrong_stream')
            if r['kind']=='memcpy' and r['copy_kind'] not in o.get('copy_kinds',[]):f.append('copy_direction')
            counts[oid][r['kind']]+=r['count'];by_api[c]+=r['count']
            joins.append(dict(offer=oid,correlation_id=c,kind=r['kind'],count=r['count']))
        require(records==report.get('activity_records'),'activity_census')
        peer_index={}
        for c,a in apis.items():
            if by_api[c]:peer_index.setdefault((a['thread'],external.get(c),api_kind(a['name'])),[]).append((c,a))
        claimed_wrappers=set()
        for c,a in apis.items():
            k=api_kind(a['name'])
            if k is None:m.append('unsupported_api:'+a['name']);continue
            if k=='host':
                # A pending event query (600) carries no GPU work and is not
                # completion; only the exact query API may report it.
                if not host_return_ok(a['name'],a['return_value']):f.append('host_api_failed')
                if by_api[c]:f.append('host_api_has_activity')
                continue
            if a['return_value']!=0:f.append('api_submission_failed');continue
            if by_api[c]:continue
            # Deduplicate only a unique nested runtime/driver wrapper of an
            # actually correlated submission on the SAME thread and offer.
            # These are API call nesting bounds, never GPU timestamp proximity.
            peers=[]
            for d,b in peer_index.get((a['thread'],external.get(c),k),[]):
                nested=(a['start']<=b['start'] and b['end']<=a['end']) or (b['start']<=a['start'] and a['end']<=b['end'])
                if (by_api[d] and a['domain']!=b['domain'] and a['thread']==b['thread']
                    and external.get(c) is not None and external.get(c)==external.get(d)
                    and api_kind(b['name'])==k and nested):peers.append(d)
            if len(peers)!=1 or peers[0] in claimed_wrappers:m.append('submission_without_unique_activity')
            else:
                claimed_wrappers.add(peers[0]);wrappers.append(dict(wrapper=c,activity_api=peers[0]))
        for i,o in om.items():
            if counts[i]!=Counter(o['expected']):f.append('offer_count_mismatch')
            families.setdefault(o['family'],Counter()).update(counts[i])
    except (KeyError,TypeError,ValueError) as exc:
        m.append(str(exc) if isinstance(exc,ValueError) else 'malformed_correlation_evidence')
    return dict(schema='s5-engine-accounting/1',verdict='FAIL' if f else 'INCONCLUSIVE' if m else 'PASS',
        failures=sorted(set(f)),missing=sorted(set(m)),families={k:dict(v) for k,v in families.items()},
        wrappers=wrappers,joined_records=sum(x['count'] for x in joins),joins=joins)


class CorrelationBridge:
    """Thread-local offers over one owned native observer subscriber.

    Configure after observer start, before the window. Does not drain, create
    streams, replay graphs, or change a runtime setting. Caller owns the final
    drain. Native failures remain sticky in the separate correlation receipt.
    """
    def __init__(self, session, *, epoch, generation, rank=0):
        import ctypes as C
        import threading
        if type(epoch) is not int or epoch<1 or type(generation) is not int or generation<1:
            raise ValueError('invalid correlation lifetime')
        self.session=session;self.epoch=epoch;self.generation=generation;self.rank=rank
        self.lock=threading.Lock();self.local=threading.local();self.offers=[];self.next_id=1
        lib=session.binding._lib;self.lib=lib;self.token=session._token
        lib.s5ro_correlation_enable.argtypes=[C.c_uint64];lib.s5ro_correlation_enable.restype=C.c_int
        lib.s5ro_correlation_thread_key.argtypes=[C.c_uint64,C.POINTER(C.c_uint32),C.POINTER(C.c_uint32)]
        lib.s5ro_correlation_thread_key.restype=C.c_int
        lib.s5ro_offer_push.argtypes=[C.c_uint64,C.c_uint64,C.c_uint32];lib.s5ro_offer_push.restype=C.c_int
        lib.s5ro_offer_pop.argtypes=[C.c_uint64,C.c_uint64];lib.s5ro_offer_pop.restype=C.c_int
        lib.s5ro_correlation_window.argtypes=[C.c_uint64,C.c_void_p,C.c_uint64,C.POINTER(C.c_uint64)];lib.s5ro_correlation_window.restype=C.c_int
        lib.s5ro_correlation_report.argtypes=[C.c_uint64,C.c_void_p,C.c_uint64,C.POINTER(C.c_uint64)];lib.s5ro_correlation_report.restype=C.c_int
        if lib.s5ro_correlation_enable(self.token):raise RuntimeError('correlation_enable')
        reserved=C.c_uint64()
        if lib.s5ro_reserved_bytes(self.token,C.byref(reserved)):raise RuntimeError('correlation_reservation')
        session._native_reserved=reserved.value

    def offer(self,family,expected,*,context_id,stream_id,graph=False,copy_kinds=()):
        from contextlib import contextmanager
        import threading
        @contextmanager
        def scope():
            stack=getattr(self.local,'stack',None)
            if stack is None:stack=[];self.local.stack=stack
            import ctypes as C
            key,system=C.c_uint32(),C.c_uint32()
            if self.lib.s5ro_correlation_thread_key(self.token,C.byref(key),C.byref(system)) or system.value!=threading.get_native_id():raise RuntimeError('native_thread_identity')
            with self.lock:
                if len(self.offers)>=32768:raise RuntimeError('offer_capacity')
                ident=self.next_id;self.next_id+=1
                row=dict(id=ident,epoch=self.epoch,generation=self.generation,rank=self.rank,
                    thread=key.value,system_thread=system.value,parent=stack[-1] if stack else 0,family=family,
                    expected=dict(expected),context_id=context_id,stream_id=stream_id,graph=graph,
                    copy_kinds=list(copy_kinds),closed=False,body_exception=False)
                self.offers.append(row)
            if self.lib.s5ro_offer_push(self.token,ident,self.generation):raise RuntimeError('offer_push')
            stack.append(ident)
            try:yield row
            except BaseException:
                row['body_exception']=True
                raise
            finally:
                if not stack or stack[-1]!=ident:raise RuntimeError('offer_stack')
                if self.lib.s5ro_offer_pop(self.token,ident):raise RuntimeError('offer_pop')
                stack.pop();row['closed']=True
        return scope()

    def window(self):
        """Native session bounds; end is zero until its drained window ends."""
        return self.session._report_call(self.lib.s5ro_correlation_window,(self.token,))

    def report(self):
        report=self.session._report_call(self.lib.s5ro_correlation_report,(self.token,),
                                         max_bytes=CORRELATION_REPORT_MAX_BYTES)
        metadata = getattr(self.session, 'native_read_metadata', None)
        read = metadata() if callable(metadata) else None
        return dict(native=report,offers=self.offers,
                    native_report_read=read,
                    grade=grade(report,self.offers,epoch=self.epoch,generation=self.generation))
