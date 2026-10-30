"""Rank-1, native-window launch fences and capture/live pointer receipts.

No engine source changes. No tensor readback/allocation. Pointer inventories are
observed references, not a claim that every inventoried tensor is a graph input.
"""
import contextlib
import functools
import inspect
import itertools
import json
import os
import sys
import threading
import traceback
import weakref

import native30_host as host
import native30_localize as localize

_TLS = threading.local()
_RUNNER = None
_RANK = None
_GRAPHS = weakref.WeakKeyDictionary()
_IDS = itertools.count(1)
_INSTALLED = False
MAX_LAUNCH_RECORDS = 200000
MAX_BYTES = 128*1024**2
_BYTES = 0
MAX_POINTERS = 2048
_RECORDS = 0
_LIMIT = False
_LIVE_METADATA = None


def active():
    if os.environ.get('FLASH_NATIVE33_DIAG') != '1' or getattr(_TLS, 'capturing', None) is not None:
        return None
    ctx = getattr(host._TLS, 'context', None)
    if not ctx or not ctx.get('state') or not ctx['state'].get('active'):
        return None
    if ctx['state'].get('rank') != 1 or ctx['fields'].get('label') in ('PP_SEND', 'PP_RECEIVE'):
        return None
    return ctx


def emit(event, **fields):
    global _RECORDS, _LIMIT, _BYTES
    charge=len(json.dumps(fields,sort_keys=True).encode())+512
    if _RECORDS >= MAX_LAUNCH_RECORDS or _BYTES+charge>MAX_BYTES or charge>4096:
        if not _LIMIT:
            _LIMIT = True
            localize.emit('native33-receipt-limit', rank=1, records=_RECORDS,
                          coverage_complete=False)
        return
    _RECORDS += 1; _BYTES += charge
    localize.emit('native33-' + event, rank=1, **fields)


def frames(exc):
    return [dict(file=f.filename[-192:], line=f.lineno, function=f.name[-96:])
            for f in traceback.extract_tb(exc.__traceback__)[-8:]]


def fence():
    import torch
    # The current stream only: never wait on an unposted native PP receive.
    torch.cuda.current_stream().synchronize()


def launch(original, name, kind, args, kwargs, source=None):
    ctx = active()
    transfer = kind == 'torch-operator' and any(word in name.lower() for word in ('send','recv','receive'))
    if ctx is None or transfer:
        return original(*args, **kwargs)
    identity = next(_IDS)
    fields = {k: v for k, v in ctx['fields'].items() if k != 'rank'}
    fields.update(launch_id=identity, operator=name[:256], launch_kind=kind, kernel_source=source,
                  kernel_name=name[:256] if kind == 'triton-jit' else None,
                  kernel_coverage='named-triton' if kind == 'triton-jit' else 'opaque-operator')
    emit('launch-begin', **fields)
    try:
        result = original(*args, **kwargs)
        fence()
    except BaseException as exc:
        emit('launch-exception', error_type=type(exc).__name__, frames=frames(exc), **fields)
        raise
    emit('launch-end', **fields)
    return result


def patch(owner, name, callback):
    original = getattr(owner, name)
    if getattr(original, '_native33_wrapped', False):
        return
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return callback(original, args, kwargs)
    wrapped._native33_wrapped = True
    setattr(owner, name, wrapped)


def pointers(roots):
    values = {}; seen = set(); nodes = 0; complete = True
    def visit(obj, path, depth):
        nonlocal nodes, complete
        nodes += 1
        if nodes > 16000 or len(values) >= MAX_POINTERS:
            complete = False; return
        if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
            return
        if callable(getattr(obj, 'data_ptr', None)) and hasattr(obj, 'shape'):
            try:
                values[path] = dict(data_ptr=hex(int(obj.data_ptr())),
                                    shape=list(obj.shape), dtype=str(obj.dtype),
                                    device=str(obj.device))
            except Exception:
                values[path] = dict(available=False)
            return
        if depth > 8:complete = False; return
        if id(obj) in seen:return
        seen.add(id(obj))
        if isinstance(obj, dict):
            for key, value in list(obj.items())[:512]:
                if isinstance(key, (str, int)):visit(value, path + '.' + str(key), depth + 1)
        elif isinstance(obj, (tuple, list)):
            for i, value in enumerate(obj[:512]):visit(value, path + '[' + str(i) + ']', depth + 1)
        elif hasattr(obj, '__dict__'):
            for key, value in vars(obj).items():
                # Exclude model/weights and callbacks. Only the named state roots.
                if key not in ('model', 'runner', 'vllm_config') and not callable(value):
                    visit(value, path + '.' + key, depth + 1)
    for name, value in roots.items():visit(value, name, 0)
    return dict(values=values, complete=complete, nodes=nodes,
                scope='bounded observed pointers; repeated object aliases elided; graph-read-set not proven')


def runner_roots():
    runner = _RUNNER() if _RUNNER else None
    if runner is None:return {}
    return {name: getattr(runner, name, None) for name in
            ('block_tables', 'req_states', 'input_buffers', 'model_state')}


def descriptor(desc):
    out = {name: getattr(desc, name, None) for name in
           ('num_tokens', 'num_reqs', 'num_ubatches', 'max_query_len', 'num_active_loras')}
    mode = getattr(desc, 'cg_mode', None)
    out['mode'] = getattr(mode, 'name', str(mode))
    return out


def capture_begin(original, args, kwargs):
    result = original(*args, **kwargs)
    graph = args[0]
    _TLS.capturing = weakref.ref(graph)
    if _RANK == 1:
        roots = runner_roots()
        roots.update(getattr(_TLS, 'forward_roots', {}))
        _GRAPHS[graph] = dict(capture_id=next(_IDS), key=getattr(_TLS, 'capture_key', None),
                              snapshot=pointers(roots), replay_seen=False)
    return result


def capture_end(original, args, kwargs):
    try:return original(*args, **kwargs)
    finally:_TLS.capturing = None


def capture_factory(original, args, kwargs):
    if _RANK != 1:return original(*args, **kwargs)
    factory = args[1] if len(args) > 1 else kwargs['create_forward_fn']
    def tracked_factory(desc, *a, **k):
        fn = factory(desc, *a, **k)
        @functools.wraps(fn)
        def forward(*fa, **fk):
            prior_key = getattr(_TLS, 'capture_key', None)
            prior_roots = getattr(_TLS, 'forward_roots', {})
            closure = inspect.getclosurevars(fn).nonlocals
            _TLS.capture_key = descriptor(desc)
            _TLS.forward_roots = {name: closure[name] for name in
                                  ('model_inputs', 'attn_metadata', 'slot_mappings') if name in closure}
            ref = getattr(_TLS, 'capturing', None)
            graph = ref() if ref else None
            if graph is not None and graph in _GRAPHS:
                roots = runner_roots(); roots.update(_TLS.forward_roots)
                _GRAPHS[graph].update(key=descriptor(desc), snapshot=pointers(roots))
            try:return fn(*fa, **fk)
            finally:_TLS.capture_key = prior_key; _TLS.forward_roots = prior_roots
        return forward
    if len(args) > 1:return original(args[0], tracked_factory, *args[2:], **kwargs)
    return original(*args, **dict(kwargs, create_forward_fn=tracked_factory))


def compare_graph(graph):
    record = _GRAPHS.get(graph)
    key = getattr(_TLS, 'replay_key', None)
    if record is None:
        emit('graph-identity', graph_id=hex(id(graph)), capture_key=key,
             capture_inventory_available=False, kernel_name=None,
             coverage='opaque-captured-graph; capture association missing')
        return
    roots = runner_roots()
    live = pointers(roots)
    live['values'].update((_LIVE_METADATA or {}).get('values', {}))
    for name in ('input_ids', 'positions'):
        value = live['values'].get('input_buffers.' + name)
        if value is not None:live['values']['model_inputs.' + name] = value
    captured = record['snapshot']
    emit('graph-identity', graph_id=hex(id(graph)), capture_id=record['capture_id'],
         capture_key=record['key'], replay_key=key, capture_inventory_available=True,
         capture_complete=captured['complete'], live_complete=live['complete'],
         live_metadata_step=(_LIVE_METADATA or {}).get('step'),
         coverage='opaque-captured-graph; inventoried references not proven kernel read-set')
    changed = missing = matched = 0
    for path, before in captured['values'].items():
        after = live['values'].get(path)
        different = after is not None and before.get('data_ptr') != after.get('data_ptr')
        missing += after is None; changed += different
        matched += after is not None and not different
        if not record['replay_seen'] or different or after is None:
            emit('graph-pointer', capture_id=record['capture_id'], path=path,
                 captured=before, live=after, pointer_changed=different if after else None,
                 association='same inventory attribute path; no device readback',
                 binding_kind='forward-closure-reference' if path.startswith(('model_inputs.','attn_metadata.','slot_mappings.')) else 'runner-inventory-unproven-graph-read')
    record['replay_seen'] = True
    emit('graph-pointer-summary', capture_id=record['capture_id'], matched=matched,
         changed=changed, missing_live=missing, captured_count=len(captured['values']))


def graph_replay(original, args, kwargs):
    if active():compare_graph(args[0])
    return launch(original, 'torch.cuda.CUDAGraph.replay', 'cuda-graph-replay', args, kwargs)


def full_replay(original, args, kwargs):
    if active() is None:return original(*args, **kwargs)
    old = getattr(_TLS, 'replay_key', None)
    _TLS.replay_key = descriptor(args[1] if len(args) > 1 else kwargs['desc'])
    try:return original(*args, **kwargs)
    finally:_TLS.replay_key = old


def register_runner(original, args, kwargs):
    global _RUNNER, _RANK
    runner = getattr(args[0], 'model_runner', None)
    if runner is not None:_RUNNER = weakref.ref(runner); _RANK = args[0].rank
    result = original(*args, **kwargs)
    runner = getattr(args[0], 'model_runner', None)
    if runner is not None:_RUNNER = weakref.ref(runner); _RANK = args[0].rank
    return result


def metadata(original, args, kwargs):
    global _LIVE_METADATA
    result = original(*args, **kwargs)
    ctx = active()
    if ctx and ctx['fields'].get('label') == 'DECODE_STEP':
        _LIVE_METADATA = pointers(dict(attn_metadata=result))
        _LIVE_METADATA['step'] = ctx['fields'].get('step')
    return result


def instrument(module):
    name = module.__name__
    if name.endswith('.gpu_worker'):patch(module.Worker, 'load_model', register_runner)
    elif name.endswith('.cudagraph_utils'):
        patch(module.CudaGraphManager, 'capture', capture_factory)
        patch(module.CudaGraphManager, 'run_fullgraph', full_replay)
    elif name.endswith('.attn_utils'):
        old = module.build_attn_metadata
        patch(module, 'build_attn_metadata', metadata)
        # Existing from-import aliases must see the same receipt wrapper.
        for loaded in list(sys.modules.values()):
            if loaded is not None and getattr(loaded, '__dict__', {}).get('build_attn_metadata') is old:
                setattr(loaded, 'build_attn_metadata', module.build_attn_metadata)


def triton_launch(old, args, kwargs):
    if kwargs.get('warmup'):return old(*args,**kwargs)
    fn=args[0].fn; code=getattr(fn,'__code__',None)
    source=dict(file=code.co_filename[-192:],line=code.co_firstlineno) if code else None
    return launch(old,getattr(fn,'__module__','')+'.'+getattr(fn,'__qualname__','<triton>'),
                  'triton-jit',args,kwargs,source=source)


def exception_emit(previous,event,**fields):
    runner=_RUNNER() if _RUNNER else None
    state=host._RUNNERS.get(runner) if runner is not None else None
    if event=='exception' and fields.get('rank')==1 and state and state.get('active'):
        exc=sys.exc_info()[1]
        if exc is not None:
            emit('boundary-exception',frames=frames(exc),**{k:v for k,v in fields.items() if k!='rank'})
    return previous(event,**fields)


def install():
    global _INSTALLED
    if _INSTALLED or os.environ.get('FLASH_NATIVE33_DIAG') != '1':return
    if os.environ.get('CUDA_LAUNCH_BLOCKING') == '1':raise RuntimeError('native33-global-launch-blocking-forbidden')
    import torch
    from triton.runtime.jit import JITFunction
    import native30_policy
    native30_policy.install()
    patch(torch.cuda.CUDAGraph, 'capture_begin', capture_begin)
    patch(torch.cuda.CUDAGraph, 'capture_end', capture_end)
    patch(torch.cuda.CUDAGraph, 'replay', graph_replay)
    patch(JITFunction, 'run', triton_launch)
    for cls in (torch._ops.OpOverload, torch._ops.OpOverloadPacket):
        patch(cls, '__call__', lambda old, a, k: launch(old,
              str(getattr(a[0], '_qualified_op_name', getattr(a[0], '__name__', '<torch-op>'))),
              'torch-operator', a, k))
    old_instrument = localize.instrument
    @functools.wraps(old_instrument)
    def combined(module):old_instrument(module); instrument(module)
    localize.instrument = combined
    names = {'vllm.v1.worker.gpu_worker', 'vllm.v1.worker.gpu.cudagraph_utils',
             'vllm.v1.worker.gpu.attn_utils'}
    localize._TARGETS.update(names)
    for name in sorted(names):
        if name in sys.modules:instrument(sys.modules[name])
    previous_emit=localize.emit
    def observed_emit(event,**fields):return exception_emit(previous_emit,event,**fields)
    localize.emit=observed_emit
    _INSTALLED = True
    localize.emit('native33-installed', target_rank=1, activation='existing row4 admission window',
                  synchronization='current stream after named launches; PP transfers excluded',
                  max_launch_records=MAX_LAUNCH_RECORDS, max_pointers=MAX_POINTERS)
