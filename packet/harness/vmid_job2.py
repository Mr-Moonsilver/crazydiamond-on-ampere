#!/usr/bin/env python3
"""Pinned observer/serializer, flat mixed graph, full8-size window, child residual."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
import time
import vmid_support as s

PLANS = {
    'mixed': dict(kernels=3, copies=3, memsets=1, child=0, replays=3),
    'capacity': dict(kernels=1386, copies=26, memsets=7, child=0, replays=40),
    'child': dict(kernels=3, copies=3, memsets=1, child=1, replays=3),
}


def expected(plan):
    # Observer intentionally counts root nodes only. Child leaf/container
    # support is a separate declared residual; never relabel it as complete.
    return {'kernel': plan['kernels'], 'memcpy': plan['copies'],
            'memset': plan['memsets'], 'graph': plan['child']}


def grade(r):
    rows = []
    for name, plan in PLANS.items():
        item = r.get('phases', {}).get(name, {})
        g = s.observation_grade(item, plan['replays'], expected(plan))
        wanted=[plan['kernels'],1,plan['copies']-2,1,plan['memsets'],plan['child'],plan['child'],0]
        if item.get('native_node_inventory') is None:
            g['missing'].append('copy_direction_inventory')
        elif item['native_node_inventory']!=wanted:
            g['failures'].append('wrong_native_node_inventory')
        if name == 'child':
            g['missing'].append('pinned_observer_has_no_recursive_child_placement_contract')
            g = s.verdict(g['failures'], g['missing'])
        rows.append(s.verdict(g['failures'],g['missing']))
    g = s.combine(rows)
    g = {**g, **s.verdict(g['failures']+r.get('failures',[]),g['missing']+r.get('missing',[]))}
    g['q17_flat_node_families'] = s.combine(rows[:2])
    return g


class GraphBinding:
    def __init__(self, path):
        self.lib = C.CDLL(str(path))
        u, i, pu = C.c_uint64, C.c_int, C.POINTER(C.c_uint64)
        self.lib.vmg_abi.argtypes = []
        self.lib.vmg_abi.restype = i
        self.lib.vmg_create.argtypes = [u]*6 + [i]*5 + [pu, pu]
        self.lib.vmg_create.restype = i
        self.lib.vmg_replay.argtypes = [u]
        self.lib.vmg_replay.restype = i
        self.lib.vmg_close.argtypes = [u]
        self.lib.vmg_close.restype = i
        self.lib.vmg_inventory.argtypes = [u,C.POINTER(i),i]
        self.lib.vmg_inventory.restype = i
        if self.lib.vmg_abi() != 1:
            raise RuntimeError('graph_helper_abi')


def phase(torch, ro, gc, native, receipt, plan):
    r = {'plan': plan, 'replays': 0, 'failures': [], 'missing': [], 'started_wall_s': time.time()}
    session = split = None
    owner = 0
    window = None
    try:
        binding = ro.ReplayBinding(str(native/'flash_s5_replay_observer.so'),
            receipt['files']['flash_s5_replay_observer.so'], receipt['files']['flash_s5_replay_observer.cpp'], s.CUPTI,
            cupti_library_path='/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcupti.so.13')
        # EXACT q17b-full9/default ring sizes, including the capacity phase.
        limits = ro.ObserverLimits()
        session = ro.ObserverSession(binding, limits)
        session.start()
        r['observer_host_reserved_bytes'] = session._native_reserved
        r['limits'] = limits.as_record()
        split = gc.open_split(0, env={'FLASH_S5_DECODE_SMS':'44'}, torch_module=torch,
                              binding=gc.DriverBinding(str(native/'flash_s5_green.so')))
        r['split'] = dict(decode=split.decode_sms, prefill=split.prefill_sms, total=split.total_sms)
        # All native GPU and pinned operands are allocated by torch and kept
        # alive until graph close. Only 5 x 4096 bytes of data storage.
        src = torch.full((1024,), 7, dtype=torch.int32, pin_memory=True)
        dst = torch.empty((1024,), dtype=torch.int32, pin_memory=True)
        a, b, z = [torch.empty((1024,), dtype=torch.int32, device='cuda') for _ in range(3)]
        torch.cuda.synchronize()
        graphlib = GraphBinding(native/'vmid_graph_probe.so')
        own, graph = C.c_uint64(), C.c_uint64()
        s.checked(graphlib.lib.vmg_create(split.decode_stream_token, a.data_ptr(), b.data_ptr(), z.data_ptr(),
                  src.data_ptr(), dst.data_ptr(), 1024, plan['kernels'], plan['copies'], plan['memsets'],
                  plan['child'], C.byref(own), C.byref(graph)), 'graph_create')
        owner = own.value
        inventory=(C.c_int*8)()
        s.checked(graphlib.lib.vmg_inventory(owner,inventory,8),'graph_inventory')
        r['native_node_inventory']=list(inventory)
        r['inventory_fields']=['kernel','h2d','d2d','d2h','memset','child','child_kernel','other']
        r['capture_instantiated_before_enable'] = True
        lib, token = binding._lib, session._token
        # Retained graph definition bytes stay private; report only length and equality.
        try:
            before = s.graph_bytes(lib, graph.value)
            r['serialized_bytes'] = len(before)
            r['serialized_stable'] = before == s.graph_bytes(lib, graph.value)
            r['serializer_status'] = 'covered'
        except RuntimeError:
            before = None
            r['serializer_status'] = 'unsupported_or_refused'
            r['missing'].append('serializer')
        s.checked(lib.s5ro_bind_context(token, split.decode_context_id, split.decode_sms), 'bind_context')
        count = C.c_uint32()
        s.checked(lib.s5ro_bind_graph(token, graph.value, 0, C.byref(count)), 'bind_graph')
        r['static'] = session._report_call(lib.s5ro_static, (token,))
        ro._validate_static(r['static'])
        start = C.c_uint64()
        s.checked(lib.s5ro_begin_window(token, 1, C.byref(start)), 'begin_window')
        window = start.value
        for _ in range(plan['replays']):
            s.checked(graphlib.lib.vmg_replay(owner), 'replay')
            r['replays'] += 1
        split.decode_stream.synchronize()
        r['window'] = session._report_call(lib.s5ro_end_window, (token, window, 1), retry=False)
        window = None
        ro._validate_window(r['window'], limits, 1)
        if before is not None:
            r['serialized_stable'] &= before == s.graph_bytes(lib, graph.value)
        # Source reset in every replay => same exact value, regardless of replay count.
        # Child runs after D2H and increments z; its execution is checked too.
        r['output_exact'] = bool(torch.all(dst == 7+plan['kernels']).item()
                                 and torch.all(b == 7+plan['kernels']).item()
                                 and torch.all(z == plan['child']).item())
        r['allocator_peak_reserved'] = torch.cuda.max_memory_reserved(0)
        if r['allocator_peak_reserved'] > 64*1024**2:
            r['failures'].append('unexpected_allocator_growth')
        time.sleep(1)
    except Exception as exc:
        r['missing'].append('phase_exception')
        r['error_type'] = type(exc).__name__
        r['error_code'] = getattr(exc, 'code', None)
        if isinstance(exc,RuntimeError) and exc.args and str(exc.args[0]).split(':')[0] in {
                'graph_create','graph_inventory','bind_context','bind_graph','begin_window','replay','graph_helper_abi'}:
            r['error_code']=str(exc.args[0])
    finally:
        errors = []
        try:
            torch.cuda.synchronize()
            if session and window is not None:
                session._report_call(session.binding._lib.s5ro_end_window, (session._token, window, 1), retry=False)
        except Exception: errors.append('drain_window')
        try:
            if session: session.close()
        except Exception: errors.append('observer_close')
        try:
            if owner: s.checked(graphlib.lib.vmg_close(owner), 'graph_close')
        except Exception: errors.append('graph_close')
        try:
            if split: split.close()
        except Exception: errors.append('split_close')
        r['cleanup_ok'] = not errors
        r['cleanup_errors'] = errors
        r['ended_wall_s'] = time.time()
    return r


def probe(modules, native, out):
    r = {'schema':'vm<vm-id>-job2-probe/1', 'phases':{}, 'image':s.IMAGE, 'failures':[], 'missing':[],
         'scope':'mixed flat node types and full8-size window; child is a separate unsupported-observer residual',
         'skipped':{'conditional':'serializer and observer have no conditional contract; child alternative exercised',
                    'collective_semantics':'single GPU synthetic kernels are not TP/PP collectives'}}
    try:
        import torch
        torch.cuda.set_per_process_memory_fraction(0.125, 0)
        if torch.cuda.device_count()!=1 or torch.cuda.get_device_capability(0)!=(8,6):
            raise RuntimeError('one_sm86_device_required')
        sys.path.insert(0,str(modules))
        import replay_observer as ro
        import green_context as gc
        receipt=json.loads((native/'q17-helper-build.json').read_text())
        if receipt['image']!=s.IMAGE or receipt['cupti_library_sha256']!=s.CUPTI:
            raise RuntimeError('identity')
        for name,digest in receipt['files'].items():
            if s.sha(native/name)!=digest: raise RuntimeError('pin')
        for name in ('flash_s5_green.cu','flash_s5_replay_observer.cpp','flash_s5_graph_bytes.cpp'):
            if s.sha(modules/'native'/name)!=receipt['files'][name]: raise RuntimeError('source_pin')
        r['build']=receipt
        r['software']=dict(torch=str(torch.__version__),cuda=torch.version.cuda)
        for name,plan in PLANS.items():
            item=phase(torch,ro,gc,native,receipt,plan)
            r['phases'][name]=item
            s.write(out/(name+'.json'),item)
            if not item.get('cleanup_ok'):
                break  # no new native session after an unproven cleanup
    except Exception as exc:
        r['error_type']=type(exc).__name__
        if isinstance(exc,RuntimeError) and exc.args and str(exc.args[0]) in ('identity','pin','source_pin'):
            r['failures'].append(str(exc.args[0]))
        else:r['missing'].append('setup_exception')
    r['grade']=grade(r)
    s.write(out/'probe.json',r)
    return r['grade']['rc']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('probe','monitor','finalize'),required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--modules',type=Path);p.add_argument('--native',type=Path)
    p.add_argument('--process-rc',type=int,default=2)
    a=p.parse_args()
    if a.mode=='monitor': return s.monitor(a.out,'flashnext-job2')
    if a.mode=='finalize': return s.finalize(a.out,a.process_rc,grade,'job2')
    return probe(a.modules,a.native,a.out)


if __name__=='__main__': sys.exit(main())
