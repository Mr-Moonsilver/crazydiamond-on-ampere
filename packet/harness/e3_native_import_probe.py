#!/usr/bin/env python3
"""Actual-image, exact-mount import proof. No GPU owners, mocks or ready flags.

Deferred clauses execute as import-only projections in the real module package;
GPU bodies/fences are not called. Pure provider import helpers are called for
real, including the exact row1 helper used after its live lifecycle fences.
"""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def error(exc):
    return dict(type=type(exc).__name__,message=str(exc).replace('\n',' ')[:512],
                stack=[dict(file=Path(f.filename).name,function=f.name,line=f.lineno)
                       for f in traceback.extract_tb(exc.__traceback__)[-6:]])


def check(graph):
    result=dict(schema='native8-real-overlay-import-proof/1',modules=[],deferred=[],
                providers=[],failures=[],missing=[],no_mocks=True,no_fixture_shims=True,
                gpu_owners_constructed=False,hardware_qualified=False,release_qualified=False)
    result['gpu_device_nodes']=[p.name for p in Path('/dev').glob('nvidia[0-9]*')]
    if result['gpu_device_nodes']:
        result.update(verdict='FAIL',rc=1,counts={},cuda_initialized=False,native7_negative_rejected=False)
        result['failures'].append('GPU-device-exposed-in-CPU-proof')
        return result
    assert graph['schema']=='native8-mounted-runtime-imports/1' and graph['verdict']=='PASS'
    forbidden=[p for p in sys.path if p and
               (p.endswith(('/e3','/e2','/e3_builder','/e2_builder','/s5_engine'))
                or 'harness/' in p)]
    if forbidden:result['failures'].append('builder-source-search-path-present')
    records={x['module']:x for x in graph['modules']
             if x['active_mount'] and x['module'] not in {b['module'] for b in graph['builder_only_proof']}}
    # The real loader imports model first. qsa -> model is a deferred constructor
    # read; importing qsa first would create an artificial circular-import test.
    order=['vllm.models.qwen4_exp.nvidia.model']+sorted(records)
    loaded={}
    for name in dict.fromkeys(order):
        if name not in records:continue
        spec=records[name];row=dict(module=name,destination=spec['destination'])
        try:
            module=importlib.import_module(name)
            origin=Path(module.__file__).resolve()
            assert origin==Path(spec['destination']).resolve(),'module-origin-differs'
            assert sha(origin)==spec['sha256'],'mounted-source-differs'
            loaded[name]=module;row.update(verdict='PASS',source_sha256=sha(origin))
        except Exception as exc:
            row.update(verdict='INCONCLUSIVE',error=error(exc));result['missing'].append('real-module-import:'+name)
        result['modules'].append(row)
    for edge in graph['edges']:
        if not edge['runtime_required'] or not edge['deferred'] or edge['module'] not in records:continue
        row={k:edge[k] for k in ('edge_id','module','scope','line','expression','qualified_target','form')}
        module=loaded.get(edge['module'])
        if module is None:
            row['verdict']='INCONCLUSIVE';result['missing'].append('deferred-owner:'+str(edge['edge_id']))
        else:
            try:
                if edge['form'] in ('from','import'):
                    tree=ast.parse(edge['expression'])
                    assert len(tree.body)==1 and isinstance(tree.body[0],(ast.Import,ast.ImportFrom))
                    # A private namespace copy prevents imported aliases from
                    # replacing any binding in the actual loaded runtime module.
                    namespace=dict(module.__dict__)
                    exec(compile(tree,'<actual-deferred-import>', 'exec'),namespace,namespace)
                else:
                    importlib.import_module(edge['qualified_target'])
                row['verdict']='PASS'
            except Exception as exc:
                row.update(verdict='INCONCLUSIVE',error=error(exc));result['missing'].append('deferred-import:'+str(edge['edge_id']))
        result['deferred'].append(row)
    # These are actual source helpers, not test replacements or injected owners.
    calls=[('vllm.v1.flash_e3.s5_startup','row1_qualification_types',()),
           ('vllm.v1.flash_e3.native_resources','_import_forward_context',()),
           ('vllm.v1.flash_e3.native_resources','_workspace_module',()),
           ('vllm.v1.flash_e3.native_resources','response_success',()),
           ('vllm.v1.flash_e3.native_qualification_frontend','_peer_module',('native_qualification',)),
           ('vllm.v1.flash_e3.native_qualification_frontend','_peer_module',('native_overlap_control',))]
    for module_name,name,args in calls:
        row=dict(module=module_name,provider=name,args=list(args))
        try:
            value=getattr(importlib.import_module(module_name),name)(*args)
            if name=='row1_qualification_types':
                backend,qualifier=value
                assert backend.__module__=='vllm.v1.flash_e2.exactness'
                assert qualifier.__module__=='vllm.v1.flash_e3.row1_oracle'
                obj=backend()  # Actual constructor only imports real torch.
                assert obj.torch is sys.modules['torch']
                row['returned_owners']=[backend.__module__,qualifier.__module__]
                row['real_torch_backend_constructed']=True
            elif name=='_peer_module':
                assert value.__name__=='vllm.v1.flash_e3.'+args[0]
                row['returned_owner']=value.__name__
            row['verdict']='PASS'
        except Exception as exc:
            row.update(verdict='INCONCLUSIVE',error=error(exc));result['missing'].append('provider-import:'+name)
        result['providers'].append(row)
    # This guard requires a real model's GPU bind_kv_cache. Import-only CPU
    # execution must preserve the refusal, never manufacture a UID tensor.
    resources=importlib.import_module('vllm.v1.flash_e3.native_resources')
    context=resources._import_forward_context()
    try:
        resources._forward_context_module()
    except resources.NativeResourcesError as exc:
        result['unbound_forward_context_refused']=(
            not hasattr(context,'QSA_REQ_UID') and 'QSA_REQ_UID' in str(exc))
    else:
        result['unbound_forward_context_refused']=False
    if not result['unbound_forward_context_refused']:
        result['failures'].append('unbound-model-context-not-refused')
    # Counterexample in the identical mounted namespace: no bare exactness
    # may become available through a test-only source path or stale sys.modules.
    try:
        exec('from exactness import TorchExactBackend',
             dict(importlib.import_module('vllm.v1.flash_e3.s5_startup').__dict__))
    except ModuleNotFoundError as exc:
        result['native7_negative_rejected']=exc.name=='exactness'
    else:result['native7_negative_rejected']=False
    if not result['native7_negative_rejected']:
        result['failures'].append('bare-native7-import-not-rejected')
    torch=sys.modules.get('torch');cuda=getattr(torch,'cuda',None)
    result['cuda_initialized']=bool(cuda is not None and cuda.is_initialized())
    if result['cuda_initialized']:result['failures'].append('CUDA-context-in-CPU-proof')
    result['verdict']='FAIL' if result['failures'] else 'INCONCLUSIVE' if result['missing'] else 'PASS'
    result['rc']=1 if result['failures'] else 2 if result['missing'] else 0
    result['counts']={k:len(result[k]) for k in ('modules','deferred','providers')}
    result['scope']='exact launcher namespace and all enumerated deferred import clauses; actual pure providers only; no GPU body/lifecycle/performance proof'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    try:
        j=check(json.loads(a.graph.read_text()))
    except Exception as exc:
        torch=sys.modules.get('torch');cuda=getattr(torch,'cuda',None)
        j=dict(schema='native8-real-overlay-import-proof/1',verdict='INCONCLUSIVE',rc=2,
               counts={},failures=[],missing=['probe_exception'],error=error(exc),
               cuda_initialized=bool(cuda is not None and cuda.is_initialized()),
               native7_negative_rejected=False,no_mocks=True,no_fixture_shims=True,
               gpu_owners_constructed=False,hardware_qualified=False,release_qualified=False)
    j['graph_sha256']=sha(a.graph);j['probe_sha256']=sha(__file__)
    with a.out.open('x') as f:json.dump(j,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:j[k] for k in ('verdict','rc','counts','failures','missing','cuda_initialized','native7_negative_rejected')}))
    return j['rc']


if __name__=='__main__':raise SystemExit(main())
