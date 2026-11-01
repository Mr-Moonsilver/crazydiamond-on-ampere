#!/usr/bin/env python3
"""Fail-closed path and AST freeze vs native39 1c334bc, source and generated overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from native37_robust_freeze import prove
from build_native30_held import verify
ROOT=Path(__file__).resolve().parents[1]
PARENT='681030c'
DECLARATION=ROOT/'tools/native41_autoarm_declarations.json'
RUNTIME={'e3/scheduler_hooks.py','e3/engine_driver.py','e3/executor_hooks.py','e3/s5_startup.py','e3/native_overlap_worker.py','e3/decode_probe.py','e3/native_overlap_control.py','e3/native_overlap_session.py','run-flash-s5-test.sh'}
ADDED={'e3/native_auto_arm.py'}
META={'SHA256SUMS','S5-OVERLAP-GENERATION.json','e3/native-implementation-closure.json',
      'e3/native-overlap-readiness.json','e3/decode-probe-evidence.json'}
def show(path):return subprocess.check_output(['git','show',PARENT+':'+path],cwd=ROOT,text=True)
def paths():return subprocess.check_output(['git','diff','--name-status',PARENT,'--','patches/s5_engine','harness','tools'],cwd=ROOT,text=True).splitlines()
def check_paths(actual,d):assert set(actual)==set(d['paths']),sorted(set(actual)^set(d['paths']))
def source_level():
    d=json.loads(DECLARATION.read_text());rows=paths();check_paths(rows,d)
    all_rows=subprocess.check_output(['git','diff','--name-status',PARENT],cwd=ROOT,text=True).splitlines()
    ancillary=set(all_rows)-set(rows)
    assert ancillary<=set(d['ancillary_paths']),sorted(ancillary-set(d['ancillary_paths']))
    modified={row.split('\t',1)[1] for row in rows if row.startswith('M\t') and row.endswith('.py')}
    assert modified==set(d['definitions'])
    defs={p:prove(show(p),(ROOT/p).read_text(),p,names) for p,names in d['definitions'].items()}
    compiled=[]
    for row in rows:
        name=row.split('\t',1)[1]
        if name.endswith('.py'):compile((ROOT/name).read_bytes(),name,'exec');compiled.append(name)
    return dict(default_off=default_off(),parent=PARENT,paths=rows,ancillary_paths=sorted(ancillary),definitions=defs,compiled=compiled,skips=0,
                declaration_sha256=hashlib.sha256(DECLARATION.read_bytes()).hexdigest())
def overlay_level(parent,candidate):
    before,after=verify(parent),verify(candidate)
    assert set(after)==set(before)|ADDED,'overlay file set changed'
    changed={n for n in before if before[n]!=after[n]}|{'SHA256SUMS'}|ADDED
    assert changed==RUNTIME|META|ADDED,sorted(changed^(RUNTIME|META|ADDED))
    d=json.loads(DECLARATION.read_text())['overlay_definitions']
    assert set(d)==RUNTIME-{'run-flash-s5-test.sh'}
    defs={p:prove((parent/p).read_text(),(candidate/p).read_text(),p,d[p]) for p in sorted(d)}
    launcher=(parent/'run-flash-s5-test.sh').read_text()
    assert launcher.count('e3/s5_startup.py ')==1
    assert (candidate/'run-flash-s5-test.sh').read_text()==launcher.replace('e3/s5_startup.py ','e3/s5_startup.py e3/native_auto_arm.py ',1),'launcher changed beyond forced module declaration'
    native=[p for p in before if p.endswith('.so') or '/native30_plugins/' in p or p=='qsa.py']
    assert native and all(before[n]==after[n] for n in native)
    return dict(parent=str(parent),changed=sorted(changed),definitions=defs,
                model_and_plugin_byte_identical=True,native_libraries_byte_identical=True)
def negative_controls():
    d=json.loads(DECLARATION.read_text());p='patches/s5_engine/scheduler_hooks.py'
    def refused(fn):
        try:fn()
        except AssertionError:return 'REFUSED'
        raise AssertionError('undeclared change accepted')
    return {'undeclared-function':refused(lambda:prove(show(p),(ROOT/p).read_text()+'\ndef native41_undeclared():\n pass\n',p,d['definitions'][p])),
            'undeclared-path':refused(lambda:check_paths(paths()+['A\tharness/native41_undeclared.py'],d)),
            'undeclared-overlay':refused(lambda:prove('x=1\n','x=2\n','core/scheduler.py',[]))}

def default_off():
    import ast,copy
    def function(text,name):
        tree=ast.parse(text)
        if '.' in name:
            cls,method=name.split('.')
            tree=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
            return copy.deepcopy(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==method))
        return copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
    def key(node):
        node=copy.deepcopy(node);node.name='f'
        return ast.dump(node)
    specs={
      'engine_driver.py': [('NativeSegmentDriver._submit_control','NativeSegmentDriver.submit_control','same'),
         ('NativeSegmentDriver._status_record_base','NativeSegmentDriver._status_record','same'),
         ('NativeSegmentDriver._arm_admission','NativeSegmentDriver._arm_admission','arm')],
      'scheduler_hooks.py':[('PrefillAdmissionController._native_admission_disabled','PrefillAdmissionController._native_admission_disabled','intent'),
         ('PrefillAdmissionController._admit_observed','PrefillAdmissionController._admit_observed','policy')],
      'executor_hooks.py':[('native_work_pending','native_work_pending','work'),('engine_idle_tick','engine_idle_tick','policy'),('before_pause','before_pause','policy')],
      's5_startup.py':[('install_engine','install_engine','startup')],
      'native_overlap_worker.py':[('_native_inventory','_native_inventory','worker')],
      'native_overlap_control.py':[('NativeControlClient._call','NativeControlClient._call','client')],
      'native_overlap_session.py':[('NativeOverlapSession.execute','NativeOverlapSession.execute','session')],
    }
    result={}
    for file,entries in specs.items():
        old=show('patches/s5_engine/'+file);new=(ROOT/'patches/s5_engine'/file).read_text()
        for current,previous,mode in entries:
            node=function(new,current);projected=[]
            for n in node.body:
                text=ast.unparse(n)
                if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('policy','auto'):continue
                if isinstance(n,ast.If):
                    condition=ast.unparse(n.test)
                    if mode=='arm' and condition=='not auto':projected+=n.body;continue
                    if mode=='arm' and condition=='auto':continue
                    if mode=='intent' and 'auto-prefill' in condition:
                        assert len(n.orelse)==1 and isinstance(n.orelse[0],ast.If)
                        projected+=n.orelse;continue
                    if mode=='policy' and condition.startswith('policy is not None'):continue
                    if mode=='startup' and condition.startswith('auto '):continue
                    if mode=='worker' and 'FLASH_S5_AUTO_ARM' in condition:continue
                    if mode=='session' and 'release_auto_arm' in condition:continue
                    if mode=='client' and condition=='action not in ACTIONS':
                        assert len(n.body)==1 and isinstance(n.body[0],ast.If)
                        n.body=n.body[0].body
                if mode=='work' and isinstance(n,ast.Return):
                    assert isinstance(n.value,ast.BoolOp) and len(n.value.values)==3
                    n.value.values=n.value.values[1:]
                projected.append(n)
            node.body=projected
            before=function(old,previous)
            assert key(node)==key(before),'default-off legacy body changed: '+file+':'+current
            result[file+':'+current]=hashlib.sha256(key(before).encode()).hexdigest()
    # No graph/model/worker-forward source is changed; overlay whitelist
    # separately enforces its exact bytes and compiled-library identity.
    return dict(verdict='PASS',legacy_function_bodies=result,functions=len(result),
                meaning='source projection with policy absent/AUTO_ARM=0; canonical connected traces checked separately')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path);p.add_argument('--overlay',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=dict(verdict='PASS',source=source_level(),negative_controls=negative_controls())
    if a.overlay:r['overlay']=overlay_level(a.parent,a.overlay)
    with a.out.open('x') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
