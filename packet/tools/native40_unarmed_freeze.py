#!/usr/bin/env python3
"""Fail-closed path and AST freeze vs the pinned native39 held-b source, source and generated overlay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from native37_robust_freeze import prove
from build_native30_held import verify
ROOT=Path(__file__).resolve().parents[1]
PARENT='1ffc394783d3d67e140056868b50ca2f74c88b3b'
DECLARATION=ROOT/'tools/native40_unarmed_declarations.json'
RUNTIME={'core/scheduler.py','e3/scheduler_hooks.py','e3/engine_driver.py'}
META={'SHA256SUMS','core/SHA256SUMS','S5-OVERLAP-GENERATION.json','e3/native-implementation-closure.json',
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
    return dict(parent=PARENT,paths=rows,ancillary_paths=sorted(ancillary),definitions=defs,compiled=compiled,skips=0,
                declaration_sha256=hashlib.sha256(DECLARATION.read_bytes()).hexdigest())
def overlay_level(parent,candidate):
    before,after=verify(parent),verify(candidate)
    assert set(before)==set(after),'overlay file set changed'
    changed={n for n in before if before[n]!=after[n]}|{'SHA256SUMS'}
    assert changed==RUNTIME|META,sorted(changed^(RUNTIME|META))
    d=json.loads(DECLARATION.read_text())['overlay_definitions']
    assert set(d)==RUNTIME
    defs={p:prove((parent/p).read_text(),(candidate/p).read_text(),p,d[p]) for p in sorted(RUNTIME)}
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
    return {'undeclared-function':refused(lambda:prove(show(p),(ROOT/p).read_text()+'\ndef native40_undeclared():\n pass\n',p,d['definitions'][p])),
            'undeclared-path':refused(lambda:check_paths(paths()+['A\tharness/native40_undeclared.py'],d)),
            'undeclared-overlay':refused(lambda:prove('x=1\n','x=2\n','core/scheduler.py',[]))}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path);p.add_argument('--overlay',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=dict(verdict='PASS',source=source_level(),negative_controls=negative_controls())
    if a.overlay:r['overlay']=overlay_level(a.parent,a.overlay)
    with a.out.open('x') as f:json.dump(r,f,sort_keys=True,indent=2);f.write('\n')
