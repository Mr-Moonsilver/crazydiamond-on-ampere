#!/usr/bin/env python3
"""Fixed source/AST declaration and exact parent-overlay diff for native38."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from native37_robust_freeze import changed, definitions, prove, manifest

ROOT=Path(__file__).resolve().parents[1]
PARENT_SOURCE='ce6deb1'
PARENT=Path('<hub-dir>/agents/flash-next/tmp/native37-robust-held-b/q18a-native37-robust-bundle-34658b07177d/native30_overlay')
DECLARATION=ROOT/'tools/native38_plain_bind_declarations.json'
RUNTIME={'e3/engine_driver.py','e3/executor_hooks.py','e3/probe_control.py',
         'e3/reply_transport.py','e3/s5_startup.py','e3_sources/executor.py','e3_sources/core.py'}
META={'S5-OVERLAP-GENERATION.json','e3/native-implementation-closure.json',
      'e3/native-overlap-readiness.json','e3/decode-probe-evidence.json','SHA256SUMS'}

def git_show(commit,path):
    return subprocess.check_output(['git','show',commit+':'+path],cwd=ROOT,text=True)

def declaration():return json.loads(DECLARATION.read_text())

def source_level():
    declared=declaration()
    paths=subprocess.check_output(['git','diff','--name-status',PARENT_SOURCE,'--','patches/s5_engine','harness','tools'],cwd=ROOT,text=True).splitlines()
    assert set(paths)==set(declared['paths']),('undeclared source paths',set(paths)^set(declared['paths']))
    result={}
    for path,names in declared['definitions'].items():
        result[path]=prove(git_show(PARENT_SOURCE,path),(ROOT/path).read_text(),path,names)
    return dict(parent=PARENT_SOURCE,paths=paths,definitions=result,
                declaration_sha256=hashlib.sha256(DECLARATION.read_bytes()).hexdigest())

def overlay_level(overlay):
    before,after=manifest(PARENT),manifest(overlay)
    assert set(before)==set(after),'overlay file set changed'
    moved={n for n in before if before[n]!=after[n]}|{'SHA256SUMS'}
    assert moved==RUNTIME|META,('unexpected overlay changes',moved^(RUNTIME|META))
    result={}
    for path in sorted(RUNTIME):
        names=declaration()['overlay_definitions'][path]
        result[path]=prove((PARENT/path).read_text(),(overlay/path).read_text(),path,names)
    return dict(changed=sorted(moved),definitions=result,
                compiled_libraries_byte_identical=all(before[n]==after[n] for n in before if n.endswith('.so')))

def negative_controls():
    path='patches/s5_engine/s5_startup.py'
    old=git_show(PARENT_SOURCE,path);current=(ROOT/path).read_text()
    names=declaration()['definitions'][path]
    try:prove(old,current+'\ndef native38_undeclared():\n    pass\n',path,names)
    except AssertionError:return {'undeclared-function':'REFUSED'}
    raise AssertionError('undeclared source accepted')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--overlay',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    record=dict(verdict='PASS',source=source_level(),negative_controls=negative_controls())
    if a.overlay:record['overlay']=overlay_level(a.overlay)
    with a.out.open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(verdict=record['verdict'],out=str(a.out))))
