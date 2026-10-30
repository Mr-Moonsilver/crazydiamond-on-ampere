#!/usr/bin/env python3
"""Static fixture/startup-flag inventory; emits symbol names, never literal values.

AST patches (including default MagicMock), positive flags, and parseable embedded
Python are inventoried. This is an audit index, not a claim that a fixture proves
runtime readiness. Read the accompanying review for each scope's disposition.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FLAG=re.compile(r'(ready|verified|qualified|armable|capabil|installed|native_route|native_offers|allow_fixture|fixture_only|require_.*backend|require_.*closure|require_.*ready|build_composition|install_engine|after_engine_startup|install_native_lane|response_success|_ROLE_KIND|auth|canary|membership)',re.I)
SENSITIVE=re.compile(r'(startup|native_integration|admission_metadata|observer_runtime|native_observation|private_auth|canary|native_performance_queue|native_readiness)')


def symbol(n):
    if isinstance(n,ast.Name):return n.id
    if isinstance(n,ast.Attribute):return symbol(n.value)+'.'+n.attr
    if isinstance(n,ast.Constant) and isinstance(n.value,str):
        return n.value if re.fullmatch(r'[A-Za-z_][\w./-]*',n.value) else '<literal>'
    if isinstance(n,ast.Call):return symbol(n.func)+'()'
    return type(n).__name__


def scan(path,text):
    rows=[]
    def walk(tree,offset=0,embedded=False):
        parents={child:parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        def scope(n):
            names=[]
            while n in parents:
                n=parents[n]
                if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):names.append(n.name)
            return '.'.join(reversed(names)) or '<module>'
        def add(n,kind,target,detail):
            rows.append({'file':path,'line':offset+n.lineno,'scope':scope(n),'kind':kind,
                         'target':target,'replacement':detail,'embedded':embedded})
        for n in ast.walk(tree):
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)):
                args=n.args.posonlyargs+n.args.args
                for arg,v in zip(args[-len(n.args.defaults):],n.args.defaults):
                    if isinstance(v,ast.Constant) and v.value is True:
                        add(n,'positive-default',arg.arg,'True')
            if isinstance(n,ast.Call):
                fun=symbol(n.func)
                if fun.endswith(('patch','patch.object','patch.dict')) or fun in ('patch','patch.object','patch.dict'):
                    args=n.args
                    target=symbol(args[0]) if args else '<keyword>'
                    idx=1
                    if fun.endswith('patch.object') and len(args)>1:
                        target+='.'+symbol(args[1]);idx=2
                    value=args[idx] if len(args)>idx else next((k.value for k in n.keywords if k.arg in ('new','return_value','side_effect')),None)
                    # All patches retained; defaults and lambdas can mask guards.
                    detail=('default-MagicMock' if value is None else
                            'lambda' if isinstance(value,ast.Lambda) else
                            str(value.value) if isinstance(value,ast.Constant) and type(value.value) in (bool,type(None)) else symbol(value))
                    add(n,'patch',target,detail)
                if fun in ('setattr','object.__setattr__') and len(n.args)>=3:
                    target=symbol(n.args[0])+'.'+symbol(n.args[1])
                    if isinstance(n.args[2],ast.Constant) and n.args[2].value is True:add(n,'positive-setattr',target,'True')
                for k in n.keywords:
                    if k.arg and isinstance(k.value,ast.Constant) and k.value.value is True:
                        add(n,'positive-keyword',fun+'.'+k.arg,'True')
            if isinstance(n,(ast.Assign,ast.AnnAssign)):
                v=n.value
                targets=n.targets if isinstance(n,ast.Assign) else [n.target]
                if isinstance(v,ast.Constant) and v.value is True:
                    for t in targets:
                        target=symbol(t)
                        add(n,'positive-assignment',target,'True')
            if isinstance(n,ast.Dict):
                for k,v in zip(n.keys,n.values):
                    if k and isinstance(k,ast.Constant) and isinstance(k.value,str) and isinstance(v,ast.Constant) and v.value is True:
                        add(n,'positive-data',k.value,'True')
            if isinstance(n,ast.Return) and isinstance(n.value,ast.Constant) and n.value.value is True :
                add(n,'positive-return',scope(n),'True')
            if not embedded and isinstance(n,ast.Constant) and isinstance(n.value,str) and ('def ' in n.value or 'import ' in n.value):
                try:sub=ast.parse(n.value)
                except SyntaxError:continue
                walk(sub,n.lineno-1,True)
    walk(ast.parse(text))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    files=subprocess.check_output(['git','ls-files','--','harness','patches/s5_engine','tools'],cwd=ROOT,text=True).splitlines()
    # The new binding test may be untracked while this audit is first run.
    files=sorted(set(files)|{'harness/test_e3_native_startup_binding.py'})
    rows=[];pins={}
    for name in files:
        path=ROOT/name
        if path.suffix!='.py' or not path.is_file():continue
        text=path.read_text();pins[name]=hashlib.sha256(text.encode()).hexdigest()
        rows.extend(scan(name,text))
    rows.sort(key=lambda x:(x['file'],x['line'],x['kind'],x['target']))
    result={'schema':'native-startup-fixture-audit/1','scope':'tracked Python plus new actual binding test; symbol-only static index',
            'source_shas':pins,'entries':rows,'entries_total':len(rows),
            'limitations':['Dynamic string-built targets require source review; index is not runtime evidence.',
                          'Positive test data and negative mutations are deliberately included, not assumed to be bypasses.']}
    with a.out.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'files':len(pins),'entries':len(rows),'audit_sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()}))

if __name__=='__main__':main()
