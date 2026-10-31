#!/usr/bin/env python3
"""Compile proof: every .py in a held bundle plus its runner byte-compiles (CPU, no import)."""
import argparse,hashlib,json,py_compile,sys
from pathlib import Path


def prove(bundle,runner,out):
    files=sorted(p for p in bundle.rglob('*.py') if p.is_file())+[runner];failures=[]
    for path in files:
        try:py_compile.compile(str(path),cfile='/dev/null',doraise=True)
        except py_compile.PyCompileError as exc:failures.append(dict(file=str(path.relative_to(bundle)) if path!=runner else runner.name,error=str(exc).splitlines()[0][:200]))
    result=dict(verdict='PASS' if not failures else 'FAIL',files=len(files),failures=failures,gpu_execution=False,
        manifest=hashlib.sha256((bundle/'SHA256SUMS').read_bytes()).hexdigest(),runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest())
    with out.open('x') as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps(result,sort_keys=True));return 0 if not failures else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--runner',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.exit(prove(a.bundle.resolve(),a.runner.resolve(),a.out))
