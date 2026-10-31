#!/usr/bin/env python3
"""Native35 P-c: under the queue's own settings env every native30/33/34 plugin stays inert. CPU only, pinned image.

Parent side (imported by native35_p_c_cpu_probe.py). settings_env() takes the
plugin switches and PYTHONPATH from one settings file's EXTRA_ENV, using the
exact queue bytes. run() starts this file as a fresh python3. Its env is the
container env with every inherited FLASH_*, CUDA_LAUNCH_BLOCKING and
PYTHONPATH removed, plus those settings tokens.

Child side (--child):
- vLLM's real entry-point discovery and load_general_plugins();
- the plugin modules' own install flags;
- sys.meta_path;
- a wrapper-marker scan of every function and class attribute in every module
  the four plugins target, plus the torch/triton classes they patch.

The same child run under the P-b settings (LOCALIZE=1) is the positive control:
the scan finds the wrappers there, so an empty P-c scan is a discriminating
result, not a blind one.
"""
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys

MARKERS=('_native30_wrapped','_native30_host_wrapped','_native33_wrapped','_native34_wrapped')
PLUGINS={'native30_localize':'native30_localize:install','native30_receipt_policy':'native30_policy:install',
         'native33_diagnostic':'native33_diag:install','native34_diagnostic':'native34_diag:install'}
# Every environment variable the plugin modules read (os.environ in native30_overlay/e3/native30_plugins/*.py).
READ_BY_PLUGINS=('FLASH_NATIVE30_LOCALIZE','FLASH_NATIVE30_SYNC','FLASH_NATIVE33_DIAG','FLASH_NATIVE34_DIAG',
                 'FLASH_NATIVE34_ARM','FLASH_NATIVE34_READBACK','CUDA_LAUNCH_BLOCKING')
SYNC_DEFAULT='${FLASH_NATIVE30_SYNC:-none}'  # the owner arms with --sync none


def settings_env(path):
    found=re.findall(r'"EXTRA_ENV=([^"]*)"',Path(path).read_text());assert len(found)==1,path
    env={}
    for token in found[0].split():
        name,sep,value=token.partition('=');assert sep,token
        assert not name.startswith('FLASH_NATIVE') or name in READ_BY_PLUGINS,'unlisted plugin switch: '+name
        if name=='PYTHONPATH' or name in READ_BY_PLUGINS:
            value='none' if value==SYNC_DEFAULT else value
            assert '$' not in value and name not in env,token
            env[name]=value
    assert 'PYTHONPATH' in env and 'FLASH_NATIVE30_LOCALIZE' in env,env
    return env


def run(settings,label):
    base={k:v for k,v in os.environ.items() if not (k.startswith('FLASH_') or k in ('CUDA_LAUNCH_BLOCKING','PYTHONPATH'))}
    switches=settings_env(settings)
    done=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--child'],env=dict(base,**switches),cwd='/tmp',
                        capture_output=True,text=True,timeout=900)
    assert done.returncode==0,(label,done.returncode,done.stderr[-4000:])
    row=json.loads(done.stdout.strip().splitlines()[-1])
    row.update(label=label,settings=str(settings),settings_sha256=hashlib.sha256(Path(settings).read_bytes()).hexdigest(),
               switches=switches,inherited_removed=sorted(k for k in os.environ if k not in base),
               native30_stderr_lines=sum('NATIVE30 ' in line for line in done.stderr.splitlines()))
    return row


def child():
    here=str(Path(__file__).resolve().parent)
    sys.path[:]=[p for p in sys.path if p and str(Path(p).resolve())!=here]  # plugins come from PYTHONPATH only
    from importlib.metadata import entry_points
    found={p.name:p.value for p in entry_points(group='vllm.general_plugins')}
    assert {k:found.get(k) for k in PLUGINS}==PLUGINS,found
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    import torch
    from triton.runtime.jit import JITFunction
    import native30_host as host
    import native30_localize as localize
    import native30_policy as policy
    import native33_diag
    import native34_diag
    flags=dict(native30_localize=localize._INSTALLED,native30_receipt_policy=policy._INSTALLED,
               native33_diagnostic=native33_diag._INSTALLED,native34_diagnostic=native34_diag.INSTALLED)
    plugin_files={m.__name__:inspect.getfile(m) for m in (host,localize,policy,native33_diag,native34_diag)}
    finders=[type(f).__module__+'.'+type(f).__name__ for f in sys.meta_path
             if type(f).__module__ in ('native30_localize','native30_policy','native30_host','native33_diag','native34_diag')]
    wrapped=[];scanned=0
    def scan(owner,label):
        nonlocal scanned
        for name,value in list(vars(owner).items()):
            fn=value.__func__ if isinstance(value,(staticmethod,classmethod)) else value
            if not callable(fn):continue
            scanned+=1
            hits=[m for m in MARKERS if getattr(fn,m,False)]
            if hits:wrapped.append(dict(owner=label,name=name,markers=hits))
    targets=sorted(set(localize._TARGETS)|set(host.HOOKS)|set(native34_diag.TARGETS))
    for name in targets:
        module=importlib.import_module(name);scan(module,name)
        for attr,value in list(vars(module).items()):
            if isinstance(value,type) and value.__module__==name:scan(value,name+'.'+attr)
    for owner in (torch.cuda.CUDAGraph,JITFunction,torch._ops.OpOverload,torch._ops.OpOverloadPacket,torch.Tensor):
        scan(owner,owner.__module__+'.'+owner.__qualname__)
    print(json.dumps(dict(entry_points={k:found[k] for k in PLUGINS},installed=flags,plugin_files=plugin_files,
        meta_path_finders=finders,target_modules=targets,callables_scanned=scanned,wrapped=wrapped,
        cuda_initialized=torch.cuda.is_initialized(),
        sync_mode=os.environ.get('FLASH_NATIVE30_SYNC'),localize=os.environ.get('FLASH_NATIVE30_LOCALIZE')),sort_keys=True))


if __name__=='__main__' and sys.argv[1:]==['--child']:child()
