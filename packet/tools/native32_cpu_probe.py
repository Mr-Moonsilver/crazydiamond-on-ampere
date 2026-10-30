"""CPU-only plugin discovery/source and wrapper tests; never used by box workers."""
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import unittest

if '--inside' not in sys.argv:
    # Extension registration only. External to the engine/plugin mount; never a GPU result.
    stub=Path('/tmp/native30-libcuda');stub.mkdir();(stub/'libcuda.so.1').symlink_to('/usr/local/cuda/lib64/stubs/libcuda.so')
    env=dict(os.environ);env['LD_LIBRARY_PATH']=str(stub)+':'+env.get('LD_LIBRARY_PATH','')
    os.execvpe('python3',['python3','-B',__file__,'--inside'],env)

sys.path.insert(0,'/source/harness/e3_builder')
from importlib.metadata import entry_points
found=[p for p in entry_points(group='vllm.general_plugins') if p.name=='native30_localize']
assert len(found)==1 and found[0].value=='native30_localize:install'
policies=[p for p in entry_points(group='vllm.general_plugins') if p.name=='native30_receipt_policy']
assert len(policies)==1 and policies[0].value=='native30_policy:install'
from vllm.plugins import load_general_plugins
load_general_plugins()
import native30_localize as plugin
assert plugin._INSTALLED
import native30_policy as policy
assert policy._INSTALLED and policy.host.MAX_BYTES_PER_RANK==1024**3
import torch
assert not torch.cuda.is_initialized()
modules={}
for name in sorted(plugin._TARGETS):
    module=importlib.import_module(name)
    methods=[]
    base={'vllm.v1.flash_e3.lane_dispatch':('LaneDispatcher',('_do_prepare','_do_prefix','_do_layer','_do_suffix','_do_state_finalize','_do_pp_send','_do_pp_receive')),
          'vllm.v1.flash_e3.worker_bridge':('NativeWorkerBridge',('open',)),
          'vllm.v1.worker.gpu_worker':('Worker',('execute_model','sample_tokens'))}
    checks=base.get(name)
    if checks:
        cls=getattr(module,checks[0])
        for method in checks[1]:assert getattr(getattr(cls,method),'_native30_wrapped',False),(name,method)
        methods.extend(checks[1])
    host_checks=dict(plugin.host.HOOKS.get(name,{}))
    if name.endswith('.lane_dispatch'):host_checks['LaneDispatcher']=('submit',)
    if name.endswith('.gpu_worker'):host_checks['Worker']=('init_device','load_model')
    for cls, names in host_checks.items():
        for method in names:assert getattr(getattr(getattr(module,cls),method),'_native30_host_wrapped',False),(name,method)
        methods.extend(cls+'.'+method for method in names)
    if name.endswith('.native_qualification'):
        assert getattr(module.handle_worker,'_native30_host_wrapped',False)
        methods.append('handle_worker')
    path=Path(inspect.getfile(module))
    if name.endswith('.buffer_utils'):
        assert path.read_bytes()==Path('/source/harness/e3_reference/v1/worker/gpu/buffer_utils.py').read_bytes(),'pinned-buffer-source-differs'
    modules[name]=dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),methods=list(methods))
assert not torch.cuda.is_initialized()
suite=unittest.defaultTestLoader.discover('/source/harness',pattern='test_native30*.py')
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native31.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native32.py'))
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert not torch.cuda.is_initialized()
identity_receipts=sys.modules['test_native30_identity'].IDENTITY_RECEIPTS
assert len(identity_receipts)==3
Path('/out/identity-check.json').write_text(json.dumps(identity_receipts,sort_keys=True,indent=2)+'\n')
cli_receipts=sys.modules['test_native30_cli'].CLI_RECEIPTS
assert len(cli_receipts)==2 and cli_receipts[0]['first_network_call']=='prepare-full'
Path('/out/cli-check.json').write_text(json.dumps(cli_receipts,sort_keys=True,indent=2)+'\n')
# Execute the real frozen owner runner's pure pre-launch checks, not copies.
spec=importlib.util.spec_from_file_location('native32_frozen_owner','/source/owner.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
overlay=Path('/source/harness/native30_overlay')
owner.verify_sources(overlay,built=True,final=True)
owner.OVERLAY=overlay
green,final_manifest,build_proof=owner.verify_final_build()
assert final_manifest==hashlib.sha256((overlay/'SHA256SUMS').read_bytes()).hexdigest()
owner_prelaunch=dict(verdict='PASS',methods_executed=['verify_sources(built=True,final=True)','verify_final_build'],runner_sha256=hashlib.sha256(Path('/source/owner.py').read_bytes()).hexdigest(),overlay_manifest=final_manifest,green_sha256=green,device_state_checks='DEFERRED to owner: stopped unit, current launcher/defaults, GPUs idle and real arm-time window',device_execution=False)
Path('/out/owner-prelaunch.json').write_text(json.dumps(owner_prelaunch,sort_keys=True,indent=2)+'\n')
import native31_cpu_check
native31_receipt=native31_cpu_check.run()
Path('/out/native31-check.json').write_text(json.dumps(native31_receipt,sort_keys=True,indent=2)+'\n')
record=dict(owner_prelaunch=owner_prelaunch,native31=native31_receipt,receipt_policy='PASS',receipt_budget_bytes=policy.BYTES_PER_RANK,receipt_tail_s=policy.TAIL_S,row4_cli_validation='PASS',cli_receipts=cli_receipts,identity_receipts=identity_receipts,verdict='PASS' if result.wasSuccessful() and not result.skipped and result.testsRun==47 else 'FAIL',tests_run=result.testsRun,skips=len(result.skipped),plugin_loader='PASS',identity_capture_check='PASS' if result.wasSuccessful() else 'FAIL',identity_timeout_negative_rejected=result.wasSuccessful(),mounted_modules=modules,cuda_initialized=False,gpu_execution=False,
    fixture='CPU driver-registration stub, synthetic launch/sync/control objects; no GPU model forward, stream or memory query')
Path('/out/probe.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
raise SystemExit(0 if record['verdict']=='PASS' else 1)
