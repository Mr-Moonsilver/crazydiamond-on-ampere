"""Native39 trunk CPU-only plugin discovery/source, wrapper, grade, builder-view, speed-arm, L2 row4, LOCALIZE=0 and trunk queue checks; never used by box workers.

Derived from native35_p_c_cpu_probe.py by tools/native39_trunk_tools_derive.py.
FLASH_NATIVE35_ARM must be TRUNK: it selects the frozen trunk owner runner, the
trunk window (native39_trunk_gate) and the provenance binding; the trunk queue
suite (test_native39_trunk_queue) runs with the inherited suites. This probe process installs the native30 and native34 plugins in their
arm-O/readback configuration so the inherited plugin checks still run. The P-c
queue itself runs with FLASH_NATIVE30_LOCALIZE=0, FLASH_NATIVE33_DIAG=0 and
FLASH_NATIVE34_DIAG=0.

P-c adds five things to P-b's receipts:
- native35_p_c_inert_check, which loads the plugins in fresh processes under
  the exact P-c settings env (nothing installs, nothing is wrapped, no
  NATIVE30 line) and under the P-b settings env (positive control);
- the P-c cleanup receipts;
- the P-c draft slot;
- the P-c row4 receipts (test_native35_p_c_window): the decode stop at the
  20-token window, the observer_end spool over real loopback HTTP (>112 MiB
  collected, >512 MiB refused with bytes and sha), the generated decode abort
  with a native chunk in flight, and the exact P-c row4 argv;
- the row4 capacity continuation receipts (test_native35_p_c_continuation):
  capacity-only INCONCLUSIVE records from the real graders continue, mixed or
  other ones withhold, the >512 MiB refusal continues, and the queue's own gates
  and both W2 rounds run under bash with the real classifier and W2 gate.
"""
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
import native34_diag as diagnostic
assert diagnostic.INSTALLED
assert all(getattr(getattr(torch.cuda.CUDAGraph,n),'_native34_wrapped',False) for n in ('capture_begin','capture_end','replay'))
from triton.runtime.jit import JITFunction
assert getattr(JITFunction.run,'_native34_wrapped',False)
assert not torch.cuda.is_initialized()
modules={}
for name in sorted(plugin._TARGETS):
    module=importlib.import_module(name)
    methods=[]
    base={'vllm.v1.flash_e3.lane_dispatch':('LaneDispatcher',('_do_prepare','_do_prefix','_do_layer','_do_suffix','_do_state_finalize','_do_pp_send','_do_pp_receive')),
          'vllm.v1.flash_e3.worker_bridge':('NativeWorkerBridge',('open',)),
          'vllm.v1.worker.gpu_worker':('Worker',('execute_model','sample_tokens'))}
    if name.endswith('.qsa_cache'):assert getattr(module.QSAMetadataBuilder.build,'_native34_wrapped',False)
    if name.endswith('.lane_chunk'):assert getattr(module.LaneChunk.__init__,'_native34_wrapped',False)
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
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native33.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native34_*.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native35_*.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native39_trunk_queue.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native40_box_queue.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native41_box_queue.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native41_unarmed_box.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_native41_progress_queue.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_perf_queue.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_step_a_queue.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_step_b_queue.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_next_queue.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_next_c1_queue.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_alpha_next_c1b_queue.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_inwait_queue.py'))
suite.addTests(unittest.defaultTestLoader.discover('/source/harness',pattern='test_compose5_queue.py'))
EXPECTED_TESTS=611  # the 585 inwait probe cases plus the 26 compose-5 queue cases; pinned into the frozen runner gate
import step_b_clock_bound
suite,clock_bound_removed=step_b_clock_bound.filtered(suite)
assert clock_bound_removed==step_b_clock_bound.IN_PROBE,clock_bound_removed
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert not torch.cuda.is_initialized()
identity_receipts=sys.modules['test_native30_identity'].IDENTITY_RECEIPTS
assert len(identity_receipts)==3
Path('/out/identity-check.json').write_text(json.dumps(identity_receipts,sort_keys=True,indent=2)+'\n')
cli_receipts=sys.modules['test_native30_cli'].CLI_RECEIPTS
assert len(cli_receipts)==2 and cli_receipts[0]['first_network_call']=='prepare-full'
Path('/out/cli-check.json').write_text(json.dumps(cli_receipts,sort_keys=True,indent=2)+'\n')
# Execute the real frozen owner runner's pure pre-launch checks, not copies.
spec=importlib.util.spec_from_file_location('native34_frozen_owner','/source/owner.py')
owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
role=os.environ['FLASH_NATIVE35_ARM'];assert role=='TRUNK' and os.environ['FLASH_NATIVE34_ARM']=='O'
provenance=json.loads(Path('/source/harness/native30-provenance.json').read_text())
assert provenance['native35_arm']==role and provenance['candidate']==owner.QNAME=='q18a-parity-n3s24'
owner.check_slot()
import native34_gate as weekend_gate
import native35p_gate
assert (native35p_gate.NOT_BEFORE,native35p_gate.HARD_END)==(weekend_gate.NOT_BEFORE,weekend_gate.HARD_END)
import native39_trunk_gate
assert (native39_trunk_gate.NOT_BEFORE,native39_trunk_gate.HARD_END)==(weekend_gate.NOT_BEFORE,weekend_gate.HARD_END)
import alpha_perf_gate
import step_a_gate
import step_b_gate
import alpha_next_gate
import alpha_next_c1_gate
import alpha_next_c1b_gate
import inwait_gate
import compose5_gate
active_and_cleanup=compose5_gate.ACTIVE_S+compose5_gate.CLEANUP_S
from unittest.mock import patch
for instant in (weekend_gate.NOT_BEFORE-1,weekend_gate.HARD_END-active_and_cleanup+1):
    with patch.object(owner.time,'time',lambda:instant):
        owner.check_slot()  # never refused by wall clock: launch time is governed by the lead (operator, 2 Nov 2026)
overlay=Path('/source/harness/native30_overlay')
owner.verify_sources(overlay,built=True,final=True)
owner.OVERLAY=overlay
green,final_manifest,build_proof=owner.verify_final_build()
assert final_manifest==hashlib.sha256((overlay/'SHA256SUMS').read_bytes()).hexdigest()
actual_bridge=sys.modules['test_native32'].NativeWorkerBridge
assert actual_bridge is sys.modules['vllm.v1.flash_e3.worker_bridge'].NativeWorkerBridge
owner_prelaunch=dict(actual_bridge_class=actual_bridge.__module__+'.'+actual_bridge.__name__,verdict='PASS',methods_executed=['check_slot (no wall-clock refusal at the two former boundaries)','verify_sources(built=True,final=True)','verify_final_build'],runner_sha256=hashlib.sha256(Path('/source/owner.py').read_bytes()).hexdigest(),overlay_manifest=final_manifest,green_sha256=green,device_state_checks='DEFERRED to owner: stopped unit, current launcher/defaults, GPUs idle and real arm-time window',device_execution=False)
Path('/out/owner-prelaunch.json').write_text(json.dumps(owner_prelaunch,sort_keys=True,indent=2)+'\n')
import native31_cpu_check
native31_receipt=native31_cpu_check.run()
Path('/out/native31-check.json').write_text(json.dumps(native31_receipt,sort_keys=True,indent=2)+'\n')
# Real CPU Tensor getters and descriptor installation, with no device init.
from unittest.mock import patch
import native34_worker as worker
from native34_storage import tensor_view,overlap
actual=torch.arange(16,dtype=torch.int32);view=actual[4:12]
a=tensor_view(actual);b=tensor_view(view)
assert a['data_ptr']==actual.data_ptr() and b['data_ptr']==view.data_ptr()
assert a['allocation_base']==b['allocation_base'] and overlap(a,b)
assert a['view_within_storage'] and b['view_within_storage']
original_zero=torch.Tensor.zero_
try:
    worker.patch(torch.Tensor,'zero_',lambda old,a,k:worker.phase(old,a,k,'OUTPUT_ZERO'))
    ordinary_zero=torch.ones(4);assert ordinary_zero.zero_() is ordinary_zero and torch.equal(ordinary_zero,torch.zeros(4))
    with patch.object(__import__('lane_comm'),'register_selection_op',lambda:torch.ops.vllm.flash_s5_selection):
        instrumented=native31_cpu_check.run()
finally:torch.Tensor.zero_=original_zero
assert instrumented['output_sha256']==native31_receipt['output_sha256']
assert not torch.cuda.is_initialized()
native34_receipt=dict(verdict='PASS',ordinary_exactness='PASS',steps=32,rows=8,
    real_cpu_tensor_storage='PASS',real_tensor_descriptor_patch='PASS',cuda_initialized=False,
    scope='real CPU Torch ordinary arithmetic, metadata getters and inactive late-hook bypass; device events/launches are declared leaves')
Path('/out/native34-check.json').write_text(json.dumps(native34_receipt,sort_keys=True,indent=2)+'\n')
causal_cli=sys.modules['test_native34_frontend'].CLI_RECEIPTS
assert len(causal_cli)==3
Path('/out/causal-cli-check.json').write_text(json.dumps(causal_cli,sort_keys=True,indent=2)+'\n')

slot_receipts=sys.modules['test_native34_queue'].SLOT_RECEIPTS+sys.modules['test_native35_queue'].SLOT_RECEIPTS+sys.modules['test_native35_performance_queue'].SLOT_RECEIPTS+sys.modules['test_native35_l2'].SLOT_RECEIPTS
assert slot_receipts==[]  # their real-clock slot tests are declared (step_b_clock_bound) and left out
assert not {r['draft'] for r in slot_receipts}&{'q18a_native35_r.draft.sh','q18a_native35_o.draft.sh','q18a_native35_l.draft.sh','q18a_native35_p.draft.sh','q18a_native35_p_b.draft.sh','q18a_native35_p_c.draft.sh','q18a_native35_l2.draft.sh'}
assert [r['active_s'] for r in slot_receipts if r['draft']=='q18a_native35_p_c.draft.sh']==[]
trunk_slots=sys.modules['test_native39_trunk_queue'].SLOT_RECEIPTS
assert trunk_slots==[]
slot_receipts=slot_receipts+trunk_slots+sys.modules['test_native40_box_queue'].SLOT_RECEIPTS+sys.modules['test_native41_box_queue'].SLOT_RECEIPTS+sys.modules['test_native41_progress_queue'].SLOT_RECEIPTS+sys.modules['test_alpha_perf_queue'].SLOT_RECEIPTS+sys.modules['test_step_a_queue'].SLOT_RECEIPTS+sys.modules['test_step_b_queue'].SLOT_RECEIPTS+sys.modules['test_alpha_next_queue'].SLOT_RECEIPTS+sys.modules['test_alpha_next_c1_queue'].SLOT_RECEIPTS+sys.modules['test_alpha_next_c1b_queue'].SLOT_RECEIPTS+sys.modules['test_inwait_queue'].SLOT_RECEIPTS+sys.modules['test_compose5_queue'].SLOT_RECEIPTS
Path('/out/draft-window-check.json').write_text(json.dumps(slot_receipts,sort_keys=True,indent=2)+'\n')
protocol_receipts=sys.modules['test_native34_runtime'].PROTOCOL_RECEIPTS
assert len(protocol_receipts)==1 and protocol_receipts[0]['verdict']=='PASS' and protocol_receipts[0]['hand_built_marker'] is False
Path('/out/protocol-check.json').write_text(json.dumps(protocol_receipts,sort_keys=True,indent=2)+'\n')
import native34_readback_cpu_check
readback_check=native34_readback_cpu_check.run()
assert readback_check['verdict']=='PASS'
Path('/out/readback-check.json').write_text(json.dumps(readback_check,sort_keys=True,indent=2)+'\n')
# Native35: the same pinned CPU builder sequence twice. Old shared storage must
# still refuse with request IDs 1-3 out of bounds; the private view must pass
# while the ordinary builder's bytes equal its ordinary-alone build.
assert readback_check['negative_refusal']['verdict']=='VIOLATION' and readback_check['negative_refusal']['first_bad']==[[1,1],[2,2],[3,3]]
assert readback_check['observed_request_ids']==[0,1,2,3]
assert readback_check['native_private_positive']['verdict']=='PASS' and readback_check['ordinary_after_native_state']=='PASS'
assert readback_check['production_private_group_helper']=='PASS' and readback_check['private_builder_tensor_count']>=7
import native35_builder_view_cpu_check
builder_view=native35_builder_view_cpu_check.run()
Path('/out/native35-builder-view.json').write_text(json.dumps(builder_view,sort_keys=True,indent=2)+'\n')
assert builder_view['verdict']=='PASS',builder_view.get('error')
assert not torch.cuda.is_initialized()
native35_cli=sys.modules['test_native35_grade'].CLI_RECEIPTS
assert len(native35_cli)==2 and all(r['device_arm']=='O' for r in native35_cli)
native35_readback=sys.modules['test_native35_readback'].READBACK_RECEIPTS
assert len(native35_readback)==1 and native35_readback[0]['drain'] is False and native35_readback[0]['fence'] is False
# Native35 P: exact W1/W2 queue argv to the first send, both rounds over the loopback APIServer leaf, aggregate grade.
perf_receipts=sys.modules['test_native35_performance'].PERF_RECEIPTS
assert sorted(r['case'] for r in perf_receipts)==['aggregate-actual-packets-shortfall','aggregate-refusals-retained','two-round-composition','w1-exact-queue-argv','w2-both-round-exact-queue-argv','w2-refusals-before-network'] and all(r['verdict']=='PASS' for r in perf_receipts)
Path('/out/native35-p-performance-check.json').write_text(json.dumps(perf_receipts,sort_keys=True,indent=2)+'\n')
# Native35 P-b: the L2 runner's own receipts (unchanged L2 suite) and the exact P-b row4 argv through it.
l2_cli=sys.modules['test_native35_l2'].CLI_RECEIPTS
assert len(l2_cli)==1 and l2_cli[0]['runner']=='ResumeWindowPlacementRunner' and l2_cli[0]['first_network_call']=='prepare-full' and l2_cli[0]['substitution_restored'] is True
l2_timing=sys.modules['test_native35_l2'].TIMING_RECEIPTS
assert sorted(r['case'] for r in l2_timing)==['box-shaped','early-finish','fast-prefill','prefill-never-opens']
assert [r for r in l2_timing if r['case']=='box-shaped'][0]['parent']=='placement-window-incomplete'
row4=sys.modules['test_native35_p_b'].ROW4_RECEIPTS
assert len(row4)==1 and row4[0]['verdict']=='PASS' and row4[0]['runner']=='ResumeWindowPlacementRunner' and row4[0]['first_network_call']=='prepare-full' and row4[0]['substitution_restored'] is True
assert row4[0]['runner_sha256']==hashlib.sha256(Path('/source/harness/e3_native35_l2_placement.py').read_bytes()).hexdigest()
Path('/out/native35-p-b-row4-check.json').write_text(json.dumps(dict(p_b_row4=row4,l2_cli=l2_cli,l2_timing=l2_timing),sort_keys=True,indent=2)+'\n')
# Native35 P-c: the four plugins are inert under the exact P-c settings env (fresh processes, real
# load_general_plugins, every declared target module scanned), and the identical check under the
# P-b settings finds the native30 wrappers (positive control). P-c's row4 receipts follow the cleanup check.
import native35_p_c_inert_check as inert
host_mod=plugin.host
expected_wrapped=sorted([('vllm.v1.flash_e3.lane_dispatch.LaneDispatcher',n) for n in ('_do_prepare','_do_prefix','_do_layer','_do_suffix','_do_state_finalize','_do_pp_send','_do_pp_receive','submit')]
    +[('vllm.v1.flash_e3.worker_bridge.NativeWorkerBridge','open'),('vllm.v1.flash_e3.native_qualification','handle_worker')]
    +[('vllm.v1.worker.gpu_worker.Worker',n) for n in ('execute_model','sample_tokens','init_device','load_model')]
    +[(module+'.'+cls,method) for module,classes in host_mod.HOOKS.items() for cls,methods in classes.items() for method in methods])
positive=inert.run('/source/harness/runs/q18a_native35_p_b_settings.sh','P-B settings (LOCALIZE=1, positive control)')
negative=inert.run('/source/harness/runs/q18a_parity_n3s24_settings.sh','TRUNK settings (LOCALIZE=0)')
for row in (positive,negative):
    assert row['entry_points']==inert.PLUGINS and row['cuda_initialized'] is False and row['sync_mode']=='none'
    assert all(f.startswith('/usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e3/native30_plugins/') for f in row['plugin_files'].values())
    assert row['switches']['PYTHONPATH']=='/usr/local/lib/python3.12/dist-packages/vllm/v1/flash_e3/native30_plugins'
    assert {k:row['switches'].get(k) for k in ('FLASH_NATIVE33_DIAG','FLASH_NATIVE34_DIAG')}=={'FLASH_NATIVE33_DIAG':'0','FLASH_NATIVE34_DIAG':'0'}
assert positive['switches']['FLASH_NATIVE30_LOCALIZE']=='1' and negative['switches']['FLASH_NATIVE30_LOCALIZE']=='0'
assert positive['callables_scanned']==negative['callables_scanned']>0 and positive['target_modules']==negative['target_modules']
assert positive['installed']=={'native30_localize':True,'native30_receipt_policy':True,'native33_diagnostic':False,'native34_diagnostic':False}
assert positive['meta_path_finders']==['native30_localize.Finder'] and positive['native30_stderr_lines']>0
assert sorted((w['owner'],w['name']) for w in positive['wrapped'])==expected_wrapped,(positive['wrapped'],expected_wrapped)
assert negative['installed']=={k:False for k in inert.PLUGINS} and negative['meta_path_finders']==[] and negative['wrapped']==[] and negative['native30_stderr_lines']==0
inert_receipt=dict(verdict='PASS',positive_control=positive,localize_off=negative,expected_positive_wrappers=len(expected_wrapped),
    scope='fresh python3 per settings file in the pinned image; plugin switches and PYTHONPATH from the exact settings EXTRA_ENV; CPU libcuda stub; no engine run')
Path('/out/native35-p-c-inert-check.json').write_text(json.dumps(inert_receipt,sort_keys=True,indent=2)+'\n')
cleanup=sys.modules['test_native35_p_c'].CLEANUP_RECEIPTS
assert sorted(r['case'] for r in cleanup)==['localize-on-rows-counterfactual','p-b-lines-on-localize-off-log','p-c-localize-off'] and all(r['verdict']=='PASS' for r in cleanup)
Path('/out/native35-p-c-cleanup-check.json').write_text(json.dumps(cleanup,sort_keys=True,indent=2)+'\n')
# Native35 P-c row4: decode stop at the window, observer_end spool, decode abort, exact argv.
window_suite=sys.modules['test_native35_p_c_window']
pc_timing=window_suite.TIMING_RECEIPTS;pc_spool=window_suite.SPOOL_RECEIPTS;pc_abort=window_suite.ABORT_RECEIPTS;pc_cli=window_suite.CLI_RECEIPTS
assert sorted(r['case'] for r in pc_timing)==['refusals-kept','stop-at-window'] and all(r['verdict']=='PASS' for r in pc_timing)
assert sorted(r['case'] for r in pc_spool)==['close-delimited-store-to-ceiling','over-112-under-512-end-to-end','over-512-refused'] and all(r['verdict']=='PASS' for r in pc_spool)
assert [r for r in pc_spool if r['case']=='over-512-refused'][0]['stored_bytes']==0
assert sorted(r['case'] for r in pc_abort)==['decode-abort-chunk-in-flight','native-uid-abort-counterfactual'] and all(r['verdict']=='PASS' for r in pc_abort)
assert len(pc_cli)==1 and pc_cli[0]['runner']=='StoppedWindowPlacementRunner' and pc_cli[0]['first_network_call']=='prepare-full'
assert pc_cli[0]['runner_sha256']==hashlib.sha256(Path('/source/harness/e3_native35_p_c_placement.py').read_bytes()).hexdigest()
pc_row4=dict(timing=pc_timing,spool=pc_spool,abort=pc_abort,cli=pc_cli)
Path('/out/native35-p-c-row4-check.json').write_text(json.dumps(pc_row4,sort_keys=True,indent=2)+'\n')
# Native35 P-c row4 capacity continuation: both ways over the real graders, the refusal path and the queue's own bytes.
cont_suite=sys.modules['test_native35_p_c_continuation']
cont_composition=cont_suite.COMPOSITION_RECEIPTS;cont_refusal=cont_suite.REFUSAL_RECEIPTS;cont_queue=cont_suite.QUEUE_RECEIPTS
assert sorted((r['case'],r['eligible']) for r in cont_composition)==[('both',True),('clean-pass',False),('correlation',True),('correlation-errors',False),('kernel',True),('kernel-raw-fail',False),('kernel-replays-60',False),('kernel-unknown-node',False)]
assert all(r['verdict']=='PASS' for r in cont_composition+cont_refusal+cont_queue)
assert [(r['case'],r['eligible'],r['body_bytes']>512*1024**2) for r in cont_refusal]==[('observer-end-over-spool-ceiling',True,True)]
assert sorted(r['case'] for r in cont_queue)==['capacity-only-continues','error-marker-withheld','mixed-withheld']
assert [r['w2_gate_passed'] for r in cont_queue if r['case']=='capacity-only-continues']==[[0,1]] and not any(r['w1_run'] for r in cont_queue if r['case']!='capacity-only-continues')
pc_continuation=dict(composition=cont_composition,refusal=cont_refusal,queue=cont_queue)
Path('/out/native35-p-c-continuation-check.json').write_text(json.dumps(pc_continuation,sort_keys=True,indent=2)+'\n')
# Native39 trunk: the queue's own segments under bash (bind receipt with the real host observer; row4 declared
# NOT_RUN, per-rank instruments, sealed native W1, ingress, typed FULL + idle before the plain W1, W2 rounds,
# returns and summary with the real checks, client and grade; W2 length guard; sourced settings through the real
# launcher render; real box access lines).
trunk_cases=sys.modules['test_native39_trunk_queue'].QUEUE_RECEIPTS
assert sorted(r['case'] for r in trunk_cases)==['bind-absent', 'bind-after-first-200', 'bind-before-first-200', 'bind-other-container', 'box-access-lines', 'guard-count-differs', 'guard-declared-counts', 'guard-no-engine', 'guard-refused-band', 'guard-tampered-declaration', 'health-deadline', 'launcher-loopback-host', 'round-wrapper-declared-not-run-only', 'tail-foreign-at-entry-no-split', 'tail-hidden-arrival-invalid', 'tail-maps-unreadable-withholds', 'tail-not-idle-withholds-w2', 'tail-observer-mapped-withholds', 'tail-observer-off-sealed-w1-plain-on-full-rounds', 'tail-swapped-member-no-split', 'tail-transient-foreign-ingress-invalid', 'tail-unqualified-withholds-all', 'tail-unrestored-split-withholds-plain'] and all(r['verdict']=='PASS' for r in trunk_cases)
assert {r['case']:r['bind'] for r in trunk_cases if r['case'].startswith('bind-')}=={'bind-absent':'INCONCLUSIVE','bind-after-first-200':'FAIL','bind-before-first-200':'PASS','bind-other-container':'INCONCLUSIVE'}
assert [r['trunk_hosts'] for r in trunk_cases if r['case']=='launcher-loopback-host']==[['0.0.0.0','127.0.0.1']]
assert [r['plain_requests'] for r in trunk_cases if r['case']=='tail-observer-off-sealed-w1-plain-on-full-rounds']==[8]
assert {r['case']:r['instruments'] for r in trunk_cases if 'instruments' in r}=={'tail-observer-mapped-withholds':'FAIL','tail-maps-unreadable-withholds':'INCONCLUSIVE'}
assert [(r['full_idle'],r['plain_requests']) for r in trunk_cases if r['case']=='tail-unrestored-split-withholds-plain']==[('INCONCLUSIVE',0)]
assert [r['split_entries'] for r in trunk_cases if r['case'] in ('tail-foreign-at-entry-no-split','tail-swapped-member-no-split')]==[0,0]
assert [r['accepted'] for r in trunk_cases if r['case']=='round-wrapper-declared-not-run-only']==[['declared']]
Path('/out/native39-trunk-queue-check.json').write_text(json.dumps(dict(cases=trunk_cases,slots=trunk_slots),sort_keys=True,indent=2)+'\n')
alpha_slots=sys.modules['test_alpha_perf_queue'].SLOT_RECEIPTS
assert alpha_slots==[]
alpha_calls=sys.modules['test_alpha_perf_queue'].ARGPARSE_RECEIPTS
assert alpha_calls and all(r['verdict']=='parsed' for r in alpha_calls)
Path('/out/alpha-perf-queue-check.json').write_text(json.dumps(dict(slots=alpha_slots,argparse=alpha_calls),sort_keys=True,indent=2)+'\n')
step_a_slots=sys.modules['test_step_a_queue'].SLOT_RECEIPTS
assert step_a_slots==[]
step_a_calls=sys.modules['test_step_a_queue'].ARGPARSE_RECEIPTS
assert step_a_calls and all(r['verdict']=='parsed' for r in step_a_calls)
Path('/out/step-a-queue-check.json').write_text(json.dumps(dict(slots=step_a_slots,argparse=step_a_calls),sort_keys=True,indent=2)+'\n')
step_b_slots=sys.modules['test_step_b_queue'].SLOT_RECEIPTS
assert step_b_slots==[]
step_b_calls=sys.modules['test_step_b_queue'].ARGPARSE_RECEIPTS
assert step_b_calls and all(r['verdict']=='parsed' for r in step_b_calls)
Path('/out/step-b-queue-check.json').write_text(json.dumps(dict(slots=step_b_slots,argparse=step_b_calls),sort_keys=True,indent=2)+'\n')
alpha_next_slots=sys.modules['test_alpha_next_queue'].SLOT_RECEIPTS
assert alpha_next_slots==[]  # alpha-next's draft has no slot (no absolute-time window), so its queue records no slot receipt
alpha_next_calls=sys.modules['test_alpha_next_queue'].ARGPARSE_RECEIPTS
assert alpha_next_calls and all(r['verdict']=='parsed' for r in alpha_next_calls)
Path('/out/alpha-next-queue-check.json').write_text(json.dumps(dict(slots=alpha_next_slots,argparse=alpha_next_calls),sort_keys=True,indent=2)+'\n')
c1_slots=sys.modules['test_alpha_next_c1_queue'].SLOT_RECEIPTS
assert c1_slots==[]  # the alpha-next-c1 draft has no slot (no absolute-time window), so its queue records no slot receipt
c1_calls=sys.modules['test_alpha_next_c1_queue'].ARGPARSE_RECEIPTS
assert c1_calls and all(r['verdict']=='parsed' for r in c1_calls)
Path('/out/alpha-next-c1-queue-check.json').write_text(json.dumps(dict(slots=c1_slots,argparse=c1_calls),sort_keys=True,indent=2)+'\n')
c1b_slots=sys.modules['test_alpha_next_c1b_queue'].SLOT_RECEIPTS
assert c1b_slots==[]  # the alpha-next-c1b draft has no slot (no absolute-time window), so its queue records no slot receipt
c1b_calls=sys.modules['test_alpha_next_c1b_queue'].ARGPARSE_RECEIPTS
assert c1b_calls and all(r['verdict']=='parsed' for r in c1b_calls)
Path('/out/alpha-next-c1b-queue-check.json').write_text(json.dumps(dict(slots=c1b_slots,argparse=c1b_calls),sort_keys=True,indent=2)+'\n')
am_slots=sys.modules['test_inwait_queue'].SLOT_RECEIPTS
assert am_slots==[]  # the inwait draft has no slot (no absolute-time window), so its queue records no slot receipt
am_calls=sys.modules['test_inwait_queue'].ARGPARSE_RECEIPTS
assert am_calls and all(r['verdict']=='parsed' for r in am_calls)
Path('/out/inwait-queue-check.json').write_text(json.dumps(dict(slots=am_slots,argparse=am_calls),sort_keys=True,indent=2)+'\n')
c5_slots=sys.modules['test_compose5_queue'].SLOT_RECEIPTS
assert c5_slots==[]  # no absolute-time window here either
c5_calls=sys.modules['test_compose5_queue'].ARGPARSE_RECEIPTS
assert c5_calls and all(r['verdict']=='parsed' for r in c5_calls)
Path('/out/compose5-queue-check.json').write_text(json.dumps(dict(slots=c5_slots,argparse=c5_calls),sort_keys=True,indent=2)+'\n')
record=dict(compose5_queue='PASS',compose5_argparse_calls=len(c5_calls),inwait_queue='PASS',inwait_queue_slots=am_slots,inwait_argparse_calls=len(am_calls),alpha_next_c1b_queue='PASS',alpha_next_c1b_queue_slots=c1b_slots,alpha_next_c1b_argparse_calls=len(c1b_calls),alpha_next_c1_queue='PASS',alpha_next_c1_queue_slots=c1_slots,alpha_next_c1_argparse_calls=len(c1_calls),alpha_next_queue='PASS',alpha_next_queue_slots=alpha_next_slots,alpha_next_argparse_calls=len(alpha_next_calls),step_b_queue='PASS',step_b_queue_slots=step_b_slots,step_b_argparse_calls=len(step_b_calls),clock_bound_frozen_excluded=clock_bound_removed,clock_bound_frozen_reason=step_b_clock_bound.REASON,step_a_queue='PASS',step_a_queue_slots=step_a_slots,step_a_argparse_calls=len(step_a_calls),alpha_perf_queue='PASS',alpha_perf_queue_slots=alpha_slots,alpha_perf_argparse_calls=len(alpha_calls),native39_trunk_queue='PASS',native39_trunk_queue_receipts=trunk_cases,native39_trunk_window_active_and_cleanup_s=active_and_cleanup,native35_p_c_continuation='PASS',native35_p_c_continuation_receipts=pc_continuation,native35_p_c_row4='PASS',native35_p_c_row4_receipts=pc_row4,native35_localize_off='PASS',native35_p_c_inert=inert_receipt,native35_p_c_cleanup_receipts=cleanup,native35_row4_l2='PASS',native35_p_b_row4_receipts=row4,native35_l2_cli_receipts=l2_cli,native35_l2_timing_receipts=l2_timing,
    native35_arm=role,native35_performance='PASS',native35_performance_receipts=perf_receipts,native35_builder='PASS',native35_builder_view=builder_view,native35_cli_receipts=native35_cli,native35_o_arm_readback=native35_readback,native35_window_active_and_cleanup_s=active_and_cleanup,probe_environment={k:os.environ[k] for k in ('FLASH_NATIVE34_DIAG','FLASH_NATIVE34_ARM','FLASH_NATIVE34_READBACK','FLASH_NATIVE35_ARM')},native34_readback='PASS',readback_check=readback_check,native34_protocol='PASS',protocol_receipts=protocol_receipts,draft_window_slots='PASS',draft_window_receipts=slot_receipts,arm=role,final_input_proof=True,native34=native34_receipt,owner_prelaunch=owner_prelaunch,native31=native31_receipt,receipt_policy='PASS',receipt_budget_bytes=policy.BYTES_PER_RANK,receipt_tail_s=policy.TAIL_S,row4_cli_validation='PASS',cli_receipts=cli_receipts,identity_receipts=identity_receipts,verdict='PASS' if result.wasSuccessful() and not result.skipped and result.testsRun==EXPECTED_TESTS else 'FAIL',tests_run=result.testsRun,skips=len(result.skipped),plugin_loader='PASS',identity_capture_check='PASS' if result.wasSuccessful() else 'FAIL',identity_timeout_negative_rejected=result.wasSuccessful(),mounted_modules=modules,cuda_initialized=False,gpu_execution=False,
    fixture='CPU driver-registration stub, synthetic launch/sync/control objects; no GPU model forward, stream or memory query')
Path('/out/probe.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
raise SystemExit(0 if record['verdict']=='PASS' else 1)
