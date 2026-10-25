"""Startup discovery/config tests using tensor metadata, never GPU allocation."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import weakref

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
from startup import discover,install,Switch
from block_mirror import CpuTables
from retention_contract import ContractError
from serial_dma import TensorFacts
import runner_hooks


class Backend:
    def __init__(self,device):self.device=str(device)
    def facts(self,tensor):return tensor.facts


def fixture(pp=0,budget=100663296):
    env=dict(FLASH_E2_ARENA='1',FLASH_E2_DMA='1',VLLM_PP_LAYER_PARTITION='24,24',VLLM_QSA_KV_OFFLOAD='1')
    ctx={};groups=[]
    for k,i in enumerate(range(3+pp*24,24+pp*24,4)):
        name=f'model.language_model.layers.{i}.self_attn.attn'
        shape=(1340,1,4048,512);strides=(4048*512,4048*512,512,1)
        host=TensorFacts('cpu',shape,strides,'torch.bfloat16',2,True,True,(k+1)*2**34)
        dev=replace(host,device='cuda:0',pinned=False,address=(k+10)*2**34)
        layer=NS(_qsa_host_kv=NS(facts=host),kv_cache=NS(facts=dev),layer_name=name,
                 _qsa_kv_offload=True,_qsa_stage_min_rows=64,_qsa_stage_arena_bytes=budget)
        ctx[name]=layer
        # Include a group with no local layer, as the hybrid planner can do.
        groups.append(NS(layer_names=[f'remote.layers.{i}.attn'],kv_cache_spec=NS(block_size=4048)))
        groups.append(NS(layer_names=[name],kv_cache_spec=NS(block_size=4048)))
    runner=NS(device='cuda:0',parallel_config=NS(tensor_parallel_size=2,pipeline_parallel_size=2,
                  data_parallel_size=1,decode_context_parallel_size=1),
              vllm_config=NS(kv_transfer_config=None,ec_transfer_config=None),
              model_config=NS(multimodal_config=NS(language_model_only=True)),
              compilation_config=NS(mode=0,cudagraph_mode=NS(name='FULL_DECODE_ONLY'),static_forward_context=ctx),
              cache_config=NS(kv_sharing_fast_prefill=False),
              scheduler_config=NS(max_num_batched_tokens=1024),
              kv_cache_config=NS(num_blocks=1340,kv_cache_groups=groups),
              block_tables=NS(_flash_e2_tables=CpuTables([1]*len(groups),100)),
              is_encoder_decoder=False,is_pooling_model=False,decode_query_len=1,
              is_first_pp_rank=pp==0,is_last_pp_rank=pp==1)
    return runner,env


class StartupTests(unittest.TestCase):
    def test_discovers_both_stages_with_empty_remote_groups_and_24_page_baseline(self):
        for pp in (0,1):
            r,e=fixture(pp);f=discover(r,e)
            self.assertEqual(f.pp,pp);self.assertEqual(f.arena_pages,24)
            self.assertEqual(set(f.layer_groups.values()),{1,3,5,7,9,11})
            self.assertEqual(f.arena_pages*4048*1024,99483648)
            self.assertEqual([(lo,min(lo+f.arena_pages,50)) for lo in range(0,50,f.arena_pages)],[(0,24),(24,48),(48,50)])

    def test_arena_geometry_is_discovered_not_forced_to_slice_or_budget(self):
        for budget,pages in ((95338496,23),(100663296,24),(192*1024**2,48)):
            r,e=fixture(budget=budget);f=discover(r,e)
            self.assertEqual(f.arena_pages,pages)

    def test_disabled_and_profile_make_no_backend_and_do_not_require_live_sources(self):
        def fail(_):self.fail('backend constructed')
        self.assertIsNone(install(object(),env={},backend_factory=fail))
        self.assertIsNone(install(object(),True,env={'FLASH_E2_ARENA':'1'},backend_factory=fail))

    def test_install_binds_weak_refs_and_declares_zero_kv_allocation(self):
        r,e=fixture();lines=[]
        x=install(r,env=e,backend_factory=Backend,emit=lines.append)
        self.assertIs(r._flash_e2_lifecycle,x)
        self.assertIsNone(x.dma.arena)
        self.assertEqual(x.mirror.arena.arena_pages,24)
        for name,layer in r.compilation_config.static_forward_context.items():
            self.assertIsInstance(layer._flash_e2_lifecycle_ref,weakref.ReferenceType)
            self.assertIs(layer._flash_e2_lifecycle_ref(),x)
            self.assertEqual(layer._flash_e2_layer_name,name)
        ledger=json.loads(lines[0].split(' ',1)[1])
        self.assertEqual(ledger['incremental_kv_bytes'],0)
        self.assertEqual(ledger['baseline_arena_bytes'],99483648)
        self.assertEqual(ledger['reserve_bytes'],536870912)
        self.assertNotIn('address',lines[0]);self.assertNotIn('request',lines[0])
        with self.assertRaises(ContractError):install(r,env=e,backend_factory=Backend,emit=lines.append)

    def test_unsupported_execution_modes_rejected_before_backend(self):
        changes=[lambda r,e:setattr(r.parallel_config,'tensor_parallel_size',4),
                 lambda r,e:setattr(r.parallel_config,'data_parallel_size',2),
                 lambda r,e:setattr(r.parallel_config,'decode_context_parallel_size',2),
                 lambda r,e:e.update(VLLM_PP_LAYER_PARTITION='26,22'),
                 lambda r,e:setattr(r.vllm_config,'kv_transfer_config',object()),
                 lambda r,e:setattr(r.vllm_config,'ec_transfer_config',object()),
                 lambda r,e:setattr(r.compilation_config,'mode',3),
                 lambda r,e:setattr(r.compilation_config,'cudagraph_mode',NS(name='FULL')),
                 lambda r,e:setattr(r.model_config.multimodal_config,'language_model_only',False),
                 lambda r,e:setattr(r.scheduler_config,'max_num_batched_tokens',512),
                 lambda r,e:setattr(r.cache_config,'kv_sharing_fast_prefill',True),
                 lambda r,e:e.update(QSA_RETAIN_HISTORY='1'),
                 lambda r,e:e.update(FLASH_S4_SEGMENT='1'),
                 lambda r,e:e.update(FLASH_PP_GROUPING='1'),
                 lambda r,e:e.update(VLLM_USE_BREAKABLE_CUDAGRAPH='1'),
                 lambda r,e:e.update(CUDA_MPS_PIPE_DIRECTORY='/mps'),
                 lambda r,e:e.update(VLLM_QSA_KV_OFFLOAD='0')]
        changes.extend((lambda r,e:e.update(FLASH_E2_VERIFY='bad'),
                        lambda r,e:e.update(FLASH_E2_VERIFY='1',FLASH_E2_REPORT_DIR='/other')))
        for name in ('speculator','speculative_config','adaptive_verification','pcp_manager','ubatch_runner','fast_prefill','lora_config'):
            changes.append(lambda r,e,name=name:setattr(r,name,object()))
        for change in changes:
            r,e=fixture();change(r,e)
            with self.assertRaises(ContractError):
                install(r,env=e,backend_factory=lambda _:self.fail('allocated before rejection'))

    def test_missing_duplicate_remote_or_misbound_layers_rejected(self):
        for defect in ('missing','duplicate','placement','name','stage-off','budget','blocksize','expanded'):
            r,e=fixture();ctx=r.compilation_config.static_forward_context;name=next(iter(ctx));layer=ctx[name]
            if defect=='missing':layer._qsa_host_kv=None
            elif defect=='duplicate':r.kv_cache_config.kv_cache_groups[0].layer_names=[name]
            elif defect=='placement':r.is_first_pp_rank=False;r.is_last_pp_rank=True
            elif defect=='name':layer.layer_name='other'
            elif defect=='stage-off':layer._qsa_stage_min_rows=0
            elif defect=='budget':layer._qsa_stage_arena_bytes=123
            elif defect=='blocksize':r.kv_cache_config.kv_cache_groups[1].kv_cache_spec.block_size=4096
            else:r.block_tables._flash_e2_tables.bpk=(2,)*12
            with self.subTest(defect=defect),self.assertRaises(ContractError):discover(r,e)

    def test_registered_source_layout_and_uva_view_checked_before_layer_attach(self):
        for defect in ('unpinned','host-alias','uva-dtype','uva-shape'):
            r,e=fixture();layers=list(r.compilation_config.static_forward_context.values())
            if defect=='unpinned':layers[0]._qsa_host_kv.facts=replace(layers[0]._qsa_host_kv.facts,pinned=False)
            elif defect=='host-alias':layers[1]._qsa_host_kv=layers[0]._qsa_host_kv
            elif defect=='uva-dtype':layers[0].kv_cache.facts=replace(layers[0].kv_cache.facts,dtype='torch.float32')
            else:layers[0].kv_cache.facts=replace(layers[0].kv_cache.facts,shape=(10,1,4048,512))
            with self.assertRaises(ContractError):install(r,env=e,backend_factory=Backend,emit=lambda _:None)
            self.assertFalse(any(hasattr(l,'_flash_e2_lifecycle_ref') for l in layers))

    def test_hot_attach_and_missing_cpu_mirror_rejected(self):
        for hot in (False,True):
            r,e=fixture()
            if hot:r.block_tables._flash_e2_tables.append(0,([1],)*12,True)
            else:r.block_tables._flash_e2_tables=None
            with self.assertRaises(ContractError):discover(r,e)

    def test_switch_is_read_only_fail_closed_boolean_and_fixed_path(self):
        with patch('startup.Path.exists',return_value=True):self.assertFalse(Switch({'FLASH_E2_DMA':'1'})())
        with patch('startup.Path.exists',return_value=False):self.assertTrue(Switch({'FLASH_E2_DMA':'1'})())
        self.assertFalse(Switch({})())
        with self.assertRaises(ContractError):Switch({'FLASH_E2_DMA':'yes'})
        with self.assertRaises(ContractError):Switch({'FLASH_E2_OFF_FILE':'/other'})

    def test_startup_hook_sits_after_real_binding_and_is_reversible(self):
        source='class GPUModelRunner:\n    def initialize_kv_cache(self, is_profiling=False):\n        original_binding()\n'
        changed=runner_hooks.runner_startup(source);ast.parse(changed)
        self.assertLess(changed.index('original_binding()'),changed.index('_install_flash_e2(self'))
        self.assertEqual(changed.replace('\n'+runner_hooks.STARTUP,''),source)
        with self.assertRaises(ValueError):runner_hooks.runner_startup(changed)


if __name__=='__main__':unittest.main()
