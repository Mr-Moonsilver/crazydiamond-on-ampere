"""Coverage conformance at the actual NativeWorkerBridge receiver; CPU only."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import native30_host as host
import native30_policy as policy
from worker_bridge import NativeWorkerBridge

class Runner:pass

def runner():
    value=Runner();value.req_states=NS(max_num_reqs=10)
    value.block_tables=NS(block_tables=[],num_blocks=NS(np=None))
    value.kv_cache_config=NS(num_blocks=12)
    return value

def bridge(value):return NativeWorkerBridge(NS(runner=value,submit=lambda *a:None,open=lambda *a:None),rank=1)

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

class Coverage(unittest.TestCase):
    def test_frozen_native31_policy_misses_actual_bridge(self):
        oldhost=load('native31_before_host',Path(__file__).parent/'fixtures/native31_before_native30_host.py')
        with patch.dict(sys.modules,{'native30_host':oldhost}):
            oldpolicy=load('native31_before_policy',Path(__file__).parent/'fixtures/native31_before_native30_policy.py')
        value=runner();receiver=bridge(value)
        self.assertFalse(hasattr(receiver,'runner'));self.assertFalse(hasattr(receiver,'model_runner'))
        self.assertIs(receiver.dispatcher.runner,value)
        oldpolicy.qualification(lambda *a:None,value,dict(action='observer_begin'))
        s=oldhost.state(value,1)
        with oldpolicy.boundary(receiver,dict(rank=1,label='FIRST_ADMISSION',call_id=1,chunk_start=0,chunk_end=1024)):
            pass
        self.assertTrue(oldpolicy.window(s)['armed']);self.assertFalse(oldpolicy.window(s)['started']);self.assertFalse(s['active'])

    def test_actual_bridge_activates_and_legacy_boundary_captures_same_runner(self):
        value=runner();receiver=bridge(value);rows=[];previous=host._EMIT
        host._EMIT=lambda event,**fields:rows.append(dict(event=event,**fields));self.addCleanup(setattr,host,'_EMIT',previous)
        policy.qualification(lambda *a:None,value,dict(action='observer_begin'))
        fields=dict(rank=1,label='FIRST_ADMISSION',call_id=7,transaction=100,chunk_start=0,chunk_end=1024)
        marker=object()
        with policy.boundary(receiver,fields):
            self.assertIs(host._TLS.context['runner'],value)
            self.assertTrue(host._TLS.context['state']['active'])
            result=marker
        self.assertIs(result,marker)
        admission=[r for r in rows if r['event']=='host-policy-admission']
        self.assertEqual(len(admission),1);self.assertEqual(admission[0]['receiver_type'],'NativeWorkerBridge')
        self.assertEqual((admission[0]['transaction'],admission[0]['chunk_start'],admission[0]['chunk_end']),(100,0,1024))
        boundaries=[r for r in rows if r['event']=='host-boundary']
        self.assertEqual([r['boundary_position'] for r in boundaries],['before','after'])
        self.assertTrue(all(r['snapshot']['max_num_reqs']==10 for r in boundaries))

    def test_real_receiver_paths_and_missing_receiver_refuse_coverage(self):
        value=runner();self.assertIs(host.runner_for(NS(model_runner=value)),value)
        self.assertIs(host.runner_for(NS(runner=value)),value)
        self.assertIs(host.runner_for(bridge(value)),value)
        self.assertIsNone(host.runner_for(NS()))

    def test_same_stream_exception_escapes_and_tls_restores(self):
        value=runner();receiver=bridge(value);policy.qualification(lambda *a:None,value,dict(action='observer_begin'))
        previous=getattr(host._TLS,'context',None)
        with self.assertRaisesRegex(ValueError,'sentinel'):
            with policy.boundary(receiver,dict(rank=1,label='FIRST_ADMISSION',call_id=8,chunk_start=0,chunk_end=1024)):
                raise ValueError('sentinel')
        self.assertIs(getattr(host._TLS,'context',None),previous)

if __name__=='__main__':unittest.main()
