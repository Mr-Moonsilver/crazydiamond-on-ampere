"""Native35 grades: native-private writer catalogue, startup ledger, lifetime consolidation and exact R/O queue argv."""
import http.client
import json
import os
from pathlib import Path
import shlex
from string import Template
import tempfile
import unittest
from unittest.mock import patch

import e3_native34_causal as causal
from native34_readback import KEYS
from native35_grade import grade,native_writer,BUILDER_WRITTEN
import native35_ledger_grade as ledger
import native35_lifetime_grade as lifetime

HERE=Path(__file__).resolve().parent
CLI_RECEIPTS=[]


def _writer(provider,**fields):
    row=dict(provider=provider,completion_observed=True)
    if provider=='QSAMetadataBuilder.build':row.update(kernel='_build_qsa_metadata_kernel',num_reqs=1,num_tokens=1024,query_start_loc_cpu=[0,1024],mode='PREFIX',step=9)
    else:row.update(transaction=100,chunk=[0,1024],request_index=8)
    row.update(fields);return row


def _rows(rank,*,ordinary_before=False,last_ordinary=False,drop=None,value_verdict='PASS'):
    rows=[]
    for name in KEYS:
        if name==drop:continue
        if name in BUILDER_WRITTEN:
            history=[]
            if ordinary_before:history.append(_writer('QSAMetadataBuilder.build',num_reqs=4,num_tokens=4,query_start_loc_cpu=[0,1,2,3,4],mode='ARMED'))
            history.append(_writer('QSAMetadataBuilder.build'))
            if last_ordinary:history.append(_writer('QSAMetadataBuilder.build',num_reqs=4,num_tokens=4,query_start_loc_cpu=[0,1,2,3,4],mode='ARMED'))
        else:history=[_writer('LaneChunk.__init__')]
        rows.append(dict(event='native34-indexer-indirect-value',rank=rank,argument=name,verdict=value_verdict,device_values_observed=True,
            writer_chain=dict(observed=True,history=history,last_catalogued_writer=history[-1],history_overflow=False)))
    return rows


def _log(path,rank_rows,readback_verdict='PASS'):
    lines=[]
    for rank in (0,1):
        lines.append(json.dumps(dict(event='native34-indexer-readback-complete',rank=rank,device_values_observed=True,arguments=7,verdict=readback_verdict,monotonic_ns=5)))
        lines.extend(json.dumps(dict(r,monotonic_ns=6)) for r in rank_rows[rank])
        lines.append(json.dumps(dict(event='native34-indexer-kernel-end',rank=rank,post_fence=False,rank1_post_fence=False,monotonic_ns=9)))
    path.write_text(''.join('NATIVE30 '+line+'\n' for line in lines))


class WriterCatalogue(unittest.TestCase):
    def test_native_one_request_last_writer_on_both_ranks_passes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);log=root/'engine.err';arm=root/'arm.json';arm.write_text(json.dumps(dict(arm='O',verdict='PASS')))
            _log(log,{0:_rows(0),1:_rows(1)})
            r=grade([log],arm,require_readback=True,require_native_writer=True)
            self.assertEqual((r['verdict'],r['failure_class'],r['attribution']),('PASS',None,'NO_DEVICE_FAULT_RECORDED'))
            self.assertTrue(r['native_writer']['coverage']);self.assertEqual(len(r['native_writer']['operands']),14)
            self.assertFalse(r['release_qualified'])

    def test_ordinary_build_anywhere_in_native_storage_history_fails(self):
        for kind in ('ordinary_before','last_ordinary'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                root=Path(d);log=root/'engine.err';arm=root/'arm.json';arm.write_text(json.dumps(dict(arm='O',verdict='PASS')))
                _log(log,{0:_rows(0),1:_rows(1,**{kind:True})})
                r=grade([log],arm,require_readback=True,require_native_writer=True)
                self.assertEqual((r['verdict'],r['failure_class']),('FAIL','ENGINE'))
                self.assertTrue(any('ordinary-build-catalogued-on-native-storage' in f for f in r['native_writer']['failures']))

    def test_missing_operand_or_writer_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);log=root/'engine.err';arm=root/'arm.json';arm.write_text(json.dumps(dict(arm='O',verdict='PASS')))
            _log(log,{0:_rows(0),1:_rows(1,drop='k_work_metadata_ptr')})
            r=grade([log],arm,require_readback=True,require_native_writer=True)
            self.assertEqual((r['verdict'],r['failure_class']),('INCONCLUSIVE','HARNESS'))
            rows=_rows(1);rows[0]['writer_chain']=dict(observed=False,history=[],last_catalogued_writer=None,history_overflow=False)
            _log(root/'two.err',{0:_rows(0),1:rows})
            r=grade([root/'two.err'],arm,require_readback=True,require_native_writer=True)
            self.assertEqual((r['verdict'],r['attribution']),('INCONCLUSIVE','NATIVE_PRIVATE_WRITER_NOT_PROVED'))

    def test_value_violation_fails_even_with_native_writer(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);log=root/'engine.err';arm=root/'arm.json';arm.write_text(json.dumps(dict(arm='O',verdict='PASS')))
            _log(log,{0:_rows(0),1:_rows(1,value_verdict='VIOLATION')},readback_verdict='VIOLATION')
            r=grade([log],arm,require_readback=True,require_native_writer=True)
            self.assertEqual((r['verdict'],r['failure_class'],r['attribution']),('FAIL','ENGINE','INDIRECT_VALUE_OR_HOST_INTENT_VIOLATION'))

    def test_device_fault_keeps_engine_precedence_and_plain_o_is_native34(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);log=root/'engine.err';arm=root/'arm.json';arm.write_text(json.dumps(dict(arm='O',verdict='INCONCLUSIVE')))
            log.write_text('NATIVE30 '+json.dumps(dict(event='native34-indexer-kernel-exception',rank=1,monotonic_ns=20))+'\n')
            r=grade([log],arm,require_readback=True,require_native_writer=True)
            self.assertEqual((r['verdict'],r['failure_class'],r['attribution']),('FAIL','ENGINE','OVERLAPPED_PHASE_OR_BOUNDARY'))
            plain=grade([log],arm);self.assertIsNone(plain['native_writer']);self.assertFalse(plain['readback_required'])
            self.assertEqual(plain['verdict'],'FAIL')
            empty=root/'empty.log';empty.write_text('');self.assertEqual(grade([empty],arm)['verdict'],'INCONCLUSIVE')


def _startup(bytes_per_rank=4096,**over):
    def rank(i):
        fields=[dict(name='token_to_req_buffer',classification='private-tensor',tensor_bytes=bytes_per_rank-16),
                dict(name='layer_names',classification='private-container',tensor_bytes=0),
                dict(name='k_work_metadata_buffer',classification='private-tensor',tensor_bytes=16),
                dict(name='vllm_config',classification='shared-read-only',tensor_bytes=0)]
        storage=dict(native_builder_bytes=bytes_per_rank,native_builder_tensor_count=2,chunks=0,metadata_generation=0,metadata_generation_in_flight=False,
            native_builder_fields=[dict(builder='qsa_cache.QSAMetadataBuilder',layers=['layers.3'],fields=fields)],
            gdn_linear_alignment=dict(enabled=True,gpu_bytes=64,host_bytes=32,stock_aliases=0),lane_tables_bytes=128)
        storage.update(over)
        return dict(native_inventory=dict(native_attention_builder_storage=storage),memory=dict(free_bytes=10**9,total_bytes=24*10**9))
    return dict(workers={str(i):rank(i) for i in range(4)})


class Ledger(unittest.TestCase):
    def test_four_rank_ledger_passes_and_reports_bytes(self):
        r=ledger.grade(_startup());self.assertEqual(r['verdict'],'PASS');self.assertEqual({k:v['native_builder_bytes'] for k,v in r['ledger'].items()},{str(i):4096 for i in range(4)})

    def test_unclassified_field_in_flight_generation_or_alias_fails(self):
        s=_startup();s['workers']['2']['native_inventory']['native_attention_builder_storage']['native_builder_fields'][0]['fields'][0]['classification']='mystery'
        self.assertEqual(ledger.grade(s)['verdict'],'FAIL')
        self.assertEqual(ledger.grade(_startup(metadata_generation_in_flight=True))['verdict'],'FAIL')
        self.assertEqual(ledger.grade(_startup(gdn_linear_alignment=dict(stock_aliases=1)))['verdict'],'FAIL')
        self.assertEqual(ledger.grade(_startup(native_builder_bytes=4095))['verdict'],'FAIL')

    def test_missing_rank_or_storage_record_is_inconclusive(self):
        s=_startup();del s['workers']['3'];self.assertEqual(ledger.grade(s)['verdict'],'INCONCLUSIVE')
        s=_startup();del s['workers']['1']['native_inventory']['native_attention_builder_storage'];self.assertEqual(ledger.grade(s)['verdict'],'INCONCLUSIVE')
        self.assertEqual(ledger.grade({})['verdict'],'INCONCLUSIVE')


class Lifetime(unittest.TestCase):
    def _queue(self,root,**over):
        files=dict(lifecycle=dict(verdict='PASS',cancel_grade=dict(verdict='PASS'),reuse=dict(complete=True)),
            hygiene=dict(prepare=dict(route='full')),partial=dict(verdict='PASS',fails=[],missing=[]),
            completion=dict(rc=0),coverage=dict(verdict='PASS',fails=[],missing=[]),leak='sessions 8 rounds 3 bad 0 in 12.0s\n')
        files.update(over)
        (root/'lifecycle.json').write_text(json.dumps(files['lifecycle']));(root/'hygiene-full.json').write_text(json.dumps(files['hygiene']))
        (root/'partial-leak.grade.json').write_text(json.dumps(files['partial']));(root/'pressure.completion.json').write_text(json.dumps(files['completion']))
        (root/'pressure.coverage.json').write_text(json.dumps(files['coverage']));(root/'cross-session-leak.log').write_text(files['leak'])

    def test_all_present_pass(self):
        with tempfile.TemporaryDirectory() as d:
            self._queue(Path(d));r=lifetime.grade(d);self.assertEqual((r['verdict'],r['failures'],r['missing']),('PASS',[],[]))

    def test_leak_or_crash_fails_and_absence_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as d:
            self._queue(Path(d),leak='s3 r2 LEAK\nsessions 8 rounds 3 bad 1 in 9s\n');self.assertEqual(lifetime.grade(d)['verdict'],'FAIL')
        with tempfile.TemporaryDirectory() as d:
            self._queue(Path(d));(Path(d)/'engine-stream.err').write_text('CUDA error: an illegal memory access was encountered\n')
            self.assertEqual(lifetime.grade(d)['verdict'],'FAIL')
        with tempfile.TemporaryDirectory() as d:
            self._queue(Path(d),lifecycle=dict(verdict='INCONCLUSIVE',missing=['x']));self.assertEqual(lifetime.grade(d)['verdict'],'INCONCLUSIVE')
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(lifetime.grade(d)['verdict'],'INCONCLUSIVE')


def queue_argv(directory,arm):
    text=(HERE/f'runs/q18a_native35_{arm}.draft.sh').read_text()
    invocation=text.split('python3 "$H/e3_native34_causal.py"',1)[1].split('> "$Q/qual-row4.log"',1)[0]
    values=dict(H=str(HERE),Q=str(directory),PROBE_URL='http://127.0.0.1:1',MODEL='qwen38-w8a8-tp4',
        RUN_ID='native35-'+arm+'-cpu',EPOCH='3',SNAPSHOT_MAX_BYTES=str(512*1024**2),Q18A_STORE=str(directory/'private/store'))
    return [Template(token).substitute(values) for token in shlex.split(invocation.replace('\\\n',' '))]


class Frontend(unittest.TestCase):
    def test_r_and_o_queue_argv_validate_real_files_before_first_http(self):
        for arm in ('r','o'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);calls=[];fixture_key='native35-cpu-fixture'
                def refuse(connection,method,url,body=None,headers=None,**kwargs):
                    command=json.loads(body);calls.append(command['action'])
                    self.assertEqual((connection.host,connection.port),('127.0.0.1',1))
                    self.assertEqual(headers['Authorization'],'Bearer '+fixture_key)
                    self.assertEqual((command['action'],command['target_route'],command['engine_epoch']),('prepare','full',3))
                    raise ConnectionRefusedError('CPU-network-boundary')
                argv=queue_argv(root,arm);self.assertEqual(argv[argv.index('--arm')+1],'O')
                with patch.dict(os.environ,VLLM_API_KEY=fixture_key),patch.object(http.client.HTTPConnection,'request',refuse):
                    self.assertEqual(causal.main(argv),2)
                record=json.loads((root/'qual-row4.json').read_text())
                self.assertEqual(calls,['prepare']);self.assertEqual((record['arm'],record['failure_class']),('O','HARNESS'))
                self.assertFalse(record['release_qualified']);self.assertNotIn(fixture_key,json.dumps(record))
                CLI_RECEIPTS.append(dict(arm=arm.upper(),device_arm='O',verdict='PASS',first_network_call='prepare-full',network_contact=False))


if __name__=='__main__':unittest.main()
