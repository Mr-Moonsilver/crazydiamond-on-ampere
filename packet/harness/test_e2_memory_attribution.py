"""CPU-only pairing/cadence tests; fake nvidia-smi, never a GPU query."""
import copy
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import e2_memory_attribution as m

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    samples=[dict(start_s=t,end_s=t+0.02,gpus={str(i):dict(used_mib=100,total_mib=200) for i in range(4)})
             for t in (9,11)]
    ledger=dict(per_rank={'Worker_'+rank:dict(gpu=i) for i,rank in enumerate(m.RANKS)})
    sides={rank:dict(rank=rank,failed=False,active=False,layers={'layer':dict(calls=1,buckets=1)},
                    memory=dict(samples=6,missing=[],highwater={})) for rank in m.RANKS}
    records=[dict(rank=rank,wall_s=10,rpc_sequence=1,layer='layer',boundary='output_reference_snapshot',
                  counters={'reserved_bytes.all.current':80*m.MIB,'allocated_bytes.all.current':70*m.MIB})
             for rank in m.RANKS]
    lines=['INFO FLASH_E2_MEMORY '+json.dumps(r) for r in records]
    return lines,samples,ledger,sides


class AttributionTests(unittest.TestCase):
    def test_pairs_current_not_peak_and_reports_both_brackets_and_skew(self):
        lines,samples,ledger,sides=fixture()
        r=m.report(lines,samples,ledger,sides)
        self.assertEqual(r['verdict'],'OBSERVED')
        p=r['pairs'][0]
        self.assertEqual(p['reserved_minus_allocated_bytes'],10*m.MIB)
        self.assertEqual(p['physical']['non_allocator_used_estimate_bytes'],20*m.MIB)
        self.assertAlmostEqual(p['physical']['skew_s'],-0.99)
        self.assertEqual(p['physical']['before']['start_s'],9)
        self.assertEqual(p['physical']['after']['start_s'],11)
        self.assertIn('not simultaneous',r['scope'])

    def test_stale_or_slow_samples_cannot_be_paired(self):
        lines,samples,ledger,sides=fixture()
        for variant in ('stale','slow','jump'):
            s=copy.deepcopy(samples)
            for p in s:
                if variant=='stale':p['start_s']+=100;p['end_s']+=100
                if variant=='slow':p['end_s']+=1
                if variant=='jump':p['error']='ClockStep'
            with self.subTest(variant=variant):
                r=m.report(lines,s,ledger,sides)
                self.assertEqual(r['verdict'],'INCONCLUSIVE')
                self.assertFalse(r['pairs'][0]['physical']['paired'])

    def test_negative_residual_remains_visible_not_a_native_memory_claim(self):
        lines,samples,ledger,sides=fixture()
        for s in samples:
            for g in s['gpus'].values():g['used_mib']=50
        r=m.report(lines,samples,ledger,sides)
        self.assertEqual(r['pairs'][0]['physical']['non_allocator_used_estimate_bytes'],-30*m.MIB)
        self.assertIn('not proven native',r['scope'])

    def test_missing_mapping_sidecar_or_counter_proof_is_inconclusive(self):
        for variant in ('mapping','sidecar','samples','missing'):
            lines,samples,ledger,sides=fixture()
            if variant=='mapping':ledger['per_rank']['Worker_PP0_TP0']['gpu']=3
            if variant=='sidecar':sides.pop('PP0_TP0')
            if variant=='samples':sides['PP0_TP0']['memory']['samples']=7
            if variant=='missing':sides['PP0_TP0']['memory']['missing']=['reserved_bytes.all.peak']
            with self.subTest(variant=variant):
                self.assertEqual(m.report(lines,samples,ledger,sides)['verdict'],'INCONCLUSIVE')

    def test_timing_is_matched_workload_difference_not_causal_overhead(self):
        args=fixture()
        base={p:dict(seconds=10,prompt_tokens=200,completion_tokens=8,cached_tokens=0) for p in ('fresh','append')}
        drive=copy.deepcopy(base);drive['fresh']['seconds']=11
        r=m.report(*args,drive,base)
        self.assertEqual(r['timing_vs_q16h']['fresh']['ratio'],1.1)
        self.assertIn('not isolated telemetry overhead',r['scope'])
        drive['append']['cached_tokens']=100
        self.assertEqual(m.report(*args,drive,base)['verdict'],'INCONCLUSIVE')

    def test_gpu_csv_rejects_missing_duplicate_invalid_rows(self):
        text='\n'.join(f'{i}, 100, 200' for i in range(4))
        self.assertEqual(len(m.parse_gpu(text)),4)
        for bad in (text+'\n0, 1, 2',text.replace('3, 100, 200',''),text.replace('100','300'),text+', extra'):
            with self.assertRaises(ValueError):m.parse_gpu(bad)

    def test_sampler_same_query_writes_legacy_csv_and_time_brackets_then_exits_on_term(self):
        out=Path(tempfile.mkdtemp(prefix='e2-clock-',dir=ROOT.parent/'tmp'))
        handlers={}
        text='\n'.join(f'{i}, 100, 200' for i in range(4))+'\n'
        def fake_run(cmd,**kwargs):
            self.assertEqual(cmd[0],'nvidia-smi')
            self.assertEqual(kwargs['timeout'],10)
            handlers[signal.SIGTERM]()
            return subprocess.CompletedProcess(cmd,0,text,'')
        with patch.object(m.signal,'signal',side_effect=lambda sig,fn:handlers.update({sig:fn})), \
             patch.object(m.subprocess,'run',side_effect=fake_run):
            m.sample(out/'gpu.csv',out/'gpu.timed.jsonl')
        self.assertEqual(m.parse_gpu((out/'gpu.csv').read_text()),m.parse_gpu(text))
        records=[json.loads(s) for s in (out/'gpu.timed.jsonl').read_text().splitlines()]
        self.assertEqual(records[-1]['kind'],'stopped')
        r=records[0]
        self.assertEqual(r['clock'],'unix_realtime_seconds')
        self.assertGreaterEqual(r['end_s'],r['start_s'])
        self.assertEqual(r['gpus'],m.parse_gpu(text))

    def test_q16i_settings_enable_only_oracle_phases_and_keep_pool(self):
        script='unset E2_KV_BYTES; E2_MEMORY_STATS=1; QNAME=test; source "$1"; declare -n a="$2"; printf "%s\\0" "${a[@]}"'
        for cfg in ('DENSE','VERIFY','CONTROL','SHIP'):
            p=subprocess.run(['bash','-uc',script,'fixture',str(ROOT/'harness/runs/q16_settings.sh'),'E2_'+cfg],
                             capture_output=True,check=True)
            words=p.stdout.decode().rstrip('\0').split('\0')
            self.assertEqual([w for w in words if w.startswith('GPU_KV_BYTES=')],['GPU_KV_BYTES=1500000000'])
            env=next(w for w in words if w.startswith('EXTRA_ENV='))
            self.assertEqual('FLASH_E2_MEMORY_STATS=1' in env,cfg in ('DENSE','VERIFY'))
            if cfg in ('DENSE','VERIFY'):self.assertIn('FLASH_E2_VERIFY=1',env)

    def test_gate_peak_parser_requires_clean_stop_and_preserves_legacy_peak_format(self):
        source=(ROOT/'harness/runs/q16_e2_serial_v3e.draft.sh').read_text()
        code=source.split("<<'PYS'\n",1)[1].split('\nPYS',1)[0]
        out=Path(tempfile.mkdtemp(prefix='e2-peak-',dir=ROOT.parent/'tmp'))
        raw=out/'raw.csv';raw.write_text(''.join(f'{i}, 100, 200\n' for i in range(4)))
        timed=out/'timed.jsonl'
        sample=dict(start_s=1,end_s=1.01,gpus=m.parse_gpu(raw.read_text()))
        timed.write_text(json.dumps(sample)+'\n')
        import sys
        p=subprocess.run([sys.executable,'-',str(raw),str(out/'bad.json'),str(timed)],
                         input=code,text=True,capture_output=True)
        self.assertNotEqual(p.returncode,0)
        self.assertFalse((out/'bad.json').exists())
        with timed.open('a') as f:f.write(json.dumps(dict(kind='stopped'))+'\n')
        p=subprocess.run([sys.executable,'-',str(raw),str(out/'good.json'),str(timed)],
                         input=code,text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr)
        result=json.loads((out/'good.json').read_text())
        self.assertEqual(result,dict(samples=1,gpus={str(i):dict(max_used_mib=100,total_mib=200) for i in range(4)}))


if __name__=='__main__':unittest.main()
