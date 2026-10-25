import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import s5_onepair as d
ROOT=Path(__file__).resolve().parents[1]
TMP=ROOT.parent/'tmp'

class Grade(unittest.TestCase):
    def test_node_missing_or_mixed_not_proven(self):
        self.assertEqual(d.audit_grade([])['verdict'],'INCONCLUSIVE')
        good=dict(type='kernel',node=0,context_known=True,context_matches=True)
        self.assertEqual(d.audit_grade([good])['verdict'],'PASS')
        self.assertEqual(d.audit_grade([good,dict(good,context_matches=False)])['verdict'],'FAIL')
        self.assertEqual(d.audit_grade([dict(good,context_known=False)])['verdict'],'INCONCLUSIVE')
    def test_strict_rank_completion_and_counts(self):
        folder=Path(tempfile.mkdtemp(prefix='s5one-',dir=TMP));p=folder/'r.json'
        ranks=[dict(rank=i,pci_bus_id=str(i),case='collectives',context='green24',max_connections='8',
                    source_sha256={'module':'hash'},completed=True,durations_ms={'decode':[1.]*30,'prefill':[2.]*30}) for i in range(2)]
        doc=dict(pair='1,3',case='collectives',ranks=ranks)
        for i in range(2):Path(str(p)+f'.rank{i}.progress.json').write_text('{"phase":"completed"}')
        def grade(value):p.write_text(json.dumps(value));return d.grade(p)['verdict']
        self.assertEqual(grade(doc),'PASS')
        for mutate in [lambda v:v['ranks'][0]['durations_ms'].pop('prefill'),
                       lambda v:v['ranks'][0]['durations_ms']['decode'].pop(),
                       lambda v:v['ranks'][1].update(rank=0),
                       lambda v:v['ranks'][1].update(max_connections='1'),
                       lambda v:v['ranks'][0].update(completed=False)]:
            v=copy.deepcopy(doc);mutate(v);self.assertNotEqual(grade(v),'PASS')
        grade(doc);Path(str(p)+'.rank0.error.json').write_text('{"fatal_correctness":true}')
        self.assertEqual(d.grade(p)['verdict'],'FAIL')
    def test_queue_has_separate_fresh_rungs_and_no_engine_control(self):
        text=(ROOT/'harness/runs/q_s5_onepair.draft.sh').read_text()
        self.assertEqual(subprocess.run(['bash','-n'],input=text,text=True).returncode,0)
        for bad in ('systemctl stop','systemctl start','docker rm','--rm','/tmp/q','flash-reconfig'):
            self.assertNotIn(bad,text)
        rungs=[line for line in text.splitlines() if line.startswith('run_case ')]
        self.assertEqual([line.split()[1] for line in rungs],['decode-alone','prefill-alone','node-audit','collectives8','collectives32','prio8','prio1'])
        self.assertIn('idle_wait || return 1',text)
        self.assertIn('flash.s5.onepair=$token',text)
    def test_gpu_diagnostics_do_not_require_production(self):
        text=(ROOT/'harness/s5_onepair_gpu.py').read_text()
        self.assertIn('keep_graph=True',text)
        self.assertIn('graph.raw_cuda_graph()',text)
        self.assertIn('graph.instantiate()',text)
        self.assertIn("for source,stream in streams.items():",text)
        self.assertIn("for dest,launch in streams.items():",text)
        self.assertNotIn('systemctl',text)
        self.assertIn('wd.arm(',text)
if __name__=='__main__':unittest.main()
