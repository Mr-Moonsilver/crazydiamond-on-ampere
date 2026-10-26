import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import vmid_job2 as j
import vmid_support as s

HERE=Path(__file__).resolve().parent


def phase(plan):
    req={k:v for k,v in j.expected(plan).items() if k!='graph'}
    n=plan['replays']
    slot={'required':req,'node_types':j.expected(plan),'kernel_executions':req['kernel']*n,
          'copy_executions':req['memcpy']*n,'memset_executions':req['memset']*n,
          'node_execution_bounds':{k:{'min':n,'max':n} for k in req}}
    return dict(output_exact=True,serialized_stable=True,serialized_bytes=400,cleanup_ok=True,replays=n,
                native_node_inventory=[plan['kernels'],1,plan['copies']-2,1,plan['memsets'],plan['child'],plan['child'],0],
                static={'complete':True,'graph_slots':[slot]},
                window={'complete':True,'graph_slots':[copy.deepcopy(slot)],'flags':{},'reasons':[],
                        'expected_context':{'is_green':True,'sm_matching':1,'records':1}})


def complete():return {'phases':{name:phase(p) for name,p in j.PLANS.items()}}


class Tests(unittest.TestCase):
    def test_flat_pass_does_not_waive_child(self):
        g=j.grade(complete())
        self.assertEqual(g['q17_flat_node_families']['verdict'],'PASS')
        self.assertEqual(g['verdict'],'INCONCLUSIVE')
        self.assertTrue(any('recursive_child' in x for x in g['missing']))

    def test_capacity_uses_40_not_3_replays(self):
        r=complete();r['phases']['capacity']['window']['graph_slots'][0]['kernel_executions']=1386*3
        self.assertEqual(j.grade(r)['q17_flat_node_families']['verdict'],'INCONCLUSIVE')

    def test_missing_node_not_hidden_by_total(self):
        r=complete();r['phases']['capacity']['window']['graph_slots'][0]['node_execution_bounds']['kernel']={'min':39,'max':41}
        self.assertEqual(j.grade(r)['q17_flat_node_families']['verdict'],'INCONCLUSIVE')

    def test_wrong_copy_directions_fail(self):
        r=complete();r['phases']['mixed']['native_node_inventory'][1]=0
        self.assertEqual(j.grade(r)['verdict'],'FAIL')

    def test_wrong_graph_bytes_fail_even_with_child_unknown(self):
        r=complete();r['phases']['capacity']['serialized_stable']=False
        self.assertEqual(j.grade(r)['verdict'],'FAIL')

    def test_sticky_loss_never_pass(self):
        r=complete();r['phases']['capacity']['window']['flags']['buffer_refused']=True
        self.assertEqual(j.grade(r)['q17_flat_node_families']['verdict'],'INCONCLUSIVE')

    def test_empty_and_identity(self):
        self.assertEqual(j.grade({})['verdict'],'INCONCLUSIVE')
        r=complete();r['failures']=['pin'];self.assertEqual(j.grade(r)['verdict'],'FAIL')

    def test_monitor_cannot_target_other_container(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):s.monitor(Path(d),'<gpu-tenant>')

    def test_monitor_stops_only_job2(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'monitor.stop').touch();calls=[]
            def command(a):
                calls.append(a)
                if any('query-compute-apps' in x for x in a):return 0,'GPU-x, 1, python3, 2900\nGPU-x, 2, UnrealEditor, 2700\n'
                if 'top' in a:return 0,'PID\n1\n'
                return 0,''
            with patch.object(s,'command',command):s.monitor(out,'flashnext-job2')
            self.assertEqual([a for a in calls if 'stop' in a],[['sudo','-n','docker','stop','-t','5','flashnext-job2']])
            self.assertEqual(json.loads((out/'monitor.json').read_text())['own_peak_mib'],2900)

    def test_queue_and_native_headers(self):
        subprocess.run(['bash','-n',str(HERE/'runs/vmid_job2.sh')],check=True)
        subprocess.run(['g++','-std=c++17','-x','c++','-fsyntax-only','-I',str(HERE.parent/'reference/cuda-runtime-13'),
                        str(HERE/'native/vmid_graph_probe.cu')],check=True)
        text=(HERE/'runs/vmid_job2.sh').read_text()
        self.assertIn('--rm --gpus all --name flashnext-job2',text)
        self.assertNotIn('systemctl',text)
        self.assertNotIn('flashnext-job1',text)
        self.assertIn('vmid_graph_probe.so',text)


if __name__=='__main__':unittest.main()
