import copy
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
import vmid_job5 as j
import vmid_support as s


def phase(n=40):
    graphs=[dict(role=role,index=i,native_node_inventory=[1386,1,24,1,7,0,0,0],
                 serialized_stable=True,serialized_bytes=215162)
            for role,count in (('startup_pressure',8),('bound',7)) for i in range(count)]
    slots=[]
    for slot in range(7):
        count=n if slot==0 else 0
        slots.append(dict(slot=slot,node_count=1419,required=dict(j.MIX),exercised=bool(count),complete=bool(count),
            kernel_executions=1386*count,copy_executions=26*count,memset_executions=7*count,
            node_execution_bounds={k:dict(min=count,max=count) for k in j.MIX}))
    counts=dict(kernel_records=n*1386,copy_records=n*33,resource_events=194280,
                ring_overflows=dict.fromkeys(('aliases','resource','kernel','copy','context'),0))
    counts.update(dict.fromkeys(('dropped_records_total','buffers_refused','drop_query_errors','parse_errors','callback_errors',
        'unknown_node_records','ambiguous_node_records','kind_mismatch_records','known_wrong_context_records',
        'unknown_activity_records','unknown_resource_events'),0))
    w=dict(complete=True,limits=dict(j.LIMITS),graph_slots=slots,counts=counts,flags={},reasons=[],
           mapping=dict(node_keys=9933,node_aliases=9933,ambiguous_graph_ids=False,node_overlap=False,alias_unknown=0),
           expected_context=dict(is_green=True,sm_matching=1,records=1))
    return dict(replays=n,bound_graphs=7,startup_pressure_graphs=8,graphs=graphs,output_exact=True,cleanup_ok=True,
                native_schema_valid=True,limits=dict(j.LIMITS),split=dict(decode=44,prefill=38,total=82),
                static=dict(complete=True,limits=dict(j.LIMITS),graph_slots=copy.deepcopy(slots)),window=w,
                occupancy=j.occupancy(w),expected_records=dict(kernel=n*1386,copy=n*33),allocator_peak_reserved=2097152)


class Tests(unittest.TestCase):
    def test_short_complete(self):
        self.assertEqual(j.phase_grade(phase(),40)['verdict'],'PASS')

    def test_long_arithmetic_cannot_fit_pinned_rings(self):
        self.assertEqual(1386*1200,1663200);self.assertEqual(33*1200,39600)
        self.assertGreater(1386*1200,j.LIMITS['max_kernel_records'])
        self.assertGreater(33*1200,j.LIMITS['max_copy_records'])
        # Even forged loss-free totals cannot pass by exceeding the capacities.
        self.assertEqual(j.phase_grade(phase(1200),1200)['verdict'],'FAIL')

    def test_overflow_is_capacity_failure_not_harness_excuse(self):
        r=phase(1200)
        r['window']['counts']['kernel_records']=131072
        r['window']['counts']['copy_records']=32768
        r['window']['counts']['ring_overflows'].update(kernel=1532128,copy=6832)
        r['window']['complete']=False;r['occupancy']=j.occupancy(r['window'])
        g=j.phase_grade(r,1200)
        self.assertEqual(g['verdict'],'FAIL')
        self.assertIn('capacity_overflow:kernel',g['failures'])
        self.assertIn('capacity_overflow:copy',g['failures'])

    def test_phase_verdicts_separate(self):
        r={'phases':{'short':phase(),'long':phase(1200)}}
        g=j.grade(r)
        self.assertEqual(g['phases']['short']['verdict'],'PASS')
        self.assertEqual(g['verdict'],'FAIL')

    def test_no_counter_loss_waiver(self):
        for field in ('dropped_records_total','buffers_refused','unknown_node_records','kind_mismatch_records'):
            r=phase();r['window']['counts'][field]=1
            self.assertEqual(j.phase_grade(r,40)['verdict'],'FAIL',field)

    def test_all_seven_serializer_checks_required(self):
        r=phase();r['graphs'][-1]['serialized_stable']=False
        self.assertEqual(j.phase_grade(r,40)['verdict'],'FAIL')
        r=phase();r['graphs'][-1].pop('serialized_stable')
        self.assertEqual(j.phase_grade(r,40)['verdict'],'INCONCLUSIVE')

    def test_bank_inventory_not_single_graph(self):
        r=phase();r['static']['graph_slots'].pop()
        self.assertEqual(j.phase_grade(r,40)['verdict'],'INCONCLUSIVE')

    def test_inactive_bank_is_not_missing_replay(self):
        r=phase();self.assertFalse(r['window']['graph_slots'][6]['complete'])
        self.assertEqual(j.phase_grade(r,40)['verdict'],'PASS')
        r['window']['graph_slots'][6]['kernel_executions']=1386
        self.assertNotEqual(j.phase_grade(r,40)['verdict'],'PASS')

    def test_totals_do_not_hide_per_node_loss(self):
        r=phase();r['window']['graph_slots'][0]['node_execution_bounds']['memcpy']={'min':39,'max':41}
        self.assertEqual(j.phase_grade(r,40)['verdict'],'INCONCLUSIVE')

    def test_resource_scale_cannot_be_silently_missed(self):
        r=phase();r['window']['counts']['resource_events']=90659;r['occupancy']=j.occupancy(r['window'])
        self.assertIn('startup_resource_volume_below_full8',j.phase_grade(r,40)['missing'])

    def test_native_limits_cannot_be_enlarged(self):
        r=phase();r['limits']['max_kernel_records']=2000000
        self.assertEqual(j.phase_grade(r,40)['verdict'],'FAIL')

    def test_occupancy_receipt_must_match(self):
        r=phase();r['occupancy']['aliases']['reported_count']=1
        self.assertEqual(j.phase_grade(r,40)['verdict'],'INCONCLUSIVE')
        self.assertEqual(j.occupancy(phase()['window'])['aliases']['remaining_lower_bound'],65536-9933)
        self.assertIn('upper bound',j.occupancy(phase()['window'])['resource_events']['basis'])

    def test_missing_counter_cannot_pass(self):
        r=phase();r['window']['counts'].pop('dropped_records_total')
        self.assertEqual(j.phase_grade(r,40)['verdict'],'INCONCLUSIVE')
        self.assertEqual(j.grade({})['verdict'],'INCONCLUSIVE')

    def test_wrong_context_and_outputs_fail(self):
        r=phase();r['window']['expected_context']['is_green']=False
        self.assertEqual(j.phase_grade(r,40)['verdict'],'FAIL')
        r=phase();r['output_exact']=False
        self.assertEqual(j.phase_grade(r,40)['verdict'],'FAIL')

    def test_queue_reuses_native_probe_and_limits(self):
        root=Path(__file__).resolve().parent
        script=root/'runs/vmid_job5.sh'
        subprocess.run(['bash','-n',str(script)],check=True)
        self.assertEqual(script.read_text(),(root/'runs/vmid_job2b.sh').read_text().replace('job2b','job5').replace('JOB2B','JOB5'))
        source=(root/'vmid_job5.py').read_text()
        self.assertIn('set_per_process_memory_fraction(.125,0)',source)
        self.assertNotIn('ObserverLimits(',source.replace('ro.ObserverLimits()',''))
        self.assertIn('ro.ObserverLimits()',source)

    def test_sidecar_owns_only_job5(self):
        import tempfile,json
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'monitor.stop').touch();calls=[]
            def command(args):
                calls.append(args)
                if any('query-compute-apps' in x for x in args):return 0,'GPU-x, 1, python3, 2900\nGPU-x, 2, UnrealEditor, 2700\n'
                if 'top' in args:return 0,'PID\n1\n'
                return 0,''
            with patch.object(s,'command',command):s.monitor(out,'flashnext-job5')
            self.assertEqual([x for x in calls if 'stop' in x],[['sudo','-n','docker','stop','-t','5','flashnext-job5']])
            self.assertEqual(json.loads((out/'monitor.json').read_text())['own_peak_mib'],2900)

if __name__=='__main__':unittest.main()
