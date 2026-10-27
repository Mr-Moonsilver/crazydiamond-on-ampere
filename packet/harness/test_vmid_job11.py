"""CPU qualification of the new owner-run composed-ledger VM probe."""
import copy
from pathlib import Path
import subprocess
import unittest
import vmid_job11 as j
import test_e3_engine_offers as f


def positive_case():
    _, bridge, ledger, _ = f.positive(graph_offers=40, graph_census=j.MIX)
    bank=copy.deepcopy(f.BANK)
    bank.update(capture_order_labels=['descriptor:'+str(i) for i in range(7)], descriptor_count=7)
    binding=ledger.binding.as_record()
    binding['bank_identity_sha256']=f.eo.bank_identity(bank)
    binding['descriptor_labels']={'graph:'+str(i):f.eo.descriptor_key(dict(f.DESCRIPTOR,num_tokens=128+i)) for i in range(7)}
    ledger.binding=f.eo._coerce_binding(binding)
    engine=ledger.seal()
    raw=f.raw_receipt(engine,replays=40,graph_census=j.MIX,verdict='INCONCLUSIVE',
                      native_reasons=['unattributed_expected_context'])
    raw['bank']=bank
    raw['static_structure']['graph_count']=7
    raw['coverage']['descriptor_inventory']=bank['capture_order_labels']
    raw['coverage']['unexercised_descriptors']=['graph:'+str(i) for i in range(1,7)]
    for i in range(1,7):
        static=copy.deepcopy(raw['static_structure']['graph_slots'][0]);static['label']='graph:'+str(i)
        raw['static_structure']['graph_slots'].append(static)
        row=copy.deepcopy(raw['replay']['graph_slots'][0])
        row.update(label='graph:'+str(i),descriptor='descriptor:'+str(i),
                   observed=dict.fromkeys(j.MIX,0),missing=dict(j.MIX),missing_nodes=['node:'+str(i)+':0'],
                   kernel_executions=0,copy_executions=0,memset_executions=0,
                   node_execution_bounds={k:dict(min=0,max=0) for k in j.MIX},
                   exercised=False,replay_qualified=False,complete=False)
        raw['replay']['graph_slots'].append(row)
    raw['replay']['mapping']['graph_ids']=7
    raw['replay']['mapping']['node_keys']=7*sum(j.MIX.values())
    ledger.deactivate()
    return dict(mode='positive',failures=[],missing=[],cleanup_ok=True,
                serialized_stable=True,output_exact=True,raw_unchanged=True,
                replays=40,graph_records=56760,raw_placement=raw,engine_accounting=engine,
                injections={name:dict(rejected=True) for name in
                            ('wrong_window','wrong_rank','one_missing_eager','raw_fail')})

class VM11Tests(unittest.TestCase):
    def tearDown(self):
        if f.eo._ACTIVE:f.eo._ACTIVE.deactivate()
    def test_real_composer_is_required(self):
        r=positive_case()
        self.assertEqual(j.case_grade(r)['verdict'],'PASS',j.case_grade(r))
        r['engine_accounting']['binding']['native_start_ns']+=1
        self.assertNotEqual(j.case_grade(r)['verdict'],'PASS')
    def test_unused_bank_cannot_hide_execution_or_missing_exercised_node(self):
        for mutate in (lambda r:r['raw_placement']['replay']['graph_slots'][1].update(kernel_executions=1),
                       lambda r:r['raw_placement']['replay']['graph_slots'][0]['missing'].update(kernel=1)):
            r=positive_case();mutate(r)
            self.assertNotEqual(j.case_grade(r)['verdict'],'PASS')
    def test_missing_evidence_negative_and_cleanup_refused(self):
        for field in ('raw_placement','engine_accounting','injections'):
            r=positive_case();r.pop(field)
            self.assertNotEqual(j.case_grade(r)['verdict'],'PASS')
        for mode in ('unoffered','primary'):
            r=positive_case();r['mode']=mode
            self.assertEqual(j.case_grade(r)['verdict'],'FAIL')
        r=positive_case();r['cleanup_ok']=False
        self.assertEqual(j.case_grade(r)['verdict'],'FAIL')
    def test_new_queue_owns_only_its_slot(self):
        p=Path(__file__).parent/'runs/vmid_job11.sh';s=p.read_text()
        subprocess.run(['bash','-n',str(p)],check=True)
        for text in ('Q=/opt/flashnext/runs/job11','flock -n 9','--name flashnext-job11',
                     '--network none','--memory 4g','PYTORCH_CUDA_ALLOC_CONF'):
            self.assertIn(text,s)
        self.assertNotIn('systemctl',s)
        self.assertIn('set_per_process_memory_fraction(.125,0)',Path(j.__file__).read_text())
        self.assertIn("'flashnext-job11'",Path(__file__).with_name('vmid_support.py').read_text())

if __name__=='__main__':unittest.main()
