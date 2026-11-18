"""Actual generated owner/client argv to final verdict, real schema defaults and IO stubs."""
import hashlib,json,os,subprocess,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'harness'),str(ROOT/'tools'),str(ROOT/'patches/s5_engine')]
import test_vision_fit_8x200k as fixture
import test_vision_fit_owner_protocol as owner_fixture
from g4_headroom_read import assess
PACKET=Path(os.environ.get('G4_HEADROOM_PACKET','<hub-dir>/cargo-target/flash-next/g4-headroom-build-20261118/build-1'))


def receipts():
    rows=[]
    for rank in range(4):
        for label in ('native-tp-pp-setup','native-tp-pp-warmup','full-capture','custom-ar-registration','split-capture'):
            for phase in ('before','after'):
                rows.append(dict(schema='g4-startup-phase/1',rank=rank,label=label,phase=phase,snapshot=dict(device_index=rank,total_bytes=24*2**30,free_bytes=2**30,allocated_bytes=22*2**30,reserved_bytes=23*2**30,observed_ns=1 if phase=='before' else 2)))
        rows.append(dict(schema='g4-score-view/1',rank=rank,workspace_lane=1,workspace_bytes=200*2**20,
            score_capacity_bytes=64*2**20,offset_bytes=136*2**20,extra_resident_bytes=0))
    return ['flash-s5 startup-vram '+json.dumps(row) for row in rows]


class CompleteProtocol(unittest.TestCase):
    def setUp(self):
        fixture.Protocol.setUp(self)
        memory=fixture.old.memory
        def sufficient(*args,**kwargs):
            value=memory(*args,**kwargs)
            value['min_free_bytes']={str(r):300*2**20 for r in range(4)}
            return value
        switch=patch.object(fixture.old,'memory',sufficient);switch.start();self.addCleanup(switch.stop)
    tearDown=fixture.Protocol.tearDown
    run_cli=fixture.Protocol.run_cli

    def test_generated_owner_whole_protocol(self):
        # The old helper provides only host control/filesystem/HTTP leaves; all clients/readers are packet bytes.
        with patch.object(fixture,'ROOT',PACKET),patch.object(fixture,'PACKET',PACKET):
            packet,run,env,argv=owner_fixture.OwnerProtocol.owner_fixture(self)
        logs=self.root/'fake-engine.log';logs.write_text('\n'.join(receipts())+'\n')
        docker=self.root/'bin/docker';text=docker.read_text()
        text=text.replace('echo "2026-11-16T17:00:00Z INFO 127.0.0.1 GET /health HTTP/1.1 200 OK";',
            'echo "2026-11-16T17:00:00Z INFO 127.0.0.1 GET /health HTTP/1.1 200 OK"; cat "'+str(logs)+'";')
        docker.write_text(text)
        result=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=180)
        (self.root/'owner.stdout').write_text(result.stdout);(self.root/'owner.stderr').write_text(result.stderr)
        detail='\n'.join(p.read_text()[-2000:] for p in run.glob('*.stderr.log'))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr+detail+' retained '+str(run))
        self.assertIn('G4_HEADROOM_OWNER FIT',result.stdout)
        read=json.loads((run/'headroom-read.json').read_text());self.assertEqual(read['verdict'],'PASS',read)
        record=json.loads((run/'fit-collect.json').read_text())
        self.assertEqual([s['prompt_tokens'] for s in record['stages']],[190320])
        self.assertEqual(len(record['stages'][0]['conversations']),8)
        self.assertIsNone(record['cold']);self.assertNotIn('reuse',record);self.assertNotIn('cancel',record)
        self.assertTrue(self.finished);self.assertTrue((run/'host-stop-intent.json').exists())
        print('G4_FULL_PROTOCOL_COMPLETE '+str(run))

    def test_readers_complete_and_missing_peer_mutant(self):
        delta=json.loads((PACKET/'vision-delta.json').read_text())
        self.assertEqual(delta['fit_plan_sha256'],hashlib.sha256((PACKET/'fit-plan.json').read_bytes()).hexdigest())
        binding=json.loads((PACKET/'g4-headroom-binding.json').read_text())
        for name in binding['changed_modules']:
            source=PACKET/'harness/native30_overlay/e3'/(name+'.py')
            self.assertEqual(delta['generated_module_shas'][name],hashlib.sha256(source.read_bytes()).hexdigest())
        raw=dict(scope='headroom-discriminator-only-not-full-G4')
        workload=dict(verdict='FIT',minimum_mib={str(r):300 for r in range(4)},raw=raw)
        complete=self.root/'synthetic-complete.json';complete.write_text(json.dumps(workload))
        logs=self.root/'synthetic-engine.log';logs.write_text('\n'.join(receipts())+'\n')
        argv=[sys.executable,'-B',str(PACKET/'tools/g4_headroom_read.py'),'--workload',str(complete),
            '--engine-log',str(logs),'--out',str(self.root/'read-complete.json')]
        result=self.run_cli(argv,'reader-complete');self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('G4_HEADROOM PASS',result.stdout)
        mutant=[line for line in receipts() if json.loads(line.split('vram ',1)[1])['rank']!=3]
        self.assertEqual(assess(workload,mutant)['verdict'],'HOLD')
        workload['minimum_mib']['1']=249.9
        self.assertEqual(assess(workload,receipts())['verdict'],'HOLD')
        workload['minimum_mib']['1']=float('nan')
        self.assertEqual(assess(workload,receipts())['verdict'],'HOLD')

if __name__=='__main__':unittest.main()
