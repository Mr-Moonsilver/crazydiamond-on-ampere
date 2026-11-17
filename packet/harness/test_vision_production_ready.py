import contextlib,hashlib,io,json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import vision_production_ready as ready

class Response:
 status=200
 def __enter__(self):return self
 def __exit__(self,*_):pass

class Readiness(unittest.TestCase):
 def setUp(self):
  self.tmp=Path(tempfile.mkdtemp(prefix='w27-ready-'));self.packet=self.tmp/'packet';self.packet.mkdir();(self.packet/'PACKET-SHA256SUMS').write_text('CPU_SCHEMA_LEAF\n')
  for folder,name in ((self.packet,'PACKET-SHA256SUMS'),(self.packet/'harness','SHA256SUMS'),(self.packet/'harness/native30_overlay','SHA256SUMS')):
   folder.mkdir(parents=True,exist_ok=True);(folder/'fixture.txt').write_text('CPU_LEAF');(folder/name).write_text(hashlib.sha256(b'CPU_LEAF').hexdigest()+'  fixture.txt\n')
  self.state=self.tmp/'control/ready.json';self.key=self.tmp/'key.env';self.key.write_text('VLLM_API_KEY='+'a'*64);self.key.chmod(0o600)
  self.argv=['--packet',str(self.packet),'--state-file',str(self.state),'--key-file',str(self.key)]
  self.released=dict(schema=1,action='status',run_id='test-boot',engine_epoch=10,native_ready=True,auto_arm=dict(manual_owned=False,selection_seq=1))
 def drive(self,statuses,leaf=None,extra=()):
  real_run=subprocess.run
  def command(argv,**kw):
   if argv[0]=='sha256sum':return real_run(argv,**kw)
   if leaf is None:raise AssertionError('bound lane must not run a mutable client')
   return leaf(argv,**kw)
  with patch.object(ready,'status',side_effect=statuses),patch.object(ready.urllib.request,'urlopen',return_value=Response()),patch.object(ready.subprocess,'run',side_effect=command),contextlib.redirect_stdout(io.StringIO()) as stdout:
   code=ready.main([*self.argv,*extra]);return code,stdout.getvalue()
 def test_existing_selected_lane_is_checked_without_pause_or_reselection(self):
  code,text=self.drive([self.released]);self.assertEqual(code,0);self.assertIn('PASS',text);self.assertTrue(json.loads(self.state.read_text())['ready'])
 def test_bound_unreleased_lane_refuses_without_owning_it(self):
  value=dict(self.released,auto_arm=dict(manual_owned=True,selection_seq=0))
  with self.assertRaisesRegex(ValueError,'bound native lane'):self.drive([value])
  self.assertFalse(json.loads(self.state.read_text())['ready'])
 def test_fresh_start_actual_subcommand_argv_and_complete_startup_idle_select(self):
  calls=[]
  def child(command,**kw):
   calls.append(command);out=Path(command[command.index('--out')+1]);out.parent.mkdir(parents=True,exist_ok=True)
   out.write_text(json.dumps(dict(verdict='PASS',engine_epoch=10)))
   if 'select' in command:self.released['run_id']=command[command.index('--run-id')+1]
   return subprocess.CompletedProcess(command,0)
  cold=dict(schema=1,action='status',run_id=None,auto_arm=dict(manual_owned=True,selection_seq=0))
  code,text=self.drive([cold,self.released],child);self.assertEqual(code,0);self.assertIn('PASS',text)
  self.assertEqual(len(calls),3);self.assertIn('e3_split_startup_grade.py',calls[0]);self.assertIn('full-idle',calls[1]);self.assertIn('select',calls[2])
  self.assertTrue(all(command[:3]==[ready.sys.executable,'-B',str(self.packet/'serve_lane.py')] for command in calls))
  self.assertTrue(json.loads(self.state.read_text())['ready'])
 def test_refused_bootstrap_never_publishes_ready_and_invalidation_is_typed(self):
  cold=dict(schema=1,action='status',run_id=None)
  with self.assertRaises(ValueError):self.drive([cold],lambda *a,**kw:subprocess.CompletedProcess(a,2))
  self.assertFalse(json.loads(self.state.read_text())['ready'])
  code,text=self.drive([],extra=['--invalidate-only']);self.assertEqual(code,0);self.assertIn('INVALIDATED',text)
 def test_missing_or_zero_selection_never_becomes_ready(self):
  for auto in ({},dict(manual_owned=False,selection_seq=0),dict(manual_owned=True,selection_seq=1)):
   self.assertFalse(ready.released(dict(self.released,auto_arm=auto)))
if __name__=='__main__':unittest.main()
