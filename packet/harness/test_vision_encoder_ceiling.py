import base64,json,tempfile,time,unittest,os,subprocess,sys,copy
from pathlib import Path
from unittest.mock import patch
import vision_encoder_ceiling as c
from test_vision_encoder_protocol import FakeBackend
class Ceiling(unittest.TestCase):
 def setUp(self):
  self.tmp=Path(tempfile.mkdtemp(prefix='ceiling-cpu-'));self.images=[]
  for i in range(8):
   p=self.tmp/(str(i)+'.png');p.write_bytes(b'\x89PNG\r\n\x1a\nCPU_LEAF');self.images.append(p)
  c.LATEST={};c.PENDING=None;c.CANCEL.clear();self.now=0;self.calls=0
 def sleep(self,n):self.now+=n;time.sleep(.001)
 def asr(self):self.calls+=1;return dict(correct=True,elapsed_s=1,observed_wall_ns=time.time_ns())
 def test_whole_serial_gateway_sweep_and_post_stop_asr(self):
  encode=[]
  def one(self,raw):encode.append(raw);result=FakeBackend().encode(raw);result['metrics']=dict(wall_ms=1,peak_allocated_bytes=100,peak_reserved_bytes=200);return result
  with patch.object(c,'remote_speech',return_value=0),patch.object(c.Gateway,'encode',one):
   result=c.sweep('http://<encoder-host>:9041','b'*64,self.images,dict(elapsed_s=1),self.asr,lambda x:'',clock=lambda:self.now,sleep=self.sleep)
  self.assertTrue(result['complete']);self.assertEqual([r['n'] for r in result['steps']],[1,2,4,6,8]);self.assertEqual(len(encode),21);self.assertEqual(self.calls,5)
  self.assertEqual(c.post_stop(self.asr,dict(elapsed_s=1))['correct'],True);self.assertEqual(self.calls,6) # The owner supplies the additional pre-load baseline (7 total).
  self.assertGreaterEqual(self.now,480);self.assertEqual(result['ceiling'],8)
 def test_first_asr_failure_stops_higher_steps(self):
  def wrong():return dict(correct=False,elapsed_s=1)
  def one(self,raw):result=FakeBackend().encode(raw);result['metrics']={};return result
  with patch.object(c,'remote_speech',return_value=0),patch.object(c.Gateway,'encode',one):
   with self.assertRaises(c.SpeechLatency):c.sweep('http://<encoder-host>:9041','b'*64,self.images,dict(elapsed_s=1),wrong,lambda x:'',clock=lambda:self.now,sleep=self.sleep)
  self.assertEqual([r['n'] for r in c.LATEST['steps']],[1]);self.assertEqual(c.LATEST['steps'][0]['status'],'FAIL')
 def test_busy_speech_skip_and_connection_parsing(self):
  text='0 0 <encoder-host>:9030 <internal-host>:42342\n0 0 <encoder-host>:9030 <encoder-host>:42343\n'
  self.assertEqual(c.remote_speech(lambda x:text),1)
  with patch.object(c,'remote_speech',return_value=1):result=c.sweep('http://<encoder-host>:9041','b'*64,self.images,dict(elapsed_s=1),self.asr,lambda x:'',clock=lambda:self.now,sleep=self.sleep)
  self.assertFalse(result['complete']);self.assertEqual(self.calls,0);self.assertTrue(all(x['status']=='SKIPPED' for x in result['steps']))
 def test_card_budget_pid_restart_and_unknown_gpu_owner(self):
  clear=dict(gpu_uuid='GPU-test',speech_gpu_pids=[10])
  def command(argv,**kw):
   if '--query-gpu=memory.used,memory.free' in argv:return '7000,17576'
   if '--query-compute-apps=pid,used_memory,gpu_uuid' in argv:return '10,5500,GPU-test\n20,1500,GPU-test'
   return 'PID\n10\n'
  self.assertEqual(c.card(clear,20,command)['used_mib'],7000)
  with self.assertRaises(c.Refused):c.card(dict(clear,speech_gpu_pids=[11]),20,command)
  def over(argv,**kw):return '23000,1576' if '--query-gpu=memory.used,memory.free' in argv else command(argv,**kw)
  with self.assertRaises(c.Refused):c.card(clear,20,over)


class WholeOwner(unittest.TestCase):
 def test_actual_owner_argv_all21_encodes_7_asrs_gpu_off_and_final_reader(self):
  import test_vision_encoder_protocol as prior
  import vision_encoder_job as job
  import vision_encoder_ceiling as ceiling
  import vision_public_gateway as gateway
  from vision_encoder_ceiling_read import read
  fixture=prior.Protocol();fixture.setUp()
  try:
   clear=Path(fixture.argv[fixture.argv.index('--clearance')+1]);value=json.loads(clear.read_text());value['speech_gpu_pids']=[10];clear.write_text(json.dumps(value))
   for i in range(8):
    p=fixture.root/('page-'+str(i)+'.png');p.write_bytes(b'\x89PNG\r\n\x1a\nCPU_IMAGE_LEAF')
    if i<len(fixture.images):fixture.images[i]=p
    else:fixture.images.append(p)
   fixture.argv=fixture.argv[:fixture.argv.index('--images')]+['--images',*map(str,fixture.images),'--ceiling']
   packet=os.environ.get('VISION_CEILING_PACKET')
   if packet:
    actual=json.loads((Path(packet)/'ceiling-owner-argv.json').read_text())[3:]
    # Only IO locations and the bound fresh run identity differ in the CPU stub.
    for flag in ('--model','--run-dir','--run-id','--clearance','--key-file','--expected-asr','--asr-url','--speech-health','--fixture'):
     actual[actual.index(flag)+1]=fixture.argv[fixture.argv.index(flag)+1]
    actual=actual[:actual.index('--images')]+['--images',*map(str,fixture.images),'--ceiling']
    fixture.argv=actual
   now=[0.0]
   def sleep(n):now[0]+=n;time.sleep(.001)
   original=ceiling.sweep
   def sweep(*a,**kw):return original(*a,**kw,clock=lambda:now[0],sleep=sleep)
   def encode(self,data):
    response=job.request(fixture.encoder_url,'/v1/encode','b'*64,dict(image_base64=base64.b64encode(data).decode()))
    response['metrics'].update(peak_allocated_bytes=100,peak_reserved_bytes=200,wall_ms=1)
    return response
   with patch.object(ceiling,'sweep',side_effect=sweep),patch.object(ceiling,'card',return_value=dict(used_mib=7300,free_mib=17276)),patch.object(ceiling,'remote_speech',return_value=0),patch.object(gateway.Gateway,'encode',encode):code,value=fixture.drive()
   self.assertEqual(code,0,value);self.assertEqual(fixture.state['asr'],7);self.assertTrue(value['owned_gpu_released']);self.assertTrue(value['ceiling']['complete'])
   memory=[dict(monotonic=(r['started_monotonic']+r['finished_monotonic'])/2,owned_mib=1800,free_mib=17000) for r in value['ceiling']['steps']]
   result=read(value,memory);self.assertEqual(result['verdict'],'PASS');self.assertEqual(result['ceiling'],8)
   missing=copy.deepcopy(value)
   for r in missing['ceiling']['steps']:
    for image in r['per_image']:image.pop('memory_before',None);image.pop('memory_after',None)
   self.assertEqual(read(missing,[])['verdict'],'HOLD')
   (fixture.root/'run/memory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in memory))
   result=subprocess.run([sys.executable,str(Path(c.__file__).with_name('vision_encoder_ceiling_read.py')),'--record',str(fixture.root/'run/result.json'),'--out',str(fixture.root/'read.json')],capture_output=True,text=True)
   self.assertEqual(result.returncode,0,result.stderr);self.assertIn('VISION_ENCODER_CEILING_READ PASS ceiling=8',result.stdout)
  finally:fixture.doCleanups()

if __name__=='__main__':unittest.main()
