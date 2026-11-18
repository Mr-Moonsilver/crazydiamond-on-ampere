import importlib.util,json,shlex,shutil,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class Packet(unittest.TestCase):
 def test_real_unit_retained_container_and_stop_ownership_guards(self):
  packet=Path('<hub-dir>/cargo-target/flash-next/gateway-queue-w28-20261118/build-7')
  lines=(packet/'flashnext-vision-gateway.service').read_text().splitlines()
  pre=next(x for x in lines if x.startswith('ExecStartPre=/usr/bin/python3 -c'))
  stop=next(x for x in lines if x.startswith('ExecStop='))
  class Completed:
   returncode=0;stdout=''
  class Fake:
   def __init__(self,answer):self.answer=answer;self.calls=[]
   def run(self,args,**kwargs):
    self.calls.append(args);r=Completed();r.stdout=self.answer;return r
  from unittest.mock import patch
  source_sha=__import__('hashlib').sha256((packet/'vision_public_gateway.py').read_bytes()).hexdigest()
  image='sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
  for line,kind in ((pre,'rename'),(stop,'stop')):
   code=shlex.split(line.split('=',1)[1])[-1]
   for answer,allowed in ((source_sha+' false '+image,True),('wrong false '+image,False),(source_sha+' true '+image,kind=='stop')):
    fake=Fake(answer)
    with patch('subprocess.run',fake.run):
     if allowed:exec(code,{})
     else:
      with self.assertRaises(AssertionError):exec(code,{})
    self.assertEqual(any(c[1]==kind for c in fake.calls),allowed)
 def test_repin_preserves_engine_rows_and_real_guard_restores_gateway(self):
  repin=module('queue_repin',ROOT/'tools/vision_gateway_queue_repin.py')
  guard=module('frozen_guard',Path('<hub-dir>/cargo-target/flash-next/vision-headroom-w28-20261118/build-6/production/tools/vision_production_pin_guard.py'))
  tmp=Path(tempfile.mkdtemp());pin=tmp/'root/flashnext-prod-pins/vision-w28';pin.mkdir(parents=True)
  (pin/'engine').write_text('frozen engine');(pin/'engine').chmod(0o600)
  engine=tmp/'engine';engine.write_text('frozen engine');engine.chmod(0o644)
  old=dict(path='/engine',blob='engine',sha256=repin.sha(engine),mode=0o644)
  (pin/'manifest.json').write_text(json.dumps(dict(schema='vision-production-pins/1',files=[old])))
  packet=tmp/repin.PATCH.lstrip('/');packet.mkdir(parents=True)
  source=Path('<hub-dir>/cargo-target/flash-next/gateway-queue-w28-20261118/build-7')
  for target,name in (('/etc/systemd/system/flashnext-vision-gateway.service','flashnext-vision-gateway.service'),('/etc/default/flashnext-vision-gateway','gateway-defaults.env'),(repin.PATCH+'/vision_public_gateway.py','vision_public_gateway.py')):
   shutil.copyfile(source/name,packet/name);p=tmp/target.lstrip('/');p.parent.mkdir(parents=True,exist_ok=True)
   if p!=packet/name:shutil.copyfile(packet/name,p)
   p.chmod(0o644)
  repin.update(tmp);self.assertEqual(json.loads((pin/'manifest.json').read_text())['files'][0],old)
  self.assertEqual(guard.guard(str(tmp),True),[])
  gateway=tmp/'etc/default/flashnext-vision-gateway';gateway.write_text('STALE=8')
  self.assertIn('/etc/default/flashnext-vision-gateway',guard.guard(str(tmp),True))
  guard.guard(str(tmp));self.assertEqual(gateway.read_bytes(),(packet/'gateway-defaults.env').read_bytes())
  self.assertTrue(list(pin.glob('manifest-before-queue7-*')))
if __name__=='__main__':unittest.main()
