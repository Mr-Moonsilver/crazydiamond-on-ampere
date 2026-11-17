import contextlib,hashlib,importlib.util,io,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('vision_pins',ROOT/'tools/vision_production_pin_guard.py');pins=importlib.util.module_from_spec(spec);spec.loader.exec_module(pins)
class Pins(unittest.TestCase):
 def setUp(self):
  self.root=Path(tempfile.mkdtemp(prefix='w27-pin-guard-'))
  for name in pins.TARGETS:
   path=pins.rooted(self.root,name);path.parent.mkdir(parents=True,exist_ok=True);path.write_text('KNOWN '+name);path.chmod(0o644)
  with contextlib.redirect_stdout(io.StringIO()):pins.register(self.root)
 def test_guard_restores_every_declared_file_and_quarantines_foreign_dropin(self):
  for name in pins.TARGETS:pins.rooted(self.root,name).write_text('FOREIGN')
  extra=pins.rooted(self.root,pins.DIRS[0]+'/99-test.conf');extra.write_text('FOREIGN test override')
  with contextlib.redirect_stdout(io.StringIO()):changed=pins.guard(self.root)
  self.assertEqual(len(changed),len(pins.TARGETS)+1);self.assertFalse(extra.exists())
  for name in pins.TARGETS:self.assertEqual(pins.rooted(self.root,name).read_text(),'KNOWN '+name)
  retained=list(pins.rooted(self.root,'/root/flashnext-vision-boot-aside').rglob('*'))
  self.assertEqual(sum(p.is_file() for p in retained),len(pins.TARGETS)+1)
  with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(pins.guard(self.root),[])
 def test_dry_run_preserves_changed_target_and_pin_tamper_refuses(self):
  target=pins.rooted(self.root,pins.TARGETS[0]);target.write_text('TEST')
  with contextlib.redirect_stdout(io.StringIO()):self.assertTrue(pins.guard(self.root,True))
  self.assertEqual(target.read_text(),'TEST')
  blob=pins.rooted(self.root,'/root/flashnext-prod-pins/vision-w27/0');blob.write_text('CORRUPT')
  with self.assertRaises(ValueError):pins.guard(self.root)
 def test_no_symlink_follows_into_other_files(self):
  target=pins.rooted(self.root,pins.TARGETS[0]);target.rename(target.with_suffix('.retained'));target.symlink_to('/etc/passwd')
  with self.assertRaises(ValueError):pins.guard(self.root)
if __name__=='__main__':unittest.main()
