"""Join-only suffix fix: one unique engine id on every rank, all frozen MM gates intact."""
import copy,hashlib,importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
PK=Path('<hub-dir>/cargo-target/flash-next/vision-headroom-w28-20261118/build-6/production/tools')
def module(name):
 s=importlib.util.spec_from_file_location(name,PK/(name+'.py'));v=importlib.util.module_from_spec(s);s.loader.exec_module(v);return v
new=module('vision_g5_boundary_read_w28s');old=module('vision_g5_boundary_read')
class Join(unittest.TestCase):
 def setUp(self):
  self.target=dict(kind='native-boundary',request_id='w27-g5-four-native-boundary',response_id='chatcmpl-w27-g5-four-native-boundary');self.record=dict(cases=[self.target]);self.id=self.target['response_id']+'-9866a3e1'
  self.rows=[dict(schema='native-mm-boundary/1',request_sha256=hashlib.sha256(self.id.encode()).hexdigest(),worker_rank=r,stage=r//2,rows=n,free_bytes=100*2**20,native_mm=dict(tower_registered_cuda_bytes=0,cache_storage_bytes=40*2**20)) for r in range(4) for n in (533,1994,82)]
 def lines(self):return [new.PREFIX+json.dumps(r)+'\n' for r in self.rows]
 def read(self,ids=None):return new.read(self.lines(),self.record,[self.id] if ids is None else ids)
 def test_suffixed_unique_id_repeated_chunk_receipts_PASS(self):
  value=self.read(['other-id',self.id,self.id]);self.assertEqual(value['verdict'],'PASS');self.assertEqual(value['matched_rows'],12)
 def test_both_original_unsuffixed_forms_match_frozen_results(self):
  for ident in (self.target['request_id'],self.target['response_id']):
   for row in self.rows:row['request_sha256']=hashlib.sha256(ident.encode()).hexdigest()
   self.assertEqual(new.read(self.lines(),self.record,[ident]),old.read(self.lines(),self.record))
 def test_zero_multiple_and_unanchored_candidates_HOLD(self):
  for ids in ([],[self.id+'0'],['x'+self.id],[self.id+'-tail'],[self.target['response_id']+'-9866A3E1'],[self.target['request_id']+'-9866a3e1'],[self.id,self.target['response_id']+'-00000000']):
   with self.subTest(ids=ids):self.assertEqual(self.read(ids)['verdict'],'HOLD')
 def test_regex_prefix_is_literal(self):
  self.target['response_id']='chatcmpl-a.b';self.id='chatcmpl-aXb-9866a3e1';self.assertEqual(self.read()['verdict'],'HOLD')
 def test_zero_or_multiple_distinct_ids_per_rank_HOLD(self):
  other=hashlib.sha256(self.target['response_id'].encode()).hexdigest()
  self.rows[0]['request_sha256']=other;self.assertEqual(self.read()['verdict'],'HOLD')
  for row in self.rows:
   if row['worker_rank']==0:row['request_sha256']=other
  self.assertEqual(self.read()['verdict'],'HOLD')
  self.rows=[r for r in self.rows if r['worker_rank']!=3];self.assertEqual(self.read()['verdict'],'HOLD')
 def test_frozen_floor_cache_tower_schema_and_wide_splice_gates(self):
  original=copy.deepcopy(self.rows)
  for key,value in [('free_bytes',39*2**20),('schema','wrong')]:
   self.rows=copy.deepcopy(original);self.rows[0][key]=value;self.assertEqual(self.read()['verdict'],'FAIL')
  for key,value in [('cache_storage_bytes',41*2**20),('tower_registered_cuda_bytes',1)]:
   self.rows=copy.deepcopy(original);self.rows[0]['native_mm'][key]=value;self.assertEqual(self.read()['verdict'],'FAIL')
  self.rows=copy.deepcopy(original)
  for row in self.rows:row['rows']=1000
  self.assertEqual(self.read()['verdict'],'HOLD')
 def test_actual_argv_complete_and_ambiguity_negative(self):
  root=Path(tempfile.mkdtemp(prefix='g5-w28s-'));record=root/'record.json';ids=root/'ids.txt';log=root/'engine.log';record.write_text(json.dumps(self.record));ids.write_text(self.id+'\n');log.write_text(''.join(self.lines()))
  args=[sys.executable,'-I','-B',str(PK/'vision_g5_boundary_read_w28s.py'),'--engine-log',str(log),'--engine-ids',str(ids),'--record',str(record),'--out',str(root/'read.json')]
  p=subprocess.run(args,capture_output=True,text=True);self.assertEqual(p.returncode,0,p.stderr);self.assertIn('W27_G5_BOUNDARY PASS four-peer-MM-boundaries-cache-floor',p.stdout)
  ids.write_text(self.id+'\n'+self.target['response_id']+'-00000000\n');args[-1]=str(root/'ambiguous.json');p=subprocess.run(args,capture_output=True,text=True);self.assertEqual(p.returncode,2);self.assertIn('W27_G5_BOUNDARY HOLD',p.stdout)
 def test_outside_join_and_cli_blocks_are_byte_identical(self):
  a=(PK/'vision_g5_boundary_read.py').read_text();b=(PK/'vision_g5_boundary_read_w28s.py').read_text()
  self.assertEqual(hashlib.sha256(a.encode()).hexdigest(),'baa5a890f74346723e9562b93bb923647a906d3cb504401b8df9f5877be09036')
  start=" for row in rows:\n  mm=";end='def main():'
  self.assertEqual(a[a.index(start):a.index(end)],b[b.index(start):b.index(end)])
if __name__=='__main__':unittest.main()
