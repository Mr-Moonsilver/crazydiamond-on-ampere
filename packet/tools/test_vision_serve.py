"""Small host checks for the derived vision packet and stricter real-argv Docker stub."""
import contextlib,io,json,os
from pathlib import Path
import subprocess,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_vision_serve as builder
import vision_mm_memory_read as memory

class PacketTests(unittest.TestCase):
    def test_duplicate_mount_is_refused_by_the_actual_stub_parser(self):
        root=Path(tempfile.mkdtemp(prefix='vision-mount-'));mount=root/'mount';mount.mkdir()
        stub=builder.ROOT/'deploy/final-engine-serve/selfrun/shims/docker'
        argv=[str(stub),'run','--name','fake','-v',str(mount)+':/same:ro','-v',str(mount)+':/same:ro','image-never-started']
        result=subprocess.run(argv,env=dict(os.environ,SR=str(root)),capture_output=True,text=True)
        self.assertEqual(result.returncode,125,result.stderr)
        self.assertIn('duplicate mount point',result.stderr)
        self.assertFalse((root/'docker/fake.json').exists())
    def test_broad_caught_exception_keeps_type_message_and_location(self):
        source='''def run():
    try:
        raise ValueError('schema differs')
    except Exception:
        return 2
'''
        namespace={};exec(builder.exception_details(source),namespace)
        stderr=io.StringIO()
        with contextlib.redirect_stderr(stderr):self.assertEqual(namespace['run'](),2)
        self.assertIn('ValueError: schema differs',stderr.getvalue())
        self.assertRegex(stderr.getvalue(),r'at .*:[0-9]+')
    def test_wrong_parent_refuses_before_creating_output(self):
        root=Path(tempfile.mkdtemp(prefix='vision-parent-'));base=root/'base';base.mkdir()
        (base/'PACKET-SHA256SUMS').write_text('wrong parent\n')
        out=root/'out'
        with self.assertRaisesRegex(ValueError,'serve-c identity'):
            builder.build(base,root/'unused-budget.py',root/'unused-encoder.py',out)
        self.assertFalse(out.exists())
    def test_declared_packet_keeps_final_engine_dimensions(self):
        self.assertEqual(builder.VISION_ARGS,'--enable-mm-embeds --limit-mm-per-prompt {"image":0,"video":0} --mm-processor-cache-gb 0 --skip-mm-profiling')
        self.assertNotEqual(builder.NEW_OVERLAY,builder.OLD_OVERLAY)
        self.assertIn('harness/vision_embeddings_export.py',builder.CLIENT_FILES)

    def test_memory_reader_actual_cli_and_unknown_peer_refusal(self):
        value=dict(verdict='PASS',workers={})
        for r in range(4):
            value['workers'][str(r)]=dict(native_inventory=dict(worker_rank=r,native_mm=dict(
                schema='native-mm-inventory/1',tower_registered_cuda_bytes=0,cache_budget_tokens=8192,
                cache_budget_bytes=40*2**20,cache_feature_bytes=0,cache_storage_bytes=0,
                gpu_encoder_cache_present=r<2,encoder_inputs_buffer_bytes=5*2**20 if r<2 else 0,
                active_splice_device_bytes=0)))
        self.assertEqual(memory.read(value)['verdict'],'PASS')
        root=Path(tempfile.mkdtemp(prefix='vision-mm-memory-'));record=root/'startup.json';record.write_text(json.dumps(value))
        result=subprocess.run([sys.executable,'-B',str(Path(memory.__file__)),'--startup-grade',str(record),
            '--out',str(root/'read.json')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('VISION_MM_MEMORY PASS',result.stdout)
        mm=value['workers']['1']['native_inventory']['native_mm']
        for field,bad in [('tower_registered_cuda_bytes',1),('cache_storage_bytes',1),('encoder_inputs_buffer_bytes',None)]:
            old=mm[field];mm[field]=bad
            self.assertEqual(memory.read(value)['verdict'],'HOLD');mm[field]=old
        value['workers'].pop('3');self.assertEqual(memory.read(value)['verdict'],'HOLD')

if __name__=='__main__':unittest.main()
