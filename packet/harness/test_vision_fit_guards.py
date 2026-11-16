"""Bound input and reader negatives/mutants; no network, GPU or authoritative-cache edits."""
import copy,inspect,json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import vision_fit_8x200k as fit
import test_vision_fit_8x200k as protocol
import test_fit_8x200k as old


def complete():
    with patch.object(old.fit,'PREFIXES',fit.PREFIXES),patch.object(old.fit,'OUTPUT',8192),patch.object(old.fit,'OUTPUT_CAPS',fit.OUTPUT_CAPS),patch.object(old.fit,'COLD_HISTORY',190000):
        row=old.complete()
    row.update(input_binding_sha256=fit.BINDING_SHA,cache_shas=fit.CACHE_SHAS,
        cancel=dict(retired=True),reuse=dict(same_image_cached=4048,different_image_verified=True),
        host=dict(samples=100,errors=[],oom_delta=0,oom_kill_delta=0),semantic=dict(verdict='MATCH'))
    row['tail']=dict(complete=True,native_admissions=8,conversations=[dict(prompt_tokens=190320,output_tokens=8192,output_cap=8192,image_tail_overlap=54) for _ in range(8)])
    for stage in row['stages']+[row['tail'],row['cold']]:
        for m in stage['conversations']:
            m.update(prompt_verified=True,image_rows=1024,output_cap=m['output_tokens'])
    row['memory'].update(mm_peaks={str(r):dict(tower_registered_cuda_bytes=0,cache_storage_bytes=5*2**20 if r<2 else 0) for r in range(4)},
        host_rss_peaks={str(r):dict(VmRSS=100,RssAnon=80,RssShmem=20,VmHWM=120) for r in range(4)})
    row['memory']['simultaneous_resident_token_peak'].update(eight_at_190k_seen=True,eight_minimum_peak=190320)
    return row

class Guards(unittest.TestCase):
    def args(self):return NS(binding=protocol.BINDING,private_cache_dir=protocol.CACHES,packet=protocol.PACKET)
    def test_wrong_input_sha_and_private_directory_refusal_mutants(self):
        a=self.args()
        with patch.object(fit,'CACHE_SHAS',dict(fit.CACHE_SHAS,semantic='0'*64)):
            with self.assertRaisesRegex(fit.Hold,'private-cache-SHA'):fit.load_inputs(a)
            source=inspect.getsource(fit.load_inputs).replace("if hashlib.sha256(raw).hexdigest()!=CACHE_SHAS[name]:",'if False:')
            namespace=dict(fit.__dict__);exec(source,namespace)
            self.assertEqual(len(namespace['load_inputs'](a)['eight-unique']),8)
        root=Path(tempfile.mkdtemp(prefix='vision-fit-bad-dir-'));(root/'alias').symlink_to(protocol.CACHES,target_is_directory=True)
        a.private_cache_dir=root/'alias'
        with self.assertRaisesRegex(fit.Hold,'private-cache-directory-mode'):fit.load_inputs(a)
        source=inspect.getsource(fit.load_inputs).replace("if directory.is_symlink() or directory.stat().st_mode & 0o777!=0o700:", 'if False:')
        namespace=dict(fit.__dict__);exec(source,namespace)
        self.assertEqual(len(namespace['load_inputs'](a)['semantic']),4)
    def test_complete_record_and_every_mm_host_work_missing_proof(self):
        self.assertEqual(fit.assess(complete())['verdict'],'FIT')
        changes=(lambda r:r['memory']['host_rss_peaks'].pop('1'),lambda r:r['memory']['mm_peaks'].pop('3'),
            lambda r:r['memory']['stop_proof'].update(all_rank_stop=False),
            lambda r:r['memory']['mm_peaks']['0'].update(cache_storage_bytes=41*2**20),
            lambda r:r['memory']['mm_peaks']['2'].update(tower_registered_cuda_bytes=1),
            lambda r:r['memory']['simultaneous_resident_token_peak'].update(eight_at_190k_seen=False),
            lambda r:r['tail'].update(native_admissions=0),lambda r:r['tail']['conversations'][2].update(image_tail_overlap=0),
            lambda r:r['stages'][1].update(native_admissions=0),lambda r:r['host'].update(errors=['unreadable']),
            lambda r:r['stages'][1]['conversations'][0].update(prompt_verified=False))
        for change in changes:
            row=complete();change(row);self.assertEqual(fit.assess(row)['verdict'],'HOLD')
        row=complete();row['host']['oom_kill_delta']=1;self.assertEqual(fit.assess(row)['verdict'],'FAIL')
    def test_missing_residency_and_prompt_proof_mutants(self):
        for reason,change in (('eight-concurrent-190k-frontiers-unproven',lambda r:r['memory']['simultaneous_resident_token_peak'].update(eight_at_190k_seen=False)),
                             ('MM-stream-unproven',lambda r:r['tail']['conversations'][0].update(prompt_verified=False))):
            row=complete();change(row);self.assertEqual(fit.assess(row)['verdict'],'HOLD')
            source=inspect.getsource(fit.assess).replace("reasons.append('"+reason+"')",'pass')
            namespace=dict(fit.__dict__);exec(source,namespace)
            self.assertEqual(namespace['assess'](row)['verdict'],'FIT')
    def test_error_messages_do_not_echo_private_http_response(self):
        class Response:
            status=400
            def read(self,*_):return b'PRIVATE_IMAGE_PAYLOAD'
        class Connection:
            def request(self,*_):pass
            def getresponse(self):return Response()
            def close(self):pass
        with patch.object(fit.base,'connection',return_value=(Connection(),'')):
            with self.assertRaisesRegex(fit.Hold,'/tokenize-HTTP-400') as error:fit.post('http://fixture','/tokenize',{})
            self.assertNotIn('PRIVATE_IMAGE_PAYLOAD',str(error.exception))

if __name__=='__main__':unittest.main()
