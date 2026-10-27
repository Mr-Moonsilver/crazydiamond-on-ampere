"""CPU-only source/build closure tests. Fake ELF bytes are never loaded."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
import native_readiness as ready
import decode_probe
import make_s5_overlap as builder
import e3_finalize_package as finalizer


def write_json(path,value):
    path.write_text(json.dumps(value,sort_keys=True))


def source_package(root):
    source=root/'source'
    e3=source/'e3'
    native=e3/'native'
    native.mkdir(parents=True)
    for name in decode_probe.GENERATED_SOURCES:
        (e3/(name+'.py')).write_text('# CPU-only fixture: '+name+'\n')
    for name in ready.SOURCES:
        (native/name).write_text('// CPU-only source fixture '+name+'\n')
    (source/'run-flash-s5-test.sh').write_text('#!/bin/sh\nexit 2\n')
    for name in ready.LIBS:
        (native/name).write_bytes(b'\x7fELF CPU fixture never loaded '+name.encode())
    review={
        'schema':ready.REVIEW_SCHEMA,'profile':ready.PROFILE,'reviewed_by':'Codex',
        'scope':'implementation closure for owner test only',
        'hardware_qualified':False,'release_qualified':False,
        'cpu_regression':{'cases':261,'required':261,'exit_code':0,'log_sha256':'1'*64},
        'vm_prerequisites':{name:{'verdict':'PASS','receipt_sha256':'2'*64,
                                  'scope':'CPU fixture of the review record, not GPU evidence'}
                            for name in ('job13c','job14b','job13f')},
        'source_shas':{'native/'+name:ready.file_sha(native/name) for name in ready.SOURCES},
        'harness_source_shas':{'runs/q18a_overlap.draft.sh':'3'*64},
        'qualified_observer_library_sha256':ready.file_sha(native/ready.LIBS[1])}
    closure={'schema':ready.SCHEMA,'profile':ready.PROFILE,'review':review,
             'review_sha256':ready.digest(review),
             'generated_source_shas':{name:ready.file_sha(e3/(name+'.py'))
                                       for name in decode_probe.GENERATED_SOURCES},
             'native_source_shas':{name:ready.file_sha(native/name) for name in ready.SOURCES}}
    write_json(e3/ready.CLOSURE,closure)
    generation={'armable':False,'generated_files':[],
                'candidate_profile':ready.PROFILE,'require_native_ready':True,
                'implementation_review_sha256':closure['review_sha256'],
                'launcher':{'name':'run-flash-s5-test.sh'}}
    write_json(source/finalizer.GENERATION,generation)
    # Match the real CPU build protocol: the original source manifest excludes
    # only the two newly built libraries and their receipt.
    (source/'SHA256SUMS').write_text(''.join(
        finalizer.sha(p)+'  '+p.relative_to(source).as_posix()+'\n'
        for p in sorted(source.rglob('*'))
        if p.is_file() and p.name not in ready.LIBS))
    receipt={'schema':1,'image':ready.IMAGE,'gpu_work':False,
             'cupti_library_sha256':ready.CUPTI,'declared_abi':{'green':1,'observer':1},
             'files':{name:ready.file_sha(native/name) for name in ready.LIBS+ready.SOURCES},
             'needed':{ready.LIBS[0]:['libcuda.so.1'],
                       ready.LIBS[1]:['libcuda.so.1','libcupti.so.13']}}
    write_json(native/'q17-helper-build.json',receipt)
    return source,closure,receipt


class NativeClosureTests(unittest.TestCase):
    def test_no_review_or_environment_escape_hatch(self):
        ready.require_implementation_closure.cache_clear()
        try:
            with patch.dict('os.environ',{'FLASH_S5_NATIVE_READY':'1',
                                         'FLASH_S5_NATIVE_CLOSURE':'/tmp/anything'}):
                with self.assertRaisesRegex(ready.NativeReadinessError,'closure missing'):
                    ready.require_implementation_closure()
                self.assertFalse(ready.implementation_status()['native_ready'])
        finally:
            ready.require_implementation_closure.cache_clear()

    def test_complete_implementation_is_not_hardware_or_release_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,_,_=source_package(Path(tmp))
            result=ready.verify_directory(source/'e3',decode_probe.GENERATED_SOURCES)
            self.assertTrue(result['native_ready'])
            self.assertFalse(result['hardware_qualified'])
            self.assertFalse(result['release_qualified'])

    def test_actual_finalizer_then_independent_payload_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,_,_=source_package(root)
            before={p.relative_to(source).as_posix():finalizer.sha(p)
                    for p in source.rglob('*') if p.is_file()}
            result=finalizer.finalize(source,root/'final')
            self.assertTrue(result['armable'])
            self.assertTrue(ready.verify_finalized_package(
                root/'final',decode_probe.GENERATED_SOURCES)['armable'])
            self.assertEqual(before,{p.relative_to(source).as_posix():finalizer.sha(p)
                                    for p in source.rglob('*') if p.is_file()})
            entries=finalizer.manifest(root/'final')
            self.assertIn('e3/'+ready.CLOSURE,entries)
            self.assertIn('e3/native-overlap-readiness.json',entries)
            with self.assertRaises(ValueError):finalizer.finalize(source,root/'final')

    def test_runtime_payload_mutations_are_refused_even_with_recomputed_manifest(self):
        mutations=(
            lambda p:(p/'e3/native/flash_s5_green.so').write_bytes(b'changed'),
            lambda p:(p/'e3/native_overlap_session.py').write_text('changed'),
            lambda p:(p/'unexpected').write_text('extra'),
            lambda p:(p/'run-flash-s5-test.sh').write_text('changed'),
            lambda p:(p/'extra-link').symlink_to(p/'run-flash-s5-test.sh'))
        for mutate in mutations:
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source,_,_=source_package(root)
                finalizer.finalize(source,root/'final');mutate(root/'final')
                with self.assertRaises(ready.NativeReadinessError):
                    ready.verify_finalized_package(root/'final',decode_probe.GENERATED_SOURCES)

    def test_native_image_abi_cupti_and_vm_observer_pin_are_bound(self):
        for field,value in (('image','foreign'),('gpu_work',True),
                            ('cupti_library_sha256','0'*64),
                            ('declared_abi',{'green':2,'observer':1})):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                source,_,receipt=source_package(Path(tmp))
                receipt[field]=value
                write_json(source/'e3/native/q17-helper-build.json',receipt)
                with self.assertRaises(ready.NativeReadinessError):
                    ready.verify_directory(source/'e3',decode_probe.GENERATED_SOURCES)
        with tempfile.TemporaryDirectory() as tmp:
            source,closure,receipt=source_package(Path(tmp))
            path=source/'e3/native'/ready.LIBS[1]
            path.write_bytes(b'\x7fELF different build')
            receipt['files'][ready.LIBS[1]]=ready.file_sha(path)
            write_json(source/'e3/native/q17-helper-build.json',receipt)
            with self.assertRaisesRegex(ready.NativeReadinessError,'VM-qualified'):
                ready.verify_directory(source/'e3',decode_probe.GENERATED_SOURCES)

    def test_review_refuses_missing_cpu_or_vm_proof_and_hardware_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            _,closure,_=source_package(Path(tmp));review=closure['review']
            for mutate in (
                lambda r:r['cpu_regression'].update(exit_code=1),
                lambda r:r['cpu_regression'].update(required=262),
                lambda r:r['cpu_regression'].update(log_sha256='z'*64),
                lambda r:r['vm_prerequisites'].pop('job14b'),
                lambda r:r['vm_prerequisites']['job13f'].update(verdict='INCONCLUSIVE'),
                lambda r:r.update(hardware_qualified=True),
                lambda r:r.update(harness_source_shas={}),
                lambda r:r['source_shas'].update({'../outside':'4'*64})):
                with self.subTest(mutate=mutate):
                    bad=copy.deepcopy(review);mutate(bad)
                    with self.assertRaises(ready.NativeReadinessError):ready.check_review(bad)

    def test_inventory_and_closure_digest_cannot_be_omitted(self):
        for mutate in (
            lambda c:c.update(review_sha256='0'*64),
            lambda c:c['generated_source_shas'].pop('native_readiness'),
            lambda c:c['native_source_shas'].pop(ready.SOURCES[0]),
            lambda c:c['review']['source_shas'].update({'native/'+ready.SOURCES[0]:'0'*64})):
            with self.subTest(mutate=mutate),tempfile.TemporaryDirectory() as tmp:
                source,closure,_=source_package(Path(tmp));mutate(closure)
                write_json(source/'e3'/ready.CLOSURE,closure)
                with self.assertRaises(ready.NativeReadinessError):
                    ready.verify_directory(source/'e3',decode_probe.GENERATED_SOURCES)

    def test_final_generation_and_receipt_are_bound_separately_from_payload(self):
        for target,field,value in (
            (finalizer.GENERATION,'armable',False),
            (finalizer.GENERATION,'implementation_review_sha256','0'*64),
            ('e3/native-overlap-readiness.json','hardware_qualified',True),
            ('e3/native-overlap-readiness.json','launcher_sha256','0'*64),
            ('e3/native-overlap-readiness.json','native_ready',False)):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);source,_,_=source_package(root)
                finalizer.finalize(source,root/'final')
                path=root/'final'/target;record=json.loads(path.read_text());record[field]=value
                if 'receipt_sha256' in record:
                    record['receipt_sha256']=ready.digest({k:v for k,v in record.items() if k!='receipt_sha256'})
                write_json(path,record)
                with self.assertRaises(ready.NativeReadinessError):
                    ready.verify_finalized_package(root/'final',decode_probe.GENERATED_SOURCES)

    def test_candidate_builder_does_not_invent_a_source_review(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(builder,'HERE',Path(tmp)):
            with self.assertRaisesRegex(ValueError,'receipt is absent'):
                builder._native_candidate_closure({})


if __name__=='__main__':unittest.main()
