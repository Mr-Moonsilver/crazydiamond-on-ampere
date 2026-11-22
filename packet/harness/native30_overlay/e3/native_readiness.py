"""Reviewed implementation closure for an owner-run native candidate.

This guard is source/build closure, not GPU/release qualification. It has no
environment override. Missing review, changed code, an unbuilt helper or a
foreign image refuses before worker allocation / engine transport ownership.
The immutable mounted package is checked once per process at startup.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

SCHEMA='e3-native-implementation-closure/1'
REVIEW_SCHEMA='e3-native-source-review/1'
PROFILE='q18a-native-performance'
CLOSURE='native-implementation-closure.json'
IMAGE='sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b'
CUPTI='e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e'
LIBS=('flash_s5_green.so','flash_s5_replay_observer.so')
SOURCES=('flash_s5_green.cu','flash_s5_replay_observer.cpp','flash_s5_graph_bytes.cpp')


class NativeReadinessError(RuntimeError):
    pass


def _require(ok,reason):
    if not ok:raise NativeReadinessError('require_native_ready: '+reason)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def file_sha(path):
    path=Path(path)
    _require(path.is_file() and not path.is_symlink(),'regular closure payload required')
    _require(path.stat().st_size<=64*1024**2,'closure payload exceeds bound')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def check_review(review):
    _require(type(review) is dict and review.get('schema')==REVIEW_SCHEMA,'source review missing')
    _require(review.get('reviewed_by')=='Codex' and review.get('profile')==PROFILE,'source review identity differs')
    _require(review.get('scope')=='implementation closure for owner test only','source review scope differs')
    _require(review.get('hardware_qualified') is False and review.get('release_qualified') is False,
             'source review cannot certify unrun engine gates')
    cpu=review.get('cpu_regression') or {}
    _require(type(cpu.get('cases')) is int and cpu['cases']>=260
             and cpu.get('required')==cpu['cases'] and cpu.get('exit_code')==0
             and _sha(cpu.get('log_sha256')),
             'full CPU regression not recorded')
    vm=review.get('vm_prerequisites') or {}
    _require(set(vm)=={'job13c','job14b','job13f'}
             and all(type(v) is dict and v.get('verdict')=='PASS'
                     and _sha(v.get('receipt_sha256'))
                     and type(v.get('scope')) is str and v['scope'] for v in vm.values()),
             'VM prerequisite review absent')
    for field in ('source_shas','harness_source_shas'):
        pins=review.get(field)
        _require(type(pins) is dict and bool(pins)
                 and all(type(name) is str and name and not Path(name).is_absolute()
                         and '..' not in Path(name).parts and _sha(value)
                         for name,value in pins.items()),'source review pins missing: '+field)
    _require(_sha(review.get('qualified_observer_library_sha256')), 'qualified observer pin missing')
    return review


def verify_directory(root, expected_modules):
    root=Path(root)
    path=root/CLOSURE
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size<1024**2,
             'reviewed implementation closure missing')
    closure=json.loads(path.read_text())
    _require(type(closure) is dict and closure.get('schema')==SCHEMA
             and closure.get('profile')==PROFILE,'native package profile differs')
    review=check_review(closure.get('review'))
    _require(closure.get('review_sha256')==digest(review),'source review digest differs')
    shas=closure.get('generated_source_shas')
    _require(type(shas) is dict and set(shas)==set(expected_modules),'generated module closure differs')
    for name in expected_modules:
        _require(type(name) is str and name.replace('_','').isalnum(),'module name invalid')
        _require(file_sha(root/(name+'.py'))==shas[name],'generated source differs: '+name)
    native=closure.get('native_source_shas')
    _require(type(native) is dict and set(native)==set(SOURCES),'native source inventory differs')
    for name in SOURCES:
        _require(file_sha(root/'native'/name)==native[name],'native source differs: '+name)
        _require(review['source_shas'].get('native/'+name)==native[name],'reviewed native source differs: '+name)
    receipt_path=root/'native/q17-helper-build.json'
    _require(receipt_path.is_file() and not receipt_path.is_symlink()
             and receipt_path.stat().st_size<1024**2,'CPU native build absent')
    receipt=json.loads(receipt_path.read_text())
    _require(receipt.get('schema')==1 and receipt.get('image')==IMAGE and receipt.get('gpu_work') is False
             and receipt.get('cupti_library_sha256')==CUPTI
             and receipt.get('declared_abi')=={'green':1,'observer':1},'CPU native build identity differs')
    files=receipt.get('files') or {}
    _require(set(files)==set(LIBS+SOURCES),'CPU native build inventory differs')
    for name in LIBS+SOURCES:
        _require(file_sha(root/'native'/name)==files[name],'built native payload differs: '+name)
    _require(files[LIBS[1]]==review.get('qualified_observer_library_sha256'),
             'observer differs from VM-qualified source build')
    return {'native_ready':True,'armable':True,'hardware_qualified':False,'release_qualified':False,
            'scope':'reviewed implementation candidate; same-queue hardware gates pending',
            'review_sha256':closure['review_sha256'],'profile':PROFILE}


@lru_cache(maxsize=1)
def require_implementation_closure():
    # The directory is the image's own RO-mounted package, never an env path.
    from .decode_probe import GENERATED_SOURCES
    return verify_directory(Path(__file__).resolve().parent,GENERATED_SOURCES)


def implementation_status():
    try:return require_implementation_closure()
    except (NativeReadinessError,OSError,ValueError,TypeError,KeyError) as exc:
        return {'native_ready':False,'armable':False,'hardware_qualified':False,'release_qualified':False,
                'reason':str(exc)[:240],'scope':'implementation closure unavailable'}


def payload_identity(root):
    """Explicit non-self-referential final payload identity, metadata excluded.

    The independent final SHA256SUMS still covers generation/readiness too.
    This identity includes all compiled payloads, build receipt and source
    closure; it is not mislabeled as the final manifest's own byte hash.
    """
    root=Path(root)
    excluded={'SHA256SUMS','S5-OVERLAP-GENERATION.json','e3/native-overlap-readiness.json'}
    rows={p.relative_to(root).as_posix():file_sha(p) for p in sorted(root.rglob('*'))
          if p.is_file() and p.relative_to(root).as_posix() not in excluded}
    _require(not any(p.is_symlink() for p in root.rglob('*')), 'candidate payload contains symlink')
    _require(bool(rows),'empty candidate payload')
    return digest(rows)


def verify_finalized_package(root, expected_modules):
    root=Path(root)
    capability=verify_directory(root/'e3',expected_modules)
    generation=json.loads((root/'S5-OVERLAP-GENERATION.json').read_text())
    ready=json.loads((root/'e3/native-overlap-readiness.json').read_text())
    payload=payload_identity(root)
    _require(generation.get('candidate_profile')==PROFILE and generation.get('armable') is True,
             'final candidate not armed by finalizer')
    _require(generation.get('package_identity_basis')=='runtime payload excluding generation/readiness/manifest',
             'candidate payload identity basis differs')
    _require(generation.get('package_sha256')==payload==ready.get('package_sha256'),
             'final candidate payload identity differs')
    _require(ready.get('review_sha256')==capability['review_sha256'],'final review binding differs')
    _require(generation.get('implementation_review_sha256')==capability['review_sha256'],
             'generation review binding differs')
    _require(ready.get('path')=='e3/native-overlap-readiness.json'
             and ready.get('reviewed_by')=='Codex'
             and all(ready.get(key) is True for key in
                     ('reviewed','require_native_ready','native_ready','full_native_overlap',
                      'sparse_dma_lease','armable')), 'final readiness capability differs')
    launcher=generation.get('launcher') or {}
    name=launcher.get('name')
    _require(type(name) is str and Path(name).name==name,
             'final launcher name differs')
    _require(ready.get('launcher_sha256')==file_sha(root/name), 'final launcher binding differs')
    _require(ready.get('receipt_sha256')==digest({k:v for k,v in ready.items() if k!='receipt_sha256'}),
             'final readiness receipt digest differs')
    _require(ready.get('hardware_qualified') is False and ready.get('release_qualified') is False,
             'unrun hardware cannot be certified by packaging')
    return capability
