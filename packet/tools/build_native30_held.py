#!/usr/bin/env python3
"""Freeze a fresh native30 harness beside byte-identical native28 runtime files."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
BASE=Path('<hub-dir>/agents/flash-next/tmp/native28-held-b')
BUNDLE=BASE/'q18a-native28-bundle-71571d7efbca'
BASE_MANIFEST='71571d7efbca089f74c3093f484ee6c9236d2819b7a136914796ca08a2c58a78'
sys.path.insert(0,str(ROOT/'patches/s5_engine'))
from native_readiness import digest,payload_identity,verify_finalized_package


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def manifest(root):
    (root/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(root).as_posix()}\n' for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'SHA256SUMS'))
def verify(root):
    rows={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        pin,name=line.split(None,1);assert sha(root/name)==pin,name;rows[name]=pin
    assert set(rows)=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p!=root/'SHA256SUMS'}
    return rows

def replace(source,names,functions={}):
    lines=source.splitlines(True)
    for node in reversed(ast.parse(source).body):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in names:
            name=node.targets[0].id;lines[node.lineno-1:node.end_lineno]=[name+'='+names[name]+'\n']
        elif isinstance(node,ast.FunctionDef) and node.name in functions:
            lines[node.lineno-1:node.end_lineno]=[functions[node.name]+'\n']
    value=''.join(lines);compile(value,'native30-owner','exec');return value


def build(out,log):
    assert not out.exists();assert sha(BUNDLE/'SHA256SUMS')==BASE_MANIFEST;verify(BUNDLE);verify(BASE/'final')
    dirty=subprocess.check_output(['git','status','--porcelain','--','harness','tools'],cwd=ROOT,text=True);assert not dirty,dirty
    text=log.read_text();summ=re.findall(r'^selftest: (\d+)/(\d+) cases as required$',text,re.M)
    suites=[json.loads(s.split(': ',1)[1]) for s in text.splitlines() if s.startswith('selftest suite: ')]
    assert len(summ)==1 and summ[0][0]==summ[0][1] and all(s['returncode']==0 and not s['skipped'] for s in suites)
    out.mkdir();bundle=out/'bundle';shutil.copytree(BUNDLE,bundle)
    for p in bundle.rglob('*'):p.chmod(0o755 if p.is_dir() else 0o644)
    bundle.chmod(0o755)
    tracked=subprocess.check_output(['git','ls-files','harness'],cwd=ROOT,text=True).splitlines()
    for name in tracked:
        source=ROOT/name
        if source.is_file():
            target=bundle/Path(name).relative_to('harness');target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    tool_dir=bundle/'native30_tools';tool_dir.mkdir()
    for name in ('native30_owner_functions.py','native30_cpu_probe.py','native30_selfrun.py'):
        shutil.copyfile(ROOT/'tools'/name,tool_dir/name)
    overlay=bundle/'native30_overlay';shutil.copytree(BASE/'final',overlay)
    for p in overlay.rglob('*'):p.chmod(0o755 if p.is_dir() else 0o644)
    overlay.chmod(0o755)
    plugins=overlay/'e3/native30_plugins';plugins.mkdir()
    shutil.copyfile(ROOT/'harness/native30_localize.py',plugins/'native30_localize.py')
    dist=plugins/'flashnext_native30_localize-0.1.dist-info';dist.mkdir()
    (dist/'METADATA').write_text('Metadata-Version: 2.1\nName: flashnext-native30-localize\nVersion: 0.1\n')
    (dist/'entry_points.txt').write_text('[vllm.general_plugins]\nnative30_localize = native30_localize:install\n')
    closure=json.loads((overlay/'e3/native-implementation-closure.json').read_text());review=closure['review']
    review['cpu_regression']=dict(basis='native30 current-source strict zero-skip regression',cases=int(summ[0][0]),required=int(summ[0][1]),exit_code=0,skips=0,suites=len(suites),reported_tests=sum(s['tests_run'] or 0 for s in suites),log_sha256=sha(log))
    review['harness_source_shas']={str(Path(n).relative_to('harness')):sha(ROOT/n) for n in tracked if (ROOT/n).is_file()}
    review['native30_localization']=dict(engine_baseline_manifest=sha(BASE/'final/SHA256SUMS'),plugin_sha256=sha(plugins/'native30_localize.py'),scope='separate diagnostic wrappers and sync; no engine source/PLE repair',queue='startup then one row4 placement diagnostic, no fresh qualification claim',active_bound_s=1200,cleanup_margin_s=420,hard_end='2026-10-30T18:00:00Z',hardware_qualified=False,release_qualified=False)
    review['night_window']=dict(hard_end='2026-10-30T18:00:00Z',queue_timeout_s=1200,cleanup_margin_s=420,placeholder=False)
    closure['review_sha256']=digest(review);dump(overlay/'e3/native-implementation-closure.json',closure)
    dump(bundle/'native30-review.json',review)
    gen=json.loads((overlay/'S5-OVERLAP-GENERATION.json').read_text());ready=json.loads((overlay/'e3/native-overlap-readiness.json').read_text())
    gen['implementation_review_sha256']=ready['review_sha256']=closure['review_sha256']
    gen['package_sha256']=ready['package_sha256']=payload_identity(overlay)
    ready['receipt_sha256']=digest({k:v for k,v in ready.items() if k!='receipt_sha256'})
    dump(overlay/'S5-OVERLAP-GENERATION.json',gen);dump(overlay/'e3/native-overlap-readiness.json',ready);manifest(overlay)
    verify_finalized_package(overlay,closure['generated_source_shas'])
    old=verify(BASE/'final');new=verify(overlay)
    changed=[name for name,pin in old.items() if new.get(name)!=pin]
    assert sorted(changed)==sorted(['e3/native-implementation-closure.json','e3/native-overlap-readiness.json','S5-OVERLAP-GENERATION.json']),changed
    runtime=[name for name in old if name.endswith(('.py','.so','.sh'))]
    assert all(new[name]==old[name] for name in runtime)
    proof=dict(schema='native30-build/1',source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),baseline_bundle_manifest=BASE_MANIFEST,baseline_overlay_manifest=sha(BASE/'final/SHA256SUMS'),runtime_byte_equal=runtime,changed_metadata=changed,added=sorted(set(new)-set(old)),overlay_manifest=sha(overlay/'SHA256SUMS'),review_sha256=closure['review_sha256'],selftest=review['cpu_regression'],gpu_work=False,box_action=False)
    dump(bundle/'native30-provenance.json',proof);manifest(bundle);pin=sha(bundle/'SHA256SUMS')
    final=out/('q18a-native30-localize-bundle-'+pin[:12]);bundle.rename(final);overlay=final/'native30_overlay'
    original=(BASE/'q18a-native28-owner-run-71571d7efbca.py').read_text().replace('q18a_native28','q18a_native30_localize').replace('q18a-native28','q18a-native30-localize').replace('14400+900','1200+420')
    functions={
      'cpu_build':'''def cpu_build():
    # Already built/checked on hub: avoid compilation during the short outage.
    import shutil
    BUILD_LOGS.mkdir(mode=0o700)
    source=H/'native30_overlay'
    verify_sources(source,built=True,final=True)
    shutil.copytree(source,OVERLAY)
    verify_sources(OVERLAY,built=True,final=True)
    green=sha(OVERLAY/'e3/native/flash_s5_green.so');final=sha(OVERLAY/'SHA256SUMS')
    return green,final,dict(schema='native30-owner-prebuilt/1',bundle_sha256=BUNDLE,manifest_sha256=final,review_sha256=REVIEW,hub_prebuilt=True,box_compilation=False,hardware_qualified=False,release_qualified=False)''',
      'verify_sources':'''def verify_sources(root,*,built=False,final=False):
    require(inventory(root)==set(EXPECTED_SOURCES)|{'SHA256SUMS'},'prebuilt-inventory-differs')
    for name,pin in EXPECTED_SOURCES.items():require(sha(root/name)==pin,'prebuilt-source-pin:'+name)
    manifest(root)
    require(sha(root/'run-flash-s5-test.sh')==LAUNCHER,'trusted-launcher-pin')
    from native_readiness import verify_finalized_package
    closure=json.loads((root/'e3/native-implementation-closure.json').read_text())
    require(closure['review_sha256']==REVIEW,'review-binding-differs')
    verify_finalized_package(root,closure['generated_source_shas'])'''}
    template=ROOT/'tools/native30_owner_functions.py'
    body=template.read_text();nodes=ast.parse(body)
    for node in nodes.body:
        if isinstance(node,ast.FunctionDef):functions[node.name]=ast.get_source_segment(body,node)
    owner=replace(original,dict(H="Path('/home/<service-user>/harness/"+pin[:12]+"')",BUNDLE=repr(pin),REVIEW=repr(closure['review_sha256']),EXPECTED_SOURCES=repr(new)),functions)
    runner=out/('q18a-native30-localize-owner-run-'+pin[:12]+'.py');runner.write_text(owner);compile(owner,str(runner),'exec')
    for p in final.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
    final.chmod(0o555);runner.chmod(0o444)
    result=dict(bundle=str(final),manifest=pin,overlay=str(overlay),overlay_manifest=sha(overlay/'SHA256SUMS'),runner=str(runner),runner_sha256=sha(runner),runner_size=runner.stat().st_size,review_sha256=closure['review_sha256'],runtime_files_unchanged=len(runtime),checksummed_files=len(verify(final)),held=True)
    dump(out/'build.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--selftest-log',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.out,a.selftest_log),sort_keys=True))
