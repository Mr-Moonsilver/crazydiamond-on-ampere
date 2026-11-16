#!/usr/bin/env python3
"""Compose fit-D observation into the frozen vision consumer; no GPU/box actions."""
import argparse,ast,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'patches/s5_engine'),str(ROOT/'harness'),str(ROOT/'tools')]
import make_s5_overlap as compose
import native_readiness as nr
from fit_8x200k_source import transform,packaged
from build_vision_serve import sums,repin_params,once
from vision_fit_8x200k import plan
PARENT_SHA='3fa651383cb1dc97c81dbf137cb6477d0084808bc66c9e20b86780ebe6242ee8'
EXTRA=('fit_memory','graph_event_floor','decode_graph_events','startup_vram_budget')
TAG='vision-fit-8x200k-20261116a'
OVERLAY='overlay-v16s5-'+TAG
NAME='q18a-vision-fit-8x200k-a'
CLIENTS=('vision_fit_8x200k.py','fit_8x200k.py','packet_client_contract.py','test_fit_memory.py','test_vision_fit_8x200k.py',
 'fit_w21_auto_tick.py.src','fit_w21_repro.json','fit_8x200k_w20.json','test_fit_8x200k.py','test_vision_answer_protocol.py','fit_death.py','test_vision_fit_owner.py','test_vision_fit_owner_protocol.py','vision_fit_tokenizer_proof.py','test_vision_fit_guards.py','test_vision_fit_stage.py','test_vision_fit_package.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

def build(parent,out):
    if sha(parent/'PACKET-SHA256SUMS')!=PARENT_SHA:raise ValueError('vision-parent-pin')
    for folder,name in ((parent,'PACKET-SHA256SUMS'),(parent/'harness','SHA256SUMS'),(parent/'harness/native30_overlay','SHA256SUMS')):
        subprocess.run(['sha256sum','--strict','--quiet','-c',name],cwd=folder,check=True)
    if out.exists():raise ValueError('fresh-fit-packet-required')
    shutil.copytree(parent,out)
    for p in out.rglob('*'):
        if p.is_symlink():raise ValueError('fit-packet-symlink')
        p.chmod(p.stat().st_mode|0o200)
    dump(out/'source-identity.json',dict(branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),review_sha256=sha(ROOT/'patches/s5_engine/native_candidate_review.json')))
    h=out/'harness';o=h/'native30_overlay';changes=[]
    for rel in ('e3_sources/core.py','e3_sources/worker.py','e3_sources/api_routers.py','e3/scheduler_hooks.py','e3/native_auto_arm.py','e3/native_measurement.py'):
        p=o/rel;before=p.read_text();after,receipt=transform(rel,before);p.write_text(after);changes.append(receipt)
        dest=h/'fit_parent'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(before)
    for name in EXTRA:
        raw=ROOT/'patches/s5_engine'/(name+'.py');(o/'e3'/(name+'.py')).write_text(packaged(compose.package(raw.read_text())));shutil.copy2(raw,h/'e3_builder'/raw.name)
    for name in CLIENTS:shutil.copy2(ROOT/'harness'/name,h/name)
    transport=out/'patches/s5_engine/transport_source.json';transport.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'patches/s5_engine/transport_source.json',transport)
    raw=h/'e3_builder/native_measurement.py';raw.write_text(transform('e3/native_measurement.py',raw.read_text())[0])
    shutil.copy2(ROOT/'tools/fit_8x200k_source.py',h/'fit_8x200k_source.py')
    shutil.copy2(ROOT/'harness/e3_startup_mount_policy.json',h/'e3_startup_mount_policy.json')
    shutil.copy2(ROOT/'harness/vision_answer_probe.py',h/'vision_answer_probe.py')
    # Explicit public MM argv classification; unknown flags remain unclassified.
    p=h/'engine_identity.py';text=p.read_text()
    text=once(text,"PUBLIC_FLAGS = set('","PUBLIC_FLAGS = set('mm-processor-cache-gb ")
    text=once(text,"PUBLIC_BOOLEANS = set('","PUBLIC_BOOLEANS = set('enable-mm-embeds skip-mm-profiling ")
    text=once(text,"    if k in {'EXTRA_ARGS', 'VISION_ARGS'}:\n        return clean_args(shlex.split(v))",
        "    if k == 'VISION_ARGS':\n        return clean_args(v.split())  # launcher expands this variable without eval\n    if k == 'EXTRA_ARGS':\n        return clean_args(shlex.split(v))")
    p.write_text(text)
    # Frozen setup gate expects this exact opt-in observer and no graph-event/stamp bank.
    env=json.loads((out/'vision-container-env.json').read_text());env['FLASH_S5_FIT_MEMORY']='e6'
    dump(out/'vision-container-env.json',env)
    p=out/'serve-launch.sh';text=p.read_text().replace('overlay-v16s5-vision-fe-20261116a',OVERLAY)
    text=once(text,'SERVE_SETTINGS+=("$kv FLASH_S5_NATIVE_MM=1")','SERVE_SETTINGS+=("$kv FLASH_S5_NATIVE_MM=1 FLASH_S5_FIT_MEMORY=e6")')
    p.write_text(text)
    p=out/'serve_readback.py';text=p.read_text();text=once(text,"'FLASH_S5_NATIVE_MM=1', 'FLASH_S5_OVERLAP=1'","'FLASH_S5_FIT_MEMORY=e6', 'FLASH_S5_NATIVE_MM=1', 'FLASH_S5_OVERLAP=1'");p.write_text(text)
    repin_params(out/'serve-params.env',dict(SERVE_TAG=TAG))
    # Register the four observer sources in the same closure as the unchanged MM bridge.
    probe=o/'e3/decode_probe.py';tree=ast.parse(probe.read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GENERATED_SOURCES' for t in n.targets))
    modules=ast.literal_eval(node.value)+EXTRA;lines=probe.read_text().splitlines(True);lines[node.lineno-1:node.end_lineno]=['GENERATED_SOURCES = '+repr(modules)+'\n'];probe.write_text(''.join(lines))
    generated={n:sha(o/'e3'/(n+'.py')) for n in modules}
    closure=json.loads((o/'e3'/nr.CLOSURE).read_text());review=closure['review']
    receipt=dict(
        harness_sources={n:sha(ROOT/'harness'/n) for n in CLIENTS+('vision_answer_probe.py',)},
        owner_sha256=sha(ROOT/'deploy/vision-fit/owner.sh'),
        public_identity_flags=('enable-mm-embeds','mm-processor-cache-gb','skip-mm-profiling'),
        vision_identity_word_split='matches unquoted Bash variable expansion; literal JSON quotes preserved',
        reviewed_by='Codex',scope='fit-D observer plus MM implementation closure only; CPU qualification recorded in delivery',
        fit_parent='f5c9db7676f59eecc418006505bac933c6bbd89d',default_off='FLASH_S5_FIT_MEMORY',changes=changes,
        transform_sha256=sha(ROOT/'tools/fit_8x200k_source.py'),added_graphs_events_streams_tensors=0,hardware_qualified=False,
        source_shas={n+'.py':sha(ROOT/'patches/s5_engine'/(n+'.py')) for n in EXTRA})
    review['vision_fit_20261116']=receipt
    closure['review_sha256']=nr.digest(review);closure['generated_source_shas']=generated;dump(o/'e3'/nr.CLOSURE,closure)
    evidence=json.loads((o/'e3/decode-probe-evidence.json').read_text());evidence['generated_source_shas']=generated;dump(o/'e3/decode-probe-evidence.json',evidence)
    dump(h/'vision-fit-engine-review.json',receipt)
    shutil.copy2(ROOT/'deploy/vision-fit/owner.sh',out/'owner.sh')
    shutil.copy2(ROOT/'deploy/vision-fit/seq.sh',out/'seq.sh')
    for name in ('vision_fit_identity.py','vision_fit_finish.py','vision_fit_stage_inputs.py','vision_fit_declaration_proof.py'):shutil.copy2(ROOT/'tools'/name,out/'tools'/name)
    binding=Path('<hub-dir>/cargo-target/flash-next/vision-native-mm-20261116/vision-fit-inputs-v<vm-id>-b-20261116.json')
    if sha(binding)!=plan()['inputs_binding_sha256']:raise ValueError('input-binding-sha')
    shutil.copy2(binding,out/'vision-fit-inputs.json')
    for src,dst in (('<hub-dir>/agents/flash-next/tmp/final-engine/OPERATOR-GRANT-w26-vision-addendum-20261116.md','w26-grant.md'),
                    ('<hub-dir>/agents/flash-next/tmp/final-engine/PLAN-vision-fit-8x200k-b-20261116.md','vision-fit-plan-b.md')):shutil.copy2(src,out/dst)
    rule=Path('<hub-dir>/agents/flash-next/tmp/final-engine/RULE-vision-fit-M-w26-20261116.md')
    shutil.copy2(rule,out/'vision-fit-rule.md')
    dump(out/'fit-plan.json',dict(plan(),rule_sha256=sha(rule)))
    (out/'image-id.txt').write_text('sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b\n')
    # Compatibility with the existing CPU owner proof; does not alter the model payload.
    delta=json.loads((out/'vision-delta.json').read_text());delta.update(overlay_name=OVERLAY,fit_observer=True,fit_plan_sha256=sha(out/'fit-plan.json'));dump(out/'vision-delta.json',delta)
    generation=json.loads((o/'S5-OVERLAP-GENERATION.json').read_text());ready=json.loads((o/'e3/native-overlap-readiness.json').read_text())
    generation['implementation_review_sha256']=ready['review_sha256']=closure['review_sha256']
    ready['launcher_sha256']=sha(o/'run-flash-s5-test.sh')
    generation['package_sha256']=ready['package_sha256']=nr.payload_identity(o)
    ready['receipt_sha256']=nr.digest({k:v for k,v in ready.items() if k!='receipt_sha256'})
    dump(o/'S5-OVERLAP-GENERATION.json',generation);dump(o/'e3/native-overlap-readiness.json',ready)
    sums(o);sums(h)
    repin_params(out/'serve-params.env',dict(BUNDLE_SUMS_SHA=sha(h/'SHA256SUMS'),OVERLAY_SUMS_SHA=sha(o/'SHA256SUMS')))
    nr.verify_finalized_package(o,modules)
    for p in (out/'owner.sh',out/'seq.sh',out/'serve-launch.sh'):subprocess.run(['bash','-n',str(p)],check=True)
    names=sorted(p.relative_to(out).as_posix() for p in out.iterdir() if p.is_file() and p.name!='PACKET-SHA256SUMS')
    names+=['harness/SHA256SUMS']+[p.relative_to(out).as_posix() for sub in ('tools','vision-assets','vision-fit-assets','patches') for p in (out/sub).rglob('*') if p.is_file()]
    (out/'PACKET-SHA256SUMS').write_text(''.join(sha(out/n)+'  '+n+'\n' for n in sorted(names)))
    print('VISION_FIT_PACKET '+str(out)+' '+sha(out/'PACKET-SHA256SUMS'))
if __name__=='__main__':
    from packet_client_contract import report_exception
    try:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
        a=p.parse_args();build(a.parent,a.out)
    except Exception as exc:report_exception(exc);raise SystemExit(2)
