#!/usr/bin/env python3
"""Derive a fresh CPU-qualified discriminator from the exact w27 G4 overlay; no host installs."""
import argparse,ast,json,shutil,subprocess,sys
from pathlib import Path
from build_vision_serve import once,sums,sha,repin_params
from g4_headroom_packet_source import client,base_reader
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'harness'),str(ROOT/'patches/s5_engine')]
PARENT=Path('<hub-dir>/cargo-target/flash-next/vision-production-w27-20261117/vision-fit-production-overlay-build-4')
PARENT_SHA='533d71e2e78f3f8b8a6fc5cf3a0bbfd930806fef62e86af73569c7f76b14c676'
CHANGED=('lane_state','native_resources','lane_private','native_startup','model_hooks','qsa_driver','split_bank','startup_vram_budget')
TAG='g4-headroom-20261118a';OVERLAY='overlay-v16s5-'+TAG

def dump(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')

def build(out):
    if sha(PARENT/'PACKET-SHA256SUMS')!=PARENT_SHA:raise ValueError('w27-G4-parent-pin')
    for folder,manifest in ((PARENT,'PACKET-SHA256SUMS'),(PARENT/'harness','SHA256SUMS'),(PARENT/'harness/native30_overlay','SHA256SUMS')):
        subprocess.run(['sha256sum','--strict','--quiet','-c',manifest],cwd=folder,check=True)
    if out.exists():raise ValueError('fresh-packet-required')
    review=json.loads((ROOT/'patches/s5_engine/native_candidate_review.json').read_text())
    if review.get('g4_headroom_20261118',{}).get('reviewed_by')!='Codex':raise ValueError('headroom-review-missing')
    for name in (*CHANGED,'make_s5_overlap'):
        if review['source_shas'].get(name+'.py')!=sha(ROOT/'patches/s5_engine'/(name+'.py')):raise ValueError('headroom-source-not-reviewed:'+name)
    for name in ('test_g4_headroom.py','test_g4_headroom_packet.py','test_n3_indexer_lifetime.py'):
        if review['harness_source_shas'].get(name)!=sha(ROOT/'harness'/name):raise ValueError('headroom-proof-not-reviewed:'+name)
    shutil.copytree(PARENT,out)
    for p in (out,*out.rglob('*')):
        if p.is_symlink():raise ValueError('packet-symlink')
        p.chmod(p.stat().st_mode|0o200)
    h=out/'harness';o=h/'native30_overlay'
    sys.path.insert(0,str(ROOT/'patches/s5_engine'))
    import make_s5_overlap as compose,native_readiness as nr
    for name in CHANGED:
        path=ROOT/'patches/s5_engine'/(name+'.py')
        (o/'e3'/path.name).write_text(compose.package(path.read_text()))
        shutil.copy2(path,h/'e3_builder'/path.name)
    raw=ROOT/'patches/s5_engine/make_s5_overlap.py';shutil.copy2(raw,h/'e3_builder'/raw.name)
    reference=ROOT/'reference/vllm-src'/compose.INDEXER_PATH
    (o/'ops_qsa_indexer.py').write_text(compose.compose_indexer(reference.read_text()))
    for name in ('vision_fit_8x200k.py','fit_8x200k.py'):
        original=(h/name).read_text()
        (h/name).write_text((client if name.startswith('vision_') else base_reader)(original))
    for rel in ('harness/test_g4_headroom.py','harness/test_g4_headroom_packet.py','harness/test_n3_indexer_lifetime.py',
                'tools/build_g4_headroom.py','tools/g4_headroom_packet_source.py','tools/g4_headroom_read.py','tools/g4_headroom_mutants.py'):
        p=ROOT/rel;dest=h/p.name if rel.startswith('harness/') else out/rel
        dest.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(p,dest)
    p=out/'serve-launch.sh';text=p.read_text().replace('overlay-v16s5-vision-prod-w27-20261117a',OVERLAY)
    text=once(text,'FLASH_S5_NATIVE_MM=1 FLASH_S5_FIT_MEMORY=e6','FLASH_S5_NATIVE_MM=1 FLASH_S5_FIT_MEMORY=e6 FLASH_S5_G4_HEADROOM=1')
    p.write_text(text)
    p=out/'serve_readback.py';p.write_text(once(p.read_text(),"'FLASH_S5_FIT_MEMORY=e6', 'FLASH_S5_NATIVE_MM=1'","'FLASH_S5_G4_HEADROOM=1', 'FLASH_S5_FIT_MEMORY=e6', 'FLASH_S5_NATIVE_MM=1'"))
    env=json.loads((out/'vision-container-env.json').read_text());env['FLASH_S5_G4_HEADROOM']='1';dump(out/'vision-container-env.json',env)
    p=out/'owner.sh';p.write_text(once(p.read_text(),'read_rc=$?','''read_rc=$?
if [[ $read_rc == 0 ]]; then
  python3 -B "$PK/tools/g4_headroom_read.py" --workload "$RUN/fit-read.json" --engine-log "$RUN/engine-stream.log" --out "$RUN/headroom-read.json" \\
    >"$RUN/headroom-read.stdout.log" 2>"$RUN/headroom-read.stderr.log"
  read_rc=$?
  cat "$RUN/headroom-read.stdout.log"
fi'''))
    p=out/'tools/vision_fit_finish.py';p.write_text(once(p.read_text(),"scope='box fit with images, encoder link on<vm-id> not tested'", "scope='headroom discriminator only, not full G4 or production qualification'").replace('VISION_FIT_OWNER','G4_HEADROOM_OWNER'))
    p=out/'tools/vision_fit_finish.py';text=p.read_text()
    text=once(text,"    if phase_rc:reasons.append('owner-phase-rc-'+str(phase_rc))", "    try:\n        headroom=json.loads((root/'headroom-read.json').read_text())\n        if headroom.get('verdict')!='PASS':reasons.append('headroom-not-PASS')\n    except (OSError,ValueError):reasons.append('headroom-reader-missing')\n    if phase_rc:reasons.append('owner-phase-rc-'+str(phase_rc))")
    p.write_text(text)
    plan=dict(schema='g4-headroom-plan/1',name='q18a-g4-headroom-a',scope='headroom discriminator only; not full G4',
        warmup='unchanged serve startup: 302 + 70903; then same FULL semantic calibration',
        stress=dict(agents=8,prompt_tokens=190320,images_per_agent=1,image_rows=1024,output_tokens=8192),
        physical_stop_mib=40,headroom_target_mib=250,kv_bytes=2150000000,rowcache=[8,6144],marlin_mib=200,
        native_workload_cap_s=2700,startup_cap_s=900,start_identity_cap_s=900,end_identity_cap_s=1200,outer_cap_s=6000,cleanup_grace_s=330,
        not_run=['growth intermediates','MM reuse/cancel device cases','cold-beside-seven','full G4','production adoption'])
    dump(out/'fit-plan.json',plan)
    (out/'headroom-protocol.md').write_text('# Prospective G4 headroom discriminator\n\n'+json.dumps(plan,indent=2)+'\n\nAll original keyed, identity, 40 MiB, no-OOM/preemption, fixed output, MM semantic/native coverage and owned-stop/death checks remain. Extra PASS requires all four ranks >=250 MiB and complete startup/first-score receipts. A PASS does not grant full G4 FIT. No launch authorization.\n')
    # The signed source review is carried unchanged; no review or hardware override.
    closure=json.loads((o/'e3'/nr.CLOSURE).read_text());closure['review']=review;closure['review_sha256']=nr.digest(review)
    generated={name:sha(o/'e3'/(name+'.py')) for name in closure['generated_source_shas']};closure['generated_source_shas']=generated
    dump(o/'e3'/nr.CLOSURE,closure);dump(h/'e3_builder/native_candidate_review.json',review)
    evidence=json.loads((o/'e3/decode-probe-evidence.json').read_text());evidence['generated_source_shas']=generated;dump(o/'e3/decode-probe-evidence.json',evidence)
    gen=json.loads((o/'S5-OVERLAP-GENERATION.json').read_text());ready=json.loads((o/'e3/native-overlap-readiness.json').read_text())
    gen['implementation_review_sha256']=ready['review_sha256']=closure['review_sha256']
    gen['package_sha256']=ready['package_sha256']=nr.payload_identity(o)
    ready['receipt_sha256']=nr.digest({k:v for k,v in ready.items() if k!='receipt_sha256'})
    dump(o/'S5-OVERLAP-GENERATION.json',gen);dump(o/'e3/native-overlap-readiness.json',ready)
    nr.verify_finalized_package(o,tuple(generated))
    delta=json.loads((out/'vision-delta.json').read_text());delta.update(overlay_name=OVERLAY,
        generated_module_shas=generated,fit_plan_sha256=sha(out/'fit-plan.json'),
        headroom_flag='FLASH_S5_G4_HEADROOM',scope=plan['scope']);dump(out/'vision-delta.json',delta)
    sums(o);sums(h)
    repin_params(out/'serve-params.env',dict(SERVE_TAG=TAG,BUNDLE_SUMS_SHA=sha(h/'SHA256SUMS'),OVERLAY_SUMS_SHA=sha(o/'SHA256SUMS')))
    dump(out/'g4-headroom-binding.json',dict(parent_packet_sha256=PARENT_SHA,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        changed_modules=list(CHANGED),generated_indexer_sha256=sha(o/'ops_qsa_indexer.py'),overlay_sha256=sha(o/'SHA256SUMS'),parent_overlay_sha256='37d33ee72fba0f411cac46d3b27a91c3b38739408651196e6c743f2434fce55a',
        no_device_qualification=True,packet_rule='headroom-protocol.md; original full G4 unchanged in parent'))
    for p in out.rglob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
    for p in list(h.glob('*.py'))+list((out/'tools').glob('*.py')):ast.parse(p.read_text())
    names=sorted(p.relative_to(out).as_posix() for p in out.iterdir() if p.is_file() and p.name!='PACKET-SHA256SUMS')
    names+=['harness/SHA256SUMS']+[p.relative_to(out).as_posix() for sub in ('tools','vision-assets','vision-fit-assets','patches') for p in (out/sub).rglob('*') if p.is_file()]
    (out/'PACKET-SHA256SUMS').write_text(''.join(sha(out/n)+'  '+n+'\n' for n in sorted(names)))
    print('G4_HEADROOM_PACKET '+str(out)+' '+sha(out/'PACKET-SHA256SUMS'))

if __name__=='__main__':
    from packet_client_contract import report_exception
    try:
        p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);build(p.parse_args().out)
    except Exception as exc:report_exception(exc);raise SystemExit(2)
