#!/usr/bin/env python3
"""Bind unchanged w27 full G4 and production wrappers to the qualified headroom overlay. CPU/files only."""
import argparse,ast,json,shutil,subprocess,sys
from pathlib import Path
from build_vision_serve import once,sums,sha,repin_params
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'harness')]
OLD=Path('<hub-dir>/cargo-target/flash-next/vision-production-w27-20261117')
PRODUCTION=OLD/'serve-vision-prod-w27-20261117a-build-10'
FIT=OLD/'vision-fit-production-overlay-build-4'
HEADROOM=Path('<hub-dir>/cargo-target/flash-next/g4-headroom-build-20261118/build-8')
PINS={PRODUCTION:'67bc07698431ed089143feb3e9b01f70fb55deeb19c0a2024692aab7948401b6',FIT:'533d71e2e78f3f8b8a6fc5cf3a0bbfd930806fef62e86af73569c7f76b14c676',HEADROOM:'37a45485f543cad8b44b8624898a625d414419a9d57f338a430451fe1bd11451'}
OVERLAY_SHA='fa65ce90a37a3f2ba99445ea298833a1dc54b8e13a004b51182669e33a51b08e'
OVERLAY='overlay-v16s5-g4-headroom-20261118a'
CT_PROD='/data/flash-next-packets/serve-vision-prod-w28-20261118a'
CT_FIT='/data/flash-next-packets/vision-fit-headroom-w28-20261118a'
OLD_CT='/data/flash-next-packets/serve-vision-prod-w27-20261117a'
FULL_FILES=('fit-plan.json','protocol-b.md','vision-fit-rule.md','vision-fit-inputs.json')
FULL_CLIENTS=('vision_fit_8x200k.py','fit_8x200k.py')

def dump(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def verify(packet):
    if sha(packet/'PACKET-SHA256SUMS')!=PINS[packet]:raise ValueError('frozen-parent-pin:'+str(packet))
    for folder,name in ((packet,'PACKET-SHA256SUMS'),(packet/'harness','SHA256SUMS'),(packet/'harness/native30_overlay','SHA256SUMS')):
        subprocess.run(['sha256sum','--strict','--quiet','-c',name],cwd=folder,check=True)
def root_copy(parent,out):
    out.mkdir(exist_ok=False)
    for path in parent.iterdir():
        if path.name=='harness':continue
        target=out/path.name
        if path.is_dir():shutil.copytree(path,target)
        else:shutil.copy2(path,target)
    shutil.copytree(HEADROOM/'harness',out/'harness')
    for path in (out,*out.rglob('*')):
        if path.is_symlink():raise ValueError('packet-symlink')
        path.chmod(path.stat().st_mode|0o200)
    # Restore the full registered workload; no discriminator simplifications travel with the overlay.
    for name in FULL_CLIENTS:shutil.copy2(FIT/'harness'/name,out/'harness'/name)

def rebind(text):
    return (text.replace(OLD_CT,CT_PROD)
        .replace('/data/flash-next-control/vision-prod-w27','/data/flash-next-control/vision-prod-w28')
        .replace('overlay-v16s5-vision-prod-w27-20261117a',OVERLAY)
        .replace('flashnext-vision-gateway-w27','flashnext-vision-gateway-w28')
        .replace('/root/flashnext-vision-w27-promotion','/root/flashnext-vision-w28-promotion')
        .replace('/root/flashnext-prod-pins/vision-w27','/root/flashnext-prod-pins/vision-w28')
        .replace('flashnext-boot-guard-base-w27.sh','flashnext-boot-guard-base-w28.sh'))

def finish(packet):
    for path in packet.rglob('*.sh'):subprocess.run(['bash','-n',str(path)],check=True)
    for path in list((packet/'harness').glob('*.py'))+list((packet/'tools').glob('*.py')):ast.parse(path.read_text())
    names=sorted(p.relative_to(packet).as_posix() for p in packet.iterdir() if p.is_file() and p.name!='PACKET-SHA256SUMS')
    names+=['harness/SHA256SUMS']+[p.relative_to(packet).as_posix() for sub in ('tools','vision-assets','vision-fit-assets','patches','units','rollback') for p in (packet/sub).rglob('*') if p.is_file()]
    (packet/'PACKET-SHA256SUMS').write_text(''.join(sha(packet/n)+'  '+n+'\n' for n in sorted(set(names))))
    subprocess.run(['sha256sum','--strict','--quiet','-c','PACKET-SHA256SUMS'],cwd=packet,check=True)

def build(out):
    for parent in PINS:verify(parent)
    if sha(HEADROOM/'harness/native30_overlay/SHA256SUMS')!=OVERLAY_SHA:raise ValueError('headroom-overlay-pin')
    out.mkdir(exist_ok=False);prod=out/'production';fit=out/'g4-full'
    root_copy(PRODUCTION,prod)
    # Only wrapper/readiness/pin namespaces move. Existing public+encoder credentials stay at their approved paths.
    wrappers=('gateway-start.sh','gateway-stop.sh','production-pin.sh','production-rollback.sh','flashnext-boot-guard-w27.sh',
              'units/flashnext-vision-gateway.service','units/native-ready.conf','tools/vision_production_pin_guard.py',
              'harness/vision_production_ready.py','harness/vision_gateway_launch_selfrun.py')
    for name in wrappers:
        path=prod/name;path.write_text(rebind(path.read_text()))
    p=prod/'serve-launch.sh';p.write_text(once(rebind(p.read_text()),'"$kv FLASH_S5_NATIVE_MM=1"','"$kv FLASH_S5_NATIVE_MM=1 FLASH_S5_G4_HEADROOM=1"'))
    p=prod/'serve_readback.py';p.write_text(once(p.read_text(),"'FLASH_S5_NATIVE_MM=1', 'FLASH_S5_OVERLAP=1'","'FLASH_S5_G4_HEADROOM=1', 'FLASH_S5_NATIVE_MM=1', 'FLASH_S5_OVERLAP=1'"))
    env=json.loads((prod/'vision-container-env.json').read_text());env['FLASH_S5_G4_HEADROOM']='1';dump(prod/'vision-container-env.json',env)
    # The public credential tag is deliberately retained; no LiteLLM/<llm-gateway> key rotation is needed.
    repin_params(prod/'serve-params.env',dict(OVERLAY_SUMS_SHA=OVERLAY_SHA,BUNDLE_SUMS_SHA='pending'))
    for rel in ('harness/test_vision_headroom_w28.py','tools/build_vision_headroom_w28.py'):
        dest=prod/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    sums(prod/'harness')
    repin_params(prod/'serve-params.env',dict(BUNDLE_SUMS_SHA=sha(prod/'harness/SHA256SUMS')))
    delta=json.loads((prod/'vision-delta.json').read_text());head=json.loads((HEADROOM/'vision-delta.json').read_text())
    delta.update(overlay_name=OVERLAY,generated_module_shas=head['generated_module_shas'],headroom_flag='FLASH_S5_G4_HEADROOM',hardware_qualified=False)
    dump(prod/'vision-delta.json',delta)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    binding=dict(schema='vision-headroom-w28/1',source_commit=source,engine_source_commit='ac652378f9bdeb7757f01ef79d7ed45098a1caec',
        headroom_packet_sha256=PINS[HEADROOM],overlay_sha256=OVERLAY_SHA,harness_sha256=sha(prod/'harness/SHA256SUMS'),
        flag='FLASH_S5_G4_HEADROOM=1',model_max_len=200000,max_seqs=8,kv_bytes=2150000000,rowcache=[8,6144],marlin_mib=200,
        fit_observer='e6 only in G4; production default off, as in w27',production_ct=CT_PROD,fit_ct=CT_FIT,
        public_auth_tag_retained='vision-prod-w27-20261117a',credential_values_in_packet=False,
        gates='w27 frozen rules and amendments; no carryover waiver in code',device_qualified=False)
    dump(prod/'w28-binding.json',dict(binding,role='production',parent_packet_sha256=PINS[PRODUCTION]))
    production=json.loads((prod/'production-delta.json').read_text());production.update(schema='vision-production-w28/1',source_commit=source,
        parent_packet_sha256=PINS[PRODUCTION],overlay_sha256=OVERLAY_SHA,headroom_enabled=True,no_device_qualification_claim=True)
    dump(prod/'production-delta.json',production)
    finish(prod)
    # Keep root G4 workload/rule/owner/death/identity semantics byte-for-byte; only the launch binding changes.
    root_copy(FIT,fit)
    # Each wrapper runs the same frozen harness/overlay. No second native source rendering.
    for path in (prod/'harness').rglob('*'):
        if path.is_file():shutil.copy2(path,fit/'harness'/path.relative_to(prod/'harness'))
    for name in ('serve-launch.sh','serve_readback.py','serve-params.env'):
        shutil.copy2(prod/name,fit/name)
    p=fit/'serve-launch.sh';p.write_text(once(p.read_text(),'"$kv FLASH_S5_NATIVE_MM=1 FLASH_S5_G4_HEADROOM=1"','"$kv FLASH_S5_NATIVE_MM=1 FLASH_S5_G4_HEADROOM=1 FLASH_S5_FIT_MEMORY=e6"'))
    p=fit/'serve_readback.py';p.write_text(once(p.read_text(),"'FLASH_S5_G4_HEADROOM=1', 'FLASH_S5_NATIVE_MM=1'","'FLASH_S5_FIT_MEMORY=e6', 'FLASH_S5_G4_HEADROOM=1', 'FLASH_S5_NATIVE_MM=1'"))
    env['FLASH_S5_FIT_MEMORY']='e6';dump(fit/'vision-container-env.json',env)
    repin_params(fit/'serve-params.env',dict(SERVE_TAG='vision-fit-headroom-w28-20261118a'))
    delta['fit_observer']=True;delta['fit_plan_sha256']=sha(fit/'fit-plan.json');dump(fit/'vision-delta.json',delta)
    preserved={name:sha(fit/name) for name in FULL_FILES}
    preserved.update({'harness/'+name:sha(fit/'harness'/name) for name in FULL_CLIENTS})
    for name,value in preserved.items():
        if value!=sha(FIT/name):raise ValueError('frozen-full-G4-moved:'+name)
    for name in ('owner.sh','seq.sh','tools/vision_fit_finish.py','tools/vision_fit_identity.py'):
        if sha(fit/name)!=sha(FIT/name):raise ValueError('full-G4-owner-moved:'+name)
    dump(fit/'production-overlay-binding.json',dict(binding,role='full-G4',production_packet_sha256=sha(prod/'PACKET-SHA256SUMS'),
        frozen_fit_parent_sha256=PINS[FIT],fit_workload_unchanged=True,preserved=preserved))
    dump(fit/'w28-binding.json',dict(binding,role='full-G4',parent_packet_sha256=PINS[FIT]))
    finish(fit)
    if sha(fit/'harness/SHA256SUMS')!=sha(prod/'harness/SHA256SUMS'):raise ValueError('harness-not-identical')
    print('W28_PAIR PASS production='+sha(prod/'PACKET-SHA256SUMS')+' full-G4='+sha(fit/'PACKET-SHA256SUMS')+' overlay='+OVERLAY_SHA)

if __name__=='__main__':
    from packet_client_contract import report_exception
    try:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);build(p.parse_args().out)
    except Exception as exc:report_exception(exc);raise SystemExit(2)
