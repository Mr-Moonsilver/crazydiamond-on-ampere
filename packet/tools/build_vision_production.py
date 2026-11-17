#!/usr/bin/env python3
"""Freeze a new raw-image production candidate from verified MM build7. No host installs."""
import argparse,ast,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
from build_vision_serve import sha,once,sums,repin_params
from vision_anthropic_overlay import transform
ROOT=Path(__file__).resolve().parents[1]
PARENT_SHA='3fa651383cb1dc97c81dbf137cb6477d0084808bc66c9e20b86780ebe6242ee8'
TAG='vision-prod-w27-20261117a';OVERLAY='overlay-v16s5-'+TAG
CLIENTS=('vision_public_gateway','vision_g6_capture','vision_g5_probe','vision_prod_a3','vision_w27_read','vision_answer_probe','vision_encoder_service','vision_encoder_backend','vision_speech_guard','packet_client_contract')
TESTS=('test_vision_public_gateway','test_vision_anthropic_overlay','test_vision_w27_protocol','test_vision_production_bundle','test_vision_encoder_protocol')

def dump(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def build(base,source,out):
    if sha(base/'PACKET-SHA256SUMS')!=PARENT_SHA:raise ValueError('MM build7 parent identity differs')
    for folder,name in ((base,'PACKET-SHA256SUMS'),(base/'harness','SHA256SUMS'),(base/'harness/native30_overlay','SHA256SUMS')):
        subprocess.run(['sha256sum','--strict','--quiet','-c',name],cwd=folder,check=True)
    if out.exists():raise ValueError('new separate packet required')
    shutil.copytree(base,out)
    for p in (out,*out.rglob('*')):
        if p.is_symlink():raise ValueError('symlink in packet')
        p.chmod(p.stat().st_mode|0o200)
    h=out/'harness';o=h/'native30_overlay'
    for name in CLIENTS+TESTS:shutil.copy2(ROOT/'harness'/(name+'.py'),h/(name+'.py'))
    shutil.copy2(ROOT/'tools/vision_anthropic_overlay.py',h/'vision_anthropic_overlay.py')
    original=source.read_text();patched=transform(original);ast.parse(patched)
    (o/'e3_sources/anthropic_serving.py').write_text(patched)
    launch=o/'run-flash-s5-test.sh';text=once(launch.read_text(),'--host 0.0.0.0 --port 8000','--host 127.0.0.1 --port 8001')
    anchor='  s5+=(-v "$OVERLAY/e3_sources/mm_encoder.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/mm/encoder_runner.py:ro")'
    addition='''
  grep -q "  e3_sources/anthropic_serving.py$" "$OVERLAY/SHA256SUMS" || { echo "Anthropic bridge not checksummed" >&2; exit 2; }
  s5+=(-v "$OVERLAY/e3_sources/anthropic_serving.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/serving.py:ro")'''
    text=once(text,anchor,anchor+addition);launch.write_text(text)
    p=h/'identity_declare.py';text,n=re.subn(r"S5_LAUNCHER_SHA='[0-9a-f]{64}'","S5_LAUNCHER_SHA='"+sha(launch)+"'",p.read_text())
    if n!=1:raise ValueError('launcher pin missing')
    p.write_text(text)
    p=out/'serve-launch.sh';text=p.read_text().replace('overlay-v16s5-vision-fe-20261116a',OVERLAY).replace('PORT=${SERVE_PORT:-8000}','PORT=${SERVE_PORT:-8001}')
    text=once(text,'[[ $ARM == engine || $ARM == stock ]]','[[ $ARM == engine ]]')
    text=text.replace('0.0.0.0:$PORT','127.0.0.1:$PORT').replace('00000000:$(printf','0100007F:$(printf')
    p.write_text(text)
    p=out/'serve_readback.py';text=p.read_text().replace("got['host'] == '0.0.0.0'","got['host'] == '127.0.0.1'").replace("argv('--host') != '0.0.0.0'","argv('--host') != '127.0.0.1' or argv('--port') != '8001'")
    p.write_text(text)
    unit=out/'units';unit.mkdir()
    shutil.copy2(ROOT/'deploy/vision-production/flashnext-vision-gateway.service',unit/'flashnext-vision-gateway.service')
    for name in ('gateway-start.sh','gateway-stop.sh','production-pin.sh'):shutil.copy2(ROOT/'deploy/vision-production'/name,out/name)
    shutil.copy2(ROOT/'tools/build_vision_production.py',out/'tools/build_vision_production.py')
    selection=Path('<hub-dir>/agents/flash-next/tmp/final-engine/A3-main100-selection-20261117.json');shutil.copy2(selection,out/selection.name)
    contract=Path('<hub-dir>/agents/flash-next/tmp/final-engine/G6-RECEIPT-CONTRACT-w27-20261117.md')
    if sha(contract)!='eb9749fc979c3167480ef221c884e435824c256f43fd3a0e199ba90b15eb27e3':raise ValueError('<review-group> frozen G6 contract differs')
    shutil.copy2(contract,out/contract.name)
    sys.path.insert(0,str(h/'e3_builder'));import native_readiness as nr
    closure=json.loads((o/'e3'/nr.CLOSURE).read_text());review=closure['review']
    review['native_mm_20261116']['rendered_harness_pins']['identity_declare.py']=sha(h/'identity_declare.py')
    review['vision_production_w27']=dict(reviewed_by='Codex',scope='public CPU gateway + loopback binding + narrow Anthropic trusted-embed conversion; native model/math/bridge unchanged',hardware_qualified=False,source_sha256=sha(source),rendered_source_sha256=sha(o/'e3_sources/anthropic_serving.py'),rendered_harness_pins={'identity_declare.py':sha(h/'identity_declare.py')})
    review['harness_source_shas']['identity_declare.py']=sha(h/'identity_declare.py')
    closure['review_sha256']=nr.digest(review);dump(o/'e3'/nr.CLOSURE,closure);dump(h/'e3_builder/native_candidate_review.json',review)
    gen=json.loads((o/'S5-OVERLAP-GENERATION.json').read_text());ready=json.loads((o/'e3/native-overlap-readiness.json').read_text())
    gen['implementation_review_sha256']=ready['review_sha256']=closure['review_sha256'];ready['launcher_sha256']=sha(launch)
    gen['package_sha256']=ready['package_sha256']=nr.payload_identity(o);ready['receipt_sha256']=nr.digest({k:v for k,v in ready.items() if k!='receipt_sha256'})
    dump(o/'S5-OVERLAP-GENERATION.json',gen);dump(o/'e3/native-overlap-readiness.json',ready)
    delta=json.loads((out/'vision-delta.json').read_text());delta.update(overlay_name=OVERLAY,production_gateway=True);dump(out/'vision-delta.json',delta)
    modules=tuple(closure['generated_source_shas']);nr.verify_finalized_package(o,modules)
    sums(o);sums(h)
    repin_params(out/'serve-params.env',dict(SERVE_TAG=TAG,BUNDLE_SUMS_SHA=sha(h/'SHA256SUMS'),OVERLAY_SUMS_SHA=sha(o/'SHA256SUMS'),S5_LAUNCHER_SHA=sha(launch)))
    dump(out/'production-delta.json',dict(schema='vision-production-w27/1',parent_packet_sha256=PARENT_SHA,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),backend='127.0.0.1:8001',public='0.0.0.0:8000',encoder='<encoder-host>:9041',thinking_defaults='ON/xhigh, unchanged; no default kwargs injected',native_generated_modules_unchanged=True,stock_reference_unchanged=True,no_device_qualification_claim=True))
    for p in out.rglob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
    names=sorted(p.relative_to(out).as_posix() for p in out.iterdir() if p.is_file() and p.name!='PACKET-SHA256SUMS')
    names+=['harness/SHA256SUMS']+[p.relative_to(out).as_posix() for d in ('tools','vision-assets','vision-fit-assets','units') for p in (out/d).rglob('*') if p.is_file()]
    (out/'PACKET-SHA256SUMS').write_text(''.join(sha(out/n)+'  '+n+'\n' for n in sorted(names)))
    print('VISION_PRODUCTION_PACKET '+str(out)+' '+sha(out/'PACKET-SHA256SUMS'))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--anthropic-source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.base,a.anthropic_source,a.out)
