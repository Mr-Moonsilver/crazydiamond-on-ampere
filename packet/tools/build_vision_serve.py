#!/usr/bin/env python3
"""Derive a NEW keyed vision consumer from released serve-c. CPU/files only; never installs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
BASE_SUMS='a2e90784073f8c317c2649373524dbcfd2ae5c5eac90fa7c3e5caf337791be1f'
VISION_ARGS='--enable-mm-embeds --limit-mm-per-prompt {"image":0,"video":0} --mm-processor-cache-gb 0 --skip-mm-profiling'
OLD_OVERLAY='overlay-v16s5-q18a-parity-n3s24q2024p-j2-cxfx42-b1'
NEW_OVERLAY='overlay-v16s5-vision-fe-20261116a'
TAG='vision-fe-20261116a'
CHANGED=('wire_v1','lane_chunk','worker_bridge','scheduler_hooks','lane_dispatch','decode_probe','native_mm','native_startup')
CLIENT_FILES=('harness/vision_answer_probe.py','tools/vision_answer_read.py',
    'tools/vision_mm_memory_read.py',
    'harness/vision_embeddings_export.py',
    'tools/build_vision_encoder_packet.py',
    'harness/vision_encoder_service.py','harness/vision_encoder_backend.py','harness/vision_speech_guard.py',
    'harness/vision_tower_probe.py','harness/test_native_mm.py','harness/test_vision_answer_protocol.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('vision render anchor missing or repeated: '+old[:70])
    return text.replace(old,new)


def sums(folder):
    names=sorted(p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p!=folder/'SHA256SUMS')
    (folder/'SHA256SUMS').write_text(''.join(sha(folder/n)+'  '+n+'\n' for n in names))


def repin_params(path,values):
    text=path.read_text()
    for k,v in values.items():
        text,count=re.subn('^'+re.escape(k)+r'=.*$',k+'='+v,text,flags=re.M)
        if count!=1:raise ValueError('one declared parameter required: '+k)
    path.write_text(text)


def exception_details(source):
    """Keep legacy verdict handling, while preserving every broad-caught exception."""
    helper='''
def _packet_error(exc):
    import sys, traceback
    frame=traceback.extract_tb(exc.__traceback__)[-1]
    print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {str(exc)[:256]} at {frame.filename}:{frame.lineno}', file=sys.stderr)

'''
    lines=source.splitlines(keepends=True)
    for node in sorted((n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.ExceptHandler)
                        and isinstance(n.type,ast.Name) and n.type.id in ('Exception','BaseException')),
                       key=lambda n:n.lineno,reverse=True):
        name=node.name or '_packet_exc'
        if node.name is None:
            lines[node.lineno-1]=lines[node.lineno-1].replace('except '+node.type.id+':',
                'except '+node.type.id+' as '+name+':')
        first=node.body[0]
        lines.insert(first.lineno-1,' '*first.col_offset+'_packet_error('+name+')\n')
    # After imports/docstring, before all function definitions and the entrypoint.
    result=''.join(lines)
    tree=ast.parse(result)
    first=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)))
    lines=result.splitlines(keepends=True);lines.insert(first.lineno-1,helper)
    result=''.join(lines);ast.parse(result)
    return result


def build(base,budget,encoder,out):
    if sha(base/'PACKET-SHA256SUMS')!=BASE_SUMS:raise ValueError('released serve-c identity differs')
    for folder,manifest in ((base,'PACKET-SHA256SUMS'),(base/'harness','SHA256SUMS'),(base/'harness/native30_overlay','SHA256SUMS')):
        subprocess.run(['sha256sum','--strict','--quiet','-c',manifest],cwd=folder,check=True)
    if out.exists() or out.resolve().is_relative_to(base.resolve()):raise ValueError('fresh separate packet required')
    sys.path.insert(0,str(ROOT/'patches/s5_engine'))
    import make_s5_overlap as compose
    import decode_probe,native_readiness,native_mm_overlay
    review=json.loads((ROOT/'patches/s5_engine/native_candidate_review.json').read_text())
    native_readiness.check_review(review)
    for name in {n+'.py' for n in compose.E3_MODULES}|{'make_s5_overlap.py','native_mm_overlay.py'}|set(compose.NATIVE_FILES):
        if review['source_shas'].get(name)!=sha(ROOT/'patches/s5_engine'/name):raise ValueError('unreviewed source: '+name)
    for name,expected in review['harness_source_shas'].items():
        if sha(ROOT/'harness'/name)!=expected:raise ValueError('unreviewed harness source: '+name)
    budget_text=native_mm_overlay.budget_source(budget.read_text())
    encoder_text=native_mm_overlay.encoder_source(encoder.read_text())
    shutil.copytree(base,out)
    for path in (out,*out.rglob('*')):
        if path.is_symlink():raise ValueError('symlink in released packet copy')
        path.chmod(path.stat().st_mode | 0o200)
    h=out/'harness';overlay=h/'native30_overlay'
    for name in CHANGED:
        source=(ROOT/'patches/s5_engine'/(name+'.py')).read_text()
        (overlay/'e3'/(name+'.py')).write_text(compose.package(source))
    for name in (*CHANGED,'make_s5_overlap','native_mm_overlay'):
        shutil.copy2(ROOT/'patches/s5_engine'/(name+'.py'),h/'e3_builder'/(name+'.py'))
    (overlay/'e3_sources/mm_budget.py').write_text(budget_text)
    (overlay/'e3_sources/mm_encoder.py').write_text(encoder_text)
    launcher=overlay/'run-flash-s5-test.sh'
    addition='''  grep -q "  e3_sources/mm_budget.py$" "$OVERLAY/SHA256SUMS" || { echo "MM budget mount not checksummed" >&2; exit 2; }
  s5+=(-v "$OVERLAY/e3_sources/mm_budget.py:/usr/local/lib/python3.12/dist-packages/vllm/multimodal/encoder_budget.py:ro")
  grep -q "  e3_sources/mm_encoder.py$" "$OVERLAY/SHA256SUMS" || { echo "MM encoder mount not checksummed" >&2; exit 2; }
  s5+=(-v "$OVERLAY/e3_sources/mm_encoder.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/mm/encoder_runner.py:ro")
'''
    launcher_text=once(launcher.read_text(),'fi\n# VISION=1: image input',addition+'fi\n# VISION=1: image input')
    launcher_text=once(launcher_text,'if [[ -f $OVERLAY/vision/model.py ]]; then',
        'if [[ -f $OVERLAY/vision/model.py && ! -d $OVERLAY/e3 ]]; then')
    launcher.write_text(launcher_text)
    # This is a derived launcher identity, never an allowlist bypass.
    identity=h/'identity_declare.py'
    text,count=re.subn(r"S5_LAUNCHER_SHA='[0-9a-f]{64}'", "S5_LAUNCHER_SHA='"+sha(launcher)+"'",identity.read_text())
    if count!=1:raise ValueError('frozen launcher declaration missing')
    identity.write_text(text)
    for name in CLIENT_FILES:
        destination=out/name if name.startswith('tools/') else h/Path(name).name
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
    for name in ('selftest.py','test_e3_package.py'):
        shutil.copy2(ROOT/'harness'/name,h/name)
    assets=out/'vision-assets';assets.mkdir()
    for name in ('table.png','diagram.png','screenshot.png'):shutil.copy2(ROOT/'harness/vision_assets'/name,assets/name)
    from build_vision_encoder_packet import red_png
    (assets/'red.png').write_bytes(red_png(512))
    manifest=json.loads((ROOT/'harness/vision_assets/manifest.json').read_text())
    cases=[dict(id='red',image='red.png',question='What color fills this image? Answer with one word.',expected='red')]
    cases += [dict(id=n[:-4],image=n,question=manifest[n]['q'],expected=manifest[n]['a']) for n in ('table.png','diagram.png','screenshot.png')]
    (assets/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    fit_assets=out/'vision-fit-assets';fit_assets.mkdir()
    fit_cases=[]
    for name,rgb in (('red',(255,0,0)),('green',(0,255,0)),('blue',(0,0,255)),('yellow',(255,255,0)),
                     ('cyan',(0,255,255)),('magenta',(255,0,255)),('black',(0,0,0)),('white',(255,255,255))):
        (fit_assets/(name+'.png')).write_bytes(red_png(1024,rgb))
        fit_cases.append(dict(id=name,image=name+'.png',question='Choose the solid color: red, green, blue, yellow, cyan, magenta, black or white. Answer with one word.',expected=name))
    (fit_assets/'cases.json').write_text(json.dumps(fit_cases,indent=2)+'\n')
    start=out/'serve-launch.sh';text=start.read_text().replace(OLD_OVERLAY,NEW_OVERLAY)
    text=once(text,'      MAX_MODEL_LEN=*|MAX_SEQS=*)','      EXTRA_ENV=*) [[ $kv != *FLASH_S5_NATIVE_MM* ]] || die settings "duplicate MM switch"\n                   SERVE_SETTINGS+=("$kv FLASH_S5_NATIVE_MM=1") ;;\n      MAX_MODEL_LEN=*|MAX_SEQS=*)')
    text=once(text,'  SERVE_SETTINGS+=("MAX_MODEL_LEN=$MAX_MODEL_LEN" "MAX_SEQS=$MAX_SEQS")',
        '  SERVE_SETTINGS+=("MAX_MODEL_LEN=$MAX_MODEL_LEN" "MAX_SEQS=$MAX_SEQS" "VISION=1" '+__import__('shlex').quote('VISION_ARGS='+VISION_ARGS)+')')
    text=once(text,'${#S5_SETTINGS[@]} + 2','${#S5_SETTINGS[@]} + 4')
    text=once(text,'--g0-env "$PK/g0-container-env.json"','--g0-env "$PK/vision-container-env.json"')
    text=once(text,'  step startup 0 "cx startup grade PASS, engine_epoch $EPOCH, session run_id bound to $RUN_ID"',
        '''  step startup 0 "cx startup grade PASS, engine_epoch $EPOCH, session run_id bound to $RUN_ID"
  python3 -B "$PK/tools/vision_mm_memory_read.py" --startup-grade "$OUT/startup-grade.json" --out "$OUT/mm-memory.json" > "$OUT/mm-memory.log" 2>&1 ||
    fail_stop mm-memory "all-rank MM construction budget not proven"
  step mm-memory 0 "$(tail -1 "$OUT/mm-memory.log")"''')
    start.write_text(text)
    readback=out/'serve_readback.py';text=readback.read_text()
    text=once(text,"        env = new.get('EXTRA_ENV', '').split()",
        "        want.update(VISION='1', VISION_ARGS="+repr(VISION_ARGS)+")\n        env = new.get('EXTRA_ENV', '').split()")
    text=once(text,"'FLASH_S5_OVERLAP=1', 'FLASH_S5_AUTO_ARM=1'", "'FLASH_S5_NATIVE_MM=1', 'FLASH_S5_OVERLAP=1', 'FLASH_S5_AUTO_ARM=1'")
    text=once(text,"    if '--speculative-config' in words or '--api-key' in words:",
        "    if a.arm == 'engine':\n        try:\n            limits=json.loads(argv('--limit-mm-per-prompt'))\n        except (TypeError,ValueError):\n            limits=None\n        if ('--language-model-only' in words or '--enable-mm-embeds' not in words\n                or '--skip-mm-profiling' not in words or limits != {'image':0,'video':0}\n                or argv('--mm-processor-cache-gb') != '0'):\n            diffs.append('vision-embedding-only-argv-differs')\n    if '--speculative-config' in words or '--api-key' in words:")
    text=once(text,"    probe = noauth_probe(a.url)",
        "    if a.arm == 'engine':\n        target='/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py'\n        model_mounts=[m for m in d.get('Mounts',[]) if m.get('Destination')==target]\n        if len(model_mounts)!=1 or model_mounts[0].get('Source')!=a.overlay+'/e3_sources/model.py':\n            diffs.append('native-model-mount-shadowed')\n    probe = noauth_probe(a.url)")
    readback.write_text(text)
    for name in ('serve_readback.py','serve_lane.py','serve_vram.py','serve_recorder.py','serve_smoke.py'):
        path=out/name;path.write_text(exception_details(path.read_text()))
    env=json.loads((out/'g0-container-env.json').read_text());env['FLASH_S5_NATIVE_MM']='1'
    (out/'vision-container-env.json').write_text(json.dumps(env,indent=2,sort_keys=True)+'\n')
    generated={name:sha(overlay/'e3'/(name+'.py')) for name in decode_probe.GENERATED_SOURCES}
    for name in compose.E3_MODULES:
        if (overlay/'e3'/(name+'.py')).read_text()!=compose.package((ROOT/'patches/s5_engine'/(name+'.py')).read_text()):
            raise ValueError('delivered module does not derive from reviewed source: '+name)
    evidence=json.loads((overlay/'e3/decode-probe-evidence.json').read_text());evidence['generated_source_shas']=generated
    (overlay/'e3/decode-probe-evidence.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    closure=json.loads((overlay/'e3'/native_readiness.CLOSURE).read_text())
    review['native_mm_20261116']['rendered_harness_pins']={'identity_declare.py':sha(identity)}
    closure.update(review=review,review_sha256=native_readiness.digest(review),generated_source_shas=generated)
    (overlay/'e3'/native_readiness.CLOSURE).write_text(json.dumps(closure,indent=2,sort_keys=True)+'\n')
    (h/'e3_builder/native_candidate_review.json').write_text(json.dumps(review,indent=2,sort_keys=True)+'\n')
    sums(overlay);sums(h)
    repin_params(out/'serve-params.env',dict(SERVE_TAG=TAG,BUNDLE_SUMS_SHA=sha(h/'SHA256SUMS'),
        OVERLAY_SUMS_SHA=sha(overlay/'SHA256SUMS'),S5_LAUNCHER_SHA=sha(launcher),
        PARAMS_STATUS='CANDIDATE:8095a93b096faf84',VRAM_NET_LIMIT_MIB='95'))
    delta=dict(schema='native-mm-serve-delta/1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        parent_packet_sha256=BASE_SUMS,scope='owner integration/fit candidate; NOT device qualified',
        overlay_name=NEW_OVERLAY,default_off_flag='FLASH_S5_NATIVE_MM',enabled_in_this_packet=True,
        vision_args=VISION_ARGS,mm_cache_tokens=8192,max_image_tokens=1024,max_images_per_request=1,
        max_model_len=200000,max_seqs=8,kv_bytes=2150000000,rowcache_slots=6144,
        physical_floor_mib=40,modeled_extra_pp0_mib=95,hardware_qualified=False,
        unchanged=['model','image','KV','max_model_len','MAX_SEQS','row cache','native libraries','SM split','MTP off','stock reference'],
        native_library_shas={n:sha(overlay/'e3/native'/n) for n in ('flash_s5_green.so','flash_s5_replay_observer.so')},
        generated_module_shas=generated,budget_source_sha256=native_mm_overlay.BUDGET_SHA,
        encoder_source_sha256=native_mm_overlay.ENCODER_SHA,ordinary_image_storage='individual async H2D; no concatenated GPU batch views',
        pending=['actual CUDA/TP/PP splice and lease checks','zero real tower allocation on every rank',
                 '8x200k image-mix physical fit and cancellation','known-label/27B image answers','encoder service clearance'])
    (out/'vision-delta.json').write_text(json.dumps(delta,indent=2,sort_keys=True)+'\n')
    native_readiness.verify_directory(overlay/'e3',decode_probe.GENERATED_SOURCES)
    for path in (start,launcher):subprocess.run(['bash','-n',str(path)],check=True)
    for path in (readback,overlay/'e3_sources/mm_budget.py',overlay/'e3_sources/mm_encoder.py'):ast.parse(path.read_text())
    root_names=sorted(p.relative_to(out).as_posix() for p in out.iterdir() if p.is_file() and p.name!='PACKET-SHA256SUMS')
    root_names += ['harness/SHA256SUMS']+[p.relative_to(out).as_posix() for p in (out/'tools').rglob('*') if p.is_file()]
    root_names += [p.relative_to(out).as_posix() for p in assets.iterdir() if p.is_file()]
    root_names += [p.relative_to(out).as_posix() for p in fit_assets.iterdir() if p.is_file()]
    (out/'PACKET-SHA256SUMS').write_text(''.join(sha(out/n)+'  '+n+'\n' for n in sorted(root_names)))
    print('VISION_SERVE_PACKET '+str(out)+' PACKET-SHA256SUMS '+sha(out/'PACKET-SHA256SUMS'))

if __name__=='__main__':
    try:
        p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True)
        p.add_argument('--budget-source',type=Path,required=True);p.add_argument('--encoder-source',type=Path,required=True)
        p.add_argument('--out',type=Path,required=True)
        a=p.parse_args();build(a.base,a.budget_source,a.encoder_source,a.out)
    except Exception as exc:
        frame=traceback.extract_tb(exc.__traceback__)[-1]
        print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}',file=sys.stderr)
        raise SystemExit(2)
