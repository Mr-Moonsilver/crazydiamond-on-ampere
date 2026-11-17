#!/usr/bin/env python3
"""Prospective G5/G6/A3 evidence gates. A missing real-caller adapter is HOLD."""
import argparse,hashlib,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from packet_client_contract import report_exception
CALLERS=('<caller-a>.text','<caller-a>.image','<caller-c>50.text','<caller-c>50.image','<caller-c>50.summary','<caller-c>80.text','<caller-c>80.image','<caller-c>80.summary','<caller-b>.dev1','<caller-d>')

REQUIRED_CALLERS=('<caller-a>.text','<caller-a>.image','<caller-c>50.summary','<caller-c>80.summary','<caller-b>.dev1','<caller-d>')

def verdict(v,r,**kw):return dict(verdict=v,reason=r,**kw)
def norm(s):return ' '.join(s.strip().lower().split())
def g5_read(value,reference,speech):
    if value.get('verdict')!='COMPLETE' or reference.get('verdict')!='COMPLETE' or len(value.get('cases',[]))!=4 or len(reference.get('cases',[]))!=4:return verdict('HOLD','incomplete-four-label-capture')
    refs={x['id']:x for x in reference['cases']};rows=[]
    if len(refs)!=4 or len({x['id'] for x in value['cases']})!=4:return verdict('HOLD','duplicate-image-case')
    for case in value['cases']:
        ref=refs.get(case['id'])
        if ref is None or any(case[k]!=ref[k] for k in ('image_sha256','question','expected')):return verdict('HOLD','unmatched-27B-reference')
        okay=norm(case['answer'])==norm(case['expected'])==norm(ref['answer'])
        rows.append(dict(id=case['id'],match=okay,reference_wall_s=ref['wall_s'],production_wall_s=case['wall_s'],reference_output_tokens=ref['output_tokens'],production_output_tokens=case['output_tokens']))
    if not all(r['match'] for r in rows):return verdict('FAIL','known-label-or-reference-MISMATCH',cases=rows)
    start=min(c['started_wall_ns'] for c in value['cases']);end=max(c['finished_wall_ns'] for c in value['cases'])
    pre=speech.get('speech',{}).get('pre_asr',{});history=speech.get('speech',{}).get('asr_history',[])
    # ASR after G5 is the scheduled15min service check, not another immediate probe.
    checks=[pre,*history]
    def valid(c):return c.get('correct') is True and 0<=c.get('elapsed_s',999)<=10
    good=[c for c in checks if valid(c)]
    if (not any(c.get('observed_wall_ns',start+1)<=start for c in good)
            or not valid(speech.get('speech',{}).get('post_load_asr',{}))
            or not speech.get('speech_overlap_images')
            or not any(c.get('observed_wall_ns',0)>=end for c in good)):
        return verdict('HOLD','speech-pre-during-post-evidence-missing',cases=rows)
    if (not 0<speech.get('physical_peak_mib',9999)<=3072
            or speech.get('verdict') not in ('PASS','RUNNING')
            or speech.get('speech',{}).get('last_health',{}).get('observed_wall_ns',0)<end):
        return verdict('HOLD','encoder-clearance-health-or-memory-not-proven',cases=rows)
    return verdict('PASS','four-labels-reference-and-speech',cases=rows)

def g6_read(pre,post):
    if pre.get('phase')!='27B' or post.get('phase')!='Flash-Next':return verdict('HOLD','caller-phase-differs')
    rows=[]
    for caller in REQUIRED_CALLERS:
        a=pre.get('calls',{}).get(caller);b=post.get('calls',{}).get(caller)
        if a is None or b is None:return verdict('HOLD','missing-real-caller:'+caller,calls=rows)
        for c in (a,b):
            if c.get('status')=='FAIL' or c.get('timed_out') is True:return verdict('FAIL','real-caller-failed:'+caller,calls=rows)
            if (c.get('status')!='PASS' or c.get('application_path') is not True
                    or type(c.get('output_tokens')) is not int or c['output_tokens']<0
                    or not isinstance(c.get('wall_s'),(int,float)) or not math.isfinite(c['wall_s']) or c['wall_s']<0
                    or not c.get('receipt_path') or not c.get('fixture_sha256') or not c.get('caller_timeout_contract')):
                return verdict('HOLD','unproven-real-caller:'+caller,calls=rows)
        if a['fixture_sha256']!=b['fixture_sha256']:return verdict('HOLD','caller-fixture-differs:'+caller,calls=rows)
        rows.append(dict(caller=caller,reference_wall_s=a['wall_s'],production_wall_s=b['wall_s'],reference_output_tokens=a['output_tokens'],production_output_tokens=b['output_tokens'],reference_receipt=a['receipt_path'],production_receipt=b['receipt_path']))
    supplementary={name:dict(reference=pre.get('calls',{}).get(name),production=post.get('calls',{}).get(name),scope='REPORT ONLY caller-config probe') for name in CALLERS if name not in REQUIRED_CALLERS}
    return verdict('PASS','six-real-callers-paired; <caller-c> text proven by summaries; <caller-c> image N/A by design',calls=rows,<caller-c>_text_proof='same summary real-path call; no separate chat',<caller-c>_image='N/A BY DESIGN; G5/<caller-a> gate images',supplementary=supplementary)

def a3_read(out,quality,selection):
    ids=json.loads(selection.read_text())['ids']
    def load(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    values=load(out/'scores/main.jsonl');raw=load(out/'results/main.jsonl');baseline=load(quality/'run/B/scores/main.jsonl')
    chosen=[r for r in values if r['id'] in ids];original=[r for r in baseline if r['id'] in ids]
    if (len(ids)!=100 or len(set(ids))!=100 or len(chosen)!=100 or len(raw)!=100 or len(original)!=100
            or any({r['id'] for r in rs}!=set(ids) for rs in (chosen,raw,original))):
        return verdict('HOLD','main100-missing-or-duplicate-id')
    if any(r.get('error') or r.get('skipped') for r in raw+chosen):return verdict('FAIL','main100-request-or-score-errors')
    if any(type(r.get('score')) not in (int,float) or not 0<=r['score']<=1 for r in chosen+original):return verdict('HOLD','invalid-score')
    before=sum(r['score'] for r in original)/100;after=sum(r['score'] for r in chosen)/100
    return verdict('PASS' if after-before>=-.03-1e-12 else 'FAIL','paired-main100-score',reference=before,production=after,delta_pp=100*(after-before),errors=0,n=100)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('gate',choices=('g5','g6','a3'));p.add_argument('--record',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--speech',type=Path);p.add_argument('--quality-packet',type=Path);p.add_argument('--selection',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args(argv)
    if a.gate=='g5':result=g5_read(json.loads(a.record.read_text()),json.loads(a.reference.read_text()),json.loads(a.speech.read_text()))
    elif a.gate=='g6':result=g6_read(json.loads(a.reference.read_text()),json.loads(a.record.read_text()))
    else:result=a3_read(a.record,a.quality_packet,a.selection)
    if a.out:a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True));print('W27_'+a.gate.upper()+' '+result['verdict']+' '+result['reason']);return 0 if result['verdict']=='PASS' else 2
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:report_exception(exc);print('W27_GATE HOLD incomplete-or-invalid-record');raise SystemExit(2)
