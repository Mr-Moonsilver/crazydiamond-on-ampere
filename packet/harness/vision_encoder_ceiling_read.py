#!/usr/bin/env python3
"""Read the serial encoder sweep; no claim about a box-side consumer or production image limit."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from packet_client_contract import report_exception

def read(value,memory):
 sweep=value.get('ceiling',{});steps=sweep.get('steps',[]);ceiling=max((r['n'] for r in steps if r.get('status')=='PASS'),default=0);rows=[]
 for step in steps:
  r=dict(step);start=step.get('started_monotonic');end=step.get('finished_monotonic')
  samples=[x for x in memory if start is not None and end is not None and start<=x.get('monotonic',-1)<=end]
  samples += [sample for image in step.get('per_image',[]) for name in ('memory_before','memory_after') if (sample:=image.get(name)) is not None]
  r['measured_samples']=len(samples)
  r['owned_physical_peak_mib']=max((x['owned_mib'] for x in samples),default=0)
  r['free_minimum_mib']=min((x['free_mib'] for x in samples),default=None)
  rows.append(r)
 after=value.get('speech',{}).get('after_stop_asr',{});bound=sweep.get('bound_s',0)
 okay=(value.get('verdict')=='PASS' and sweep.get('complete') is True and [r['n'] for r in steps]==[1,2,4,6,8]
       and all(r.get('status')=='PASS' and r['asr'].get('correct') is True and r['asr'].get('elapsed_s',99)<=bound for r in steps)
       and value.get('speech',{}).get('asr_calls')==7
       and all(len(r.get('per_image',[]))==r['n'] and r.get('request_bytes',9*2**20)<=8*2**20 and all(0<i['shape'][0]<=1024 and 0<i['metrics']['peak_reserved_bytes']<=2048*2**20 and 0<i['metrics']['peak_allocated_bytes']<=2048*2**20 for i in r['per_image']) for r in rows)
       and all(r.get('measured_samples',0)>0 and 0<r.get('owned_physical_peak_mib',0)<=3072 for r in rows) and value.get('owned_gpu_released') is True
       and value.get('handoff',{}).get('state')=='OFF' and after.get('correct') is True and after.get('elapsed_s',999)<=bound
       and 0<value.get('physical_peak_mib',9999)<=3072 and not value.get('cleanup_errors'))
 return dict(verdict='PASS' if okay else 'HOLD',ceiling=ceiling,steps=rows,physical_peak_mib=value.get('physical_peak_mib'),memory_samples=value.get('memory_samples'),card=sweep.get('card'),post_asr=after,owned_gpu_released=value.get('owned_gpu_released'),reason=value.get('reason'),scope='serial encoder + real speech only; no box consumer tested; production <= actual G5 box ceiling')

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args(argv)
 memory_path=a.record.parent/'memory.jsonl';memory=[json.loads(x) for x in memory_path.read_text().splitlines()] if memory_path.exists() else []
 result=read(json.loads(a.record.read_text()),memory);a.out.write_text(json.dumps(result,indent=2)+'\n');print('VISION_ENCODER_CEILING_READ '+result['verdict']+' ceiling='+str(result['ceiling']));return 0 if result['verdict']=='PASS' else 2
if __name__=='__main__':
 try:raise SystemExit(main())
 except Exception as exc:report_exception(exc);print('VISION_ENCODER_CEILING_READ HOLD exception');raise SystemExit(2)
