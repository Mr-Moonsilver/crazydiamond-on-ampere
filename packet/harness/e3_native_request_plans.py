#!/usr/bin/env python3
"""Freeze the new row1 oracle request before measurement, not W1/W2 inputs.

One source-seeded text tokenization, no retries/selection from outputs. Slice
exactly 96*1024+8 tokens so the native bulk tail is seven tokens after the
stock final-token handback. This reaches beyond the existing24-page arena and
bounds the sparse tail union independently of token-content predictions. DMA
fragmentation refusals and unexercised routes remain INCONCLUSIVE in row1.
"""
import argparse
import json
import random
from pathlib import Path
import e3_decode_correctness as correctness
from e3_teacher_decode_client import Instrument
from e3_native_qualification_driver import freeze_row1_plan,canonical_sha,exception_details

EXACT_ROW1_TOKENS=96*1024+8


def build(template,url,model,run_id,post=None):
    if (type(template) is not dict or template.get('schema')!='s5-native-row1-request-plan/1'
            or template.get('workload',{}).get('streams')!=1
            or template['workload'].get('prompt',{}).get('tokens')!=EXACT_ROW1_TOKENS):
        raise ValueError('declared-exact-row1-template-differs')
    post=post or Instrument._post
    seed=template['workload']['prompt']['seed']
    # agent_replay.text is deliberately an estimate. Overproduce once, tokenize
    # once, then retain a fixed prefix; no second call if it is insufficient.
    text=correctness.AR.text(random.Random(seed*1000),EXACT_ROW1_TOKENS*2)
    result=post(url,'/tokenize',{'model':model,'prompt':text,'add_special_tokens':True})
    ids=result.get('tokens') if type(result) is dict else None
    if (type(ids) is not list or len(ids)<EXACT_ROW1_TOKENS
            or any(type(x) is not int or x<0 for x in ids)):
        raise ValueError('row1-single-tokenization-insufficient')
    plan=dict(template,prompts=[ids[:EXACT_ROW1_TOKENS]])
    frozen=freeze_row1_plan(plan,run_id=run_id,model=model)
    return plan,{'schema':'s5-native-row1-plan-freeze/1','run_id':run_id,
        'template_sha256':canonical_sha(template),'plan_sha256':frozen['plan_sha256'],
        'request_sha256':frozen['request_sha256'],'actual_prompt_tokens':EXACT_ROW1_TOKENS,
        'native_bulk_tail_tokens':(EXACT_ROW1_TOKENS-1)%1024,
        'tokenization_calls':1,'no_retry':True,'settings_unchanged':True,
        'scope':'new row1 diagnostic only; teacher, placement, W1 and W2 bytes/settings unchanged'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('url','model','run-id','template','out','receipt'):p.add_argument('--'+name,required=True)
    a=p.parse_args(argv)
    if Path(a.out).exists() or Path(a.receipt).exists():return 2
    try:
        plan,receipt=build(json.loads(Path(a.template).read_text()),a.url,a.model,a.run_id)
    except Exception as exc:
        receipt={'verdict':'INCONCLUSIVE','error':exception_details(exc)}
        with open(a.receipt,'x') as handle:json.dump(receipt,handle,sort_keys=True)
        return 2
    with open(a.out,'x') as handle:json.dump(plan,handle,sort_keys=True)
    with open(a.receipt,'x') as handle:json.dump(receipt,handle,sort_keys=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
