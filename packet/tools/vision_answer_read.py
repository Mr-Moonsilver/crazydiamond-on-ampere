#!/usr/bin/env python3
"""Read known-label answers; optional same-image27B comparison, never a box-fit certificate."""
import argparse,json,sys,traceback
from pathlib import Path


def read(value, reference=None):
    if value.get('verdict')!='COMPLETE' or not value.get('cases'):
        return dict(verdict='HOLD',reason='incomplete-answer-probe',cases=[])
    comparisons=[]
    refs={} if reference is None else {x['id']:x for x in reference.get('cases',[])}
    for case in value['cases']:
        normal=lambda text:' '.join(text.strip().lower().split())
        row=dict(id=case['id'],answer=case['answer'],expected=case['expected'],
                 label_match=normal(case['answer'])==normal(case['expected']))
        if reference is not None:
            ref=refs.get(case['id'])
            if (reference.get('verdict')!='COMPLETE' or ref is None
                    or ref['image_sha256']!=case['image_sha256'] or ref['question']!=case['question']):
                return dict(verdict='HOLD',reason='unmatched-reference',cases=comparisons)
            row.update(reference_answer=ref['answer'],same_answer=normal(ref['answer'])==normal(case['answer']))
        comparisons.append(row)
    return dict(verdict='MATCH' if all(r['label_match'] for r in comparisons) else 'MISMATCH',
        scope='known-label answers only; native receipt/TPPP/MM-fit/lifetimes remain separate gates',cases=comparisons)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--record',type=Path,required=True)
    p.add_argument('--reference-record',type=Path);a=p.parse_args()
    result=read(json.loads(a.record.read_text()),None if a.reference_record is None else json.loads(a.reference_record.read_text()))
    print(json.dumps(result,sort_keys=True))
    print('VISION_ANSWER_READ '+result['verdict'])
    return 0 if result['verdict']=='MATCH' else 2

if __name__=='__main__':
    try:code=main()
    except (Exception,SystemExit) as exc:
        if not isinstance(exc,SystemExit) or exc.code:
            frame=traceback.extract_tb(exc.__traceback__)[-1]
            print(f'PACKET_CLIENT_ERROR {type(exc).__name__}: {exc} at {frame.filename}:{frame.lineno}',file=sys.stderr)
        if isinstance(exc,SystemExit):raise
        print('VISION_ANSWER_READ HOLD '+type(exc).__name__)
        raise SystemExit(2)
    raise SystemExit(code)
