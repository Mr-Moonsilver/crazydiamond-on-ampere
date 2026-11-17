#!/usr/bin/env python3
"""Pinned CPU hybrid-cache/checkpoint proof; device state bytes are a declared shadow leaf."""
import argparse,json,sys
from pathlib import Path

def main():
    import torch
    import test_append_no_block_stop_stack as rig
    from vllm.multimodal.inputs import MultiModalFeatureSpec,PlaceholderRange
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    assert rig.STACK,'pinned vLLM CPU stack required'
    stack=rig.Stack('1',blocks=2048);stack.manager.max_model_len=200000
    stack.manager.max_in_flight_tokens=200000
    prompts=[]
    def request(tokens,i,owner=None):
        owner=stack if owner is None else owner
        owner.seq+=1
        feature=MultiModalFeatureSpec(data=None,modality='image',identifier='image-owner-'+str(i),mm_position=PlaceholderRange(offset=4,length=1024))
        return rig.Request('r'+str(owner.seq),list(tokens),rig.SamplingParams(max_tokens=8192),None,mm_features=[feature],block_hasher=owner.hasher)
    # Same placeholder token IDs, different MM identities: hash separation is real.
    for i in range(7):
        tokens=[50,51,52,53]+[248056]*1024+list(range(1028,190320))
        prompts.append(tokens)
        hit,pieces=stack.native_turn(request(tokens[:190144],i),reply=1)
        assert hit==0
    hits=[]
    for i,tokens in enumerate(prompts):
        stack.manager.new_step_starts()
        req=request(tokens,i)
        _,hit,junction=stack.manager.get_computed_blocks(req)
        hits.append(dict(member=i,hit=hit,junction=junction))
    assert all(x['hit']==190080 for x in hits),hits
    for i,tokens in enumerate(prompts):
        req=request(tokens,i)
        blocks,hit,junction,lowered=rig.sh.residue_hit(stack.host,req,64,190256)
        assert hit==190080 and not lowered,(hit,junction,lowered)
        actual,pieces=stack.native_turn(req,reply=1)
        assert actual==190080 and pieces[0][0]==190080
    # A prime at the full prompt's boundary is too late: admission lowers its hit.
    fresh=rig.Stack('1',blocks=256);fresh.manager.max_model_len=200000;fresh.manager.max_in_flight_tokens=200000
    req=fresh.request(prompts[0]);fresh.native_turn(req,reply=1)
    again=fresh.request(prompts[0]);fresh.manager.new_step_starts()
    _,late,_,lowered=rig.sh.residue_hit(fresh.host,again,64,190256)
    assert lowered and late<190256
    assert not fresh.violations,fresh.violations[:8]
    small=rig.Stack('1',blocks=128)
    small.native_turn(request(prompts[0][:4096],0,small),reply=128)
    small.manager.new_step_starts()
    _,same_hit,_=small.manager.get_computed_blocks(request(prompts[0][:4096],0,small))
    _,different_hit,_=small.manager.get_computed_blocks(request(prompts[0][:4096],1,small))
    assert same_hit==4048 and different_hit==0,(same_hit,different_hit)
    assert not small.violations,small.violations[:8]
    assert not stack.violations,stack.violations[:8]
    assert not torch.cuda.is_initialized()
    args.out.write_text(json.dumps(dict(verdict='PASS',members=hits,FULL_calibration=dict(same_hit=same_hit,different_hit=different_hit),shadow_violations=stack.violations,cuda_initialized=False,
        scope='real pinned hybrid cache/hash/CoW; CPU state-position shadow, no device recurrence'),indent=2)+'\n')
    print('VISION_FIT_CHECKPOINT PASS')
if __name__=='__main__':
    from packet_client_contract import report_exception
    try:main()
    except Exception as exc:report_exception(exc);raise SystemExit(2)
