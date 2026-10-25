"""Compare completed concurrent trajectories with exact-count solo controls.

All snapshots/diff readbacks happen AFTER overlap. No tolerance silently passes.
"""
def verify(dec,pre,graph,sd,sp,counts,progress,solo_runs=4):
    import torch
    import s5_segment_gpu as b
    from s5_terminal_mode import NullTrace
    torch.cuda.synchronize();got={'decode':b.snapshot(dec),'prefill':b.snapshot(pre)}
    result={}
    for name,lane,stream,n in (('decode',dec,sd,counts[0]),('prefill',pre,sp,counts[1])):
        solo=[];state_ok=True;finite=True;ref=None;comparisons=[]
        for trial in range(solo_runs):
            progress.update('terminal-solo-'+name,trial=trial,steps=n)
            lane.reset();torch.cuda.synchronize()
            with torch.cuda.stream(stream):
                for _ in range(n):
                    if name=='decode':graph.replay()
                    else:lane.forward(NullTrace())
            stream.synchronize();snap=b.snapshot(lane)
            if ref is None:ref=snap
            state_ok=state_ok and b.state_equal(snap[1:],ref[1:])
            finite=finite and all(bool(torch.isfinite(x).all()) for x in snap)
            equal=torch.equal(got[name][0],snap[0])
            comparisons.append(dict(solo=trial,output_bitwise=equal,
                diff=None if equal else b.ulp_close(got[name][0],snap[0])[1]))
            if not any(torch.equal(snap[0],x) for x in solo):solo.append(snap[0])
        state_vs_solo=b.state_equal(got[name][1:],ref[1:])
        finite=finite and all(bool(torch.isfinite(x).all()) for x in got[name])
        result[name]=dict(steps=n,solo_runs=solo_runs,distinct_solo_outputs=len(solo),
            bitwise_first=comparisons[0]['output_bitwise'],output_matches_solo=any(x['output_bitwise'] for x in comparisons),
            solo_states_identical=state_ok,concurrent_states_match=state_vs_solo,finite=finite,comparisons=comparisons)
    result['passed']=all(x['output_matches_solo'] and x['solo_states_identical'] and x['concurrent_states_match'] and x['finite'] for x in result.values())
    return result
