#!/usr/bin/env python3
"""For each big miss (>20k computed, cached small) find the previous request of the same session and report WHERE
they diverge (system block index / tools / message index + block type), sizes only, never text.
usage: miss_diff.py capture.pcap  (imports the rows of cache_by_request.py from the same directory)"""
import json, sys, time
sys.argv = [sys.argv[0], sys.argv[1]]
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import cache_by_request as cbr
rows = cbr.rows
last = {}
for t, sess, nm, tl, it, cr, b in rows:
    if it is not None and cr is not None and it > 20000 and sess in last:
        pt, pb = last[sess]
        out = [f"{time.strftime('%H:%M:%S', time.gmtime(t))} session {sess} gap {t - pt:.0f}s msgs {len(pb['messages'])}->{len(b['messages'])} computed {it} cached {cr}"]
        sa, sb = pb.get("system"), b.get("system")
        if json.dumps(sa) != json.dumps(sb):
            if isinstance(sa, list) and isinstance(sb, list):
                for i, (x, y) in enumerate(zip(sa, sb)):
                    if x != y:
                        xa, ya = x.get("text", ""), y.get("text", "")
                        k = next((j for j in range(min(len(xa), len(ya))) if xa[j] != ya[j]), min(len(xa), len(ya)))
                        out.append(f"  system[{i}] differs at char {k} of {len(xa)}/{len(ya)}")
                        break
            else:
                out.append("  system differs (type)")
        if json.dumps(pb.get("tools")) != json.dumps(b.get("tools")):
            out.append(f"  tools differ: {[x.get('name') for x in pb.get('tools', [])]} -> {[x.get('name') for x in b.get('tools', [])]}")
        for i, (x, y) in enumerate(zip(pb["messages"], b["messages"])):
            if x != y:
                cx, cy = x.get("content"), y.get("content")
                desc = lambda c: (f"str{len(c)}" if isinstance(c, str) else [blk.get("type") for blk in c][:6])
                out.append(f"  first differing message {i}: role {x.get('role')}->{y.get('role')} {desc(cx)} -> {desc(cy)}")
                break
        else:
            out.append("  messages: earlier request is a strict prefix")
        for k in ("model", "thinking", "max_tokens", "temperature", "tool_choice"):
            if pb.get(k) != b.get(k):
                out.append(f"  {k}: {pb.get(k)} -> {b.get(k)}")
        print("\n".join(out))
    last[sess] = (t, b)
