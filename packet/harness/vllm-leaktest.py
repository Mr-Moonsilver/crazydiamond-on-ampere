# Cross-session prefix-cache leak test on the local vLLM (via <llm-gateway>).
# N sessions share a long common prefix (like Claude Code's system prompt), each carries a unique
# Agent-like mode (matches the 10-23 incidents: ~7.5k shared prefix, short own task, 2nd request reuses the cache):
#   RULES=45 FILL=60 SEED=$RANDOM GAP=2 python3 vllm-leaktest.py 10 3
# secret + unique filler, then 3 follow-up turns (the 3rd request is where both foreign-task cases hit).
# Each turn asks for the session's own secret; any answer naming another session's secret = leak.
import json, random, string, sys, concurrent.futures as cf, urllib.request, time
URL="http://<gateway-host>:8000/v1/chat/completions"; MODEL="qwen38-w8a8-tp4"
N=int(sys.argv[1]) if len(sys.argv)>1 else 8; ROUNDS=int(sys.argv[2]) if len(sys.argv)>2 else 3
rnd=random.Random(int(__import__("os").environ.get("SEED","7")))
words=[ ''.join(rnd.choice(string.ascii_lowercase) for _ in range(rnd.randint(3,9))) for _ in range(4000)]
common="You are a careful assistant for a software project. Rules follow.\n" + "\n".join(
    f"Rule {i}: " + " ".join(rnd.choice(words) for _ in range(40)) for i in range(int(__import__("os").environ.get("RULES","120"))))   # ~20k tokens shared
secrets=[f"{rnd.choice(['AMBER','COBALT','JUNIPER','OSPREY','TUNDRA','VIOLET','MAGPIE','QUARTZ','SIENNA','FALCON','HAZEL','RIVET'])}-{rnd.randint(1000,9999)}" for _ in range(N)]
def filler(i):
    r=random.Random(1000+i); return " ".join(r.choice(words) for _ in range(int(__import__("os").environ.get("FILL","1500"))))   # ~6k tokens unique
def call(msgs):
    body=json.dumps({"model":MODEL,"messages":msgs,"max_tokens":40,"temperature":0,
                     "chat_template_kwargs":{"enable_thinking":False}}).encode()
    req=urllib.request.Request(URL,body,{"Content-Type":"application/json","Authorization":"Bearer local"})
    d=json.load(urllib.request.urlopen(req,timeout=240)); return d["choices"][0]["message"]["content"].strip(), d.get("usage",{})
def session(i):
    msgs=[{"role":"system","content":common},
          {"role":"user","content":f"Your private session code is {secrets[i]}. Background notes (ignore): {filler(i)}\n\nReply with only your private session code."}]
    out=[]
    for k in range(ROUNDS):
        t0=time.time()
        try: a,u=call(msgs)
        except Exception as e: a,u=f"ERR {type(e).__name__} after {time.time()-t0:.0f}s",{}
        out.append(a); print(f"  s{i} r{k+1} {time.time()-t0:.0f}s cached={((u or {}).get('prompt_tokens_details') or {}).get('cached_tokens')} prompt={(u or {}).get('prompt_tokens')}", flush=True)
        msgs+= [{"role":"assistant","content":a},
                {"role":"user","content":f"Follow-up {k+1}: notes {filler(i*31+k)[:4000]}\n\nReply with only your private session code."}]
        time.sleep(random.uniform(0,float(__import__("os").environ.get("GAP","20"))))   # idle gaps so contexts get evicted and re-used
    return i,out
bad=0; t=time.time()
with cf.ThreadPoolExecutor(N) as ex:
    for i,out in ex.map(session, range(N)):
        for k,a in enumerate(out):
            own=secrets[i] in a; foreign=[s for j,s in enumerate(secrets) if j!=i and s in a]
            if foreign or not own: bad+=1
            if a.startswith('ERR'): print(f's{i} r{k+1} ERR'); continue
            print(f"s{i} r{k+1} {'OK ' if own and not foreign else 'LEAK' if foreign else 'MISS'} {a[:60]!r}")
print(f"sessions {N} rounds {ROUNDS} bad {bad} in {time.time()-t:.0f}s")
