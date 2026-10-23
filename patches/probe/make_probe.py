#!/usr/bin/env python3
"""Build qsa.probe.py from the image's qsa.py: logs how much of each decode step's QSA
selection was already selected in the previous step / the previous 8 steps, per request.
Eager mode only (the hook is Python). Enabled by env QSA_OVERLAP_PROBE=<layer-name suffix>.
usage: make_probe.py qsa.image.py qsa.probe.py
"""
import sys

src = open(sys.argv[1]).read()
anchor = "        token_to_req = token_to_req[:num_tokens]\n"
assert src.count(anchor) == 1
hook = anchor + "        if _PROBE_LAYER and not use_prefill_config:\n            _overlap_probe(layer, logical_indices, token_to_req, num_tokens)\n"
probe = '''
_PROBE_LAYER = os.environ.get("QSA_OVERLAP_PROBE", "")
_probe_hist: dict = {}
_probe_acc = {"steps": 0, "n": 0, "prev": 0.0, "win8": 0.0, "sel": 0}


def _overlap_probe(layer, logical_indices, token_to_req, num_tokens):
    import collections
    import logging
    name = getattr(layer, "layer_name", "")
    if not name.endswith(_PROBE_LAYER):
        return
    idx = logical_indices[:num_tokens].cpu()
    req = token_to_req[:num_tokens].cpu().tolist()
    cur: dict = {}
    for t in range(num_tokens):
        valid = int(idx[t, -1])
        cur.setdefault(req[t], set()).update(idx[t, :valid].tolist())
    acc = _probe_acc
    for r, sel in cur.items():
        hist = _probe_hist.setdefault(r, collections.deque(maxlen=8))
        if hist and sel:
            acc["prev"] += len(sel & hist[-1]) / len(sel)
            acc["win8"] += len(sel & set().union(*hist)) / len(sel)
            acc["sel"] += len(sel)
            acc["n"] += 1
        hist.append(sel)
    acc["steps"] += 1
    if acc["steps"] % 100 == 0 and acc["n"]:
        logging.getLogger("vllm").warning(
            "QSA_OVERLAP layer=%s steps=%d samples=%d mean_selected=%.0f overlap_prev=%.3f overlap_last8=%.3f",
            name, acc["steps"], acc["n"], acc["sel"] / acc["n"], acc["prev"] / acc["n"], acc["win8"] / acc["n"])
'''
marker = "\n\nclass "
pos = src.index(marker)
out = src[:pos] + "\n" + probe + src[pos:]
out = out.replace(anchor, hook)
open(sys.argv[2], "w").write(out)
print("ok")
