#!/usr/bin/env python3
"""New overlay from v16g with opt-in scheduler cost budget; no deployment."""
import argparse,ast,hashlib,json,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE_SHA='4d797e5ab6c5abaa06229dd88035ceede7562f1dbe054bdf7ad59be659c04c87'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sub(s,a,b):
    assert s.count(a)==1,(a[:60],s.count(a));return s.replace(a,b)
def generate(src,model):
    assert hashlib.sha256(src.encode()).hexdigest()==BASE_SHA,'unreviewed scheduler'
    src=sub(src,'class Scheduler(SchedulerInterface):',
        '_FLASH_COST_MODEL = '+repr(model)+'\n\n'+(HERE/'controller.py').read_text()+'\n\nclass Scheduler(SchedulerInterface):')
    anchor='        self._sf_max = int(os.environ.get("FLASH_SHORT_FIRST", "0"))'
    src=sub(src,anchor,'''        self._flash_cost = None
        if os.environ.get("FLASH_COST_BUDGET", "0") == "1":
            if not (self.pd_alternate and self.parallel_config.pipeline_parallel_size == 2
                    and self.parallel_config.tensor_parallel_size == 2 and self.num_spec_tokens == 0):
                raise ValueError("FLASH_COST_BUDGET requires PP2 TP2 MTP0 with alternation")
            self._flash_cost = _FlashCostBudget(_FLASH_COST_MODEL,
                float(os.environ.get("FLASH_COST_DEADLINE_MS", "27")),
                os.environ.get("FLASH_COST_X8", "1") == "1")
            logger.warning("FLASH_COST uses a client-gap tail proxy, NOT calibrated GPU-step deadlines")
        self._flash_cost_history = {}
        self._flash_cost_active = False
        self._flash_cost_calls = 0
'''+anchor)
    anchor='    def _mamba_block_aligned_split(\n'
    src=sub(src,anchor,'''    def _flash_cost_chunk(self, request, history, proposal, scheduled):
        if self._flash_cost is None or not self._flash_cost_active or proposal <= 0:
            return proposal
        # Scheduled-token dict is authoritative: preempted requests were popped
        # from it already, so their reservations cannot leak into this budget.
        spent = sum(self._flash_cost.predict(self._flash_cost_history[k], n)["ms"]
                    for k,n in scheduled.items() if k in self._flash_cost_history)
        decision = self._flash_cost.choose(history, proposal, spent)
        self._flash_cost_history[request.request_id] = history
        self._flash_cost_calls += 1
        if self._flash_cost_calls <= 4 or self._flash_cost_calls % 200 == 0:
            logger.info("FLASH_COST step=%d history=%d proposal=%d decision=%s counts=%s",
                        self.current_step, history, proposal, decision, self._flash_cost.counts)
        return decision["chunk"]

'''+anchor)
    anchor='        num_scheduled_tokens: dict[str, int] = {}\n'
    src=sub(src,anchor,anchor+'''        if self._flash_cost is not None:
            self._flash_cost_history = {}
            self._flash_cost_active = any(not r.is_prefill_chunk for r in self.running)
''')
    anchor='\n            # Apply Mamba alignment before encoder caps.\n'
    src=sub(src,anchor,'''
            if request.is_prefill_chunk:
                num_new_tokens = self._flash_cost_chunk(request, request.num_computed_tokens,
                                                        num_new_tokens, num_scheduled_tokens)
'''+anchor)
    anchor='\n                    # Apply Mamba alignment before encoder caps.\n'
    src=sub(src,anchor,'''
                    if request.num_tokens - num_computed_tokens > 1:
                        num_new_tokens = self._flash_cost_chunk(request, num_computed_tokens,
                                                                num_new_tokens, num_scheduled_tokens)
'''+anchor)
    ast.parse(src);return src

def build(base,out):
    assert not out.exists(),'new output required'
    for folder in (base,base/'core'):
        for line in (folder/'SHA256SUMS').read_text().splitlines():
            h,n=line.split(maxsplit=1);p=folder/n
            assert p.resolve().is_relative_to(folder.resolve()) and sha(p)==h
    src=generate((base/'core/scheduler.py').read_text(),json.loads((HERE/'model.json').read_text()))
    shutil.copytree(base,out);(out/'core/scheduler.py').write_text(src)
    lines=[]
    for line in (out/'core/SHA256SUMS').read_text().splitlines():
        _,n=line.split(maxsplit=1);lines.append(sha(out/'core'/n)+'  '+n)
    (out/'core/SHA256SUMS').write_text('\n'.join(lines)+'\n')
    return sha(out/'core/scheduler.py')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path);p.add_argument('out',type=Path);a=p.parse_args();print(build(a.base,a.out))
