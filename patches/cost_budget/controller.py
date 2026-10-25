# Pure stdlib runtime, embedded with a pinned model in scheduler.py by generator.
class _FlashCostBudget:
    def __init__(self, model, deadline_ms, x8=True):
        import math
        if not math.isfinite(deadline_ms) or deadline_ms<=0: raise ValueError('invalid decode deadline')
        self.model=model;self.deadline=deadline_ms;self.x8=x8
        self.counts=dict(feasible=0,infeasible=0,out_of_domain=0)

    def predict(self, history, chunk, coverage=None):
        import math
        if history<0 or chunk<=0 or (coverage is not None and not 0<=coverage<=1):raise ValueError('invalid cost inputs')
        pages=math.ceil((history+chunk)/self.model['page_size'])
        rank_cost={}
        for rank,cells in self.model['copy_cells'].items():
            if not self.x8 and rank=='1:0':continue
            cell=next((c for c in cells if pages<=c['pages_le']),cells[-1])
            cov=cell['coverage_upper'] if coverage is None else coverage
            rank_cost[rank]=self.model['qsa_layers_per_rank']*pages*cov*cell['copy_ms_per_page']
        knots=self.model['other_ms_knots']
        # Ceiling piecewise choice: never extrapolate a faster tiny-chunk path
        # from a throughput measurement at a large chunk.
        other=next((v for c,v in knots if chunk<=c),knots[-1][1]*chunk/knots[-1][0])
        return {'ms':max(rank_cost.values())+other,'copy_ms':max(rank_cost.values()),'ranks_copy_ms':rank_cost,
                'tail_kind':self.model['tail_kind']}

    def choose(self,history,proposal,spent=0.,has_decode=True):
        if not has_decode:return {'chunk':proposal,'status':'no_decode','predicted_ms':None}
        budget=self.deadline-self.model['decode_gap99_ms']-spent
        candidates=sorted(set([min(proposal,c) for c,_ in self.model['other_ms_knots'] if c<=proposal]+[proposal]))
        if history+proposal>self.model['max_fit_history']:
            self.counts['out_of_domain']+=1
            return {'chunk':proposal,'status':'out_of_domain','predicted_ms':self.predict(history,proposal)['ms'],'budget_ms':budget}
        estimates=[(n,self.predict(history,n)['ms']) for n in candidates]
        fitting=[(n,t) for n,t in estimates if t<=budget]
        if fitting:
            n,t=fitting[-1];status='feasible'
        else:
            # Report the impossible budget; keep efficient service. Never turn
            # an infeasible deadline into zero tokens or repeated tiny chunks.
            n,t=proposal,estimates[-1][1];status='infeasible'
        self.counts[status]+=1
        return {'chunk':n,'status':status,'predicted_ms':t,'min_predicted_ms':estimates[0][1],'budget_ms':budget,
                'calibrated_gpu_step_tail':False}
