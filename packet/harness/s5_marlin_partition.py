"""CPU transcription of the pinned Marlin DP/Stream-K tile partition.

Not a GEMM emulator or GPU result. Supply actual launch dimensions to attribute
an observed token. Default/example dimensions are explicitly hypothetical.
Source: dc36fcce9 marlin_moe_wna16/marlin_template.h:355-385,535-553,599-620.
"""
import argparse
from dataclasses import asdict, dataclass
import json


def ceildiv(a, b): return (a+b-1)//b


@dataclass(frozen=True)
class Layout:
    padded_tokens: int
    moe_block: int
    n: int
    k: int
    thread_n: int
    thread_k: int
    grid: int
    group_size: int = 128

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in
               (self.padded_tokens,self.moe_block,self.n,self.k,self.thread_n,self.thread_k,self.grid)):
            raise ValueError('positive integer dimensions required')
        if self.measure_invalid(): raise ValueError('unaligned dimensions')

    def measure_invalid(self):
        return (self.padded_tokens%self.moe_block or self.n%self.thread_n or
                self.k%self.thread_k or self.thread_k%16 or self.thread_n%16 or
                self.group_size != -1 and (self.group_size < 16 or self.group_size%16 or self.k%self.group_size))

    @property
    def nt(self): return self.n//self.thread_n
    @property
    def kt(self): return self.k//self.thread_k
    @property
    def total(self): return self.padded_tokens//self.moe_block*self.nt
    @property
    def tail(self):
        if self.total <= self.grid: return self.total
        n = self.total%self.grid
        return n+self.grid if n*3 <= self.grid else n
    @property
    def prefix(self): return self.total-self.tail
    @property
    def iters(self):
        n = ceildiv(self.kt*self.tail,self.grid)
        if self.group_size != -1 and self.group_size >= self.thread_k:
            factor = self.group_size//self.thread_k
            n = factor*ceildiv(n,factor)
        return n

    def slices(self, m_block, n_tile):
        if not 0 <= m_block < self.padded_tokens//self.moe_block or not 0 <= n_tile < self.nt:
            raise ValueError('tile outside padded launch')
        tile = m_block*self.nt+n_tile
        if tile < self.prefix:
            return dict(path='DP',tile=tile,partials=[dict(cta=tile%self.grid,
                        slice_idx=0,k_begin=0,k_end=self.k)])
        relative=tile-self.prefix; lo=relative*self.kt; hi=lo+self.kt
        parts=[]
        for cta in range(lo//self.iters,min(ceildiv(hi,self.iters),self.grid)):
            start=max(lo,cta*self.iters); end=min(hi,(cta+1)*self.iters)
            if end<=start:continue
            # Same slice_count/slice_idx calculation as init_part2_slice.
            col_first=self.iters*ceildiv(lo,self.iters)
            col_off=col_first-lo
            count=1; idx=0
            if col_first<=hi:
                count=ceildiv(self.kt-col_off,self.iters)+(col_off>0)
                delta=self.iters*cta-col_first
                if delta<0 or col_off==0 and delta==0:idx=count-1
                else:idx=count-1-delta//self.iters-(col_off>0)
            parts.append(dict(cta=cta,slice_idx=idx,k_begin=(start-lo)*self.thread_k,
                              k_end=(end-lo)*self.thread_k))
        return dict(path='Stream-K',tile=tile,partials=sorted(parts,key=lambda p:p['slice_idx']))

    def token(self, sorted_position):
        if not 0<=sorted_position<self.padded_tokens:raise ValueError('position outside sorted tokens')
        block,row=divmod(sorted_position,self.moe_block)
        return dict(sorted_position=sorted_position,m_block=block,row_in_tile=row,
                    columns=[self.slices(block,n) for n in range(self.nt)])


def example():
    # Valid shape option for the slice's second GEMM, NOT its measured launch:
    # 257 padded M blocks, 2 CTAs/SM on 82 SM, thread N=128/K=64.
    config=Layout(257*32,32,2560,640,128,64,164)
    return dict(scope='hypothetical valid tile geometry; not row710 attribution',
                config=asdict(config),dp_tiles=config.prefix,stream_k_tiles=config.tail,
                stripe_k_tiles=config.iters,
                before=config.slices(253,11),after=config.slices(254,11))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',help='JSON file containing measured Layout fields')
    p.add_argument('--position',type=int)
    a=p.parse_args()
    if a.config:
        with open(a.config) as f:layout=Layout(**json.load(f))
        if a.position is None:p.error('--position required with --config')
        result=layout.token(a.position)
    else:result=example()
    print(json.dumps(result,indent=2))
