"""Explicit diagnostic shapes; ablated output is compared only with itself."""
def configure(spec,kind):
    if kind not in ('mixed','gdn','qsa'):raise ValueError('unknown layer kind')
    if kind!='mixed':spec['mix']=[{'gdn':'linear_attention','qsa':'full_attention'}[kind]]*spec['layers']


def apply(dec,pre,moe):
    import torch
    if moe not in ('on','off'):raise ValueError('unknown MoE mode')
    # Slice originally prepares row caches only at 3,7,...; a QSA-only slice
    # needs its own independently reset cache for every attention layer.
    for i,kind in enumerate(dec.spec['mix']):
        if kind=='full_attention' and i not in dec.rowcaches:
            dec.rowcaches[i]=tuple(t.clone() for t in dec.rowcache)
    if moe=='off':
        # Retain both TP collectives per layer and all HC/PLE/attention. Removes
        # routed+shared expert compute; not an equivalent language model.
        dec.moe=lambda x:torch.zeros_like(x)
        pre.moe=lambda x:torch.zeros_like(x)
