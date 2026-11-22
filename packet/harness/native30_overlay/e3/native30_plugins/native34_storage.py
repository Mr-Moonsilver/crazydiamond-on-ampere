"""Compact host metadata for storage aliases and bounded allocator lifetimes.

No Tensor/Storage owners are retained. No device values are read. An allocator
incarnation is certified only within a non-overflowing history window.
"""
import os

MAX_HISTORY = 8192
MAX_STORAGES = 4096


def tensor_view(tensor):
    storage=tensor.untyped_storage()
    base=int(storage.data_ptr());extent=int(storage.nbytes())
    pointer=int(tensor.data_ptr());item=int(tensor.element_size())
    shape=tuple(int(n) for n in tensor.shape)
    stride=tuple(int(n) for n in tensor.stride())
    if len(shape)!=len(stride) or any(n<0 for n in shape) or item<=0:
        raise ValueError('invalid-view-metadata')
    empty=any(n==0 for n in shape)
    low=high=pointer
    if not empty:
        for n,s in zip(shape,stride):
            delta=(n-1)*s*item;low+=min(0,delta);high+=max(0,delta)
        high+=item
    return dict(allocation_base=base,storage_extent=extent,
        storage_impl=int(storage._cdata),data_ptr=pointer,shape=list(shape),
        stride=list(stride),dtype=str(tensor.dtype),device=str(tensor.device),
        element_size=item,storage_offset=int(tensor.storage_offset()),
        view_start=low,view_end=high,empty=empty,
        view_within_storage=empty or base<=low<=high<=base+extent,
        range_kind='strided bounding interval; holes may exist')


def overlap(a,b):
    if a['device']!=b['device'] or a['empty'] or b['empty']:return False
    return max(a['view_start'],b['view_start'])<min(a['view_end'],b['view_end'])


class AllocationWindow:
    """Consume actual Torch host allocator snapshots without retaining them."""
    def __init__(self):
        self.seen={};self.allocations={};self.complete=True;self.started=False

    def start(self,snapshot):
        if self.started:raise ValueError('allocation-window-already-started')
        self.started=True
        # Older events belong to another observation window. The caller must
        # not clear external history to manufacture a fresh starting point.
        self.seen={i:len(rows) for i,rows in enumerate(snapshot.get('device_traces',()))}
        for segment in snapshot.get('segments',()):
            device=segment.get('device')
            for block in segment.get('blocks',()):
                if block.get('state') not in ('active_allocated','active_awaiting_free'):continue
                key=(device,block['address'])
                if len(self.allocations)>=MAX_STORAGES:
                    self.complete=False;break
                self.allocations[key]=dict(base=block['address'],extent=block['size'],
                    generation='baseline',state=block['state'],pool=segment.get('segment_pool_id'),
                    stream=segment.get('stream'),creation_observed=False)
        return self.update(snapshot)

    def update(self,snapshot):
        for device,rows in enumerate(snapshot.get('device_traces',())):
            previous=self.seen.get(device,0)
            if len(rows)>=MAX_HISTORY or len(rows)<previous:
                self.complete=False;continue
            for index,entry in enumerate(rows[previous:],previous):
                action=entry.get('action');address=entry.get('addr')
                if action not in ('alloc','free_requested','free_completed') or address is None:continue
                key=(device,address)
                if action=='alloc':
                    if key not in self.allocations and len(self.allocations)>=MAX_STORAGES:
                        self.complete=False;continue
                    self.allocations[key]=dict(base=address,extent=entry['size'],
                        generation=f'alloc-{index}',state='active_allocated',
                        pool=entry.get('pool_id'),stream=entry.get('stream'),creation_observed=True)
                elif key in self.allocations:
                    self.allocations[key]['state']='active_awaiting_free' if action=='free_requested' else 'inactive'
            self.seen[device]=len(rows)
        return dict(complete=self.complete,events_seen=dict(self.seen),tracked=len(self.allocations))

    def identify(self,view):
        device=view['device']
        if not device.startswith('cuda:'):
            return dict(available=False,reason='non-cuda-or-unindexed-device')
        try:index=int(device.split(':',1)[1])
        except ValueError:return dict(available=False,reason='device-index-unavailable')
        # Different Storage objects can be subviews of one allocator block.
        matches=[value for (dev,_),value in self.allocations.items()
                 if dev==index and value['base']<=view['allocation_base']
                 and view['allocation_base']+view['storage_extent']<=value['base']+value['extent']]
        active=[row for row in matches if row['state']!='inactive']
        if active:matches=active
        if len(matches)!=1:return dict(available=False,reason='allocator-range-ambiguous-or-unobserved')
        value=matches[0]
        return dict(value,available=True,incarnation=f"{os.getpid()}:{index}:{value['base']}:{value['generation']}",
            lifecycle_complete=self.complete,
            pre_window_creation='unknown' if not value['creation_observed'] else 'observed',
            scope='Torch caching allocator only; external/IPC allocations not certified')


def compact(tensors,*,window=None,writer=None,reader=None,event=None,captured=None,peer=None):
    rows=[]
    for name,tensor in list(tensors.items())[:64]:
        try:
            view=tensor_view(tensor)
            view.update(name=name,writer=writer,intended_reader=reader,
                completion_event=event,captured_graph=captured,peer_registration=peer,
                allocation_incarnation=window.identify(view) if window else dict(available=False,reason='history-not-enabled'))
            rows.append(view)
        except Exception as exc:rows.append(dict(name=name,available=False,error_type=type(exc).__name__))
    aliases=[]
    valid=[row for row in rows if row.get('view_within_storage') is not None]
    for i,a in enumerate(valid):
        for b in valid[i+1:]:
            if overlap(a,b):aliases.append([a['name'],b['name']])
    return dict(storages=rows,overlapping_ranges=aliases,
        tensors_complete=getattr(tensors,'complete',True) and len(tensors)<=64,
        alias_kind='bounding intervals, not a proof of harmful access',
        tensor_retention=False,device_readback=False)


def compare_manifests(captured,live,limit=128):
    """Bounded interval comparison, including differing subview pointers."""
    rows=[];complete=True
    for a in captured.get('storages',()):
        if 'view_start' not in a:continue
        for b in live.get('storages',()):
            if 'view_start' not in b or not overlap(a,b):continue
            if len(rows)>=limit:complete=False;continue
            ai=a.get('allocation_incarnation') or {};bi=b.get('allocation_incarnation') or {}
            rows.append(dict(captured=a['name'],live=b['name'],
                captured_ptr=a['data_ptr'],live_ptr=b['data_ptr'],
                same_storage_impl=a['storage_impl']==b['storage_impl'],
                captured_incarnation=ai.get('incarnation'),live_incarnation=bi.get('incarnation'),
                incarnation_comparable=bool(ai.get('lifecycle_complete') and bi.get('lifecycle_complete')),
                scope='overlap only; not harmful access or writer proof'))
    return dict(overlaps=rows,complete=complete and captured.get('tensors_complete',False)
                and live.get('tensors_complete',False),device_readback=False)
