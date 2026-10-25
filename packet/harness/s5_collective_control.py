"""Scoped local-return collective control, never used by an engine."""
from contextlib import contextmanager


class LocalPrefillReducer:
    def __init__(self,original,stream_id):
        self.progress=original.progress;self.step='setup';self.collective=0
        self.stream_id=stream_id
    def __call__(self,x):
        self.collective+=1
        fields=dict(step=self.step,collective=self.collective,shape=list(x.shape),
                    backend='local-noop',stream=int(self.stream_id()))
        self.progress.event('prefill','collective-enter',**fields)
        # Deliberately no GPU op and no rank exchange. This changes the model's
        # arithmetic; compare only with this control's own solo trajectory.
        self.progress.event('prefill','collective-issued',**fields)
        return x


@contextmanager
def local_prefill_collectives(lane,stream_id):
    original=lane.reduce
    replacement=LocalPrefillReducer(original,stream_id)
    lane.reduce=replacement
    try:yield replacement
    finally:lane.reduce=original
