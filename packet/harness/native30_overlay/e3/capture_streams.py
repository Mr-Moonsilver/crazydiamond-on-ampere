"""Pinned side-stream owners for split graph capture; no torch import at startup.

Only object-owned SharedExperts/PLE streams are replaced, and only while the
split bank warms/captures. Event topology is untouched. Streams are retained
through process exit once exposed to tensor work. Graph drain is necessary but
not sufficient to release CUDA allocator block-to-stream records.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
from e3_types import E3ContractError

PINS = {
    'model_executor/layers/fused_moe/runner/shared_experts.py': 'd89e4c79c46f6fe61262c51f3facf102e71ce1e4eb3db67f72b6bd8470937be9',
    'models/qwen4_exp/nvidia/ngram_embedding.py': '40320647b13c31e4f94bf611b4bed386292a5b3bf97ed138249d710493501b17',
    'utils/torch_utils.py': 'eec8b1e3c7c673d3d171b92530f8b07acabdad5b0e3ab07cc0e717f827af8e5a',
}

class CaptureStreamError(E3ContractError):
    pass


def pinned_types():
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
    from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpPLEPinnedHostEmbedding
    import vllm.utils.torch_utils as tu
    for obj, rel in zip((SharedExperts, Qwen4ExpPLEPinnedHostEmbedding, tu), PINS):
        if hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest() != PINS[rel]:
            raise CaptureStreamError('capture stream owner source pin differs: ' + rel)
    return SharedExperts, Qwen4ExpPLEPinnedHostEmbedding


class CaptureStreams:
    def __init__(self, split, owners):
        # owners: (object, exact attribute, role). Objects/handles never in receipts.
        self.split = split
        self.owners = tuple(owners)
        self.originals = []
        self.streams = []
        self.replacements = []
        self.state = 'new'
        self.retained = False
        self.graphs = set()
        seen = set()
        for obj, attr, role in self.owners:
            if (attr, role) not in (('_stream', 'shared'), ('_prefetch_stream', 'ple')):
                raise CaptureStreamError('unsupported stream owner')
            if (id(obj), attr) in seen: raise CaptureStreamError('duplicate stream owner')
            seen.add((id(obj), attr))
            old = getattr(obj, attr, None)
            if old is None or not getattr(old, 'cuda_stream', 0):
                raise CaptureStreamError('stream owner has no stream')
            self.originals.append(old)

    def prepare(self):
        if self.state != 'new': raise CaptureStreamError('capture streams already prepared')
        # Drain original work before touching any owner or creating graph resources.
        self.split.decode_stream.synchronize()
        for old in self.originals: old.synchronize()
        groups = {}
        try:
            for old in self.originals:
                key = old.cuda_stream
                if key not in groups:
                    stream = self.split.create_decode_aux()
                    self.streams.append(stream)
                    groups[key] = stream
                self.replacements.append(groups[key])
            self.state = 'ready'
        except BaseException:
            self.state = 'failed'
            # Usually pre-exposure; a diagnostic may already have used the main
            # green stream. In that case preserve all native resources as well.
            if not self.split.lifetime_record()['allocator_exposed']:
                for stream in reversed(self.streams): self.split.destroy_decode_aux(stream)
            self.streams.clear()
            raise

    @contextmanager
    def capture(self):
        if self.state != 'ready': raise CaptureStreamError('capture scope unavailable')
        for (obj, attr, _), old in zip(self.owners, self.originals):
            if getattr(obj, attr) is not old: raise CaptureStreamError('owner stream changed before capture')
        self.split.check_decode_stream(self.split.decode_stream)
        for stream in self.streams: self.split.check_decode_stream(stream)
        self.split.retain_graph_owner(self)
        self.retained = True
        self.state = 'capturing'
        installed = []
        try:
            for (obj, attr, _), stream, old in zip(self.owners, self.replacements, self.originals):
                setattr(obj, attr, stream)
                installed.append((obj, attr, old))
            yield self
            self.state = 'captured'
        except BaseException:
            self.state = 'failed'
            raise
        finally:
            for obj, attr, old in reversed(installed): setattr(obj, attr, old)

    def consumer_streams(self):
        return tuple(self.streams) + tuple(self.originals)

    def attach_graph(self, graph):
        if not self.retained or id(graph) in self.graphs:
            raise CaptureStreamError('graph lease unavailable or duplicated')
        self.graphs.add(id(graph))

    def retire_graph(self, graph):
        if id(graph) not in self.graphs: raise CaptureStreamError('unknown retained graph')
        self.graphs.remove(id(graph))

    def drain(self):
        self.split.decode_stream.synchronize()
        for stream in self.consumer_streams(): stream.synchronize()

    def close_after_graphs(self):
        # Caller must have reset its graphs; retain prevents premature split close.
        if self.state == 'capturing' or self.graphs: raise CaptureStreamError('capture or graphs still active')
        if self.state == 'closed': return
        self.drain()
        if self.retained:
            self.split.release_graph_owner(self)
            self.retained = False
        # Exposure is irreversible even when capture raises: tensors may already
        # carry record_stream entries. The split owns wrappers until process exit.
        if not self.split.lifetime_record()['allocator_exposed']:
            for stream in reversed(self.streams): self.split.destroy_decode_aux(stream)
        self.streams.clear()
        self.state = 'closed'

    def record(self):
        return {'state': self.state, 'shared_owners': sum(r == 'shared' for _, _, r in self.owners),
                'ple_owners': sum(r == 'ple' for _, _, r in self.owners),
                'auxiliary_streams': len(self.streams), 'graph_lease': self.retained,
                'allocator_lifetime': self.split.lifetime_record(),
                'original_owners_restored': all(getattr(o, a) is v for (o,a,_),v in zip(self.owners,self.originals))}


def for_runner(runner, split):
    shared_type, ple_type = pinned_types()
    modules = tuple(runner.model.modules())
    # Exact pinned model family, not a generic count of arbitrary decoder layers.
    models = [m for m in modules if type(m).__name__ == 'Qwen4ExpModel'
              and type(m).__module__ == 'vllm.models.qwen4_exp.nvidia.model']
    if len(models) != 1: raise CaptureStreamError('expected one local Qwen4Exp model')
    model = models[0]
    start, end = model.start_layer, model.end_layer
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise CaptureStreamError('invalid local layer interval')
    shared = [m for m in modules if type(m) is shared_type]
    ple = [m for m in modules if type(m) is ple_type]
    expected_ple = sum(start < n <= end for n in set(model.config.ple_layer_ids))
    if len(shared) != end-start or len(ple) != expected_ple:
        raise CaptureStreamError('side-stream owner census differs from local model')
    owners = [(m, '_stream', 'shared') for m in shared] + [(m, '_prefetch_stream', 'ple') for m in ple]
    result = CaptureStreams(split, owners)
    result.prepare()
    return result
