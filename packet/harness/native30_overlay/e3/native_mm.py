"""Bounded image-only CPU snapshot and per-native-frame splice ownership.

No imports of torch/vLLM on the text path. Bytes never appear in repr/logs.
"""
from dataclasses import dataclass, field
import hashlib
import logging
import os

from e3_types import E3ContractError, TransactionId

WIDTH = 2560
MAX_IMAGE_ROWS = 1024
CACHE_ROWS = 8 * MAX_IMAGE_ROWS
ROW_BYTES = WIDTH * 2


class NativeMMError(E3ContractError):
    pass


def enabled():
    return os.environ.get('FLASH_S5_NATIVE_MM', '0') == '1'


def cap_encoder_budget(compute, cache):
    if not enabled():
        return compute, cache
    if type(compute) is not int or type(cache) is not int or min(compute, cache) <= 0:
        raise NativeMMError('ordinary MM encoder budget is not positive')
    return min(compute, CACHE_ROWS), min(cache, CACHE_ROWS)


@dataclass(frozen=True, repr=False)
class MMChunk:
    transaction: TransactionId
    state_epoch: int
    rows: tuple
    data: bytes = field(repr=False)
    sha256: str

    def __post_init__(self):
        self.validate(self.transaction, self.state_epoch)

    def validate(self, transaction, state_epoch):
        if (type(self.transaction) is not TransactionId or self.transaction != transaction
                or type(self.state_epoch) is not int or self.state_epoch < 1 or self.state_epoch != state_epoch):
            raise NativeMMError('MM chunk owner differs')
        size = transaction.chunk_end - transaction.chunk_start
        if (type(self.rows) is not tuple or not 0 < len(self.rows) <= MAX_IMAGE_ROWS
                or any(type(i) is not int or not 0 <= i < size for i in self.rows)
                or self.rows != tuple(sorted(set(self.rows)))):
            raise NativeMMError('MM chunk rows are not unique ordered positions')
        if (type(self.data) is not bytes or len(self.data) != len(self.rows) * ROW_BYTES
                or hashlib.sha256(self.data).hexdigest() != self.sha256):
            raise NativeMMError('MM chunk bytes differ from their frozen digest/shape')
        return self

    def __repr__(self):
        return f'MMChunk(rows={len(self.rows)}, bytes={len(self.data)})'


@dataclass(frozen=True, repr=False)
class ImagePlan:
    signature: tuple
    positions: tuple
    data: bytes = field(repr=False)

    def chunk(self, transaction, epoch):
        picked = [(i, p - transaction.chunk_start) for i, p in enumerate(self.positions)
                  if transaction.chunk_start <= p < transaction.chunk_end]
        if not picked:
            return None
        data = b''.join(self.data[i*ROW_BYTES:(i+1)*ROW_BYTES] for i, _ in picked)
        return MMChunk(transaction, epoch, tuple(p for _, p in picked), data, hashlib.sha256(data).hexdigest())


def request_plan(request):
    if not enabled():
        raise NativeMMError('native MM switch is off')
    features = request.mm_features
    if len(features) != 1:
        raise NativeMMError('one image per request required')
    feature = features[0]
    place = feature.mm_position
    item = feature.data
    if feature.modality != 'image' or item is None or set(item) != {'image_embeds', 'image_grid_thw'}:
        raise NativeMMError('explicit image embeddings and grid required; no tower or cached-data guess')
    tensor, grid = item['image_embeds'].data, item['image_grid_thw'].data
    signature = (request.request_id, id(feature), id(item), id(tensor), id(grid),
                 feature.identifier, place.offset, place.length, id(place.is_embed))
    old = getattr(request, '_flash_s5_mm_plan', None)
    if old is not None:
        if type(old) is not ImagePlan or old.signature != signature:
            raise NativeMMError('MM request metadata changed after snapshot')
        return old
    import torch
    image_tensor(tensor)
    if (not isinstance(grid, torch.Tensor) or grid.device.type != 'cpu'
            or grid.dtype != torch.int64 or grid.numel() != 3):
        raise NativeMMError('MM image grid must be three CPU int64 values')
    t, h, w = grid.reshape(-1).tolist()
    if t != 1 or h <= 0 or w <= 0 or h % 2 or w % 2 or h*w//4 != tensor.shape[0]:
        raise NativeMMError('MM image grid disagrees with BF16 rows')
    if (type(place.offset) is not int or place.offset < 0 or type(place.length) is not int
            or not 0 < place.length <= 4*MAX_IMAGE_ROWS
            or place.offset + place.length > request.num_prompt_tokens):
        raise NativeMMError('MM placeholder is outside prompt')
    mask = place.is_embed
    if mask is None:
        selected = tuple(range(place.length))
    else:
        if (not isinstance(mask, torch.Tensor) or mask.device.type != 'cpu'
                or mask.dtype != torch.bool or tuple(mask.shape) != (place.length,)):
            raise NativeMMError('MM placeholder mask must be CPU bool vector')
        selected = tuple(i for i, use in enumerate(mask.tolist()) if use)
    if len(selected) != tensor.shape[0]:
        raise NativeMMError('MM placeholder mask disagrees with BF16 rows')
    plan = ImagePlan(signature, tuple(place.offset+i for i in selected),
                     tensor.contiguous().view(torch.uint8).numpy().tobytes())
    request._flash_s5_mm_plan = plan
    return plan


def image_tensor(tensor):
    import torch
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
            or tensor.dtype != torch.bfloat16 or tensor.ndim != 2
            or tensor.shape[1] != WIDTH or not 0 < tensor.shape[0] <= MAX_IMAGE_ROWS
            or not bool(torch.isfinite(tensor).all())):
        raise NativeMMError('MM embeddings must be finite bounded CPU BF16 rows')
    return tensor


def stage_image(feature, device, upload):
    """Mirror prompt_embeds passthrough, keeping each GPU feature's storage private."""
    if not enabled() or feature.modality != 'image' or set(feature.data) != {'image_embeds','image_grid_thw'}:
        raise NativeMMError('ordinary MM path requires explicit image embeddings')
    tensor=image_tensor(feature.data['image_embeds'].data)
    if feature.mm_position.get_num_embeds() != tensor.shape[0]:
        raise NativeMMError('ordinary MM mask disagrees with image rows')
    result=upload(tensor.contiguous(), device=device)
    if result.untyped_storage().nbytes() != result.nbytes:
        raise NativeMMError('ordinary MM feature retained a larger shared storage')
    return result


def inventory(runner):
    visual=getattr(runner.model,'visual',None)
    if visual is None:raise NativeMMError('vision module missing from MM consumer')
    # StageMissingLayer deliberately hides the original module from named_parameters.
    visual=visual.__dict__.get('module',visual)
    if visual is None:raise NativeMMError('language-only tower stub is not an MM consumer')
    gpu=lambda tensor:getattr(getattr(tensor,'device',None),'type',None)=='cuda'
    tower=sum(t.nbytes for t in (*visual.parameters(),*visual.buffers()) if gpu(t))
    cache=getattr(runner,'encoder_cache',None)
    values=[] if cache is None else list(cache.encoder_outputs.values())
    def storage_bytes(tensors):
        storage={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors if gpu(t)}
        return sum(storage.values())
    state=getattr(runner,'model_state',None)
    inputs=getattr(getattr(state,'encoder_runner',None),'inputs_embeds',None)
    lane=getattr(getattr(runner,'_flash_s5_dispatcher',None),'lane',None)
    lease=getattr(lane,'_native_mm_lease',None)
    return dict(schema='native-mm-inventory/1',tower_registered_cuda_bytes=tower,
        gpu_encoder_cache_present=cache is not None,cache_budget_tokens=CACHE_ROWS,
        cache_budget_bytes=CACHE_ROWS*ROW_BYTES,cache_feature_bytes=sum(t.nbytes for t in values if gpu(t)),
        cache_storage_bytes=storage_bytes(values),
        encoder_inputs_buffer_bytes=inputs.nbytes if inputs is not None and gpu(inputs) else 0,
        active_splice_device_bytes=storage_bytes([] if lease is None else lease.keepalive),
        basis='host tensor/storage metadata only; no CUDA synchronization or tensor values read')


def eligible(request):
    if not enabled():
        return False
    try:
        request_plan(request)
        return True
    except NativeMMError as exc:
        request._flash_s5_mm_refusal = str(exc)
        return False


class SpliceLease:
    """The frame retains all H2D sources/inputs until its queried prefix event."""
    def __init__(self, payload):
        self.payload = payload
        self.keepalive = []
        self.event = None

    def bind(self, event):
        if self.event is not None or not callable(getattr(event, 'query', None)):
            raise NativeMMError('MM prefix needs its own completion event')
        self.event = event

    def release(self):
        if self.event is None or not self.event.query():
            raise NativeMMError('MM prefix consumers have not completed')
        self.keepalive.clear()
        self.payload = None


def prefix_inputs(lane, admission, torch):
    payload = admission.mm
    payload.validate(admission.transaction, admission.state_epoch)
    if not enabled() or getattr(lane, '_native_mm_lease', None) is not None:
        raise NativeMMError('MM prefix switch/owner differs')
    lease = lane._native_mm_lease = SpliceLease(payload)
    host = torch.frombuffer(bytearray(payload.data), dtype=torch.bfloat16).reshape(len(payload.rows), WIDTH)
    lease.keepalive.append(host)
    ids = lane.inputs['input_ids']
    if ids.device.type == 'cuda':
        host = host.pin_memory()
        lease.keepalive.append(host)
    device = host.to(device=ids.device, non_blocking=ids.device.type == 'cuda')
    lease.keepalive.append(device)
    rows = torch.tensor(payload.rows, dtype=torch.int64, device=ids.device)
    lease.keepalive.append(rows)
    embeds = lane.model.embed_input_ids(ids)
    # The ordinary embedding temporary is consumed on its allocating prefill stream,
    # exactly as in the existing prefix; do not pin a second full-chunk scratch.
    if embeds.dtype != torch.bfloat16 or tuple(embeds.shape) != (len(admission.input_ids), WIDTH):
        raise NativeMMError('native token embedding shape/dtype differs')
    embeds.index_copy_(0, rows, device)
    return embeds


def note_prefix(admission, stage):
    payload = admission.mm
    logging.getLogger(__name__).info(
        'flash-s5 native-mm-prefix txn=%d generation=%d stage=%d rows=%d device_feature_bytes=%d digest=%s',
        admission.transaction.txn_seq, admission.transaction.request_generation, stage,
        len(payload.rows), len(payload.data) if stage == 0 else 0, payload.sha256)
