"""Decode-step levers (lead 13 Nov 2026, branch claude/n3-decode-step); both switches default off.

``FLASH_S5_DECODE_SAMPLE_FAST=1`` (lever D, the sampling tail). On the last PP stage the V2 sampler sends every replay
step (temperature != 0, no explicit seed, top_k 20 and top_p 0.95 from the model's generation config, no processed
logprobs) to ``flashinfer_sample(logits, top_k_tensor, top_p_tensor)``. With a per-row top_k tensor FlashInfer runs
its full-vocab path: ``top_k_mask_logits`` over 248,320 columns, a full-row softmax, ``top_p_sampling_from_probs``
with one CTA per row. That is about 0.6-0.8 ms of the 1.76 ms device tail measured by the handler anatomy
(SIZING.md). D samples the same truncated distribution from the top ``WIDTH`` columns only:
  * exact top-WIDTH values and ids per row (``torch.topk``, sorted, no JIT);
  * mask entries below the K-th value (ties at the pivot kept, as a ">= pivot" top-k mask keeps them);
  * softmax over the WIDTH survivors, then FlashInfer's own ``top_p_sampling_from_probs``
    (``sampling_from_probs`` without top-p), already loaded by the stock path;
  * the chosen column's vocab id.
  Logits, logit bias, temperature, logprobs (from the raw logits) and everything else are stock and unchanged. A step is
  eligible only when FlashInfer was chosen anyway, every row's top_k is one K with 0 < K <= K_MAX, there is one fp32
  logit row per request and the vocab is at least MIN_VOCAB wide; any other step runs the stock call. More than
  WIDTH - K ties at the K-th value would drop tied entries; a device counter sees it (``tie_overflow``), never a sync.
  Each sampling rank logs a cumulative ``flash-s5 sample-fast`` line (its rank, fast/stock counts by reason, the
  lagged tie counter) at every power of two of the fast count and every LOG_EVERY sampler steps (about 9 s at the
  ~110 steps/s of 8-stream decode): about 200 lines per sampling rank per j2 run, so a reader takes per-arm deltas
  from the docker stamps, each arm edge within one period.

``FLASH_S5_DECODE_ANATOMY=1`` (the free read). The stage-1 handler anatomy (``FLASH_S5_HANDLER_ANATOMY=1``, required)
only records while native progress rides the RPC, so clean decode has no rows. With this switch the engine core opens
a clean window: at most ``MAX_WINDOWS`` windows and ``WINDOW_BUDGET_S`` seconds per run, each closed at ``WINDOW_S``
or as soon as the condition breaks. The condition is the auto-arm released for the arm, no native progress
recording, and the policy idle. In the j2 packet the first such window is the sm arm's 20 s decode-only window, right
after its release_auto_arm and resume. Inside a window the core tags every ``execute_model`` / ``sample_tokens`` RPC.
The workers record those RPCs as handler-anatomy rows: per-step device forward and sample time, host wall, CPU and
seams. Those rows stay out of the native progress trace. At the window's close the workers flush their lines, and the
core logs its own step stamps: monotonic ns, batch queue depth, running and waiting counts. Bounded per run on the
j2 packet: core <= MAX_WINDOWS x 26 lines (MAX_STAMPS stamps of about 20 bytes, packed under LINE_BUDGET); workers
<= MAX_WINDOWS x handler_anatomy.MAX_ROWS rows per rank, about 25 rows per line, so <= 3 x 165 lines per rank. That is
under 2,100 lines per run. The 20 s budget usually ends it inside the first window: 20 s of clean decode is about 2,200
engine steps, so about 14 core lines plus 4 x 90 worker lines.

Per-arm gate, both switches: the engine core reads ``auto_arm.manual_owned`` before each step (the gate
ingress_incremental uses). The stock reference arm (sp) holds the manual lease, so its RPCs are never tagged and its
workers run the stock sampler and the stock anatomy. The tag rides in the RPC kwargs under ``WIRE_KEY``. The worker pops
it before dispatch (``worker_rpc_begin``), so stock methods never see it. An untagged RPC means: D off, no clean row.
Neither switch set: executor_hooks never imports this module, and RPC bytes and outputs are stock.
"""
import json
import os
import threading
import time

SAMPLE_FAST_ENV = 'FLASH_S5_DECODE_SAMPLE_FAST'
ANATOMY_ENV = 'FLASH_S5_DECODE_ANATOMY'
SCHEMA = 's5-decode-step/1'
WIRE_KEY = '_flash_s5_ds'
LINE_SAMPLE = 'flash-s5 sample-fast '
LINE_ANATOMY = 'flash-s5 decode-anatomy '
LINE_GATE = 'flash-s5 decode-step-gate '
LINE_BYTES = 3968            # == native_measurement.PROGRESS_LINE_BYTES
WIDTH = 256                  # columns kept per row (FlashInfer's own top-k-first fast path cap)
K_MAX = 128                  # eligible top_k; leaves WIDTH - K >= 128 slots for ties at the pivot
MIN_VOCAB = 65536            # FlashInfer's own fast-path floor
WINDOW_S = 20.0
WINDOW_BUDGET_S = 20.0
MAX_WINDOWS = 3
MAX_STAMPS = 4096
LINE_BUDGET = LINE_BYTES - 512   # core stamp lines are packed by size under this
ANATOMY_METHODS = ('execute_model', 'sample_tokens')
# wire tag: [sample_fast 0/1, anatomy code, engine transport sequence]
TAG_NONE, TAG_RECORD, TAG_CLOSE = 0, 1, 2
LOG_EVERY = 1024             # sampler steps (fast or stock) between periodic sample-fast lines (compose line, 13 Nov)
REASONS = ('not_gated', 'not_flashinfer', 'no_top_k', 'rows', 'dtype', 'vocab', 'k_mixed', 'k_range', 'error')


def sample_fast_enabled(env=None):
    return (os.environ if env is None else env).get(SAMPLE_FAST_ENV, '0') == '1'


def anatomy_enabled(env=None):
    return (os.environ if env is None else env).get(ANATOMY_ENV, '0') == '1'


def installed(env=None):
    return sample_fast_enabled(env) or anatomy_enabled(env)


def check_startup(env):
    """s5_startup: the anatomy window needs the handler anatomy and the progress transport it rides on."""
    if anatomy_enabled(env):
        if env.get('FLASH_S5_HANDLER_ANATOMY', '') != '1':
            raise RuntimeError('FLASH_S5_DECODE_ANATOMY=1 needs FLASH_S5_HANDLER_ANATOMY=1')
        if not env.get('FLASH_S5_NATIVE_PROGRESS'):
            raise RuntimeError('FLASH_S5_DECODE_ANATOMY=1 needs FLASH_S5_NATIVE_PROGRESS set')
    for key in (SAMPLE_FAST_ENV, ANATOMY_ENV):
        if env.get(key, '0') not in ('0', '1'):
            raise RuntimeError('%s must be 0 or 1' % key)


def _log_line(logger, prefix, body):
    try:
        logger.info(prefix + json.dumps(body, sort_keys=True, separators=(',', ':')))
    except Exception:  # noqa: BLE001 - a log line never fails the step
        pass


# -- engine core -----------------------------------------------------------------------------------------------------
class AnatomyWindow:
    """Engine-core clean window: opens when ``eligible``, bounded in count, length, total time and stamps."""

    def __init__(self, *, window_s=WINDOW_S, budget_s=WINDOW_BUDGET_S, max_windows=MAX_WINDOWS,
                 max_stamps=MAX_STAMPS):
        self.window_s, self.budget_ns = window_s, int(budget_s * 1e9)
        self.max_windows, self.max_stamps = max_windows, max_stamps
        self.open_ns = None
        self.used_ns = 0
        self.windows = 0
        self.close_pending = False
        self.stamps = []
        self.counters = dict(windows=0, steps=0, stamps_dropped=0, tagged=0, closes=0, lines=0)

    def exhausted(self):
        return self.windows >= self.max_windows or self.used_ns >= self.budget_ns

    def step(self, now_ns, eligible, stats, logger):
        """Once per engine step, before the stock step. Returns True while the window is open."""
        if self.open_ns is not None:
            if not eligible or now_ns - self.open_ns >= self.window_s * 1e9 or \
                    self.used_ns + (now_ns - self.open_ns) >= self.budget_ns:
                self._close(now_ns, logger)
            else:
                self.counters['steps'] += 1
                self._stamp(now_ns, stats)
                return True
        if eligible and self.open_ns is None and not self.close_pending and not self.exhausted():
            self.open_ns = now_ns
            self.windows += 1
            self.counters['windows'] += 1
            self.stamps = []
            self.counters['steps'] += 1
            self._stamp(now_ns, stats)
            return True
        return False

    def _stamp(self, now_ns, stats):
        if len(self.stamps) >= self.max_stamps:
            self.counters['stamps_dropped'] += 1
            return
        self.stamps.append((now_ns,) + tuple(stats))

    def _close(self, now_ns, logger):
        self.used_ns += now_ns - self.open_ns
        self.close_pending = True       # the next eligible-method RPC carries the close tag (the workers flush)
        self.counters['closes'] += 1
        for line in self.lines(now_ns):
            _log_line(logger, LINE_ANATOMY, line)
            self.counters['lines'] += 1
        self.open_ns = None
        self.stamps = []

    def lines(self, now_ns):
        base = self.open_ns
        rows = [[(t - base) // 1000] + list(rest) for t, *rest in self.stamps]
        head = dict(schema=SCHEMA, kind='core-steps', window=self.windows, open_ns=base, close_ns=now_ns,
                    fields=['t_us', 'batch_queue', 'running', 'waiting'], counters=dict(self.counters))
        room = LINE_BUDGET - len(json.dumps(dict(head, part=99999, parts=99999, stamps=[]), separators=(',', ':')))
        parts, current, size = [], [], 0
        for row in rows:
            width = len(json.dumps(row, separators=(',', ':'))) + 1
            if current and size + width > room:
                parts.append(current)
                current, size = [], 0
            current.append(row)
            size += width
        parts.append(current)
        return [dict(head, part=i, parts=len(parts), stamps=p) for i, p in enumerate(parts)]

    def tag(self, method):
        """The anatomy code for one stock RPC sent now."""
        if method not in ANATOMY_METHODS:
            return TAG_NONE
        if self.close_pending:
            self.close_pending = False
            return TAG_CLOSE
        if self.open_ns is not None:
            self.counters['tagged'] += 1
            return TAG_RECORD
        return TAG_NONE


class EngineSide:
    """One per engine core process (``executor._flash_s5_decode_step``)."""

    def __init__(self, env=None, *, logger=None, clock=time.monotonic_ns):
        env = os.environ if env is None else env
        self.fast = sample_fast_enabled(env)
        self.window = AnatomyWindow() if anatomy_enabled(env) else None
        self.native_arm = False
        self.logger = logger
        self.clock = clock
        self.counters = dict(steps=0, gate_changes=0, fast_tags=0, errors=0)

    def step(self, engine):
        """before_step_with_batch_queue: the per-arm gate and the anatomy window. Never raises."""
        try:
            self.counters['steps'] += 1
            driver = getattr(engine, '_flash_s5_driver', None)
            policy = getattr(driver, 'auto_arm', None)
            native = policy is not None and policy.manual_owned is False
            if native != self.native_arm:
                self.native_arm = native
                self.counters['gate_changes'] += 1
                _log_line(self.logger, LINE_GATE, dict(schema=SCHEMA, native_arm=native, sample_fast=self.fast,
                                                       anatomy=self.window is not None, counters=dict(self.counters)))
            if self.window is not None:
                executor = getattr(engine, 'model_executor', None)
                progress = getattr(executor, '_flash_s5_progress_recording', None)
                eligible = native and progress is None and not policy.active
                scheduler = getattr(engine, 'scheduler', None)
                stats = (len(getattr(engine, 'batch_queue', None) or ()),
                         len(getattr(scheduler, 'running', None) or ()),
                         len(getattr(scheduler, 'waiting', None) or ()))
                self.window.step(self.clock(), eligible, stats, self.logger)
        except Exception:  # noqa: BLE001 - the gate fails closed: off
            self.counters['errors'] += 1
            self.native_arm = False

    def wire(self, method, sequence):
        """The tag for one stock RPC (collective_rpc send), or None: untagged."""
        fast = int(self.fast and self.native_arm and method == 'sample_tokens')
        anatomy = self.window.tag(method) if self.window is not None else TAG_NONE
        if not fast and anatomy == TAG_NONE:
            return None
        self.counters['fast_tags'] += fast
        return [fast, anatomy, int(sequence)]


# -- worker ------------------------------------------------------------------------------------------------------------
def _torch():
    import torch
    return torch


def _flashinfer_sampling():
    from flashinfer import sampling
    return sampling


class FastSampler:
    """Wraps one Sampler instance's ``_sample_random`` on the last PP stage (instance attribute, never the class)."""

    def __init__(self, sampler, *, torch=None, top_p_sample=None, sample=None, logger=None, width=WIDTH,
                 k_max=K_MAX, min_vocab=MIN_VOCAB, rank=None, log_every=LOG_EVERY):
        self.sampler = sampler
        self.rank, self.log_every = rank, log_every
        self.torch = torch if torch is not None else _torch()
        self.stock = sampler._sample_random
        self._top_p_sample, self._sample = top_p_sample, sample
        self.logger = logger
        self.width, self.k_max, self.min_vocab = width, k_max, min_vocab
        self.gated = False                   # set per RPC by the worker from the wire tag
        self.counters = dict(fast=0, stock=0, **{'stock_' + r: 0 for r in REASONS})
        self.tie_overflow = None             # device counter, read lagged at log points only
        self._host_overflow = None
        self._next_log = 1
        sampler._sample_random = self.sample_random

    def _kernels(self):
        """(top_p_sample(probs, top_p, ids), sample(probs, ids)): FlashInfer's, already loaded by the stock path. ``ids``
        are the columns' vocab ids; FlashInfer does not read them (a CPU test's token-keyed sampler does)."""
        if self._top_p_sample is None or self._sample is None:
            fi = _flashinfer_sampling()
            self._top_p_sample = self._top_p_sample or (
                lambda probs, top_p, ids: fi.top_p_sampling_from_probs(probs, top_p, indices=None, deterministic=True))
            self._sample = self._sample or (
                lambda probs, ids: fi.sampling_from_probs(probs, indices=None, deterministic=True))
        return self._top_p_sample, self._sample

    def eligible(self, logits, idx_mapping_np, top_k, use_flashinfer):
        """(K, None) when D samples this step, else (None, reason). Host-side reads only; never a sync."""
        if not self.gated:
            return None, 'not_gated'
        if not use_flashinfer:
            return None, 'not_flashinfer'
        if top_k is None:
            return None, 'no_top_k'
        if getattr(logits, 'dim', lambda: 0)() != 2 or logits.shape[0] != len(idx_mapping_np) or logits.shape[0] < 1:
            return None, 'rows'
        if logits.dtype != self.torch.float32:
            return None, 'dtype'
        vocab = logits.shape[1]
        if vocab < self.min_vocab or vocab < self.width:
            return None, 'vocab'
        ks = self.sampler.sampling_states.top_k.np[idx_mapping_np]
        k = int(ks[0])
        if not bool((ks == k).all()):
            return None, 'k_mixed'
        if not 0 < k <= self.k_max or k >= vocab:
            return None, 'k_range'
        return k, None

    def sample_random(self, processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p, use_flashinfer):
        try:
            k, why = self.eligible(processed_logits, idx_mapping_np, top_k, use_flashinfer)
        except Exception:  # noqa: BLE001 - an unreadable step is a stock step
            k, why = None, 'error'
        if k is None:
            self.counters['stock'] += 1
            self.counters['stock_' + why] += 1
            result = self.stock(processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k, top_p,
                                use_flashinfer)
            self._periodic()
            return result
        sampled = self.sample_top(processed_logits, k, top_p)
        self.counters['fast'] += 1
        if self.counters['fast'] >= self._next_log:
            self._next_log *= 2
            self.log()
        else:
            self._periodic()
        return sampled, processed_logits

    def _periodic(self):
        """A cumulative line every ``log_every`` sampler steps (fast or stock), so a reader can take per-arm deltas
        from the docker-stamped lines; the power-of-two lines above stay as they were."""
        if self.log_every and (self.counters['fast'] + self.counters['stock']) % self.log_every == 0:
            self.log()

    def sample_top(self, logits, k, top_p):
        torch = self.torch
        values, ids = torch.topk(logits, self.width, dim=-1, largest=True, sorted=True)
        pivot = values[:, k - 1:k]
        kept = values.masked_fill(values < pivot, float('-inf'))
        overflow = (values[:, -1:] >= pivot).sum()
        self.tie_overflow = overflow if self.tie_overflow is None else self.tie_overflow + overflow
        probs = torch.softmax(kept, dim=-1)
        top_p_sample, sample = self._kernels()
        local = top_p_sample(probs, top_p, ids) if top_p is not None else sample(probs, ids)
        return ids.gather(1, local.view(-1, 1).to(torch.int64)).view(-1).to(torch.int64)

    def overflow_seen(self):
        """The device tie counter, lagged by one log point (non-blocking copy; never a sync)."""
        torch = self.torch
        if self.tie_overflow is None:
            return 0
        if self._host_overflow is None:
            try:
                self._host_overflow = torch.zeros(1, dtype=torch.int64, pin_memory=self.tie_overflow.is_cuda)
            except Exception:  # noqa: BLE001 - no pinned memory (CPU): a plain host tensor
                self._host_overflow = torch.zeros(1, dtype=torch.int64)
            seen = None
        else:
            seen = int(self._host_overflow[0])
        self._host_overflow.copy_(self.tie_overflow.view(1), non_blocking=True)
        return seen

    def log(self):
        if self.logger is not None:
            _log_line(self.logger, LINE_SAMPLE, dict(schema=SCHEMA, width=self.width, k_max=self.k_max, rank=self.rank,
                                                     counters=dict(self.counters),
                                                     tie_overflow_lagged=self.overflow_seen()))


class WorkerSide:
    """One per worker process: pops the tag, gates the sampler, opens clean-window anatomy rows."""

    def __init__(self, env=None, *, logger=None):
        env = os.environ if env is None else env
        self.fast_on = sample_fast_enabled(env)
        self.anatomy_on = anatomy_enabled(env)
        self.logger = logger
        self.fast = None                     # FastSampler, installed at the first gated sample_tokens on a sampling rank
        self.install_refused = False
        self.lock = threading.Lock()
        self.counters = dict(tags=0, fast_rpcs=0, clean_rows=0, flushes=0, no_sampler=0, errors=0)

    def pop(self, proc, name, kwargs):
        """Before dispatch: (fast, anatomy code, sequence) from the tag, and the sampler gate for this RPC."""
        tag = kwargs.pop(WIRE_KEY, None) if type(kwargs) is dict else None
        if self.fast is not None:
            self.fast.gated = False
        if tag is None:
            return None
        self.counters['tags'] += 1
        try:
            fast, anatomy, sequence = int(tag[0]), int(tag[1]), int(tag[2])
        except Exception:  # noqa: BLE001
            self.counters['errors'] += 1
            return None
        if fast and self.fast_on and name == 'sample_tokens':
            sampler = self._sampler(proc)
            if sampler is not None:
                sampler.gated = True
                self.counters['fast_rpcs'] += 1
        return fast, anatomy if self.anatomy_on else TAG_NONE, sequence

    def _sampler(self, proc):
        if self.fast is not None or self.install_refused:
            return self.fast
        with self.lock:
            if self.fast is None and not self.install_refused:
                runner = getattr(getattr(proc, 'worker', None), 'model_runner', None)
                sampler = getattr(runner, 'sampler', None)
                if sampler is None or not callable(getattr(sampler, '_sample_random', None)) or \
                        getattr(sampler, 'sampling_states', None) is None:
                    # not the last PP stage (no sampler), or another runner: stock, once
                    self.install_refused = True
                    self.counters['no_sampler'] += 1
                    return None
                self.fast = FastSampler(sampler, logger=self.logger, rank=getattr(proc, 'rank', None))
        return self.fast


__all__ = ['ANATOMY_ENV', 'AnatomyWindow', 'EngineSide', 'FastSampler', 'K_MAX', 'LINE_ANATOMY',
           'LINE_GATE', 'LINE_SAMPLE', 'MAX_WINDOWS', 'MIN_VOCAB', 'SAMPLE_FAST_ENV', 'TAG_CLOSE', 'TAG_NONE',
           'TAG_RECORD', 'WIDTH', 'WINDOW_S', 'WIRE_KEY', 'WorkerSide', 'anatomy_enabled', 'check_startup', 'installed',
           'sample_fast_enabled']
