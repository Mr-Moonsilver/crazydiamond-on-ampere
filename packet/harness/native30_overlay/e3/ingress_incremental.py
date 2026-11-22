"""Incremental ingress for appends (FLASH_S5_INGRESS_INCREMENTAL, default 0) and the append common-prefix counter.

API process (Splicer, Binding, install): on an append the rendered chat text repeats the previous prompt's text up to
the generation prompt. HF fast tokenizers extract added tokens before normalisation and pre-tokenisation, so the text
on either side of a special added token (special, not normalized, no lstrip/rstrip/single_word, contained in no other
added token) is tokenised independently. The splice is therefore exact: the cached ids up to the last such token wholly
inside the common character prefix, plus the tokenisation of the rest of the new text. Anything else (no cached entry,
no join, other encode arguments, offsets requested, a result above the request's truncation length, a verify mismatch)
takes the stock call unchanged. Correctness rests on the text comparison only; the cache key (a hash of the opening
text, and add_special_tokens) only finds the candidate. The length bound is per request, not per conversation (vLLM's
max_length = max_model_len - max_tokens + 1 follows each turn's reply budget), so it is not in the key (ingress key fix,
13 Nov): an entry holds untruncated ids only (a stock result at or past its bound is not stored), the tail is tokenised
without the bound, and each request's own bound is applied to the spliced ids (above it: the stock call, which
truncates as stock does). Every VERIFY_EVERY-th splice also runs the stock call and uses its ids (a mismatch drops the
entry and is counted). The verify rate is settable (FLASH_S5_INGRESS_VERIFY_EVERY, default 16; 1 = every splice, 0 =
none); with FLASH_S5_INGRESS_VERIFY_ASYNC=1 the spliced ids are returned at once and the stock tokenisation runs on one
separate verifier thread (the latency reading survives), and the first mismatch disables the splice for the launch
(disabled='verify-mismatch'; one log line with the conversation index and the first differing position only).

Bound to the engine selection: the splice runs only while the engine's auto-arm is released (the native engine arms).
The API process mirrors the engine's own rule (native_auto_arm.manual_control) from the controls it forwards: a
release_auto_arm reply with released True selects; any other non-read control (pause, selections, transitions)
deselects; a status reply sets it from auto_arm.manual_owned. The stock reference arm (sp) holds the manual lease,
so its turns take the stock call. A flip to selected clears the cache: each arm starts empty.

Engine core (PrefixCounter, log-only, recorded only while the auto-arm is released): at admission, the common-prefix
length between the new prompt ids and (previous prompt ids + generated ids) of the same conversation (cache_salt),
as a histogram relative to the previous prompt end, plus the count of appends where the match runs through the
whole previous reply. Found by the chained block hashes (equality at block i implies all earlier blocks), then tokens
inside the first differing block. Never consulted by any scheduling decision.
"""
import array
import collections
import hashlib
import logging
import os
import threading
import time

ENV = 'FLASH_S5_INGRESS_INCREMENTAL'
VERIFY_EVERY = 16
VERIFY_EVERY_ENV = 'FLASH_S5_INGRESS_VERIFY_EVERY'     # night-pack fix 3: settable; unset = VERIFY_EVERY
VERIFY_ASYNC_ENV = 'FLASH_S5_INGRESS_VERIFY_ASYNC'     # night-pack fix 3: 1 = off-path verifier thread; unset = 0
VERIFY_QUEUE = 64                                       # pending off-path verifications; past it a splice goes unverified
MISMATCH_KEEP = 8
MAX_ENTRIES = 16
MAX_BYTES = 128 << 20
KEY_CHARS = 4096
KEY_JOINS = 2               # the key text ends at the second join end (the system or first user message), capped
PER_KEY = 4                 # candidates per key: conversations sharing a system prompt do not thrash
STOCK_KWARGS = frozenset(('truncation', 'max_length', 'add_special_tokens'))
LENGTH_KWARGS = frozenset(('truncation', 'max_length'))  # per request: applied to each result, never in the key
NO_TRUNCATION = (False, 'do_not_truncate')
SESSION_NO_CHANGE = frozenset(('resume',))      # stock resume_generation: no native command, no lease
COUNTER_ENTRIES = 32           # night-pack fix 4 (was 12: an evicted conversation's next append counted as 'first')
EVICTED_KEEP = 1024
WINDOW_APPENDS = 8             # night-pack fix 4: the id window around the first divergence, first appends only
WINDOW_HALF = 6
REPLY_OFFSETS = 16             # night-pack fix 4: reply-aligned match, offsets from the previous prompt end searched
MS_BUCKETS = (1, 2, 5, 10, 20, 50, 100, 200, 500)


def enabled(env=None):
    env = os.environ if env is None else env
    return env.get(ENV, '0') == '1'


def verify_settings(env=None):
    """(every, async) from the launch env; strict: a malformed value raises (the install then reports install-error)."""
    env = os.environ if env is None else env
    every = env.get(VERIFY_EVERY_ENV)
    every = VERIFY_EVERY if every is None else int(every)
    if every < 0:
        raise ValueError(VERIFY_EVERY_ENV)
    mode = env.get(VERIFY_ASYNC_ENV, '0')
    if mode not in ('0', '1'):
        raise ValueError(VERIFY_ASYNC_ENV)
    return every, mode == '1'


def _logger():
    return logging.getLogger('vllm.flash_s5.ingress')


def _first_difference(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _bucket(ms):
    for b in MS_BUCKETS:
        if ms <= b:
            return '<=%d' % b
    return '>%d' % MS_BUCKETS[-1]


class Splicer:
    """Per-conversation (text, ids, special-token ends) cache and the exact splice. Thread-safe."""

    def __init__(self, tokenizer, *, max_entries=MAX_ENTRIES, max_bytes=MAX_BYTES, verify_every=VERIFY_EVERY,
                 verify_async=False):
        self.tokenizer = tokenizer
        self.max_entries, self.max_bytes, self.verify_every = max_entries, max_bytes, verify_every
        self.verify_async = bool(verify_async)
        self.vcounts = collections.Counter()      # written by the verifier thread only
        self.verify_ms = collections.Counter()
        self.mismatches = []                       # (conversation index, first differing position), at most MISMATCH_KEEP
        self.conversations = {}                    # key -> conversation index (first seen order), bounded
        self.vqueue = collections.deque()
        self.vcond = threading.Condition()
        self.vthread = None
        self.lock = threading.Lock()
        self.cache = collections.OrderedDict()     # entry id -> (key, text, packed ids, join ends); LRU order
        self.by_key = {}                            # key -> [entry ids], most recent last
        self.serial = 0
        self.bytes = 0
        self.sizes = {}
        self.counts = collections.Counter()
        self.fallbacks = collections.Counter()
        self.sums = collections.Counter()
        self.splice_ms = collections.Counter()
        self.full_ms = collections.Counter()
        self.disabled = None
        self.joins = {}
        try:
            decoder = tokenizer.added_tokens_decoder
            contents = [t.content for t in decoder.values()]
            for tid, t in decoder.items():
                if (t.special and not t.normalized and not t.lstrip and not t.rstrip and not t.single_word
                        and t.content and not any(t.content in c and c != t.content for c in contents)):
                    self.joins[t.content] = tid
            if getattr(tokenizer, 'backend_tokenizer', None) is None:
                self.disabled = 'not-a-fast-tokenizer'
            elif not self.joins:
                self.disabled = 'no-join-token'
            elif list(tokenizer('', add_special_tokens=False)['input_ids']) != []:
                self.disabled = 'empty-encode-adds-tokens'
        except Exception as exc:  # the stock call stays in place
            self.disabled = 'probe-error:' + type(exc).__name__
        self.join_ids = {tid: content for content, tid in self.joins.items()}

    # -- the splice ---------------------------------------------------------------------------------------------
    def ends(self, text, ids, start_char=0, start_index=0, prefix=()):
        """(char end, id index + 1) of each join token, in order; prefix carries already-known ends."""
        out = list(prefix)
        found = []
        for tid, name in self.join_ids.items():
            if name not in text:        # C-level text search first: only present joins scan the ids
                continue
            i = start_index
            while True:
                try:
                    i = ids.index(tid, i)
                except ValueError:
                    break
                found.append(i)
                i += 1
        found.sort()
        at = start_char
        for i in found:
            name = self.join_ids[ids[i]]
            at = text.find(name, at)
            if at < 0:
                return None
            at += len(name)
            out.append((at, i + 1))
        return out

    @staticmethod
    def length_bound(kwargs):
        """The request's truncation bound on the ids: max_length unless truncation is off (stock then ignores it).
        Conservative: truncation unset with a max_length counts as bounded (HF then truncates longest_first)."""
        if kwargs.get('truncation') in NO_TRUNCATION:
            return None
        return kwargs.get('max_length')

    def key(self, text, kwargs):
        """Stable across a conversation's turns: the opening text up to the KEY_JOINS-th join end, at most KEY_CHARS,
        and the encode arguments other than the per-request length bound (LENGTH_KWARGS)."""
        head = text[:KEY_CHARS]
        at = 0
        for _ in range(KEY_JOINS):
            hits = [(i, len(name)) for name in self.joins for i in (head.find(name, at),) if i >= 0]
            if not hits:
                break
            i, n = min(hits)
            at = i + n
        else:
            head = head[:at]
        raw = head.encode('utf-8', 'surrogatepass')
        return (hashlib.blake2b(raw, digest_size=16).digest(),
                tuple(sorted(item for item in kwargs.items() if item[0] not in LENGTH_KWARGS)))

    def cut(self, entry, text):
        """(cut char, cut index, position in ends) of the largest join end inside the common prefix, or None."""
        old_text, old_ids, ends = entry
        lo, hi = 0, len(ends) - 1
        best = -1                       # prefix equality is monotone in the length: bisect over the join ends
        while lo <= hi:
            mid = (lo + hi) // 2
            end = ends[mid][0]
            if end <= len(text) and text[:end] == old_text[:end]:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        return None if best < 0 else (ends[best][0], ends[best][1], best)

    def splice(self, entry, text, kwargs, found=None):
        """((ids, cut char, cut index, ends kept, packed ids, continues the whole cached text), None) or (None, reason)."""
        old_text, old_ids, ends = entry
        found = self.cut(entry, text) if found is None else found
        if found is None:
            return None, 'no-join'
        cut, k, best = found
        plain = {name: value for name, value in kwargs.items() if name not in LENGTH_KWARGS}
        tail = list(self.tokenizer(text[cut:], **plain)['input_ids']) if cut < len(text) else []
        head = old_ids[:k]
        ids = head.tolist() + tail
        bound = self.length_bound(kwargs)
        if bound is not None and len(ids) > bound:      # stock truncates: its call, unchanged
            return None, 'over-truncation-length'
        head.extend(tail)
        return (ids, cut, k, ends[:best + 1], head, best == len(ends) - 1), None

    def tokenize(self, text, kwargs, full):
        """ids for text; full() is the stock call returning ids. Never raises from the splice side."""
        if self.disabled is not None:
            self.fallbacks[self.disabled] += 1
            return full()
        if (set(kwargs) - STOCK_KWARGS or kwargs.get('add_special_tokens') is not False
                or (kwargs.get('max_length') is None and kwargs.get('truncation') not in NO_TRUNCATION + (None,))):
            self.fallbacks['encode-arguments'] += 1      # (truncation to the tokenizer's own model_max_length too)
            return full()
        bound = self.length_bound(kwargs)
        key = self.key(text, kwargs)
        with self.lock:
            candidates = [(eid, self.cache[eid][1:]) for eid in reversed(self.by_key.get(key, ()))]
        t0 = time.perf_counter()
        got, reason, used = None, 'no-entry', None
        best = -1
        for eid, entry in candidates:   # the candidate with the longest join cut
            cut = self._safe_cut(entry, text)
            if cut is not None and cut[0] > best:
                best, used = cut[0], (eid, entry, cut)
        if candidates and used is None:
            reason = 'no-join'
        if used is not None:
            eid, entry, cut = used
            got, reason = self._safe_splice(entry, text, kwargs, cut)
        if got is not None:
            ids, cut, k, prefix_ends, packed, continues = got
            replace = used[0] if continues else None  # the same conversation's next turn replaces its entry
            ms = (time.perf_counter() - t0) * 1000
            self.counts['spliced'] += 1
            self.sums['chars_total'] += len(text)
            self.sums['chars_retokenised'] += len(text) - cut
            self.sums['ids_reused'] += k
            self.sums['ids_total'] += len(ids)
            self.sums['splice_us'] += int(ms * 1000)
            self.splice_ms[_bucket(ms)] += 1
            if self.verify_every and self.counts['spliced'] % self.verify_every == 0 and self.verify_async:
                self._store(key, text, ids, (cut, k, prefix_ends), packed, replace=replace)
                self._submit(self._conversation(key), text, kwargs, packed)
                return ids
            if self.verify_every and self.counts['spliced'] % self.verify_every == 0:
                stock = list(full())
                self.counts['verified'] += 1
                if stock != ids:
                    self.counts['verify_mismatch'] += 1
                    if bound is None or len(stock) < bound:
                        self._store(key, text, stock, None, replace=used[0])   # never splice from it again
                    else:
                        self.counts['store_length_bound'] += 1
                        with self.lock:
                            self._drop(used[0])
                    return stock
            self._store(key, text, ids, (cut, k, prefix_ends), packed, replace=replace)
            return ids
        self.fallbacks[reason] += 1
        t0 = time.perf_counter()
        ids = full()
        ms = (time.perf_counter() - t0) * 1000
        self.counts['full'] += 1
        self.sums['full_us'] += int(ms * 1000)
        self.sums['full_ids'] += len(ids)
        self.full_ms[_bucket(ms)] += 1
        if bound is not None and len(ids) >= bound:         # may be truncated: never a splice source
            self.counts['store_length_bound'] += 1
        else:
            self._store(key, text, ids, None)
        return ids

    # -- off-path verification (night-pack fix 3) -------------------------------------------------------------------
    def _conversation(self, key):
        with self.lock:
            index = self.conversations.get(key)
            if index is None:
                if len(self.conversations) >= 4096:
                    return -1
                index = self.conversations[key] = len(self.conversations)
            return index

    def _submit(self, conversation, text, kwargs, packed):
        """Queue one stock tokenisation to compare with the spliced ids already returned; never blocks the caller."""
        try:
            with self.vcond:
                if len(self.vqueue) >= VERIFY_QUEUE:
                    self.counts['verify_dropped'] += 1
                    return
                self.vqueue.append((conversation, text, dict(kwargs), packed, time.perf_counter()))
                self.counts['verify_submitted'] += 1
                if self.vthread is None:
                    self.vthread = threading.Thread(target=self._verifier, name='flash-s5-ingress-verify', daemon=True)
                    self.vthread.start()
                self.vcond.notify()
        except Exception as exc:
            self.counts['verify_submit_error'] += 1
            _ = exc

    def _verifier(self):
        while True:
            with self.vcond:
                while not self.vqueue:
                    self.vcond.wait()
                job = self.vqueue.popleft()
            self.verify_one(*job)

    def verify_one(self, conversation, text, kwargs, packed, submitted):
        """The stock call on the verifier thread; the first mismatch disables the splice (counted, one line)."""
        try:
            t0 = time.perf_counter()
            stock = self.tokenizer(text, **kwargs)['input_ids']
            ms = (time.perf_counter() - t0) * 1000
            self.vcounts['verified'] += 1
            self.vcounts['verify_us'] += int(ms * 1000)
            self.vcounts['verify_lag_us'] += int((t0 - submitted) * 1e6)
            self.verify_ms[_bucket(ms)] += 1
            if array.array('I', stock) != packed:
                pos = _first_difference(stock, packed)
                self.vcounts['verify_mismatch'] += 1
                if len(self.mismatches) < MISMATCH_KEEP:
                    self.mismatches.append((conversation, pos))
                if self.disabled is None:
                    self.disabled = 'verify-mismatch'
                    self.clear()
                _logger().warning('flash-s5 ingress-verify-mismatch conversation=%d position=%d', conversation, pos)
        except Exception as exc:
            self.vcounts['verify_error'] += 1
            _ = exc

    def drain(self, timeout=5.0):
        """Tests and the snapshot reader: wait until the verifier queue is empty (bounded)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.vcond:
                if not self.vqueue and self.vcounts['verified'] + self.vcounts['verify_error'] >= \
                        self.counts['verify_submitted']:
                    return True
            time.sleep(0.002)
        return False

    def _safe_cut(self, entry, text):
        try:
            return self.cut(entry, text)
        except Exception:
            return None

    def _safe_splice(self, entry, text, kwargs, cut):
        try:
            return self.splice(entry, text, kwargs, cut)
        except Exception as exc:
            return None, 'splice-error:' + type(exc).__name__

    def _store(self, key, text, ids, known, packed=None, replace=None):
        try:
            if known is None:
                ends = self.ends(text, ids)
            else:
                cut, k, prefix_ends = known
                ends = self.ends(text, ids, start_char=cut, start_index=k, prefix=prefix_ends)
            if ends is None:
                self.counts['store_refused'] += 1
                if replace is not None:
                    with self.lock:
                        self._drop(replace)
                return
            if packed is None:
                packed = array.array('I', ids)
            size = len(text) * 4 + len(packed) * packed.itemsize + len(ends) * 16
            with self.lock:
                if replace is not None:
                    self._drop(replace)
                self.serial += 1
                eid = self.serial
                self.cache[eid] = (key, text, packed, ends)
                self.by_key.setdefault(key, []).append(eid)
                self.sizes[eid] = size
                self.bytes += size
                while len(self.by_key[key]) > PER_KEY:
                    self._drop(self.by_key[key][0])
                    self.counts['evicted_per_key'] += 1
                while self.cache and (len(self.cache) > self.max_entries or self.bytes > self.max_bytes):
                    self._drop(next(iter(self.cache)))
                    self.counts['evicted'] += 1
        except Exception:
            self.counts['store_error'] += 1

    def _drop(self, eid):
        entry = self.cache.pop(eid, None)
        if entry is not None:
            self.bytes -= self.sizes.pop(eid, 0)
            ids = self.by_key.get(entry[0])
            if ids is not None:
                ids.remove(eid)
                if not ids:
                    del self.by_key[entry[0]]

    def clear(self):
        with self.lock:
            n = len(self.cache)
            self.cache.clear()
            self.by_key.clear()
            self.sizes.clear()
            self.bytes = 0
        if n:
            self.counts['cleared_entries'] += n

    def snapshot(self):
        with self.lock:
            entries, size = len(self.cache), self.bytes
        with self.vcond:
            pending = len(self.vqueue)
        counts = dict(self.counts)
        for name in ('verified', 'verify_mismatch'):        # inline and off-path verification add up
            if self.vcounts.get(name):
                counts[name] = counts.get(name, 0) + self.vcounts[name]
        return dict(schema='s5-ingress-incremental/1', disabled=self.disabled, joins=sorted(self.joins),
                    entries=entries, bytes=size, max_entries=self.max_entries, max_bytes=self.max_bytes,
                    verify_every=self.verify_every, verify_async=self.verify_async, counts=counts,
                    fallbacks=dict(self.fallbacks), sums=dict(self.sums), splice_ms=dict(self.splice_ms),
                    full_ms=dict(self.full_ms),
                    verify=dict(pending=pending, verified=self.vcounts.get('verified', 0),
                                errors=self.vcounts.get('verify_error', 0), verify_us=self.vcounts.get('verify_us', 0),
                                lag_us=self.vcounts.get('verify_lag_us', 0), ms=dict(self.verify_ms),
                                mismatches=[list(m) for m in self.mismatches]))


class Binding:
    """The API process's mirror of the engine selection (auto-arm released = an engine arm)."""

    def __init__(self, on_select=None):
        self.selected = False
        self.on_select = on_select
        self.flips = collections.Counter()

    def _set(self, value, why):
        if value and not self.selected and self.on_select is not None:
            self.on_select()
        if value != self.selected:
            self.flips[('on:' if value else 'off:') + why] += 1
        self.selected = value

    def observe(self, action, result):
        if type(result) is not dict:
            return
        if action == 'release_auto_arm':
            if result.get('released') is True:
                self._set(True, 'release')
            return
        if action == 'status':
            auto = result.get('auto_arm')
            if type(auto) is dict and type(auto.get('manual_owned')) is bool:
                self._set(not auto['manual_owned'], 'status')
            return
        if action in SESSION_NO_CHANGE:
            return
        self._set(False, str(action))

    def snapshot(self):
        return dict(selected=self.selected, flips=dict(self.flips))


class Ingress:
    """The installed API-process state: one renderer wrapped, its splicer and its binding."""

    def __init__(self, renderer, env=None):
        every, verify_async = verify_settings(env)
        self.renderer = renderer
        self.splicer = Splicer(renderer.get_tokenizer(), verify_every=every, verify_async=verify_async)
        self.binding = Binding(on_select=self.splicer.clear)
        self.counts = collections.Counter()
        self.timing = collections.Counter()      # night-pack fix 3: submit -> done around the async wrapper, by class
        self.original = renderer._tokenize_prompt

    def tokenize_prompt(self, prompt, params):
        """The renderer's _tokenize_prompt, timed on the executor thread (night-pack fix 3)."""
        t0 = time.perf_counter()
        selected = self.binding.selected
        try:
            return self._tokenize_prompt(prompt, params)
        finally:
            self._time('run', 'n3' if selected else 'stock', t0)

    def _time(self, what, cls, t0):
        try:
            self.timing['%s_%s_n' % (what, cls)] += 1
            self.timing['%s_%s_us' % (what, cls)] += int((time.perf_counter() - t0) * 1e6)
        except Exception:
            pass

    def async_wrapper(self, inner):
        """submit -> done of one async tokenisation (the executor queue, the run and the loop pickup), by class."""
        async def tokenize_prompt_async(*args, **kwargs):
            t0 = time.perf_counter()
            selected = self.binding.selected
            try:
                return await inner(*args, **kwargs)
            finally:
                self._time('async', 'n3' if selected else 'stock', t0)
        return tokenize_prompt_async

    def _tokenize_prompt(self, prompt, params):
        """The stock call unless selected and the prompt is a plain text prompt."""
        if not self.binding.selected:
            return self.original(prompt, params)
        r = self.renderer
        if (type(prompt) is not dict or type(prompt.get('prompt')) is not str
                or prompt.get('multi_modal_data') or prompt.get('multi_modal_uuids') or r._wants_offsets(prompt, params)):
            self.counts['stock_shape'] += 1
            return self.original(prompt, params)
        kwargs = params.get_encode_kwargs()
        stock = []

        def full():
            out = self.original(prompt, params)
            stock.append(out)
            return out['prompt_token_ids']
        ids = self.splicer.tokenize(prompt['prompt'], kwargs, full)
        if stock and stock[-1]['prompt_token_ids'] is ids:
            return stock[-1]
        if stock and list(stock[-1]['prompt_token_ids']) == ids:
            return stock[-1]
        self.counts['spliced_prompts'] += 1
        return r._build_tokens_prompt(ids, prompt)

    def snapshot(self):
        return dict(self.splicer.snapshot(), process='api', binding=self.binding.snapshot(), wrapper=dict(self.counts),
                    timing=dict(self.timing))


def install(engine_client, env=None):
    """Wrap the engine client's renderer (flag 1 only); returns the Ingress or None."""
    if not enabled(env):
        return None
    renderer = getattr(engine_client, 'renderer', None)
    if renderer is None or getattr(renderer, '_flash_s5_ingress', None) is not None:
        return getattr(renderer, '_flash_s5_ingress', None)
    from vllm.utils.async_utils import make_async
    ingress = Ingress(renderer, env)
    saved = {name: renderer.__dict__[name] for name in ('_tokenize_prompt', '_tokenize_prompt_async')
             if name in getattr(renderer, '__dict__', {})}
    try:
        renderer._tokenize_prompt = ingress.tokenize_prompt
        renderer._tokenize_prompt_async = ingress.async_wrapper(
            make_async(ingress.tokenize_prompt, executor=renderer._executor))
        renderer._flash_s5_ingress = ingress
    except BaseException:           # night-pack fix 2: never leave a half-wrapped renderer
        for name in ('_tokenize_prompt', '_tokenize_prompt_async', '_flash_s5_ingress'):
            if name in saved:
                setattr(renderer, name, saved[name])
            elif name in getattr(renderer, '__dict__', {}):
                delattr(renderer, name)
        raise
    return ingress


# -- engine core: the append common-prefix counter (log-only) -------------------------------------------------------
DELTA_BUCKETS = ((-1024, '<=-1024'), (-64, '-1023..-64'), (-8, '-63..-8'), (-1, '-7..-1'), (0, '0'))


class PrefixCounter:
    """Per conversation (cache_salt) it keeps only references to the previous request's block-hash list and output-token
    list (both grow until it finishes) and a copy of its last TAIL_BLOCKS blocks of prompt tokens: a few KB, never the
    whole prompt. The match is found by bisection over the chained hashes (full blocks of prompt + output), then token by
    token from there when those tokens are held (the prompt tail or the output); a divergence deeper in the prompt is
    resolved to the block only (counted block_resolved)."""
    TAIL_BLOCKS = 2

    def __init__(self, unit, max_entries=COUNTER_ENTRIES):
        self.unit, self.max_entries = unit, max_entries
        self.prev = collections.OrderedDict()
        self.evicted = collections.OrderedDict()   # salts evicted from prev (bounded): their next append is not 'first'
        self.counts = collections.Counter()
        self.delta = collections.Counter()
        self.sums = collections.Counter()
        self.reply = collections.Counter()          # night-pack fix 4: reply-aligned match class
        self.reply_offset = collections.Counter()   # offset of the best reply alignment from the previous prompt end
        self.windows = []                           # the id window around the first divergence, first WINDOW_APPENDS

    def lcp(self, new_ids, new_hashes, prev):
        """(common prefix length, exact) against prev = (prompt len, prompt tail, outputs, hashes)."""
        p, tail, outputs, old_hashes = prev
        total = p + len(outputs)
        lo, hi = 0, min(len(new_hashes), len(old_hashes))
        while lo < hi:                  # leading equal chained hashes: equality at a block implies all earlier ones
            mid = (lo + hi + 1) // 2
            if new_hashes[mid - 1] == old_hashes[mid - 1]:
                lo = mid
            else:
                hi = mid - 1
        i = lo * self.unit
        held = p - len(tail)
        if i < held:
            return i, False
        n = min(len(new_ids), total)
        while i < n:
            old = tail[i - held] if i < p else outputs[i - p]
            if new_ids[i] != old:
                break
            i += 1
        return i, True

    def record(self, request):
        try:
            salt = getattr(request, 'cache_salt', None)
            if not salt:
                self.counts['no_salt'] += 1
                return
            if not self.unit:
                self.counts['no_hash_unit'] += 1
                return
            prev = self.prev.pop(salt, None)
            p_new = request.num_prompt_tokens
            keep = self.TAIL_BLOCKS * self.unit
            self.prev[salt] = (p_new, list(request.prompt_token_ids[max(0, p_new - keep):p_new]),
                               request.output_token_ids, request.block_hashes)
            while len(self.prev) > self.max_entries:
                gone, _ = self.prev.popitem(last=False)
                self.counts['evicted'] += 1
                self.evicted[gone] = True
                while len(self.evicted) > EVICTED_KEEP:
                    self.evicted.popitem(last=False)
            if prev is None:
                if self.evicted.pop(salt, None):
                    self.counts['after_eviction'] += 1
                else:
                    self.counts['first'] += 1
                return
            p, t = prev[0], prev[0] + len(prev[2])
            k, exact = self.lcp(request.prompt_token_ids, request.block_hashes, prev)
            d = k - p
            self.counts['compared'] += 1
            if not exact:
                self.counts['block_resolved'] += 1
            for bound, name in DELTA_BUCKETS:
                if d <= bound:
                    self.delta[name] += 1
                    break
            else:
                self.delta['into_reply' if k < t - 1 else ('reply_but_last' if k < t else 'whole_reply')] += 1
            if k >= t:
                self.counts['through_reply'] += 1
            if k >= t - 1:
                self.counts['through_reply_but_last'] += 1
            self.sums['lcp'] += k
            self.sums['prev_prompt'] += p
            self.sums['prev_output'] += t - p
            self.sums['new_prompt'] += len(request.prompt_token_ids)
            self._reply_aligned(request.prompt_token_ids, p, prev[2])
            if len(self.windows) < WINDOW_APPENDS:
                self.windows.append(self._window(request.prompt_token_ids, k, exact, prev))
        except Exception:
            self.counts['errors'] += 1

    def _reply_aligned(self, new_ids, p, outputs):
        """night-pack fix 4: the previous reply's ids searched in the new prompt at offsets -R..+R from the previous
        prompt end; the longest leading run (ties: the smallest |offset|) is the reply-aligned match."""
        n_out = len(outputs)
        if not n_out:
            self.reply['no-output'] += 1
            return
        best, best_o = 0, None
        for o in sorted(range(-REPLY_OFFSETS, REPLY_OFFSETS + 1), key=abs):
            a = p + o
            if a < 0 or a >= len(new_ids):
                continue
            run, limit = 0, min(n_out, len(new_ids) - a)
            while run < limit and new_ids[a + run] == outputs[run]:
                run += 1
            if run > best:
                best, best_o = run, o
                if run == n_out:
                    break
        name = ('whole' if best == n_out else 'but_last' if best == n_out - 1 else 'into' if best else 'none')
        self.reply[name] += 1
        if best_o is not None:
            self.reply_offset[str(best_o)] += 1
        self.sums['reply_run'] += best
        self.sums['reply_len'] += n_out

    def _window(self, new_ids, k, exact, prev):
        """night-pack fix 4: ids around the first divergence k (new prompt and previous prompt + reply, where held)."""
        p, tail, outputs = prev[0], prev[1], prev[2]
        held = p - len(tail)
        lo, hi = max(0, k - WINDOW_HALF), k + WINDOW_HALF
        old = []
        for i in range(lo, hi):
            if i < held:
                old.append(None)
            elif i < p:
                old.append(tail[i - held])
            elif i - p < len(outputs):
                old.append(outputs[i - p])
        return dict(k=k, d=k - p, exact=exact, prev_prompt=p, prev_output=len(outputs), start=lo,
                    new=list(new_ids[lo:hi]), old=old)

    def snapshot(self):
        return dict(schema='s5-ingress-prefix/1', unit=self.unit, entries=len(self.prev), counts=dict(self.counts),
                    delta_vs_prev_prompt_end=dict(self.delta), sums=dict(self.sums), max_entries=self.max_entries,
                    reply_aligned=dict(self.reply), reply_offset=dict(self.reply_offset),
                    divergence_windows=[dict(w) for w in self.windows],
                    basis='engine-lifetime monotonic counts while the auto-arm is released; subtract matching snapshots')
