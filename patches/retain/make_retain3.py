#!/usr/bin/env python3
"""S3 accounting + byte check (24 Oct review): time what the stage timer did not see, verify the retained pages.

The v2 timer began inside the staged function, after qsa.py had already (1) synced the request id + block ids to the
host (`.tolist()`, which blocks the CPU until the GPU reaches that point) and (2) filled the retained buffer. Logs
showing `moved=1.0p copy=0.00` therefore hid both costs. Now both run inside ops_qsa (qsa_retained_history_from):
  meta_host  host wall ms of the id/block-id transfer (the launch-ahead the CPU loses)
  fill       GPU ms of the retained-buffer refresh (CUDA events from the timer's pool)
appended per bin to the timing line as `ret=<calls> fill=<ms> meta_host=<ms>`.
QSA_RETAIN_VERIFY=1 (test only, syncs): after every refresh, compare the retained view byte for byte against a fresh
gather of the same pages from host; counts `verify n=<checked> bad=<mismatches>` on the timing line.
Codex review (reviews/codex-20261024.md) fixes in the retention class itself:
  * an allocation failure releases every layer's buffer (v2 kept the earlier layers' VRAM while all of them fell
    back to host staging);
  * rollback / rewrite under the same request id: a shorter sequence than last time re-arms that layer (v2 could
    skip a rewritten page that was complete when copied), and only pages complete now AND when copied are reused;
  * the first active use per process logs once that retention is live (it silently needs the row cache's request
    ids: QSA_ROWCACHE_SLOTS=0 disables it).
Unset, the patched files behave exactly like their inputs.
usage: make_retain3.py ops_qsa.py.v7 ops_qsa.py.v8 qsa.py.v4 qsa.py.v5
"""
import difflib, sys

ops_in, ops_out, qsa_in, qsa_out = sys.argv[1:5]


def sub(text, old, new, count=1):
    assert text.count(old) == count, (old[:80], text.count(old))
    return text.replace(old, new)


src = orig = open(ops_in).read()
src = sub(src, """        old_uid, old_ids, complete = self.state.get(key, (None, (), 0))
        if self.failed or uid < 1:
            self.state.pop(key, None)
            return None, 0
        if old_uid != uid:
            self.state[key] = (uid, (), 0)  # first chunk seen: arm only
            return None, 0
""", """        old_uid, old_ids, complete, old_len = self.state.get(key, (None, (), 0, 0))
        if self.failed or uid < 1:
            self.state.pop(key, None)
            return None, 0
        if old_uid != uid or seq_len < old_len:
            self.state[key] = (uid, (), 0, seq_len)  # first chunk seen, or a rollback: arm only
            return None, 0
""")
src = sub(src, """            except torch.OutOfMemoryError:
                self.failed = True
                self.state.clear()
                return None, 0
            self.buf[key] = buf
""", """            except torch.OutOfMemoryError:
                self.failed = True
                self.state.clear()
                self.buf.clear()  # release every layer's buffer: all of them fall back to host staging now
                import logging
                logging.getLogger("vllm.qsa.retain").warning("QSA retained history disabled: buffer allocation failed")
                return None, 0
            if not self.buf:
                import logging
                logging.getLogger("vllm.qsa.retain").info("QSA retained history active: %d pages per layer", self.cap)
            self.buf[key] = buf
""")
src = sub(src, """        first, lim = 0, min(complete, keep, len(old_ids))
""", """        first, lim = 0, min(complete, keep, len(old_ids), seq_len // kv_cache.shape[2])
""")
src = sub(src, """        self.state[key] = (uid, tuple(ids[:keep]), min(keep, seq_len // kv_cache.shape[2]))
""", """        self.state[key] = (uid, tuple(ids[:keep]), min(keep, seq_len // kv_cache.shape[2]), seq_len)
""")
src = sub(src, """        self.last = time.monotonic()
""", """        self.last = time.monotonic()
        self.pre_retain = None  # (meta host ms, fill start event, fill end event) for the next staged call
        self.verify = [0, 0]    # retained pages checked, mismatches (QSA_RETAIN_VERIFY=1)
""")
src = sub(src, """            a = self.agg.setdefault(k, [0, 0, 0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0])
""", """            a = self.agg.setdefault(k, [0, 0, 0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0])
            if "ret" in t:
                host_ms, f0, f1 = t["ret"]
                a[11] += 1; a[12] += f0.elapsed_time(f1); a[13] += host_ms
                self.free.extend((f0, f1))
""")
src = sub(src, """                n, rows, pages, sp, cov, moved, pre, copy, attn, span, ncov = self.agg[k]
""", """                n, rows, pages, sp, cov, moved, pre, copy, attn, span, ncov, nret, fill, mhost = self.agg[k]
""")
src = sub(src, """                             f"pre={pre / n:.2f} copy={copy / n:.2f} attn={attn / n:.2f} span={span / n:.2f} ms")
""", """                             f"pre={pre / n:.2f} copy={copy / n:.2f} attn={attn / n:.2f} span={span / n:.2f} ms"
                             + (f" ret={nret} fill={fill / nret:.2f} meta_host={mhost / nret:.2f} ms" if nret else ""))
            if self.verify[0]:
                parts.append(f"verify n={self.verify[0]} bad={self.verify[1]}")
""")
src = sub(src, """def qsa_retained_history(kv_cache, block_table_row, num_pages, seq_len, uid, ids):
    if _RETAINED is None:
        return None, 0
    return _RETAINED.get(kv_cache, block_table_row, num_pages, seq_len, uid, ids)
""", """def qsa_retained_history(kv_cache, block_table_row, num_pages, seq_len, uid, ids):
    if _RETAINED is None:
        return None, 0
    return _RETAINED.get(kv_cache, block_table_row, num_pages, seq_len, uid, ids)


_RETAIN_VERIFY = os.environ.get("QSA_RETAIN_VERIFY", "0") == "1"


def qsa_retained_history_from(kv_cache, table, num_pages, seq_len, uids, best):
    \"\"\"qsa_retained_history with the id/block-id host transfer done here, timed (see make_retain3.py).\"\"\"
    if _RETAINED is None or _RETAINED.failed:
        return None, 0  # v3.1: no metadata transfer once retention is off for the process
    t0 = time.perf_counter()
    meta = torch.cat([uids[best:best + 1].to(table.dtype), table[:num_pages]]).tolist()
    host_ms = (time.perf_counter() - t0) * 1000
    timed = _STAGE_TIMER is not None and not torch.cuda.is_current_stream_capturing()
    f0 = _STAGE_TIMER.mark() if timed else None
    view, copied = _RETAINED.get(kv_cache, table, num_pages, seq_len, int(meta[0]), meta[1:])
    if timed:
        _STAGE_TIMER.pre_retain = (host_ms, f0, _STAGE_TIMER.mark()) if view is not None else None
        if view is None:
            _STAGE_TIMER.free.append(f0)
    if _RETAIN_VERIFY and view is not None:
        # kv_cache is the host pages' device view (as in get()): a fresh gather of the current host content; a
        # mismatch means a reused page changed after it was retained (the identity rules failed)
        fresh = torch.index_select(kv_cache, 0, table[: view.shape[0]].to(torch.int64))
        ok = torch.equal(fresh, view)
        if _STAGE_TIMER is not None:
            _STAGE_TIMER.verify[0] += view.shape[0]
            _STAGE_TIMER.verify[1] += 0 if ok else 1
        if not ok:
            from vllm.logger import init_logger
            init_logger("vllm.qsa.retain").error("QSA retained history MISMATCH: %d pages, request id %d",
                                                 view.shape[0], int(meta[0]))
            _RETAINED.state.pop(kv_cache.data_ptr(), None)
            return None, 0  # v3.1: never attend over a suspect view; this call stages from host as before
    return view, copied
""")
src = sub(src, """    _t = _STAGE_TIMER.begin(num_rows, num_pages) if _STAGE_TIMER is not None else None
""", """    _t = _STAGE_TIMER.begin(num_rows, num_pages) if _STAGE_TIMER is not None else None
    if _STAGE_TIMER is not None and _STAGE_TIMER.pre_retain is not None:
        if _t is not None:
            _t["ret"] = _STAGE_TIMER.pre_retain
        _STAGE_TIMER.pre_retain = None
""")
open(ops_out, 'w').write(src)
sys.stdout.writelines(difflib.unified_diff(orig.splitlines(True), src.splitlines(True), 'ops_qsa.py.v7', 'ops_qsa.py.v8'))

src = orig = open(qsa_in).read()
src = sub(src, """    from .ops.qsa import qsa_get_staging_arena, qsa_retained_history, qsa_sparse_paged_attention_staged
""", """    from .ops.qsa import qsa_get_staging_arena, qsa_retained_history_from, qsa_sparse_paged_attention_staged
""")
src = sub(src, """            table = attn_metadata.block_table[best]
            meta = torch.cat([uids[best:best + 1].to(table.dtype), table[: max_logical_page + 1]]).tolist()
            retained, copied = qsa_retained_history(
                kv_cache, table, max_logical_page + 1, int(lengths[best]), int(meta[0]), meta[1:])
""", """            retained, copied = qsa_retained_history_from(
                kv_cache, attn_metadata.block_table[best], max_logical_page + 1, int(lengths[best]), uids, best)
""")
open(qsa_out, 'w').write(src)
sys.stdout.writelines(difflib.unified_diff(orig.splitlines(True), src.splitlines(True), 'qsa.py.v4', 'qsa.py.v5'))
