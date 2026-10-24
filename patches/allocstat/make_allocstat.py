#!/usr/bin/env python3
"""Add PyTorch allocator counters to the QSA stage-timing log line (QSA_STAGE_TIMING=1): alloc retries (the allocator
freed its cache and retried -- a device-wide cudaFree + sync), OOMs, reserved and allocated GiB, peak allocated since
the previous line. Host-side counters only (torch.cuda.memory_stats), no GPU sync. Question (24 Oct): does S3's VRAM
use cause allocator retries under the 8 x 153k + 200k load (1.3-2.1 s gaps, no TTFT gain)?
usage: make_allocstat.py ops_qsa.py.in ops_qsa.py.out
"""
import difflib, sys
src = orig = open(sys.argv[1]).read()
old = '''            self.log.info("QSA stage timing (%.1f s, per call): %s | dropped %d", now - self.last, " ; ".join(parts),
                          self.dropped)'''
new = '''            m = torch.cuda.memory_stats()
            torch.cuda.reset_peak_memory_stats()
            self.log.info("QSA stage timing (%.1f s, per call): %s | dropped %d | alloc retries %d ooms %d reserved "
                          "%.2fG allocated %.2fG peak %.2fG", now - self.last, " ; ".join(parts), self.dropped,
                          m.get("num_alloc_retries", -1), m.get("num_ooms", -1),
                          m.get("reserved_bytes.all.current", 0) / 2**30, m.get("allocated_bytes.all.current", 0) / 2**30,
                          m.get("allocated_bytes.all.peak", 0) / 2**30)'''
assert src.count(old) == 1
src = src.replace(old, new)
open(sys.argv[2], 'w').write(src)
sys.stdout.writelines(difflib.unified_diff(orig.splitlines(True), src.splitlines(True), 'in', 'out'))
