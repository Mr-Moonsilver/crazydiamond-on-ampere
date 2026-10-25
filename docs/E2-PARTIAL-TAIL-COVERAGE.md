# Partial-tail coverage diagnostic (reviewed; unarmed)

q16j's feature-off control completed 80 answers (zero errors), but only 15
reported partial hits; the unchanged leak oracle returned INCONCLUSIVE. The
pressure streams each completed 15 requests and the lifetime grader observed
12514 evictions / 10901 state evictions. q16c had 44/80 hits. These observations
do not prove which allocation or scheduler event lost a particular cached tail.
Neither a leak nor a grouping-specific root cause is established.

The worker draft was reviewed and repaired. The new queue is deliberately a
single **feature-off control engine** at the existing 1.5 GB pool, not another
complete E2 gate. It reuses overlay-v16e2f. No runtime, pool, allocator, grouping,
model, oracle or threshold change. Native E3 work proceeds independently.

## Declared experiment

1. Qualification: two fresh-salted 64k pressure streams, max generation128;
   10 sessions ×2 rounds ×2 answers =40 answers. The deterministic producer
   prompt has the original short tail after boundary7568; generation200–399.
   Fixed one-token refresh before each consumer; all 40 outcomes retained,
   cancellation fraction0.3. Cold controls after both consumers. Fresh namespace
   per session preserves session isolation; it does not exercise shared cache
   blocks across namespaces. The original stress phase below does.
2. Stress: original `leak_partial_tail.py` requests/oracle, 10 sessions ×4 rounds,
   2×200k pressure, cancellation0.3, GEN200, NEED_PARTIAL1. A metadata sink
   adapter replaces only console printing and terminal JSON serialization;
   source AST inverse is tested. No change to requests, retries or leakgrade.
3. Both phases require every pressure stream to complete and its first/last
   output timestamps to bracket the workload; the existing lifetime grader
   requires state eviction. Readiness, driver, wait, artifact and identity
   failures remain failures. Both workloads and all waits are bounded.
4. Each subphase has its own verdict. The combined verdict **never promotes an
   INCONCLUSIVE stress result to PASS**. A lower-load PASS only qualifies that
   exposure. Raw memory reserve FAIL remains recorded under the operator's
   existing diagnostic-continuation policy. Engine ends STOPPED, launcher and
   defaults restored; no canary waiter is used.

The smaller pressure size is a hypothesis to test, not a guaranteed solution.
If it creates no state evictions, or still misses the coverage threshold, the
result is INCONCLUSIVE. Do not silently lower pressure or retry until it passes.
The qualification retains leakgrade's 80% threshold and additionally requires
80% of attempts to hit the intended partial source page; a different earlier
partial hit is not enough to claim the targeted copy was exercised.

## Evidence and limits

Artifacts carry case keys, prompt/generated lengths, cached/refresh hit counts,
construction flags, first/settled oracle verdicts, transition counts, identity,
pressure statistics and timing. No identifiers, prompts, token arrays or answer
text. Errors carry type/category only. All rounds' identifiers are drawn before
any request so future-round identifiers are included in in-memory foreign checks.

`eligible_cases` means the constructed shape can expose a partial boundary; it
is not proof of registration. Transition labels describe observed cached lengths,
not a physical eviction diagnosis. Refresh is not proof a source survived. A
client disconnect count does not prove cancellation at a GPU copy boundary.
First/last token bracketing does not prove uninterrupted device occupancy; retain
pressure2 occupancy summaries and actual pool/state-eviction observations.

The workload deadline bounds new calls and their I/O timeouts. The queue's outer
process timeout is the hard wall bound, including slow-trickle HTTP responses.
The same timeout covers retries; no best-of selection and no dropped misses.
Payload-free metadata cannot independently rejudge answer semantics; the trusted
unchanged leakgrade adjudicates in memory, and the grader checks stored verdicts,
case coverage/uniqueness, raw lengths and consistency. CPU fakes cannot prove live
cache coverage or safety. No new GPU evidence exists yet.

## Run request for the owner

After merge, build a NEW harness65 with `tools/build_harness_bundle.py OUT` and
manifest-verify it on the CT; do not overwrite an existing bundle. Reuse the
manifest-verified overlay-v16e2f, generated launcher SHA2630f86c... and production
found-launcher SHA45cc5d81...; no generator run or runtime change.

Install the bundle's NEW `runs/q16k_declare.sh` and `runs/q16k.sh` as new wrappers,
write the new bundle identifier to `/tmp/harness65.sha`, and create an owned
chain: stopped/idle, manifest/wrapper equality, no q16k artifacts or other queue,
no inherited E2_KV_BYTES/E2_OVERLAY override, then declarations → gate only on
rc0. The four declarations are retained for compatible source preflight, though
only control starts. Do not start a hub waiter.

```bash
bash /tmp/q16k_declare.sh > /home/<service-user>/s0/q16k-declare.log 2>&1
# Only on Q16K-DECLARE DONE rc=0:
bash /tmp/q16k.sh > /home/<service-user>/s0/q16k.log 2>&1
```

Queue: `runs/q16_e2_serial_v7.draft.sh`; settings unchanged. Qualification
parameters explicitly default to64000 tokens,2400s pressure ceiling,900s driver
budget with1500s outer timeout. Stress keeps200000 tokens/1800s pressure ceiling,
1800s leak timeout. Completion waits900s; outer queue18000s includes startup and
source hashing. Expected duration is not yet measured; do not promise a short
slot from CPU timing. The long bounds are ceilings, not sleeps.

Record qualification driver/grade/lifetime, original stress oracle/lifetime,
combined verdict, identities before/after, final error scan, memory/peaks and
all cleanup steps. Failure in qualification stops before stress; terminal phase
identity/error/memory evidence is still attempted. No overall PASS means release
qualification. Integration into a future E2/E3 gate follows measured evidence,
not merely this source merge.

## Offline review commands

```bash
python3 harness/test_partial_tail_cov.py
bash -n harness/runs/q16_e2_serial_v7.draft.sh
bash -n harness/runs/q16k_declare.sh
bash -n harness/runs/q16k.sh
(cd harness && python3 selftest.py)
```

The old leak/pressure tools, all earlier queue files, and patches/ stay unchanged.
