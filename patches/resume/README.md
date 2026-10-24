# Resumable prefill: offline prototype

Read [the design and test plan](../../reviews/design-resumable-prefill.md) before using this code.

**Not deployable.** `FLASH_PREFILL_SEGMENT_LAYERS=0` preserves the existing model forward and runner execute/sample bodies. A positive value fails at runner construction because the scheduler/KV/PP adapters do not exist yet. The model continuation is for a dedicated harness with private metadata, state ownership and completion fences; it is not a serving entry point.

Generate into a **new** directory (existing output directories are rejected):

```bash
PYTHONDONTWRITEBYTECODE=1 python3 patches/resume/make_resume.py \
  reference/vllm-src <hub-dir>/agents/flash-next/tmp/resume-new-candidate
```

The generator validates unique anchors and Python syntax before writing `model.py.v1`, `model_runner.py.v1` and a SHA256 manifest. The checked-in `generated/` files reproduce from the source hashes in that manifest. No reference files are modified.

Run the CPU-only tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 harness/test_resume_prefill.py
TMPDIR=<hub-dir>/agents/flash-next/tmp PYTHONDONTWRITEBYTECODE=1 \
  python3 harness/selftest.py \
  <hub-dir>/agents/flash-next/investigations/flash-current-review-20261024T0226Z/offline-checks/canary-fixture
```

The first command uses fake tensors with the actual generated Python methods; it does not prove CUDA numerics or leak freedom. `protocol.py` models fences and transactions, with callbacks standing in for the real worker and cache adapters. The second command includes the repository's existing localhost mock servers. Test artifacts are retained under the Flash-Next home.
