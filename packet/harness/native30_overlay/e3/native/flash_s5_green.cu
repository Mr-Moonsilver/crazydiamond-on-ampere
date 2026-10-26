// Flash-Next E3 green-context helper. Derived from harness/s5_green.cu (the S5
// slice shim); built against the image's CUDA headers on the box by the overlay
// generation step, never on the hub. Exports a small C ABI that
// patches/s5_engine/green_context.py binds with ctypes. The ABI version below
// is checked at load; bump it with any signature change.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>
#include <vector>

#define FLASH_S5_GREEN_ABI 1

struct Pair { CUgreenCtx d{}, p{}; CUstream ds{}, ps{}; int dc{}, pc{}; std::vector<CUstream> aux; };

extern "C" int s5_abi(void) { return FLASH_S5_GREEN_ABI; }

// Split the device's SMs: `requested` for decode, the remainder for prefill.
// Two green contexts, one non-blocking stream each. On any failure nothing is
// leaked and a CUresult is returned.
extern "C" int s5_create(int dev, unsigned requested, void **handle,
                          uint64_t *ds, uint64_t *ps, int *dc, int *pc,
                          uint64_t *dctx, uint64_t *pctx) {
  CUdevResource whole{}, group{}, rest{};
  unsigned n = 1; CUresult r;
  if ((r = cuDeviceGetDevResource(dev, &whole, CU_DEV_RESOURCE_TYPE_SM))) return r;
  if ((r = cuDevSmResourceSplitByCount(&group, &n, &whole, &rest, 0, requested))) return r;
  if (n != 1 || group.sm.smCount < requested || !rest.sm.smCount ||
      group.sm.smCount + rest.sm.smCount != whole.sm.smCount) return CUDA_ERROR_INVALID_VALUE;
  Pair *p = new Pair;
  CUdevResourceDesc dd{}, pd{};
  if ((r = cuDevResourceGenerateDesc(&dd, &group, 1))) { delete p; return r; }
  if ((r = cuDevResourceGenerateDesc(&pd, &rest, 1))) { delete p; return r; }
  if ((r = cuGreenCtxCreate(&p->d, dd, dev, CU_GREEN_CTX_DEFAULT_STREAM))) { delete p; return r; }
  if ((r = cuGreenCtxCreate(&p->p, pd, dev, CU_GREEN_CTX_DEFAULT_STREAM))) {
    cuGreenCtxDestroy(p->d); delete p; return r;
  }
  if ((r = cuGreenCtxStreamCreate(&p->ds, p->d, CU_STREAM_NON_BLOCKING, 0)) ||
      (r = cuGreenCtxStreamCreate(&p->ps, p->p, CU_STREAM_NON_BLOCKING, 0))) {
    if (p->ds) cuStreamDestroy(p->ds);
    cuGreenCtxDestroy(p->p); cuGreenCtxDestroy(p->d); delete p; return r;
  }
  *dc = group.sm.smCount; *pc = rest.sm.smCount;
  *ds = (uint64_t)p->ds; *ps = (uint64_t)p->ps;
  *dctx = (uint64_t)p->d; *pctx = (uint64_t)p->p;
  *handle = p; return 0;
}

// Additive ABI-1 extension: bounded streams on the EXISTING decode context.
// Handles stay private. A stream is destroyed only through its owning Pair.
extern "C" int s5_decode_aux_create(void *handle, uint64_t *out) {
  if (!handle || !out) return CUDA_ERROR_INVALID_VALUE;
  Pair *p = static_cast<Pair*>(handle);
  if (p->aux.size() >= 8) return CUDA_ERROR_OUT_OF_MEMORY;
  CUstream stream{};
  CUresult r = cuGreenCtxStreamCreate(&stream, p->d, CU_STREAM_NON_BLOCKING, 0);
  if (r) return r;
  try { p->aux.push_back(stream); }
  catch (...) { cuStreamDestroy(stream); return CUDA_ERROR_OUT_OF_MEMORY; }
  *out = reinterpret_cast<uint64_t>(stream); return CUDA_SUCCESS;
}
extern "C" int s5_decode_stream_check(void *handle, uint64_t token) {
  if (!handle || !token) return CUDA_ERROR_INVALID_VALUE;
  Pair *p = static_cast<Pair*>(handle);
  CUcontext primary{}; CUgreenCtx green{};
  CUresult r = cuStreamGetCtx_v2(reinterpret_cast<CUstream>(token), &primary, &green);
  if (r) return r;
  return primary && green == p->d ? CUDA_SUCCESS : CUDA_ERROR_INVALID_CONTEXT;
}
extern "C" int s5_decode_aux_destroy(void *handle, uint64_t token) {
  if (!handle || !token) return CUDA_ERROR_INVALID_VALUE;
  Pair *p = static_cast<Pair*>(handle);
  CUstream stream = reinterpret_cast<CUstream>(token);
  auto it = std::find(p->aux.begin(), p->aux.end(), stream);
  if (it == p->aux.end()) return CUDA_ERROR_INVALID_VALUE;
  CUresult r = cuStreamSynchronize(stream); if (r) return r;
  r = cuStreamDestroy(stream); if (r) return r;
  p->aux.erase(it); return CUDA_SUCCESS;
}

// Teardown order: drain both streams, destroy streams, destroy contexts.
extern "C" int s5_close(void *handle) {
  Pair *p = (Pair*)handle;
  if (!p || !p->aux.empty()) return CUDA_ERROR_INVALID_VALUE;
  CUresult r = cuStreamSynchronize(p->ds); if (r) return r;
  r = cuStreamSynchronize(p->ps); if (r) return r;
  r = cuStreamDestroy(p->ds); if (r) return r;
  r = cuStreamDestroy(p->ps); if (r) return r;
  r = cuGreenCtxDestroy(p->d); if (r) return r;
  r = cuGreenCtxDestroy(p->p); if (r) return r;
  delete p; return 0;
}

// SM membership probe: each block records the SM it ran on after spinning
// `cycles`, so a stream's partition can be measured rather than assumed.
__global__ void sm_probe(int *out, unsigned long long cycles) {
  unsigned sm; asm("mov.u32 %0, %%smid;" : "=r"(sm));
  unsigned long long begin = clock64();
  while (clock64() - begin < cycles) asm volatile("");
  out[blockIdx.x] = sm;
}
extern "C" int s5_probe(uint64_t ptr, int blocks, uint64_t stream, uint64_t cycles) {
  sm_probe<<<blocks, 1, 0, (cudaStream_t)stream>>>((int*)ptr, cycles);
  return (int)cudaPeekAtLastError();
}

extern "C" int s5_pci_bus_id(int dev, char *out, int len) {
  return (int)cuDeviceGetPCIBusId(out, len, dev);
}

extern "C" int s5_runtime(int *runtime, int *driver, int *device) {
  cudaError_t r = cudaRuntimeGetVersion(runtime); if (r) return r;
  r = cudaDriverGetVersion(driver); if (r) return r;
  return (int)cudaGetDevice(device);
}
