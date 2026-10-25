// Diagnostic interposer, NOT an engine-wide device-property override.
// Preload before Python. Enabled only by a thread-local guard around the exact
// vLLM moe_wna16_marlin_gemm call. No GPU work or synchronization here.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdint.h>
#include <algorithm>

struct Guard { bool active=false; int cap=0, device=-1, hits=0, original=0; };
static thread_local Guard guard;
extern "C" int s5_marlin_begin(uint64_t stream, int device, int *cap) {
  if (guard.active) return -1;
  CUgreenCtx green{}; CUdevResource resource{};
  CUresult r=cuStreamGetGreenCtx((CUstream)stream,&green);
  if (r) return (int)r;
  if (green) {
    r=cuGreenCtxGetDevResource(green,&resource,CU_DEV_RESOURCE_TYPE_SM);
    if (r) return (int)r;
    if (!resource.sm.smCount) return -2;
  }
  guard=Guard{true,green?(int)resource.sm.smCount:0,device,0,0};
  *cap=guard.cap; return 0;
}
extern "C" int s5_marlin_end(int *hits, int *original) {
  if (!guard.active) return -3;
  *hits=guard.hits; *original=guard.original; guard=Guard{}; return 0;
}
extern "C" cudaError_t CUDARTAPI cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) {
  using Fn=cudaError_t (CUDARTAPI *)(int*,cudaDeviceAttr,int);
  static Fn real=(Fn)dlsym(RTLD_NEXT,"cudaDeviceGetAttribute");
  if (!real) return cudaErrorUnknown;
  cudaError_t r=real(value,attr,device);
  if (r==cudaSuccess && guard.active && guard.cap>0 && guard.device==device &&
      attr==cudaDevAttrMultiProcessorCount) {
    ++guard.hits;guard.original=*value;*value=std::min(*value,guard.cap);
  }
  return r;
}
