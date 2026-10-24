// Runtime-built against the image's CUDA headers: no guessed ctypes struct layout.
// The queue compiles this; it is never compiled/executed on the hub.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
struct Pair { CUgreenCtx d{}, p{}; CUstream ds{}, ps{}; int dc{}, pc{}; };
extern "C" int s5_create(int dev, unsigned requested, void **handle,
                          uint64_t *ds, uint64_t *ps, int *dc, int *pc) {
  CUdevResource whole{}, group{}, rest{};
  unsigned n=1; CUresult r;
  if ((r=cuDeviceGetDevResource(dev,&whole,CU_DEV_RESOURCE_TYPE_SM))) return r;
  if ((r=cuDevSmResourceSplitByCount(&group,&n,&whole,&rest,0,requested))) return r;
  if (n!=1 || group.sm.smCount<requested || !rest.sm.smCount ||
      group.sm.smCount+rest.sm.smCount!=whole.sm.smCount) return CUDA_ERROR_INVALID_VALUE;
  Pair *p = new Pair;
  CUdevResourceDesc dd{}, pd{};
  if ((r=cuDevResourceGenerateDesc(&dd,&group,1))) { delete p; return r; }
  if ((r=cuDevResourceGenerateDesc(&pd,&rest,1))) { delete p; return r; }
  if ((r=cuGreenCtxCreate(&p->d,dd,dev,CU_GREEN_CTX_DEFAULT_STREAM))) { delete p; return r; }
  if ((r=cuGreenCtxCreate(&p->p,pd,dev,CU_GREEN_CTX_DEFAULT_STREAM))) {
    cuGreenCtxDestroy(p->d); delete p; return r;
  }
  if ((r=cuGreenCtxStreamCreate(&p->ds,p->d,CU_STREAM_NON_BLOCKING,0)) ||
      (r=cuGreenCtxStreamCreate(&p->ps,p->p,CU_STREAM_NON_BLOCKING,0))) {
    if (p->ds) cuStreamDestroy(p->ds);
    cuGreenCtxDestroy(p->p); cuGreenCtxDestroy(p->d); delete p; return r;
  }
  *dc=group.sm.smCount; *pc=rest.sm.smCount;
  *ds=(uint64_t)p->ds; *ps=(uint64_t)p->ps; *handle=p; return 0;
}
extern "C" int s5_close(void *handle) {
  Pair *p=(Pair*)handle;
  CUresult r=cuStreamSynchronize(p->ds); if(r) return r;
  r=cuStreamSynchronize(p->ps); if(r) return r;
  r=cuStreamDestroy(p->ds); if(r) return r;
  r=cuStreamDestroy(p->ps); if(r) return r;
  r=cuGreenCtxDestroy(p->d); if(r) return r;
  r=cuGreenCtxDestroy(p->p); if(r) return r;
  delete p; return 0;
}
__global__ void sm_probe(int *out, unsigned long long cycles) {
  unsigned sm; asm("mov.u32 %0, %%smid;" : "=r"(sm));
  unsigned long long begin=clock64();
  while(clock64()-begin < cycles) asm volatile("");
  out[blockIdx.x]=sm;
}
extern "C" int s5_probe(uint64_t ptr, int blocks, uint64_t stream, uint64_t cycles) {
  sm_probe<<<blocks,1,0,(cudaStream_t)stream>>>((int*)ptr,cycles);
  return (int)cudaPeekAtLastError();
}
__global__ void global_stamp(unsigned long long *out) {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); *out=t;
}
extern "C" int s5_stamp(uint64_t ptr, uint64_t stream) {
  global_stamp<<<1,1,0,(cudaStream_t)stream>>>((unsigned long long*)ptr);
  return (int)cudaPeekAtLastError();
}
extern "C" int s5_pci_bus_id(int dev, char *out, int len) {
  return (int)cuDeviceGetPCIBusId(out,len,dev);
}
