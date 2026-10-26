// VM-only bounded eager families. Explicit streams, no allocation or engine.
#include <cuda_runtime.h>
#include <cstdint>
#ifdef __CUDACC__
__global__ void vm8_pre(int *p) { if(threadIdx.x==0)p[0]+=1; }
__global__ void vm8_post(int *p) { if(threadIdx.x==0)p[0]+=2; }
__global__ void vm8_sentinel(int *p) { if(threadIdx.x==0)p[0]+=3; }
#else
void vm8_pre(int *p); void vm8_post(int *p); void vm8_sentinel(int *p); // CPU header syntax only
#endif
extern "C" int vm8_eager_submit(uint64_t stream,uint64_t a,uint64_t b,
 uint64_t h_in,uint64_t h_out,int family,int count) {
 if(!stream||!a||!b||!h_in||!h_out||family<1||family>7||count<1||count>128)return cudaErrorInvalidValue;
 auto st=reinterpret_cast<cudaStream_t>(stream);auto p=reinterpret_cast<int*>(a);
 for(int i=0;i<count;++i){cudaError_t rc=cudaSuccess;
  if(family<=3){void*args[]={&p};
   const void*fn=family==1?reinterpret_cast<const void*>(vm8_pre):family==2?reinterpret_cast<const void*>(vm8_post):reinterpret_cast<const void*>(vm8_sentinel);
   rc=cudaLaunchKernel(fn,dim3(1),dim3(32),args,0,st);
  }else if(family==4)rc=cudaMemcpyAsync(p,reinterpret_cast<void*>(h_in),4096,cudaMemcpyHostToDevice,st);
  else if(family==5)rc=cudaMemcpyAsync(reinterpret_cast<void*>(b),p,4096,cudaMemcpyDeviceToDevice,st);
  else if(family==6)rc=cudaMemcpyAsync(reinterpret_cast<void*>(h_out),p,4096,cudaMemcpyDeviceToHost,st);
  else rc=cudaMemsetAsync(reinterpret_cast<void*>(b),0,4096,st);
  if(rc!=cudaSuccess)return rc;
 }return cudaSuccess;
}
