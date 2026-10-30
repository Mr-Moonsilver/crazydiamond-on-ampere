// VM-only host queries. Own ABI; IDs only leave this helper, never handles.
#include <cuda.h>
#include <cupti_activity.h>
#include <cstdint>
extern "C" int vmc_stream(uint64_t stream,uint64_t*out,unsigned count){
 if(!stream||!out||count!=7)return 1;
 CUcontext primary=nullptr,converted=nullptr;CUgreenCtx green=nullptr;
 if(cuStreamGetCtx_v2(reinterpret_cast<CUstream>(stream),&primary,&green)!=CUDA_SUCCESS)return 2;
 uint32_t p=0,g=0,s=0;unsigned long long pd=0,gd=0;
 if(cuptiGetContextId(primary,&p)!=CUPTI_SUCCESS||cuCtxGetId(primary,&pd)!=CUDA_SUCCESS)return 3;
 if(green){if(cuCtxFromGreenCtx(&converted,green)!=CUDA_SUCCESS||
  cuptiGetContextId(converted,&g)!=CUPTI_SUCCESS||cuGreenCtxGetId(green,&gd)!=CUDA_SUCCESS)return 4;}
 if(cuptiGetStreamIdEx(nullptr,reinterpret_cast<CUstream>(stream),0,&s)!=CUPTI_SUCCESS)return 5;
 out[0]=p;out[1]=green?1:0;out[2]=g;out[3]=s;out[4]=pd;out[5]=gd;out[6]=1;return 0;
}
extern "C" int vmc_timestamp(uint64_t*out){return out?static_cast<int>(cuptiGetTimestamp(out)):1;}
