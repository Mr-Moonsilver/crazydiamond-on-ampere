// Built only in the owner's GPU job. Clone a captured graph; replace its memory
// nodes, preserving default dependency edges. Never modify the original graph.
#include "s5_graph_audit.cu"
#include <algorithm>
struct CopyJob { unsigned char *src,*dst;size_t sp,sh,dp,dh,w,h,d;unsigned value,element;int fill; };
__global__ void byte_copy(CopyJob p) {
  size_t n=p.w*p.h*p.d;
  for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x) {
    size_t x=i%p.w,y=(i/p.w)%p.h,z=i/(p.w*p.h);
    unsigned char v=p.fill?(unsigned char)(p.value>>(8*(x%p.element))):p.src[z*p.sh*p.sp+y*p.sp+x];
    p.dst[z*p.dh*p.dp+y*p.dp+x]=v;
  }
  // Preserve host visibility if the original node was device-to-host.
  __threadfence_system();
}
struct CopyGraph { CUgraph graph{};CUgraphExec exec{}; };
static CUresult pointer(CUmemorytype kind,const void *host,CUdeviceptr device,CUdeviceptr *out) {
  if(kind==CU_MEMORYTYPE_DEVICE){*out=device;return CUDA_SUCCESS;}
  if(kind==CU_MEMORYTYPE_HOST || kind==CU_MEMORYTYPE_UNIFIED)
    return cuPointerGetAttribute(out,CU_POINTER_ATTRIBUTE_DEVICE_POINTER,kind==CU_MEMORYTYPE_HOST?(CUdeviceptr)host:device);
  return CUDA_ERROR_NOT_SUPPORTED; // Never guess array layout or register memory.
}
static CUresult convert(CopyGraph *g,CUcontext context,unsigned *copies,unsigned *sets) {
  size_t n=0;CUresult r=cuGraphGetNodes(g->graph,nullptr,&n);if(r)return r;
  std::vector<CUgraphNode> nodes(n);r=cuGraphGetNodes(g->graph,nodes.data(),&n);if(r)return r;
  cudaKernel_t kernel{};
  if(cudaGetKernel(&kernel,(const void*)byte_copy)!=cudaSuccess)return CUDA_ERROR_INVALID_HANDLE;
  for(auto node:nodes) {
    CUgraphNodeType type;r=cuGraphNodeGetType(node,&type);if(r)return r;
    if(type==CU_GRAPH_NODE_TYPE_GRAPH || type==CU_GRAPH_NODE_TYPE_MEM_ALLOC || type==CU_GRAPH_NODE_TYPE_MEM_FREE)return CUDA_ERROR_NOT_SUPPORTED;
    if(type!=CU_GRAPH_NODE_TYPE_MEMCPY && type!=CU_GRAPH_NODE_TYPE_MEMSET)continue;
    CopyJob j{};CUdeviceptr src{},dst{};
    if(type==CU_GRAPH_NODE_TYPE_MEMCPY) {
      CUDA_MEMCPY3D p{};r=cuGraphMemcpyNodeGetParams(node,&p);if(r)return r;
      if((r=pointer(p.srcMemoryType,p.srcHost,p.srcDevice,&src)) || (r=pointer(p.dstMemoryType,p.dstHost,p.dstDevice,&dst)))return r;
      j.w=p.WidthInBytes;j.h=p.Height;j.d=p.Depth;
      j.sp=p.srcPitch?p.srcPitch:j.w+p.srcXInBytes;j.dp=p.dstPitch?p.dstPitch:j.w+p.dstXInBytes;
      j.sh=p.srcHeight?p.srcHeight:j.h+p.srcY;j.dh=p.dstHeight?p.dstHeight:j.h+p.dstY;
      if(j.sp<j.w+p.srcXInBytes || j.dp<j.w+p.dstXInBytes || j.sh<j.h+p.srcY || j.dh<j.h+p.dstY)return CUDA_ERROR_INVALID_VALUE;
      j.src=(unsigned char*)(src+p.srcZ*j.sh*j.sp+p.srcY*j.sp+p.srcXInBytes);
      j.dst=(unsigned char*)(dst+p.dstZ*j.dh*j.dp+p.dstY*j.dp+p.dstXInBytes);j.element=1;
      ++*copies;
    } else {
      CUDA_MEMSET_NODE_PARAMS p{};r=cuGraphMemsetNodeGetParams(node,&p);if(r)return r;
      if(p.elementSize!=1 && p.elementSize!=2 && p.elementSize!=4)return CUDA_ERROR_NOT_SUPPORTED;
      j.dst=(unsigned char*)p.dst;j.w=p.width*p.elementSize;j.h=p.height;j.d=1;
      j.dp=p.pitch?p.pitch:j.w;j.dh=j.h;j.element=p.elementSize;j.value=p.value;j.fill=1;
      if(j.dp<j.w)return CUDA_ERROR_INVALID_VALUE;
      ++*sets;
    }
    if(!j.w || !j.h || !j.d || j.w>1024ULL*1024*1024 || j.h>1024ULL*1024*1024/j.w || j.d>1024ULL*1024*1024/(j.w*j.h))return CUDA_ERROR_NOT_SUPPORTED;
    size_t ni=0,no=0;
    if((r=cuGraphNodeGetDependencies(node,nullptr,nullptr,&ni)) || (r=cuGraphNodeGetDependentNodes(node,nullptr,nullptr,&no)))return r;
    std::vector<CUgraphNode> in(ni),out(no);
    // NULL edgeData deliberately rejects non-default edges (LOSSY_QUERY).
    if((r=cuGraphNodeGetDependencies(node,in.data(),nullptr,&ni)) || (r=cuGraphNodeGetDependentNodes(node,out.data(),nullptr,&no)))return r;
    void *args[]={&j};CUDA_KERNEL_NODE_PARAMS kp{};
    kp.kern=(CUkernel)kernel;kp.ctx=context;kp.gridDimX=(unsigned)std::min<size_t>(128,(j.w*j.h*j.d+255)/256);
    kp.gridDimY=kp.gridDimZ=kp.blockDimY=kp.blockDimZ=1;kp.blockDimX=256;kp.kernelParams=args;
    CUgraphNode replacement{};r=cuGraphAddKernelNode(&replacement,g->graph,in.data(),ni,&kp);if(r)return r;
    for(auto dependent:out){r=cuGraphAddDependencies(g->graph,&replacement,&dependent,nullptr,1);if(r)return r;}
    r=cuGraphDestroyNode(node);if(r)return r;
  }
  return CUDA_SUCCESS;
}
extern "C" int s5_copy_graph(uint64_t original,uint64_t stream,void **handle,uint64_t *raw,unsigned *copies,unsigned *sets) {
  auto *g=new CopyGraph;CUgreenCtx green{};CUcontext context{};CUresult r;
  if((r=cuStreamGetGreenCtx((CUstream)stream,&green)) || !green){delete g;return r?r:CUDA_ERROR_INVALID_CONTEXT;}
  if((r=cuCtxFromGreenCtx(&context,green)) || (r=cuGraphClone(&g->graph,(CUgraph)original)) ||
     (r=convert(g,context,copies,sets)) || (r=cuGraphInstantiateWithFlags(&g->exec,g->graph,0))) {
    if(g->graph)cuGraphDestroy(g->graph);delete g;return r;
  }
  *raw=(uint64_t)g->graph;*handle=g;return 0;
}
extern "C" int s5_copy_launch(void *handle,uint64_t stream){return cuGraphLaunch(((CopyGraph*)handle)->exec,(CUstream)stream);}
extern "C" int s5_copy_close(void *handle){auto *g=(CopyGraph*)handle;CUresult r=cuGraphExecDestroy(g->exec);if(r)return r;r=cuGraphDestroy(g->graph);if(r)return r;delete g;return 0;}
