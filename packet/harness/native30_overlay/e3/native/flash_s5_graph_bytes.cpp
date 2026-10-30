// Private, bounded serialization of a RETAINED graph definition. Own ABI 1.
// No graph replay, no device read, no pointer dereference except kernel parameter
// arrays returned by the driver under its documented GetParams contract. The
// output contains pointers/arguments: it stays in worker memory, NEVER a log.
#include <cuda.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
namespace {
constexpr size_t LIMIT=16u*1024u*1024u, NODES=32768, PARAMS=256;
struct Bytes {
 std::vector<unsigned char> v; size_t nodes=0;
 void raw(const void*p,size_t n) { if(n>LIMIT-v.size())throw 1;
  const auto*b=static_cast<const unsigned char*>(p);v.insert(v.end(),b,b+n); }
 template<class T>void put(T x){raw(&x,sizeof x);}
 void ok(CUresult r){if(r!=CUDA_SUCCESS)throw 2;}
 void graph(CUgraph g,int depth=0){
  if(depth>8)throw 3;size_t n=0;ok(cuGraphGetNodes(g,nullptr,&n));
  if(n>NODES-nodes)throw 4;nodes+=n;std::vector<CUgraphNode>a(n);
  size_t got=n;ok(cuGraphGetNodes(g,a.data(),&got));if(got!=n)throw 5;
  std::sort(a.begin(),a.end());put(n);
  for(auto node:a){put(reinterpret_cast<uintptr_t>(node));CUgraphNodeType t;
   ok(cuGraphNodeGetType(node,&t));put(t);
   size_t nd=0;ok(cuGraphNodeGetDependencies(node,nullptr,nullptr,&nd));if(nd>NODES)throw 6;
   std::vector<CUgraphNode>d(nd);std::vector<CUgraphEdgeData>e(nd);got=nd;ok(cuGraphNodeGetDependencies(node,d.data(),e.data(),&got));
   if(got!=nd)throw 7;std::vector<size_t>order(nd);for(size_t i=0;i<nd;i++)order[i]=i;
   std::sort(order.begin(),order.end(),[&](size_t i,size_t j){return d[i]<d[j];});put(nd);
   for(auto i:order){put(reinterpret_cast<uintptr_t>(d[i]));put(e[i].from_port);put(e[i].to_port);put(e[i].type);}
   if(t==CU_GRAPH_NODE_TYPE_KERNEL){CUDA_KERNEL_NODE_PARAMS p{};ok(cuGraphKernelNodeGetParams(node,&p));
    put(reinterpret_cast<uintptr_t>(p.func));put(reinterpret_cast<uintptr_t>(p.kern));put(reinterpret_cast<uintptr_t>(p.ctx));
    put(p.gridDimX);put(p.gridDimY);put(p.gridDimZ);put(p.blockDimX);put(p.blockDimY);put(p.blockDimZ);put(p.sharedMemBytes);
    // Packed `extra` parameters are deliberately unsupported, not guessed.
    if(p.extra)throw 8;bool ended=false;
    for(size_t i=0;i<PARAMS;i++){size_t off=0,size=0;CUresult r=p.func?cuFuncGetParamInfo(p.func,i,&off,&size):
       p.kern?cuKernelGetParamInfo(p.kern,i,&off,&size):CUDA_ERROR_INVALID_HANDLE;
     if(r==CUDA_ERROR_INVALID_VALUE){if(i==0 && p.kernelParams)throw 12;put(i);ended=true;break;}
     ok(r);if(!p.kernelParams||!p.kernelParams[i]||size==0||size>4096||off>65536||size>65536-off)throw 9;
     put(i);put(off);put(size);raw(p.kernelParams[i],size);
    }if(!ended)throw 10;
   }else if(t==CU_GRAPH_NODE_TYPE_MEMCPY){CUDA_MEMCPY3D p{};ok(cuGraphMemcpyNodeGetParams(node,&p));
    put(p.srcXInBytes);put(p.srcY);put(p.srcZ);put(p.srcLOD);put(p.srcMemoryType);
    put(reinterpret_cast<uintptr_t>(p.srcHost));put(p.srcDevice);put(reinterpret_cast<uintptr_t>(p.srcArray));
    put(p.srcPitch);put(p.srcHeight);put(p.dstXInBytes);put(p.dstY);put(p.dstZ);put(p.dstLOD);put(p.dstMemoryType);
    put(reinterpret_cast<uintptr_t>(p.dstHost));put(p.dstDevice);put(reinterpret_cast<uintptr_t>(p.dstArray));
    put(p.dstPitch);put(p.dstHeight);put(p.WidthInBytes);put(p.Height);put(p.Depth);
   }else if(t==CU_GRAPH_NODE_TYPE_MEMSET){CUDA_MEMSET_NODE_PARAMS p{};ok(cuGraphMemsetNodeGetParams(node,&p));
    put(p.dst);put(p.pitch);put(p.value);put(p.elementSize);put(p.width);put(p.height);
   }else if(t==CU_GRAPH_NODE_TYPE_GRAPH){CUgraph child;ok(cuGraphChildGraphNodeGetGraph(node,&child));graph(child,depth+1);
   }else if(t==CU_GRAPH_NODE_TYPE_WAIT_EVENT||t==CU_GRAPH_NODE_TYPE_EVENT_RECORD){CUevent event;
    ok(t==CU_GRAPH_NODE_TYPE_WAIT_EVENT?cuGraphEventWaitNodeGetEvent(node,&event):cuGraphEventRecordNodeGetEvent(node,&event));put(reinterpret_cast<uintptr_t>(event));
   }else if(t!=CU_GRAPH_NODE_TYPE_EMPTY)throw 11;
  }
 }
};
}
extern "C" int s5_graph_bytes(uint64_t handle,void*out,size_t capacity,size_t*written){
 if(!handle||!written||capacity>LIMIT)return 1;*written=0;
 try{Bytes b;b.graph(reinterpret_cast<CUgraph>(handle));*written=b.v.size();
  if(!out)return 0;if(capacity<b.v.size())return 2;std::memcpy(out,b.v.data(),b.v.size());return 0;
 }catch(...){return 3;}
}
