// Diagnostic only. Built against the pinned image headers, never on the hub.
#include "s5_green.cu"
#include <vector>
#include <cstdio>
#include <cstring>

static const char* memkind(CUmemorytype t) {
  switch(t) { case CU_MEMORYTYPE_HOST:return "host";case CU_MEMORYTYPE_DEVICE:return "device";
    case CU_MEMORYTYPE_ARRAY:return "array";case CU_MEMORYTYPE_UNIFIED:return "unified";default:return "unknown"; }
}

static CUresult audit_nodes(CUgraph graph, CUcontext expected, FILE *out, unsigned *index, unsigned depth) {
  if (depth > 16) return CUDA_ERROR_INVALID_VALUE;
  size_t count=0; CUresult r=cuGraphGetNodes(graph,nullptr,&count); if(r) return r;
  std::vector<CUgraphNode> nodes(count);
  r=cuGraphGetNodes(graph,nodes.data(),&count); if(r) return r;
  for(auto node:nodes) {
    CUgraphNodeType type; r=cuGraphNodeGetType(node,&type); if(r) return r;
    unsigned id=(*index)++;
    if(type==CU_GRAPH_NODE_TYPE_KERNEL) {
      CUDA_KERNEL_NODE_PARAMS p{}; r=cuGraphKernelNodeGetParams(node,&p); if(r) return r;
      // Do not infer placement from the launch stream or a CUfunction's primary
      // context. Missing ctx remains unknown, never a successful restriction.
      const char *name=nullptr;
      if(p.func) (void)cuFuncGetName(&name,p.func);
      CUdevResource resource{};
      CUresult query=p.ctx?cuCtxGetDevResource(p.ctx,&resource,CU_DEV_RESOURCE_TYPE_SM):CUDA_ERROR_INVALID_CONTEXT;
      fprintf(out,"{\"node\":%u,\"depth\":%u,\"type\":\"kernel\",\"context_known\":%s,\"context_matches\":%s,\"resource_query_rc\":%d,\"sm_count\":%u,\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],\"shared_bytes\":%u,\"custom_ar_name\":%s}\n",
        id,depth,p.ctx?"true":"false",p.ctx&&p.ctx==expected?"true":"false",(int)query,
        query?0:resource.sm.smCount,p.gridDimX,p.gridDimY,p.gridDimZ,p.blockDimX,p.blockDimY,p.blockDimZ,p.sharedMemBytes,
        name&&strstr(name,"allreduce")?"true":"false");
    } else if(type==CU_GRAPH_NODE_TYPE_MEMCPY) {
      CUDA_MEMCPY3D p{};r=cuGraphMemcpyNodeGetParams(node,&p);if(r)return r;
      fprintf(out,"{\"node\":%u,\"depth\":%u,\"type\":\"memcpy\",\"type_id\":%d,\"direction\":\"%s-to-%s\",\"width_bytes\":%zu,\"height\":%zu,\"depth_count\":%zu,\"bytes\":%zu,\"src_pitch\":%zu,\"dst_pitch\":%zu}\n",
        id,depth,(int)type,memkind(p.srcMemoryType),memkind(p.dstMemoryType),p.WidthInBytes,p.Height,p.Depth,p.WidthInBytes*p.Height*p.Depth,p.srcPitch,p.dstPitch);
    } else if(type==CU_GRAPH_NODE_TYPE_MEMSET) {
      CUDA_MEMSET_NODE_PARAMS p{};r=cuGraphMemsetNodeGetParams(node,&p);if(r)return r;
      fprintf(out,"{\"node\":%u,\"depth\":%u,\"type\":\"memset\",\"type_id\":%d,\"element_bytes\":%u,\"width_elements\":%zu,\"height\":%zu,\"bytes\":%zu,\"pitch\":%zu}\n",
        id,depth,(int)type,p.elementSize,p.width,p.height,p.width*p.height*p.elementSize,p.pitch);
    } else {
      fprintf(out,"{\"node\":%u,\"depth\":%u,\"type\":\"other\",\"type_id\":%d}\n",id,depth,(int)type);
      if(type==CU_GRAPH_NODE_TYPE_GRAPH) {
        CUgraph child{}; r=cuGraphChildGraphNodeGetGraph(node,&child); if(r) return r;
        r=audit_nodes(child,expected,out,index,depth+1);if(r) return r;
      }
    }
  }
  return CUDA_SUCCESS;
}
extern "C" int s5_audit(uint64_t graph, uint64_t stream, const char *path) {
  CUgreenCtx green{}; CUcontext context{};
  CUresult r=cuStreamGetGreenCtx((CUstream)stream,&green);if(r) return r;
  r=green?cuCtxFromGreenCtx(&context,green):cuCtxGetCurrent(&context);if(r) return r;
  FILE *out=fopen(path,"wx");if(!out) return CUDA_ERROR_INVALID_VALUE;
  unsigned index=0;r=audit_nodes((CUgraph)graph,context,out,&index,0);
  fclose(out);return (int)r;
}
