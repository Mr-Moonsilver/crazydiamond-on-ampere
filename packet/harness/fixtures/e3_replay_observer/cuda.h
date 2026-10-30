/* Minimal CPU fixture: the subset of the pinned CUDA 13 driver API used by the
 * replay observer.  This is a test ABI, not a copy of the real header; the
 * observer is compiled against the real headers separately. */
#ifndef S5RO_FIXTURE_CUDA_H
#define S5RO_FIXTURE_CUDA_H

#include <stddef.h>
#include <stdint.h>

#define CUDAAPI
#define CUDA_SUCCESS 0

typedef int CUresult;
typedef void* CUfunction;
typedef void* CUkernel;
typedef struct CUctx_st *CUcontext;
typedef struct CUstream_st *CUstream;
typedef struct CUgraph_st *CUgraph;
typedef struct CUgraphExec_st *CUgraphExec;
typedef struct CUgraphNode_st *CUgraphNode;
typedef struct CUgreenCtx_st *CUgreenCtx;

typedef struct { CUfunction func; CUkernel kern; CUcontext ctx; } CUDA_KERNEL_NODE_PARAMS;

typedef enum CUgraphNodeType {
  CU_GRAPH_NODE_TYPE_KERNEL = 0,
  CU_GRAPH_NODE_TYPE_MEMCPY = 1,
  CU_GRAPH_NODE_TYPE_MEMSET = 2,
  CU_GRAPH_NODE_TYPE_HOST = 3,
  CU_GRAPH_NODE_TYPE_GRAPH = 4,
  CU_GRAPH_NODE_TYPE_EMPTY = 5,
  CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
  CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
  CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
  CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
  CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
  CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
  CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12,
  CU_GRAPH_NODE_TYPE_CONDITIONAL = 13
} CUgraphNodeType;

#ifdef __cplusplus
extern "C" {
#endif

CUresult cuCtxFromGreenCtx(CUcontext *pContext, CUgreenCtx hCtx);
CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes);
CUresult cuGraphKernelNodeGetParams(CUgraphNode node, CUDA_KERNEL_NODE_PARAMS *p);
CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type);

#ifdef __cplusplus
}
#endif

#endif
