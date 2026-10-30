// Synthetic graph constructor, own ABI 1. No allocation of GPU/pinned memory:
// Python owns bounded torch tensors and keeps them alive through close.
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace {
struct Graphs {
  cudaGraph_t root{}, child{};
  cudaGraphExec_t exec{};
  cudaStream_t stream{};
};
#ifdef __CUDACC__
__global__ void add_one(int *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += 1;
}
#else
void add_one(int *, int); // header/API syntax check only; never linked on hub
#endif
void check(cudaError_t e) { if (e != cudaSuccess) throw e; }
void launch_add(int *p, int n, cudaStream_t stream) {
  void *args[] = {&p, &n};
  check(cudaLaunchKernel(reinterpret_cast<const void*>(add_one), dim3(4), dim3(256), args, 0, stream));
}
cudaError_t dispose(Graphs *p) {
  cudaError_t first = cudaSuccess;
  auto keep = [&](cudaError_t r) { if (first == cudaSuccess) first = r; };
  if (p->exec) keep(cudaGraphExecDestroy(p->exec));
  if (p->root) keep(cudaGraphDestroy(p->root));
  if (p->child) keep(cudaGraphDestroy(p->child));
  delete p;
  return first;
}
}
extern "C" int vmg_abi() { return 1; }
extern "C" int vmg_create(uint64_t stream, uint64_t a, uint64_t b, uint64_t z,
                          uint64_t host_in, uint64_t host_out, int n,
                          int kernels, int copies, int memsets, int child,
                          uint64_t *owner, uint64_t *graph) {
  if (!stream || !a || !b || !z || !host_in || !host_out || !owner || !graph ||
      n != 1024 || kernels < 1 || kernels > 1386 || copies < 3 || copies > 26 ||
      memsets < 1 || memsets > 7 || (child != 0 && child != 1)) return cudaErrorInvalidValue;
  *owner = *graph = 0;
  auto *p = new Graphs;
  p->stream = reinterpret_cast<cudaStream_t>(stream);
  bool capturing = false;
  try {
    const size_t bytes = n * sizeof(int);
    check(cudaStreamBeginCapture(p->stream, cudaStreamCaptureModeThreadLocal));
    capturing = true;
    check(cudaMemcpyAsync(reinterpret_cast<void*>(a), reinterpret_cast<void*>(host_in), bytes,
                          cudaMemcpyHostToDevice, p->stream));
    for (int i=0; i<copies-2; ++i)
      check(cudaMemcpyAsync(reinterpret_cast<void*>(b), reinterpret_cast<void*>(a), bytes,
                            cudaMemcpyDeviceToDevice, p->stream));
    for (int i=0; i<memsets; ++i)
      check(cudaMemsetAsync(reinterpret_cast<void*>(z), 0, bytes, p->stream));
    for (int i=0; i<kernels; ++i) {
      launch_add(reinterpret_cast<int*>(b), n, p->stream);
    }
    check(cudaMemcpyAsync(reinterpret_cast<void*>(host_out), reinterpret_cast<void*>(b), bytes,
                          cudaMemcpyDeviceToHost, p->stream));
    cudaError_t end = cudaStreamEndCapture(p->stream, &p->root);
    capturing = false;
    check(end);
    if (child) {
      check(cudaStreamBeginCapture(p->stream, cudaStreamCaptureModeThreadLocal));
      capturing = true;
      launch_add(reinterpret_cast<int*>(z), n, p->stream);
      end = cudaStreamEndCapture(p->stream, &p->child);
      capturing = false;
      check(end);
      size_t count=0;
      check(cudaGraphGetNodes(p->root, nullptr, &count));
      std::vector<cudaGraphNode_t> nodes(count);
      check(cudaGraphGetNodes(p->root, nodes.data(), &count));
      cudaGraphNode_t added;
      // Captured root plus a child after all existing nodes. The child is
      // explicit additional coverage, not claimed to be present in full8.
      check(cudaGraphAddChildGraphNode(&added, p->root, nodes.data(), count, p->child));
    }
    check(cudaGraphInstantiate(&p->exec, p->root, 0));
    *owner = reinterpret_cast<uint64_t>(p);
    *graph = reinterpret_cast<uint64_t>(p->root);
    return 0;
  } catch (cudaError_t e) {
    if (capturing) {
      cudaGraph_t abandoned{};
      cudaStreamEndCapture(p->stream, &abandoned);
      if (abandoned) cudaGraphDestroy(abandoned);
    }
    dispose(p);
    return static_cast<int>(e);
  } catch (...) {
    if (capturing) {
      cudaGraph_t abandoned{};
      cudaStreamEndCapture(p->stream, &abandoned);
      if (abandoned) cudaGraphDestroy(abandoned);
    }
    dispose(p); return cudaErrorUnknown;
  }
}
extern "C" int vmg_replay(uint64_t owner) {
  if (!owner) return cudaErrorInvalidValue;
  auto *p = reinterpret_cast<Graphs*>(owner);
  return cudaGraphLaunch(p->exec, p->stream);
}
extern "C" int vmg_inventory(uint64_t owner, int *out, int length) {
  if (!owner || !out || length != 8) return cudaErrorInvalidValue;
  for (int i=0; i<8; ++i) out[i]=0;
  auto *p=reinterpret_cast<Graphs*>(owner);
  try {
    size_t n=0; check(cudaGraphGetNodes(p->root,nullptr,&n));
    if(n>1500) return cudaErrorInvalidValue;
    std::vector<cudaGraphNode_t> nodes(n); check(cudaGraphGetNodes(p->root,nodes.data(),&n));
    for(auto node:nodes){
      cudaGraphNodeType t; check(cudaGraphNodeGetType(node,&t));
      if(t==cudaGraphNodeTypeKernel) ++out[0];
      else if(t==cudaGraphNodeTypeMemcpy){
        cudaMemcpy3DParms q{};check(cudaGraphMemcpyNodeGetParams(node,&q));
        if(q.kind==cudaMemcpyHostToDevice) ++out[1];
        else if(q.kind==cudaMemcpyDeviceToDevice) ++out[2];
        else if(q.kind==cudaMemcpyDeviceToHost) ++out[3];
        else ++out[7];
      }else if(t==cudaGraphNodeTypeMemset) ++out[4];
      else if(t==cudaGraphNodeTypeGraph){
        ++out[5];cudaGraph_t child;check(cudaGraphChildGraphNodeGetGraph(node,&child));
        size_t cn=0;check(cudaGraphGetNodes(child,nullptr,&cn));
        if(cn>1) return cudaErrorInvalidValue;
        std::vector<cudaGraphNode_t> cs(cn);check(cudaGraphGetNodes(child,cs.data(),&cn));
        for(auto c:cs){cudaGraphNodeType ct;check(cudaGraphNodeGetType(c,&ct));
          if(ct==cudaGraphNodeTypeKernel)++out[6];else ++out[7];}
      }else ++out[7];
    }
    return 0;
  }catch(cudaError_t e){return static_cast<int>(e);}catch(...){return cudaErrorUnknown;}
}
extern "C" int vmg_close(uint64_t owner) {
  if (!owner) return cudaErrorInvalidValue;
  auto *p = reinterpret_cast<Graphs*>(owner);
  cudaError_t e = cudaStreamSynchronize(p->stream);
  if (e != cudaSuccess) return e; // refuse destruction with unproven consumers
  return dispose(p);
}
