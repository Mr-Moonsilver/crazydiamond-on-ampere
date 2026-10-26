// Flash-Next E3 raw retained-graph audit helper (own C ABI 1).
//
// Inspection only: this file walks CUDA graphs with the driver API and writes
// bounded, sanitised JSON metadata into a caller-owned byte buffer. It never
// launches a kernel, allocates device memory, replays/synchronises/repairs a
// graph, or accepts a filesystem path. There is no fallback to
// cuCtxGetCurrent: the supplied stream must belong to EXACTLY the expected
// CUgreenCtx handle, and node contexts are compared with the CUcontext that
// handle converts to. A mismatched/absent context is reported, never guessed.
//
// Exported ABI (fixed; bump FLASH_S5_GRAPH_AUDIT_ABI with any change):
//
//   int s5ga_abi(void);
//   int s5ga_inspect(uint64_t graph, uint64_t stream,
//                    uint64_t expected_green_ctx,
//                    uint32_t max_nodes, uint32_t max_depth,
//                    uint64_t max_bytes,
//                    char *out, uint64_t out_cap, uint64_t *out_len);
//
// s5ga_inspect returns one of the S5GA_* status codes below. On S5GA_OK the
// JSON is complete, the caller must honour `complete` in the report, and
// *out_len is the exact byte count (no NUL included). S5GA_TRUNCATED means the
// report did not fit; *out_len is the exact required size and no partial report
// is written. Any other status means no report was produced.
//
// Handles, addresses, CUfunction values, kernel arguments and kernel names are
// never serialised. Kernel names are reduced to a candidate substring class.
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <set>

#if !defined(CUDA_VERSION) || CUDA_VERSION < 13000
#error "flash_s5_graph_audit.cu requires the pinned CUDA 13 headers"
#endif

#define FLASH_S5_GRAPH_AUDIT_ABI 1

enum {
  S5GA_OK = 0,
  S5GA_TRUNCATED = 1,
  S5GA_BAD_ARGS = 2,
  S5GA_GREEN_QUERY_FAILED = 3,
  S5GA_GREEN_MISSING = 4,
  S5GA_GREEN_MISMATCH = 5,
  S5GA_CONTEXT_CONVERT_FAILED = 6,
  S5GA_GRAPH_QUERY_FAILED = 7,
  S5GA_INTERNAL_ERROR = 8,
  S5GA_STREAM_CONTEXT_QUERY_FAILED = 9,
  S5GA_STREAM_CONTEXT_MISMATCH = 10
};

static const char *kSchema = "s5ga-native-audit/1";

static void put_u(std::string &s, unsigned long long v) {
  char b[32];
  int n = snprintf(b, sizeof(b), "%llu", v);
  if (n > 0) s.append(b, (size_t)n);
}

static void put_i(std::string &s, long long v) {
  char b[32];
  int n = snprintf(b, sizeof(b), "%lld", v);
  if (n > 0) s.append(b, (size_t)n);
}

static void put_b(std::string &s, bool v) { s += v ? "true" : "false"; }

static void put_q(std::string &s, const char *v) {
  s.push_back('"');
  s.append(v);
  s.push_back('"');
}

struct Walk {
  uint32_t max_nodes = 0;
  uint32_t max_depth = 0;
  uint64_t budget = 0;
  std::string nodes;
  unsigned node_count = 0;
  unsigned root_nodes = 0;
  unsigned child_graphs = 0;
  unsigned n_kernel = 0;
  unsigned n_memcpy = 0;
  unsigned n_memset = 0;
  unsigned n_graph = 0;
  unsigned n_other = 0;
  unsigned n_unknown_type = 0;
  unsigned unsupported = 0;
  unsigned unknown = 0;
  unsigned max_depth_seen = 0;
  unsigned collective_unknown = 0;
  unsigned c_allreduce = 0;
  unsigned c_allgather = 0;
  unsigned c_reducescatter = 0;
  unsigned c_broadcast = 0;
  unsigned c_other_collective = 0;
  unsigned c_none = 0;
  unsigned c_unavailable = 0;
  bool limits_exceeded = false;
  std::vector<std::string> reasons;
  std::vector<CUgraph> stack;
};

static void add_reason(Walk &w, const char *reason) {
  for (size_t i = 0; i < w.reasons.size(); ++i) {
    if (w.reasons[i] == reason) return;
  }
  w.reasons.push_back(std::string(reason));
}

static bool emit_node(Walk &w, const std::string &node) {
  // Keep envelope room; the caller buffer remains the hard limit and is
  // checked again before copying.
  if (w.nodes.size() + node.size() + 1 > w.budget) {
    w.limits_exceeded = true;
    add_reason(w, "limits_output");
    return false;
  }
  if (!w.nodes.empty()) w.nodes.push_back(',');
  w.nodes += node;
  return true;
}

static const char *mem_kind(CUmemorytype t) {
  switch (t) {
    case CU_MEMORYTYPE_HOST: return "host";
    case CU_MEMORYTYPE_DEVICE: return "device";
    case CU_MEMORYTYPE_ARRAY: return "array";
    case CU_MEMORYTYPE_UNIFIED: return "unified";
    default: return "unknown";
  }
}

// A substring is only a candidate classifier. The report states the basis and
// never calls it confirmed collective proof.
static const char *collective_candidate(const char *name) {
  if (name == nullptr) return "unavailable";
  std::string lower;
  lower.reserve(64);
  for (const char *p = name; *p && lower.size() < 200; ++p) {
    char c = *p;
    if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    lower.push_back(c);
  }
  if (lower.find("allreduce") != std::string::npos ||
      lower.find("all_reduce") != std::string::npos) return "allreduce";
  if (lower.find("allgather") != std::string::npos ||
      lower.find("all_gather") != std::string::npos) return "allgather";
  if (lower.find("reducescatter") != std::string::npos ||
      lower.find("reduce_scatter") != std::string::npos) return "reducescatter";
  if (lower.find("broadcast") != std::string::npos) return "broadcast";
  if (lower.find("nccl") != std::string::npos ||
      lower.find("sendrecv") != std::string::npos ||
      lower.find("send_recv") != std::string::npos) return "other_collective";
  return "none";
}

static void count_candidate(Walk &w, const char *candidate) {
  std::string c(candidate);
  if (c == "allreduce") ++w.c_allreduce;
  else if (c == "allgather") ++w.c_allgather;
  else if (c == "reducescatter") ++w.c_reducescatter;
  else if (c == "broadcast") ++w.c_broadcast;
  else if (c == "other_collective") ++w.c_other_collective;
  else if (c == "unavailable") { ++w.c_unavailable; ++w.collective_unknown; }
  else ++w.c_none;
}

static bool mul_ok(size_t a, size_t b, size_t &out) {
  if (a != 0 && b > (size_t)-1 / a) return false;
  out = a * b;
  return true;
}

static void put_u3(std::string &s, unsigned a, unsigned b, unsigned c) {
  s.push_back('[');
  put_u(s, a); s.push_back(',');
  put_u(s, b); s.push_back(',');
  put_u(s, c); s.push_back(']');
}

static std::string node_prefix(unsigned id, unsigned graph, int parent,
                               unsigned depth, const char *type, int type_id,
                               bool supported, bool unknown, const char *params,
                               CUresult rc) {
  std::string s = "{";
  s += "\"id\":"; put_u(s, id);
  s += ",\"graph\":"; put_u(s, graph);
  s += ",\"parent\":"; put_i(s, parent);
  s += ",\"depth\":"; put_u(s, depth);
  s += ",\"type\":"; put_q(s, type);
  s += ",\"type_id\":"; put_i(s, type_id);
  s += ",\"supported\":"; put_b(s, supported);
  s += ",\"unknown\":"; put_b(s, unknown);
  s += ",\"params\":"; put_q(s, params);
  s += ",\"query_rc\":"; put_u(s, (unsigned long long)rc);
  return s;
}

static void walk_graph(Walk &w, CUgraph graph, CUcontext expected_ctx,
                       unsigned graph_index, unsigned depth, int parent,
                       bool root, bool &ok);

static bool walk_node(Walk &w, CUgraphNode node, CUcontext expected_ctx,
                      unsigned id, unsigned graph_index, unsigned depth,
                      int parent) {
  CUgraphNodeType type = (CUgraphNodeType)-1;
  CUresult trc = cuGraphNodeGetType(node, &type);
  if (trc != CUDA_SUCCESS) {
    std::string s = node_prefix(id, graph_index, parent, depth, "unknown", -1,
                                false, true, "not_applicable", trc);
    s += "}";
    add_reason(w, "node_type_error");
    add_reason(w, "unsupported_node");
    if (emit_node(w, s)) {
      ++w.node_count;
      ++w.n_unknown_type;
      ++w.unsupported;
      ++w.unknown;
      ++w.collective_unknown;
      if (graph_index == 0) ++w.root_nodes;
      if (depth > w.max_depth_seen) w.max_depth_seen = depth;
    }
    return true;
  }

  if (type == CU_GRAPH_NODE_TYPE_KERNEL) {
    CUDA_KERNEL_NODE_PARAMS p{};
    CUresult rc = cuGraphKernelNodeGetParams(node, &p);
    if (rc != CUDA_SUCCESS) {
      std::string s = node_prefix(id, graph_index, parent, depth, "kernel",
                                  (int)CU_GRAPH_NODE_TYPE_KERNEL, true, true,
                                  "error", rc);
      s += ",\"context_basis\":\"unavailable\",\"context_known\":false,\"context_matches\":false";
      s += ",\"sm_query\":\"unknown\",\"sm_query_rc\":0,\"sm_count\":null";
      s += ",\"grid\":null,\"block\":null,\"shared_bytes\":null";
      s += ",\"collective_candidate\":\"unavailable\"";
      s += ",\"collective_basis\":\"function_name_substring\"";
      s += ",\"collective_confirmed\":false}";
      add_reason(w, "node_query_error");
      if (emit_node(w, s)) {
        ++w.node_count; ++w.n_kernel; ++w.unknown;
        ++w.collective_unknown; ++w.c_unavailable;
        if (graph_index == 0) ++w.root_nodes;
        if (depth > w.max_depth_seen) w.max_depth_seen = depth;
      }
      return true;
    }
    // CUDA 13 CUDA_KERNEL_NODE_PARAMS_v2 explicitly ignores ctx when func
    // is set. A CUfunction's module context is not green-node placement proof.
    const char *context_basis = p.func ? "ignored_for_func" :
                               (p.kern && p.ctx ? "kernel_ctx" : "unavailable");
    bool ctx_known = p.func == nullptr && p.kern != nullptr && p.ctx != nullptr;
    bool ctx_matches = ctx_known && p.ctx == expected_ctx;
    const char *sm_query = "unknown";
    CUresult sm_rc = CUDA_SUCCESS;
    unsigned sm_count = 0;
    bool sm_known = false;
    if (ctx_known) {
      CUdevResource resource{};
      sm_rc = cuCtxGetDevResource(p.ctx, &resource, CU_DEV_RESOURCE_TYPE_SM);
      if (sm_rc == CUDA_SUCCESS) {
        sm_query = "ok";
        sm_known = true;
        sm_count = resource.sm.smCount;
      } else if (sm_rc == CUDA_ERROR_NOT_SUPPORTED) {
        sm_query = "unsupported";
      } else {
        sm_query = "error";
      }
    }
    bool unknown = !ctx_known || !sm_known;
    if (!ctx_known) add_reason(w, "kernel_context_unknown");
    else if (!sm_known) add_reason(w, "kernel_sm_unknown");

    const char *name = nullptr;
    CUresult name_rc = p.func ? cuFuncGetName(&name, p.func)
                              : CUDA_ERROR_INVALID_HANDLE;
    if (!p.func && p.kern) name_rc = cuKernelGetName(&name, p.kern);
    const char *candidate = name_rc == CUDA_SUCCESS
                                ? collective_candidate(name) : "unavailable";
    if (name_rc != CUDA_SUCCESS || name == nullptr) {
      candidate = "unavailable";
      add_reason(w, "kernel_name_unavailable");
    }

    std::string s = node_prefix(id, graph_index, parent, depth, "kernel",
                                (int)CU_GRAPH_NODE_TYPE_KERNEL, true, unknown,
                                "ok", CUDA_SUCCESS);
    s += ",\"context_basis\":"; put_q(s, context_basis);
    s += ",\"context_known\":"; put_b(s, ctx_known);
    s += ",\"context_matches\":"; put_b(s, ctx_matches);
    s += ",\"sm_query\":"; put_q(s, sm_query);
    s += ",\"sm_query_rc\":"; put_u(s, (unsigned long long)sm_rc);
    s += ",\"sm_count\":";
    if (sm_known) put_u(s, sm_count); else s += "null";
    s += ",\"grid\":"; put_u3(s, p.gridDimX, p.gridDimY, p.gridDimZ);
    s += ",\"block\":"; put_u3(s, p.blockDimX, p.blockDimY, p.blockDimZ);
    s += ",\"shared_bytes\":"; put_u(s, p.sharedMemBytes);
    s += ",\"collective_candidate\":"; put_q(s, candidate);
    s += ",\"collective_basis\":\"function_name_substring\"";
    s += ",\"collective_confirmed\":false}";
    if (emit_node(w, s)) {
      ++w.node_count; ++w.n_kernel;
      if (unknown) ++w.unknown;
      count_candidate(w, candidate);
      if (graph_index == 0) ++w.root_nodes;
      if (depth > w.max_depth_seen) w.max_depth_seen = depth;
    }
    return true;
  }

  if (type == CU_GRAPH_NODE_TYPE_MEMCPY) {
    CUDA_MEMCPY3D p{};
    CUresult rc = cuGraphMemcpyNodeGetParams(node, &p);
    char direction[64] = "unknown";
    size_t bytes = 0;
    bool dims = false;
    bool have_params = rc == CUDA_SUCCESS;
    if (have_params) {
      snprintf(direction, sizeof(direction), "%s-to-%s",
               mem_kind(p.srcMemoryType), mem_kind(p.dstMemoryType));
      dims = mul_ok(p.WidthInBytes, p.Height, bytes) &&
             mul_ok(bytes, p.Depth, bytes);
    }
    // These getter APIs return dimensions, not the node execution context.
    // Pointer ownership or the expected stream cannot supply the missing proof.
    bool unknown = true;
    add_reason(w, "memory_context_unavailable");
    if (!have_params) add_reason(w, "node_query_error");
    else if (!dims) add_reason(w, "node_query_error");
    std::string s = node_prefix(id, graph_index, parent, depth, "memcpy",
                                (int)CU_GRAPH_NODE_TYPE_MEMCPY, true, unknown,
                                have_params ? "ok" : "error", rc);
    s += ",\"context_basis\":\"not_exposed\",\"context_known\":false";
    if (!have_params) {
      s += ",\"direction\":\"unknown\",\"width_bytes\":null,\"height\":null";
      s += ",\"depth_count\":null,\"bytes\":null,\"src_pitch\":null,\"dst_pitch\":null}";
    } else {
      s += ",\"direction\":"; put_q(s, direction);
      s += ",\"width_bytes\":"; put_u(s, p.WidthInBytes);
      s += ",\"height\":"; put_u(s, p.Height);
      s += ",\"depth_count\":"; put_u(s, p.Depth);
      s += ",\"bytes\":";
      if (dims) put_u(s, bytes); else s += "null";
      s += ",\"src_pitch\":"; put_u(s, p.srcPitch);
      s += ",\"dst_pitch\":"; put_u(s, p.dstPitch);
      s += "}";
    }
    if (emit_node(w, s)) {
      ++w.node_count; ++w.n_memcpy;
      if (unknown) ++w.unknown;
      if (graph_index == 0) ++w.root_nodes;
      if (depth > w.max_depth_seen) w.max_depth_seen = depth;
    }
    return true;
  }

  if (type == CU_GRAPH_NODE_TYPE_MEMSET) {
    CUDA_MEMSET_NODE_PARAMS p{};
    CUresult rc = cuGraphMemsetNodeGetParams(node, &p);
    bool have_params = rc == CUDA_SUCCESS;
    bool dims = false;
    size_t bytes = 0;
    if (have_params) {
      dims = mul_ok(p.width, p.height, bytes) &&
             mul_ok(bytes, (size_t)p.elementSize, bytes);
    }
    // These getter APIs return dimensions, not the node execution context.
    // Pointer ownership or the expected stream cannot supply the missing proof.
    bool unknown = true;
    add_reason(w, "memory_context_unavailable");
    if (!have_params || !dims) add_reason(w, "node_query_error");
    std::string s = node_prefix(id, graph_index, parent, depth, "memset",
                                (int)CU_GRAPH_NODE_TYPE_MEMSET, true, unknown,
                                have_params ? "ok" : "error", rc);
    s += ",\"context_basis\":\"not_exposed\",\"context_known\":false";
    if (!have_params) {
      s += ",\"element_bytes\":null,\"width_elements\":null,\"height\":null";
      s += ",\"bytes\":null,\"pitch\":null}";
    } else {
      s += ",\"element_bytes\":"; put_u(s, p.elementSize);
      s += ",\"width_elements\":"; put_u(s, p.width);
      s += ",\"height\":"; put_u(s, p.height);
      s += ",\"bytes\":";
      if (dims) put_u(s, bytes); else s += "null";
      s += ",\"pitch\":"; put_u(s, p.pitch);
      s += "}";
    }
    if (emit_node(w, s)) {
      ++w.node_count; ++w.n_memset;
      if (unknown) ++w.unknown;
      if (graph_index == 0) ++w.root_nodes;
      if (depth > w.max_depth_seen) w.max_depth_seen = depth;
    }
    return true;
  }

  if (type == CU_GRAPH_NODE_TYPE_GRAPH) {
    CUgraph child = nullptr;
    CUresult rc = cuGraphChildGraphNodeGetGraph(node, &child);
    bool unknown = false;
    bool can_descend = (depth < w.max_depth);
    int child_index = -1;
    const char *params = "ok";
    if (rc != CUDA_SUCCESS || child == nullptr) {
      params = "error";
      unknown = true;
      add_reason(w, "child_graph_error");
    } else if (!can_descend) {
      unknown = true;
      w.limits_exceeded = true;
      add_reason(w, "limits_depth");
      add_reason(w, "child_graph_not_expanded");
    } else {
      bool cycle = false;
      for (size_t i = 0; i < w.stack.size(); ++i) {
        if (w.stack[i] == child) { cycle = true; break; }
      }
      if (cycle) {
        unknown = true;
        add_reason(w, "child_graph_cycle");
      } else {
        size_t child_count = 0;
        CUresult crc = cuGraphGetNodes(child, nullptr, &child_count);
        if (crc != CUDA_SUCCESS) {
          unknown = true;
          add_reason(w, "child_graph_error");
        } else if (child_count >=
                   (size_t)(w.max_nodes - w.node_count)) {
          // Keep the receipt representable: do not claim a child graph whose
          // nodes cannot all be reported.
          unknown = true;
          w.limits_exceeded = true;
          add_reason(w, "limits_nodes");
        } else {
          child_index = (int)(w.child_graphs + 1);
        }
      }
    }
    std::string s = node_prefix(id, graph_index, parent, depth, "graph",
                                (int)CU_GRAPH_NODE_TYPE_GRAPH, true, unknown,
                                params, rc);
    s += ",\"child\":"; put_i(s, child_index);
    s += "}";
    bool emitted = emit_node(w, s);
    if (emitted) {
      ++w.node_count; ++w.n_graph;
      if (unknown) ++w.unknown;
      if (graph_index == 0) ++w.root_nodes;
      if (depth > w.max_depth_seen) w.max_depth_seen = depth;
    }
    if (!emitted) return false;
    if (child_index > 0) {
      w.child_graphs = (unsigned)child_index;
      bool child_ok = true;
      walk_graph(w, child, expected_ctx, (unsigned)child_index, depth + 1,
                 (int)id, false, child_ok);
    }
    return true;
  }

  // Recognised by the driver but not parameterised here (events, host nodes,
  // batch mem ops, ...). Never omitted: recorded as unsupported/unknown.
  std::string s = node_prefix(id, graph_index, parent, depth, "other",
                              (int)type, false, true, "not_applicable",
                              CUDA_SUCCESS);
  s += "}";
  add_reason(w, "unsupported_node");
  if (emit_node(w, s)) {
    ++w.node_count;
    ++w.n_other;
    ++w.unsupported;
    ++w.unknown;
    ++w.collective_unknown;
    if (graph_index == 0) ++w.root_nodes;
    if (depth > w.max_depth_seen) w.max_depth_seen = depth;
  }
  return true;
}

static void walk_graph(Walk &w, CUgraph graph, CUcontext expected_ctx,
                       unsigned graph_index, unsigned depth, int parent,
                       bool root, bool &ok) {
  ok = true;
  if (depth > w.max_depth) {
    w.limits_exceeded = true;
    add_reason(w, "limits_depth");
    return;
  }
  for (size_t i = 0; i < w.stack.size(); ++i) {
    if (w.stack[i] == graph) {
      add_reason(w, "child_graph_cycle");
      return;
    }
  }
  size_t count = 0;
  CUresult rc = cuGraphGetNodes(graph, nullptr, &count);
  if (rc != CUDA_SUCCESS) {
    if (root) {
      ok = false;
    } else {
      add_reason(w, "child_graph_error");
    }
    return;
  }
  if (count == 0) return;
  if (count > (size_t)(w.max_nodes - w.node_count)) {
    w.limits_exceeded = true;
    add_reason(w, "limits_nodes");
    return;
  }
  std::vector<CUgraphNode> nodes(count);
  const size_t capacity = count;
  rc = cuGraphGetNodes(graph, nodes.data(), &count);
  if (rc == CUDA_SUCCESS && count != capacity) {
    add_reason(w, "graph_changed");
    return;
  }
  if (rc != CUDA_SUCCESS) {
    if (root) {
      ok = false;
    } else {
      add_reason(w, "child_graph_error");
    }
    return;
  }
  std::set<CUgraphNode> seen;
  w.stack.push_back(graph);
  for (size_t i = 0; i < count; ++i) {
    if (!nodes[i] || !seen.insert(nodes[i]).second) {
      add_reason(w, "node_inventory_invalid");
      continue;
    }
    if (w.node_count >= w.max_nodes) {
      w.limits_exceeded = true;
      add_reason(w, "limits_nodes");
      break;
    }
    if (!walk_node(w, nodes[i], expected_ctx, w.node_count, graph_index, depth,
                   parent)) {
      break;  // output budget exhausted
    }
  }
  w.stack.pop_back();
}

extern "C" int s5ga_abi(void) { return FLASH_S5_GRAPH_AUDIT_ABI; }

static int inspect_impl(uint64_t graph_handle, uint64_t stream_handle,
                            uint64_t expected_green_handle, uint32_t max_nodes,
                            uint32_t max_depth, uint64_t max_bytes, char *out,
                            uint64_t out_cap, uint64_t *out_len) {
  if (out_len) *out_len = 0;
  if (graph_handle == 0 || stream_handle == 0 || expected_green_handle == 0)
    return S5GA_BAD_ARGS;
  if (max_nodes == 0 || max_nodes > 1000000u || max_depth > 64u)
    return S5GA_BAD_ARGS;
  if (max_bytes < 256 || max_bytes > 64u * 1024u * 1024u || out == nullptr || out_cap == 0 || out_len == nullptr)
    return S5GA_BAD_ARGS;

  CUgraph graph = (CUgraph)(uintptr_t)graph_handle;
  CUstream stream = (CUstream)(uintptr_t)stream_handle;
  CUgreenCtx expected_green = (CUgreenCtx)(uintptr_t)expected_green_handle;

  CUgreenCtx observed_green = nullptr;
  CUresult rc = cuStreamGetGreenCtx(stream, &observed_green);
  if (rc != CUDA_SUCCESS) return S5GA_GREEN_QUERY_FAILED;
  if (observed_green == nullptr) return S5GA_GREEN_MISSING;
  if (observed_green != expected_green) return S5GA_GREEN_MISMATCH;
  CUcontext expected_ctx = nullptr;
  rc = cuCtxFromGreenCtx(&expected_ctx, expected_green);
  if (rc != CUDA_SUCCESS || expected_ctx == nullptr)
    return S5GA_CONTEXT_CONVERT_FAILED;

  CUcontext stream_ctx = nullptr;
  rc = cuStreamGetCtx(stream, &stream_ctx);
  if (rc != CUDA_SUCCESS || stream_ctx == nullptr)
    return S5GA_STREAM_CONTEXT_QUERY_FAILED;
  if (stream_ctx != expected_ctx) return S5GA_STREAM_CONTEXT_MISMATCH;

  bool stream_sm_known = false;
  const char *stream_sm_query = "unknown";
  CUresult stream_sm_rc = CUDA_SUCCESS;
  unsigned stream_sm_count = 0;
  CUdevResource stream_resource{};
  stream_sm_rc = cuCtxGetDevResource(stream_ctx, &stream_resource,
                                     CU_DEV_RESOURCE_TYPE_SM);
  if (stream_sm_rc == CUDA_SUCCESS) {
    stream_sm_known = true;
    stream_sm_query = "ok";
    stream_sm_count = stream_resource.sm.smCount;
  } else if (stream_sm_rc == CUDA_ERROR_NOT_SUPPORTED) {
    stream_sm_query = "unsupported";
  } else {
    stream_sm_query = "error";
  }

  Walk w;
  w.max_nodes = max_nodes;
  w.max_depth = max_depth;
  uint64_t budget = max_bytes < out_cap ? max_bytes : out_cap;
  // Reserve envelope room so a report that fills the node budget can still
  // fit the caller's buffer; the final size check below is authoritative.
  w.budget = budget > 2048 ? budget - 1024 : (budget > 2 ? budget / 2 : 1);
  bool ok = true;
  walk_graph(w, graph, expected_ctx, 0, 0, -1, true, ok);
  if (!ok) return S5GA_GRAPH_QUERY_FAILED;

  if (!stream_sm_known) add_reason(w, "stream_sm_unknown");

  bool collective_complete = (w.collective_unknown == 0) &&
                             !w.limits_exceeded && w.unsupported == 0;
  bool complete = !w.limits_exceeded && w.unsupported == 0 && w.unknown == 0 &&
                  stream_sm_known && collective_complete && w.reasons.empty();

  std::string s;
  s.reserve(w.nodes.size() + 1024);
  s += "{\"schema\":"; put_q(s, kSchema);
  s += ",\"root_graph_nodes\":"; put_u(s, w.root_nodes);
  s += ",\"child_graphs\":"; put_u(s, w.child_graphs);
  s += ",\"counts\":{\"kernel\":"; put_u(s, w.n_kernel);
  s += ",\"memcpy\":"; put_u(s, w.n_memcpy);
  s += ",\"memset\":"; put_u(s, w.n_memset);
  s += ",\"graph\":"; put_u(s, w.n_graph);
  s += ",\"other\":"; put_u(s, w.n_other);
  s += ",\"unknown\":"; put_u(s, w.n_unknown_type);
  s += "},\"node_count\":"; put_u(s, w.node_count);
  s += ",\"unknown_nodes\":"; put_u(s, w.unknown);
  s += ",\"unsupported_nodes\":"; put_u(s, w.unsupported);
  s += ",\"max_depth\":"; put_u(s, w.max_depth_seen);
  s += ",\"limits_exceeded\":"; put_b(s, w.limits_exceeded);
  s += ",\"stream_context\":{\"green_query\":\"ok\",\"green_matches\":true";
  s += ",\"context_convert\":\"ok\",\"sm_query\":";
  put_q(s, stream_sm_query);
  s += ",\"sm_query_rc\":"; put_u(s, (unsigned long long)stream_sm_rc);
  s += ",\"sm_count\":";
  if (stream_sm_known) put_u(s, stream_sm_count); else s += "null";
  s += "},\"collective\":{\"basis\":\"function_name_substring\"";
  s += ",\"confirmed\":false,\"complete\":"; put_b(s, collective_complete);
  s += ",\"unknown_nodes\":"; put_u(s, w.collective_unknown);
  s += ",\"candidates\":{\"allreduce\":"; put_u(s, w.c_allreduce);
  s += ",\"allgather\":"; put_u(s, w.c_allgather);
  s += ",\"reducescatter\":"; put_u(s, w.c_reducescatter);
  s += ",\"broadcast\":"; put_u(s, w.c_broadcast);
  s += ",\"other_collective\":"; put_u(s, w.c_other_collective);
  s += ",\"none\":"; put_u(s, w.c_none);
  s += ",\"unavailable\":"; put_u(s, w.c_unavailable);
  s += "}},\"reasons\":[";
  for (size_t i = 0; i < w.reasons.size(); ++i) {
    if (i) s.push_back(',');
    put_q(s, w.reasons[i].c_str());
  }
  s += "],\"complete\":"; put_b(s, complete);
  s += ",\"nodes\":["; s += w.nodes; s += "]}";

  *out_len = (uint64_t)s.size();
  if (s.size() > budget) return S5GA_TRUNCATED;
  memcpy(out, s.data(), s.size());
  if (s.size() < out_cap) out[s.size()] = '\0';
  return S5GA_OK;
}

// Never unwind a host allocation/library exception across the C/ctypes boundary.
extern "C" int s5ga_inspect(uint64_t graph, uint64_t stream,
                            uint64_t expected_green, uint32_t max_nodes,
                            uint32_t max_depth, uint64_t max_bytes, char *out,
                            uint64_t out_cap, uint64_t *out_len) {
  try {
    return inspect_impl(graph, stream, expected_green, max_nodes, max_depth,
                        max_bytes, out, out_cap, out_len);
  } catch (...) {
    if (out_len) *out_len = 0;
    return S5GA_INTERNAL_ERROR;
  }
}
