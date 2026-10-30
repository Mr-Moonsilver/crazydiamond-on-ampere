// CPU fake CUDA driver + CUPTI used only by the replay-observer tests.
//
// It models the pinned seam's data flow: the observer registers activity
// callbacks, requests buffers, receives complete() with a buffer it owns, and
// parses records through cuptiActivityGetNextRecord.  All handles and IDs are
// fabricated.  Nothing here loads a driver or launches work.

#include <cstdint>
#include <cstring>
#include <set>
#include <thread>
#include <vector>
#include <unistd.h>
#include <sys/syscall.h>

#include "cuda.h"
#include "cupti_activity.h"
#include "cupti_callbacks.h"
#include "cupti_result.h"
#include "cupti_version.h"

namespace {

struct ContextEntry {
  uint64_t handle = 0;
  uint32_t id = 0;
  uint64_t green = 0;
  uint32_t sms = 0;
};

struct GraphEntry {
  uint64_t handle = 0;
  uint32_t id = 0;
  std::vector<uint64_t> nodes;
};

struct NodeEntry {
  uint64_t handle = 0;
  uint64_t id = 0;
  int32_t type = 0;
  int params_rc=0;
  CUDA_KERNEL_NODE_PARAMS params{reinterpret_cast<void*>(1),nullptr,nullptr};
};

struct ExecEntry {
  uint64_t handle = 0;
  uint32_t id = 0;
};

struct HeldBuffer {
  uint8_t *buffer = nullptr;
  size_t size = 0;
  std::vector<uint8_t> pending;
};

std::vector<ContextEntry> g_contexts;
std::vector<GraphEntry> g_graphs;
std::vector<NodeEntry> g_nodes;
std::vector<ExecEntry> g_execs;

CUpti_BuffersCallbackRequestFunc g_request = nullptr;
CUpti_BuffersCallbackCompleteFunc g_complete = nullptr;
CUpti_CallbackFunc g_callback = nullptr;
void *g_userdata = nullptr;
CUpti_SubscriberHandle g_subscriber = nullptr;
std::set<CUpti_CallbackId> g_resource_callbacks;
std::set<CUpti_ActivityKind> g_activity_kinds;

std::vector<uint8_t> g_pending;
std::vector<HeldBuffer> g_held;

uint64_t g_next_handle = 0x100000;
uint64_t g_next_green = 0x500000;
uint64_t g_timestamp = 1;
size_t g_dropped = 0;
uint32_t g_version = CUPTI_API_VERSION;

int g_version_rc = 0;
int g_subscribe_rc = 0;
int g_unsubscribe_rc = 0;
int g_enable_callback_rc = 0;
int g_enable_activity_rc = 0;
int g_register_rc = 0;
int g_flush_rc = 0;
int g_timestamp_rc = 0;
int g_drop_rc = 0;
int g_disable_activity_rc = 0;
int g_deliver_on_flush = 1;
int g_reenter = 0;
int g_reentered = 0;
int g_hold_next = 0;
int g_foreign_subscriber = 0;
int g_threaded_flush = 0;
size_t g_refused_requests = 0;
size_t g_foreign_completions = 0;
uint8_t g_foreign_buffer[65536];

uint8_t *g_current_buffer = nullptr;
size_t g_current_valid = 0;

uint64_t alloc_handle() { return g_next_handle++; }

ContextEntry *find_context(uint64_t handle) {
  for (ContextEntry &entry : g_contexts) {
    if (entry.handle == handle) return &entry;
  }
  return nullptr;
}

GraphEntry *find_graph(uint64_t handle) {
  for (GraphEntry &entry : g_graphs) {
    if (entry.handle == handle) return &entry;
  }
  return nullptr;
}

NodeEntry *find_node(uint64_t handle) {
  for (NodeEntry &entry : g_nodes) {
    if (entry.handle == handle) return &entry;
  }
  return nullptr;
}

ExecEntry *find_exec(uint64_t handle) {
  for (ExecEntry &entry : g_execs) {
    if (entry.handle == handle) return &entry;
  }
  return nullptr;
}

void CUPTIAPI foreign_buffer_requested(uint8_t **buffer, size_t *size,
                                         size_t *max_num_records) {
  if (buffer != nullptr) *buffer = g_foreign_buffer;
  if (size != nullptr) *size = sizeof(g_foreign_buffer);
  if (max_num_records != nullptr) *max_num_records = 0;
}

void CUPTIAPI foreign_buffer_completed(CUcontext context, uint32_t stream_id,
                                       uint8_t *buffer, size_t size,
                                       size_t valid_size) {
  (void)context;
  (void)stream_id;
  (void)buffer;
  (void)size;
  (void)valid_size;
  ++g_foreign_completions;
}

size_t round16(size_t value) { return (value + 15u) & ~static_cast<size_t>(15u); }

void queue_bytes(const void *data, size_t size) {
  const size_t offset = round16(g_pending.size());
  const size_t end = offset + size;
  g_pending.resize(round16(end), 0);
  std::memcpy(g_pending.data() + offset, data, size);
}

template <typename T>
void queue_record(const T &record) {
  queue_bytes(&record, sizeof(record));
}

int deliver_pending() {
  if (g_complete == nullptr || g_request == nullptr || g_pending.empty()) {
    return 0;
  }
  int delivered = 0;
  size_t offset = 0;
  while (offset < g_pending.size()) {
    uint8_t *buffer = nullptr;
    size_t size = 0;
    size_t max_records = 0;
    g_request(&buffer, &size, &max_records);
    if (buffer == nullptr || size == 0) {
      ++g_refused_requests;
      break;
    }
    if (g_hold_next != 0) {
      g_hold_next = 0;
      HeldBuffer held;
      held.buffer = buffer;
      held.size = size;
      held.pending.assign(g_pending.begin() + offset, g_pending.end());
      g_held.push_back(std::move(held));
      // The held buffer owns these records. A later flush must not deliver a
      // second copy while its original completion is still withheld.
      g_pending.clear();
      offset = 0;
      break;
    }
    size_t take = 0;
    while (offset + take < g_pending.size()) {
      CUpti_Activity header;
      std::memcpy(&header, g_pending.data() + offset + take, sizeof(header));
      const size_t next = round16(header.record_size);
      if (next == 0 || take + next > size) break;
      take += next;
    }
    if (take == 0) {
      g_complete(nullptr, 0, buffer, size, 0);
      ++g_refused_requests;
      break;
    }
    std::memcpy(buffer, g_pending.data() + offset, take);
    g_complete(nullptr, 0, buffer, size, take);
    offset += take;
    ++delivered;
  }
  if (offset > 0) {
    g_pending.erase(g_pending.begin(), g_pending.begin() + offset);
  }
  return delivered;
}

}  // namespace

extern "C" {

void stub_reset(void) {
  g_contexts.clear();
  g_graphs.clear();
  g_nodes.clear();
  g_execs.clear();
  g_pending.clear();
  g_held.clear();
  g_request = nullptr;
  g_complete = nullptr;
  g_callback = nullptr;
  g_userdata = nullptr;
  g_subscriber = nullptr;
  g_resource_callbacks.clear();
  g_activity_kinds.clear();
  g_next_handle = 0x100000;
  g_next_green = 0x500000;
  g_timestamp = 1;
  g_dropped = 0;
  g_version = CUPTI_API_VERSION;
  g_version_rc = 0;
  g_subscribe_rc = 0;
  g_unsubscribe_rc = 0;
  g_enable_callback_rc = 0;
  g_enable_activity_rc = 0;
  g_register_rc = 0;
  g_flush_rc = 0;
  g_timestamp_rc = 0;
  g_drop_rc = 0;
  g_disable_activity_rc = 0;
  g_deliver_on_flush = 1;
  g_reenter = 0;
  g_reentered = 0;
  g_hold_next = 0;
  g_foreign_subscriber = 0;
  g_threaded_flush = 0;
  g_refused_requests = 0;
  g_foreign_completions = 0;
  g_current_buffer = nullptr;
  g_current_valid = 0;
}

void stub_set_version(uint32_t version) { g_version = version; }
int stub_activity_enabled(int kind) {
  return g_activity_kinds.count(static_cast<CUpti_ActivityKind>(kind)) != 0;
}
void stub_set_version_rc(int rc) { g_version_rc = rc; }
void stub_set_subscribe_rc(int rc) { g_subscribe_rc = rc; }
void stub_set_unsubscribe_rc(int rc) { g_unsubscribe_rc = rc; }
void stub_set_enable_callback_rc(int rc) { g_enable_callback_rc = rc; }
void stub_set_enable_activity_rc(int rc) { g_enable_activity_rc = rc; }
void stub_set_register_rc(int rc) { g_register_rc = rc; }
void stub_set_flush_rc(int rc) { g_flush_rc = rc; }
void stub_set_timestamp_rc(int rc) { g_timestamp_rc = rc; }
void stub_set_drop_rc(int rc) { g_drop_rc = rc; }
void stub_set_deliver_on_flush(int enabled) { g_deliver_on_flush = enabled; }
void stub_set_timestamp(uint64_t timestamp) { g_timestamp = timestamp; }
void stub_set_reenter(int enabled) { g_reenter = enabled; }
void stub_set_foreign_subscriber(int enabled) { g_foreign_subscriber = enabled; }
int stub_has_subscriber(void) { return g_subscriber != nullptr; }
void stub_set_dropped(size_t count) { g_dropped = count; }
void stub_set_hold_next(int enabled) { g_hold_next = enabled; }
void stub_set_disable_activity_rc(int rc) { g_disable_activity_rc = rc; }
void stub_set_threaded_flush(int enabled) { g_threaded_flush = enabled; }
void stub_install_foreign_callbacks(void) {
  g_request = foreign_buffer_requested;
  g_complete = foreign_buffer_completed;
}

size_t stub_held_count(void) { return g_held.size(); }
size_t stub_refused_requests(void) { return g_refused_requests; }
size_t stub_pending_count(void) { return g_pending.size(); }
size_t stub_foreign_completions(void) { return g_foreign_completions; }

uint64_t stub_new_context(uint32_t context_id) {
  ContextEntry entry;
  entry.handle = alloc_handle();
  entry.id = context_id;
  g_contexts.push_back(entry);
  return entry.handle;
}

uint64_t stub_new_green(uint32_t context_id, uint32_t sms) {
  ContextEntry entry;
  entry.handle = alloc_handle();
  entry.green = g_next_green++;
  entry.id = context_id;
  entry.sms = sms;
  g_contexts.push_back(entry);
  return entry.green;
}

uint64_t stub_context_of_green(uint64_t green) {
  for (ContextEntry &entry : g_contexts) {
    if (entry.green == green) return entry.handle;
  }
  return 0;
}

uint64_t stub_new_graph(uint32_t graph_id) {
  GraphEntry entry;
  entry.handle = alloc_handle();
  entry.id = graph_id;
  g_graphs.push_back(entry);
  return entry.handle;
}

uint64_t stub_new_node(uint64_t node_id, int node_type) {
  NodeEntry entry;
  entry.handle = alloc_handle();
  entry.id = node_id;
  entry.type = node_type;
  g_nodes.push_back(entry);
  return entry.handle;
}

uint64_t stub_new_exec(uint32_t exec_id) {
  ExecEntry entry;
  entry.handle = alloc_handle();
  entry.id = exec_id;
  g_execs.push_back(entry);
  return entry.handle;
}

void stub_graph_add_node(uint64_t graph, uint64_t node) {
  GraphEntry *entry = find_graph(graph);
  if (entry != nullptr) entry->nodes.push_back(node);
}

void stub_emit_context(uint32_t context_id, uint32_t is_green, uint32_t sms,
                       uint32_t parent) {
  CUpti_ActivityContext3 record;
  std::memset(&record, 0, sizeof(record));
  record.kind = CUPTI_ACTIVITY_KIND_CONTEXT;
  record.record_size = sizeof(record);
  record.contextId = context_id;
  record.deviceId = 0;
  record.parentContextId = parent;
  record.isGreenContext = static_cast<uint8_t>(is_green);
  record.numMultiprocessors = static_cast<uint16_t>(sms);
  queue_record(record);
}

void stub_emit_kernel(uint32_t context_id, uint32_t graph_id, uint64_t node_id,
                      uint64_t start, uint64_t end, uint64_t completed) {
  CUpti_ActivityKernel10 record;
  std::memset(&record, 0, sizeof(record));
  record.kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
  record.record_size = sizeof(record);
  record.contextId = context_id;
  record.deviceId = 0;
  record.graphId = graph_id;
  record.graphNodeId = node_id;
  record.start = start;
  record.end = end;
  record.completed = completed;
  record.name="fixture_kernel";record.gridX=4;record.gridY=1;record.gridZ=1;
  record.blockX=32;record.blockY=1;record.blockZ=1;
  queue_record(record);
}

void stub_emit_copy(uint32_t context_id, uint32_t graph_id, uint64_t node_id,
                    uint64_t start, uint64_t end) {
  CUpti_ActivityMemcpy6 record;
  std::memset(&record, 0, sizeof(record));
  record.kind = CUPTI_ACTIVITY_KIND_MEMCPY;
  record.record_size = sizeof(record);
  record.contextId = context_id;
  record.graphId = graph_id;
  record.graphNodeId = node_id;
  record.start = start;
  record.end = end;
  queue_record(record);
}

void stub_emit_memset(uint32_t context_id, uint32_t graph_id, uint64_t node_id,
                      uint64_t start, uint64_t end) {
  CUpti_ActivityMemset4 record;
  std::memset(&record, 0, sizeof(record));
  record.kind = CUPTI_ACTIVITY_KIND_MEMSET;
  record.record_size = sizeof(record);
  record.contextId = context_id;
  record.graphId = graph_id;
  record.graphNodeId = node_id;
  record.start = start;
  record.end = end;
  queue_record(record);
}

void stub_emit_unknown(uint32_t kind) {
  struct UnknownRecord {
    uint32_t kind;
    uint32_t record_size;
    uint32_t padding;
  } record;
  record.kind = kind;
  record.record_size = sizeof(record);
  record.padding = 0;
  queue_record(record);
}

void stub_emit_corrupt(void) {
  struct CorruptRecord {
    uint32_t kind;
    uint32_t record_size;
    uint32_t padding;
  } record;
  record.kind = CUPTI_ACTIVITY_KIND_CONTEXT;
  record.record_size = 4;
  record.padding = 0;
  queue_record(record);
}

int stub_deliver(void) { return deliver_pending(); }

void stub_release_held(void) {
  for (const HeldBuffer &held : g_held) {
    const size_t valid = held.pending.size() < held.size
                             ? held.pending.size()
                             : held.size;
    if (valid > 0) std::memcpy(held.buffer, held.pending.data(), valid);
    if (g_complete != nullptr) {
      g_complete(nullptr, 0, held.buffer, held.size, valid);
    }
  }
  g_held.clear();
}

void stub_resource_graph(uint32_t cbid, uint64_t graph, uint64_t original_graph,
                         uint64_t node, uint64_t original_node,
                         int node_type, uint64_t exec) {
  if (g_callback == nullptr || g_resource_callbacks.count(cbid) == 0) return;
  CUpti_GraphData data;
  std::memset(&data, 0, sizeof(data));
  data.graph = reinterpret_cast<CUgraph>(static_cast<uintptr_t>(graph));
  data.originalGraph =
      reinterpret_cast<CUgraph>(static_cast<uintptr_t>(original_graph));
  data.node = reinterpret_cast<CUgraphNode>(static_cast<uintptr_t>(node));
  data.originalNode =
      reinterpret_cast<CUgraphNode>(static_cast<uintptr_t>(original_node));
  data.nodeType = static_cast<CUgraphNodeType>(node_type);
  data.graphExec = reinterpret_cast<CUgraphExec>(static_cast<uintptr_t>(exec));
  // Real CUPTI supplies a resource envelope, not GraphData as cbdata.
  // Padding makes the old erroneous cast deterministic in the CPU fixture:
  // it queries the descriptor address as a node instead of reading beyond it.
  struct Envelope {
    CUpti_ResourceData resource;
    unsigned char padding[sizeof(CUpti_GraphData)];
  } envelope{};
  envelope.resource.resourceDescriptor = &data;
  g_callback(g_userdata, CUPTI_CB_DOMAIN_RESOURCE, cbid, &envelope.resource);
}

void stub_resource_missing_descriptor(uint32_t cbid) {
  if (g_callback == nullptr || g_resource_callbacks.count(cbid) == 0) return;
  struct Envelope {
    CUpti_ResourceData resource;
    unsigned char padding[sizeof(CUpti_GraphData)];
  } envelope{};
  g_callback(g_userdata, CUPTI_CB_DOMAIN_RESOURCE, cbid, &envelope.resource);
}

void stub_resource_context(uint32_t cbid, uint64_t context) {
  if (g_callback == nullptr || g_resource_callbacks.count(cbid) == 0) return;
  CUpti_ResourceData data;
  std::memset(&data, 0, sizeof(data));
  data.context = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(context));
  g_callback(g_userdata, CUPTI_CB_DOMAIN_RESOURCE, cbid, &data);
}

}  // extern "C"

extern "C" {

CUresult cuCtxFromGreenCtx(CUcontext *pContext, CUgreenCtx hCtx) {
  if (pContext == nullptr || hCtx == nullptr) return 1;
  const uint64_t handle = reinterpret_cast<uintptr_t>(hCtx);
  for (ContextEntry &entry : g_contexts) {
    if (entry.green == handle) {
      *pContext = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(entry.handle));
      return 0;
    }
  }
  return 1;
}

CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes) {
  if (numNodes == nullptr) return 1;
  GraphEntry *entry =
      find_graph(reinterpret_cast<uintptr_t>(hGraph));
  if (entry == nullptr) return 1;
  if (nodes == nullptr) {
    *numNodes = entry->nodes.size();
    return 0;
  }
  if (*numNodes < entry->nodes.size()) return 1;
  for (size_t i = 0; i < entry->nodes.size(); ++i) {
    nodes[i] = reinterpret_cast<CUgraphNode>(
        static_cast<uintptr_t>(entry->nodes[i]));
  }
  *numNodes = entry->nodes.size();
  return 0;
}

void stub_node_params(uint64_t node,int rc,int func,int kern,uint64_t ctx){
 auto*n=find_node(node);if(!n)return;n->params_rc=rc;
 n->params={func?reinterpret_cast<void*>(1):nullptr,kern?reinterpret_cast<void*>(2):nullptr,reinterpret_cast<CUcontext>(ctx)};
}
CUresult cuGraphKernelNodeGetParams(CUgraphNode node,CUDA_KERNEL_NODE_PARAMS*p){
 auto*n=find_node(reinterpret_cast<uintptr_t>(node));if(!n||!p)return 1;
 *p=n->params;return n->params_rc;
}
CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
  if (type == nullptr) return 1;
  NodeEntry *entry = find_node(reinterpret_cast<uintptr_t>(hNode));
  if (entry == nullptr) return 1;
  *type = static_cast<CUgraphNodeType>(entry->type);
  return 0;
}

CUptiResult cuptiGetVersion(uint32_t *version) {
  if (g_version_rc != 0) return static_cast<CUptiResult>(g_version_rc);
  if (version == nullptr) return CUPTI_ERROR_INVALID_PARAMETER;
  *version = g_version;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc request,
    CUpti_BuffersCallbackCompleteFunc complete) {
  if (g_register_rc != 0) return static_cast<CUptiResult>(g_register_rc);
  g_request = request;
  g_complete = complete;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiSubscribe(CUpti_SubscriberHandle *subscriber,
                           CUpti_CallbackFunc callback, void *userdata) {
  if (g_foreign_subscriber != 0 || g_subscriber != nullptr) {
    return CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED;
  }
  if (g_subscribe_rc != 0) return static_cast<CUptiResult>(g_subscribe_rc);
  if (subscriber == nullptr || callback == nullptr) {
    return CUPTI_ERROR_INVALID_PARAMETER;
  }
  g_subscriber = reinterpret_cast<CUpti_SubscriberHandle>(
      static_cast<uintptr_t>(0x900000));
  *subscriber = g_subscriber;
  g_callback = callback;
  g_userdata = userdata;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber) {
  (void)subscriber;
  if (g_unsubscribe_rc != 0) return static_cast<CUptiResult>(g_unsubscribe_rc);
  g_subscriber = nullptr;
  g_callback = nullptr;
  g_userdata = nullptr;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiEnableCallback(uint32_t enable,
                                CUpti_SubscriberHandle subscriber,
                                CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid) {
  (void)subscriber;
  if (g_enable_callback_rc != 0) {
    return static_cast<CUptiResult>(g_enable_callback_rc);
  }
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    if (enable) g_resource_callbacks.insert(cbid);
    else g_resource_callbacks.erase(cbid);
  }
  return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind) {
  if (g_enable_activity_rc != 0) {
    return static_cast<CUptiResult>(g_enable_activity_rc);
  }
  g_activity_kinds.insert(kind);
  return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind) {
  if (g_disable_activity_rc != 0) {
    return static_cast<CUptiResult>(g_disable_activity_rc);
  }
  g_activity_kinds.erase(kind);
  return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityGetNextRecord(uint8_t *buffer,
                                       size_t validBufferSizeBytes,
                                       CUpti_Activity **record) {
  if (buffer == nullptr || record == nullptr) {
    return CUPTI_ERROR_INVALID_PARAMETER;
  }
  size_t offset = 0;
  if (*record != nullptr) {
    const auto *current = reinterpret_cast<const CUpti_Activity *>(*record);
    const size_t current_offset =
        reinterpret_cast<const uint8_t *>(current) - buffer;
    offset = round16(current_offset + current->record_size);
  }
  if (offset >= validBufferSizeBytes) return CUPTI_ERROR_MAX_LIMIT_REACHED;
  if (offset + sizeof(CUpti_Activity) > validBufferSizeBytes) {
    return CUPTI_ERROR_INVALID_KIND;
  }
  auto *candidate = reinterpret_cast<CUpti_Activity *>(buffer + offset);
  if (candidate->record_size < sizeof(CUpti_Activity) ||
      offset + candidate->record_size > validBufferSizeBytes) {
    return CUPTI_ERROR_INVALID_KIND;
  }
  *record = candidate;
  if (g_reenter != 0 && g_reentered == 0) {
    g_reentered = 1;
    g_current_buffer = buffer;
    g_current_valid = validBufferSizeBytes;
    if (g_complete != nullptr) {
      g_complete(nullptr, 0, g_current_buffer, g_current_valid, g_current_valid);
    }
  }
  return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext context,
                                              uint32_t streamId,
                                              size_t *dropped) {
  (void)context;
  (void)streamId;
  if (g_drop_rc != 0) return static_cast<CUptiResult>(g_drop_rc);
  if (dropped == nullptr) return CUPTI_ERROR_INVALID_PARAMETER;
  *dropped = g_dropped;
  g_dropped = 0;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiActivityFlushAll(uint32_t flag) {
  (void)flag;
  if (g_flush_rc != 0) return static_cast<CUptiResult>(g_flush_rc);
  if (g_deliver_on_flush != 0) {
    if (g_threaded_flush != 0) {
      std::thread deliver([] { deliver_pending(); });
      deliver.join();
    } else {
      deliver_pending();
    }
  }
  return CUPTI_SUCCESS;
}

CUptiResult cuptiGetTimestamp(uint64_t *timestamp) {
  if (g_timestamp_rc != 0) {
    return static_cast<CUptiResult>(g_timestamp_rc);
  }
  if (timestamp == nullptr) return CUPTI_ERROR_INVALID_PARAMETER;
  *timestamp = g_timestamp;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiGetContextId(CUcontext context, uint32_t *contextId) {
  if (context == nullptr || contextId == nullptr) {
    return CUPTI_ERROR_INVALID_PARAMETER;
  }
  ContextEntry *entry = find_context(reinterpret_cast<uintptr_t>(context));
  if (entry == nullptr) return CUPTI_ERROR_NOT_INITIALIZED;
  *contextId = entry->id;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiGetGraphId(CUgraph graph, uint32_t *pId) {
  if (graph == nullptr || pId == nullptr) {
    return CUPTI_ERROR_INVALID_PARAMETER;
  }
  GraphEntry *entry = find_graph(reinterpret_cast<uintptr_t>(graph));
  if (entry == nullptr) return CUPTI_ERROR_NOT_INITIALIZED;
  *pId = entry->id;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiGetGraphExecId(CUgraphExec graphExec, uint32_t *pId) {
  if (graphExec == nullptr || pId == nullptr) {
    return CUPTI_ERROR_INVALID_PARAMETER;
  }
  ExecEntry *entry = find_exec(reinterpret_cast<uintptr_t>(graphExec));
  if (entry == nullptr) return CUPTI_ERROR_NOT_INITIALIZED;
  *pId = entry->id;
  return CUPTI_SUCCESS;
}

CUptiResult cuptiGetGraphNodeId(CUgraphNode node, uint64_t *nodeId) {
  if (node == nullptr || nodeId == nullptr) {
    return CUPTI_ERROR_INVALID_PARAMETER;
  }
  NodeEntry *entry = find_node(reinterpret_cast<uintptr_t>(node));
  if (entry == nullptr) return CUPTI_ERROR_NOT_INITIALIZED;
  *nodeId = entry->id;
  return CUPTI_SUCCESS;
}

}  // extern "C"

namespace {thread_local std::vector<uint64_t> stub_external_stack;}
extern "C" {
CUptiResult cuptiActivityPushExternalCorrelationId(CUpti_ExternalCorrelationKind,uint64_t id){stub_external_stack.push_back(id);return CUPTI_SUCCESS;}
CUptiResult cuptiActivityPopExternalCorrelationId(CUpti_ExternalCorrelationKind,uint64_t*out){
 if(stub_external_stack.empty())return CUPTI_ERROR_INVALID_PARAMETER;
 *out=stub_external_stack.back();stub_external_stack.pop_back();return CUPTI_SUCCESS;}
CUptiResult cuptiGetCallbackName(CUpti_CallbackDomain d,CUpti_CallbackId cb,const char**out){
 *out=cb==1?(d==CUPTI_CB_DOMAIN_DRIVER_API?"cuLaunchKernel":"cudaLaunchKernel_v7000"):
      cb==2?(d==CUPTI_CB_DOMAIN_DRIVER_API?"cuGraphLaunch":"cudaGraphLaunch"):"cuStreamSynchronize";
 return CUPTI_SUCCESS;}
void stub_emit_correlated(uint32_t c,uint64_t offer,uint32_t domain,uint32_t thread,
 uint64_t start,uint64_t end,uint32_t ctx,uint32_t graph,uint64_t node,uint32_t activity){
 CUpti_ActivityAPI a{};a.kind=static_cast<CUpti_ActivityKind>(domain);a.record_size=sizeof(a);
 a.cbid=1;a.correlationId=c;a.threadId=thread;a.start=start;a.end=end;
 queue_record(a);
 CUpti_ActivityExternalCorrelation e{};e.kind=CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION;e.record_size=sizeof(e);
 e.externalKind=CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;e.correlationId=c;e.externalId=offer;queue_record(e);
 if(activity){CUpti_ActivityKernel10 k{};k.kind=CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;k.record_size=sizeof(k);
 k.contextId=ctx;k.graphId=graph;k.graphNodeId=node;k.correlationId=c;k.streamId=9;k.name="correlated_fixture";
 k.start=end+1;k.end=end+10;k.completed=end+10;queue_record(k);}
}
}

static int stub_thread_id_mode=0;
extern "C" void stub_set_thread_id_mode(int mode){stub_thread_id_mode=mode;}
extern "C" CUptiResult cuptiGetThreadIdType(CUpti_ActivityThreadIdType*out){if(!out)return CUPTI_ERROR_INVALID_PARAMETER;*out=static_cast<CUpti_ActivityThreadIdType>(stub_thread_id_mode);return CUPTI_SUCCESS;}

// One graph launch correlation shared by both retained kernel-node records.
extern "C" void stub_emit_correlated_graph(uint32_t c,uint64_t offer,uint32_t thread,
 uint32_t ctx,uint32_t graph,uint64_t node1,uint64_t node2){
 CUpti_ActivityAPI a{};a.kind=CUPTI_ACTIVITY_KIND_RUNTIME;a.record_size=sizeof(a);
 a.cbid=2;a.correlationId=c;a.threadId=thread;a.start=1600;a.end=1650;queue_record(a);
 CUpti_ActivityExternalCorrelation e{};e.kind=CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION;e.record_size=sizeof(e);
 e.externalKind=CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;e.correlationId=c;e.externalId=offer;queue_record(e);
 for(uint64_t node:{node1,node2}){
  CUpti_ActivityKernel10 k{};k.kind=CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;k.record_size=sizeof(k);
  k.contextId=ctx;k.graphId=graph;k.graphNodeId=node;k.correlationId=c;k.streamId=9;k.name="graph_fixture";
  k.start=1700;k.end=1800;k.completed=1800;queue_record(k);
 }
}
