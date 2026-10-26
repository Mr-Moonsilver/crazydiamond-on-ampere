// E3 actual-replay context observer (own C ABI 1, default off).
//
// This translation unit is a bounded CUPTI activity observer for the retained
// green-context decode bank.  It is compiled CPU-only against the pinned CUDA
// 13 / CUPTI 13 headers and is never loaded, started or exercised by import.
//
// Evidence boundary:
//   * A window receipt can join every observed CONCURRENT_KERNEL / MEMCPY /
//     MEMSET execution of the bound retained graphs to the owner's expected
//     green context (via cuCtxFromGreenCtx + cuptiGetContextId) and to the
//     static node inventory captured before the window.
//   * It cannot prove that the scheduler was paused, that the caller's drain
//     proof is true, that the caller's worker/engine incarnation is the one
//     observed, or that instrumentation overhead is acceptable.  Those stay
//     caller obligations in the Python receipt.
//   * Missing context activity, unknown IDs, clone/update/destroy ambiguity,
//     refused buffers, dropped records, ring overflow, parser failures and
//     boundary-spanning records are retained as INCONCLUSIVE evidence.
//     A known wrong context or a known wrong green SM count is FAIL evidence.
//   * Nothing here launches, replays, allocates on a device, synchronises,
//     resets CUPTI globally, or reads argument pointees/tensor contents.
//     Launch reports contain names/configuration for private worker comparison;
//     Python redacts policy pointers before any public receipt is emitted.
//
// Storage is allocated once in s5ro_create().  Callbacks copy only the fixed
// facts below into preallocated rings while the CUPTI data is valid; they do
// not allocate, call Python, or perform CUDA work beyond the read-only ID
// queries that map retained originals to clones/executables.

#include <cuda.h>
#include <cupti_activity.h>
#include <cupti_callbacks.h>
#include <cupti_result.h>
#include <cupti_version.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <map>
#include <tuple>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/syscall.h>
#include <pthread.h>

namespace {

constexpr uint32_t kAbi = 1;
constexpr uint32_t kMaxBuffers = 64;
constexpr uint64_t kMinBufferBytes = 64ull * 1024ull;
constexpr uint64_t kMaxBufferBytes = 64ull * 1024ull * 1024ull;
constexpr uint32_t kMaxContextRecords = 1000000u;
constexpr uint32_t kMaxExecRecords = 8000000u;
constexpr uint32_t kMaxEventRecords = 4000000u;
constexpr uint32_t kMaxGraphs = 1024u;
constexpr uint32_t kMaxNodesPerGraph = 1000000u;
constexpr uint32_t kMaxNodesTotal = 2000000u;
constexpr uint32_t kMaxLinks = 4000000u;
constexpr uint64_t kMaxReservedBytes = 512ull * 1024ull * 1024ull;
constexpr uint32_t kMaxMissingIndices = 64u;
constexpr uint32_t kMaxGraphPasses = 32u;
constexpr uint32_t kNoSlot = 0xffffffffu;

constexpr uint64_t kTsUnknown = CUPTI_TIMESTAMP_UNKNOWN;

enum Status : int {
  S5RO_OK = 0,
  S5RO_TRUNCATED = 1,
  S5RO_BAD_ARGS = 2,
  S5RO_UNKNOWN_SESSION = 3,
  S5RO_STATE = 4,
  S5RO_VERSION_QUERY_FAILED = 5,
  S5RO_VERSION_MISMATCH = 6,
  S5RO_ACTIVITY_REGISTER_FAILED = 7,
  S5RO_SUBSCRIBE_FAILED = 8,
  S5RO_SUBSCRIBER_CONFLICT = 9,
  S5RO_CALLBACK_ENABLE_FAILED = 10,
  S5RO_ACTIVITY_ENABLE_FAILED = 11,
  S5RO_CONTEXT_BIND_FAILED = 12,
  S5RO_GRAPH_BIND_FAILED = 13,
  S5RO_LIMIT_EXCEEDED = 14,
  S5RO_BUSY = 15,
  S5RO_FLUSH_FAILED = 16,
  S5RO_UNSUBSCRIBE_FAILED = 17,
  S5RO_GRAPH_DUPLICATE = 18,
  S5RO_CONTEXT_CONFLICT = 19,
  S5RO_TIMESTAMP_FAILED = 20,
  S5RO_TOKEN_MISMATCH = 21,
  S5RO_INTERNAL_ERROR = 22,
  S5RO_NOT_BOUND = 23,
  S5RO_WINDOW_ACTIVE = 24,
  S5RO_CLEANUP_FAILED = 25,
};

enum State : uint32_t {
  kCreated = 0,
  kStarted = 1,
  kWindow = 2,
  kEnded = 3,
  kStopped = 4,
  kQuarantined = 5,
};

enum RecKind : uint32_t {
  kRecKernel = 1,
  kRecMemcpy = 2,
  kRecMemset = 3,
};

enum EventKind : uint32_t {
  kEvGraphCreated = 1,
  kEvGraphCloned = 2,
  kEvGraphDestroy = 3,
  kEvNodeCreateStarting = 4,
  kEvNodeCreated = 5,
  kEvNodeCloned = 6,
  kEvNodeDestroy = 7,
  kEvNodeDependencyCreated = 8,
  kEvNodeDependencyDestroy = 9,
  kEvExecCreateStarting = 10,
  kEvExecCreated = 11,
  kEvExecDestroy = 12,
  kEvNodeUpdated = 13,
  kEvNodeSetParams = 14,
  kEvContextCreated = 15,
  kEvContextDestroy = 16,
  kEvStreamAttribute = 17,
};

// Reason bits are an allowlisted vocabulary shared with replay_observer.py.
enum Reason : uint32_t {
  kRDropSeen,
  kRDropQueryError,
  kRBufferRefused,
  kRRingOverflow,
  kRParseError,
  kRUnknownActivity,
  kRUnknownResource,
  kRFlushFailed,
  kRContextUnbound,
  kRContextMissing,
  kRContextDuplicate,
  kRContextContradictory,
  kRContextNotGreen,
  kRContextSmMismatch,
  kRContextDestroyed,
  kRWrongContext,
  kRUnknownGraph,
  kRAmbiguousGraph,
  kRUnknownNode,
  kRAmbiguousNode,
  kRKindMismatch,
  kRMissingNodeId,
  kRBoundaryOverlap,
  kRUnknownTimestamp,
  kRIncompleteRecord,
  kRInvalidInterval,
  kRUnknownResourceTime,
  kRUpdateSeen,
  kRDestroySeen,
  kRInventoryChanged,
  kRUnsupportedNode,
  kRUnknownNodeType,
  kRMappingUnknown,
  kRNoExercisedGraph,
  kRUnattributedExpected,
  kRLinkOverflow,
  kRNodeOverlap,
  kRMissingExecution,
  kRNoGraphs,
  kRGraphIdUnknown,
  kRNodeIdUnknown,
  kRGraphChanged,
  kREmptyGraph,
  kRDuplicateGraph,
  kRBufferOutstanding,
  kRReasonCount,
};

const char *const kReasonNames[kRReasonCount] = {
    "drop_seen", "drop_query_error", "buffer_refused", "ring_overflow",
    "parse_error", "unknown_activity_record", "unknown_resource_event",
    "flush_failed", "expected_context_unbound", "expected_context_missing",
    "expected_context_duplicate", "expected_context_contradictory",
    "expected_context_not_green", "expected_context_sm_mismatch",
    "context_destroyed", "known_wrong_context", "unknown_graph",
    "ambiguous_graph", "unknown_node", "ambiguous_node", "kind_mismatch",
    "missing_node_id", "boundary_overlap", "unknown_timestamp",
    "incomplete_record", "invalid_interval", "unknown_resource_timestamp",
    "update_seen", "destroy_seen", "inventory_changed", "unsupported_node",
    "unknown_node_type", "mapping_unknown", "no_exercised_graph",
    "unattributed_expected_context", "link_overflow", "node_overlap",
    "missing_execution", "no_graphs", "graph_id_unknown", "node_id_unknown", "graph_changed",
    "empty_graph", "duplicate_graph", "buffer_outstanding",
};

static_assert(kRReasonCount <= 64, "reason mask must fit uint64_t");

inline void mark(uint64_t &mask, Reason reason) {
  mask |= (1ull << static_cast<uint32_t>(reason));
}

// Argument validation lives in one place so every entry point fails closed.

struct Limits {
  uint32_t buffer_count = 0;
  uint64_t buffer_bytes = 0;
  uint32_t max_context_records = 0;
  uint32_t max_kernel_records = 0;
  uint32_t max_copy_records = 0;
  uint32_t max_graph_events = 0;
  uint32_t max_graphs = 0;
  uint32_t max_nodes_per_graph = 0;
  uint32_t max_nodes_total = 0;
  uint32_t max_links = 0;
};

struct ContextRec {
  uint32_t context_id = 0;
  uint32_t device_id = 0;
  uint32_t parent_context_id = 0;
  uint16_t num_multiprocessors = 0;
  uint8_t is_green = 0;
};

constexpr unsigned kLaunchFields=36;
struct ExecRec {
  uint32_t correlation_id=0;
  uint64_t launch[kLaunchFields]={0};
  char name[1024]={0};
  uint8_t launch_complete=0;
  uint32_t stream_id=0;
  uint32_t kind = 0;
  uint32_t context_id = 0;
  uint32_t device_id = 0;
  uint32_t graph_id = 0;
  uint64_t graph_node_id = 0;
  uint64_t start = 0;
  uint64_t end = 0;
  uint64_t completed = 0;
};

struct CopyRec {
  uint32_t correlation_id=0;
  uint32_t stream_id = 0;
  uint32_t copy_kind = 0; // CUPTI direction for memcpy; zero for memset
  uint32_t kind = 0;  // kRecMemcpy or kRecMemset
  uint32_t context_id = 0;
  uint32_t device_id = 0;
  uint32_t graph_id = 0;
  uint64_t graph_node_id = 0;
  uint64_t start = 0;
  uint64_t end = 0;
};

struct EventRec {
  uint32_t kind = 0;
  uint32_t graph_id = 0;
  uint32_t original_graph_id = 0;
  uint32_t exec_id = 0;
  uint64_t node_id = 0;
  uint64_t original_node_id = 0;
  int32_t node_type = -1;
  uint64_t ts = 0;
  uint8_t has_graph = 0;
  uint8_t has_original_graph = 0;
  uint8_t has_exec = 0;
  uint8_t has_node = 0;
  uint8_t has_original_node = 0;
  uint8_t has_node_type = 0;
};

struct NodeInfo {
  CUgraphNode retained_handle=nullptr; // private; never serialized
  uint64_t node_id = 0;
  uint32_t local_index = 0;
  int32_t node_type = -1;
  uint8_t observed = 0;  // bit 0 kernel, bit 1 memcpy, bit 2 memset
  uint64_t executions = 0;  // complete records for this exact static node
};

struct GraphSlot {
  uint64_t graph_handle = 0;
  uint32_t graph_id = 0;
  uint32_t first_node = 0;
  uint32_t node_count = 0;
  uint32_t type_counts[11] = {0};  // kernel..unknown
  uint32_t required_kernel = 0;
  uint32_t required_copy = 0;
  uint32_t required_memset = 0;
  uint32_t unsupported = 0;
  uint32_t unknown_type = 0;
  uint64_t static_reasons = 0;
  uint64_t window_reasons = 0;
  uint32_t missing_kernel = 0;
  uint32_t missing_copy = 0;
  uint32_t missing_memset = 0;
  uint32_t missing_kernel_indices[kMaxMissingIndices] = {0};
  uint32_t missing_copy_indices[kMaxMissingIndices] = {0};
  uint32_t missing_memset_indices[kMaxMissingIndices] = {0};
  uint32_t context_mismatch_indices[kMaxMissingIndices] = {0};
  uint32_t context_mismatch_total = 0;
  uint32_t observed_kernel = 0;
  uint32_t observed_copy = 0;
  uint32_t observed_memset = 0;
  uint64_t kernel_executions = 0;
  uint64_t copy_executions = 0;
  uint64_t memset_executions = 0;
  uint8_t graph_id_known = 0;
  uint8_t node_ids_known = 0;
  uint8_t inventory_changed = 0;
  uint8_t update_seen = 0;
  uint8_t destroy_seen = 0;
  uint8_t mapping_ambiguous = 0;
  uint8_t exercised = 0;
  uint8_t used = 0;
  uint8_t invalidated = 0;
  uint8_t retired = 0;
  uint8_t recycled = 0;
  uint64_t bind_event_count = 0;
  uint64_t retire_event_count = 0;
};

struct GraphMap {
  uint32_t id = 0;
  uint32_t slot = 0;
};

struct NodeKey {
  uint64_t id = 0;
  uint32_t slot = 0;
  uint32_t index = 0;
};

struct GraphAlias {
  uint64_t id = 0;
  uint32_t slot = 0;
  uint32_t index = 0;
};

constexpr CUpti_CallbackId kResourceIds[] = {
    CUPTI_CBID_RESOURCE_CONTEXT_CREATED,
    CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING,
    CUPTI_CBID_RESOURCE_GRAPH_CREATED,
    CUPTI_CBID_RESOURCE_GRAPH_CLONED,
    CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING,
    CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING,
    CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED,
    CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED,
    CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING,
    CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED,
    CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING,
    CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING,
    CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED,
    CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING,
    CUPTI_CBID_RESOURCE_GRAPH_NODE_UPDATED,
    CUPTI_CBID_RESOURCE_GRAPH_NODE_SET_PARAMS,
    CUPTI_CBID_RESOURCE_STREAM_ATTRIBUTE_CHANGED,
};
constexpr uint32_t kResourceIdCount =
    static_cast<uint32_t>(sizeof(kResourceIds) / sizeof(kResourceIds[0]));

constexpr uint32_t kCorrCapacity=131072;
struct ApiRec {uint32_t domain=0,correlation=0,thread=0,result=0;uint64_t start=0,end=0;char name[128]={0};};
struct ExternalRec {uint32_t correlation=0,kind=0;uint64_t external=0;};
struct ScopeRec {uint64_t id=0,parent=0;uint32_t thread=0,action=0,system_thread=0;};
struct ScopeEntry {uint64_t session,id;};
thread_local std::vector<ScopeEntry> tl_scopes;

struct Session {
  uint64_t token = 0;
  uint32_t state = kCreated;
  Limits limits;

  // API operations are serialised here; callbacks never take this lock.
  std::mutex op_mu;
  // Protects the capture rings, buffer pool and expected-context fields.  It
  // is never held across a CUPTI call that can invoke or wait on callbacks.
  std::recursive_mutex cb_mu;
  std::atomic<uint8_t> window_active{0};
  std::atomic<uint8_t> accept_activity{0};
  std::atomic<uint8_t> destroyed{0};

  uint8_t **buffers = nullptr;
  uint8_t *buffer_in_use = nullptr;
  uint32_t *free_stack = nullptr;
  uint32_t free_count = 0;
  uint32_t in_use_count = 0;

  ContextRec *contexts = nullptr;
  uint32_t context_count = 0;
  ExecRec *kernels = nullptr;
  uint32_t kernel_count = 0;
  CopyRec *copies = nullptr;
  uint32_t copy_count = 0;
  EventRec *events = nullptr;
  uint32_t event_count = 0;

  GraphSlot *slots = nullptr;
  uint32_t slot_count = 0;
  NodeInfo *nodes = nullptr;
  uint32_t node_used = 0;
  CUgraphNode *node_handles = nullptr;

  GraphMap *graph_map = nullptr;
  NodeKey *node_keys = nullptr;
  GraphAlias *node_aliases = nullptr;

  uint64_t reserved_bytes = 0;

  uint32_t runtime_api = 0;
  CUpti_SubscriberHandle subscriber = nullptr;
  uint8_t subscribed = 0;
  uint8_t activities_registered = 0;
  uint8_t activity_enabled[7] = {};
  bool correlation_enabled=false;
  CUpti_ActivityThreadIdType correlation_thread_type=CUPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT;
  ApiRec *apis=nullptr;ExternalRec *external=nullptr;ScopeRec *scopes=nullptr;
  uint32_t api_count=0,external_count=0,scope_count=0,open_scopes=0;
  std::atomic<uint64_t> correlation_overflow{0},correlation_errors{0};
  uint8_t callback_enabled[kResourceIdCount] = {0};
  uint8_t cleanup_failed = 0;
  uint64_t context_bind_event_count = 0;

  uint64_t expected_green = 0;
  uint32_t expected_context_id = 0;
  uint32_t expected_sms = 0;
  uint8_t expected_known = 0;

  uint32_t diag_graphs=0,diag_keys=0,diag_aliases=0;
  uint32_t window_route_generation = 0;
  uint64_t window_token = 0;
  uint64_t sample_start=0, sample_end=0;
  std::atomic<uint8_t> launch_sampling{0};
  uint64_t window_start = 0;
  uint64_t window_end = 0;
  uint8_t end_flushing = 0;
  uint64_t window_issued_base = 0;
  uint64_t window_completed_base = 0;
  uint64_t window_buffers_issued = 0;
  uint64_t window_buffers_completed = 0;

  // Counters.  Saturating increments are unnecessary at these bounds, but the
  // drop total is kept separate from CUPTI's reset-on-read queue counter.
  uint64_t buffers_requested = 0;
  uint64_t buffers_issued = 0;
  uint64_t buffers_completed = 0;
  uint64_t buffers_refused = 0;
  uint64_t dropped_total = 0;
  uint64_t drop_query_errors = 0;
  uint64_t parse_errors = 0;
  uint64_t invalid_records = 0;
  uint64_t callback_errors = 0;
  uint64_t unknown_activity_records = 0;
  uint64_t unknown_resource_events = 0;
  uint64_t resource_time_unknown = 0;
  uint64_t id_query_errors = 0;
  uint64_t records_before_window = 0;
  uint64_t records_after_window = 0;
  uint64_t records_in_window = 0;
  uint64_t ring_context_overflow = 0;
  uint64_t ring_kernel_overflow = 0;
  uint64_t ring_copy_overflow = 0;
  uint64_t ring_event_overflow = 0;
  uint64_t ring_alias_overflow = 0;
  uint64_t duplicate_buffer_completions = 0;
  uint64_t callback_reentrant_refused = 0;
  uint64_t flush_failures = 0;
  uint64_t unsubscribe_failures = 0;

  // Window analysis counters.
  uint64_t outside_scope_records = 0;
  uint64_t outside_scope_graph_records = 0;
  uint64_t unattributed_expected = 0;
  uint64_t unattributed_other = 0;
  uint64_t unknown_graph_records = 0;
  uint64_t ambiguous_graph_records = 0;
  uint64_t unknown_node_records = 0;
  uint64_t ambiguous_node_records = 0;
  uint64_t kind_mismatch_records = 0;
  uint64_t missing_node_id_records = 0;
  uint64_t boundary_overlap_records = 0;
  uint64_t unknown_timestamp_records = 0;
  uint64_t incomplete_records = 0;
  uint64_t invalid_interval_records = 0;
  uint64_t known_wrong_context_records = 0;
  uint64_t resource_events_seen = 0;
  uint64_t dependency_events = 0;
  uint64_t outside_scope_updates = 0;
  uint64_t alias_unknown = 0;
  // Recomputed from retained callback facts for each alias census. Diagnostic
  // only: type authority is cuGraphNodeGetType on the retained original plus
  // the independently typed activity record, never the clone callback hint.
  uint64_t alias_type_disagreements = 0;
  uint64_t alias_type_memcpy_disagreements = 0;
  uint64_t alias_type_memset_disagreements = 0;
  uint64_t alias_type_reported_kernel_disagreements = 0;
  uint64_t graph_passes = 0;
  uint8_t graph_pass_limit_hit = 0;
  uint8_t ambiguous_graph_ids = 0;
  uint8_t node_overlap = 0;
  uint8_t expected_context_destroyed = 0;
  uint8_t flush_failed = 0;
  uint8_t buffer_outstanding = 0;
};

std::mutex g_mu;
std::condition_variable g_cv;
std::shared_ptr<Session> g_session;
uint32_t g_callback_active = 0;
bool g_accept_callbacks = false;
bool g_retiring = false;
uint64_t g_next_token = 1;
uint64_t g_next_window_token = 1;

// A completion callback reentered from inside another completion callback is
// refused: CUPTI buffer ownership is single-threaded here and nested parsing
// would alias the same session rings.
thread_local bool tl_in_complete = false;

bool valid_limit(const Limits &l) {
  if (l.buffer_count < 1 || l.buffer_count > kMaxBuffers) return false;
  if (l.buffer_bytes < kMinBufferBytes || l.buffer_bytes > kMaxBufferBytes) return false;
  if (l.max_context_records < 1 || l.max_context_records > kMaxContextRecords) return false;
  if (l.max_kernel_records < 1 || l.max_kernel_records > kMaxExecRecords) return false;
  if (l.max_copy_records < 1 || l.max_copy_records > kMaxExecRecords) return false;
  if (l.max_graph_events < 1 || l.max_graph_events > kMaxEventRecords) return false;
  if (l.max_graphs < 1 || l.max_graphs > kMaxGraphs) return false;
  if (l.max_nodes_per_graph < 1 || l.max_nodes_per_graph > kMaxNodesPerGraph) return false;
  if (l.max_nodes_total < l.max_nodes_per_graph || l.max_nodes_total > kMaxNodesTotal) return false;
  if (l.max_links < 1 || l.max_links > kMaxLinks) return false;
  return true;
}

template <typename T>
T *alloc_array(size_t count) {
  if (count == 0) return nullptr;
  return new (std::nothrow) T[count]();
}

void free_session_storage(Session *s) {
  if (s == nullptr) return;
  for (uint32_t i = 0; i < s->limits.buffer_count; ++i) {
    if (s->buffers != nullptr && s->buffers[i] != nullptr) std::free(s->buffers[i]);
  }
  delete[] s->buffers;
  delete[] s->buffer_in_use;
  delete[] s->free_stack;
  delete[] s->contexts;
  delete[] s->kernels;
  delete[] s->copies;
  delete[] s->events;
  delete[] s->apis;delete[] s->external;delete[] s->scopes;
  delete[] s->slots;
  delete[] s->nodes;
  delete[] s->node_handles;
  delete[] s->graph_map;
  delete[] s->node_keys;
  delete[] s->node_aliases;
  s->buffers = nullptr;
  s->buffer_in_use = nullptr;
  s->free_stack = nullptr;
  s->contexts = nullptr;
  s->kernels = nullptr;
  s->copies = nullptr;
  s->events = nullptr;
  s->apis=nullptr;s->external=nullptr;s->scopes=nullptr;
  s->slots = nullptr;
  s->nodes = nullptr;
  s->node_handles = nullptr;
  s->graph_map = nullptr;
  s->node_keys = nullptr;
  s->node_aliases = nullptr;
}

struct SessionDeleter {
  void operator()(Session *s) const {
    if (s == nullptr) return;
    {
      std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
      free_session_storage(s);
    }
    // Unlock while the mutex still exists. The last shared owner invokes this
    // deleter only after callback/API owners have released their references.
    delete s;
  }
};

using SessionPtr = std::shared_ptr<Session>;

// Callback admission is deliberately brief: copying the session reference and
// incrementing the active count under g_mu means a concurrent destroy can wait
// for callbacks without holding a lock the callback itself needs.
struct CallbackRef {
  SessionPtr session;

  CallbackRef() {
    std::lock_guard<std::mutex> lock(g_mu);
    if (!g_accept_callbacks || g_retiring || !g_session) return;
    session = g_session;
    ++g_callback_active;
  }

  ~CallbackRef() {
    if (!session) return;
    {
      std::lock_guard<std::mutex> lock(g_mu);
      if (g_callback_active > 0) --g_callback_active;
      if (g_callback_active == 0) g_cv.notify_all();
    }
    session.reset();
  }
};

struct ReentryGuard {
  bool &flag;
  explicit ReentryGuard(bool &value) : flag(value) { flag = true; }
  ~ReentryGuard() { flag = false; }
};

SessionPtr acquire_session(uint64_t token) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_session || g_session->token != token) return SessionPtr();
  return g_session;
}

// --- bounded JSON writer ----------------------------------------------------

struct Json {
  char *out;
  uint64_t cap;
  uint64_t len = 0;
  bool bad = false;
  bool need_value = false;
  bool first[32];
  int depth = 0;

  Json(char *buffer, uint64_t capacity) : out(buffer), cap(capacity) {
    first[0] = true;
  }

  void put(char c) {
    if (len < cap) out[len] = c;
    ++len;
  }

  void text(const char *s) {
    if (s == nullptr) return;
    while (*s != '\0') {
      put(*s);
      ++s;
    }
  }

  void comma() {
    if (need_value) {
      need_value = false;
      return;
    }
    if (depth > 0) {
      if (!first[depth]) put(',');
      first[depth] = false;
    }
  }

  void open_obj() {
    comma();
    put('{');
    if (depth + 1 < 32) {
      ++depth;
      first[depth] = true;
    } else {
      bad = true;
    }
  }

  void close_obj() {
    if (depth > 0) {
      put('}');
      --depth;
    } else {
      bad = true;
    }
  }

  void open_arr() {
    comma();
    put('[');
    if (depth + 1 < 32) {
      ++depth;
      first[depth] = true;
    } else {
      bad = true;
    }
  }

  void close_arr() {
    if (depth > 0) {
      put(']');
      --depth;
    } else {
      bad = true;
    }
  }

  void key(const char *name) {
    comma();
    put('"');
    text(name);
    put('"');
    put(':');
    need_value = true;
  }

  void raw_u64(uint64_t value) {
    char tmp[24];
    int i = 24;
    do {
      tmp[--i] = static_cast<char>('0' + (value % 10));
      value /= 10;
    } while (value != 0);
    while (i < 24) put(tmp[i++]);
  }

  void u64(uint64_t value) {
    comma();
    raw_u64(value);
  }

  void i64(int64_t value) {
    comma();
    if (value < 0) {
      put('-');
      raw_u64(static_cast<uint64_t>(-(value + 1)) + 1ull);
    } else {
      raw_u64(static_cast<uint64_t>(value));
    }
  }

  void boolean(bool value) {
    comma();
    text(value ? "true" : "false");
  }

  void null() {
    comma();
    text("null");
  }

  void string(const char *value) {
    comma();
    put('"');
    text(value);
    put('"');
  }

  bool truncated() const { return len > cap; }
};

void emit_reasons(Json &j, const char *name, uint64_t mask,
                  const char *const *names, uint32_t count) {
  j.key(name);
  j.open_arr();
  for (uint32_t i = 0; i < count; ++i) {
    if ((mask & (1ull << i)) != 0) j.string(names[i]);
  }
  j.close_arr();
}

void emit_limits(Json &j, const Limits &l) {
  j.key("limits");
  j.open_obj();
  j.key("buffer_count");
  j.u64(l.buffer_count);
  j.key("buffer_bytes");
  j.u64(l.buffer_bytes);
  j.key("max_context_records");
  j.u64(l.max_context_records);
  j.key("max_kernel_records");
  j.u64(l.max_kernel_records);
  j.key("max_copy_records");
  j.u64(l.max_copy_records);
  j.key("max_graph_events");
  j.u64(l.max_graph_events);
  j.key("max_graphs");
  j.u64(l.max_graphs);
  j.key("max_nodes_per_graph");
  j.u64(l.max_nodes_per_graph);
  j.key("max_nodes_total");
  j.u64(l.max_nodes_total);
  j.key("max_links");
  j.u64(l.max_links);
  j.close_obj();
}

void emit_type_counts(Json &j, const GraphSlot &slot) {
  static const char *const names[11] = {
      "kernel", "memcpy", "memset", "graph", "empty", "event",
      "host", "conditional", "memory", "other", "unknown"};
  j.open_obj();
  for (uint32_t i = 0; i < 11; ++i) {
    j.key(names[i]);
    j.u64(slot.type_counts[i]);
  }
  j.close_obj();
}

void emit_required(Json &j, const GraphSlot &slot) {
  j.open_obj();
  j.key("kernel");
  j.u64(slot.required_kernel);
  j.key("memcpy");
  j.u64(slot.required_copy);
  j.key("memset");
  j.u64(slot.required_memset);
  j.close_obj();
}

// Emitted by the loops below so the slot number stays caller-owned.
void emit_static_slot_body(Json &j, uint32_t slot_index, const GraphSlot &slot) {
  j.key("slot");
  j.u64(slot_index);
  j.key("node_count");
  j.u64(slot.node_count);
  j.key("node_types");
  emit_type_counts(j, slot);
  j.key("required");
  emit_required(j, slot);
  j.key("unsupported");
  j.u64(slot.unsupported);
  j.key("unknown_type");
  j.u64(slot.unknown_type);
  j.key("graph_id_known");
  j.boolean(slot.graph_id_known != 0);
  j.key("node_ids_known");
  j.boolean(slot.node_ids_known != 0);
  j.key("inventory_changed");
  j.boolean(slot.inventory_changed != 0);
  j.key("static_complete");
  j.boolean(slot.static_reasons == 0 && slot.node_count > 0
             && slot.graph_id_known != 0 && slot.node_ids_known != 0);
  emit_reasons(j, "reasons", slot.static_reasons, kReasonNames, kRReasonCount);
}

// --- activity record storage ------------------------------------------------

void add_context(Session *s, const CUpti_ActivityContext3 *record) {
  if (s->context_count >= s->limits.max_context_records) {
    ++s->ring_context_overflow;
    return;
  }
  ContextRec &out = s->contexts[s->context_count++];
  out.context_id = record->contextId;
  out.device_id = record->deviceId;
  out.parent_context_id = record->parentContextId;
  out.num_multiprocessors = record->numMultiprocessors;
  out.is_green = record->isGreenContext;
}

void add_exec(Session *s, const CUpti_ActivityKernel10 *record) {
  if (s->kernel_count >= s->limits.max_kernel_records) {
    ++s->ring_kernel_overflow;
    return;
  }
  ExecRec &out = s->kernels[s->kernel_count++];
  out.kind = kRecKernel;
  out.correlation_id=record->correlationId;
  out.stream_id = record->streamId;
  out.context_id = record->contextId;
  out.device_id = record->deviceId;
  out.graph_id = record->graphId;
  out.graph_node_id = record->graphNodeId;
  out.start = record->start;
  out.end = record->end;
  out.completed = record->completed;
  std::memset(out.launch,0,sizeof(out.launch));out.name[0]=0;out.launch_complete=0;
  unsigned li=0;
  out.launch[li++]=static_cast<uint64_t>(record->cacheConfig.both);
  out.launch[li++]=static_cast<uint64_t>(record->sharedMemoryConfig);
  out.launch[li++]=static_cast<uint64_t>(record->registersPerThread);
  out.launch[li++]=static_cast<uint64_t>(record->partitionedGlobalCacheRequested);
  out.launch[li++]=static_cast<uint64_t>(record->partitionedGlobalCacheExecuted);
  out.launch[li++]=static_cast<uint64_t>(record->gridX);
  out.launch[li++]=static_cast<uint64_t>(record->gridY);
  out.launch[li++]=static_cast<uint64_t>(record->gridZ);
  out.launch[li++]=static_cast<uint64_t>(record->blockX);
  out.launch[li++]=static_cast<uint64_t>(record->blockY);
  out.launch[li++]=static_cast<uint64_t>(record->blockZ);
  out.launch[li++]=static_cast<uint64_t>(record->staticSharedMemory);
  out.launch[li++]=static_cast<uint64_t>(record->dynamicSharedMemory);
  out.launch[li++]=static_cast<uint64_t>(record->localMemoryPerThread);
  out.launch[li++]=static_cast<uint64_t>(record->localMemoryTotal);
  out.launch[li++]=static_cast<uint64_t>(record->launchType);
  out.launch[li++]=static_cast<uint64_t>(record->isSharedMemoryCarveoutRequested);
  out.launch[li++]=static_cast<uint64_t>(record->sharedMemoryCarveoutRequested);
  out.launch[li++]=static_cast<uint64_t>(record->sharedMemoryExecuted);
  out.launch[li++]=static_cast<uint64_t>(record->shmemLimitConfig);
  out.launch[li++]=static_cast<uint64_t>(record->channelID);
  out.launch[li++]=static_cast<uint64_t>(record->channelType);
  out.launch[li++]=static_cast<uint64_t>(record->clusterX);
  out.launch[li++]=static_cast<uint64_t>(record->clusterY);
  out.launch[li++]=static_cast<uint64_t>(record->clusterZ);
  out.launch[li++]=static_cast<uint64_t>(record->clusterSchedulingPolicy);
  out.launch[li++]=static_cast<uint64_t>(record->localMemoryTotal_v2);
  out.launch[li++]=static_cast<uint64_t>(record->maxPotentialClusterSize);
  out.launch[li++]=static_cast<uint64_t>(record->maxActiveClusters);
  out.launch[li++]=static_cast<uint64_t>(record->isDeviceLaunched);
  out.launch[li++]=record->pAccessPolicyWindow?1:0;
  if(record->pAccessPolicyWindow){
    auto p=record->pAccessPolicyWindow;uint32_t ratio=0;std::memcpy(&ratio,&p->hitRatio,4);
    out.launch[li++]=reinterpret_cast<uintptr_t>(p->base_ptr);out.launch[li++]=p->num_bytes;
    out.launch[li++]=ratio;out.launch[li++]=p->hitProp;out.launch[li++]=p->missProp;
  }
  if(record->name){size_t n=strnlen(record->name,sizeof(out.name));
    if(n<sizeof(out.name)){std::memcpy(out.name,record->name,n+1);out.launch_complete=1;}}

}

void add_copy(Session *s, RecKind kind, uint32_t context_id, uint32_t device_id,
              uint32_t graph_id, uint64_t graph_node_id, uint64_t start,
              uint64_t end, uint32_t stream_id = 0, uint32_t copy_kind = 0, uint32_t correlation_id = 0) {
  if (s->copy_count >= s->limits.max_copy_records) {
    ++s->ring_copy_overflow;
    return;
  }
  CopyRec &out = s->copies[s->copy_count++];
  out.correlation_id=correlation_id;
  out.stream_id = stream_id;
  out.copy_kind = copy_kind;
  out.kind = kind;
  out.context_id = context_id;
  out.device_id = device_id;
  out.graph_id = graph_id;
  out.graph_node_id = graph_node_id;
  out.start = start;
  out.end = end;
}

void parse_buffer(Session *s, uint8_t *buffer, size_t valid_size) {
  CUpti_Activity *record = nullptr;
  for (;;) {
    CUptiResult rc = cuptiActivityGetNextRecord(buffer, valid_size, &record);
    if (rc == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
    if (rc == CUPTI_ERROR_INVALID_KIND) {
      ++s->invalid_records;
      ++s->parse_errors;
      break;
    }
    if (rc != CUPTI_SUCCESS) {
      ++s->parse_errors;
      break;
    }
    if (record == nullptr) {
      ++s->parse_errors;
      break;
    }
    const bool in_window = s->window_active != 0;
    switch (record->kind) {
      case CUPTI_ACTIVITY_KIND_CONTEXT:
        add_context(s, reinterpret_cast<const CUpti_ActivityContext3 *>(record));
        break;
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        if (in_window || s->launch_sampling.load()) {
          add_exec(s, reinterpret_cast<const CUpti_ActivityKernel10 *>(record));
          if(in_window) ++s->records_in_window;
        } else {
          ++s->records_after_window;
        }
        break;
      case CUPTI_ACTIVITY_KIND_MEMCPY: {
        const auto *r = reinterpret_cast<const CUpti_ActivityMemcpy6 *>(record);
        if (in_window) {
          add_copy(s, kRecMemcpy, r->contextId, r->deviceId, r->graphId,
                   r->graphNodeId, r->start, r->end, r->streamId, r->copyKind, r->correlationId);
          ++s->records_in_window;
        } else {
          ++s->records_after_window;
        }
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMSET: {
        const auto *r = reinterpret_cast<const CUpti_ActivityMemset4 *>(record);
        if (in_window) {
          add_copy(s, kRecMemset, r->contextId, r->deviceId, r->graphId,
                   r->graphNodeId, r->start, r->end, r->streamId, 0, r->correlationId);
          ++s->records_in_window;
        } else {
          ++s->records_after_window;
        }
        break;
      }
      case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
        if(!s->correlation_enabled){++s->unknown_activity_records;break;}
        if(!in_window){++s->records_after_window;break;}
        const auto*r=reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(record);
        if(s->external_count>=kCorrCapacity){++s->correlation_overflow;break;}
        auto&o=s->external[s->external_count++];o.correlation=r->correlationId;o.kind=r->externalKind;o.external=r->externalId;
        break;
      }
      case CUPTI_ACTIVITY_KIND_DRIVER:
      case CUPTI_ACTIVITY_KIND_RUNTIME: {
        if(!s->correlation_enabled){++s->unknown_activity_records;break;}
        if(!in_window){++s->records_after_window;break;}
        const auto*r=reinterpret_cast<const CUpti_ActivityAPI*>(record);
        if(s->api_count>=kCorrCapacity){++s->correlation_overflow;break;}
        auto&o=s->apis[s->api_count++];o.domain=record->kind;o.correlation=r->correlationId;
        o.thread=r->threadId;o.result=r->returnValue;o.start=r->start;o.end=r->end;
        const char*name=nullptr;
        auto domain=record->kind==CUPTI_ACTIVITY_KIND_DRIVER?CUPTI_CB_DOMAIN_DRIVER_API:CUPTI_CB_DOMAIN_RUNTIME_API;
        if(cuptiGetCallbackName(domain,r->cbid,&name)!=CUPTI_SUCCESS||!name||strnlen(name,sizeof(o.name))>=sizeof(o.name))++s->correlation_errors;
        else std::memcpy(o.name,name,std::strlen(name)+1);
        break;
      }
      default:
        ++s->unknown_activity_records;
        break;
    }
  }
}

// --- callbacks --------------------------------------------------------------

void CUPTIAPI buffer_requested(uint8_t **buffer, size_t *size,
                               size_t *max_num_records) {
  if (buffer != nullptr) *buffer = nullptr;
  if (size != nullptr) *size = 0;
  if (max_num_records != nullptr) *max_num_records = 0;

  CallbackRef ref;
  Session *s = ref.session.get();
  if (s == nullptr || buffer == nullptr || size == nullptr) return;
  std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
  if (s->accept_activity.load() == 0) return;
  ++s->buffers_requested;
  if (s->free_count == 0) {
    ++s->buffers_refused;
    return;
  }
  const uint32_t index = s->free_stack[--s->free_count];
  if (s->buffer_in_use[index] != 0 || s->buffers[index] == nullptr) {
    ++s->callback_errors;
    return;
  }
  s->buffer_in_use[index] = 1;
  ++s->in_use_count;
  ++s->buffers_issued;
  *buffer = s->buffers[index];
  *size = static_cast<size_t>(s->limits.buffer_bytes);
}

void CUPTIAPI buffer_completed(CUcontext context, uint32_t stream_id,
                               uint8_t *buffer, size_t size, size_t valid_size) {
  (void)context;
  (void)stream_id;
  (void)size;
  if (tl_in_complete) {
    CallbackRef ref;
    Session *s = ref.session.get();
    if (s != nullptr) {
      std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
      ++s->callback_reentrant_refused;
    }
    return;
  }
  ReentryGuard reentry(tl_in_complete);
  CallbackRef ref;
  Session *s = ref.session.get();
  if (s == nullptr) return;
  std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
  if (buffer == nullptr) {
    ++s->callback_errors;
    return;
  }
  uint32_t index = s->limits.buffer_count;
  for (uint32_t i = 0; i < s->limits.buffer_count; ++i) {
    if (s->buffers != nullptr && s->buffers[i] == buffer) {
      index = i;
      break;
    }
  }
  if (index >= s->limits.buffer_count || s->buffer_in_use[index] == 0) {
    ++s->callback_errors;
    ++s->duplicate_buffer_completions;
    return;
  }
  ++s->buffers_completed;
  // Context provenance is collected from startup onward, not just while a
  // replay window is active. parse_buffer gates execution records separately.
  if (valid_size > 0 && s->accept_activity.load() != 0) {
    parse_buffer(s, buffer, valid_size);
  }
  s->buffer_in_use[index] = 0;
  if (s->in_use_count > 0) --s->in_use_count;
  if (s->free_count < s->limits.buffer_count) {
    s->free_stack[s->free_count++] = index;
  } else {
    ++s->callback_errors;
  }
}

void add_event(Session *s, uint32_t kind, const CUpti_GraphData *data) {
  if (s->event_count >= s->limits.max_graph_events) {
    ++s->ring_event_overflow;
    return;
  }
  EventRec &event = s->events[s->event_count++];
  event.kind = kind;
  uint64_t ts = 0;
  if (cuptiGetTimestamp(&ts) != CUPTI_SUCCESS) {
    ts = 0;
    ++s->resource_time_unknown;
  }
  event.ts = ts;
  if (data == nullptr) return;
  event.node_type = static_cast<int32_t>(data->nodeType);
  event.has_node_type = data->node != nullptr ? 1 : 0;
  if (data->graph != nullptr) {
    uint32_t id = 0;
    if (cuptiGetGraphId(data->graph, &id) == CUPTI_SUCCESS) {
      event.graph_id = id;
      event.has_graph = 1;
    } else {
      ++s->id_query_errors;
    }
  }
  if (data->originalGraph != nullptr) {
    uint32_t id = 0;
    if (cuptiGetGraphId(data->originalGraph, &id) == CUPTI_SUCCESS) {
      event.original_graph_id = id;
      event.has_original_graph = 1;
    } else {
      ++s->id_query_errors;
    }
  }
  if (data->node != nullptr) {
    uint64_t id = 0;
    if (cuptiGetGraphNodeId(data->node, &id) == CUPTI_SUCCESS) {
      event.node_id = id;
      event.has_node = 1;
    } else {
      ++s->id_query_errors;
    }
  }
  if (data->originalNode != nullptr) {
    uint64_t id = 0;
    if (cuptiGetGraphNodeId(data->originalNode, &id) == CUPTI_SUCCESS) {
      event.original_node_id = id;
      event.has_original_node = 1;
    } else {
      ++s->id_query_errors;
    }
  }
  if (data->graphExec != nullptr) {
    uint32_t id = 0;
    if (cuptiGetGraphExecId(data->graphExec, &id) == CUPTI_SUCCESS) {
      event.exec_id = id;
      event.has_exec = 1;
    } else {
      ++s->id_query_errors;
    }
  }
  (void)data->dependency;
}

void CUPTIAPI resource_callback(void *userdata, CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid, const void *cbdata) {
  CallbackRef ref;
  Session *s = ref.session.get();
  if (s == nullptr) return;
  if (userdata != nullptr && userdata != s) return;
  std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
  if (s->accept_activity.load() == 0) return;
  if (domain != CUPTI_CB_DOMAIN_RESOURCE) {
    ++s->unknown_resource_events;
    return;
  }
  ++s->resource_events_seen;

  if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_CREATED ||
      cbid == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING) {
    if (cbdata == nullptr) {
      ++s->unknown_resource_events;
      return;
    }
    const auto *resource = static_cast<const CUpti_ResourceData *>(cbdata);
    if (cbid == CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING &&
        resource->context != nullptr) {
      uint32_t id = 0;
      if (cuptiGetContextId(resource->context, &id) == CUPTI_SUCCESS) {
        if (s->expected_known != 0 && id == s->expected_context_id) {
          s->expected_context_destroyed = 1;
        }
      } else {
        ++s->id_query_errors;
        ++s->unknown_resource_events;
      }
    }
    return;
  }

  if (cbid == CUPTI_CBID_RESOURCE_STREAM_ATTRIBUTE_CHANGED) {
    if (s->event_count >= s->limits.max_graph_events) {
      ++s->ring_event_overflow;
    } else {
      EventRec &event = s->events[s->event_count++];
      event.kind = kEvStreamAttribute;
      if (cuptiGetTimestamp(&event.ts) != CUPTI_SUCCESS) {
        event.ts = 0;
        ++s->resource_time_unknown;
      }
    }
    return;
  }

  uint32_t kind = 0;
  switch (cbid) {
    case CUPTI_CBID_RESOURCE_GRAPH_CREATED: kind = kEvGraphCreated; break;
    case CUPTI_CBID_RESOURCE_GRAPH_CLONED: kind = kEvGraphCloned; break;
    case CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING: kind = kEvGraphDestroy; break;
    case CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING: kind = kEvNodeCreateStarting; break;
    case CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED: kind = kEvNodeCreated; break;
    case CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED: kind = kEvNodeCloned; break;
    case CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING: kind = kEvNodeDestroy; break;
    case CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED:
      kind = kEvNodeDependencyCreated;
      break;
    case CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING:
      kind = kEvNodeDependencyDestroy;
      break;
    case CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING: kind = kEvExecCreateStarting; break;
    case CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED: kind = kEvExecCreated; break;
    case CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING: kind = kEvExecDestroy; break;
    case CUPTI_CBID_RESOURCE_GRAPH_NODE_UPDATED: kind = kEvNodeUpdated; break;
    case CUPTI_CBID_RESOURCE_GRAPH_NODE_SET_PARAMS: kind = kEvNodeSetParams; break;
    default:
      ++s->unknown_resource_events;
      return;
  }
  if (kind == kEvNodeDependencyCreated || kind == kEvNodeDependencyDestroy) {
    ++s->dependency_events;
  }
  if (cbdata == nullptr) {
    ++s->unknown_resource_events;
    return;
  }
  // Resource callbacks carry CUpti_ResourceData; the graph-specific payload
  // lives behind resourceDescriptor (NVIDIA cuda_graphs_trace sample). Casting
  // the envelope itself to GraphData treats this descriptor pointer as a node.
  const auto *resource = static_cast<const CUpti_ResourceData *>(cbdata);
  if (resource->resourceDescriptor == nullptr) {
    ++s->unknown_resource_events;
    return;
  }
  add_event(s, kind, static_cast<const CUpti_GraphData *>(
                         resource->resourceDescriptor));
}

void sample_dropped(Session *s) {
  size_t dropped = 0;
  CUptiResult rc = cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped);
  if (rc != CUPTI_SUCCESS) {
    std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
    ++s->drop_query_errors;
    return;
  }
  std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
  s->dropped_total += static_cast<uint64_t>(dropped);
}

// --- static binding ---------------------------------------------------------

int classify_node_type(int32_t type) {
  switch (type) {
    case CU_GRAPH_NODE_TYPE_KERNEL: return 0;
    case CU_GRAPH_NODE_TYPE_MEMCPY: return 1;
    case CU_GRAPH_NODE_TYPE_MEMSET: return 2;
    case CU_GRAPH_NODE_TYPE_GRAPH: return 3;
    case CU_GRAPH_NODE_TYPE_EMPTY:
    case CU_GRAPH_NODE_TYPE_WAIT_EVENT:
    case CU_GRAPH_NODE_TYPE_EVENT_RECORD:
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL:
    case CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT: return 4;
    case CU_GRAPH_NODE_TYPE_HOST: return 5;
    case CU_GRAPH_NODE_TYPE_CONDITIONAL: return 6;
    case CU_GRAPH_NODE_TYPE_MEM_ALLOC:
    case CU_GRAPH_NODE_TYPE_MEM_FREE:
    case CU_GRAPH_NODE_TYPE_BATCH_MEM_OP: return 7;
    case -1: return 10;
    default: return 9;
  }
}

void finish_static_slot(GraphSlot &slot) {
  if (slot.graph_id_known == 0) mark(slot.static_reasons, kRGraphIdUnknown);
  if (slot.node_ids_known == 0) mark(slot.static_reasons, kRNodeIdUnknown);
  if (slot.unsupported > 0) mark(slot.static_reasons, kRUnsupportedNode);
  if (slot.unknown_type > 0) mark(slot.static_reasons, kRUnknownNodeType);
  if (slot.inventory_changed != 0) mark(slot.static_reasons, kRGraphChanged);
  if (slot.node_count == 0) mark(slot.static_reasons, kREmptyGraph);
  if (slot.mapping_ambiguous != 0) mark(slot.static_reasons, kRNodeOverlap);
}

bool compare_node_key(const NodeKey &a, const NodeKey &b);

int bind_graph_impl(Session *s, uint64_t graph_handle, uint32_t slot_index,
                    uint32_t *node_count_out) {
  if (node_count_out != nullptr) *node_count_out = 0;
  if (s->state != kStarted) return S5RO_STATE;
  if (s->window_active != 0) return S5RO_WINDOW_ACTIVE;
  if (slot_index >= s->limits.max_graphs) return S5RO_LIMIT_EXCEEDED;
  if (graph_handle == 0) return S5RO_BAD_ARGS;
  if (s->slots[slot_index].used != 0) return S5RO_STATE;
  for (uint32_t i = 0; i < s->slot_count; ++i) {
    if (s->slots[i].used != 0 && s->slots[i].graph_handle == graph_handle) {
      return S5RO_GRAPH_DUPLICATE;
    }
  }

  uint64_t bind_event_count = 0;
  {
    std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
    bind_event_count = s->event_count;
  }

  CUgraph graph = reinterpret_cast<CUgraph>(static_cast<uintptr_t>(graph_handle));
  size_t count = 0;
  if (cuGraphGetNodes(graph, nullptr, &count) != CUDA_SUCCESS) {
    return S5RO_GRAPH_BIND_FAILED;
  }
  if (count > s->limits.max_nodes_per_graph
      || s->node_used + count > s->limits.max_nodes_total) {
    return S5RO_LIMIT_EXCEEDED;
  }
  size_t enumerated = count;
  if (count > 0 &&
      cuGraphGetNodes(graph, s->node_handles, &enumerated) != CUDA_SUCCESS) {
    return S5RO_GRAPH_BIND_FAILED;
  }
  if (enumerated != count || enumerated > s->limits.max_nodes_per_graph) {
    return S5RO_GRAPH_BIND_FAILED;
  }

  GraphSlot &slot = s->slots[slot_index];
  slot.used = 1;
  slot.graph_handle = graph_handle;
  slot.first_node = s->node_used;
  slot.node_count = static_cast<uint32_t>(enumerated);
  slot.graph_id_known = 0;
  slot.node_ids_known = 1;
  slot.bind_event_count = bind_event_count;
  slot.invalidated = 0;
  slot.retired = 0;
  slot.recycled = 0;
  slot.retire_event_count = 0;
  uint32_t graph_id = 0;
  if (cuptiGetGraphId(graph, &graph_id) == CUPTI_SUCCESS) {
    slot.graph_id = graph_id;
    slot.graph_id_known = 1;
  } else {
    ++s->id_query_errors;
  }

  for (uint32_t i = 0; i < slot.node_count; ++i) {
    NodeInfo &node = s->nodes[s->node_used + i];
    node.local_index = i;
    node.retained_handle=s->node_handles[i];
    node.node_type = -1;
    CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
    if (cuGraphNodeGetType(s->node_handles[i], &type) == CUDA_SUCCESS) {
      node.node_type = static_cast<int32_t>(type);
    } else {
      slot.unknown_type++;
      slot.node_ids_known = 0;
      mark(slot.static_reasons, kRUnknownNodeType);
    }
    const int bucket = classify_node_type(node.node_type);
    if (bucket >= 0 && bucket < 11) slot.type_counts[bucket]++;
    if (bucket == 0) slot.required_kernel++;
    else if (bucket == 1) slot.required_copy++;
    else if (bucket == 2) slot.required_memset++;
    else if (bucket >= 3) slot.unsupported++;
    if (bucket == 10) slot.unknown_type++;
    if (cuptiGetGraphNodeId(s->node_handles[i], &node.node_id) != CUPTI_SUCCESS) {
      node.node_id = 0;
      slot.node_ids_known = 0;
    }
    if (node.node_id == 0) slot.node_ids_known = 0;
    node.observed = 0;
  }
  // Duplicate original node IDs inside one retained graph are ambiguous.  Use
  // the preallocated node-key scratch, sorted here; the full cross-slot index
  // is rebuilt at window end.
  uint32_t dup_count = 0;
  for (uint32_t i = 0; i < slot.node_count; ++i) {
    const NodeInfo &node = s->nodes[slot.first_node + i];
    if (node.node_id == 0) continue;
    s->node_keys[dup_count].id = node.node_id;
    s->node_keys[dup_count].slot = slot_index;
    s->node_keys[dup_count].index = i;
    ++dup_count;
  }
  std::sort(s->node_keys, s->node_keys + dup_count, compare_node_key);
  for (uint32_t i = 1; i < dup_count; ++i) {
    if (s->node_keys[i].id == s->node_keys[i - 1].id) {
      slot.mapping_ambiguous = 1;
      s->node_overlap = 1;
      mark(slot.static_reasons, kRNodeOverlap);
      break;
    }
  }
  // A second bound graph with the same graph ID is an ambiguous join.
  for (uint32_t i = 0; i < slot_index; ++i) {
    if (s->slots[i].used != 0 && s->slots[i].graph_id_known != 0 &&
        slot.graph_id_known != 0 && s->slots[i].graph_id == slot.graph_id) {
      slot.mapping_ambiguous = 1;
      s->ambiguous_graph_ids = 1;
      mark(slot.static_reasons, kRDuplicateGraph);
    }
  }

  finish_static_slot(slot);
  s->node_used += slot.node_count;
  if (s->slot_count <= slot_index) s->slot_count = slot_index + 1;
  if (node_count_out != nullptr) *node_count_out = slot.node_count;
  return S5RO_OK;
}

// --- graph / node mapping at window end ------------------------------------

bool compare_graph_map(const GraphMap &a, const GraphMap &b) {
  if (a.id != b.id) return a.id < b.id;
  return a.slot < b.slot;
}

bool compare_node_key(const NodeKey &a, const NodeKey &b) {
  if (a.id != b.id) return a.id < b.id;
  if (a.slot != b.slot) return a.slot < b.slot;
  return a.index < b.index;
}

bool compare_alias(const GraphAlias &a, const GraphAlias &b) {
  if (a.id != b.id) return a.id < b.id;
  if (a.slot != b.slot) return a.slot < b.slot;
  return a.index < b.index;
}

struct Lookup {
  uint32_t slot = kNoSlot;
  bool ambiguous = false;
};

Lookup graph_lookup(Session *s, uint32_t id, uint32_t count) {
  Lookup out;
  if (id == 0 || count == 0) return out;
  uint32_t lo = 0, hi = count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (s->graph_map[mid].id < id) lo = mid + 1;
    else hi = mid;
  }
  bool found = false;
  for (uint32_t i = lo; i < count && s->graph_map[i].id == id; ++i) {
    if (!found) {
      out.slot = s->graph_map[i].slot;
      found = true;
    } else if (out.slot != s->graph_map[i].slot) {
      out.ambiguous = true;
    }
  }
  if (out.ambiguous) out.slot = kNoSlot;
  return out;
}

void compact_graph_map(Session *s, uint32_t *count) {
  if (*count < 2) return;
  uint32_t out = 1;
  for (uint32_t i = 1; i < *count; ++i) {
    const GraphMap &previous = s->graph_map[out - 1];
    const GraphMap &current = s->graph_map[i];
    if (current.id == previous.id && current.slot == previous.slot) continue;
    s->graph_map[out++] = current;
  }
  *count = out;
}

bool build_graph_map(Session *s, uint32_t *count) {
  *count = 0;
  for (uint32_t i = 0; i < s->slot_count; ++i) {
    if (s->slots[i].used != 0 && s->slots[i].graph_id_known != 0) {
      if (*count >= s->limits.max_graph_events + s->limits.max_graphs) {
        ++s->ring_alias_overflow;
        return false;
      }
      s->graph_map[*count].id = s->slots[i].graph_id;
      s->graph_map[*count].slot = i;
      ++(*count);
    }
  }
  std::sort(s->graph_map, s->graph_map + *count, compare_graph_map);
  compact_graph_map(s, count);
  for (uint32_t pass = 0; pass < kMaxGraphPasses; ++pass) {
    ++s->graph_passes;
    const uint32_t search_count = *count;
    bool added = false;
    for (uint32_t e = 0; e < s->event_count; ++e) {
      const EventRec &event = s->events[e];
      uint32_t id = 0;
      uint32_t from_id = 0;
      if ((event.kind == kEvGraphCreated || event.kind == kEvGraphCloned) &&
          event.has_graph != 0 && event.has_original_graph != 0) {
        id = event.graph_id;
        from_id = event.original_graph_id;
      } else if ((event.kind == kEvExecCreateStarting ||
                  event.kind == kEvExecCreated) &&
                 event.has_exec != 0 && event.has_graph != 0) {
        id = event.exec_id;
        from_id = event.graph_id;
      } else {
        continue;
      }
      Lookup from = graph_lookup(s, from_id, search_count);
      if (from.slot == kNoSlot || from.ambiguous) continue;
      Lookup existing = graph_lookup(s, id, search_count);
      if (existing.ambiguous ||
          (existing.slot != kNoSlot && existing.slot != from.slot)) {
        s->ambiguous_graph_ids = 1;
        continue;
      }
      if (existing.slot == from.slot) continue;
      if (*count >= s->limits.max_graph_events + s->limits.max_graphs) {
        ++s->ring_alias_overflow;
        return false;
      }
      s->graph_map[*count].id = id;
      s->graph_map[*count].slot = from.slot;
      ++(*count);
      added = true;
    }
    std::sort(s->graph_map, s->graph_map + *count, compare_graph_map);
    compact_graph_map(s, count);
    if (!added) break;
    if (pass + 1 == kMaxGraphPasses) s->graph_pass_limit_hit = 1;
  }
  // Duplicate IDs mapping to different slots are ambiguous at lookup time.
  for (uint32_t i = 1; i < *count; ++i) {
    if (s->graph_map[i].id == s->graph_map[i - 1].id &&
        s->graph_map[i].slot != s->graph_map[i - 1].slot) {
      s->ambiguous_graph_ids = 1;
    }
  }
  return s->ring_alias_overflow == 0;
}

void build_node_keys(Session *s, uint32_t *count) {
  *count = 0;
  for (uint32_t slot = 0; slot < s->slot_count; ++slot) {
    if (s->slots[slot].used == 0) continue;
    for (uint32_t i = 0; i < s->slots[slot].node_count; ++i) {
      const NodeInfo &node = s->nodes[s->slots[slot].first_node + i];
      if (node.node_id == 0) continue;
      if (*count >= s->limits.max_nodes_total) {
        ++s->ring_alias_overflow;
        break;
      }
      s->node_keys[*count].id = node.node_id;
      s->node_keys[*count].slot = slot;
      s->node_keys[*count].index = i;
      ++(*count);
    }
  }
  std::sort(s->node_keys, s->node_keys + *count, compare_node_key);
}

bool find_original_node(Session *s, uint32_t key_count, uint64_t id,
                        uint32_t slot, uint32_t *index) {
  if (id == 0) return false;
  uint32_t lo = 0, hi = key_count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    const NodeKey &key = s->node_keys[mid];
    if (key.id < id || (key.id == id && key.slot < slot)) lo = mid + 1;
    else hi = mid;
  }
  bool found = false;
  uint32_t found_index = 0;
  for (uint32_t i = lo; i < key_count; ++i) {
    const NodeKey &key = s->node_keys[i];
    if (key.id != id || key.slot != slot) break;
    if (!found) {
      found = true;
      found_index = key.index;
    } else if (found_index != key.index) {
      return false;
    }
  }
  if (found && index != nullptr) *index = found_index;
  return found;
}

void build_node_aliases(Session *s, uint32_t key_count, uint32_t graph_count,
                        uint32_t *alias_count) {
  *alias_count = 0;
  s->alias_type_disagreements = 0;
  s->alias_type_memcpy_disagreements = 0;
  s->alias_type_memset_disagreements = 0;
  s->alias_type_reported_kernel_disagreements = 0;
  for (uint32_t e = 0; e < s->event_count; ++e) {
    const EventRec &event = s->events[e];
    if (event.kind != kEvNodeCloned && event.kind != kEvNodeCreated) continue;
    if (event.has_node == 0 || event.has_original_node == 0) continue;
    if (event.node_id == event.original_node_id) continue;
    uint32_t slot = kNoSlot;
    bool ambiguous = false;
    if (event.has_original_graph != 0) {
      Lookup l = graph_lookup(s, event.original_graph_id, graph_count);
      slot = l.slot;
      ambiguous = l.ambiguous;
    } else if (event.has_graph != 0) {
      Lookup l = graph_lookup(s, event.graph_id, graph_count);
      slot = l.slot;
      ambiguous = l.ambiguous;
    } else if (s->slot_count == 1) {
      slot = 0;
    }
    if (slot == kNoSlot || ambiguous) {
      if (slot != kNoSlot && ambiguous) ++s->ambiguous_node_records;
      continue;
    }
    uint32_t index = 0;
    if (!find_original_node(s, key_count, event.original_node_id, slot, &index)) {
      ++s->alias_unknown;
      s->slots[slot].mapping_ambiguous = 1;
      continue;
    }
    if (event.has_node_type != 0 && event.node_type >= 0) {
      const NodeInfo &original =
          s->nodes[s->slots[slot].first_node + index];
      if (original.node_type != event.node_type) {
        // A callback type disagreement is not an identity overlap. Identity
        // comes from the graph clone link and originalNode ID above. Type
        // comes from the retained driver's inventory; analyze_window checks
        // EVERY kernel/copy/memset activity against that inventory before it
        // counts. Do not replace either proof with callback nodeType (or query
        // ephemeral clone handles / call the CUDA driver from this callback).
        // Keep the disagreement visible so the VM rerun can localize it.
        ++s->alias_type_disagreements;
        if (original.node_type == CU_GRAPH_NODE_TYPE_MEMCPY)
          ++s->alias_type_memcpy_disagreements;
        if (original.node_type == CU_GRAPH_NODE_TYPE_MEMSET)
          ++s->alias_type_memset_disagreements;
        if (event.node_type == CU_GRAPH_NODE_TYPE_KERNEL)
          ++s->alias_type_reported_kernel_disagreements;
      }
    }
    if (s->node_aliases == nullptr ||
        *alias_count >= s->limits.max_links) {
      ++s->ring_alias_overflow;
      continue;
    }
    s->node_aliases[*alias_count].id = event.node_id;
    s->node_aliases[*alias_count].slot = slot;
    s->node_aliases[*alias_count].index = index;
    ++(*alias_count);
  }
  std::sort(s->node_aliases, s->node_aliases + *alias_count, compare_alias);
}

struct NodeLookup {
  uint32_t slot = kNoSlot;
  uint32_t index = 0;
  bool found = false;
  bool ambiguous = false;
};

// --- report construction ----------------------------------------------------

struct ContextSummary {
  uint64_t records = 0;
  uint64_t green_true = 0;
  uint64_t green_false = 0;
  uint64_t sm_known = 0;
  uint64_t sm_matching = 0;
  uint64_t sm_mismatched = 0;
  bool is_green_known = false;
  bool is_green = false;
  bool sm_known_out = false;
  uint32_t num_multiprocessors = 0;
  bool parent_present = false;
  bool duplicate = false;
  bool contradictory = false;
};

NodeLookup resolve_activity_node(Session *s, uint32_t slot_index, uint64_t id,
                                 uint32_t key_count, uint32_t alias_count,
                                 uint32_t *out_index);

void analyze_events(Session *s, uint32_t graph_count, uint32_t key_count,
                    uint32_t alias_count) {
  for (uint32_t slot = 0; slot < s->slot_count; ++slot) {
    GraphSlot &g = s->slots[slot];
    if (g.used == 0) continue;
    g.window_reasons = 0;
    g.inventory_changed = 0;
    g.update_seen = 0;
    g.destroy_seen = 0;
    g.exercised = 0;
    g.invalidated = 0;
    g.retired = 0;
    g.recycled = 0;
    g.retire_event_count = 0;
    g.observed_kernel = g.observed_copy = g.observed_memset = 0;
    g.kernel_executions = g.copy_executions = g.memset_executions = 0;
    g.missing_kernel = g.missing_copy = g.missing_memset = 0;
    g.context_mismatch_total = 0;
    for (uint32_t i = 0; i < g.node_count; ++i) {
      s->nodes[g.first_node + i].observed = 0;
      s->nodes[g.first_node + i].executions = 0;
    }
  }

  auto invalidate = [s](GraphSlot &g, uint32_t event_index) {
    g.invalidated = 1;
    if (g.retired != 0 && event_index > g.retire_event_count) {
      g.recycled = 1;
      s->ambiguous_graph_ids = 1;
      s->node_overlap = 1;
    }
  };

  auto apply = [&](uint32_t slot_index, uint32_t event_index,
                   const EventRec &event) {
    GraphSlot &g = s->slots[slot_index];
    if (g.used == 0 || event_index < g.bind_event_count) return;
    switch (event.kind) {
      case kEvNodeUpdated:
      case kEvNodeSetParams:
        g.update_seen = 1;
        invalidate(g, event_index);
        break;
      case kEvGraphDestroy:
      case kEvExecDestroy:
      case kEvNodeDestroy:
        g.destroy_seen = 1;
        g.retired = 1;
        g.retire_event_count = event_index;
        invalidate(g, event_index);
        break;
      case kEvNodeCreateStarting:
      case kEvNodeCreated:
      case kEvGraphCreated:
      case kEvExecCreateStarting:
      case kEvExecCreated:
      case kEvNodeDependencyCreated:
      case kEvNodeDependencyDestroy:
        g.inventory_changed = 1;
        invalidate(g, event_index);
        break;
      case kEvGraphCloned:
      case kEvNodeCloned:
        // A clone creates a new identity; it is not a mutation of the bound
        // original.  If the original was already destroyed, the recycled ID
        // cannot be joined to this generation.
        invalidate(g, event_index);
        if (g.retired == 0) g.invalidated = 0;
        break;
      default:
        break;
    }
  };

  for (uint32_t e = 0; e < s->event_count; ++e) {
    const EventRec &event = s->events[e];
    if (event.kind == kEvStreamAttribute) continue;
    bool matched = false;

    auto by_graph = [&](uint32_t id, bool has_id) {
      if (!has_id || id == 0) return;
      Lookup l = graph_lookup(s, id, graph_count);
      if (l.ambiguous) {
        s->ambiguous_graph_ids = 1;
        matched = true;
        return;
      }
      if (l.slot != kNoSlot) {
        apply(l.slot, e, event);
        matched = true;
      }
    };
    by_graph(event.graph_id, event.has_graph != 0);
    by_graph(event.original_graph_id, event.has_original_graph != 0);
    by_graph(event.exec_id, event.has_exec != 0);

    auto by_node = [&](uint64_t id, bool has_id) {
      if (!has_id || id == 0) return;
      for (uint32_t slot = 0; slot < s->slot_count; ++slot) {
        if (s->slots[slot].used == 0) continue;
        uint32_t index = 0;
        NodeLookup lookup = resolve_activity_node(s, slot, id, key_count,
                                                  alias_count, &index);
        if (lookup.ambiguous) {
          s->slots[slot].mapping_ambiguous = 1;
          matched = true;
        } else if (lookup.found) {
          apply(slot, e, event);
          matched = true;
        }
      }
    };
    by_node(event.node_id, event.has_node != 0);
    by_node(event.original_node_id, event.has_original_node != 0);

    if (!matched && (event.kind == kEvNodeUpdated ||
                     event.kind == kEvNodeSetParams)) {
      ++s->outside_scope_updates;
    }
  }
}

NodeLookup resolve_activity_node(Session *s, uint32_t slot_index, uint64_t id,
                                 uint32_t key_count, uint32_t alias_count,
                                 uint32_t *out_index) {
  NodeLookup result;
  result.slot = slot_index;
  if (id == 0) return result;
  uint32_t lo = 0, hi = key_count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    const NodeKey &key = s->node_keys[mid];
    if (key.id < id || (key.id == id && key.slot < slot_index)) lo = mid + 1;
    else hi = mid;
  }
  bool original_found = false;
  uint32_t original_index = 0;
  uint32_t original_hits = 0;
  for (uint32_t i = lo; i < key_count; ++i) {
    const NodeKey &key = s->node_keys[i];
    if (key.id != id || key.slot != slot_index) break;
    if (original_hits == 0) original_index = key.index;
    ++original_hits;
  }
  if (original_hits == 1) {
    original_found = true;
  }
  uint32_t alo = 0, ahi = alias_count;
  while (alo < ahi) {
    uint32_t mid = alo + (ahi - alo) / 2;
    const GraphAlias &a = s->node_aliases[mid];
    if (a.id < id || (a.id == id && a.slot < slot_index)) alo = mid + 1;
    else ahi = mid;
  }
  uint32_t alias_hits = 0;
  uint32_t alias_index = 0;
  for (uint32_t i = alo; i < alias_count; ++i) {
    const GraphAlias &a = s->node_aliases[i];
    if (a.id != id || a.slot != slot_index) break;
    if (alias_hits == 0) alias_index = a.index;
    ++alias_hits;
  }
  if (original_hits > 1 || alias_hits > 1 ||
      (original_hits == 1 && alias_hits == 1)) {
    result.ambiguous = true;
    return result;
  }
  if (original_found) {
    result.found = true;
    if (out_index != nullptr) *out_index = original_index;
    return result;
  }
  if (alias_hits == 1) {
    result.found = true;
    if (out_index != nullptr) *out_index = alias_index;
  }
  return result;
}

void context_summary(Session *s, ContextSummary *out) {
  for (uint32_t i = 0; i < s->context_count; ++i) {
    const ContextRec &record = s->contexts[i];
    if (s->expected_known == 0 || record.context_id != s->expected_context_id) {
      continue;
    }
    if (out->records == 0) {
      out->is_green_known = true;
      out->is_green = record.is_green != 0;
      out->parent_present = record.parent_context_id != 0;
      if (record.is_green != 0) {
        out->sm_known_out = true;
        out->num_multiprocessors = record.num_multiprocessors;
      }
    } else {
      if ((record.is_green != 0) != out->is_green) out->contradictory = true;
      if (record.is_green != 0 && out->sm_known_out &&
          record.num_multiprocessors != out->num_multiprocessors) {
        out->contradictory = true;
      }
      if ((record.parent_context_id != 0) != out->parent_present) {
        out->contradictory = true;
      }
    }
    ++out->records;
    if (record.is_green != 0) {
      ++out->green_true;
      if (s->expected_sms != 0) {
        ++out->sm_known;
        if (record.num_multiprocessors == s->expected_sms) ++out->sm_matching;
        else ++out->sm_mismatched;
      }
    } else {
      ++out->green_false;
    }
  }
  if (out->records > 1 && !out->contradictory) out->duplicate = true;
}

void emit_context_summary(Json &j, const Session *s, const ContextSummary &c) {
  j.key("expected_context");
  j.open_obj();
  j.key("bound");
  j.boolean(s->expected_known != 0);
  j.key("context_known");
  j.boolean(s->expected_known != 0);
  j.key("expected_sms");
  j.u64(s->expected_sms);
  j.key("records");
  j.u64(c.records);
  j.key("green_true");
  j.u64(c.green_true);
  j.key("green_false");
  j.u64(c.green_false);
  j.key("sm_known");
  j.u64(c.sm_known);
  j.key("sm_matching");
  j.u64(c.sm_matching);
  j.key("sm_mismatched");
  j.u64(c.sm_mismatched);
  j.key("is_green");
  if (!c.is_green_known || c.contradictory) j.null();
  else j.boolean(c.is_green);
  j.key("num_multiprocessors");
  if (!c.sm_known_out || c.contradictory) j.null();
  else j.u64(c.num_multiprocessors);
  j.key("parent_context_present");
  j.boolean(c.parent_present);
  j.key("duplicate");
  j.boolean(c.duplicate);
  j.key("contradictory");
  j.boolean(c.contradictory);
  j.close_obj();
}

void emit_static_document(Session *s, Json &j) {
  j.open_obj();
  j.key("schema");
  j.string("s5ro-native-static/1");
  j.key("abi");
  j.u64(kAbi);
  j.key("compiled_api");
  j.u64(CUPTI_API_VERSION);
  j.key("runtime_api");
  j.u64(s->runtime_api);
  emit_limits(j, s->limits);
  j.key("reserved_bytes");
  j.u64(s->reserved_bytes);
  j.key("started");
  j.boolean(s->state != kCreated);
  j.key("expected_context");
  j.open_obj();
  j.key("bound");
  j.boolean(s->expected_known != 0);
  j.key("known");
  j.boolean(s->expected_known != 0);
  j.key("expected_sms");
  j.u64(s->expected_sms);
  j.close_obj();
  uint64_t root_reasons = 0;
  if (s->expected_known == 0) mark(root_reasons, kRContextUnbound);
  if (s->slot_count == 0) mark(root_reasons, kRNoGraphs);
  j.key("graph_slots");
  j.open_arr();
  for (uint32_t i = 0; i < s->slot_count; ++i) {
    if (s->slots[i].used == 0) continue;
    root_reasons |= s->slots[i].static_reasons;
    j.open_obj();
    emit_static_slot_body(j, i, s->slots[i]);
    j.close_obj();
  }
  j.close_arr();
  j.key("complete");
  j.boolean(root_reasons == 0 && s->slot_count > 0 && s->expected_known != 0);
  emit_reasons(j, "reasons", root_reasons, kReasonNames, kRReasonCount);
  j.close_obj();
}

void analyze_window(Session *s, uint32_t graph_count, uint32_t key_count,
                    uint32_t alias_count) {
  analyze_events(s, graph_count, key_count, alias_count);
  for (uint32_t i = 0; i < s->kernel_count; ++i) {
    const ExecRec &r = s->kernels[i];
    const bool context_match =
        s->expected_known != 0 && r.context_id == s->expected_context_id;
    if (r.start == 0 || r.end == 0) {
      ++s->unknown_timestamp_records;
      continue;
    }
    if (r.end < r.start) {
      ++s->invalid_interval_records;
      continue;
    }
    // A caller drain can deliver a warm-up record after the window opened.
    // Classify wholly outside records separately from boundary straddlers so
    // late history neither passes for window coverage nor becomes a boundary
    // failure.
    if (s->window_start != 0 && r.end < s->window_start) {
      ++s->records_before_window;
      continue;
    }
    if (s->window_start != 0 && r.start > s->window_end) {
      ++s->records_after_window;
      continue;
    }
    const bool boundary =
        s->window_start != 0 &&
        (r.start < s->window_start || r.end > s->window_end);
    Lookup graph = graph_lookup(s, r.graph_id, graph_count);
    if (graph.ambiguous) {
      ++s->ambiguous_graph_records;
      continue;
    }
    if (graph.slot == kNoSlot) {
      if (r.graph_id == 0) {
        if (context_match) {
          ++s->unattributed_expected;
        } else {
          ++s->unattributed_other;
          ++s->outside_scope_records;
        }
      } else if (context_match) {
        ++s->unknown_graph_records;
      } else {
        ++s->outside_scope_graph_records;
        ++s->outside_scope_records;
      }
      continue;
    }
    GraphSlot &slot = s->slots[graph.slot];
    slot.exercised = 1;
    if (slot.recycled != 0) {
      ++s->ambiguous_graph_records;
      continue;
    }
    uint32_t mismatch_index = 0;
    bool mismatch_found = false;
    if (r.graph_node_id != 0) {
      NodeLookup mismatch_node = resolve_activity_node(
          s, graph.slot, r.graph_node_id, key_count, alias_count,
          &mismatch_index);
      mismatch_found = mismatch_node.found;
    }
    if (!context_match) {
      ++s->known_wrong_context_records;
      slot.window_reasons |= (1ull << kRWrongContext);
      if (slot.context_mismatch_total < kMaxMissingIndices) {
        slot.context_mismatch_indices[slot.context_mismatch_total] =
            mismatch_found ? mismatch_index : 0;
      }
      ++slot.context_mismatch_total;
      continue;
    }
    if (slot.invalidated != 0) continue;
    if (boundary) {
      ++s->boundary_overlap_records;
      continue;
    }
    if (r.graph_node_id == 0) {
      ++s->missing_node_id_records;
      mark(slot.window_reasons, kRMissingNodeId);
      continue;
    }
    uint32_t node_index = 0;
    NodeLookup node = resolve_activity_node(s, graph.slot, r.graph_node_id,
                                            key_count, alias_count, &node_index);
    if (node.ambiguous) {
      ++s->ambiguous_node_records;
      slot.mapping_ambiguous = 1;
      mark(slot.window_reasons, kRAmbiguousNode);
      continue;
    }
    if (!node.found) {
      ++s->unknown_node_records;
      mark(slot.window_reasons, kRUnknownNode);
      continue;
    }
    const NodeInfo &info = s->nodes[slot.first_node + node_index];
    if (info.node_type != CU_GRAPH_NODE_TYPE_KERNEL) {
      ++s->kind_mismatch_records;
      mark(slot.window_reasons, kRKindMismatch);
      continue;
    }
    ++slot.kernel_executions;
    if (r.completed == 0) {
      ++s->incomplete_records;
      mark(slot.window_reasons, kRIncompleteRecord);
      continue;
    }
    if (r.completed < r.end) {
      ++s->invalid_interval_records;
      mark(slot.window_reasons, kRInvalidInterval);
      continue;
    }
    if (r.completed > s->window_end) {
      ++s->incomplete_records;
      mark(slot.window_reasons, kRIncompleteRecord);
      continue;
    }
    s->nodes[slot.first_node + node_index].observed |= 1;
    ++s->nodes[slot.first_node + node_index].executions;
  }

  for (uint32_t i = 0; i < s->copy_count; ++i) {
    const CopyRec &r = s->copies[i];
    const bool context_match =
        s->expected_known != 0 && r.context_id == s->expected_context_id;
    if (r.start == 0 || r.end == 0) {
      ++s->unknown_timestamp_records;
      continue;
    }
    if (r.end < r.start) {
      ++s->invalid_interval_records;
      continue;
    }
    if (s->window_start != 0 && r.end < s->window_start) {
      ++s->records_before_window;
      continue;
    }
    if (s->window_start != 0 && r.start > s->window_end) {
      ++s->records_after_window;
      continue;
    }
    const bool boundary =
        s->window_start != 0 &&
        (r.start < s->window_start || r.end > s->window_end);
    Lookup graph = graph_lookup(s, r.graph_id, graph_count);
    if (graph.ambiguous) {
      ++s->ambiguous_graph_records;
      continue;
    }
    if (graph.slot == kNoSlot) {
      if (r.graph_id == 0) {
        if (context_match) {
          ++s->unattributed_expected;
        } else {
          ++s->unattributed_other;
          ++s->outside_scope_records;
        }
      } else if (context_match) {
        ++s->unknown_graph_records;
      } else {
        ++s->outside_scope_graph_records;
        ++s->outside_scope_records;
      }
      continue;
    }
    GraphSlot &slot = s->slots[graph.slot];
    slot.exercised = 1;
    if (slot.recycled != 0) {
      ++s->ambiguous_graph_records;
      continue;
    }
    uint32_t mismatch_index = 0;
    bool mismatch_found = false;
    if (r.graph_node_id != 0) {
      NodeLookup mismatch_node = resolve_activity_node(
          s, graph.slot, r.graph_node_id, key_count, alias_count,
          &mismatch_index);
      mismatch_found = mismatch_node.found;
    }
    if (!context_match) {
      ++s->known_wrong_context_records;
      slot.window_reasons |= (1ull << kRWrongContext);
      if (slot.context_mismatch_total < kMaxMissingIndices) {
        slot.context_mismatch_indices[slot.context_mismatch_total] =
            mismatch_found ? mismatch_index : 0;
      }
      ++slot.context_mismatch_total;
      continue;
    }
    if (slot.invalidated != 0) continue;
    if (boundary) {
      ++s->boundary_overlap_records;
      continue;
    }
    if (r.graph_node_id == 0) {
      ++s->missing_node_id_records;
      mark(slot.window_reasons, kRMissingNodeId);
      continue;
    }
    uint32_t node_index = 0;
    NodeLookup node = resolve_activity_node(s, graph.slot, r.graph_node_id,
                                            key_count, alias_count, &node_index);
    if (node.ambiguous) {
      ++s->ambiguous_node_records;
      slot.mapping_ambiguous = 1;
      mark(slot.window_reasons, kRAmbiguousNode);
      continue;
    }
    if (!node.found) {
      ++s->unknown_node_records;
      mark(slot.window_reasons, kRUnknownNode);
      continue;
    }
    const NodeInfo &info = s->nodes[slot.first_node + node_index];
    const int32_t expected_type = r.kind == kRecMemcpy
                                      ? CU_GRAPH_NODE_TYPE_MEMCPY
                                      : CU_GRAPH_NODE_TYPE_MEMSET;
    if (info.node_type != expected_type) {
      ++s->kind_mismatch_records;
      mark(slot.window_reasons, kRKindMismatch);
      continue;
    }
    if (r.kind == kRecMemcpy) {
      ++slot.copy_executions;
      s->nodes[slot.first_node + node_index].observed |= 2;
    } else {
      ++slot.memset_executions;
      s->nodes[slot.first_node + node_index].observed |= 4;
    }
    ++s->nodes[slot.first_node + node_index].executions;
  }

  for (uint32_t slot = 0; slot < s->slot_count; ++slot) {
    GraphSlot &g = s->slots[slot];
    if (g.used == 0) continue;
    for (uint32_t i = 0; i < g.node_count; ++i) {
      const NodeInfo &node = s->nodes[g.first_node + i];
      if (node.node_type == CU_GRAPH_NODE_TYPE_KERNEL) {
        if ((node.observed & 1) != 0) {
          ++g.observed_kernel;
        } else {
          if (g.missing_kernel < kMaxMissingIndices) {
            g.missing_kernel_indices[g.missing_kernel] = i;
          }
          ++g.missing_kernel;
        }
      } else if (node.node_type == CU_GRAPH_NODE_TYPE_MEMCPY) {
        if ((node.observed & 2) != 0) {
          ++g.observed_copy;
        } else {
          if (g.missing_copy < kMaxMissingIndices) {
            g.missing_copy_indices[g.missing_copy] = i;
          }
          ++g.missing_copy;
        }
      } else if (node.node_type == CU_GRAPH_NODE_TYPE_MEMSET) {
        if ((node.observed & 4) != 0) {
          ++g.observed_memset;
        } else {
          if (g.missing_memset < kMaxMissingIndices) {
            g.missing_memset_indices[g.missing_memset] = i;
          }
          ++g.missing_memset;
        }
      }
    }
    if (g.static_reasons != 0 && g.exercised != 0) {
      g.window_reasons |= g.static_reasons;
    }
    if (g.inventory_changed != 0) mark(g.window_reasons, kRInventoryChanged);
    if (g.update_seen != 0) mark(g.window_reasons, kRUpdateSeen);
    if (g.destroy_seen != 0) mark(g.window_reasons, kRDestroySeen);
    if (g.mapping_ambiguous != 0) mark(g.window_reasons, kRAmbiguousNode);
    // An unexercised descriptor is inventory, not evidence of missing work in
    // this window.  Only descriptors actually exercised can be qualified or
    // judged incomplete.
    if (g.exercised != 0 && g.missing_kernel != 0) {
      mark(g.window_reasons, kRMissingExecution);
    }
    if (g.exercised != 0 && g.missing_copy != 0) {
      mark(g.window_reasons, kRMissingExecution);
    }
    if (g.exercised != 0 && g.missing_memset != 0) {
      mark(g.window_reasons, kRMissingExecution);
    }
    if (g.exercised != 0 &&
        (g.missing_kernel != 0 || g.missing_copy != 0 || g.missing_memset != 0)) {
      mark(g.window_reasons, kRIncompleteRecord);
    }
  }
}


// Diagnostic only: CUPTI IDs, never CUDA handles/arguments. Allocations occur
// after drain, never in callbacks; capped output cannot masquerade as complete.
void emit_context_diagnostics(Session *s,uint32_t graphs,uint32_t keys,
 uint32_t aliases,uint64_t start,uint64_t end,bool census,Json &j){
 constexpr size_t cap=4096;
 using Key=std::tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint64_t>;
 std::map<Key,uint64_t> rows;uint64_t eligible=0,omitted=0,boundary=0;
 for(uint32_t i=0;i<s->kernel_count;++i){const auto&r=s->kernels[i];
  if(!start||!end||!r.start||!r.end||r.end<r.start||r.end<start||r.start>end)continue;
  if(r.start<start||r.end>end){++boundary;continue;}++eligible;
  auto g=graph_lookup(s,r.graph_id,graphs);uint32_t slot=kNoSlot,index=kNoSlot;
  if(!g.ambiguous&&g.slot!=kNoSlot){slot=g.slot;uint32_t n=0;
   auto found=resolve_activity_node(s,slot,r.graph_node_id,keys,aliases,&n);
   if(found.found&&!found.ambiguous)index=n;}
  Key k(slot,index,r.context_id,r.stream_id,r.graph_id,r.graph_node_id);
  auto it=rows.find(k);if(it!=rows.end())++it->second;
  else if(rows.size()<cap)rows.emplace(k,1);else ++omitted;
 }
 j.open_obj();j.key("schema");j.string("s5ro-context-diagnostics/1");
 j.key("id_namespace");j.string("CUPTI");j.key("expected_context_id");j.u64(s->expected_context_id);
 j.key("start_ns");j.u64(start);j.key("end_ns");j.u64(end);
 j.key("eligible_kernel_records");j.u64(eligible);j.key("omitted_records");j.u64(omitted);
 j.key("boundary_records");j.u64(boundary);j.key("row_cap");j.u64(cap);
 j.key("contexts");j.open_arr();for(uint32_t i=0;i<s->context_count;++i){const auto&c=s->contexts[i];
  j.open_obj();j.key("context_id");j.u64(c.context_id);j.key("parent_context_id");j.u64(c.parent_context_id);
  j.key("is_green");j.boolean(c.is_green!=0);j.key("sms");j.u64(c.num_multiprocessors);j.close_obj();}
 j.close_arr();j.key("executions");j.open_arr();for(const auto&e:rows){const auto&k=e.first;j.open_obj();
  j.key("slot");j.u64(std::get<0>(k));j.key("node_index");j.u64(std::get<1>(k));
  j.key("context_id");j.u64(std::get<2>(k));j.key("stream_id");j.u64(std::get<3>(k));
  j.key("graph_id");j.u64(std::get<4>(k));j.key("graph_node_id");j.u64(std::get<5>(k));
  j.key("count");j.u64(e.second);j.close_obj();}
 j.close_arr();j.key("nodes");j.open_arr();size_t emitted=0,skipped=0;
 if(census)for(uint32_t slot=0;slot<s->slot_count;++slot){const auto&g=s->slots[slot];if(!g.used)continue;
  for(uint32_t n=0;n<g.node_count;++n){size_t ix=g.first_node+n;
   if(s->nodes[ix].node_type!=CU_GRAPH_NODE_TYPE_KERNEL)continue;
   if(emitted>=cap){++skipped;continue;}++emitted;
   CUDA_KERNEL_NODE_PARAMS p{};CUresult rc=cuGraphKernelNodeGetParams(s->nodes[ix].retained_handle,&p);
   const bool applicable=rc==CUDA_SUCCESS&&!p.func&&p.kern&&p.ctx;
   uint32_t id=0;int idrc=-1;if(applicable)idrc=static_cast<int>(cuptiGetContextId(p.ctx,&id));
   j.open_obj();j.key("slot");j.u64(slot);j.key("node_index");j.u64(n);
   j.key("graph_node_id");j.u64(s->nodes[ix].node_id);j.key("query_rc");j.u64(static_cast<unsigned>(rc));
   j.key("func_present");j.boolean(p.func!=nullptr);j.key("kern_present");j.boolean(p.kern!=nullptr);
   j.key("ctx_present");j.boolean(p.ctx!=nullptr);j.key("ctx_semantics");
   j.string(rc!=CUDA_SUCCESS?"unavailable":p.func?"ignored_func_set":!p.kern?"unavailable":
            !p.ctx?"null_current_context_semantics":"explicit_kernel_context");
   j.key("context_id_available");j.boolean(applicable&&idrc==CUPTI_SUCCESS);j.key("context_id");j.u64(id);j.close_obj();}}
 j.close_arr();j.key("nodes_omitted");j.u64(skipped);j.key("census_requested");j.boolean(census);
 j.key("complete");j.boolean(!omitted&&!boundary&&!skipped);j.close_obj();
}

void emit_window_document(Session *s, uint32_t graph_count, uint32_t key_count,
                          uint32_t alias_count, Json &j) {
  ContextSummary context;
  context_summary(s, &context);
  uint64_t root_reasons = 0;

  if (s->dropped_total != 0) mark(root_reasons, kRDropSeen);
  if (s->drop_query_errors != 0) mark(root_reasons, kRDropQueryError);
  if (s->buffers_refused != 0) mark(root_reasons, kRBufferRefused);
  if (s->ring_context_overflow != 0 || s->ring_kernel_overflow != 0 ||
      s->ring_copy_overflow != 0 || s->ring_event_overflow != 0 ||
      s->ring_alias_overflow != 0) {
    mark(root_reasons, kRRingOverflow);
  }
  if (s->parse_errors != 0) mark(root_reasons, kRParseError);
  if (s->unknown_activity_records != 0) mark(root_reasons, kRUnknownActivity);
  if (s->unknown_resource_events != 0) mark(root_reasons, kRUnknownResource);
  if (s->flush_failed != 0) mark(root_reasons, kRFlushFailed);
  if (s->resource_time_unknown != 0) mark(root_reasons, kRUnknownResourceTime);
  if (s->id_query_errors != 0) mark(root_reasons, kRMappingUnknown);
  if (s->callback_reentrant_refused != 0 ||
      s->duplicate_buffer_completions != 0) {
    mark(root_reasons, kRRingOverflow);
  }
  if (s->ambiguous_graph_ids != 0) mark(root_reasons, kRAmbiguousGraph);
  if (s->node_overlap != 0) mark(root_reasons, kRNodeOverlap);
  if (s->graph_pass_limit_hit != 0) mark(root_reasons, kRLinkOverflow);

  if (s->expected_known == 0) {
    mark(root_reasons, kRContextUnbound);
  } else {
    if (context.records == 0) mark(root_reasons, kRContextMissing);
    if (context.duplicate) mark(root_reasons, kRContextDuplicate);
    if (context.contradictory) mark(root_reasons, kRContextContradictory);
    if (context.records != 0 && context.green_true == 0) {
      mark(root_reasons, kRContextNotGreen);
    }
    if (context.green_true != 0 && context.sm_known != 0 &&
        context.sm_matching == 0 && context.sm_mismatched != 0) {
      mark(root_reasons, kRContextSmMismatch);
    }
    if (s->expected_context_destroyed != 0) {
      mark(root_reasons, kRContextDestroyed);
    }
  }

  analyze_window(s, graph_count, key_count, alias_count);
  // analyze_events()/analyze_window() can discover recycled identities and
  // ambiguous node mappings, so re-mark these mapping flags after analysis.
  if (s->node_overlap != 0) mark(root_reasons, kRNodeOverlap);
  if (s->ambiguous_graph_ids != 0) mark(root_reasons, kRAmbiguousGraph);
  if (s->in_use_count != 0 ||
      s->window_buffers_issued > s->window_buffers_completed) {
    s->buffer_outstanding = 1;
  }
  if (s->buffer_outstanding != 0) mark(root_reasons, kRBufferOutstanding);
  if (s->alias_unknown != 0) mark(root_reasons, kRMappingUnknown);
  if (s->unknown_graph_records != 0) mark(root_reasons, kRUnknownGraph);
  if (s->ambiguous_graph_records != 0) mark(root_reasons, kRAmbiguousGraph);
  if (s->unknown_node_records != 0) mark(root_reasons, kRUnknownNode);
  if (s->ambiguous_node_records != 0) mark(root_reasons, kRAmbiguousNode);
  if (s->kind_mismatch_records != 0) mark(root_reasons, kRKindMismatch);
  if (s->missing_node_id_records != 0) mark(root_reasons, kRMissingNodeId);
  if (s->boundary_overlap_records != 0) mark(root_reasons, kRBoundaryOverlap);
  if (s->unknown_timestamp_records != 0) mark(root_reasons, kRUnknownTimestamp);
  if (s->incomplete_records != 0) mark(root_reasons, kRIncompleteRecord);
  if (s->invalid_interval_records != 0) mark(root_reasons, kRInvalidInterval);
  if (s->unattributed_expected != 0) mark(root_reasons, kRUnattributedExpected);
  if (s->ring_alias_overflow != 0) mark(root_reasons, kRRingOverflow);
  bool any_exercised = false;
  bool any_update = false;
  bool any_destroy = false;
  bool any_inventory = false;
  for (uint32_t slot = 0; slot < s->slot_count; ++slot) {
    GraphSlot &g = s->slots[slot];
    if (g.used == 0) continue;
    if (g.exercised != 0) any_exercised = true;
    if (g.update_seen != 0) any_update = true;
    if (g.destroy_seen != 0) any_destroy = true;
    if (g.inventory_changed != 0) any_inventory = true;
    root_reasons |= g.window_reasons;
  }
  if (!any_exercised) mark(root_reasons, kRNoExercisedGraph);
  if (s->known_wrong_context_records != 0) mark(root_reasons, kRWrongContext);

  j.open_obj();
  j.key("schema");
  j.string("s5ro-native-window/4");
  j.key("context_diagnostics");
  try{emit_context_diagnostics(s,graph_count,key_count,alias_count,s->window_start,s->window_end,false,j);}
  catch(...){j.bad=true;}
  j.key("abi");
  j.u64(kAbi);
  j.key("compiled_api");
  j.u64(CUPTI_API_VERSION);
  j.key("runtime_api");
  j.u64(s->runtime_api);
  emit_limits(j, s->limits);
  j.key("reserved_bytes");
  j.u64(s->reserved_bytes);
  j.key("window");
  j.open_obj();
  j.key("start_ns");
  j.u64(s->window_start);
  j.key("end_ns");
  j.u64(s->window_end);
  j.key("route_generation");
  j.u64(s->window_route_generation);
  j.key("flushed");
  j.boolean(s->flush_failed == 0);
  j.key("flush_forced");
  j.boolean(false);
  j.close_obj();
  emit_context_summary(j, s, context);

  j.key("counts");
  j.open_obj();
  struct CountEntry { const char *name; uint64_t value; };
  const CountEntry counts[] = {
      {"context_records", s->context_count},
      {"kernel_records", s->kernel_count},
      {"copy_records", s->copy_count},
      {"resource_events", s->resource_events_seen},
      {"unknown_activity_records", s->unknown_activity_records},
      {"unknown_resource_events", s->unknown_resource_events},
      {"records_before_window", s->records_before_window},
      {"records_after_window", s->records_after_window},
      {"records_in_window", s->records_in_window},
      {"outside_scope_records", s->outside_scope_records},
      {"outside_scope_graph_records", s->outside_scope_graph_records},
      {"unattributed_expected_context", s->unattributed_expected},
      {"unattributed_other_context", s->unattributed_other},
      {"unknown_graph_records", s->unknown_graph_records},
      {"ambiguous_graph_records", s->ambiguous_graph_records},
      {"unknown_node_records", s->unknown_node_records},
      {"ambiguous_node_records", s->ambiguous_node_records},
      {"kind_mismatch_records", s->kind_mismatch_records},
      {"missing_node_id_records", s->missing_node_id_records},
      {"boundary_overlap_records", s->boundary_overlap_records},
      {"unknown_timestamp_records", s->unknown_timestamp_records},
      {"incomplete_records", s->incomplete_records},
      {"invalid_interval_records", s->invalid_interval_records},
      {"known_wrong_context_records", s->known_wrong_context_records},
      {"buffers_requested", s->buffers_requested},
      {"buffers_issued", s->buffers_issued},
      {"buffers_completed", s->buffers_completed},
      {"buffers_issued_in_window", s->window_buffers_issued},
      {"buffers_completed_in_window", s->window_buffers_completed},
      {"buffers_refused", s->buffers_refused},
      {"buffers_outstanding", s->in_use_count},
      {"dropped_records_total", s->dropped_total},
      {"drop_query_errors", s->drop_query_errors},
      {"parse_errors", s->parse_errors},
      {"invalid_records", s->invalid_records},
      {"callback_errors", s->callback_errors},
      {"callback_reentrant_refused", s->callback_reentrant_refused},
      {"duplicate_buffer_completions", s->duplicate_buffer_completions},
      {"dependency_events", s->dependency_events},
      {"outside_scope_updates", s->outside_scope_updates},
      {"graph_passes", s->graph_passes},
      {"alias_type_disagreements", s->alias_type_disagreements},
      {"alias_type_memcpy_disagreements", s->alias_type_memcpy_disagreements},
      {"alias_type_memset_disagreements", s->alias_type_memset_disagreements},
      {"alias_type_reported_kernel_disagreements", s->alias_type_reported_kernel_disagreements},
  };
  for (const CountEntry &entry : counts) {
    j.key(entry.name);
    j.u64(entry.value);
  }
  j.key("ring_overflows");
  j.open_obj();
  j.key("context");
  j.u64(s->ring_context_overflow);
  j.key("kernel");
  j.u64(s->ring_kernel_overflow);
  j.key("copy");
  j.u64(s->ring_copy_overflow);
  j.key("resource");
  j.u64(s->ring_event_overflow);
  j.key("aliases");
  j.u64(s->ring_alias_overflow);
  j.close_obj();
  j.close_obj();

  j.key("mapping");
  j.open_obj();
  j.key("graph_ids");
  j.u64(graph_count);
  j.key("node_keys");
  j.u64(key_count);
  j.key("node_aliases");
  j.u64(alias_count);
  j.key("alias_unknown");
  j.u64(s->alias_unknown);
  j.key("ambiguous_graph_ids");
  j.boolean(s->ambiguous_graph_ids != 0);
  j.key("node_overlap");
  j.boolean(s->node_overlap != 0);
  j.close_obj();

  j.key("graph_slots");
  j.open_arr();
  for (uint32_t slot = 0; slot < s->slot_count; ++slot) {
    const GraphSlot &g = s->slots[slot];
    if (g.used == 0) continue;
    const bool complete = g.static_reasons == 0 && g.window_reasons == 0 &&
                          g.exercised != 0 && g.inventory_changed == 0 &&
                          g.update_seen == 0 && g.destroy_seen == 0 &&
                          g.mapping_ambiguous == 0 && g.missing_kernel == 0 &&
                          g.missing_copy == 0 && g.missing_memset == 0;
    j.open_obj();
    j.key("slot");
    j.u64(slot);
    j.key("node_count");
    j.u64(g.node_count);
    j.key("node_types");
    emit_type_counts(j, g);
    j.key("required");
    emit_required(j, g);
    j.key("observed");
    j.open_obj();
    j.key("kernel");
    j.u64(g.observed_kernel);
    j.key("memcpy");
    j.u64(g.observed_copy);
    j.key("memset");
    j.u64(g.observed_memset);
    j.close_obj();
    j.key("missing");
    j.open_obj();
    j.key("kernel");
    j.u64(g.missing_kernel);
    j.key("memcpy");
    j.u64(g.missing_copy);
    j.key("memset");
    j.u64(g.missing_memset);
    j.close_obj();
    j.key("missing_kernel_indices");
    j.open_arr();
    for (uint32_t i = 0; i < g.missing_kernel && i < kMaxMissingIndices; ++i) {
      j.u64(g.missing_kernel_indices[i]);
    }
    j.close_arr();
    j.key("missing_copy_indices");
    j.open_arr();
    for (uint32_t i = 0; i < g.missing_copy && i < kMaxMissingIndices; ++i) {
      j.u64(g.missing_copy_indices[i]);
    }
    j.close_arr();
    j.key("missing_memset_indices");
    j.open_arr();
    for (uint32_t i = 0; i < g.missing_memset && i < kMaxMissingIndices; ++i) {
      j.u64(g.missing_memset_indices[i]);
    }
    j.close_arr();
    j.key("context_mismatch_nodes");
    j.open_arr();
    for (uint32_t i = 0; i < g.context_mismatch_total && i < kMaxMissingIndices; ++i) {
      j.u64(g.context_mismatch_indices[i]);
    }
    j.close_arr();
    j.key("kernel_executions");
    j.u64(g.kernel_executions);
    j.key("copy_executions");
    j.u64(g.copy_executions);
    j.key("memset_executions");
    j.u64(g.memset_executions);
    // Exact multiplicity bounds include zero-count nodes. min == max == the
    // independently counted replays proves every node has that many records;
    // a duplicate node cannot compensate for a missing occurrence elsewhere.
    j.key("node_execution_bounds");
    j.open_obj();
    const char *names[] = {"kernel", "memcpy", "memset"};
    const int32_t types[] = {CU_GRAPH_NODE_TYPE_KERNEL,
                            CU_GRAPH_NODE_TYPE_MEMCPY, CU_GRAPH_NODE_TYPE_MEMSET};
    for (uint32_t kind = 0; kind < 3; ++kind) {
      uint64_t low = 0, high = 0;
      bool found = false;
      for (uint32_t n = 0; n < g.node_count; ++n) {
        const NodeInfo &node = s->nodes[g.first_node + n];
        if (node.node_type != types[kind]) continue;
        if (!found || node.executions < low) low = node.executions;
        high = std::max(high, node.executions);
        found = true;
      }
      j.key(names[kind]); j.open_obj();
      j.key("min"); j.u64(low);
      j.key("max"); j.u64(high);
      j.close_obj();
    }
    j.close_obj();
    j.key("graph_id_known");
    j.boolean(g.graph_id_known != 0);
    j.key("node_ids_known");
    j.boolean(g.node_ids_known != 0);
    j.key("static_complete");
    j.boolean(g.static_reasons == 0 && g.node_count > 0 &&
              g.graph_id_known != 0 && g.node_ids_known != 0);
    j.key("unsupported");
    j.u64(g.unsupported);
    j.key("unknown_type");
    j.u64(g.unknown_type);
    j.key("inventory_changed");
    j.boolean(g.inventory_changed != 0);
    j.key("update_seen");
    j.boolean(g.update_seen != 0);
    j.key("destroy_seen");
    j.boolean(g.destroy_seen != 0);
    j.key("mapping_ambiguous");
    j.boolean(g.mapping_ambiguous != 0);
    j.key("exercised");
    j.boolean(g.exercised != 0);
    j.key("complete");
    j.boolean(complete);
    emit_reasons(j, "reasons", g.window_reasons, kReasonNames, kRReasonCount);
    j.close_obj();
  }
  j.close_arr();

  j.key("flags");
  j.open_obj();
  j.key("permanent_overflow");
  j.boolean(s->ring_context_overflow != 0 || s->ring_kernel_overflow != 0 ||
            s->ring_copy_overflow != 0 || s->ring_event_overflow != 0 ||
            s->ring_alias_overflow != 0);
  j.key("buffer_refused");
  j.boolean(s->buffers_refused != 0);
  j.key("buffer_outstanding");
  j.boolean(s->buffer_outstanding != 0);
  j.key("parse_error");
  j.boolean(s->parse_errors != 0);
  j.key("unknown_activity");
  j.boolean(s->unknown_activity_records != 0);
  j.key("unknown_resource");
  j.boolean(s->unknown_resource_events != 0);
  j.key("drop_seen");
  j.boolean(s->dropped_total != 0);
  j.key("drop_query_error");
  j.boolean(s->drop_query_errors != 0);
  j.key("flush_failed");
  j.boolean(s->flush_failed != 0);
  j.key("update_seen");
  j.boolean(any_update);
  j.key("destroy_seen");
  j.boolean(any_destroy || s->expected_context_destroyed != 0);
  j.key("inventory_changed");
  j.boolean(any_inventory);
  j.key("expected_context_missing");
  j.boolean(s->expected_known != 0 && context.records == 0);
  j.key("expected_context_duplicate");
  j.boolean(context.duplicate);
  j.key("expected_context_contradictory");
  j.boolean(context.contradictory);
  j.key("known_wrong_context");
  j.boolean(s->known_wrong_context_records != 0);
  j.key("known_wrong_greenness");
  j.boolean(context.records != 0 && context.green_true == 0);
  j.key("known_wrong_sm");
  j.boolean(context.green_true != 0 && context.sm_known != 0 &&
            context.sm_matching == 0 && context.sm_mismatched != 0);
  j.key("context_destroyed");
  j.boolean(s->expected_context_destroyed != 0);
  j.key("outside_scope_graph_records");
  j.boolean(s->outside_scope_graph_records != 0);
  j.key("no_exercised_graph");
  j.boolean(!any_exercised);
  j.key("link_overflow");
  j.boolean(s->graph_pass_limit_hit != 0 || s->ring_alias_overflow != 0);
  j.key("reentrant_refused");
  j.boolean(s->callback_reentrant_refused != 0);
  j.close_obj();

  j.key("complete");
  j.boolean(root_reasons == 0);
  emit_reasons(j, "reasons", root_reasons, kReasonNames, kRReasonCount);
  j.close_obj();
}

uint64_t compute_reserved(const Limits &l) {
  uint64_t total = sizeof(Session);
  total += sizeof(uint8_t *) * l.buffer_count;
  total += sizeof(uint8_t) * l.buffer_count;
  total += sizeof(uint32_t) * l.buffer_count;
  total += l.buffer_bytes * l.buffer_count;
  total += sizeof(ContextRec) * l.max_context_records;
  total += sizeof(ExecRec) * l.max_kernel_records;
  total += sizeof(CopyRec) * l.max_copy_records;
  total += sizeof(EventRec) * l.max_graph_events;
  total += sizeof(GraphSlot) * l.max_graphs;
  total += sizeof(NodeInfo) * l.max_nodes_total;
  total += sizeof(CUgraphNode) * l.max_nodes_per_graph;
  total += sizeof(GraphMap) * (static_cast<uint64_t>(l.max_graph_events) + l.max_graphs);
  total += sizeof(NodeKey) * l.max_nodes_total;
  total += sizeof(GraphAlias) * l.max_links;
  return total;
}

void reset_capture_state(Session *s) {
  std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
  s->context_count = 0;
  s->kernel_count = 0;
  s->copy_count = 0;
  s->event_count = 0;
  s->buffers_requested = 0;
  s->buffers_issued = 0;
  s->buffers_completed = 0;
  s->buffers_refused = 0;
  s->dropped_total = 0;
  s->drop_query_errors = 0;
  s->parse_errors = 0;
  s->invalid_records = 0;
  s->callback_errors = 0;
  s->unknown_activity_records = 0;
  s->unknown_resource_events = 0;
  s->resource_time_unknown = 0;
  s->id_query_errors = 0;
  s->records_before_window = 0;
  s->records_after_window = 0;
  s->records_in_window = 0;
  s->ring_context_overflow = 0;
  s->ring_kernel_overflow = 0;
  s->ring_copy_overflow = 0;
  s->ring_event_overflow = 0;
  s->ring_alias_overflow = 0;
  s->duplicate_buffer_completions = 0;
  s->callback_reentrant_refused = 0;
  s->flush_failures = 0;
  s->unsubscribe_failures = 0;
  s->outside_scope_records = 0;
  s->outside_scope_graph_records = 0;
  s->unattributed_expected = 0;
  s->unattributed_other = 0;
  s->unknown_graph_records = 0;
  s->ambiguous_graph_records = 0;
  s->unknown_node_records = 0;
  s->ambiguous_node_records = 0;
  s->kind_mismatch_records = 0;
  s->missing_node_id_records = 0;
  s->boundary_overlap_records = 0;
  s->unknown_timestamp_records = 0;
  s->incomplete_records = 0;
  s->invalid_interval_records = 0;
  s->known_wrong_context_records = 0;
  s->resource_events_seen = 0;
  s->dependency_events = 0;
  s->outside_scope_updates = 0;
  s->alias_unknown = 0;
  s->graph_passes = 0;
  s->graph_pass_limit_hit = 0;
  s->ambiguous_graph_ids = 0;
  s->node_overlap = 0;
  s->expected_context_destroyed = 0;
  s->flush_failed = 0;
  s->buffer_outstanding = 0;
  s->window_token = 0;
  s->window_start = 0;
  s->window_end = 0;
  s->window_active.store(0);
  s->window_route_generation = 0;
  s->window_issued_base = 0;
  s->window_completed_base = 0;
  s->window_buffers_issued = 0;
  s->window_buffers_completed = 0;
  s->expected_green = 0;
  s->expected_context_id = 0;
  s->expected_sms = 0;
  s->expected_known = 0;
  s->context_bind_event_count = 0;
  s->slot_count = 0;
  s->node_used = 0;
  for (uint32_t i = 0; i < s->limits.max_graphs; ++i) s->slots[i].used = 0;
  s->free_count = s->limits.buffer_count;
  s->in_use_count = 0;
  for (uint32_t i = 0; i < s->limits.buffer_count; ++i) {
    s->buffer_in_use[i] = 0;
    s->free_stack[i] = s->limits.buffer_count - 1 - i;
  }
  s->accept_activity.store(0);
  s->cleanup_failed = 0;
  s->state = kCreated;
}

int cleanup_start_failure(Session *s, int original_status) {
  bool ok = true;
  for (uint32_t i = 0; i < kResourceIdCount; ++i) {
    if (s->callback_enabled[i] == 0) continue;
    if (cuptiEnableCallback(0, s->subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                            kResourceIds[i]) == CUPTI_SUCCESS) {
      s->callback_enabled[i] = 0;
    } else {
      ok = false;
      ++s->callback_errors;
    }
  }
  const CUpti_ActivityKind kinds[] = {
      CUPTI_ACTIVITY_KIND_CONTEXT,
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
      CUPTI_ACTIVITY_KIND_MEMCPY,
      CUPTI_ACTIVITY_KIND_MEMSET,
      CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DRIVER,
  };
  for (uint32_t i = 0; i < 7; ++i) {
    if (s->activity_enabled[i] == 0) continue;
    if (cuptiActivityDisable(kinds[i]) == CUPTI_SUCCESS) {
      s->activity_enabled[i] = 0;
    } else {
      ok = false;
      ++s->callback_errors;
    }
  }
  if (s->activities_registered != 0) {
    if (cuptiActivityFlushAll(0) != CUPTI_SUCCESS) {
      ok = false;
      ++s->flush_failures;
    }
  }
  bool outstanding = false;
  {
    std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
    outstanding = s->in_use_count != 0;
  }
  // Retain the exclusive subscriber while activity cleanup is unresolved.
  // A retry must not operate on global callbacks owned by a new subscriber.
  if (!ok || outstanding) {
    s->cleanup_failed = 1;
    s->state = kQuarantined;
    return S5RO_CLEANUP_FAILED;
  }
  if (s->subscribed != 0) {
    if (cuptiUnsubscribe(s->subscriber) == CUPTI_SUCCESS) {
      s->subscribed = 0;
    } else {
      ok = false;
      ++s->unsubscribe_failures;
    }
  }
  {
    std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
    outstanding = s->in_use_count != 0;
    if (ok && !outstanding) s->accept_activity.store(0);
  }
  if (ok && !outstanding) {
    std::unique_lock<std::mutex> lock(g_mu);
    if (g_session.get() == s) g_accept_callbacks = false;
    g_cv.wait(lock, [] { return g_callback_active == 0; });
    s->activities_registered = 0;
    s->state = kCreated;
    s->cleanup_failed = 0;
    return original_status;
  }
  s->cleanup_failed = 1;
  s->state = kQuarantined;
  return S5RO_CLEANUP_FAILED;
}

int start_impl(Session *s) {
  if (s->state != kCreated) return S5RO_STATE;
  uint32_t version = 0;
  if (cuptiGetVersion(&version) != CUPTI_SUCCESS) return S5RO_VERSION_QUERY_FAILED;
  if (version != CUPTI_API_VERSION) return S5RO_VERSION_MISMATCH;
  s->runtime_api = version;

  reset_capture_state(s);

  // Acquire subscription ownership before touching the process-global activity
  // callback pointers.  A conflicting subscriber therefore keeps its callbacks
  // and its deliveries; we do not claim to restore a prior callback pointer.
  CUptiResult rc = cuptiSubscribe(&s->subscriber, resource_callback, s);
  if (rc == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
    return S5RO_SUBSCRIBER_CONFLICT;
  }
  if (rc != CUPTI_SUCCESS) return S5RO_SUBSCRIBE_FAILED;
  s->subscribed = 1;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_session.get() != s) {
      cuptiUnsubscribe(s->subscriber);
      s->subscribed = 0;
      return S5RO_UNKNOWN_SESSION;
    }
    g_accept_callbacks = true;
  }
  {
    std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
    s->accept_activity.store(1);
  }

  rc = cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
  if (rc != CUPTI_SUCCESS) {
    return cleanup_start_failure(s, S5RO_ACTIVITY_REGISTER_FAILED);
  }
  s->activities_registered = 1;

  for (uint32_t i = 0; i < kResourceIdCount; ++i) {
    if (cuptiEnableCallback(1, s->subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                            kResourceIds[i]) != CUPTI_SUCCESS) {
      return cleanup_start_failure(s, S5RO_CALLBACK_ENABLE_FAILED);
    }
    s->callback_enabled[i] = 1;
  }

  const CUpti_ActivityKind kinds[] = {
      CUPTI_ACTIVITY_KIND_CONTEXT,
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
      CUPTI_ACTIVITY_KIND_MEMCPY,
      CUPTI_ACTIVITY_KIND_MEMSET,
  };
  // Context provenance must start before green-context creation. Execution
  // activity must NOT span model loading, warmup or graph capture: it exhausted
  // the bounded callback buffers before the first replay in full8.
  for (uint32_t i = 0; i < 1; ++i) {
    if (cuptiActivityEnable(kinds[i]) != CUPTI_SUCCESS) {
      return cleanup_start_failure(s, S5RO_ACTIVITY_ENABLE_FAILED);
    }
    s->activity_enabled[i] = 1;
  }
  sample_dropped(s);
  s->state = kStarted;
  return S5RO_OK;
}

int stop_impl(Session *s) {
  if(s->correlation_enabled&&s->open_scopes)return S5RO_BUSY;
  if (s->state == kWindow) return S5RO_WINDOW_ACTIVE;
  if (s->state == kCreated || s->state == kStopped) return S5RO_STATE;
  if (s->state != kStarted && s->state != kEnded &&
      s->state != kQuarantined) {
    return S5RO_STATE;
  }
  const CUpti_ActivityKind kinds[] = {
      CUPTI_ACTIVITY_KIND_CONTEXT,
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
      CUPTI_ACTIVITY_KIND_MEMCPY,
      CUPTI_ACTIVITY_KIND_MEMSET,
      CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DRIVER,
  };
  bool disable_ok = true;
  for (uint32_t i = 0; i < 7; ++i) {
    if (s->activity_enabled[i] == 0) continue;
    if (cuptiActivityDisable(kinds[i]) == CUPTI_SUCCESS) {
      s->activity_enabled[i] = 0;
    } else {
      disable_ok = false;
      ++s->callback_errors;
    }
  }
  bool flush_ok = true;
  if (s->activities_registered != 0) {
    if (cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED) != CUPTI_SUCCESS) {
      flush_ok = false;
      ++s->flush_failures;
    }
  }
  sample_dropped(s);
  bool outstanding = false;
  {
    std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
    outstanding = s->in_use_count != 0;
  }
  if (outstanding) return S5RO_BUSY;
  if (!disable_ok || !flush_ok) {
    s->cleanup_failed = 1;
    s->state = kQuarantined;
    return S5RO_CLEANUP_FAILED;
  }
  if (s->subscribed != 0) {
    if (cuptiUnsubscribe(s->subscriber) != CUPTI_SUCCESS) {
      ++s->unsubscribe_failures;
      s->cleanup_failed = 1;
      s->state = kQuarantined;
      return S5RO_UNSUBSCRIBE_FAILED;
    }
    s->subscribed = 0;
  }
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_session.get() == s) g_accept_callbacks = false;
  }
  {
    std::unique_lock<std::mutex> lock(g_mu);
    g_cv.wait(lock, [] { return g_callback_active == 0; });
  }
  {
    std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
    s->accept_activity.store(0);
    s->window_active.store(0);
    s->window_token = 0;
    s->cleanup_failed = 0;
    s->state = kStopped;
  }
  s->activities_registered = 0;
  return S5RO_OK;
}

}  // namespace

extern "C" {


int s5ro_launch_begin(uint64_t session){
 auto s=acquire_session(session);if(!s)return S5RO_UNKNOWN_SESSION;
 std::lock_guard<std::mutex> op(s->op_mu);
 if(s->state!=kStarted||s->launch_sampling||s->sample_start)return S5RO_STATE;
 if(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)!=CUPTI_SUCCESS){s->flush_failed=1;return S5RO_FLUSH_FAILED;}
 sample_dropped(s.get());{std::lock_guard<std::recursive_mutex> lock(s->cb_mu);if(s->in_use_count)return S5RO_BUSY;s->kernel_count=0;}
 if(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)!=CUPTI_SUCCESS)return S5RO_ACTIVITY_ENABLE_FAILED;
 s->activity_enabled[1]=1;
 if(cuptiGetTimestamp(&s->sample_start)!=CUPTI_SUCCESS||!s->sample_start)return S5RO_TIMESTAMP_FAILED;
 s->launch_sampling=true;return S5RO_OK;
}

int s5ro_launch_report(uint64_t session,uint32_t finish_sample,char*out,uint64_t cap,uint64_t*len){
 if(!out||!len||finish_sample>1)return S5RO_BAD_ARGS;
 *len=0;
 auto s=acquire_session(session);if(!s)return S5RO_UNKNOWN_SESSION;
 std::lock_guard<std::mutex> op(s->op_mu);
 uint64_t begin=s->window_start,end=s->window_end;
 if(finish_sample){
  if(s->state!=kStarted||(!s->launch_sampling&&!s->sample_end))return S5RO_STATE;
  if(s->launch_sampling){
  if(cuptiGetTimestamp(&s->sample_end)!=CUPTI_SUCCESS)return S5RO_TIMESTAMP_FAILED;
  if(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)!=CUPTI_SUCCESS){s->flush_failed=1;return S5RO_ACTIVITY_ENABLE_FAILED;}
  s->activity_enabled[1]=0;
  if(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)!=CUPTI_SUCCESS){s->flush_failed=1;return S5RO_FLUSH_FAILED;}
  sample_dropped(s.get());s->launch_sampling=false;}
  begin=s->sample_start;end=s->sample_end;
 }else if(s->state!=kEnded)return S5RO_STATE;
 std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
 Json j(out,cap);j.open_obj();j.key("schema");j.u64(1);
 j.key("complete");j.boolean(s->in_use_count==0&&!s->ring_kernel_overflow&&!s->ring_event_overflow&&!s->dropped_total&&!s->flush_failed&&!s->buffers_refused&&!s->drop_query_errors&&!s->callback_errors&&!s->parse_errors&&!s->invalid_records&&!s->unknown_activity_records&&!s->ring_context_overflow&&!s->ring_copy_overflow&&!s->ring_alias_overflow&&!s->duplicate_buffer_completions&&!s->callback_reentrant_refused&&!s->id_query_errors);
 j.key("begin");j.u64(begin);j.key("end");j.u64(end);
 j.key("aliases");j.open_arr();
 for(uint32_t i=0;i<s->event_count;i++){auto&e=s->events[i];if((e.has_graph&&e.has_original_graph)||(e.has_node&&e.has_original_node)){
  j.open_arr();j.u64(e.has_graph?e.graph_id:0);j.u64(e.has_original_graph?e.original_graph_id:0);
  j.u64(e.has_node?e.node_id:0);j.u64(e.has_original_node?e.original_node_id:0);j.close_arr();}}
 j.close_arr();j.key("nodes");j.open_arr();
 std::sort(s->kernels,s->kernels+s->kernel_count,[](const ExecRec&a,const ExecRec&b){return a.graph_id<b.graph_id||(a.graph_id==b.graph_id&&a.graph_node_id<b.graph_node_id);});
 for(uint32_t i=0;i<s->kernel_count;){uint32_t endi=i+1;auto&r=s->kernels[i];
  while(endi<s->kernel_count&&s->kernels[endi].graph_id==r.graph_id&&s->kernels[endi].graph_node_id==r.graph_node_id)endi++;
  bool same=true,valid=true;for(uint32_t k=i;k<endi;k++){auto&q=s->kernels[k];
   valid=valid&&q.launch_complete&&q.graph_id&&q.graph_node_id&&q.start>=begin&&q.end<=end&&q.end>q.start;
   same=same&&std::memcmp(q.launch,r.launch,sizeof(r.launch))==0&&std::strcmp(q.name,r.name)==0;}
  j.open_obj();j.key("graph");j.u64(r.graph_id);j.key("node");j.u64(r.graph_node_id);
  j.key("complete");j.boolean(valid);j.key("stable");j.boolean(same);j.key("records");j.u64(endi-i);
  j.key("name");j.string(r.name);j.key("fields");j.open_arr();for(auto v:r.launch)j.u64(v);j.close_arr();j.close_obj();i=endi;
 }j.close_arr();j.close_obj();*len=j.len;if(j.truncated()||j.bad)return S5RO_TRUNCATED;if(cap>j.len)out[j.len]=0;
 return S5RO_OK;
}

int s5ro_graph_ids(uint64_t handle,char*out,uint64_t cap,uint64_t*len){
 if(!handle||!out||!len)return S5RO_BAD_ARGS;
 *len=0;CUgraph g=reinterpret_cast<CUgraph>(handle);
 uint32_t id=0;size_t n=0;if(cuptiGetGraphId(g,&id)!=CUPTI_SUCCESS||cuGraphGetNodes(g,nullptr,&n)!=CUDA_SUCCESS||n>32768)return S5RO_GRAPH_BIND_FAILED;
 std::unique_ptr<CUgraphNode[]>nodes(new(std::nothrow) CUgraphNode[n]);if(!nodes)return S5RO_INTERNAL_ERROR;
 size_t got=n;if(cuGraphGetNodes(g,nodes.get(),&got)!=CUDA_SUCCESS||got!=n)return S5RO_GRAPH_BIND_FAILED;
 Json j(out,cap);j.open_obj();j.key("graph");j.u64(id);j.key("nodes");j.open_arr();
 for(size_t i=0;i<n;i++){CUgraphNodeType t;uint64_t nid=0;
  if(cuGraphNodeGetType(nodes[i],&t)!=CUDA_SUCCESS||cuptiGetGraphNodeId(nodes[i],&nid)!=CUPTI_SUCCESS)return S5RO_GRAPH_BIND_FAILED;
  if(t==CU_GRAPH_NODE_TYPE_KERNEL)j.u64(nid);
 }j.close_arr();j.close_obj();*len=j.len;if(j.truncated()||j.bad)return S5RO_TRUNCATED;if(cap>j.len)out[j.len]=0;
 return S5RO_OK;
}

int s5ro_abi(void) { return static_cast<int>(kAbi); }

int s5ro_create(uint32_t buffer_count, uint64_t buffer_bytes,
                uint32_t max_context_records, uint32_t max_kernel_records,
                uint32_t max_copy_records, uint32_t max_graph_events,
                uint32_t max_graphs, uint32_t max_nodes_per_graph,
                uint32_t max_nodes_total, uint32_t max_links,
                uint64_t *session_out) {
  if (session_out == nullptr) return S5RO_BAD_ARGS;
  *session_out = 0;
  Limits limits;
  limits.buffer_count = buffer_count;
  limits.buffer_bytes = buffer_bytes;
  limits.max_context_records = max_context_records;
  limits.max_kernel_records = max_kernel_records;
  limits.max_copy_records = max_copy_records;
  limits.max_graph_events = max_graph_events;
  limits.max_graphs = max_graphs;
  limits.max_nodes_per_graph = max_nodes_per_graph;
  limits.max_nodes_total = max_nodes_total;
  limits.max_links = max_links;
  if (!valid_limit(limits)) return S5RO_BAD_ARGS;

  std::lock_guard<std::mutex> lock(g_mu);
  if (g_session) return S5RO_STATE;

  Session *s = new (std::nothrow) Session();
  if (s == nullptr) return S5RO_INTERNAL_ERROR;
  s->limits = limits;
  uint64_t reserved = compute_reserved(limits);
  if (reserved > kMaxReservedBytes) {
    delete s;
    return S5RO_LIMIT_EXCEEDED;
  }
  s->reserved_bytes = reserved;
  s->buffers = new (std::nothrow) uint8_t *[buffer_count]();
  s->buffer_in_use = new (std::nothrow) uint8_t[buffer_count]();
  s->free_stack = new (std::nothrow) uint32_t[buffer_count]();
  s->contexts = alloc_array<ContextRec>(max_context_records);
  s->kernels = alloc_array<ExecRec>(max_kernel_records);
  s->copies = alloc_array<CopyRec>(max_copy_records);
  s->events = alloc_array<EventRec>(max_graph_events);
  s->slots = alloc_array<GraphSlot>(max_graphs);
  s->nodes = alloc_array<NodeInfo>(max_nodes_total);
  s->node_handles = alloc_array<CUgraphNode>(max_nodes_per_graph);
  s->graph_map = alloc_array<GraphMap>(
      static_cast<size_t>(max_graph_events) + max_graphs);
  s->node_keys = alloc_array<NodeKey>(max_nodes_total);
  s->node_aliases = alloc_array<GraphAlias>(max_links);
  bool ok = s->buffers != nullptr && s->buffer_in_use != nullptr &&
            s->free_stack != nullptr && s->contexts != nullptr &&
            s->kernels != nullptr && s->copies != nullptr &&
            s->events != nullptr && s->slots != nullptr &&
            s->nodes != nullptr && s->node_handles != nullptr &&
            s->graph_map != nullptr && s->node_keys != nullptr &&
            s->node_aliases != nullptr;
  if (ok) {
    for (uint32_t i = 0; i < buffer_count; ++i) {
      void *buffer = nullptr;
      if (posix_memalign(&buffer, 64, static_cast<size_t>(buffer_bytes)) != 0) {
        ok = false;
        break;
      }
      s->buffers[i] = static_cast<uint8_t *>(buffer);
      s->free_stack[i] = buffer_count - 1 - i;
    }
    s->free_count = buffer_count;
  }
  if (!ok) {
    free_session_storage(s);
    delete s;
    return S5RO_INTERNAL_ERROR;
  }
  s->token = g_next_token++;
  if (s->token == 0) s->token = g_next_token++;
  g_session = SessionPtr(s, SessionDeleter{});
  *session_out = s->token;
  return S5RO_OK;
}

int s5ro_reserved_bytes(uint64_t session, uint64_t *bytes_out) {
  if (bytes_out == nullptr) return S5RO_BAD_ARGS;
  *bytes_out = 0;
  SessionPtr s = acquire_session(session);
  if (!s) return S5RO_UNKNOWN_SESSION;
  std::lock_guard<std::mutex> op(s->op_mu);
  if (s->destroyed.load() != 0 || s->token != session) {
    return S5RO_UNKNOWN_SESSION;
  }
  *bytes_out = s->reserved_bytes;
  return S5RO_OK;
}

int s5ro_start(uint64_t session) {
  SessionPtr s = acquire_session(session);
  if (!s) return S5RO_UNKNOWN_SESSION;
  std::lock_guard<std::mutex> op(s->op_mu);
  if (s->destroyed.load() != 0 || s->token != session) {
    return S5RO_UNKNOWN_SESSION;
  }
  return start_impl(s.get());
}

int s5ro_bind_context(uint64_t session, uint64_t green_ctx, uint32_t expected_sms) {
  SessionPtr s = acquire_session(session);
  if (!s) return S5RO_UNKNOWN_SESSION;
  std::lock_guard<std::mutex> op(s->op_mu);
  if (s->destroyed.load() != 0 || s->token != session) {
    return S5RO_UNKNOWN_SESSION;
  }
  if (s->state != kStarted) return S5RO_STATE;
  if (s->window_active.load() != 0) return S5RO_WINDOW_ACTIVE;
  if (s->expected_known != 0) return S5RO_CONTEXT_CONFLICT;
  if (green_ctx == 0 || expected_sms == 0) return S5RO_BAD_ARGS;
  CUcontext context = nullptr;
  CUresult cr = cuCtxFromGreenCtx(
      &context, reinterpret_cast<CUgreenCtx>(static_cast<uintptr_t>(green_ctx)));
  if (cr != CUDA_SUCCESS || context == nullptr) return S5RO_CONTEXT_BIND_FAILED;
  uint32_t context_id = 0;
  if (cuptiGetContextId(context, &context_id) != CUPTI_SUCCESS) {
    return S5RO_CONTEXT_BIND_FAILED;
  }
  {
    std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
    s->expected_green = green_ctx;
    s->expected_context_id = context_id;
    s->expected_sms = expected_sms;
    s->expected_known = 1;
    s->context_bind_event_count = s->event_count;
    s->expected_context_destroyed = 0;
  }
  return S5RO_OK;
}

int s5ro_bind_graph(uint64_t session, uint64_t graph, uint32_t slot,
                    uint32_t *node_count_out) {
  SessionPtr s = acquire_session(session);
  if (!s) return S5RO_UNKNOWN_SESSION;
  std::lock_guard<std::mutex> op(s->op_mu);
  if (s->destroyed.load() != 0 || s->token != session) {
    return S5RO_UNKNOWN_SESSION;
  }
  if (s->expected_known == 0) return S5RO_NOT_BOUND;
  return bind_graph_impl(s.get(), graph, slot, node_count_out);
}

int s5ro_static(uint64_t session, char *out, uint64_t out_cap,
                uint64_t *out_len) {
  if (out == nullptr || out_len == nullptr) return S5RO_BAD_ARGS;
  *out_len = 0;
  SessionPtr s = acquire_session(session);
  if (!s) return S5RO_UNKNOWN_SESSION;
  std::lock_guard<std::mutex> op(s->op_mu);
  if (s->destroyed.load() != 0 || s->token != session) {
    return S5RO_UNKNOWN_SESSION;
  }
  if (s->state != kStarted || s->expected_known == 0) return S5RO_NOT_BOUND;
  Json j(out, out_cap);
  emit_static_document(s.get(), j);
  *out_len = j.len;
  if (j.truncated() || j.bad) return S5RO_TRUNCATED;
  if (out_cap > j.len) out[j.len] = '\0';
  return S5RO_OK;
}


// Read-only retained-node/range query. Never resets loss, opens a second
// window, replays a graph, or changes the placement decision.
int s5ro_context_diagnostics(uint64_t session,uint64_t start,uint64_t end,
 char*out,uint64_t cap,uint64_t*len){
 if(!out||!len||(start==0)!=(end==0)||end<start)return S5RO_BAD_ARGS;
 *len=0;auto s=acquire_session(session);if(!s)return S5RO_UNKNOWN_SESSION;
 std::lock_guard<std::mutex> op(s->op_mu);
 if(s->destroyed.load()||(s->state!=kStarted&&s->state!=kEnded))return S5RO_STATE;
 if(start&&(s->state!=kEnded||start<s->window_start||end>s->window_end))return S5RO_BAD_ARGS;
 std::lock_guard<std::recursive_mutex> cb(s->cb_mu);
 uint32_t graphs=start?s->diag_graphs:0,keys=start?s->diag_keys:0,aliases=start?s->diag_aliases:0;
 try{Json j(out,cap);emit_context_diagnostics(s.get(),graphs,keys,aliases,start,end,true,j);
  *len=j.len;if(j.truncated()||j.bad)return S5RO_TRUNCATED;
  if(cap>j.len)out[j.len]='\0';
  return S5RO_OK;
 }catch(...){return S5RO_BAD_ARGS;}
}

// Additive diagnostic only: every retained record in a drained interval,
// including graphId==0. No name/graph whitelist and no placement-rule change.
// Keys contain scoped CUPTI ids, names and copy directions, never pointers.
int s5ro_activity_census(uint64_t session,uint64_t start,uint64_t end,
 char*out,uint64_t cap,uint64_t*len){
 if(!out||!len||!start||end<=start)return S5RO_BAD_ARGS;
 *len=0;auto s=acquire_session(session);if(!s)return S5RO_UNKNOWN_SESSION;
 std::lock_guard<std::mutex> op(s->op_mu);
 if(s->destroyed.load()||s->state!=kEnded||start<s->window_start||end>s->window_end)return S5RO_STATE;
 std::lock_guard<std::recursive_mutex> cb(s->cb_mu);
 try{
 using Key=std::tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint64_t,uint32_t,std::string>;
 std::map<Key,uint64_t> rows;uint64_t boundary=0,invalid=0,omitted=0,matched=0;
 constexpr size_t row_cap=4096;
 auto add=[&](uint32_t kind,uint32_t ctx,uint32_t stream,uint32_t graph,uint64_t node,
              uint32_t direction,const char*name,uint64_t a,uint64_t b){
   if(!a||b<=a){++invalid;return;}
   if(b<=start||a>=end)return;
   if(a<start||b>end){++boundary;return;}
   ++matched;Key key(kind,ctx,stream,graph,node,direction,std::string(name));
   auto it=rows.find(key);if(it!=rows.end())++it->second;
   else if(rows.size()<row_cap)rows.emplace(std::move(key),1);
   else ++omitted;
 };
 for(uint32_t i=0;i<s->kernel_count;++i){const auto&r=s->kernels[i];
   if(!r.launch_complete)++invalid;
   add(kRecKernel,r.context_id,r.stream_id,r.graph_id,r.graph_node_id,0,r.name,r.start,r.end);}
 for(uint32_t i=0;i<s->copy_count;++i){const auto&r=s->copies[i];
   add(r.kind,r.context_id,r.stream_id,r.graph_id,r.graph_node_id,r.copy_kind,"",r.start,r.end);}
 Json j(out,cap);j.open_obj();j.key("schema");j.string("s5ro-activity-census/1");
 j.key("complete");j.boolean(!boundary&&!invalid&&!omitted);
 j.key("scope");j.string("retained records only; caller must also require native loss counters and guards");
 j.key("start");j.u64(start);j.key("end");j.u64(end);
 j.key("boundary_records");j.u64(boundary);j.key("invalid_records");j.u64(invalid);
 j.key("omitted_records");j.u64(omitted);j.key("matched_records");j.u64(matched);
 j.key("row_cap");j.u64(row_cap);j.key("rows");j.open_arr();
 for(const auto&it:rows){const auto&k=it.first;j.open_obj();
   j.key("kind");j.string(std::get<0>(k)==kRecKernel?"kernel":std::get<0>(k)==kRecMemcpy?"memcpy":"memset");
   j.key("context_id");j.u64(std::get<1>(k));j.key("stream_id");j.u64(std::get<2>(k));
   j.key("graph_id");j.u64(std::get<3>(k));j.key("node_id");j.u64(std::get<4>(k));
   j.key("copy_kind");j.u64(std::get<5>(k));j.key("name");j.string(std::get<6>(k).c_str());
   j.key("count");j.u64(it.second);j.close_obj();}
 j.close_arr();j.close_obj();*len=j.len;if(j.truncated()||j.bad)return S5RO_TRUNCATED;
 if(cap>j.len)out[j.len]=0;
 return S5RO_OK;
 }catch(...){return S5RO_INTERNAL_ERROR;}
}

int s5ro_correlation_enable(uint64_t session){
 auto s=acquire_session(session);if(!s)return S5RO_UNKNOWN_SESSION;
 std::lock_guard<std::mutex> op(s->op_mu);
 if(s->state!=kStarted||s->correlation_enabled)return S5RO_STATE;
 CUpti_ActivityThreadIdType type;
 if(cuptiGetThreadIdType(&type)!=CUPTI_SUCCESS||
    (type!=CUPTI_ACTIVITY_THREAD_ID_TYPE_DEFAULT&&type!=CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM))return S5RO_STATE;
 s->correlation_thread_type=type;
 const uint64_t bytes=kCorrCapacity*(sizeof(ApiRec)+sizeof(ExternalRec)+sizeof(ScopeRec));
 if(s->reserved_bytes+bytes>kMaxReservedBytes)return S5RO_LIMIT_EXCEEDED;
 std::unique_ptr<ApiRec[]>a(alloc_array<ApiRec>(kCorrCapacity));
 std::unique_ptr<ExternalRec[]>e(alloc_array<ExternalRec>(kCorrCapacity));
 std::unique_ptr<ScopeRec[]>q(alloc_array<ScopeRec>(kCorrCapacity));
 if(!a||!e||!q)return S5RO_INTERNAL_ERROR;
 s->apis=a.release();s->external=e.release();s->scopes=q.release();
 s->reserved_bytes+=bytes;s->correlation_enabled=true;return S5RO_OK;
}

int s5ro_correlation_thread_key(uint64_t session,uint32_t*key,uint32_t*system){
 if(!key||!system)return S5RO_BAD_ARGS;
 auto s=acquire_session(session);if(!s)return S5RO_UNKNOWN_SESSION;
 std::lock_guard<std::mutex> op(s->op_mu);
 CUpti_ActivityThreadIdType type;
 if(!s->correlation_enabled||cuptiGetThreadIdType(&type)!=CUPTI_SUCCESS||type!=s->correlation_thread_type){++s->correlation_errors;return S5RO_STATE;}
 *system=static_cast<uint32_t>(syscall(SYS_gettid));
 *key=type==CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM?*system:static_cast<uint32_t>(pthread_self());
 return S5RO_OK;
}

int s5ro_offer_push(uint64_t session,uint64_t id,uint32_t generation){
 auto s=acquire_session(session);if(!s)return S5RO_UNKNOWN_SESSION;
 std::lock_guard<std::mutex> op(s->op_mu);
 if(s->state!=kWindow||!s->correlation_enabled||generation!=s->window_route_generation||!id||tl_scopes.size()>=32){++s->correlation_errors;return S5RO_STATE;}
 if(!tl_scopes.empty()&&tl_scopes.back().session!=session){++s->correlation_errors;return S5RO_STATE;}
 try{if(tl_scopes.capacity()<32)tl_scopes.reserve(32);}catch(...){++s->correlation_errors;return S5RO_INTERNAL_ERROR;}
 CUpti_ActivityThreadIdType type;
 if(cuptiGetThreadIdType(&type)!=CUPTI_SUCCESS||type!=s->correlation_thread_type){++s->correlation_errors;return S5RO_STATE;}
 uint32_t system=static_cast<uint32_t>(syscall(SYS_gettid));
 uint32_t thread=type==CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM?system:static_cast<uint32_t>(pthread_self());
 if(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,id)!=CUPTI_SUCCESS){++s->correlation_errors;return S5RO_STATE;}
 uint64_t parent=tl_scopes.empty()?0:tl_scopes.back().id;
 tl_scopes.push_back({session,id});++s->open_scopes;
 std::lock_guard<std::recursive_mutex> cb(s->cb_mu);
 if(s->scope_count>=kCorrCapacity)++s->correlation_overflow;
 else s->scopes[s->scope_count++]={id,parent,thread,1,system};
 return S5RO_OK;
}
int s5ro_offer_pop(uint64_t session,uint64_t id){
 auto s=acquire_session(session);if(!s)return S5RO_UNKNOWN_SESSION;
 std::lock_guard<std::mutex> op(s->op_mu);
 if(!s->correlation_enabled||tl_scopes.empty()||tl_scopes.back().session!=session||tl_scopes.back().id!=id){++s->correlation_errors;return S5RO_STATE;}
 CUpti_ActivityThreadIdType type;
 if(cuptiGetThreadIdType(&type)!=CUPTI_SUCCESS||type!=s->correlation_thread_type){++s->correlation_errors;return S5RO_STATE;}
 uint32_t system=static_cast<uint32_t>(syscall(SYS_gettid));
 uint32_t thread=type==CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM?system:static_cast<uint32_t>(pthread_self());
 uint64_t got=0;if(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0,&got)!=CUPTI_SUCCESS||got!=id){++s->correlation_errors;return S5RO_STATE;}
 tl_scopes.pop_back();--s->open_scopes;
 std::lock_guard<std::recursive_mutex> cb(s->cb_mu);
 if(s->scope_count>=kCorrCapacity)++s->correlation_overflow;
 else s->scopes[s->scope_count++]={id,0,thread,2,system};
 return S5RO_OK;
}
int s5ro_correlation_report(uint64_t session,char*out,uint64_t cap,uint64_t*len){
 if(!out||!len)return S5RO_BAD_ARGS;
 *len=0;
 auto s=acquire_session(session);if(!s)return S5RO_UNKNOWN_SESSION;
 std::lock_guard<std::mutex> op(s->op_mu);
 if(!s->correlation_enabled||s->state!=kEnded)return S5RO_STATE;
 std::lock_guard<std::recursive_mutex> cb(s->cb_mu);
 try{
 CUpti_ActivityThreadIdType type;
 if(cuptiGetThreadIdType(&type)!=CUPTI_SUCCESS||type!=s->correlation_thread_type){++s->correlation_errors;}
 using Key=std::tuple<uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,std::string>;
 std::map<Key,uint64_t> rows;uint64_t boundary=0,records=0;
 auto add=[&](uint32_t corr,uint32_t kind,uint32_t ctx,uint32_t stream,uint32_t graph,uint32_t direction,const char*name,uint64_t a,uint64_t b){
   if(!a||b<=a||a<s->window_start||b>s->window_end){++boundary;return;}
   ++rows[Key(corr,kind,ctx,stream,graph,direction,std::string(name))];++records;
 };
 for(uint32_t i=0;i<s->kernel_count;i++){const auto&r=s->kernels[i];add(r.correlation_id,kRecKernel,r.context_id,r.stream_id,r.graph_id,0,r.name,r.start,r.end);}
 for(uint32_t i=0;i<s->copy_count;i++){const auto&r=s->copies[i];add(r.correlation_id,r.kind,r.context_id,r.stream_id,r.graph_id,r.copy_kind,"",r.start,r.end);}
 for(uint32_t i=0;i<s->api_count;i++){const auto&r=s->apis[i];if(!r.start||r.end<r.start||r.start<s->window_start||r.end>s->window_end)++boundary;}
 Json j(out,cap);j.open_obj();j.key("schema");j.string("s5ro-correlation/2");
 j.key("thread_id_type");j.string(s->correlation_thread_type==CUPTI_ACTIVITY_THREAD_ID_TYPE_SYSTEM?"system-gettid":"default-pthread-low32");
 j.key("complete");j.boolean(!s->correlation_errors&&!s->correlation_overflow&&!s->open_scopes&&!boundary);
 j.key("capacity");j.u64(kCorrCapacity);j.key("reserved_bytes");j.u64(s->reserved_bytes);
 j.key("errors");j.u64(s->correlation_errors);j.key("overflow");j.u64(s->correlation_overflow);
 j.key("open_scopes");j.u64(s->open_scopes);j.key("boundary_records");j.u64(boundary);
 j.key("activity_records");j.u64(records);j.key("apis");j.open_arr();
 for(uint32_t i=0;i<s->api_count;i++){const auto&r=s->apis[i];j.open_obj();
  j.key("domain");j.string(r.domain==CUPTI_ACTIVITY_KIND_RUNTIME?"runtime":"driver");j.key("name");j.string(r.name);
  j.key("correlation_id");j.u64(r.correlation);j.key("thread");j.u64(r.thread);j.key("return_value");j.u64(r.result);
  j.key("start");j.u64(r.start);j.key("end");j.u64(r.end);j.close_obj();}j.close_arr();
 j.key("external");j.open_arr();for(uint32_t i=0;i<s->external_count;i++){const auto&r=s->external[i];j.open_obj();
  j.key("kind");j.string(r.kind==CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0?"custom0":"unsupported");
  j.key("correlation_id");j.u64(r.correlation);j.key("external_id");j.u64(r.external);j.close_obj();}j.close_arr();
 j.key("scopes");j.open_arr();for(uint32_t i=0;i<s->scope_count;i++){const auto&r=s->scopes[i];j.open_obj();
  j.key("id");j.u64(r.id);j.key("parent");j.u64(r.parent);j.key("thread");j.u64(r.thread);
  j.key("system_thread");j.u64(r.system_thread);
  j.key("action");j.string(r.action==1?"push":"pop");j.close_obj();}j.close_arr();
 j.key("activities");j.open_arr();for(const auto&it:rows){const auto&k=it.first;j.open_obj();
  j.key("correlation_id");j.u64(std::get<0>(k));j.key("kind");j.string(std::get<1>(k)==kRecKernel?"kernel":std::get<1>(k)==kRecMemcpy?"memcpy":"memset");
  j.key("context_id");j.u64(std::get<2>(k));j.key("stream_id");j.u64(std::get<3>(k));j.key("graph_id");j.u64(std::get<4>(k));
  j.key("copy_kind");j.u64(std::get<5>(k));j.key("name");j.string(std::get<6>(k).c_str());j.key("count");j.u64(it.second);j.close_obj();}j.close_arr();
 j.close_obj();*len=j.len;if(j.truncated()||j.bad)return S5RO_TRUNCATED;
 if(cap>j.len)out[j.len]=0;
 return S5RO_OK;
 }catch(...){return S5RO_INTERNAL_ERROR;}
}

int s5ro_begin_window(uint64_t session, uint32_t route_generation,
                      uint64_t *token_out) {
  if (token_out == nullptr) return S5RO_BAD_ARGS;
  *token_out = 0;
  SessionPtr s = acquire_session(session);
  if (!s) return S5RO_UNKNOWN_SESSION;
  std::lock_guard<std::mutex> op(s->op_mu);
  if (s->destroyed.load() != 0 || s->token != session) {
    return S5RO_UNKNOWN_SESSION;
  }
  if (s->state != kStarted) return S5RO_STATE;
  if (s->window_active.load() != 0 || s->window_token != 0 || s->launch_sampling.load() != 0) {
    return S5RO_WINDOW_ACTIVE;
  }
  if (s->expected_known == 0 || s->slot_count == 0) return S5RO_NOT_BOUND;
  // The caller has paused and drained the engine. Return startup/context
  // buffers before borrowing replay credit. Never hold cb_mu across CUPTI:
  // flush can synchronously wait for a callback on another thread.
  if (cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED) != CUPTI_SUCCESS) {
    ++s->flush_failures;
    s->flush_failed = 1;
    s->state = kQuarantined;
    return S5RO_FLUSH_FAILED;
  }
  sample_dropped(s.get());
  {
    std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
    if (s->in_use_count != 0) return S5RO_BUSY;
  }
  const CUpti_ActivityKind execution_kinds[] = {
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
      CUPTI_ACTIVITY_KIND_MEMCPY,
      CUPTI_ACTIVITY_KIND_MEMSET,
      CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DRIVER,
  };
  for (uint32_t i = 0; i < (s->correlation_enabled?6u:3u); ++i) {
    if (cuptiActivityEnable(execution_kinds[i]) != CUPTI_SUCCESS) {
      ++s->callback_errors;
      s->state = kQuarantined;
      return S5RO_ACTIVITY_ENABLE_FAILED;
    }
    s->activity_enabled[i + 1] = 1;
  }
  uint64_t start = 0;
  if (cuptiGetTimestamp(&start) != CUPTI_SUCCESS || start == 0) {
    s->state = kQuarantined;
    return S5RO_TIMESTAMP_FAILED;
  }
  sample_dropped(s.get());
  {
    std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
    s->kernel_count = 0;
    s->copy_count = 0;
    s->window_token = g_next_window_token++;
    if (s->window_token == 0) s->window_token = g_next_window_token++;
    s->window_start = start;
    s->window_end = 0;
    s->window_route_generation = route_generation;
    s->window_issued_base = s->buffers_issued;
    s->window_completed_base = s->buffers_completed;
    s->window_buffers_issued = 0;
    s->window_buffers_completed = 0;
    // Loss/error counters are lifetime evidence; never reset them at begin.
    s->window_active.store(1);
    s->state = kWindow;
    *token_out = s->window_token;
  }
  return S5RO_OK;
}

int s5ro_end_window(uint64_t session, uint64_t window_token,
                    uint32_t route_generation, char *out, uint64_t out_cap,
                    uint64_t *out_len) {
  if (out == nullptr || out_len == nullptr) return S5RO_BAD_ARGS;
  *out_len = 0;
  SessionPtr s = acquire_session(session);
  if (!s) return S5RO_UNKNOWN_SESSION;
  std::lock_guard<std::mutex> op(s->op_mu);
  if (s->destroyed.load() != 0 || s->token != session) {
    return S5RO_UNKNOWN_SESSION;
  }
  if (s->state != kWindow || s->window_active.load() == 0) {
    return S5RO_STATE;
  }
  if (s->window_token != window_token) return S5RO_TOKEN_MISMATCH;
  if (s->window_route_generation != route_generation) return S5RO_TOKEN_MISMATCH;
  uint64_t end = 0;
  if (cuptiGetTimestamp(&end) != CUPTI_SUCCESS || end < s->window_start) {
    // The native window token is not consumed here; Python keeps its caller
    // token so the same window can be retried after a transient clock error.
    return S5RO_TIMESTAMP_FAILED;
  }
  {
    std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
    s->window_end = end;
  }
  if(s->correlation_enabled&&s->open_scopes)++s->correlation_errors;
  s->end_flushing = 1;
  // Work is drained, but retain window_active until all queued records have
  // been delivered. A disable failure remains a permanent incomplete proof.
  const CUpti_ActivityKind execution_kinds[] = {
      CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
      CUPTI_ACTIVITY_KIND_MEMCPY,
      CUPTI_ACTIVITY_KIND_MEMSET,
      CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DRIVER,
  };
  for (uint32_t i = 0; i < (s->correlation_enabled?6u:3u); ++i) {
    if (s->activity_enabled[i + 1] == 0) continue;
    if (cuptiActivityDisable(execution_kinds[i]) == CUPTI_SUCCESS) {
      s->activity_enabled[i + 1] = 0;
    } else {
      ++s->callback_errors;
      s->flush_failed = 1;
    }
  }
  // Flush and all callback-waiting CUPTI operations happen with only the
  // per-session operation lock held.  A callback thread can always take its
  // own cb_mu and complete.
  if (cuptiActivityFlushAll(0) != CUPTI_SUCCESS) {
    s->flush_failed = 1;
    ++s->flush_failures;
  }
  s->end_flushing = 0;
  sample_dropped(s.get());

  std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
  s->window_active.store(0);
  s->window_buffers_issued =
      s->buffers_issued >= s->window_issued_base
          ? s->buffers_issued - s->window_issued_base
          : 0;
  s->window_buffers_completed =
      s->buffers_completed >= s->window_completed_base
          ? s->buffers_completed - s->window_completed_base
          : 0;

  uint32_t graph_count = 0;
  build_graph_map(s.get(), &graph_count);
  uint32_t key_count = 0;
  build_node_keys(s.get(), &key_count);
  uint32_t alias_count = 0;
  build_node_aliases(s.get(), key_count, graph_count, &alias_count);

  Json j(out, out_cap);
  s->diag_graphs=graph_count;s->diag_keys=key_count;s->diag_aliases=alias_count;
  emit_window_document(s.get(), graph_count, key_count, alias_count, j);
  *out_len = j.len;
  s->window_token = 0;
  s->state = kEnded;
  if (j.truncated() || j.bad) return S5RO_TRUNCATED;
  if (out_cap > j.len) out[j.len] = '\0';
  return S5RO_OK;
}

int s5ro_stop(uint64_t session) {
  SessionPtr s = acquire_session(session);
  if (!s) return S5RO_UNKNOWN_SESSION;
  std::lock_guard<std::mutex> op(s->op_mu);
  if (s->destroyed.load() != 0 || s->token != session) {
    return S5RO_UNKNOWN_SESSION;
  }
  return stop_impl(s.get());
}

int s5ro_destroy(uint64_t session) {
  SessionPtr s = acquire_session(session);
  if (!s) return S5RO_UNKNOWN_SESSION;
  std::lock_guard<std::mutex> op(s->op_mu);
  if (s->destroyed.load() != 0 || s->token != session) {
    return S5RO_UNKNOWN_SESSION;
  }
  if (s->state != kStopped && s->state != kCreated) return S5RO_STATE;
  {
    std::lock_guard<std::recursive_mutex> lock(s->cb_mu);
    if (s->in_use_count != 0 || s->subscribed != 0) return S5RO_BUSY;
  }
  {
    std::unique_lock<std::mutex> lock(g_mu);
    if (g_session.get() != s.get()) return S5RO_UNKNOWN_SESSION;
    g_accept_callbacks = false;
    g_retiring = true;
    g_session.reset();
    g_cv.wait(lock, [] { return g_callback_active == 0; });
    g_retiring = false;
  }
  s->destroyed.store(1);
  return S5RO_OK;
}

}  // extern "C"
