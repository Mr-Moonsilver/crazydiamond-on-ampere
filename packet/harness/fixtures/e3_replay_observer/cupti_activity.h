#ifndef S5RO_FIXTURE_CUPTI_ACTIVITY_H
#define S5RO_FIXTURE_CUPTI_ACTIVITY_H

#include <stddef.h>
#include <stdint.h>

#include "cuda.h"
#include "cupti_result.h"

#define CUPTIAPI
#define CUPTI_TIMESTAMP_UNKNOWN 0LL
#define CUPTI_ACTIVITY_FLAG_FLUSH_FORCED (1u << 0)

typedef enum CUpti_ActivityKind {
  CUPTI_ACTIVITY_KIND_MEMCPY = 1,
  CUPTI_ACTIVITY_KIND_MEMSET = 2,
  CUPTI_ACTIVITY_KIND_CONTEXT = 9,
  CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10
} CUpti_ActivityKind;

typedef struct CUpti_Activity {
  CUpti_ActivityKind kind;
  uint32_t record_size;
} CUpti_Activity;

typedef struct CUpti_ActivityContext3 {
  CUpti_ActivityKind kind;
  uint32_t record_size;
  uint32_t contextId;
  uint32_t deviceId;
  uint32_t computeApiKind;
  uint32_t nullStreamId;
  uint32_t parentContextId;
  uint8_t isGreenContext;
  uint8_t padding[7];
  uint16_t numMultiprocessors;
} CUpti_ActivityContext3;

typedef struct CUpti_ActivityKernel10 {
  CUpti_ActivityKind kind;
  uint32_t record_size;
  uint64_t start;
  uint64_t end;
  uint64_t completed;
  uint32_t deviceId;
  uint32_t contextId;
  uint32_t streamId;
  int32_t gridX, gridY, gridZ;
  int32_t blockX, blockY, blockZ;
  uint32_t staticSharedMemory;
  uint32_t dynamicSharedMemory;
  uint32_t localMemoryPerThread;
  uint32_t localMemoryTotal;
  uint32_t correlationId;
  uint32_t gridId;
  const char *name;
  uint64_t graphNodeId;
  uint32_t graphId;
  uint32_t channelID;
  uint32_t channelType;
} CUpti_ActivityKernel10;

typedef struct CUpti_ActivityMemcpy6 {
  CUpti_ActivityKind kind;
  uint32_t record_size;
  uint32_t copyKind;
  uint32_t srcKind;
  uint32_t dstKind;
  uint32_t flags;
  uint64_t bytes;
  uint64_t start;
  uint64_t end;
  uint32_t deviceId;
  uint32_t contextId;
  uint32_t streamId;
  uint32_t correlationId;
  uint64_t graphNodeId;
  uint32_t graphId;
  uint32_t channelID;
  uint32_t channelType;
} CUpti_ActivityMemcpy6;

typedef struct CUpti_ActivityMemset4 {
  CUpti_ActivityKind kind;
  uint32_t record_size;
  uint32_t value;
  uint64_t bytes;
  uint64_t start;
  uint64_t end;
  uint32_t deviceId;
  uint32_t contextId;
  uint32_t streamId;
  uint32_t correlationId;
  uint32_t flags;
  uint32_t memoryKind;
  uint64_t graphNodeId;
  uint32_t graphId;
  uint32_t channelID;
  uint32_t channelType;
} CUpti_ActivityMemset4;

#ifdef __cplusplus
extern "C" {
#endif

CUptiResult cuptiActivityGetNextRecord(uint8_t *buffer,
                                       size_t validBufferSizeBytes,
                                       CUpti_Activity **record);
CUptiResult cuptiActivityGetNumDroppedRecords(CUcontext context,
                                              uint32_t streamId,
                                              size_t *dropped);
CUptiResult cuptiActivityFlushAll(uint32_t flag);
CUptiResult cuptiActivityEnable(CUpti_ActivityKind kind);
CUptiResult cuptiActivityDisable(CUpti_ActivityKind kind);
CUptiResult cuptiGetTimestamp(uint64_t *timestamp);
CUptiResult cuptiGetContextId(CUcontext context, uint32_t *contextId);
CUptiResult cuptiGetGraphId(CUgraph graph, uint32_t *pId);
CUptiResult cuptiGetGraphExecId(CUgraphExec graphExec, uint32_t *pId);
CUptiResult cuptiGetGraphNodeId(CUgraphNode node, uint64_t *nodeId);

#ifdef __cplusplus
}
#endif

#endif
