#ifndef S5RO_FIXTURE_CUPTI_CALLBACKS_H
#define S5RO_FIXTURE_CUPTI_CALLBACKS_H

#include <stddef.h>
#include <stdint.h>

#include "cuda.h"
#include "cupti_activity.h"
#include "cupti_result.h"

typedef uint32_t CUpti_CallbackId;

typedef enum CUpti_CallbackDomain {
  CUPTI_CB_DOMAIN_DRIVER_API=1, CUPTI_CB_DOMAIN_RUNTIME_API=2,
  CUPTI_CB_DOMAIN_RESOURCE = 3
} CUpti_CallbackDomain;

typedef struct CUpti_Subscriber_st *CUpti_SubscriberHandle;

typedef void (CUPTIAPI *CUpti_CallbackFunc)(void *userdata,
                                            CUpti_CallbackDomain domain,
                                            CUpti_CallbackId cbid,
                                            const void *cbdata);
typedef void (CUPTIAPI *CUpti_BuffersCallbackRequestFunc)(uint8_t **buffer,
                                                          size_t *size,
                                                          size_t *maxNumRecords);
typedef void (CUPTIAPI *CUpti_BuffersCallbackCompleteFunc)(CUcontext context,
                                                           uint32_t streamId,
                                                           uint8_t *buffer,
                                                           size_t size,
                                                           size_t validSize);

typedef struct CUpti_ResourceData {
  CUcontext context;
  union {
    CUstream stream;
  } resourceHandle;
  void *resourceDescriptor;
} CUpti_ResourceData;

typedef struct CUpti_GraphData {
  CUgraph graph;
  CUgraph originalGraph;
  CUgraphNode node;
  CUgraphNode originalNode;
  CUgraphNodeType nodeType;
  CUgraphNode dependency;
  CUgraphExec graphExec;
} CUpti_GraphData;

enum {
  CUPTI_CBID_RESOURCE_CONTEXT_CREATED = 1,
  CUPTI_CBID_RESOURCE_CONTEXT_DESTROY_STARTING = 2,
  CUPTI_CBID_RESOURCE_GRAPH_CREATED = 9,
  CUPTI_CBID_RESOURCE_GRAPH_DESTROY_STARTING = 10,
  CUPTI_CBID_RESOURCE_GRAPH_CLONED = 11,
  CUPTI_CBID_RESOURCE_GRAPHNODE_CREATE_STARTING = 12,
  CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED = 13,
  CUPTI_CBID_RESOURCE_GRAPHNODE_DESTROY_STARTING = 14,
  CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_CREATED = 15,
  CUPTI_CBID_RESOURCE_GRAPHNODE_DEPENDENCY_DESTROY_STARTING = 16,
  CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATE_STARTING = 17,
  CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED = 18,
  CUPTI_CBID_RESOURCE_GRAPHEXEC_DESTROY_STARTING = 19,
  CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED = 20,
  CUPTI_CBID_RESOURCE_STREAM_ATTRIBUTE_CHANGED = 21,
  CUPTI_CBID_RESOURCE_GRAPH_NODE_UPDATED = 22,
  CUPTI_CBID_RESOURCE_GRAPH_NODE_SET_PARAMS = 23
};

#ifdef __cplusplus
extern "C" {
#endif

CUptiResult cuptiGetCallbackName(CUpti_CallbackDomain,CUpti_CallbackId,const char**);
CUptiResult cuptiSubscribe(CUpti_SubscriberHandle *subscriber,
                           CUpti_CallbackFunc callback, void *userdata);
CUptiResult cuptiUnsubscribe(CUpti_SubscriberHandle subscriber);
CUptiResult cuptiEnableCallback(uint32_t enable,
                                CUpti_SubscriberHandle subscriber,
                                CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid);
CUptiResult cuptiActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc request,
    CUpti_BuffersCallbackCompleteFunc complete);

#ifdef __cplusplus
}
#endif

#endif
