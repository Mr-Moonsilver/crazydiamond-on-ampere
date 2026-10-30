#ifndef S5RO_FIXTURE_CUPTI_VERSION_H
#define S5RO_FIXTURE_CUPTI_VERSION_H

#include <stdint.h>

#include "cupti_result.h"

#define CUPTI_API_VERSION 130001u

#ifdef __cplusplus
extern "C" {
#endif

CUptiResult cuptiGetVersion(uint32_t *version);

#ifdef __cplusplus
}
#endif

#endif
