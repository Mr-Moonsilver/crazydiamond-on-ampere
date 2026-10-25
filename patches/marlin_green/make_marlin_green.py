#!/usr/bin/env python3
"""Emit an opt-in native source delta; requires owner wheel rebuild, no install."""
import argparse
import hashlib
from pathlib import Path

PIN='edbfdfdeed22baf84547311f3e1922bb914cf56cdddc29eba3ac25f9c6fe1bf0'
ANCHOR='  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, a.get_device());'
DELTA=r'''
  // Default off. Query the ACTUAL launch stream's green resources, not a
  // requested split or whole-device attribute. No stream/global synchronization.
  static const bool flash_green_cap = [] {
    const char* flag = std::getenv("FLASH_MARLIN_GREEN_CAP");
    return flag && std::strcmp(flag, "1") == 0;
  }();
  if (flash_green_cap) {
    CUgreenCtx green{};
    auto r = cuStreamGetGreenCtx((CUstream)get_current_cuda_stream(a.get_device()), &green);
    STD_TORCH_CHECK(r == CUDA_SUCCESS, "Marlin green stream query failed: ", int(r));
    if (green) {
      CUdevResource resource{};
      r = cuGreenCtxGetDevResource(green, &resource, CU_DEV_RESOURCE_TYPE_SM);
      STD_TORCH_CHECK(r == CUDA_SUCCESS && resource.sm.smCount > 0,
                      "Marlin green SM query failed: ", int(r));
      sms = std::min(sms, int(resource.sm.smCount));
    }
  }
'''


def transform(source):
    if hashlib.sha256(source.encode()).hexdigest()!=PIN:raise ValueError('pinned upstream ops.cu required')
    if source.count(ANCHOR)!=1:raise ValueError('nonunique native anchor')
    return '#include <cuda.h>\n#include <cstdlib>\n#include <cstring>\n#include <algorithm>\n'+source.replace(ANCHOR,ANCHOR+DELTA)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    result=transform(a.source.read_text());a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'ops.cu').write_text(result)
    (a.out/'SHA256SUMS').write_text(hashlib.sha256(result.encode()).hexdigest()+'  ops.cu\n')
