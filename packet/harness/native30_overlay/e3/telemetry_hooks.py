"""Reuse reviewed pool counters without changing the baseline grouping planner."""
import ast
import hashlib
import importlib.util
from pathlib import Path
from mirror_hooks import once

SOURCE_SHA='2c22709970bcff8123f43ee7724441cf430c17f1da02963d88d48168e80b9086'


def source_path():
    here=Path(__file__).resolve().parent
    bundled=here/'pool_stats_source.py'
    return bundled if bundled.is_file() else here.parent/'ppgrouping/make_ppgrouping.py'


def block_pool(source):
    path=source_path()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise ValueError('unreviewed pool telemetry generator')
    spec=importlib.util.spec_from_file_location('e2_pinned_pool_source',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Only the already-reviewed counters. No kv_cache_utils/planner transform.
    return module.transform('block_pool.py',source)


LAYOUT='''        # E2 lifetime evidence: original scheduler group IDs, no regrouping.
        if self.kv_cache_manager.block_pool._flash_pool_stats_every:
            _e2_groups = []
            for _gid, _group in enumerate(self.kv_cache_config.kv_cache_groups):
                _spec = _group.kv_cache_spec
                _first = getattr(_spec, "first_spec", _spec)
                _e2_groups.append({"id": _gid, "kind": type(_first).__name__,
                                   "layers": _group.layer_names,
                                   "block_tokens": _spec.block_size})
            logger.info("FLASH_PP_LAYOUT %s", __import__("json").dumps({
                "enabled": False, "scope": "E2 baseline scheduler groups; no PP grouping delta",
                "num_blocks": self.kv_cache_config.num_blocks, "groups": _e2_groups}, sort_keys=True))
'''
STEP='''        self.kv_cache_manager.block_pool.flash_pool_step()
'''


def scheduler(source):
    result=once(source,'        # Bind after construction so connectors can access the cache manager.\n',
                LAYOUT+'        # Bind after construction so connectors can access the cache manager.\n')
    result=once(result,'        self.kv_cache_manager.new_step_starts()\n',
                STEP+'        self.kv_cache_manager.new_step_starts()\n')
    ast.parse(result)
    return result
