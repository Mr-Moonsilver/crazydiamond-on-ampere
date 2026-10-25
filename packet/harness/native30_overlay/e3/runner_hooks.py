"""E2 execution hooks only; startup binding/complete overlay generator pending.

The generator must first validate the full file digests below. Transform anchors
are fail-closed and individually reversible in CPU tests. These functions do
not create or install a RunnerLifecycle themselves.
"""
from mirror_hooks import once, replace_method


IMPORT = 'from vllm.v1.flash_e2.runner_lifecycle import execute_with_e2\n\n'
UPDATED = '''            _e2 = getattr(self, "_flash_e2_lifecycle", None)
            if _e2 is not None:
                _e2.updated(self.req_states.req_id_to_index, self.block_tables._flash_e2_tables)
'''
BATCH = '''        _e2 = getattr(self, "_flash_e2_lifecycle", None)
        if _e2 is not None and not dummy_run:
            _e2.batch(input_batch)
'''
COW = '''        _e2 = getattr(self, "_flash_e2_lifecycle", None)
        if _e2 is not None:
            _e2.actual_cow(len(copies), len(views))
'''
SHUTDOWN = '''        _e2 = getattr(self, "_flash_e2_lifecycle", None)
        if _e2 is not None:
            # The preceding real device synchronize has already completed.
            _e2.shutdown(lambda: None)
'''
STAGED = '''    _e2_ref = getattr(layer, "_flash_e2_lifecycle_ref", None)
    _e2 = _e2_ref() if _e2_ref is not None else None
    if _e2 is not None and _e2.frame is not None:
        from .ops.qsa import flash_e2_arena_attention
        _e2.staged(
            layer._flash_e2_layer_name, best, int(start), int(end), int(lengths[best]),
            (query[:num_tokens], kv_cache, logical_indices, attn_metadata.block_table[best],
             rows, max_logical_page, arena, use_prefill_config, output[:num_tokens]),
            qsa_sparse_paged_attention_staged, flash_e2_arena_attention)
        return start, end
'''


def runner_execution(source):
    if '_flash_e2_lifecycle' in source or 'execute_with_e2' in source:
        raise ValueError('runner already has E2 execution hooks')
    source = once(source, 'class GPUModelRunner(LoRAModelRunnerMixin):',
                  IMPORT + 'class GPUModelRunner(LoRAModelRunnerMixin):')
    source = once(source, '    @torch.inference_mode()\n    def execute_model(',
                  '    @execute_with_e2\n    @torch.inference_mode()\n    def execute_model(')
    def execution(s):
        s = once(s, '            self.block_tables.apply_staged_writes()\n',
                 '            self.block_tables.apply_staged_writes()\n' + UPDATED)
        return once(s, '        attn_metadata = None\n', BATCH + '        attn_metadata = None\n')
    source = replace_method(source, 'GPUModelRunner', 'execute_model', execution)
    source = replace_method(source, 'GPUModelRunner', '_qsa_copy_host_pages', lambda s: once(s,
        '        num_blocks = self.kv_cache_config.num_blocks\n',
        COW + '        num_blocks = self.kv_cache_config.num_blocks\n'))
    return replace_method(source, 'GPUModelRunner', 'shutdown', lambda s: once(s,
        '        torch.accelerator.synchronize()\n',
        '        torch.accelerator.synchronize()\n' + SHUTDOWN))


def qsa_staged(source):
    if '_flash_e2_lifecycle_ref' in source:
        raise ValueError('QSA already has an E2 staged hook')
    return once(source, '    retained, copied = None, 0\n', STAGED + '    retained, copied = None, 0\n')
