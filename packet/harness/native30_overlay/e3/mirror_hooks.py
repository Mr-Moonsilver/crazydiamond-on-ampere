"""Source transforms for E2 metadata hooks; NOT a deployable overlay generator.

The full step-4 generator must install the worker/fence adapter and reject
unsupported configs before enabling FLASH_E2_ARENA. These transforms alone do
not implement DMA or establish GPU completion. Core refcounts are untouched.
"""
import ast


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('source anchor changed: ' + before[:80])
    return text.replace(before, after, 1)


def replace_method(source, cls, name, transform):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = source.splitlines(keepends=True)
    original = ''.join(lines[method.lineno - 1:method.end_lineno])
    return ''.join(lines[:method.lineno - 1]) + transform(original) + ''.join(lines[method.end_lineno:])


def block_pool(source):
    source = replace_method(source, 'BlockPool', '__init__', lambda s: s + '''
        self._flash_e2_journal = None
        if __import__("os").environ.get("FLASH_E2_ARENA", "0") == "1":
            from vllm.v1.flash_e2.block_mirror import AllocationJournal
            self._flash_e2_journal = AllocationJournal(1, num_gpu_blocks)
''')
    return replace_method(source, 'BlockPool', 'get_new_blocks', lambda s: once(s,
        '        return ret\n',
        '        if self._flash_e2_journal is not None:\n'
        '            self._flash_e2_journal.allocated(b.block_id for b in ret)\n'
        '        return ret\n'))


def scheduler_output(source):
    return once(source,
        '    num_spec_tokens_to_schedule: int = 0\n',
        '    num_spec_tokens_to_schedule: int = 0\n\n'
        '    # E2 immutable CPU mirror frame; no tensors, defaults to stock None.\n'
        '    flash_e2_frame: object | None = None\n')


def scheduler(source):
    # AFTER the virtual _update_after_schedule (AsyncScheduler overrides it).
    # No allocator decision, step/parity counter or free route is modified.
    return replace_method(source, 'Scheduler', 'schedule', lambda s: once(s,
        '        return scheduler_output\n',
        '        _e2_journal = getattr(self.kv_cache_manager.block_pool, "_flash_e2_journal", None)\n'
        '        if _e2_journal is not None:\n'
        '            scheduler_output.flash_e2_frame = _e2_journal.frame(\n'
        '                scheduler_output, self.kv_cache_manager.get_block_ids)\n'
        '        return scheduler_output\n'))


def block_tables(source):
    source = replace_method(source, 'BlockTables', '__init__', lambda s: s + '''
        self._flash_e2_tables = None
        if __import__("os").environ.get("FLASH_E2_ARENA", "0") == "1":
            from vllm.v1.flash_e2.block_mirror import CpuTables
            if cp_size != 1:
                raise RuntimeError("E2 arena does not support context parallelism")
            self._flash_e2_tables = CpuTables(
                self.blocks_per_kv_block,
                max(t.gpu.shape[1] for t in self.block_tables))
''')
    return replace_method(source, 'BlockTables', 'append_block_ids', lambda s: s + '''
        if self._flash_e2_tables is not None:
            self._flash_e2_tables.append(req_index, new_block_ids, overwrite)
''')
