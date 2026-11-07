"""Image-only: actual pinned state constructors and real CPU torch storage, no CUDA."""
import ast
import functools
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np
import torch

from test_native_uva_retirement import Device, ROOT
from native_uva import NativeUvaSlots


def pinned_classes():
    class CpuUvaBuffer:
        def __init__(self, size, dtype):
            self.cpu = torch.zeros(size, dtype=dtype, device='cpu')
            self.np = self.cpu.numpy()
            self.uva = self.cpu

    namespace = dict(torch=torch, np=np, UvaBuffer=CpuUvaBuffer,
                     _DEFAULT_MAX_CONCURRENCY=3, partial=functools.partial)

    def load(path, names):
        source = ROOT / 'reference/vllm-src/v1/worker/gpu' / path
        tree = ast.parse(source.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
        assert len(nodes) == len(names)
        namespace['__name__'] = 'vllm.v1.worker.gpu.' + path[:-3].replace('/', '.')
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
                            type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)

    load('buffer_utils.py', ('UvaBufferPool', 'UvaBackedTensor', 'StagedWriteTensor', 'FusedStagedWriter'))
    load('states.py', ('RequestState',))
    load('block_table.py', ('BlockTables',))
    load('mm/rope.py', ('RopeState',))
    load('model_states/prompt_embeds.py', ('PromptEmbedsState',))
    return NS(**namespace)


class PinnedInventory(unittest.TestCase):
    def test_real_pool_asts_match_the_pinned_image_installation(self):
        base = Path(importlib.util.find_spec('vllm').origin).parent
        installed = ast.parse((base / 'v1/worker/gpu/buffer_utils.py').read_text())
        reference = ast.parse((ROOT / 'reference/vllm-src/v1/worker/gpu/buffer_utils.py').read_text())
        for name in ('UvaBufferPool', 'UvaBackedTensor', 'StagedWriteTensor', 'FusedStagedWriter'):
            get = lambda tree: next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            self.assertEqual(ast.dump(get(installed)), ast.dump(get(reference)), name)

    def test_real_constructors_classify_every_pinned_pool_and_report_production_shape_cost(self):
        self.assertFalse(torch.cuda.is_initialized())
        b = pinned_classes()
        for groups in (3, 39):
            with self.subTest(groups=groups):
                runner = NS(req_states=b.RequestState(10, 64, 2048, 0, 32, torch.device('cpu')),
                            block_tables=b.BlockTables([4048] * groups, 10, 2048, [16] * groups,
                                                       torch.device('cpu'), [4048] * groups),
                            model_state=NS(rope_state=None, prompt_embeds_state=None))
                d = Device()
                owner = NativeUvaSlots(runner, d.torch, b)
                inventory = owner.inventory()
                self.assertEqual(inventory['pools'], 17 + 3 * groups)
                self.assertEqual(inventory['host_bytes'], 1332 + 960 * groups)
                self.assertEqual(inventory['gpu_bytes'], 0)
                # Actual 262144 model-length bound, only the small descriptor-content pool grows.
                content = owner.pools[id(runner.req_states.all_token_ids.write_contents)]
                for _ in range(3):
                    with d.scope(d.native), owner.scope():
                        content.copy_to_uva(list(range(262144)))
                self.assertEqual(owner.inventory()['host_bytes'], inventory['host_bytes'] + 3 * (1048576 - 4))
                # Optional pinned branches must classify too, even though the production text model disables them.
                runner.model_state.rope_state = b.RopeState(3, 10, 2048, 64, torch.device('cpu'))
                runner.model_state.prompt_embeds_state = b.PromptEmbedsState(10, 8, torch.float32, torch.device('cpu'))
                rich_runner = NS(req_states=runner.req_states, block_tables=runner.block_tables,
                                 model_state=runner.model_state)
                richer = NativeUvaSlots(rich_runner, d.torch, b).inventory()
                self.assertEqual(richer['host_bytes'], inventory['host_bytes'] + 1812)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
