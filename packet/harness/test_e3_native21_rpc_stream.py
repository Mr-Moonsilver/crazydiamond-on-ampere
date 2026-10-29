"""RPC-thread profile query against a device with a real stream guard shape.

The token leaf replaces NativeQsaDevice.validate_compute's CUDA current_stream
comparison. Unlike older operation fixtures, it refuses every compute-bound
method outside the lane scope. It cannot prove CUDA thread-local behavior.
"""

import ast
from contextlib import contextmanager
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'harness'), str(ROOT / 'patches/s5_engine')]

import qsa_operation as qo
import test_e3_native_schedule_profile as profile
import test_e3_sparse_history_dma as sparse


class StreamCheckedDevice(sparse.FakeSparseDevice):
    def __init__(self):
        super().__init__(auto_ready=False)
        self.owning_compute = None
        self.current_stream = 'worker-rpc-default'
        self.validations = 0

    def validate_compute(self):
        self.validations += 1
        if self.current_stream != self.owning_compute:
            raise qo.OperationError('operation called outside its owning compute stream')


class StreamCheckedStack(profile.HistoryStack):
    def __init__(self, harness, **kwargs):
        super().__init__(harness, **kwargs)
        self.device = StreamCheckedDevice()
        self.device.owning_compute = self.split.prefill_stream.cuda_stream

        @contextmanager
        def scope(stream):
            previous = self.device.current_stream
            self.device.current_stream = stream.cuda_stream
            try:
                yield stream
            finally:
                self.device.current_stream = previous

        self.torch.cuda.stream = scope


class CheckedProfile(profile.ScheduleProfile):
    stack_type = StreamCheckedStack

    def test_only_host_queries_bypass_compute_stream_check(self):
        source = (ROOT / 'patches/s5_engine/qsa_operation.py').read_text()
        tree = ast.parse(source)
        operation = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef)
                         and node.name == 'QsaPrefillOperation')
        opt_out = set()
        for method in operation.body:
            if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(method):
                if (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == '_check_identity'):
                    for keyword in call.keywords:
                        if keyword.arg == 'compute_required':
                            self.assertIsInstance(keyword.value, ast.Constant)
                            self.assertIs(keyword.value.value, False)
                            opt_out.add(method.name)
        self.assertEqual(opt_out, {'schedule_profile', 'submission_credit'})

    def test_rpc_query_and_host_credit_do_not_claim_compute_stream(self):
        world, driver, _ = self.boundary((True, True))
        for stack in world.stacks:
            self.assertEqual(stack.device.current_stream, 'worker-rpc-default')
            self.assertGreater(stack.device.validations, 0)
        prior = [stack.device.validations for stack in world.stacks]
        observations = self.query(world, driver)
        self.assertEqual([row.status for row in observations[:2]], ['ready', 'ready'])
        grant = self.agree(world, driver)
        for stack, before in zip(world.stacks, prior):
            # These are host event/plan reads on the RPC thread. They must
            # preserve all identity/epoch checks and submit no GPU work.
            self.assertEqual(stack.device.current_stream, 'worker-rpc-default')
            self.assertEqual(stack.device.validations, before)
            self.assertEqual(stack.operation.schedule_profile(), grant.shape)
            self.assertTrue(stack.operation.submission_credit().can_submit)
            self.assertEqual(stack.device.validations, before)
            # The same guarded device still refuses actual continuation work
            # or source mutation on that thread.
            with self.assertRaisesRegex(qo.OperationError,
                                        'outside its owning compute stream'):
                stack.operation.advance()
            with self.assertRaisesRegex(qo.OperationError,
                                        'outside its owning compute stream'):
                stack.operation.register_source('qsa.0', stack.source)
            self.assertEqual(stack.device.validations, before + 2)


if __name__ == '__main__':
    unittest.main()
