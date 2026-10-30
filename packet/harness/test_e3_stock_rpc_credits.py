"""Stock batch-queue bodies with real owned transport; CPU reply leaves only.

The scheduler/output and MQ payload leaves replace GPU model outputs. They
preserve the execute_model/sample_tokens future lifecycle and do not prove
model numerics, graph routing or native completion/publication.
"""
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
import types
import unittest
from typing import cast
import test_e3_executor_threads as fixture
import executor_hooks as eh


class StockRpcCreditTests(unittest.TestCase):
    def test_actual_stock_batch_queue_reclaims_discarded_execute_futures(self):
        e,t=fixture.install();self.addCleanup(t.close,1)
        e.output_rank=0;e.kv_output_aggregator=None;e.ec_output_aggregator=None
        env=types.SimpleNamespace(VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=2)
        for name in ('execute_model','sample_tokens'):
            setattr(e,name,types.MethodType(fixture.method(eh.EXECUTOR_PATH,'MultiprocExecutor',name,{'envs':env}),e))
        outputs=[]
        scheduler=types.SimpleNamespace(has_requests=lambda:True,
            schedule=lambda throttle:types.SimpleNamespace(total_num_scheduled_tokens=1,pending_structured_output_tokens=False),
            get_grammar_bitmask=lambda output:None,
            update_from_output=lambda output,value:outputs.append(value) or {})
        core=types.SimpleNamespace(batch_queue=deque(),batch_queue_size=2,scheduler=scheduler,
            model_executor=e,is_ec_consumer=True,is_pooling_model=False,
            _should_throttle_prefills=lambda:False,log_error_detail=lambda output:nullcontext(),
            capture_iteration_details=lambda output:nullcontext(None),
            _attach_iteration_details=lambda *args:None,_process_aborts_queue=lambda:None)
        step=fixture.method(eh.CORE_PATH,'EngineCore','step_with_batch_queue',{'cast':cast,'Future':Future})
        high=0
        for i in range(200):
            step(core);high=max(high,len(t._calls))
        self.assertGreater(len(outputs),100)
        self.assertLessEqual(high,4)
        for future,_,execute_future in core.batch_queue:
            future.result(2);execute_future.result(2)
        self.assertEqual(t._calls,{})
        self.assertEqual(e._flash_s5_stock_futures,{})
        self.assertEqual([x[0] for x in e.rpc_broadcast_mq.writes],['execute_model','sample_tokens']*200)
        self.assertEqual(len(e.response_mqs[0].readers),1)
        self.assertTrue(all(len(q.readers)<=1 for q in e.response_mqs))

    def test_native_ready_credit_is_not_claimed_by_stock_result(self):
        e,t=fixture.install();self.addCleanup(t.close,1)
        def native_send(seq):
            for q in e.response_mqs:q.q.put((e.SUCCESS,'native'))
        native=t.submit('native',native_send,timeout_s=2)
        first=e.collective_rpc('execute_model',non_block=True,timeout=2)
        second=e.collective_rpc('sample_tokens',non_block=True,timeout=2)
        second.result(2)
        self.assertEqual(set(t._calls),{native})
        self.assertTrue(first.done())
        self.assertEqual(first.result(2),[(r,'execute_model',()) for r in range(4)])
        self.assertEqual(set(t._calls),{native})
        self.assertEqual(t.take(native)[0],['native']*4)
        self.assertEqual(t._calls,{})

    def test_earlier_aggregate_failure_cached_without_failing_later_result(self):
        e,t=fixture.install();self.addCleanup(t.close,1)
        first=e.collective_rpc('first',non_block=True,timeout=2)
        seen=[]
        def refuse(value):
            seen.append(value)
            raise ValueError('earlier aggregate')
        first.aggregate=refuse
        second=e.collective_rpc('second',non_block=True,timeout=2)
        self.assertEqual(second.result(2),[(r,'second',()) for r in range(4)])
        for _ in range(2):
            with self.assertRaisesRegex(ValueError,'earlier aggregate'):first.result(2)
        self.assertEqual(len(seen),1)
        self.assertEqual(t._calls,{})

    def test_pending_earlier_reply_keeps_credit_and_timeout_is_retryable(self):
        e,t=fixture.install();self.addCleanup(t.close,1)
        first=e.collective_rpc('first',non_block=True,timeout=2)
        second=e.collective_rpc('second',non_block=True,timeout=2)
        # Receiver delivery remains real; zero budget can time out while its
        # rank replies are still pending in the local queue. A later retry
        # must claim each payload once, never silently discard its credit.
        try:second.result(0)
        except TimeoutError:pass
        second.result(2)
        first.result(2)
        self.assertEqual(t._calls,{})
        self.assertEqual(e._flash_s5_stock_futures,{})

    def test_two_stock_threads_share_ordered_claims_without_native_reader(self):
        e,t=fixture.install();self.addCleanup(t.close,1)
        def run():
            for _ in range(40):
                execute=e.collective_rpc('execute_model',non_block=True,timeout=2)
                sampled=e.collective_rpc('sample_tokens',non_block=True,timeout=2)
                sampled.result(2)
                self.assertTrue(execute.done())
        rows=[fixture.threaded(run),fixture.threaded(run)]
        errors=[]
        for row in rows:
            _,err=fixture.join(self,row,6);errors.extend(err)
        self.assertEqual(errors,[])
        self.assertEqual(t._calls,{})
        self.assertEqual(e._flash_s5_stock_futures,{})
        self.assertTrue(all(len(q.readers)==1 for q in e.response_mqs))

if __name__=='__main__':unittest.main()
