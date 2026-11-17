"""Narrow, reversible fit-only seams over the frozen cxfx42 source files."""
import ast
import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
EXPORTED = next((p for p in (_ROOT/'fit_parent',_ROOT/'harness/fit_parent') if p.is_dir()),_ROOT/'fit_parent')
PARENT = EXPORTED if EXPORTED.is_dir() else Path('<hub-dir>/cargo-target/flash-next/vision-native-mm-20261116/serve-vision-fe-20261116a-build-7/harness/native30_overlay')


def once(text,old,new):
    if text.count(old)!=1:raise ValueError('fit-source-anchor:'+old[:80])
    return text.replace(old,new,1)


def transform(rel,text):
    if ('_flash_fit' in text or 'def flash_s5_fit_memory(' in text
            or 'flash-s5 fit-memory ' in text or 'not _fit_floor.before_arm()' in text):
        raise ValueError('fit-source-already-instrumented')
    original=text
    if rel=='e3_sources/core.py':
        anchor='import vllm.v1.flash_e3.executor_hooks as _flash_s5_hooks\n'
        text=once(text,anchor,anchor+'import vllm.v1.flash_e3.fit_memory as _flash_fit\n')
        anchor='        _flash_s5_hooks.after_engine_startup(self)\n'
        text=once(text,anchor,anchor+'        _flash_fit.install(self)\n')
        anchor='    def flash_s5_native_control(self, command):\n'
        text=once(text,anchor,'    def flash_s5_fit_control(self, command):\n        return _flash_fit.control(self, command)\n\n'+anchor)
        anchor='        _flash_s5_tick = _flash_s5_hooks.engine_idle_tick(self)\n'
        if text.count(anchor)!=2:raise ValueError('fit-two-core-step-seams')
        text=text.replace(anchor,'        _fit_result = _flash_fit.before_step(self)\n        if _fit_result is not None:\n            return _fit_result\n'+anchor)
        anchor='        future = self.model_executor.execute_model(scheduler_output, non_block=True)\n'
        text=once(text,anchor,'        _flash_fit.note_schedule(self, scheduler_output)\n'+anchor)
        anchor='                exec_future = self.model_executor.execute_model(\n'
        text=once(text,anchor,'                _flash_fit.note_schedule(self, scheduler_output)\n'+anchor)
        anchor='            or _flash_s5_hooks.native_work_pending(self)\n'
        text=once(text,anchor,anchor+'            or _flash_fit.pending(self)\n')
    elif rel=='e3_sources/worker.py':
        anchor='    def flash_s5_native_overlap(self, command):\n'
        text=once(text,anchor,'    def flash_s5_fit_memory(self, command):\n        from vllm.v1.flash_e3.fit_memory import worker_measure\n        return worker_measure(self, command)\n\n'+anchor)
    elif rel=='e3_sources/api_routers.py':
        anchor='    _flash_s5_probe_router(args, app)\n'
        text=once(text,anchor,anchor+'    from vllm.v1.flash_e3.fit_memory import install_api as _flash_fit_api\n    _flash_fit_api(args, app)\n')
    elif rel=='e3/native_measurement.py':
        text=once(text,'RECEIPT_JSON_PREFIXES = (',"RECEIPT_JSON_PREFIXES = ('flash-s5 fit-memory ', 'flash-s5 graph-events ', 'flash-s5 startup-vram ', ")
    elif rel=='e3/native_auto_arm.py':
        # Both FULL->SPLIT and kept-SPLIT O3 entries must join the permit first.
        # Transfer the SPLIT stop obligation first; a low-memory reply must use
        # the policy's typed return, not try a forbidden manual leave_split.
        for indent in ('        ','            '):
            anchor=indent+'self._internal = True\n'+indent+'try:\n'+indent+'    d._arm_admission(PHASE)\n'
            gate=(indent+"_fit_floor = getattr(d, 'graph_event_floor', None)\n"
                  +indent+'if _fit_floor is not None and not _fit_floor.before_arm():\n'
                  +indent+'    return False\n')
            text=once(text,anchor,gate+anchor)
    elif rel=='e3/scheduler_hooks.py':
        anchor='        if scoped:\n            metadata.begin(scheduler, request)\n'
        text=once(text,anchor,"        _fit_floor = getattr(owner, 'graph_event_floor', None)\n"
                  "        if _fit_floor is not None and not _fit_floor.before((request.request_id, start, chunk_end)):\n"
                  '            return None\n'+anchor)
        anchor='''        if cost is not None:
            with cost.span('prepare_ns'):
                return self._admit_observed(scheduler)
        return self._admit_observed(scheduler)
'''
        replacement='''        if cost is not None:
            with cost.span('prepare_ns'):
                record = self._admit_observed(scheduler)
        else:
            record = self._admit_observed(scheduler)
        _fit_floor = getattr(owner, 'graph_event_floor', None)
        if _fit_floor is not None and record is not None:
            _fit_floor.admitted(record)
        return record
'''
        text=once(text,anchor,replacement)
        anchor='        self.admitted += 1\n'
        text=once(text,anchor,anchor+
            "        # Native's first admission bypasses stock's initial PrefillStats.set.\n"
            "        _stats = getattr(request, 'prefill_stats', None)\n"
            "        if request.num_computed_tokens == 0 and _stats is not None and request.num_preemptions <= 0:\n"
            "            _stats.set(num_prompt_tokens=request.num_prompt_tokens,\n"
            "                       num_local_cached_tokens=num_new_computed, num_external_cached_tokens=0)\n")
    else:raise ValueError('unlisted-fit-source-file')
    ast.parse(text)
    return text,dict(path=rel,before=hashlib.sha256(original.encode()).hexdigest(),after=hashlib.sha256(text.encode()).hexdigest())


def packaged(source):
    for peer in ('graph_event_floor','startup_vram_budget','native_measurement','decode_graph_events'):
        source=source.replace('from '+peer+' import ','from .'+peer+' import ')
    ast.parse(source)
    return source


def stock_transform(rel,text):
    """Only fit observer seams; no native model/runner/attention transformations."""
    if '_flash_fit' in text or 'flash_s5_fit_memory' in text:
        raise ValueError('fit-stock-source-already-instrumented')
    original=text
    if rel=='core.py':
        anchor='logger = init_logger(__name__)\n'
        text=once(text,anchor,anchor+'\nimport vllm.v1.flash_e3.fit_memory as _flash_fit\n')
        anchor='        self.is_ec_consumer = (\n'
        text=once(text,anchor,'        _flash_fit.install(self)\n'+anchor)
        tree=ast.parse(text)
        owner=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EngineCore')
        inserts=[]
        for name in ('step','step_with_batch_queue'):
            method=next(n for n in owner.body if isinstance(n,ast.FunctionDef) and n.name==name)
            doc=method.body[0]
            if not isinstance(doc,ast.Expr) or not isinstance(doc.value,ast.Constant):
                raise ValueError('stock-step-doc-anchor')
            inserts.append(doc.end_lineno)
        lines=text.splitlines(True)
        for line in sorted(inserts,reverse=True):
            lines.insert(line,'        _fit_result = _flash_fit.before_step(self)\n        if _fit_result is not None:\n            return _fit_result\n')
        text=''.join(lines)
        anchor='    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:\n'
        text=once(text,anchor,'    def flash_s5_fit_control(self, command):\n        return _flash_fit.control(self, command)\n\n'+anchor)
        anchor='        future = self.model_executor.execute_model(scheduler_output, non_block=True)\n'
        text=once(text,anchor,'        _flash_fit.note_schedule(self, scheduler_output)\n'+anchor)
        anchor='                exec_future = self.model_executor.execute_model(\n'
        text=once(text,anchor,'                _flash_fit.note_schedule(self, scheduler_output)\n'+anchor)
        anchor='            or bool(self.batch_queue)\n'
        text=once(text,anchor,anchor+'            or _flash_fit.pending(self)\n')
    elif rel=='worker.py':
        anchor='    def shutdown(self) -> None:\n'
        text=once(text,anchor,'    def flash_s5_fit_memory(self, command):\n        from vllm.v1.flash_e3.fit_memory import worker_measure\n        return worker_measure(self, command)\n\n'+anchor)
    elif rel=='api_routers.py':
        anchor='    register_vllm_serve_api_routers(app)\n'
        text=once(text,anchor,anchor+'    from vllm.v1.flash_e3.fit_memory import install_api as _flash_fit_api\n    _flash_fit_api(args, app)\n')
    else:raise ValueError('fit-stock-source-not-allowlisted')
    ast.parse(text)
    return text,dict(path=rel,before=hashlib.sha256(original.encode()).hexdigest(),after=hashlib.sha256(text.encode()).hexdigest())
