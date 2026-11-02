#!/usr/bin/env python3
"""Stage-1 handler anatomy one-card VM input: Step A's B5 scenario with the switch on, decode through WorkerProc.

Derived from tools/step_a_vm_input.py (its overlay composition, probe patches, wrappers, staging and fetch are reused
unchanged). Deltas, all in the probe:

* inside the GPU composition only (as FLASH_S5_AUTO_ARM): FLASH_S5_NATIVE_PROGRESS=burst and
  FLASH_S5_HANDLER_ANATOMY=1;
* a WorkerProc built without __init__ over the real worker (the overlay's generated vllm/v1/executor/
  multiproc_executor.py, so the real _execute_worker_rpc / handle_output / enqueue_output / async_output_busy_loop
  and the real executor_hooks); its response MQ is a list writer, its async output thread is the stock loop when the
  config schedules async;
* every decode RPC (execute_model, sample_tokens) and each chunk's release go through proc._execute_worker_rpc with
  the engine's progress meta (sequence, not direct), so worker_rpc_begin / returned / handle_output / enqueue_reply
  and the recorder run as on the box; the release reaches the same bridge.release through the generated
  Worker.flash_s5_release -> executor_hooks.worker_rpc;
* the handler and progress lines are captured in-process from the hooks' logger (and still printed to the container
  log), archived as stage1-anatomy-lines.jsonl.xz next to probe.json, and checked into probe.json's
  ``stage1_anatomy`` (parse, part sizes, rows, heads, counters, resolved device times);
* after the last decode step, one probe-only flush of the rows recorded since the last release.

    python3 tools/stage1_anatomy_vm_input.py --overlay DIR --name stage1-anatomy-x \\
        --work DIR --vm-run --record DIR

What it shows: the mechanics on the real vLLM worker and model runner (the wrapped names exist, events resolve by
query, lines are written and parse, observer cost is real, nothing deadlocks or syncs). Not: stage-specific numbers.
One rank, no PP, no TP: the PP receive/send and token-helper seams are absent by construction. See ``SCOPE``.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
import step_a_vm_input as sa  # noqa: E402
from build_native30_held import sha, dump, manifest, verify  # noqa: E402
from native_readiness import digest, payload_identity, verify_finalized_package  # noqa: E402
from audit_native_runtime_imports import audit  # noqa: E402

PREFIX = 'stage1-anatomy-'
COMPOSE_ENV = sa.COMPOSE_ENV
MARK = "\n    mark('compose-env-installed'"
assert COMPOSE_ENV.endswith(MARK)
ANATOMY_ENV = (COMPOSE_ENV[:-len(MARK)] + "\n    os.environ['FLASH_S5_NATIVE_PROGRESS'] = 'burst'"
               "\n    os.environ['FLASH_S5_HANDLER_ANATOMY'] = '1'" + MARK)
SCOPE = dict(
    exercised=['the real generated WorkerProc reply path over the real gpu_worker.Worker and model runner',
               'executor_hooks worker_rpc_begin / worker_rpc_returned / worker_handle_output / worker_enqueue_reply '
               'with progress meta on every decode RPC and every chunk release',
               'handler_anatomy install (calibration, wrapped instance attributes), seams, CUDA event pairs resolved '
               'by query, stream queries, observer self-timing, flush at flash_s5_release, line parts',
               'the async output thread (stock async_output_busy_loop) when the config schedules async',
               'Step A B5 scenario otherwise unchanged (stepped)'],
    not_exercised=['PP receive/send, handle waits and PP token helpers (one rank, PP1: absent by construction)',
                   'TP: the slower-rank comparison (one rank)', 'the engine driver, its sequences and its s rows',
                   'the MQ transport (the response MQ is a list writer)', 'box numbers: stage-specific shares'],
    record_says='mechanics only: proves the recorder on the real worker and runner; no stage-1 number comes from it')

PROBE_HELPERS = '''

# --- stage-1 handler anatomy: decode RPCs through the generated WorkerProc -------------------------------------

STAGE1 = dict(proc=None, seq=0, replies=None, lines=None, rpcs=0, releases=0)


def stage1_anatomy_proc(worker, cfg):
    """A WorkerProc without __init__ over the real worker; the reply MQ is a list writer."""
    import logging
    import queue
    import threading
    from vllm.v1.executor.multiproc_executor import WorkerProc
    from vllm.v1.flash_e3 import executor_hooks as eh
    assert callable(getattr(worker, 'flash_s5_release', None)), 'generated Worker.flash_s5_release absent'
    proc = WorkerProc.__new__(WorkerProc)
    proc.rank = 0
    proc.worker = worker
    proc.use_async_scheduling = bool(cfg.scheduler_config.async_scheduling)
    proc.async_output_queue = queue.Queue()
    replies = []
    proc.worker_response_mq = SimpleNamespace(enqueue=replies.append)
    lines = []

    class Capture(logging.Handler):
        def emit(self, record):
            text = record.getMessage()
            if text.startswith(('flash-s5 handler ', 'flash-s5 progress ')):
                lines.append(text)
    logger = eh._progress_logger()
    logger.addHandler(Capture(logging.INFO))
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    if proc.use_async_scheduling:
        threading.Thread(target=proc.async_output_busy_loop, name='stage1-anatomy-async-output',
                         daemon=True).start()
    STAGE1.update(proc=proc, replies=replies, lines=lines)
    mark('stage1-anatomy-proc', async_scheduling=proc.use_async_scheduling,
         executor_module=WorkerProc.__module__, logger=logger.name)
    return proc


def stage1_rpc(method, args, output_rank=0, timeout=WAIT_SECONDS):
    """One stock RPC as the engine sends it: progress meta, the generated reply path, the reply read back."""
    from vllm.v1.flash_e3 import native_measurement as nm
    proc, replies = STAGE1['proc'], STAGE1['replies']
    STAGE1['seq'] += 1
    STAGE1['rpcs'] += 1
    before = len(replies)
    kwargs = {nm.PROGRESS_META_KEY: nm.encode_progress_meta(STAGE1['seq'], False)}
    proc._execute_worker_rpc((method, args, kwargs, output_rank))
    end = time.monotonic() + timeout
    while len(replies) <= before:
        assert time.monotonic() < end, 'stage1 reply timeout: ' + method
        time.sleep(0.0002)
    status, value = replies[before]
    assert getattr(status, 'name', str(status)).endswith('SUCCESS'), (method, str(value)[:400])
    return value


def stage1_release(request):
    STAGE1['releases'] += 1
    return stage1_rpc('flash_s5_release', (request,), output_rank=None)


def stage1_final_flush():
    """Probe-only: the rows recorded since the last chunk release (the full-after steps)."""
    from vllm.v1.flash_e3 import executor_hooks as eh
    if eh._ANATOMY:
        eh._ANATOMY.flush(eh._progress_logger())


def stage1_anatomy_summary(out_dir):
    """Archive the captured lines and check them: parse, part sizes, rows, heads, counters, device times."""
    import lzma
    from vllm.v1.flash_e3 import executor_hooks as eh, handler_anatomy as ha
    lines = list(STAGE1['lines'] or ())
    with lzma.open(out_dir / 'stage1-anatomy-lines.jsonl.xz', 'wt') as handle:
        for text in lines:
            handle.write(json.dumps(text) + '\\n')
    handler = [t for t in lines if t.startswith(ha.LINE_PREFIX)]
    progress = [t for t in lines if t.startswith('flash-s5 progress ')]
    bodies, errors = [], []
    for text in handler:
        try:
            bodies.append(json.loads(text[len(ha.LINE_PREFIX):]))
        except ValueError:
            errors.append('unparsable')
    by_line = {}
    for body in bodies:
        by_line.setdefault(body['line'], []).append(body)
    incomplete = [n for n, parts in by_line.items()
                  if sorted(b['part'] for b in parts) != list(range(parts[0]['parts']))]
    main = [dict(zip(ha.M_FIELDS, row)) for b in bodies for row in b['rows'].get('m', [])]
    out = [dict(zip(ha.A_FIELDS, row)) for b in bodies for row in b['rows'].get('a', [])]
    heads = [b['head'] for b in bodies if 'head' in b]
    last = heads[-1] if heads else {}
    kinds = {}
    for row in main:
        kinds[row['kind']] = kinds.get(row['kind'], 0) + 1
    seams = {}
    for row in main:
        for cell in row['seams']:
            seams[ha.SEAMS[cell[0]]] = seams.get(ha.SEAMS[cell[0]], 0) + 1
    forwards = [r for r in main if r['kind'] == 'e' and r['dev_forward_ns'] is not None]
    walls = sorted(r['wall_ns'] for r in main if r['wall_ns'] is not None)
    observer = sorted(r['observer_ns'] for r in main if r['observer_ns'] is not None)
    return dict(
        handler_lines=len(handler), progress_lines=len(progress), flushes=len(by_line), parse_errors=errors,
        incomplete_flushes=incomplete, max_line_bytes=max((len(t.encode()) for t in handler), default=0),
        line_limit_bytes=ha.LINE_BYTES, main_rows=len(main), output_rows=len(out), kinds=kinds, seam_rows=seams,
        rows_with_device_forward=len(forwards), execute_rows=kinds.get('e', 0),
        output_rows_async_thread=sum(1 for r in out if r['thread'] == 0),
        output_rows_main_thread=sum(1 for r in out if r['thread'] == 1),
        wall_ns_p50=walls[len(walls) // 2] if walls else None,
        observer_ns_p50=observer[len(observer) // 2] if observer else None,
        rpcs=STAGE1['rpcs'], releases=STAGE1['releases'],
        identity=last.get('identity'), calibration=last.get('calibration'), counters=last.get('counters'),
        installed=last.get('installed'), missing=last.get('missing'),
        recorder_installed=bool(eh._ANATOMY), scope=%r)
''' % (SCOPE,)


def patch_anatomy(out):
    """On top of Step A's stepped probe: env, the proc, decode and release through it, the summary."""
    path = out / 'proof/e3_native29_overlap_probe.py'
    text = path.read_text()
    text = sa.replace_once(text, COMPOSE_ENV, ANATOMY_ENV)
    text = sa.replace_once(text, '\n\n# --- configuration', PROBE_HELPERS + '\n\n# --- configuration')
    text = sa.replace_once(text, "        mark('cohort-added', streams=len(cohort))\n",
                           "        mark('cohort-added', streams=len(cohort))\n"
                           "        stage1_anatomy_proc(worker, cfg)\n")
    text = sa.replace_once(text, '            result = worker.execute_model(output)\n'
                                 '            if result is None:\n'
                                 '                result = worker.sample_tokens(None)\n',
                           "            result = stage1_rpc('execute_model', (output,))\n"
                           '            if result is None:\n'
                           "                result = stage1_rpc('sample_tokens', (None,))\n")
    text = sa.replace_once(text, '        ack = bridge.release(request)\n', '        ack = stage1_release(request)\n')
    text = sa.replace_once(text, '        torch.cuda.synchronize()\n        memory = memory_summary(torch)\n',
                           '        stage1_final_flush()\n'
                           '        torch.cuda.synchronize()\n        memory = memory_summary(torch)\n')
    text = sa.replace_once(text, '        return dict(step_a_fused=STEP_A_FUSED,',
                           '        return dict(stage1_anatomy=stage1_anatomy_summary(args.out.parent), '
                           'step_a_fused=STEP_A_FUSED,')
    path.write_text(text)


def build(overlay_src, bundle_manifest, out):
    """step_a_vm_input.build with the stage1-anatomy name and the anatomy probe deltas; stepped (no --fused)."""
    candidate = out.name.removesuffix('-input')
    if not candidate.startswith(PREFIX) or candidate == out.name or out.exists():
        raise ValueError('need a fresh --work/<%s*>-input' % PREFIX)
    verify(sa.BASE)
    shutil.copytree(sa.BASE, out)
    for path in [out, *out.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    overlay = out / 'overlay'
    previous = verify(sa.BASE / 'overlay')
    source_files = {p.relative_to(overlay_src).as_posix() for p in overlay_src.rglob('*') if p.is_file()}
    added = sorted(n for n in source_files - set(previous) - {'SHA256SUMS'} if not n.startswith(sa.PLUGINS))
    plugin_files = sorted(n for n in source_files if n.startswith(sa.PLUGINS))
    for name in [*previous, *added]:
        (overlay / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(overlay_src / name, overlay / name)
    gen = json.loads((overlay / 'S5-OVERLAP-GENERATION.json').read_text())
    ready = json.loads((overlay / 'e3/native-overlap-readiness.json').read_text())
    gen['package_sha256'] = ready['package_sha256'] = payload_identity(overlay)
    ready['receipt_sha256'] = digest({k: v for k, v in ready.items() if k != 'receipt_sha256'})
    dump(overlay / 'S5-OVERLAP-GENERATION.json', gen)
    dump(overlay / 'e3/native-overlap-readiness.json', ready)
    manifest(overlay)
    closure = json.loads((overlay / 'e3/native-implementation-closure.json').read_text())
    verify_finalized_package(overlay, closure['generated_source_shas'])
    launcher = sha(overlay / 'run-flash-s5-test.sh')
    graph = audit(overlay, ROOT, launcher_sha256=launcher)
    if graph['verdict'] != 'PASS':
        raise ValueError('import graph: ' + str(graph['runtime_ownership_errors']))
    dump(out / 'proof/import-graph.json', graph)
    sa.patch_probe(out, False)
    patch_anatomy(out)
    sa.patch_wrappers(out, candidate, launcher)
    path = out / 'run_cpu.py'
    path.write_text(sa.replace_once(path.read_text(), "refusal['line']!=1474",
                                    "refusal['line']!=%d" % sa.refusal_line(overlay)))
    provenance = json.loads((out / 'provenance.json').read_text())
    env_delta = dict(sa.ENV_DELTA, FLASH_S5_AUTO_ARM='1 (GPU composition only)',
                     FLASH_S5_NATIVE_PROGRESS='burst (GPU composition only)',
                     FLASH_S5_HANDLER_ANATOMY='1 (GPU composition only)')
    provenance.update(
        name=candidate, base_input=str(sa.BASE), base_manifest_sha256=sha(sa.BASE / 'SHA256SUMS'),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        overlay_source=str(overlay_src), bundle_manifest_sha256=bundle_manifest,
        engine_scope='composed engine Python/native bytes; plugins not mounted in VM; Step A B5 scenario, stepped',
        scope=SCOPE, step_a_scope=sa.SCOPE, env_delta=env_delta, launcher_sha256=launcher, overlay_added=added,
        overlay_plugin_files_omitted=len(plugin_files), gpu_run=False, box_action=False,
        builder_source_sha256=sha(Path(__file__)), step_a_builder_sha256=sha(Path(sa.__file__)),
        built_unix=int(time.time()))
    for wrapper in provenance['wrappers'].values():
        wrapper['container'] = wrapper['container'].replace(sa.OLD_NAME, candidate)
    dump(out / 'provenance.json', provenance)
    sa.declare_leaves(out)
    leaves = json.loads((out / 'fixture-leaves.json').read_text())
    leaves['leaves'].append(dict(
        name='stage1-anatomy-list-writer', replaces='WorkerProc.worker_response_mq (shared-memory MessageQueue)',
        equivalence='enqueue(result) appends the exact (status, output) tuple the stock enqueue_output builds; the '
                    'generated reply path, its reply lock and the hooks run unchanged',
        cannot_prove='MQ write cost and backpressure (the box run measures mq_write_ns on the real queue)',
        more_convenient=True, box_evidence=False))
    leaves['leaves'].append(dict(
        name='stage1-anatomy-workerproc-without-init', replaces='WorkerProc.__init__ (process spawn, MQs, device init)',
        equivalence='the instance carries rank, worker, use_async_scheduling, async_output_queue and the writer: the '
                    'attributes the four reply-path methods read; the worker is the real initialised Worker',
        cannot_prove='executor process topology, RPC broadcast dequeue', more_convenient=True, box_evidence=False))
    dump(out / 'fixture-leaves.json', leaves)
    dump(out / 'proof/native29_overlap_fixture_leaves.json', leaves)
    manifest(out)
    verify(out)
    result = dict(verdict='PASS', path=str(out), manifest=sha(out / 'SHA256SUMS'), overlay_source=str(overlay_src),
                  bundle_manifest_sha256=bundle_manifest, env_delta=env_delta, scope=SCOPE, overlay_added=added,
                  overlay_plugin_files_omitted=len(plugin_files), launcher_sha256=launcher, gpu_execution=False)
    dump(out.parent / (out.name + '-build.json'), result)
    for path in [out, *out.rglob('*')]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--overlay', type=Path, required=True, help='overlay directory')
    parser.add_argument('--name', required=True, help='fresh run name, ' + PREFIX + '*')
    parser.add_argument('--work', type=Path, required=True, help='hub directory for <name>-input')
    parser.add_argument('--vm-run', action='store_true')
    parser.add_argument('--wrapper', default='run_gpu_async.py', choices=sa.WRAPPERS)
    parser.add_argument('--record', type=Path, help='fresh local directory for the fetched VM run')
    args = parser.parse_args()
    if args.vm_run and args.record is None:
        parser.error('--vm-run needs --record')
    out = args.work.resolve() / (args.name + '-input')
    overlay, bundle_manifest = sa.resolve_overlay(None, args.overlay)
    build(overlay.resolve(), bundle_manifest, out)
    if args.vm_run:
        rc, run_dir = sa.vm_run(out, args.wrapper)
        sa.fetch(run_dir, args.record.resolve())
        print(json.dumps(dict(vm_rc=rc, run_dir=run_dir, record=str(args.record.resolve()))), flush=True)
        return rc
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
