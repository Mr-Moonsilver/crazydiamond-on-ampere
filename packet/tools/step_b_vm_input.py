#!/usr/bin/env python3
"""Step B one-card VM input: the B5 scenario on an overlay, with layer groups.

Derived from tools/step_a_vm_input.py. Scenario: 98,312-token prefill, 8 cohort streams,
real worker and real graphs, reduced model, driven through NativeWorkerBridge by the probe,
dense QSA policy. Three deltas from Step A's scenario, because of what groups need to show:

* the model has 4 layers, linear x3 + full (B5's 3 layers give the same groups at sizes 2 and 4),
  which is the real model's repeating pattern: (0,1),(2,3*) at size 2 and (0,1,2,3*) at size 4;
* dummy weights are seeded, so runs compare, and each chunk's final hidden state is hashed after
  that chunk's completion poll, so the read cannot be the sync that makes it pass. B5's tokens
  alone show little: dummy weights give few distinct tokens;
* the groups come from the real engine_driver.layer_group_plan over PREPARE's dense_profiles, and
  each group is one command through OpenRequest(qsa_fused=True, layer_group=size).

    python3 tools/step_b_vm_input.py --overlay DIR --name step-b-x --group 2 \\
        --work DIR --vm-run --record DIR

Size 1 run twice is the determinism control: the hashes mean nothing if two size 1 runs differ.

What cannot run here: the engine driver, burst, auto-arm, direct replies, TP and PP
collectives (one rank, TP1Identity). See ``SCOPE``.
"""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'patches/s5_engine')]
from build_native30_held import sha, dump, manifest, verify  # noqa: E402
from native_readiness import digest, payload_identity, verify_finalized_package  # noqa: E402
from audit_native_runtime_imports import audit  # noqa: E402

BASE = Path('<hub-dir>/agents/flash-next/tmp/native29-overlap-codex-b5-input')
CPU_FIXTURES = Path('<hub-dir>/agents/flash-next/tmp/native29-overlap-codex-b5-cpu-fixtures')
OLD_NAME = 'native29-overlap-codex-b5'
OLD_LAUNCHER_PIN = '83ca0128798fe184b87c006149800994c825979ecedae24a72387b4d7a7085a4'
WRAPPERS = ('run_gpu.py', 'run_gpu_async.py', 'run_gpu_pressure.py')
ENV_ANCHOR = "'--env','VLLM_NO_USAGE_STATS=1'"
ENV_DELTA = (('FLASH_S5_NATIVE_QSA_POLICY', 'dense'),)
# Set inside the GPU composition only: the CPU-shape bootstrap builds a real
# EngineCore with no FLASH_S5_OVERLAP, whose startup correctly refuses auto-arm.
COMPOSE_ANCHOR = "    os.environ.update(native_env())\n    mark('compose-env-installed'"
COMPOSE_ENV = "    os.environ.update(native_env())\n    os.environ['FLASH_S5_AUTO_ARM'] = '1'\n    mark('compose-env-installed'"
PLUGINS = 'e3/native30_plugins/'
REFUSAL_RETURN = "return self._refuse(request, 'no-staged-plan', memoised=False, **receipt)"
SCOPE = dict(
    exercised=['composed worker/QSA/native code on the real device',
               'OpenRequest(qsa_fused=True, layer_group=size), PREPARE dense_profiles, one command per group',
               'per-chunk hash of the final hidden state from seeded dummy weights, read after the completion poll',
               'FLASH_S5_NATIVE_QSA_POLICY=dense: dense_dma route where a selection exceeds the 24-page arena',
               'worker memory receipts and auto-decode capability census (FLASH_S5_AUTO_ARM=1, GPU case only)'],
    not_exercised=['the engine driver plan rewrite and its TP obligation (the probe drives the bridge)',
                   'burst progress (NativeSegmentDriver.progress_in_wait, StockLocalFuture.result)',
                   'direct replies / executor RPC meta (no executor, no worker RPC transport)',
                   'AutoArmPolicy (no EngineCore driver)', 'plugins (not mounted in VM)',
                   'TP2/PP2 collectives (one rank, TP1Identity)'],
    scenario_differs_from_queue='probe keeps FLASH_S5_REPLAY_OBSERVER=1 and FLASH_S5_ENGINE_OFFERS=1 as B5; the queue has both 0',
    deltas_from_b5=['model: 4 layers, linear x3 + full, seeded dummy weights',
                    'run_gpu*: container name, launcher pin, dense and trace env',
                    'probe: observer archive and TP1Identity.broadcast (as native31), FLASH_S5_AUTO_ARM=1 in the GPU composition',
                    'run_cpu.py and superseded CPU fixture: staged-plan refusal line pin moved to the composed scheduler_hooks line'])
VM = '<service-user>@<encoder-host>'
SSH = ['ssh', '-F', '/dev/null', '-i', str(Path.home() / '.ssh/<ssh-key>'),
       '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=30', VM]
GPU_LOCK = '/home/<service-user>/.flashnext-vm-gpu.lock'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('anchor count differs: ' + old[:80])
    return text.replace(old, new)


def resolve_overlay(bundle, overlay):
    """Bundle-level manifest is verified when the bundle has one (frozen does)."""
    if overlay is not None:
        verify(overlay)
        return overlay, None
    overlay = bundle / 'native30_overlay'
    if (bundle / 'SHA256SUMS').is_file():
        verify(bundle)
        return overlay, sha(bundle / 'SHA256SUMS')
    verify(overlay)
    return overlay, None


LAYER_TYPES = "layer_types=['linear_attention', 'full_attention', 'linear_attention'],"
GROUP_TYPES = "layer_types=['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'],"
SEED = 20261101


def patch_probe(out, fused, group, ticks=None):
    """Same two output/identity patches as tools/build_native31_vm.py."""
    shutil.copyfile(ROOT / 'tools/native31_probe_archive.py', out / 'proof/native31_probe_archive.py')
    path = out / 'proof/e3_native29_overlap_probe.py'
    text = path.read_text()
    anchor = '            args.out.write_text(json.dumps(record, indent=2, sort_keys=True, default=str)'
    text = replace_once(text, anchor, '            from native31_probe_archive import archive_observer\n'
                        '            archive_observer(record,args.out.parent)\n' + anchor)
    text = replace_once(text, COMPOSE_ANCHOR, COMPOSE_ENV)
    text = replace_once(text, '\n\n# --- configuration',
                        '\n    def broadcast(self, value, src, stream=None):\n'
                        '        assert src==0 and stream is not None\n'
                        '        self.calls += 1\n        return None\n\n\n# --- configuration')
    text = replace_once(text, 'LAYERS = 3\n', 'LAYERS = 4\n')
    text = replace_once(text, LAYER_TYPES, GROUP_TYPES)
    text = replace_once(text, 'worker.load_model(load_dummy_weights=True)',
                        'import torch; torch.manual_seed(%d); torch.cuda.manual_seed_all(%d); '
                        'worker.load_model(load_dummy_weights=True)' % (SEED, SEED))
    # Stepped or fused, group size, plus the output tokens and hidden-state hashes for the comparison.
    text = replace_once(text, '\n\n# --- configuration',
                        '\n\nSTEP_A_FUSED = %r\nSTEP_B_SIZE = %d\n\n# --- configuration' % (fused, group))
    text = replace_once(text, 'opened = bridge.open(w.OpenRequest(record, (0,), record.state_epoch, 1))',
                        'opened = bridge.open(w.OpenRequest(record, (0,), record.state_epoch, 1, qsa_fused=STEP_A_FUSED, layer_group=STEP_B_SIZE))')
    text = replace_once(text, "state = dict(seq=0, bound=None, profile=None)",
                        "state = dict(seq=0, bound=None, profile=None, grants={}, groups={}, group_end=0)")
    text = replace_once(text, "        descriptor = w.SegmentDescriptor(record.transaction, kind, 0, layer_start=layer,\n"
                        "                                         layer_end=None if layer is None else layer + 1)\n",
                        "        descriptor = w.SegmentDescriptor(record.transaction, kind, 0, layer_start=layer,\n"
                        "                                         layer_end=None if layer is None else state['group_end'],\n"
                        "                                         layer_group=1 if layer is None else STEP_B_SIZE)\n")
    text = replace_once(text, "        ack = wait(attempt, 'submit')\n",
                        "        ack = wait(attempt, 'submit')\n"
                        "        if STEP_A_FUSED and kind is w.SegmentKind.PREPARE:\n"
                        "            state['grants'] = {row[0]: w.QsaProfileGrant(w.QsaProfileQuery(\n"
                        "                record.transaction, record.state_epoch, 0, row[0], cmd.op_seq, (0,)), row[1], fused=True)\n"
                        "                for row in ack.dense_profiles}\n"
                        "            from vllm.v1.flash_e3.engine_driver import layer_group_plan\n"
                        "            state['groups'] = dict(layer_group_plan(0, LAYERS, set(state['grants']), STEP_B_SIZE))\n"
                        "            mark('fused-grants', chunk=chunk_index, layers=sorted(state['grants']),\n"
                        "                 buckets=[len(row[1].bucket_waves) for row in ack.dense_profiles],\n"
                        "                 groups=sorted(state['groups'].items()))\n"
                        "        assert not (STEP_A_FUSED and cmd.profile is not None and ack.paused), 'a fused layer paused'\n")
    text = replace_once(text, "            state['profile'] = None\n            while command(kind_name, layer).paused:",
                        "            if layer < state['group_end']:\n"
                        "                continue\n"
                        "            state['group_end'] = state['groups'].get(layer, layer + 1)\n"
                        "            state['profile'] = state['grants'].get(state['group_end'] - 1)\n"
                        "            while command(kind_name, layer).paused:")
    if ticks is not None:
        # A matched decode interleave: the tick follows the boundaries of size `ticks`, whatever this run's size is.
        text = replace_once(text, "        decode_step('after-' + kind_name)\n",
                            "        if kind_name != 'layer' or state['group_end'] in state['tick_ends']:\n"
                            "            decode_step('after-' + kind_name)\n")
        text = replace_once(text, "            state['groups'] = dict(layer_group_plan(0, LAYERS, set(state['grants']), STEP_B_SIZE))\n",
                            "            state['groups'] = dict(layer_group_plan(0, LAYERS, set(state['grants']), STEP_B_SIZE))\n"
                            "            state['tick_ends'] = {end for _, end in layer_group_plan(0, LAYERS, set(state['grants']), %d)}\n" % ticks)
    # After the completion poll, so this read cannot be the sync that makes the chunk pass.
    text = replace_once(text, "    report = wait(complete, 'completion')\n",
                        "    report = wait(complete, 'completion')\n"
                        "    import torch\n"
                        "    hidden = stack.dispatcher.lane.hidden_states.detach().cpu().to(torch.float32)\n"
                        "    output_digest = dict(sha256=hashlib.sha256(hidden.numpy().tobytes()).hexdigest(),\n"
                        "                         shape=list(hidden.shape), abs_max=float(hidden.abs().max()),\n"
                        "                         nonfinite=int((~torch.isfinite(hidden)).sum()))\n")
    text = replace_once(text, "    return dict(chunk=chunk_index, start=record.transaction.chunk_start,",
                        "    return dict(output_digest=output_digest, chunk=chunk_index, start=record.transaction.chunk_start,")
    text = replace_once(text, "        return dict(verdict='PASS', gpu_execution=True, engine_constructed=False,",
                        "        return dict(step_b_size=STEP_B_SIZE, step_a_fused=STEP_A_FUSED, prefill_output_ids=list(prefill.output_token_ids),\n"
                        "                    cohort_output_ids=[list(r.output_token_ids) for r in cohort],\n"
                        "                    verdict='PASS', gpu_execution=True, engine_constructed=False,")
    path.write_text(text)


def patch_wrappers(out, candidate, launcher):
    delta = ','.join("'--env','%s=%s'" % row for row in ENV_DELTA)
    for name in WRAPPERS:
        path = out / name
        text = path.read_text().replace(OLD_NAME, candidate)
        # The native31-consensus-vm-d refusal: this pin predated the launcher.
        text = replace_once(text, OLD_LAUNCHER_PIN, launcher)
        text = replace_once(text, ENV_ANCHOR, ENV_ANCHOR + ',' + delta)
        path.write_text(text)


def declare_leaves(out):
    leaves = json.loads((out / 'fixture-leaves.json').read_text())
    next(row for row in leaves['leaves'] if row['name'] == 'TP1Identity')['equivalence'] += (
        ' The broadcast warm-up is the exact one-rank identity, returning None like real '
        'PyNccl.broadcast; no collective is qualified.')
    leaves['leaves'].append(dict(
        name='native31-observer-archive', replaces='primary probe embedding of full observer_end',
        equivalence='lossless XZ archive plus original verdict and first 32 failure summaries, checksum and archive path',
        cannot_prove='no placement/performance promotion; existing observer overflow remains visible',
        more_convenient=True, box_evidence=False))
    dump(out / 'fixture-leaves.json', leaves)
    dump(out / 'proof/native29_overlap_fixture_leaves.json', leaves)


def refusal_line(overlay):
    """Line of the staged-plan refusal; the CPU pin predates the refusal receipts."""
    rows = [i for i, line in enumerate((overlay / 'e3/scheduler_hooks.py').read_text().splitlines(), 1)
            if line.strip() == REFUSAL_RETURN]
    if len(rows) != 1:
        raise ValueError('staged-plan refusal return is not unique')
    return rows[0]


def supersede_cpu_fixtures(out, source):
    """New fixture copy: only pins that name composed-engine source lines move."""
    target = out.parent / (out.name.removesuffix('-input') + '-cpu-fixtures')
    shutil.copytree(source, target)
    for path in [target, *target.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    line = refusal_line(out / 'overlay')
    path = target / 'native29_scenario_cpu.py'
    path.write_text(replace_once(path.read_text(), "n['refusal']['line']==1474", "n['refusal']['line']==%d" % line))
    manifest(target)
    for path in [target, *target.rglob('*')]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    return target


def build(overlay_src, bundle_manifest, out, fused, group, ticks=None):
    candidate = out.name.removesuffix('-input')
    if not candidate.startswith('step-b-') or candidate == out.name or out.exists():
        raise ValueError('need a fresh --work/<step-b-*>-input')
    verify(BASE)
    shutil.copytree(BASE, out)
    for path in [out, *out.rglob('*')]:
        path.chmod(0o755 if path.is_dir() else 0o644)
    overlay = out / 'overlay'
    previous = verify(BASE / 'overlay')
    source_files = {p.relative_to(overlay_src).as_posix() for p in overlay_src.rglob('*') if p.is_file()}
    added = sorted(n for n in source_files - set(previous) - {'SHA256SUMS'} if not n.startswith(PLUGINS))
    plugin_files = sorted(n for n in source_files if n.startswith(PLUGINS))
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
    patch_probe(out, fused, group, ticks)
    patch_wrappers(out, candidate, launcher)
    path = out / 'run_cpu.py'
    path.write_text(replace_once(path.read_text(), "refusal['line']!=1474", "refusal['line']!=%d" % refusal_line(overlay)))
    provenance = json.loads((out / 'provenance.json').read_text())
    provenance.update(
        name=candidate, base_input=str(BASE), base_manifest_sha256=sha(BASE / 'SHA256SUMS'),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        overlay_source=str(overlay_src), bundle_manifest_sha256=bundle_manifest,
        engine_scope='composed bundle engine Python/native bytes; plugins not mounted in VM; same B5 scenario',
        group_size=group, seed=SEED,
        scope=SCOPE, env_delta=dict(ENV_DELTA, FLASH_S5_AUTO_ARM='1 (GPU composition only)'), launcher_sha256=launcher, overlay_added=added,
        overlay_plugin_files_omitted=len(plugin_files), gpu_run=False, box_action=False,
        builder_source_sha256=sha(Path(__file__)), built_unix=int(time.time()))
    for wrapper in provenance['wrappers'].values():
        wrapper['container'] = wrapper['container'].replace(OLD_NAME, candidate)
    dump(out / 'provenance.json', provenance)
    declare_leaves(out)
    manifest(out)
    verify(out)
    result = dict(verdict='PASS', path=str(out), manifest=sha(out / 'SHA256SUMS'), overlay_source=str(overlay_src),
                  bundle_manifest_sha256=bundle_manifest,
                  env_delta=dict(ENV_DELTA, FLASH_S5_AUTO_ARM='1 (GPU composition only)'), scope=SCOPE,
                  overlay_added=added, overlay_plugin_files_omitted=len(plugin_files),
                  launcher_sha256=launcher, gpu_execution=False)
    dump(out.parent / (out.name + '-build.json'), result)
    for path in [out, *out.rglob('*')]:
        path.chmod(0o555 if path.is_dir() else 0o444)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def cpu_selfrun(out, source_fixtures):
    fixtures = supersede_cpu_fixtures(out, source_fixtures)
    target = out.parent / (out.name.removesuffix('-input') + '-cpu')
    run = subprocess.run([sys.executable, '-B', str(out / 'run_cpu.py'), '--input', str(out), '--out', str(target),
                          '--cpu-fixtures', str(fixtures)])
    return run.returncode, target


def vm(command, **kwargs):
    return subprocess.run([*SSH, command], **kwargs)


def stage(out):
    """Tar over ssh, root-owned and read-only, manifest re-verified on the VM."""
    candidate = out.name.removesuffix('-input')
    manifest_sha = sha(out / 'SHA256SUMS')
    owned = candidate + '-' + manifest_sha[:12] + '-owner'
    remote = '/opt/flashnext/gpu/' + owned
    script = ('set -e; D=%s; test ! -e $D; tar xf - -C /opt/flashnext/gpu; chown -R root:root $D; chmod -R a-w $D; '
              'chmod 500 $D; cd $D; test "$(sha256sum SHA256SUMS | cut -d" " -f1)" = %s; '
              'sha256sum -c SHA256SUMS --quiet; echo VM_SUMS_OK' % (remote, manifest_sha))
    tar = subprocess.Popen(['tar', 'cf', '-', '-C', str(out.parent), out.name,
                            '--transform', 's|^' + out.name + '|' + owned + '|'], stdout=subprocess.PIPE)
    done = vm('sudo -n bash -c ' + shlex.quote(script), stdin=tar.stdout)
    tar.stdout.close()
    if tar.wait() or done.returncode:
        raise RuntimeError('stage failed rc=%s' % done.returncode)
    return remote


def vm_run(out, wrapper):
    candidate = out.name.removesuffix('-input')
    remote = stage(out)
    # The GPU is shared tonight: every GPU command waits on the common lock.
    run_dir = '/opt/flashnext/runs/' + candidate
    command = 'flock -w 5400 %s sudo -n python3 -B %s/%s --input %s --out %s' % (GPU_LOCK, remote, wrapper, remote, run_dir)
    done = vm(command)
    return done.returncode, run_dir


def fetch(run_dir, record):
    """Fetch the whole run dir: nothing is left behind or filtered."""
    record.mkdir(parents=True, exist_ok=False)
    parent, name = run_dir.rsplit('/', 1)
    pull = subprocess.Popen([*SSH, 'sudo -n tar cf - -C %s %s' % (parent, name)], stdout=subprocess.PIPE)
    untar = subprocess.run(['tar', 'xf', '-', '--strip-components=1', '--no-same-owner', '-C', str(record)], stdin=pull.stdout)
    pull.stdout.close()
    if pull.wait() or untar.returncode:
        raise RuntimeError('fetch failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bundle', type=Path, help='bundle holding native30_overlay/')
    source.add_argument('--overlay', type=Path, help='overlay directory directly')
    parser.add_argument('--name', required=True, help='fresh run name, step-b-*')
    parser.add_argument('--work', type=Path, required=True, help='hub directory for <name>-input and <name>-cpu')
    parser.add_argument('--cpu-fixtures', type=Path, default=CPU_FIXTURES)
    parser.add_argument('--cpu-selfrun', action='store_true')
    parser.add_argument('--vm-run', action='store_true')
    parser.add_argument('--group', type=int, choices=(1, 2, 4), default=1, help='layer group size; always fused')
    parser.add_argument('--ticks', type=int, choices=(2, 4), default=None,
                        help='one decode tick per this group size instead of one per command, so runs of different sizes '
                             'interleave decode the same way; a tick after every command otherwise')
    # Launch blocking off: the fused path's stream order must hold with real overlap.
    parser.add_argument('--wrapper', default='run_gpu_async.py', choices=WRAPPERS)
    parser.add_argument('--record', type=Path, help='fresh local directory for the fetched VM run')
    args = parser.parse_args()
    if args.vm_run and args.record is None:
        parser.error('--vm-run needs --record')
    out = args.work.resolve() / (args.name + '-input')
    overlay, bundle_manifest = resolve_overlay(args.bundle, args.overlay)
    build(overlay.resolve(), bundle_manifest, out, True, args.group, args.ticks)
    if args.cpu_selfrun:
        rc, target = cpu_selfrun(out, args.cpu_fixtures)
        print(json.dumps(dict(cpu_selfrun_rc=rc, cpu_record=str(target))), flush=True)
        if rc:
            return rc
    if args.vm_run:
        rc, run_dir = vm_run(out, args.wrapper)
        fetch(run_dir, args.record.resolve())
        print(json.dumps(dict(vm_rc=rc, run_dir=run_dir, record=str(args.record.resolve()))), flush=True)
        return rc
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
