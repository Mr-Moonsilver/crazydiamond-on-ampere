#!/usr/bin/env python3
"""Compose the opt-in S5 overlap overlay from a reviewed E2 package.

Offline generation only. This builds NEW files next to the inert E2 payload; it
never installs, mounts, starts or arms an engine. Inputs are full-file pinned:
the E2 package must carry the exact reviewed generation record and manifest, and
the three engine sources come from the approved reference tree. Source drift or
a duplicate/missing anchor aborts before an output directory is created.

Composition order:
  1. E2's serial overlay is the base (left byte-identical where unchanged).
  2. A's lane hooks transform the E2 runner.
  3. B's reviewed graph hooks compose after A; C's startup builder is inserted between E2's install and A's init_lane.
  4. C's scheduler/block-pool leases and executor/core/worker hooks are applied.
  5. D2's bounded QSA operation call sites are generated into the E2 QSA layer,
     the E2 attention module and the S5-composed runner (full-file pinned).
  6. Optional additional pinned transforms can be applied last through this interface:
     a module with ``SOURCE_PINS`` (relative path -> sha256 of the composed
     input it reviewed) and ``generate(files) -> {relative path: source}``.
     B may only replace paths it pinned. No B module is guessed or imported by
     default; absence fails closed when ``--graph`` is requested.

The output manifest records source/output hashes, installed settings and the
allocation budget. ``FLASH_S5_OVERLAP`` stays 0; enabling requires the owner's
native factory and GPU gates. The generated launcher keeps its S5 branch off by
default and requires ``S5=1`` to mount anything.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import async_handoff
import decode_probe
import executor_hooks
import engine_offers
import graph_hooks
import green_context
import lane_hooks
import model_hooks
import operation_source
import qsa_driver
import scheduler_hooks
import worker_hooks
import probe_control_hooks


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

E3_MODULES = ('wire_v1', 'e3_types', 'lane_state', 'reply_transport', 'native_backend',
              'lane_hooks', 'async_handoff', 'native_resources', 'engine_driver',
              'executor_hooks', 'scheduler_hooks', 's5_startup',
              'admission_metadata', 'worker_bridge',
              'graph_router', 'graph_hooks', 'qsa_operation', 'sparse_history_dma',
              'native_source', 'row1_oracle',
              # D4 worker/native side (Claude): contexts, split bank, worker seams,
              # lane forward/QSA driver, communicators, startup, chunk, dispatcher.
              'green_context', 'capture_streams', 'split_bank', 'worker_hooks', 'model_hooks', 'lane_private', 'qsa_driver', 'lane_comm', 'native_startup', 'lane_chunk', 'lane_dispatch', 'operation_source', 'decode_probe', 'probe_control', 'probe_control_hooks', 'probe_session', 'launch_diagnostic', 'route_invariant', 'teacher_decode', 'replay_observer', 'observer_runtime', 'activity_accounting', 'engine_offers',
              # Native control over the pinned EngineCore utility channel.
              'native_overlap_metrics', 'native_overlap_worker',
              'native_overlap_control', 'native_overlap_session',
              'native_prefill_offers', 'native_placement', 'native_measurement')
NATIVE_FILES = ('native/flash_s5_graph_bytes.cpp', 'native/flash_s5_green.cu', 'native/flash_s5_replay_observer.cpp',
                'native/build_q17_helpers.sh')
MODEL_PATH = 'models/qwen4_exp/nvidia/model.py'
CUDAGRAPH_PATH = 'v1/worker/gpu/cudagraph_utils.py'
MODEL_DEST = '/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py'
BUILD_GREEN_HELPER = '''#!/bin/bash
# Owner-only CPU compiler container; new output, no GPU or engine access.
# Same sm_86/shared-cudart recipe qualified by q17a. Container is retained.
# Final overlay manifest and declared FLASH_S5_GREEN_LIB_SHA256 must cover
# the new library BEFORE source declaration or any test-engine start.
set -eu
img=$1; dir=$(cd "$2" && pwd)
[[ "$img" == sha256:ad067db5d64ed30b52f5aa9ae88d7645e71ff1842c876d7db116fed617a5cd0b ]]
[[ ! -e "$dir/flash_s5_green.so" && ! -L "$dir/flash_s5_green.so" ]]
name="e3-green-build-$(date -u +%Y%m%dT%H%M%SZ)-$$"
# Inner timeout terminates compiler before the client bound.
timeout --signal=TERM --kill-after=20s 300s docker run --name "$name" --network none \
  -v "$dir:/native" --entrypoint timeout "$img" --signal=TERM --kill-after=10s 240s \
  nvcc -O2 -std=c++17 -shared -Xcompiler=-fPIC \
  --generate-code=arch=compute_86,code=sm_86 --cudart=shared -lcuda \
  /native/flash_s5_green.cu -o /native/flash_s5_green.so
sha256sum "$dir/flash_s5_green.so"
echo "retained build container: $name"
'''


# Exact reviewed E2-composed inputs. Any other bytes are refused.
E2_RUNNER_SHA256 = lane_hooks.E2_STARTUP_SHA256
E2_SCHEDULER_SHA256 = scheduler_hooks.SCHEDULER_E2_SHA256
E2_BLOCK_POOL_SHA256 = scheduler_hooks.BLOCK_POOL_E2_SHA256
E2_INPUTS = {
    'model_runner_v2.py': E2_RUNNER_SHA256,
    'core/scheduler.py': E2_SCHEDULER_SHA256,
    'core/block_pool.py': E2_BLOCK_POOL_SHA256,
}
# E2 QSA worker sources the S5 operation call sites are generated into.
E2_WORKER_INPUTS = {
    'qsa.py': operation_source.E2_QSA_SHA256,
    'ops_qsa.py': operation_source.E2_OPS_SHA256,
}
import capture_streams
REFERENCE_INPUTS = dict(executor_hooks.PINS)
REFERENCE_INPUTS.update(engine_offers.PINS)
REFERENCE_INPUTS.update(capture_streams.PINS)
REFERENCE_INPUTS.update(worker_hooks.PINS)
REFERENCE_INPUTS.update(worker_hooks.CAPTURE_REFERENCE_PINS)
REFERENCE_INPUTS.update(model_hooks.PINS)
REFERENCE_INPUTS.update(probe_control_hooks.PINS)
LAUNCHER_NAME = 'run-flash-e2-test.sh'
S5_LAUNCHER_NAME = 'run-flash-s5-test.sh'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def manifest(folder):
    folder = Path(folder)
    result = {}
    for line in (folder / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        path = folder / name
        if (len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)
                or name in result or Path(name).is_absolute()
                or '..' in Path(name).parts or path.is_symlink()
                or not path.is_file() or sha(path) != digest):
            raise ValueError('invalid baseline manifest entry')
        result[name] = digest
    if not result:
        raise ValueError('empty baseline manifest')
    return result


def package(source):
    """Rewrite intra-package absolute imports for ``vllm.v1.flash_e3``."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in E3_MODULES:
            source = source.replace('from ' + node.module + ' import ',
                                    'from .' + node.module + ' import ')
    ast.parse(source)
    return source


def _replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('source anchor changed or duplicated: ' +
                         old[:100].replace('\n', '\\n'))
    return text.replace(old, new, 1)


_RUNNER_IMPORT = 'import vllm.v1.flash_e3.lane_hooks as _flash_e3_hooks\n'
_RUNNER_IMPORT_S5 = (_RUNNER_IMPORT +
                     'import vllm.v1.flash_e3.s5_startup as _flash_s5_startup\n')
_RUNNER_INIT = ('        _flash_e3_hooks.init_lane(self, kv_caches_dict, is_profiling)\n')
_RUNNER_INIT_S5 = (
    '        _flash_s5_startup.install_builder(self, kv_caches_dict, is_profiling)\n' +
    _RUNNER_INIT)


def compose_runner(source):
    """A's lane transform, then C's startup-builder insertion."""
    if _sha_text(source) != E2_RUNNER_SHA256:
        raise ValueError('runner is not the reviewed E2 composition')
    text = graph_hooks.generate(lane_hooks.generate(source))
    text = _replace(text, _RUNNER_IMPORT, _RUNNER_IMPORT_S5)
    text = _replace(text, _RUNNER_INIT, _RUNNER_INIT_S5)
    ast.parse(text)
    return text


def revert_runner(source, expected):
    text = _replace(source, _RUNNER_INIT_S5, _RUNNER_INIT)
    text = _replace(text, _RUNNER_IMPORT_S5, _RUNNER_IMPORT)
    text = graph_hooks.revert(text)
    text = lane_hooks.revert(text, expected=expected)
    return text


def _s5_launcher(source):
    """Extend the reviewed E2 test launcher with an off-by-default S5 branch."""
    # Keep the E2 branch and all its checks byte-identical; only add a guarded
    # mount set. e2 is still mounted when present; S5=1 does not imply E2.
    addition = '''# E3/S5 offline-unqualified payload. Inert unless S5=1.
s5=(); s5_auth=()
if [[ -d $OVERLAY/e3 ]]; then
  [[ ${S5:-0} == 1 ]] || { echo "S5 needs the declared overlap launcher" >&2; exit 2; }
  [[ ${FLASH_S5_AUTH_ENV_FILE:-} == /* ]] || { echo "S5 needs private auth file" >&2; exit 2; }
  [[ ${EXTRA_ARGS:-} != *--api-key* ]] || { echo "S5 refuses argv authentication" >&2; exit 2; }
  [[ ${EXTRA_ENV:-} != *VLLM_API_KEY* ]] || { echo "S5 requires private-file authentication" >&2; exit 2; }
  s5_auth=(--env-file "$FLASH_S5_AUTH_ENV_FILE")
  for f in e3/__init__.py e3/decode-probe-evidence.json ''' + ''.join('e3/' + name + '.py ' for name in E3_MODULES) + ''.join('e3/' + name + ' ' for name in NATIVE_FILES) + '''e3/native/build_green_helper.sh e3_sources/executor.py e3_sources/core.py e3_sources/worker.py e3_sources/cudagraph_utils.py e3_sources/sampler.py e3_sources/async_utils.py e3_sources/model.py e3_sources/api_routers.py qsa.py ops_qsa.py; do
    grep -q "  $f$" "$OVERLAY/SHA256SUMS" || { echo "S5 mount not checksummed" >&2; exit 2; }
  done
  V=/usr/local/lib/python3.12/dist-packages/vllm/v1
  s5=(-v "$OVERLAY/e3:$V/flash_e3:ro"
      -v "$OVERLAY/e3_sources/executor.py:$V/executor/multiproc_executor.py:ro"
      -v "$OVERLAY/e3_sources/core.py:$V/engine/core.py:ro"
      -v "$OVERLAY/e3_sources/worker.py:$V/worker/gpu_worker.py:ro"
      -v "$OVERLAY/e3_sources/cudagraph_utils.py:$V/worker/gpu/cudagraph_utils.py:ro"
      -v "$OVERLAY/e3_sources/sampler.py:$V/worker/gpu/sample/sampler.py:ro"
      -v "$OVERLAY/e3_sources/async_utils.py:$V/worker/gpu/async_utils.py:ro"
      -v "$OVERLAY/e3_sources/api_routers.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/launchers/api_server/routers.py:ro"
      -v "$OVERLAY/e3_sources/model.py:''' + MODEL_DEST + ''':ro")
fi
'''
    text = _replace(source, '# VISION=1: image input', addition + '# VISION=1: image input')
    text = _replace(text, '  "${runner[@]}" "${e2[@]}" \\\n',
                    '  "${runner[@]}" "${e2[@]}" "${s5[@]}" \\\n')
    text = _replace(text, '  "${extra_env[@]}" "${mps[@]}"',
                    '  "${extra_env[@]}" "${s5_auth[@]}" "${mps[@]}"')
    return text


GRAPH_API = ('graph transform must expose SOURCE_PINS and generate(files)')


def load_graph(path):
    """Load B's declared transform module by path; never a guessed default."""
    path = Path(path)
    if not path.is_file():
        raise ValueError('graph transform module not found: ' + path.name)
    spec = importlib.util.spec_from_file_location('_flash_s5_graph_transform', path)
    if spec is None or spec.loader is None:
        raise ValueError('cannot load graph transform module')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'generate', None)):
        raise ValueError(GRAPH_API[0])
    pins = getattr(module, 'SOURCE_PINS', None)
    if type(pins) is not dict or not pins:
        raise ValueError(GRAPH_API[0])
    return module


def compose_graph(module, files):
    """Apply B last; it may only touch paths whose supplied input it pinned."""
    changed = module.generate(dict(files))
    if type(changed) is not dict or not changed:
        raise ValueError('graph transform returned no files')
    result = dict(files)
    for name, text in changed.items():
        if name not in module.SOURCE_PINS:
            raise ValueError('graph transform wrote an unpinned path: ' + name)
        if name not in files:
            raise ValueError('graph transform wrote a path outside the composition')
        if _sha_text(files[name]) != module.SOURCE_PINS[name]:
            raise ValueError('graph transform input pin differs: ' + name)
        if type(text) is not str:
            raise ValueError('graph transform did not return source text')
        try:
            ast.parse(text, filename=name)
        except SyntaxError as exc:
            raise ValueError('graph transform produced invalid source: ' + name) from exc
        result[name] = text
    return result


def _decode_probe_declaration(files):
    """Structured declaration of the diagnostic decode-only mode only.

    It qualifies ``diagnostic_modes.decode_only`` for an explicitly requested
    GPU diagnostic; it does not qualify, arm or change the full package.
    """
    runtime_modules = tuple(name for name in E3_MODULES
                            if name not in decode_probe.BUILD_ONLY_SOURCES)
    if set(decode_probe.GENERATED_SOURCES) != set(runtime_modules):
        raise ValueError('diagnostic evidence module list drifted')
    if not set(decode_probe.BUILD_ONLY_SOURCES) <= set(E3_MODULES):
        raise ValueError('build-only module is not in the generated package')
    pins = dict(worker_hooks.PINS)
    pins.update(engine_offers.PINS)
    pins.update(worker_hooks.CAPTURE_REFERENCE_PINS)
    pins.update(probe_control_hooks.PINS)
    pins['v1/worker/gpu/model_runner.py'] = graph_hooks.PINNED_SHA256
    shas = {}
    for name in decode_probe.GENERATED_SOURCES:
        key = 'e3/' + name + '.py'
        if key not in files:
            raise ValueError('decode probe declaration lacks generated ' + key)
        shas[name] = _sha_text(files[key])
    build_only = {}
    for name in decode_probe.BUILD_ONLY_SOURCES:
        key = 'e3/' + name + '.py'
        if key not in files:
            raise ValueError('decode probe declaration lacks generated ' + key)
        build_only[name] = _sha_text(files[key])
    return {
        'declared': True,
        'qualifies': 'diagnostic_modes.decode_only',
        'diagnostic_only': True,
        'scope': ('separately declared diagnostic mode; it does not qualify the full '
                  'FLASH_S5_OVERLAP package, which stays armable:false'),
        'enabled_by': {decode_probe.PROBE_ENV: '1',
                       decode_probe.OVERLAP_ENV: '0',
                       decode_probe.DECODE_SMS_ENV: '<explicit, no default>',
                       'FLASH_S5_GREEN_LIB_SHA256': '<compiled artifact pin, required>'},
        'requires': {
            'flag_conflict': 'probe=1 with overlap=1 refuses before any allocation',
            'decode_sms': 'FLASH_S5_DECODE_SMS must be declared; code supplies no default',
            'storage_epoch': decode_probe.STORAGE_EPOCH,
            'route': 'drained RouteController transitions; unsupported dispatch invalidates the interval',
        },
        'gate': 's5_startup.require_decode_probe_ready',
        'startup': 'native_startup.install_decode_probe',
        'constructs': ['green split (worker_hooks.after_runner_init)',
                       'split decode graph bank, own pool (split_bank)',
                       'drained RouteController'],
        'excluded': list(decode_probe.EXCLUDED),
        'replay_observation': {
            'opt_in': 'FLASH_S5_REPLAY_OBSERVER=1',
            'helper_source': 'e3/native/flash_s5_replay_observer.cpp',
            'helper_library': 'e3/native/flash_s5_replay_observer.so',
            'library_pin_env': 'FLASH_S5_REPLAY_LIB_SHA256',
            'cupti_library_sha256': 'e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e',
            'scope': 'first split correctness window only; owned stop required before W1',
            'build': 'e3/native/build_q17_helpers.sh; final manifest after e3_finalize_package.py',
        },
        'module_inventory': {
            'runtime': list(runtime_modules),
            'build_only': list(decode_probe.BUILD_ONLY_SOURCES),
            'basis': ('build-only sources generate runtime text and are shipped/checksummed '
                      'for the builder; they are not imported by the runtime package'),
        },
        'evidence': {
            'pins': pins,
            'helper_source_sha256': green_context.helper_source_sha256(),
            'generated_source_shas': shas,
            'build_only_source_shas': build_only,
            'tests': list(decode_probe.REQUIRED_TESTS),
        },
    }


def build(base, out, reference, base_manifest_sha256, graph=None,
          settings=None):
    base, out, reference = map(Path, (base, out, reference))
    if out.exists():
        raise ValueError('new output directory required; nothing overwritten')
    if out.resolve().is_relative_to(base.resolve()):
        raise ValueError('output cannot be inside baseline')
    if sha(base / 'SHA256SUMS') != base_manifest_sha256:
        raise ValueError('baseline manifest differs from declaration')

    root = manifest(base)
    core = manifest(base / 'core')
    base_generation_path = base / 'E2-GENERATION.json'
    if not base_generation_path.is_file():
        raise ValueError('baseline is not an E2 generation')
    base_generation = json.loads(base_generation_path.read_text())
    if base_generation.get('default_feature') != 'FLASH_E2_ARENA=0':
        raise ValueError('baseline E2 generation is not default-off')
    if base_generation.get('incremental_kv_bytes') != 0:
        raise ValueError('baseline E2 generation is not the reviewed zero-growth package')
    covered = set(root) | {'core/' + name for name in core} | {'SHA256SUMS', 'core/SHA256SUMS'}
    if (base / 'vision').exists() or (base / 'vision').is_symlink():
        vision = manifest(base / 'vision')
        covered |= {'vision/' + name for name in vision} | {'vision/SHA256SUMS'}
    for path in base.rglob('*'):
        if path.is_symlink():
            raise ValueError('baseline symlinks not supported')
        if path.is_file() and path.relative_to(base).as_posix() not in covered:
            raise ValueError('unmanifested baseline file')
    for rel, digest in E2_INPUTS.items():
        if rel not in covered or sha(base / rel) != digest:
            raise ValueError('unreviewed E2-composed source: ' + rel)
    for rel, digest in E2_WORKER_INPUTS.items():
        if rel not in covered or sha(base / rel) != digest:
            raise ValueError('unreviewed E2-composed worker source: ' + rel)
    for rel, digest in REFERENCE_INPUTS.items():
        if sha(reference / rel) != digest:
            raise ValueError('unreviewed image source: ' + rel)

    launcher_source = (base / LAUNCHER_NAME).read_text()
    if 'S5 needs the declared overlap launcher' in launcher_source:
        raise ValueError('baseline launcher already carries S5 mounts')

    _MANIFESTS = {'SHA256SUMS', 'core/SHA256SUMS', 'vision/SHA256SUMS',
                  'E2-GENERATION.json'}
    base_text = {}
    for path in sorted(base.rglob('*')):
        if path.is_symlink():
            raise ValueError('baseline symlinks not supported')
        rel = path.relative_to(base).as_posix()
        if path.is_file() and rel not in _MANIFESTS and rel in (
                *E2_INPUTS, *E2_WORKER_INPUTS, LAUNCHER_NAME):
            base_text[rel] = path.read_text()
    files = dict(base_text)
    files['model_runner_v2.py'] = compose_runner(files['model_runner_v2.py'])
    files['core/scheduler.py'] = scheduler_hooks.generate_scheduler(files['core/scheduler.py'])
    files['core/block_pool.py'] = scheduler_hooks.generate_block_pool(files['core/block_pool.py'])
    # D2: generated native QSA operation call sites.  The layer/attention
    # transforms are E2-byte-preserving outside their inserted boundaries; the
    # runner seam is inserted into the S5-composed runner before B's optional
    # final graph transform.
    files['qsa.py'] = qsa_driver.generate_one(
        'models/qwen4_exp/nvidia/qsa.py', operation_source.generate_layer(files['qsa.py']))
    files['ops_qsa.py'] = operation_source.generate_ops(files['ops_qsa.py'])
    files['model_runner_v2.py'] = operation_source.generate_runner(
        files['model_runner_v2.py'])
    files['e3/__init__.py'] = '"""Opt-in S5 overlap lane; owner review and GPU gates required."""\n'
    for name in E3_MODULES:
        files['e3/' + name + '.py'] = package((HERE / (name + '.py')).read_text())
    files['e3_sources/executor.py'] = executor_hooks.generate_executor(
        (reference / executor_hooks.EXECUTOR_PATH).read_text())
    files['e3_sources/core.py'] = executor_hooks.generate_core(
        (reference / executor_hooks.CORE_PATH).read_text())
    # C's worker RPC methods, then D4's worker seams (lane, split creation,
    # decode scope) on that composition; plus the capture-context seam and the
    # lane forward methods on their pinned sources; plus the native helper.
    files['e3_sources/worker.py'] = worker_hooks.generate_one(
        executor_hooks.WORKER_PATH,
        executor_hooks.generate_worker((reference / executor_hooks.WORKER_PATH).read_text()))
    files['e3_sources/cudagraph_utils.py'] = worker_hooks.generate_one(
        CUDAGRAPH_PATH, (reference / CUDAGRAPH_PATH).read_text())
    files['e3_sources/model.py'] = model_hooks.generate_one(
        MODEL_PATH, (reference / MODEL_PATH).read_text())
    files['e3_sources/api_routers.py'] = probe_control_hooks.generate(
        (reference / probe_control_hooks.PATH).read_text())
    for name in NATIVE_FILES:
        files['e3/' + name] = (HERE / name).read_text()
    files['e3/native/build_green_helper.sh'] = BUILD_GREEN_HELPER
    files[S5_LAUNCHER_NAME] = _s5_launcher(launcher_source)
    graph_record = {'module': 'graph_hooks.py',
                    'sha256': sha(HERE / 'graph_hooks.py'),
                    'router_sha256': sha(HERE / 'graph_router.py'),
                    'source_pins': dict(graph_hooks.A_SOURCE_SHA256)}
    if graph is not None:
        module = load_graph(graph)
        files = compose_graph(module, files)
        graph_record = {'module': Path(graph).name,
                        'source_pins': dict(module.SOURCE_PINS)}
    # Reviewed source-owned offers compose last, after E2/A/B/D2 and capture
    # stream transforms. Full-file input pins and exact inverses are enforced.
    files['model_runner_v2.py'] = engine_offers.generate_one(
        'v1/worker/gpu/model_runner.py', files['model_runner_v2.py'])
    files['e3_sources/cudagraph_utils.py'] = engine_offers.generate_one(
        CUDAGRAPH_PATH, files['e3_sources/cudagraph_utils.py'])
    for relative, output in (
            ('v1/worker/gpu/sample/sampler.py', 'e3_sources/sampler.py'),
            ('v1/worker/gpu/async_utils.py', 'e3_sources/async_utils.py')):
        files[output] = engine_offers.generate_one(relative, (reference / relative).read_text())
    probe_declaration = _decode_probe_declaration(files)
    files['e3/' + decode_probe.EVIDENCE_FILE] = json.dumps(probe_declaration['evidence'], sort_keys=True) + '\n'
    for name, text in files.items():
        if name.endswith('.py'):
            ast.parse(text, filename=name)
    changed = {name for name, text in files.items() if base_text.get(name) != text}
    changed |= {S5_LAUNCHER_NAME, 'S5-OVERLAP-GENERATION.json',
                'SHA256SUMS', 'core/SHA256SUMS'}

    installed = {
        async_handoff.OVERLAP_ENV: '0',
        decode_probe.PROBE_ENV: '0',
        async_handoff.TRANSPORT_ENV: async_handoff.QUALIFIED_TRANSPORT,
        async_handoff.SPARSE_OVERLAP_ENV: '0',
        async_handoff.COMMAND_SLOTS_ENV: str(async_handoff.PINNED_RING_SLOTS),
        async_handoff.REPLY_TIMEOUT_ENV: '300.0',
        async_handoff.ACK_TIMEOUT_ENV: '30.0',
        async_handoff.CHUNK_TOKENS_ENV: '1024',
        async_handoff.BACKEND_ENV: 'native',
    }
    budget = async_handoff.allocation_budget(
        int(installed[async_handoff.CHUNK_TOKENS_ENV]),
        int(installed[async_handoff.COMMAND_SLOTS_ENV]))
    budget_record = {
        'entries': [dict(name=e.name, owner=e.owner, category=e.category,
                         nbytes=e.nbytes, basis=e.basis,
                         new_allocation=e.new_allocation,
                         reuse_condition=e.reuse_condition)
                    for e in budget.entries],
        'total_bytes': budget.total_bytes,
        'new_bytes': budget.new_bytes,
        'existing_bytes': budget.existing_bytes,
        'by_category': budget.by_category(),
        'unmeasured': list(budget.unmeasured()),
        'funding': 'no claim on the unmet 512-MiB reserve or the existing arena',
    }
    record = {
        'schema': 1,
        'kind': 'e3-s5-overlap',
        'base_manifest_sha256': base_manifest_sha256,
        'base_generation_sha256': sha(base_generation_path),
        'base_e2_record': base_generation,
        'base_core_manifest_sha256': sha(base / 'core/SHA256SUMS'),
        'e2_composed_pins': E2_INPUTS,
        'e2_worker_composed_pins': E2_WORKER_INPUTS,
        'reference_source_pins': REFERENCE_INPUTS,
        'engine_offer_composition': {
            'enabled_by': engine_offers.ENABLE_ENV,
            'window': 'short placement window only; inactive before and after',
            'reference_pins': dict(engine_offers.PINS),
            'runner_input_sha256': engine_offers.MODEL_RUNNER_PACKAGE_SHA256,
            'cudagraph_input_sha256': engine_offers.CUDAGRAPH_AFTER_WORKER_SHA256,
            'outputs': {key: _sha_text(files[key]) for key in (
                'model_runner_v2.py', 'e3_sources/cudagraph_utils.py',
                'e3_sources/sampler.py', 'e3_sources/async_utils.py')},
        },
        'generated_files': sorted(changed),
        'default_feature': 'FLASH_S5_OVERLAP=0',
        'installed_settings': installed,
        'allocation_budget': budget_record,
        'route': {'admitted': async_handoff.QUALIFIED_TRANSPORT,
                  'serial_fallback': 'sparse-UVA and unqualified transports are counted, not overlapped'},
        'graph_composition': graph_record,
        'launcher': {'name': S5_LAUNCHER_NAME,
                     'mounts_require': 'S5=1',
                     'e2_launcher_preserved': LAUNCHER_NAME,
                     'status': 'offline-generated; not mounted or started'},
        'status': ('offline checkpoint; bounded dense QSA producer -> CPU-source DMA -> '
                   'attention generator yields once per bucket; stock runner/layer entry refuses; '
                   'cross-layer host continuation, production admission and capture are '
                   'NOT IMPLEMENTED; NOT ARMABLE'),
        'armable': False,
        'native_gaps': [
            'enabled startup is hard-refused before allocation/transport ownership; CPU fixtures explicitly bypass this refusal',
            'production scheduler admission marker and mixed-batch extraction not implemented; without SchedulerOutput.flash_s5_prefill_admission the driver parks nothing',
            'runner._flash_s5_graph_capture is supplied by split_bank (second stock-manager instance on the green decode stream, own pool); split-bank capture, green contexts, lane communicators, lane forward/QSA driver, chunk admission and the segment dispatcher are implemented offline with fake-CUDA tests only; the native helper library is built on the box by e3/native/build_green_helper.sh and its sha must be recorded by the queue',
            'SOURCE_HISTORY (COPY_EVENT) and PP-buffer leases are not acquired or released; only ARENA leases are wired',
            'real model persistent-state commit value: admission must supply an immutable state snapshot (A accepts CPU snapshots only)',
            'bounded attention generator is implemented; stock runner/layer path explicitly refuses until prefix/layer/suffix dispatcher is wired',
            'native admission must supply explicit complete per-layer CPU tables and real allocation/source leases; flat block ids are not a QSA history map',
            'PP handoff output/parity validation not implemented; PrefillPPChannel is constructed but carries no traffic',
            'SM membership, split-pool bytes, host launch quanta and ACK lag are unmeasured until the owner GPU smoke',
            'no GPU, CUDA graph, PP ring, DMA qualification or performance run',
        ],
        'incremental_kv_bytes': 0,
    }
    # Diagnostic-only declaration, separate from the full package's armability.
    record['diagnostic_modes'] = {
        'decode_only': probe_declaration,
    }
    if settings:
        record['requested_settings'] = dict(settings)

    # Validation is complete: only now create the new output.
    shutil.copytree(base, out)
    for name, text in files.items():
        if base_text.get(name) == text:
            continue
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (out / S5_LAUNCHER_NAME).chmod(0o755)
    (out / 'S5-OVERLAP-GENERATION.json').write_text(
        json.dumps(record, indent=2, sort_keys=True) + '\n')
    core_out = dict(core)
    for name in core_out:
        core_out[name] = sha(out / 'core' / name)
    (out / 'core/SHA256SUMS').write_text(
        ''.join(core_out[name] + '  ' + name + '\n' for name in sorted(core_out)))
    entries = []
    for path in sorted(out.rglob('*')):
        if path.is_file() and path != out / 'SHA256SUMS':
            entries.append(sha(path) + '  ' + path.relative_to(out).as_posix() + '\n')
    (out / 'SHA256SUMS').write_text(''.join(entries))
    out.chmod(0o755)
    for path in out.rglob('*'):
        path.chmod(0o755 if path.is_dir() else 0o644 | (path.stat().st_mode & 0o111))
    return dict(manifest_sha256=sha(out / 'SHA256SUMS'),
                generated_files=sorted(changed),
                settings=installed,
                allocation_budget=budget_record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--base-manifest-sha256', required=True)
    parser.add_argument('--graph', type=Path,
                        help='B graph transform module with SOURCE_PINS/generate')
    args = parser.parse_args()
    print(json.dumps(build(args.base, args.out, args.reference,
                           args.base_manifest_sha256, args.graph),
                     sort_keys=True))


if __name__ == '__main__':
    main()
