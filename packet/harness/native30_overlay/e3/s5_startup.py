"""C startup adapter: install A's lane builder and C's handoff composition.

Called from the generated runner just before A's ``init_lane``. Nothing is
allocated here. The owner-reviewed native bundle supplies the real transport,
completion backend, lanes, native allocation inventory and A ``LaneRuntime``;
without it enabled startup fails instead of falling back to the CPU fixture.

This module deliberately has no torch/vLLM import so the offline tests can load
it. The generated runner imports it as ``vllm.v1.flash_e3.s5_startup``.
"""
import os
import time

from async_handoff import (
    Composition,
    HandoffConfig,
    HandoffError,
    build_composition,
)
from lane_hooks import LaneRuntime, _NullBackend


OVERLAP_ENV = 'FLASH_S5_OVERLAP'
_NATIVE_ATTR = '_flash_s5_native_factory'
_BUILDER_ATTR = '_flash_e3_build_state'
_COMPOSITION_ATTR = '_flash_s5_composition'


class StartupError(HandoffError):
    pass


def _enabled(env):
    value = env.get(OVERLAP_ENV, '0')
    if value not in ('0', '1'):
        raise StartupError('invalid boolean setting: ' + OVERLAP_ENV)
    return value == '1'


def _bundle_field(bundle, name):
    value = getattr(bundle, name, None)
    if value is None:
        raise StartupError('native bundle is missing ' + name)
    return value


def install_builder(runner, kv_caches, is_profiling=False, *, env=None):
    """Install the builder A's ``init_lane`` calls at the end of runner init."""
    env = os.environ if env is None else env
    if is_profiling or not _enabled(env):
        return None
    if getattr(runner, _BUILDER_ATTR, None) is not None:
        raise StartupError('E3 lane builder is already installed')
    factory = getattr(runner, _NATIVE_ATTR, None)
    if factory is None or not callable(factory):
        raise StartupError(
            'FLASH_S5_OVERLAP=1 requires the owner native allocation/transport factory')

    def build(current, caches):
        if getattr(current, _BUILDER_ATTR, None) is not build:
            raise StartupError('E3 lane builder was replaced during startup')
        bundle = factory(current, caches)
        if bundle is None:
            raise StartupError('native factory returned no bundle')
        transport = _bundle_field(bundle, 'transport')
        backend = _bundle_field(bundle, 'backend')
        lanes = _bundle_field(bundle, 'lanes')
        runtime = _bundle_field(bundle, 'runtime')
        if not isinstance(runtime, LaneRuntime):
            raise StartupError('native bundle runtime is not A LaneRuntime')
        if isinstance(runtime.backend, _NullBackend):
            raise StartupError('native bundle supplied the null fixture backend')
        allocations = _bundle_field(bundle, 'native_allocations')
        composition = build_composition(
            env, transport=transport, backend=backend, lanes=lanes,
            allocation=getattr(bundle, 'allocation', None),
            native_allocations=allocations, runtime=runtime,
            scheduler_leases=getattr(bundle, 'scheduler_leases', None),
            clock=getattr(bundle, 'clock', None) or time.monotonic,
            telemetry=getattr(bundle, 'telemetry', None),
        )
        if not isinstance(composition, Composition):
            raise StartupError('enabled startup did not produce a composition')
        if getattr(current, _COMPOSITION_ATTR, None) is not None:
            raise StartupError('runner already carries an E3 composition')
        current._flash_s5_composition = composition
        return runtime

    runner._flash_e3_build_state = build
    return build


def settings(env=None):
    """Metadata-only settings echo; no secret or path is read."""
    env = os.environ if env is None else env
    config = HandoffConfig.from_env(env)
    return dict(enabled=config.enabled, transport=config.transport,
                command_slots=config.command_slots,
                reply_timeout_s=config.reply_timeout_s,
                ack_timeout_s=config.ack_timeout_s,
                chunk_tokens=config.chunk_tokens,
                sparse_overlap=False, backend='native' if config.enabled else 'inert')


__all__ = ['OVERLAP_ENV', 'StartupError', 'install_builder', 'settings']
