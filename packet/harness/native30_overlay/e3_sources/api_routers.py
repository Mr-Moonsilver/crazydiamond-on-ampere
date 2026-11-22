# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# [crazydiamond-on-ampere] MODIFIED vLLM FILE. Base: vLLM commit dc36fcce902a63eab06c1b93a5c4a5ee178a0c56,
# [crazydiamond-on-ampere] upstream path vllm/entrypoints/launchers/api_server/routers.py. Changed by the Flash-Next engine overlay (build-6).
# [crazydiamond-on-ampere] Apache-2.0 like the original; see NOTICE and PROVENANCE.md in the repository root.
from argparse import Namespace

from fastapi import FastAPI

from vllm import envs
from vllm.config import ModelConfig
from vllm.tasks import POOLING_TASKS, SupportedTask


def register_api_routers(
    args: Namespace,
    app: FastAPI,
    supported_tasks: tuple["SupportedTask", ...],
    model_config: ModelConfig | None = None,
):
    from vllm.entrypoints.serve import register_vllm_serve_api_routers

    register_vllm_serve_api_routers(app)

    from vllm.v1.flash_e3.probe_control import attach_router as _flash_s5_probe_router
    _flash_s5_probe_router(args, app)
    from vllm.v1.flash_e3.fit_memory import install_api as _flash_fit_api
    _flash_fit_api(args, app)

    from vllm.entrypoints.openai.models.api_router import (
        attach_router as register_models_api_router,
    )

    register_models_api_router(app)

    from vllm.entrypoints.serve.sagemaker.api_router import (
        attach_router as register_sagemaker_api_router,
    )

    register_sagemaker_api_router(app, supported_tasks, model_config)

    if envs.VLLM_SERVER_DEV_MODE:
        from vllm.entrypoints.serve import register_vllm_dev_api_routers

        register_vllm_dev_api_routers(app)

    if "generate" in supported_tasks:
        from vllm.entrypoints.generate.api_router import (
            register_generate_api_routers,
        )

        register_generate_api_routers(app)

        from vllm.entrypoints.serve.elastic_ep.api_router import (
            attach_router as elastic_ep_attach_router,
        )

        elastic_ep_attach_router(app)

    if "generate" in supported_tasks or "render" in supported_tasks:
        from vllm.entrypoints.scale_out.factories import register_scale_out_api_routers

        register_scale_out_api_routers(app, supported_tasks)

    if "transcription" in supported_tasks or "realtime" in supported_tasks:
        from vllm.entrypoints.speech_to_text.factories import (
            register_speech_to_text_api_routers,
        )

        register_speech_to_text_api_routers(app, supported_tasks)

    if any(task in POOLING_TASKS for task in supported_tasks):
        from vllm.entrypoints.pooling.factories import register_pooling_api_routers

        register_pooling_api_routers(app, supported_tasks, model_config)

    if getattr(args, "enable_fault_tolerance", False):
        from vllm.entrypoints.serve.fault_tolerance.api_router import (
            register_fault_tolerance_api_router,
        )

        register_fault_tolerance_api_router(app)
