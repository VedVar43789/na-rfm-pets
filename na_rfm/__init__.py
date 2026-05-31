"""NA-RFM helpers for camera-ready diffusion steering reproduction."""

from .block_utils import resolve_block
from .ddim_sampling import (
    DDIMSchedule,
    build_ddim_schedule,
    build_ddim_schedule_from_scheduler,
    ddim_step,
    predict_eps,
    sample_ddim,
    slerp,
    space_timesteps,
)
from .gaussian_x0_guidance import (
    GaussianX0Guidance,
    GaussianX0GuidanceSimple,
    load_pca_filter,
)
from .hooks import (
    ActivationGuidanceHook,
    GuidanceWindow,
    RFMCFGWrapper,
)
from .activation_collection import (
    ActivationCollector,
    collect_activations,
)
from .dual_rfm import DualRFM, DualRFMResult, compute_guidance_direction

__all__ = [
    # Block utils
    "resolve_block",
    # DDIM sampling
    "DDIMSchedule",
    "build_ddim_schedule",
    "build_ddim_schedule_from_scheduler",
    "ddim_step",
    "predict_eps",
    "sample_ddim",
    "slerp",
    "space_timesteps",
    # Gaussian X0 guidance
    "GaussianX0Guidance",
    "GaussianX0GuidanceSimple",
    "load_pca_filter",
    # Hooks
    "ActivationGuidanceHook",
    "GuidanceWindow",
    "RFMCFGWrapper",
    # Activation collection
    "ActivationCollector",
    "collect_activations",
    # RFM
    "DualRFM",
    "DualRFMResult",
    "compute_guidance_direction",
]
