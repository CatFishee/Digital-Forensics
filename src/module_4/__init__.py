from .base import BaseModule4Detector, CropSpec, compute_suspicion_level
from .forensic_translator import Module4ForensicTranslator
from .runner import (
    DETECTOR_REGISTRY,
    get_required_crop_specs,
    list_available_detectors,
    register_detector,
    run_module4,
)
from .weight_store import (
    MODEL_REGISTRY,
    ensure_hf_model_dir,
    ensure_model_weights,
    find_existing_hf_model,
    find_existing_weights,
    get_cache_dir,
    set_cache_dir,
)

__all__ = [
    "BaseModule4Detector",
    "CropSpec",
    "Module4ForensicTranslator",
    "DETECTOR_REGISTRY",
    "MODEL_REGISTRY",
    "compute_suspicion_level",
    "ensure_hf_model_dir",
    "ensure_model_weights",
    "find_existing_hf_model",
    "find_existing_weights",
    "get_cache_dir",
    "get_required_crop_specs",
    "list_available_detectors",
    "register_detector",
    "run_module4",
    "set_cache_dir",
]
