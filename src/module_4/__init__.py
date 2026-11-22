from .base import BaseModule4Detector, compute_suspicion_level
from .f3net_detector import F3NetDetector
from .forensic_translator import Module4ForensicTranslator
from .runner import (
    DETECTOR_REGISTRY,
    list_available_detectors,
    register_detector,
    run_module4,
)
from .weight_store import (
    MODEL_REGISTRY,
    ensure_model_weights,
    find_existing_weights,
    get_cache_dir,
    set_cache_dir,
)

__all__ = [
    "BaseModule4Detector",
    "F3NetDetector",
    "Module4ForensicTranslator",
    "DETECTOR_REGISTRY",
    "MODEL_REGISTRY",
    "compute_suspicion_level",
    "ensure_model_weights",
    "find_existing_weights",
    "get_cache_dir",
    "list_available_detectors",
    "register_detector",
    "run_module4",
    "set_cache_dir",
]
