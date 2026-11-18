"""Annotation pipeline with confidence-gated human review."""

from .backends import MockVLMBackend, QwenVLBackend, build_backend  # noqa: F401
from .pipeline import AnnotationPipeline  # noqa: F401
from .quality import cohen_kappa, fleiss_kappa, localization_report  # noqa: F401
