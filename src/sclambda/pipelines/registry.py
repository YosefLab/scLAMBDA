"""
Pipeline registry for discovering and registering pipeline classes.

Built-in pipelines are registered at import time.
Third-party pipelines are discovered via entry_points on first lookup miss.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sclambda.framework.base_pipeline import Pipeline

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[Pipeline]] = {}
_ENTRY_POINTS_LOADED = False
_BUILTINS = {
    "celltype": "sclambda.pipelines.celltype.pipeline:CTPipeline",
    "annotate_spatial": "sclambda.pipelines.annotate_spatial.pipeline:AnnotateSpatialPipeline",
}

def _load_builtin(name):
    from importlib import import_module
    if name in _BUILTINS and name not in _REGISTRY:
        module, attribute = _BUILTINS[name].split(":")
        try:
            register_pipeline(name, getattr(import_module(module), attribute))
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith(("langgraph", "langchain")):
                raise ImportError("Python pipelines require sclambda[llm].") from exc
            raise



def register_pipeline(name: str, pipeline_class: type[Pipeline]) -> None:
    """Register a pipeline class under the given name.

    Args:
        name: Short identifier (e.g., "celltype", "annotate_spatial").
              Used as the ``pipeline=`` argument in Lambda.annotate().
        pipeline_class: A concrete subclass of Pipeline.

    Raises:
        TypeError: If pipeline_class is not a Pipeline subclass.
        ValueError: If name is already registered to a different class.
    """
    from sclambda.framework.base_pipeline import Pipeline as BasePipeline

    if not (isinstance(pipeline_class, type) and issubclass(pipeline_class, BasePipeline)):
        raise TypeError(
            f"pipeline_class must be a subclass of Pipeline, got {pipeline_class}"
        )

    existing = _REGISTRY.get(name)
    if existing is not None and existing is not pipeline_class:
        raise ValueError(
            f"Pipeline name '{name}' is already registered to {existing.__name__}. "
            f"Cannot re-register to {pipeline_class.__name__}."
        )

    _REGISTRY[name] = pipeline_class
    logger.debug("Registered pipeline '%s' -> %s", name, pipeline_class.__name__)


def _load_entry_points() -> None:
    """Discover and register third-party pipelines from entry_points."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True

    from importlib.metadata import entry_points

    eps = entry_points(group="sclambda.pipelines")
    for ep in eps:
        if ep.name in _REGISTRY or ep.name in _BUILTINS:
            continue  # built-in takes precedence
        try:
            pipeline_class = ep.load()
            register_pipeline(ep.name, pipeline_class)
            logger.info("Loaded third-party pipeline '%s' from %s", ep.name, ep.value)
        except Exception as e:
            logger.warning("Failed to load pipeline entry_point '%s': %s", ep.name, e)


def get_pipeline_class(name: str) -> type[Pipeline]:
    """Look up a pipeline class by name.

    First checks the local registry, then loads entry_points on miss.

    Raises:
        ValueError: If no pipeline is registered under name.
    """
    _load_builtin(name)
    cls = _REGISTRY.get(name)
    if cls is not None:
        return cls

    # Try entry_points discovery
    _load_entry_points()

    cls = _REGISTRY.get(name)
    if cls is not None:
        return cls

    available = sorted(_REGISTRY.keys())
    raise ValueError(
        f"Unknown pipeline: '{name}'. Available pipelines: {available}"
    )


def list_pipelines() -> list[str]:
    """Return names of all registered pipelines (including entry_points)."""
    _load_entry_points()
    return sorted(set(_REGISTRY) | set(_BUILTINS))
