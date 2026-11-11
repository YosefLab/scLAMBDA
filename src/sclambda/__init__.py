"""Single-cell annotation through Python pipelines or the lambdakit CLI."""
from sclambda._version import __version__

__all__ = ["Lambda", "register_pipeline", "list_pipelines"]

def __getattr__(name):
    if name == "Lambda":
        from sclambda.main import Lambda
        return Lambda
    if name in ("register_pipeline", "list_pipelines"):
        from sclambda.pipelines import registry
        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
