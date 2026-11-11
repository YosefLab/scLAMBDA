"""
Framework module: general abstractions for building LLM-powered pipelines.

Components:
- BaseNode: Base class for all pipeline nodes
- LLMNode: Base class for nodes that interact with LLMs
- ParseSchema: Declarative parsing configuration for LLM responses
- Pipeline: Base orchestration class for hierarchical pipelines
- LLMPromptChain: Orchestrates prompt loading, LLM querying, and parsing
- ChainResult: Structured result from LLM prompt chain
- JSONResponseParser: Extracts JSON from LLM responses
"""

from .base_node import BaseNode, LLMNode, ParseSchema
from .base_pipeline import Pipeline
from .llm_chain import LLMPromptChain, ChainResult, JSONResponseParser
from sclambda.pipelines.registry import register_pipeline, get_pipeline_class, list_pipelines

__all__ = [
    "BaseNode",
    "LLMNode",
    "ParseSchema",
    "Pipeline",
    "LLMPromptChain",
    "ChainResult",
    "JSONResponseParser",
    "register_pipeline",
    "get_pipeline_class",
    "list_pipelines",
]
