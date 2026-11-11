"""
Timing utilities for tool execution logging.
"""
from __future__ import annotations
import logging
import time
from functools import wraps
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


def log_execution(func: F) -> F:
    """
    Decorator to log tool execution start, finish, and duration.
    
    Usage:
        @log_execution
        def my_tool(...):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.info(f"Started execution: {func_name}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Finished execution: {func_name} (spent {elapsed:.2f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Failed execution: {func_name} (spent {elapsed:.2f}s) - {e}")
            raise
    
    return wrapper  # type: ignore
