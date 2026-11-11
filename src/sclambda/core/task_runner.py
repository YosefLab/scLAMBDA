"""
Unified concurrency manager for CPU-bound parallel work.

Simplifies the previous asyncio + multiple semaphores approach
to a single class with clear responsibilities.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed, Future, TimeoutError
from threading import RLock
from typing import Callable, Iterable, TypeVar, Any
from dataclasses import dataclass

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class TaskResult:
    """Result wrapper with optional error tracking."""

    value: Any = None
    error: Exception | None = None

    @property
    def success(self) -> bool:
        return self.error is None


class TaskRunner:
    """
    Unified concurrency manager for CPU-bound parallel work.

    Replaces the previous pattern of:
    - asyncio.to_thread() for CPU tasks
    - Multiple semaphores (thread_semaphore, cpu_semaphore)
    - Nested async context managers

    With a simpler interface:
    - submit() for single tasks
    - map_parallel() for batch processing
    - with_lock() for shared state mutations

    Example:
        with TaskRunner(max_workers=4) as runner:
            # Process items in parallel
            results = runner.map_parallel(process_item, items)

            # Mutate shared state safely
            runner.with_lock(accessor.set_annotation, cluster_id, label)

    Workers may use with_lock() for short shared-state operations. Never wait
    for other tasks while holding that lock, or hold it across expensive work.
    """

    def __init__(self, max_workers: int = 4, default_timeout: float | None = None):
        """
        Initialize TaskRunner.

        Args:
            max_workers: Maximum concurrent threads
            default_timeout: Default timeout in seconds for blocking operations (None = no timeout)
        """
        self._max_workers = max_workers
        self._default_timeout = default_timeout
        self._executor: ThreadPoolExecutor | None = None
        self._lock = RLock()  # Reentrant lock for thread safety and nested calls
        self._shutdown = False

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """
        Lazily create executor (thread-safe).

        Returns:
            The executor instance

        Raises:
            RuntimeError: If TaskRunner has been shut down
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("TaskRunner has been shut down")
            if self._executor is None:
                from sclambda.core.native_threads import prepare_native_threads
                prepare_native_threads()
                self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
            return self._executor

    def submit(self, func: Callable[..., R], *args, **kwargs) -> Future[R]:
        """
        Submit a task for execution.

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Future that resolves to the result

        Raises:
            RuntimeError: If TaskRunner has been shut down
        """
        executor = self._ensure_executor()
        return executor.submit(func, *args, **kwargs)

    def submit_and_wait(
        self, func: Callable[..., R], *args, timeout: float | None = ..., **kwargs  # type: ignore[assignment]
    ) -> R:
        """
        Submit task and wait for result.

        Args:
            func: Function to execute
            *args: Positional arguments to pass to function
            timeout: Timeout in seconds (None = no timeout, ... = use default_timeout)
            **kwargs: Keyword arguments to pass to function

        Returns:
            Result of function execution

        Raises:
            TimeoutError: If timeout is exceeded
            RuntimeError: If TaskRunner has been shut down
        """
        actual_timeout = self._default_timeout if timeout is ... else timeout
        return self.submit(func, *args, **kwargs).result(timeout=actual_timeout)

    def map_parallel(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        return_exceptions: bool = False,
        timeout: float | None = ...,  # type: ignore[assignment]
    ) -> list[R | TaskResult]:
        """
        Process items in parallel, return results in completion order.

        Args:
            func: Function to apply to each item
            items: Items to process
            return_exceptions: If True, wrap errors in TaskResult instead of raising
            timeout: Total timeout for all tasks (None = no timeout, ... = use default_timeout)

        Returns:
            List of results (in completion order, not input order)

        Raises:
            TimeoutError: If timeout is exceeded (when return_exceptions=False)
            RuntimeError: If TaskRunner has been shut down
        """
        executor = self._ensure_executor()
        items_list = list(items)

        if not items_list:
            return []

        actual_timeout = self._default_timeout if timeout is ... else timeout
        futures = {executor.submit(func, item): item for item in items_list}
        results: list[R | TaskResult] = []

        try:
            for future in as_completed(futures, timeout=actual_timeout):
                if return_exceptions:
                    try:
                        results.append(TaskResult(value=future.result()))
                    except Exception as e:
                        results.append(TaskResult(error=e))
                else:
                    results.append(future.result())
        except TimeoutError:
            if return_exceptions:
                # Mark remaining futures as timed out
                for future in futures:
                    if not future.done():
                        results.append(TaskResult(error=TimeoutError("Task timed out")))
            else:
                # Cancel remaining futures and re-raise
                for future in futures:
                    future.cancel()
                raise

        return results

    def map_parallel_ordered(
        self,
        func: Callable[[T], R],
        items: Iterable[T],
        timeout: float | None = ...,  # type: ignore[assignment]
    ) -> list[R]:
        """
        Process items in parallel, return results in input order.

        Args:
            func: Function to apply to each item
            items: Items to process
            timeout: Timeout per individual task (None = no timeout, ... = use default_timeout)

        Returns:
            List of results (in same order as input)

        Raises:
            TimeoutError: If any task exceeds timeout
            RuntimeError: If TaskRunner has been shut down
        """
        executor = self._ensure_executor()
        items_list = list(items)

        if not items_list:
            return []

        actual_timeout = self._default_timeout if timeout is ... else timeout
        futures = [executor.submit(func, item) for item in items_list]

        try:
            return [f.result(timeout=actual_timeout) for f in futures]
        except TimeoutError:
            # Cancel remaining futures on timeout
            for f in futures:
                f.cancel()
            raise

    def with_lock(self, func: Callable[..., R], *args, **kwargs) -> R:
        """
        Run function with exclusive lock (for shared state mutations).

        Use this when mutating shared state that isn't thread-safe.

        Safe from worker threads for short state updates. Do not wait for
        other workers while holding the lock.

        Args:
            func: Function to execute under lock
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result of function execution
        """
        with self._lock:
            return func(*args, **kwargs)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False):
        """
        Shutdown executor.

        Args:
            wait: If True, wait for pending tasks to complete
            cancel_futures: If True, cancel pending futures (Python 3.9+)
        """
        with self._lock:
            self._shutdown = True
            executor = self._executor
        # Workers may need with_lock() to finish. Never join them under it.
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=cancel_futures)

    @property
    def is_shutdown(self) -> bool:
        """Check if TaskRunner has been shut down."""
        return self._shutdown

    @property
    def max_workers(self) -> int:
        """Get the maximum number of worker threads."""
        return self._max_workers

    def __enter__(self) -> "TaskRunner":
        return self

    def __exit__(self, *args):
        self.shutdown()

    def __repr__(self) -> str:
        status = "shutdown" if self._shutdown else "active"
        return f"TaskRunner(max_workers={self._max_workers}, status={status})"
