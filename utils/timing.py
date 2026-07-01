"""
timing.py

Small reusable decorator to time pipeline steps and log execution
duration consistently, satisfying the project's logging requirements
(execution time per step).
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from utils.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def log_execution_time(step_name: str) -> Callable[[F], F]:
    """
    Decorator factory that logs the start, end, and duration of a
    pipeline step.

    Parameters
    ----------
    step_name : str
        Human-readable name of the step, e.g. "Kaggle Ingestion".
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.info("START | %s", step_name)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception:
                logger.exception("FAILED | %s", step_name)
                raise
            elapsed = time.perf_counter() - start
            logger.info("END | %s | duration=%.2fs", step_name, elapsed)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
