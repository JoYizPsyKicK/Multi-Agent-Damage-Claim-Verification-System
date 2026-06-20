import asyncio
import random
import logging
from functools import wraps
from typing import Callable, Any
import pydantic

logger = logging.getLogger(__name__)

def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 30.0
):
    """
    An async decorator that retries a function call using exponential backoff with random jitter.
    """
    def decorator(func: Callable[..., Any]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            while True:
                try:
                    return await func(*args, **kwargs)
                except (AttributeError, TypeError, KeyError, ValueError, NameError, pydantic.ValidationError) as e:
                    # Do not retry programming or validation errors
                    logger.error(
                        f"Non-retryable exception in {func.__name__}: {e}. Skipping retries."
                    )
                    raise e
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} attempts. Error: {e}"
                        )
                        raise e
                    
                    # Calculate delay with exponential backoff and random jitter
                    jitter = random.uniform(0.1, 0.5)
                    current_delay = min(delay * (backoff_factor ** (retries - 1)) + jitter, max_delay)
                    
                    logger.warning(
                        f"Attempt {retries} for {func.__name__} failed: {e}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    
                    await asyncio.sleep(current_delay)
        return wrapper
    return decorator
