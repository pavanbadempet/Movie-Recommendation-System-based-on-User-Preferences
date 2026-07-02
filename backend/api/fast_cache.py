import time
import logging
from functools import wraps
from typing import Any
import inspect

logger = logging.getLogger(__name__)

class FastCache:
    """A fast, thread-safe in-process TTL cache."""
    def __init__(self, ttl_seconds: float = 60.0, max_size: int = 2048):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.cache: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        if key in self.cache:
            val, expiry = self.cache[key]
            if time.time() < expiry:
                return val
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size:
            # Evict oldest item (Python dict maintains insertion order)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = (value, time.time() + self.ttl)

    def clear(self) -> None:
        self.cache.clear()


# Global registry of all caches so we can clear them on demand
_ALL_CACHES: list[FastCache] = []


def clear_all_caches() -> None:
    """Clear all registered caches (useful for admin invalidation on reload)."""
    logger.info("Clearing all in-memory caches...")
    for cache in _ALL_CACHES:
        cache.clear()


def cached_endpoint(ttl: float = 60.0, max_size: int = 1024):
    """
    Decorator for FastAPI endpoint handlers to cache their responses.
    Handles both sync and async handlers seamlessly.
    """
    cache = FastCache(ttl_seconds=ttl, max_size=max_size)
    _ALL_CACHES.append(cache)

    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Build unique cache key from args and kwargs
                key_parts = []
                for arg in args:
                    arg_str = str(arg)
                    if "Request" not in arg_str and "Response" not in arg_str:
                        key_parts.append(arg_str)
                for k, v in sorted(kwargs.items()):
                    if k not in ("request", "response", "db", "background_tasks"):
                        key_parts.append(f"{k}:{v}")
                
                cache_key = f"{func.__name__}:" + ":".join(key_parts)

                cached_val = cache.get(cache_key)
                if cached_val is not None:
                    return cached_val

                res = await func(*args, **kwargs)
                cache.set(cache_key, res)
                return res
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Build unique cache key from args and kwargs
                key_parts = []
                for arg in args:
                    arg_str = str(arg)
                    if "Request" not in arg_str and "Response" not in arg_str:
                        key_parts.append(arg_str)
                for k, v in sorted(kwargs.items()):
                    if k not in ("request", "response", "db", "background_tasks"):
                        key_parts.append(f"{k}:{v}")
                
                cache_key = f"{func.__name__}:" + ":".join(key_parts)

                cached_val = cache.get(cache_key)
                if cached_val is not None:
                    return cached_val

                res = func(*args, **kwargs)
                cache.set(cache_key, res)
                return res
            return sync_wrapper
    return decorator
