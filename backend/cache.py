"""
backend/cache.py

Provides the AsyncLRUCache decorator for caching results of async functions.
Uses an OrderedDict to maintain insertion order and evict the least-recently-used
entry when the cache exceeds its configured maximum size.
"""

from collections import OrderedDict
from functools import wraps


class AsyncLRUCache:
    def __init__(self, maxsize=1000):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            result = await func(*args, **kwargs)
            if result is not None:
                self.cache[key] = result
                if len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)
            return result

        return wrapper
