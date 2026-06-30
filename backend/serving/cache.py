"""
backend/cache.py

Provides AsyncLRUCache and AsyncTTLCache decorators for caching results of async functions.
Uses an OrderedDict to maintain insertion order and evict the oldest or least-recently-used
entries.
"""

from collections import OrderedDict
from functools import wraps
import time


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


class AsyncTTLCache:
    def __init__(self, maxsize=1000, ttl_seconds=60):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl_seconds

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()
            if key in self.cache:
                val, expiry = self.cache[key]
                if now < expiry:
                    self.cache.move_to_end(key)
                    return val
                else:
                    del self.cache[key]
            result = await func(*args, **kwargs)
            if result is not None:
                self.cache[key] = (result, now + self.ttl)
                if len(self.cache) > self.maxsize:
                    self.cache.popitem(last=False)
            return result

        return wrapper
