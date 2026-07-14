import asyncio

import pytest

from backend.api.fast_cache import FastCache, cached_endpoint, clear_all_caches


@pytest.mark.asyncio
async def test_fast_cache_ttl_and_eviction():
    cache = FastCache(ttl_seconds=0.2, max_size=2)
    cache.set("key1", "val1")
    cache.set("key2", "val2")

    # Verify retrieval
    assert cache.get("key1") == "val1"

    # Verify eviction (key1 and key2 are in cache, setting key3 should evict key1)
    cache.set("key3", "val3")
    assert cache.get("key1") is None
    assert cache.get("key2") == "val2"
    assert cache.get("key3") == "val3"

    # Verify TTL expiry
    await asyncio.sleep(0.25)
    assert cache.get("key2") is None
    assert cache.get("key3") is None


@pytest.mark.asyncio
async def test_cached_endpoint_decorator():
    call_count = 0

    @cached_endpoint(ttl=1.0)
    async def get_val(param: int):
        nonlocal call_count
        call_count += 1
        return f"result-{param}"

    # First call: executes function
    res1 = await get_val(10)
    assert res1 == "result-10"
    assert call_count == 1

    # Second call: cached
    res2 = await get_val(10)
    assert res2 == "result-10"
    assert call_count == 1  # Call count should not increase

    # Different parameter: executes function
    res3 = await get_val(20)
    assert res3 == "result-20"
    assert call_count == 2

    # Clear caches
    clear_all_caches()
    res4 = await get_val(10)
    assert res4 == "result-10"
    assert call_count == 3  # Executed again because cache was cleared
