import pytest

from backend.serving.cache import AsyncLRUCache


@pytest.mark.asyncio
async def test_async_lru_cache_basic():
    call_count = 0

    @AsyncLRUCache(maxsize=2)
    async def fetch_data(x):
        nonlocal call_count
        call_count += 1
        return f"data-{x}"

    # First call: cache miss, increments count
    r1 = await fetch_data(1)
    assert r1 == "data-1"
    assert call_count == 1

    # Second call: cache hit, same count
    r2 = await fetch_data(1)
    assert r2 == "data-1"
    assert call_count == 1

    # Call with new arg: cache miss, increments count
    r3 = await fetch_data(2)
    assert r3 == "data-2"
    assert call_count == 2

    # Call with 3rd arg: evicts key 1 (since maxsize=2)
    r4 = await fetch_data(3)
    assert r4 == "data-3"
    assert call_count == 3

    # Call with key 1 again: cache miss (was evicted), increments count
    r5 = await fetch_data(1)
    assert r5 == "data-1"
    assert call_count == 4
