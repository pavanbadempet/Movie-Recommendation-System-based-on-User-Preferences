"""Tests for high-throughput gRPC Recommendation Service."""

from __future__ import annotations

import asyncio
import pytest
import grpc

from backend.grpc_server import serve_grpc
from backend.proto import recommendation_pb2 as pb2
from backend.proto import recommendation_pb2_grpc as pb2_grpc


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_grpc_server_health_and_recommendation():
    """Test starting gRPC server on port 50055 and querying recommendations."""
    port = 50055
    server = await serve_grpc(host="127.0.0.1", port=port)

    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = pb2_grpc.RecommendationServiceStub(channel)

    try:
        # Test GetRecommendations
        req = pb2.RecommendationRequest(movie_id=550, limit=5)
        response = await stub.GetRecommendations(req)

        assert response.request_id != ""
        assert response.serving_tier == "gRPC-HNSW"
        assert response.query_movie.id == 550
        assert len(response.recommendations) > 0
        assert response.latency_ms > 0.0

        # Test SearchCatalog
        search_req = pb2.SearchRequest(query="Fight Club", limit=5)
        search_res = await stub.SearchCatalog(search_req)
        assert search_res.total_count >= 1
        assert len(search_res.results) >= 1

    finally:
        await channel.close()
        await server.stop(grace=1.0)
