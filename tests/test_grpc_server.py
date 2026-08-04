"""Tests for high-throughput gRPC Recommendation Service."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import grpc
import pytest

from backend.grpc_server import serve_grpc
from backend.proto import recommendation_pb2 as pb2
from backend.proto import recommendation_pb2_grpc as pb2_grpc


def create_mock_recommender():
    mock_rec = MagicMock()
    sample_movie = {
        "id": 550,
        "title": "Fight Club",
        "genres": "Drama, Thriller",
        "vote_average": 8.4,
        "release_date": "1999-10-15",
        "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
        "retrieval_stage": "FAISS-HNSW",
        "explanation": ["Shares psychological thriller vector cluster."],
    }
    rec_movie = {
        "id": 680,
        "title": "Pulp Fiction",
        "genres": "Crime, Drama",
        "vote_average": 8.5,
        "release_date": "1994-09-10",
        "poster_path": "/d5iIlFn5s0ImszYzBPb8Su121io.jpg",
        "similarity_score": 0.92,
        "retrieval_stage": "LightGCN",
        "explanation": ["High graph co-occurrence score."],
    }
    mock_rec.get_movie_by_id.side_effect = lambda mid: sample_movie if mid == 550 else None
    mock_rec.recommend_by_id.return_value = [rec_movie]
    mock_rec.search_by_title.return_value = [sample_movie]
    return mock_rec


@pytest.mark.asyncio
async def test_grpc_server_health_and_recommendation():
    """Test starting gRPC server on port 50055 and querying recommendations."""
    port = 50055
    mock_rec = create_mock_recommender()

    with patch("backend.grpc_server.get_recommender", return_value=mock_rec):
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
            assert len(response.recommendations) == 1
            assert response.recommendations[0].id == 680
            assert response.latency_ms > 0.0

            # Test SearchCatalog
            search_req = pb2.SearchRequest(query="Fight Club", limit=5)
            search_res = await stub.SearchCatalog(search_req)
            assert search_res.total_count == 1
            assert search_res.results[0].title == "Fight Club"

        finally:
            await channel.close()
            await server.stop(grace=1.0)
