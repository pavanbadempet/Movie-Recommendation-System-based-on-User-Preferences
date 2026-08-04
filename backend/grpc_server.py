"""Async gRPC Server for APEX High-Throughput Recommendation Engine."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
import uuid

import grpc

from backend.pipeline.recommender import get_recommender
from backend.proto import recommendation_pb2 as pb2
from backend.proto import recommendation_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


def _dict_to_movie_item(movie: dict[str, Any]) -> pb2.MovieItem:
    """Convert movie dict to Protobuf MovieItem message."""
    explanations = movie.get("explanation") or []
    if isinstance(explanations, str):
        explanations = [explanations]

    return pb2.MovieItem(
        id=int(movie.get("id", 0)),
        title=str(movie.get("title", "")),
        genres=str(movie.get("genres", "")),
        similarity_score=float(movie.get("similarity_score") or 0.0),
        poster_path=str(movie.get("poster_path") or ""),
        release_date=str(movie.get("release_date") or ""),
        vote_average=float(movie.get("vote_average") or 0.0),
        retrieval_stage=str(movie.get("retrieval_stage") or "gRPC"),
        explanation=[str(e) for e in explanations],
    )


class RecommendationServiceServicer(pb2_grpc.RecommendationServiceServicer):
    """gRPC Servicer implementation for APEX Recommendation Service."""

    def __init__(self):
        self._recommender = None

    def _get_engine(self):
        if self._recommender is None:
            self._recommender = get_recommender()
        return self._recommender

    async def GetRecommendations(
        self, request: pb2.RecommendationRequest, context: grpc.aio.ServicerContext
    ) -> pb2.RecommendationResponse:
        """Retrieve ranked recommendations for a movie ID or session."""
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())

        try:
            rec_engine = await asyncio.to_thread(self._get_engine)
            movie_id = request.movie_id
            limit = request.limit if request.limit > 0 else 10

            query_movie = await asyncio.to_thread(rec_engine.get_movie_by_id, movie_id)
            if query_movie is None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Movie with ID {movie_id} not found")
                return pb2.RecommendationResponse()

            recs = await asyncio.to_thread(rec_engine.recommend_by_id, movie_id, n=limit)
            rec_items = [_dict_to_movie_item(m) for m in recs]
            query_item = _dict_to_movie_item(query_movie)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            return pb2.RecommendationResponse(
                request_id=request_id,
                query_movie=query_item,
                recommendations=rec_items,
                latency_ms=latency_ms,
                serving_tier="gRPC-HNSW",
            )
        except Exception as exc:
            logger.exception("gRPC GetRecommendations failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return pb2.RecommendationResponse()

    async def SearchCatalog(self, request: pb2.SearchRequest, context: grpc.aio.ServicerContext) -> pb2.SearchResponse:
        """Search catalog items via vector embeddings / semantic query."""
        start_time = time.perf_counter()

        try:
            rec_engine = await asyncio.to_thread(self._get_engine)
            query = request.query
            limit = request.limit if request.limit > 0 else 10

            if not query.strip():
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Search query cannot be empty")
                return pb2.SearchResponse()

            results = await asyncio.to_thread(rec_engine.search_by_title, query, top_n=limit)
            result_items = [_dict_to_movie_item(m) for m in results]
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            return pb2.SearchResponse(
                results=result_items,
                total_count=len(result_items),
                latency_ms=latency_ms,
            )
        except Exception as exc:
            logger.exception("gRPC SearchCatalog failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return pb2.SearchResponse()

    async def StreamEvents(
        self,
        request_iterator: Any,
        context: grpc.aio.ServicerContext,
    ) -> pb2.EventResponse:
        """Stream real-time behavior telemetry events."""
        event_count = 0
        try:
            async for event in request_iterator:
                event_count += 1
                logger.debug(f"Received gRPC telemetry event: {event.event_type} from user {event.user_id}")

            return pb2.EventResponse(
                success=True,
                message=f"Successfully ingested {event_count} telemetry events over gRPC stream",
            )
        except Exception as exc:
            logger.exception("gRPC StreamEvents failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return pb2.EventResponse(success=False, message=str(exc))


async def serve_grpc(host: str = "0.0.0.0", port: int = 50051) -> grpc.aio.Server:
    """Initialize and start the async gRPC server."""
    server = grpc.aio.server()
    pb2_grpc.add_RecommendationServiceServicer_to_server(RecommendationServiceServicer(), server)
    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info(f"Starting APEX gRPC Recommendation Server listening on {listen_addr}")
    await server.start()
    return server
