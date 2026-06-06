"""Deterministic semantic item twins for recommendation quality.

This module is intentionally model-free and cheap to run in a free-tier API.
It gives the recommender structured signals that sit between plain genres and
heavy LLM enrichment: concepts, emotional arcs, viewer jobs, confidence, and
risk tags.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "and",
    "are",
    "around",
    "back",
    "become",
    "becomes",
    "been",
    "before",
    "being",
    "between",
    "but",
    "can",
    "city",
    "during",
    "each",
    "find",
    "finds",
    "for",
    "from",
    "has",
    "have",
    "her",
    "him",
    "his",
    "into",
    "its",
    "life",
    "lives",
    "man",
    "movie",
    "must",
    "new",
    "one",
    "only",
    "out",
    "own",
    "set",
    "she",
    "that",
    "the",
    "their",
    "them",
    "then",
    "they",
    "this",
    "through",
    "two",
    "when",
    "where",
    "who",
    "with",
    "years",
}


EMOTIONAL_ARC_LEXICON = {
    "adventure": {"adventure", "journey", "quest", "mission", "explore", "expedition"},
    "wonder": {"wonder", "magical", "mystical", "planet", "space", "future", "fantasy", "dream"},
    "survival": {"survival", "survive", "stranded", "escape", "disaster", "apocalypse", "wilderness"},
    "tension": {"war", "battle", "conflict", "enemy", "threat", "danger", "conspiracy", "chase"},
    "mystery": {"mystery", "detective", "secret", "investigation", "hidden", "murder", "missing"},
    "humor": {"comedy", "funny", "comic", "hilarious", "satire", "parody"},
    "romance": {"romance", "love", "relationship", "marriage", "heart", "couple"},
    "melancholy": {"grief", "loss", "lonely", "memory", "past", "regret", "death"},
    "fear": {"horror", "terror", "haunted", "monster", "demon", "nightmare", "killer"},
    "heroism": {"hero", "save", "protect", "rescue", "justice", "legend", "chosen"},
}


VIEWER_JOB_LEXICON = {
    "escape_and_spectacle": {"adventure", "action", "space", "fantasy", "epic", "battle", "planet"},
    "world_immersion": {"world", "kingdom", "planet", "civilization", "future", "myth", "universe"},
    "intellectual_puzzle": {"mystery", "detective", "investigation", "conspiracy", "mind", "memory"},
    "emotional_catharsis": {"family", "love", "loss", "friendship", "grief", "home"},
    "adrenaline": {"chase", "fight", "war", "mission", "crime", "revenge", "explosion"},
    "comfort_laughs": {"comedy", "funny", "family", "romance", "friendship"},
    "dark_thrill": {"horror", "thriller", "killer", "haunted", "terror", "danger"},
}


RISK_KEYWORDS = {
    "documentary_spinoff": {"making", "behind", "documentary", "interview"},
    "franchise_saturation": {"sequel", "prequel", "reboot", "superhero", "mutant", "justice", "league"},
    "thin_metadata": set(),
    "low_confidence": set(),
}


@dataclass(frozen=True)
class SemanticAffinity:
    """Structured semantic-twin comparison result."""

    score: float
    concept_overlap: float
    genre_overlap: float
    emotional_overlap: float
    viewer_job_overlap: float
    risk_penalty: float
    shared_concepts: list[str]
    shared_emotional_arcs: list[str]
    shared_viewer_jobs: list[str]
    reasons: list[str]
    cautions: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "concept_overlap": round(self.concept_overlap, 4),
            "genre_overlap": round(self.genre_overlap, 4),
            "emotional_overlap": round(self.emotional_overlap, 4),
            "viewer_job_overlap": round(self.viewer_job_overlap, 4),
            "risk_penalty": round(self.risk_penalty, 4),
            "shared_concepts": self.shared_concepts,
            "shared_emotional_arcs": self.shared_emotional_arcs,
            "shared_viewer_jobs": self.shared_viewer_jobs,
            "reasons": self.reasons,
            "cautions": self.cautions,
        }


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def tokenize(text: str) -> list[str]:
    """Extract stable content tokens from movie text."""
    tokens = re.findall(r"[a-z][a-z0-9]{2,}", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) >= 3]


def parse_genres(value: Any) -> list[str]:
    """Parse a comma/string/list genre field into normalized labels."""
    if isinstance(value, list):
        raw = value
    else:
        raw = re.split(r"[,|;/]", _safe_text(value))
    genres = []
    for item in raw:
        genre = str(item).strip().lower()
        if genre:
            genres.append(genre)
    return sorted(set(genres))


def _top_concepts(movie: dict[str, Any], limit: int = 14) -> list[str]:
    text_parts = [
        _safe_text(movie.get("title")),
        _safe_text(movie.get("tagline")),
        _safe_text(movie.get("overview")),
        _safe_text(movie.get("genres")),
        _safe_text(movie.get("director")),
    ]
    if movie.get("keywords"):
        text_parts.append(_safe_text(movie.get("keywords")))
    text = " ".join(text_parts)
    tokens = tokenize(text)
    if not tokens:
        return []

    title_tokens = set(tokenize(_safe_text(movie.get("title"))))
    genre_tokens = set(tokenize(_safe_text(movie.get("genres"))))
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
        if token in title_tokens:
            counts[token] += 1.25
        if token in genre_tokens:
            counts[token] += 0.75
    return [token for token, _score in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _labels_from_lexicon(tokens: set[str], lexicon: dict[str, set[str]]) -> list[str]:
    labels = []
    for label, keywords in lexicon.items():
        if tokens & keywords:
            labels.append(label)
    return labels


def _risk_tags(movie: dict[str, Any], concepts: set[str], genres: set[str]) -> list[str]:
    risks = []
    overview_len = len(_safe_text(movie.get("overview")).strip())
    title_len = len(_safe_text(movie.get("title")).strip())
    try:
        vote_count = float(movie.get("vote_count") or 0)
    except (TypeError, ValueError):
        vote_count = 0.0

    if title_len == 0 or overview_len < 30:
        risks.append("thin_metadata")
    if vote_count < 25:
        risks.append("low_confidence")
    if "documentary" in genres or concepts & RISK_KEYWORDS["documentary_spinoff"]:
        risks.append("documentary_spinoff")
    if concepts & RISK_KEYWORDS["franchise_saturation"]:
        risks.append("franchise_saturation")
    return sorted(set(risks))


def build_semantic_twin(movie: dict[str, Any]) -> dict[str, Any]:
    """Build a model-free semantic item twin from catalog metadata."""
    genres = parse_genres(movie.get("genres"))
    concepts = _top_concepts(movie)
    concept_set = set(concepts)
    genre_set = set(genres)
    all_tokens = concept_set | set(tokenize(_safe_text(movie.get("overview")))) | genre_set
    emotional_arcs = _labels_from_lexicon(all_tokens, EMOTIONAL_ARC_LEXICON)
    viewer_jobs = _labels_from_lexicon(all_tokens, VIEWER_JOB_LEXICON)
    risk_tags = _risk_tags(movie, concept_set, genre_set)

    completeness = movie.get("metadata_completeness")
    quality = movie.get("content_quality_score")
    try:
        confidence = float(quality if quality is not None else completeness if completeness is not None else 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence <= 0:
        confidence = min(1.0, 0.25 + 0.04 * len(concepts) + 0.08 * len(genres))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "item_id": movie.get("id"),
        "title": movie.get("title"),
        "genres": genres,
        "concepts": concepts,
        "emotional_arcs": emotional_arcs,
        "viewer_jobs": viewer_jobs,
        "risk_tags": risk_tags,
        "confidence": round(confidence, 4),
        "generated_by": {
            "method": "deterministic_catalog_semantic_twin",
            "version": "1.0",
            "llm_in_hot_path": False,
        },
    }


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def compare_semantic_twins(query_twin: dict[str, Any], candidate_twin: dict[str, Any]) -> SemanticAffinity:
    """Score candidate semantic affinity to a query item twin."""
    q_concepts = set(query_twin.get("concepts") or [])
    c_concepts = set(candidate_twin.get("concepts") or [])
    q_genres = set(query_twin.get("genres") or [])
    c_genres = set(candidate_twin.get("genres") or [])
    q_arcs = set(query_twin.get("emotional_arcs") or [])
    c_arcs = set(candidate_twin.get("emotional_arcs") or [])
    q_jobs = set(query_twin.get("viewer_jobs") or [])
    c_jobs = set(candidate_twin.get("viewer_jobs") or [])
    c_risks = set(candidate_twin.get("risk_tags") or [])

    concept_overlap = _overlap_ratio(q_concepts, c_concepts)
    genre_overlap = _overlap_ratio(q_genres, c_genres)
    emotional_overlap = _overlap_ratio(q_arcs, c_arcs)
    viewer_job_overlap = _overlap_ratio(q_jobs, c_jobs)

    risk_penalty = 0.0
    if "thin_metadata" in c_risks:
        risk_penalty += 0.05
    if "low_confidence" in c_risks:
        risk_penalty += 0.04
    if "documentary_spinoff" in c_risks and "documentary_spinoff" not in set(query_twin.get("risk_tags") or []):
        risk_penalty += 0.08
    if "franchise_saturation" in c_risks and concept_overlap < 0.15 and genre_overlap < 0.5:
        risk_penalty += 0.04

    confidence = float(candidate_twin.get("confidence") or 0.0)
    score = (
        0.36 * concept_overlap
        + 0.24 * genre_overlap
        + 0.17 * emotional_overlap
        + 0.15 * viewer_job_overlap
        + 0.08 * confidence
        - risk_penalty
    )
    score = max(0.0, min(1.0, score))

    shared_concepts = sorted(q_concepts & c_concepts)[:5]
    shared_arcs = sorted(q_arcs & c_arcs)[:4]
    shared_jobs = sorted(q_jobs & c_jobs)[:4]

    reasons = []
    if shared_concepts:
        reasons.append(f"Semantic twin concepts: {', '.join(shared_concepts[:3])}")
    if shared_arcs:
        reasons.append(f"Similar emotional arc: {', '.join(label.replace('_', ' ') for label in shared_arcs[:2])}")
    if shared_jobs:
        reasons.append(f"Same viewer job: {', '.join(label.replace('_', ' ') for label in shared_jobs[:2])}")

    cautions = []
    for risk in sorted(c_risks):
        if risk in {"thin_metadata", "low_confidence", "documentary_spinoff"}:
            cautions.append(risk.replace("_", " "))

    return SemanticAffinity(
        score=score,
        concept_overlap=concept_overlap,
        genre_overlap=genre_overlap,
        emotional_overlap=emotional_overlap,
        viewer_job_overlap=viewer_job_overlap,
        risk_penalty=risk_penalty,
        shared_concepts=shared_concepts,
        shared_emotional_arcs=shared_arcs,
        shared_viewer_jobs=shared_jobs,
        reasons=reasons,
        cautions=cautions,
    )
