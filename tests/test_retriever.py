import pytest
from src.retriever import MusicRetriever


def test_retriever_returns_results():
    """Retriever should return results for a valid query."""
    retriever = MusicRetriever()
    results = retriever.retrieve("happy pop music", top_k=5)
    assert len(results) > 0


def test_retriever_returns_correct_count():
    """Retriever should return exactly top_k results."""
    retriever = MusicRetriever()
    results = retriever.retrieve("chill lofi beats", top_k=3)
    assert len(results) == 3


def test_retriever_results_have_required_fields():
    """Each result should have title, artist, genre, mood, and retrieval_score."""
    retriever = MusicRetriever()
    results = retriever.retrieve("acoustic guitar music", top_k=3)
    for r in results:
        assert "title" in r
        assert "artist" in r
        assert "genre" in r
        assert "mood" in r
        assert "retrieval_score" in r


def test_retriever_scores_are_between_0_and_1():
    """All retrieval scores should be between 0.0 and 1.0."""
    retriever = MusicRetriever()
    results = retriever.retrieve("intense rock music", top_k=5)
    for r in results:
        assert 0.0 <= r["retrieval_score"] <= 1.0


def test_retriever_top_result_is_most_relevant():
    """First result should have the highest retrieval score."""
    retriever = MusicRetriever()
    results = retriever.retrieve("upbeat happy pop dancing", top_k=5)
    scores = [r["retrieval_score"] for r in results]
    assert scores[0] == max(scores)


def test_retrieve_by_profile_pop_happy():
    """Profile-based retrieval should return pop songs for a pop profile."""
    retriever = MusicRetriever()
    results = retriever.retrieve_by_profile("pop", "happy", top_k=5)
    genres = [r["genre"] for r in results]
    assert "pop" in genres


def test_retrieve_by_profile_rock():
    """Profile-based retrieval should return rock songs for a rock profile."""
    retriever = MusicRetriever()
    results = retriever.retrieve_by_profile("rock", "intense", top_k=5)
    genres = [r["genre"] for r in results]
    assert "rock" in genres


def test_retriever_empty_query_still_returns_results():
    """Even a vague query should return some results."""
    retriever = MusicRetriever()
    results = retriever.retrieve("music", top_k=3)
    assert len(results) == 3