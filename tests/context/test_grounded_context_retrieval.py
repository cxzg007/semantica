"""Grounded retrieval coordination: filtering before ranking and version gating."""

from copy import deepcopy

import pytest
from semantica.context import (
    ContextRetriever,
    RetrievedContext,
    TruthMaintenanceContextFilter,
)
from semantica.reasoning import FactSupport, TruthMaintenanceSession
from semantica.utils.exceptions import ProcessingError, ValidationError

SESSION_ID = "employment-session-1"


class StaticVectorStore:
    def __init__(self, rows):
        self.rows = deepcopy(rows)
        self.calls = 0

    def search(self, *, query, limit):
        self.calls += 1
        return deepcopy(self.rows[:limit])


def row(identifier, fact, score, text):
    return {
        "id": identifier,
        "score": score,
        "content": text,
        "metadata": {
            "truth_maintenance": {
                "schema_version": 1,
                "session_id": SESSION_ID,
                "required_facts": [fact],
                "required_support_ids": [],
            }
        },
    }


def make_session(*rules):
    return TruthMaintenanceSession(rules=list(rules))


def make_gate(session):
    return TruthMaintenanceContextFilter(session, session_id=SESSION_ID)


def test_invalid_candidate_is_removed_before_top_k():
    session = make_session()
    session.apply(assertions=[FactSupport("live", "Current(x)")])
    store = StaticVectorStore([
        row("stale", "Old(x)", 0.99, "stale assertion"),
        row("live", "Current(x)", 0.6, "current assertion"),
    ])
    gate = make_gate(session)
    retriever = ContextRetriever(vector_store=store, use_graph_expansion=False)
    result = retriever.retrieve("assertion", max_results=1, truth_filter=gate)
    assert [r.content for r in result] == ["current assertion"]
    assert len(store.rows) == 2
    assert (
        result[0].metadata["truth_maintenance_validation"]["version"]
        == session.version
    )


def test_duplicate_text_prefixes_are_both_kept():
    session = make_session()
    session.apply(assertions=[FactSupport("s1", "A(x)")])
    shared = "same first hundred characters " * 4
    store = StaticVectorStore([
        row("one", "A(x)", 0.8, shared + " tail one"),
        row("two", "A(x)", 0.7, shared + " tail two"),
    ])
    gate = make_gate(session)
    retriever = ContextRetriever(vector_store=store, use_graph_expansion=False)
    result = retriever.retrieve("query", max_results=5, truth_filter=gate)
    assert sorted(r.content[-8:] for r in result) == ["tail one", "tail two"]


def test_same_graph_node_id_candidates_are_not_merged():
    session = make_session()
    session.apply(assertions=[FactSupport("s1", "A(x)")])
    gate = make_gate(session)
    retriever = ContextRetriever(use_graph_expansion=False)

    def graph_candidate(node_id, extra_metadata, entities):
        return gate.filter_contexts(
            [RetrievedContext(
                content=f"node {node_id}",
                score=0.8,
                source=f"graph:{node_id}",
                metadata={
                    "node_id": node_id,
                    "truth_maintenance": {
                        "schema_version": 1,
                        "session_id": SESSION_ID,
                        "required_facts": ["A(x)"],
                        "required_support_ids": [],
                    },
                    **extra_metadata,
                },
                related_entities=entities,
            )],
            snapshot=gate.snapshot(),
        )[0]

    first = graph_candidate("n1", {"marker": "first"}, [{"id": "e1", "metadata": {
        "truth_maintenance": {
            "schema_version": 1,
            "session_id": SESSION_ID,
            "required_facts": [],
            "required_support_ids": ["s1"],
        }
    }}])
    second = graph_candidate("n1", {"marker": "second"}, [{"id": "e2", "metadata": {
        "truth_maintenance": {
            "schema_version": 1,
            "session_id": SESSION_ID,
            "required_facts": [],
            "required_support_ids": ["s1"],
        }
    }}])

    merged = retriever._rank_and_merge(
        [first, second], "query", merge_duplicates=False
    )
    assert len(merged) == 2
    markers = sorted(c.metadata["marker"] for c in merged)
    assert markers == ["first", "second"]
    entity_ids = {e["id"] for c in merged for e in c.related_entities}
    assert entity_ids == {"e1", "e2"}


def test_default_merge_duplicates_keeps_original_dedup():
    retriever = ContextRetriever(use_graph_expansion=False)
    first = RetrievedContext(
        content="shared content", score=0.9, source="vector:a"
    )
    second = RetrievedContext(
        content="shared content", score=0.5, source="vector:b"
    )
    merged = retriever._rank_and_merge([first, second], "query")
    assert len(merged) == 1


class MutatingVectorStore(StaticVectorStore):
    """Vector store whose search mutates the session mid-retrieval."""

    def __init__(self, rows, session, on_search):
        super().__init__(rows)
        self.session = session
        self.on_search = on_search

    def search(self, *, query, limit):
        results = super().search(query=query, limit=limit)
        self.on_search(self.session)
        return results


def test_fact_change_during_search_raises_processing_error():
    session = make_session()
    session.apply(assertions=[FactSupport("s1", "A(x)")])
    store = MutatingVectorStore(
        [row("live", "A(x)", 0.9, "live")],
        session,
        lambda s: s.apply(retractions=["s1"]),
    )
    gate = make_gate(session)
    retriever = ContextRetriever(vector_store=store, use_graph_expansion=False)
    with pytest.raises(ProcessingError):
        retriever.retrieve("query", max_results=5, truth_filter=gate)


def test_support_only_change_during_search_raises_processing_error():
    session = make_session()
    session.apply(assertions=[FactSupport("s1", "A(x)")])
    store = MutatingVectorStore(
        [row("live", "A(x)", 0.9, "live")],
        session,
        lambda s: s.apply(
            assertions=[FactSupport("s2", "A(x)")], retractions=["s1"]
        ),
    )
    gate = make_gate(session)
    retriever = ContextRetriever(vector_store=store, use_graph_expansion=False)
    with pytest.raises(ProcessingError):
        retriever.retrieve("query", max_results=5, truth_filter=gate)


def test_noop_apply_during_search_succeeds():
    session = make_session()
    session.apply(assertions=[FactSupport("s1", "A(x)")])
    store = MutatingVectorStore(
        [row("live", "A(x)", 0.9, "live")],
        session,
        lambda s: s.apply(assertions=[]),
    )
    gate = make_gate(session)
    retriever = ContextRetriever(vector_store=store, use_graph_expansion=False)
    result = retriever.retrieve("query", max_results=5, truth_filter=gate)
    assert [r.content for r in result] == ["live"]


def test_failed_apply_during_search_does_not_fail_retrieval():
    session = make_session()
    session.apply(assertions=[FactSupport("s1", "A(x)")])

    def failed_apply(s):
        with pytest.raises(ValidationError):
            s.apply(assertions=[FactSupport("bad", "NotAFact")])

    store = MutatingVectorStore(
        [row("live", "A(x)", 0.9, "live")], session, failed_apply
    )
    gate = make_gate(session)
    retriever = ContextRetriever(vector_store=store, use_graph_expansion=False)
    result = retriever.retrieve("query", max_results=5, truth_filter=gate)
    assert [r.content for r in result] == ["live"]


def test_rerank_embed_callback_change_raises_processing_error():
    session = make_session()
    session.apply(assertions=[FactSupport("s1", "A(x)")])

    class EmbeddingMutatingStore(StaticVectorStore):
        def embed(self, text):
            session.apply(retractions=["s1"])
            return [0.1, 0.2]

    store = EmbeddingMutatingStore([row("live", "A(x)", 0.9, "live")])
    gate = make_gate(session)
    retriever = ContextRetriever(vector_store=store, use_graph_expansion=False)
    with pytest.raises(ProcessingError):
        retriever.retrieve("query", max_results=5, truth_filter=gate)


def test_invalid_truth_filter_type_raises_validation_error():
    store = StaticVectorStore([])
    retriever = ContextRetriever(vector_store=store, use_graph_expansion=False)
    with pytest.raises(ValidationError):
        retriever.retrieve("query", max_results=5, truth_filter="gate")


def test_filter_none_matches_default_behavior():
    rows = [
        {"id": "a", "score": 0.9, "content": "first", "metadata": {}},
        {"id": "b", "score": 0.4, "content": "second", "metadata": {}},
    ]
    default_results = ContextRetriever(
        vector_store=StaticVectorStore(rows), use_graph_expansion=False
    ).retrieve("query", max_results=5)
    none_results = ContextRetriever(
        vector_store=StaticVectorStore(rows), use_graph_expansion=False
    ).retrieve("query", max_results=5, truth_filter=None)
    assert [r.content for r in default_results] == [r.content for r in none_results]


def test_all_invalid_candidates_return_empty_without_recall():
    session = make_session()
    store = StaticVectorStore([
        row("stale1", "Old(x)", 0.99, "stale one"),
        row("stale2", "Gone(x)", 0.9, "stale two"),
    ])
    gate = make_gate(session)
    retriever = ContextRetriever(vector_store=store, use_graph_expansion=False)
    assert retriever.retrieve("query", max_results=5, truth_filter=gate) == []
    assert store.calls == 1


def test_old_validation_stamp_is_overwritten_by_current_check():
    session = make_session()
    session.apply(assertions=[FactSupport("s1", "A(x)")])
    session.apply(retractions=["s1"])
    session.apply(assertions=[FactSupport("s2", "A(x)")])
    stale_row = row("live", "A(x)", 0.9, "live")
    stale_row["metadata"]["truth_maintenance_validation"] = {
        "session_id": "someone-else",
        "version": 99,
    }
    store = StaticVectorStore([stale_row])
    gate = make_gate(session)
    retriever = ContextRetriever(vector_store=store, use_graph_expansion=False)
    result = retriever.retrieve("query", max_results=5, truth_filter=gate)
    assert result[0].metadata["truth_maintenance_validation"] == {
        "session_id": SESSION_ID,
        "version": session.version,
    }


def test_empty_store_returns_empty_list():
    session = make_session()
    store = StaticVectorStore([])
    gate = make_gate(session)
    retriever = ContextRetriever(vector_store=store, use_graph_expansion=False)
    assert retriever.retrieve("query", max_results=5, truth_filter=gate) == []