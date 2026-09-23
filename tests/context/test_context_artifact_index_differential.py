"""Differential test: incremental reconcile against a full-recompute oracle.

Randomized batches mutate a real session's active support set (including
same-apply assertion/retraction pairs), register immutable content versions
of every validity category, and sometimes skip alignment entirely. Every
reconcile is then compared against an independent oracle that ignores the
index's reverse indices and eligibility cache and simply recomputes
eligibility from the full registry and the authoritative view.
"""

import random

from semantica.context import ContextArtifactIndex, TruthSnapshotProvider
from semantica.reasoning import FactSupport, TruthMaintenanceSession
from tests.context.grounded_helpers import artifact

NAMESPACE = "differential-rag"
SUPPORT_FACTS = {"s1": "A(x)", "s2": "A(x)", "s3": "B(x)"}
SEEDS = tuple(range(12))
BATCHES = 30
REGISTER_CHANCE = 0.3
SKIP_CHANCE = 0.25


def full_validity(registry, view):
    """Recompute eligibility from every immutable record, ignoring the index."""
    records = {a.artifact_id: a for a in registry.artifacts}
    support_ids = {s.support_id for s in view.active_supports}

    def own_valid(record):
        return (
            record.built_from.namespace == view.stamp.namespace
            and record.built_from.source_kind == view.stamp.source_kind
            and record.dependencies.required_facts <= view.facts
            and record.dependencies.required_support_ids <= support_ids
            and (
                record.validity_policy == "dependencies"
                or record.built_from == view.stamp
            )
        )

    return {
        key: own_valid(record)
        and all(own_valid(records[c]) for c in record.citation_ids)
        for key, record in records.items()
    }


class Harness:
    """One randomized timeline of support mutations and registrations."""

    def __init__(self, rng):
        self.rng = rng
        self.session = TruthMaintenanceSession(rules=[])
        self.provider = TruthSnapshotProvider(self.session, namespace=NAMESPACE)
        self.index = ContextArtifactIndex(namespace=NAMESPACE)
        self.session.apply(
            assertions=[
                FactSupport(support, fact) for support, fact in SUPPORT_FACTS.items()
            ]
        )
        view = self.provider.capture()
        citation_a = artifact(
            view,
            "cite-s1",
            content="Support s1 asserts A(x).",
            kind="citation",
            facts=(),
            supports=("s1",),
            policy="dependencies",
        )
        citation_b = artifact(
            view,
            "cite-s3",
            content="Support s3 asserts B(x).",
            kind="citation",
            facts=(),
            supports=("s3",),
            policy="dependencies",
        )
        fact_only_a = artifact(
            view,
            "fact-a",
            content="A(x) holds.",
            facts=("A(x)",),
            policy="dependencies",
        )
        fact_only_b = artifact(
            view,
            "fact-b",
            content="B(x) holds.",
            facts=("B(x)",),
            policy="dependencies",
        )
        bound = artifact(
            view,
            "bound-a",
            content="A(x) holds per the cited record.",
            facts=("A(x)",),
            citations=("cite-s1",),
            policy="dependencies",
        )
        pinned = artifact(
            view,
            "pinned-a",
            content="A(x) held when this was captured.",
            facts=("A(x)",),
            policy="snapshot",
        )
        for record in (
            citation_a,
            citation_b,
            fact_only_a,
            fact_only_b,
            bound,
            pinned,
        ):
            self.index.register(record)
        self.believed = full_validity(self.index.snapshot(), view)
        self.active = set(SUPPORT_FACTS)
        self.versions = 0

    def step(self):
        """Run one batch: mutate, maybe register, maybe skip alignment."""
        self._mutate_supports()
        self._maybe_register()
        if self.rng.random() < SKIP_CHANCE:
            return
        self._reconcile_and_compare()

    def _mutate_supports(self):
        chosen = {s for s in SUPPORT_FACTS if self.rng.random() < 0.5}
        assertions = [
            FactSupport(s, SUPPORT_FACTS[s]) for s in sorted(chosen - self.active)
        ]
        retractions = sorted(self.active - chosen)
        self.session.apply(assertions=assertions, retractions=retractions)
        self.active = chosen

    def _maybe_register(self):
        if self.rng.random() >= REGISTER_CHANCE:
            return
        self.versions += 1
        view = self.provider.capture()
        template = self.rng.choice(("fact", "bound", "pinned"))
        if template == "fact":
            record = artifact(
                view,
                f"fact-a-v{self.versions}",
                content=f"A(x) holds, version {self.versions}.",
                facts=("A(x)",),
                policy="dependencies",
            )
        elif template == "bound":
            record = artifact(
                view,
                f"bound-a-v{self.versions}",
                content=f"A(x) holds per record, version {self.versions}.",
                facts=("A(x)",),
                citations=("cite-s1",),
                policy="dependencies",
            )
        else:
            record = artifact(
                view,
                f"pinned-a-v{self.versions}",
                content=f"A(x) held, version {self.versions}.",
                facts=("A(x)",),
                policy="snapshot",
            )
        # A never-reconciled id is treated as previously eligible by the
        # index, so the oracle baseline for it is True as well.
        self.index.register(record)

    def _reconcile_and_compare(self):
        view = self.provider.capture()
        oracle = full_validity(self.index.snapshot(), view)
        report = self.index.reconcile(view)

        invalidated = sorted(
            key
            for key, valid in oracle.items()
            if not valid and self.believed.get(key, True)
        )
        reactivated = sorted(
            key
            for key, valid in oracle.items()
            if valid and not self.believed.get(key, True)
        )
        assert list(report.invalidated_ids) == invalidated
        assert list(report.reactivated_ids) == reactivated

        # Aligning the same view again must not report any further change.
        repeat = self.index.reconcile(self.provider.capture())
        assert repeat.invalidated_ids == ()
        assert repeat.reactivated_ids == ()

        self.believed = oracle


def test_incremental_reconcile_matches_a_full_recompute_oracle():
    for seed in SEEDS:
        rng = random.Random(seed)
        harness = Harness(rng)
        for batch in range(1, BATCHES + 1):
            try:
                harness.step()
            except AssertionError as exc:
                raise AssertionError(f"seed={seed} batch={batch}: {exc}") from exc
