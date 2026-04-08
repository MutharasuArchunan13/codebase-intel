"""Tests for the drift detector module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codebase_intel.core.config import DriftConfig
from codebase_intel.core.types import CodeAnchor, DecisionStatus, LineRange
from codebase_intel.decisions.models import Constraint, DecisionRecord
from codebase_intel.drift.detector import DriftDetector, DriftItem, DriftReport, DriftLevel


# ---------------------------------------------------------------------------
# DriftReport tests
# ---------------------------------------------------------------------------


class TestDriftReport:
    def test_empty_report(self) -> None:
        report = DriftReport()
        assert report.overall_level == DriftLevel.NONE
        assert report.summary == "No drift detected. All records are current."
        assert report.to_context_string() == ""

    def test_overall_level_takes_max(self) -> None:
        report = DriftReport(items=[
            DriftItem(component="graph", level=DriftLevel.LOW, description="a"),
            DriftItem(component="decision", level=DriftLevel.HIGH, description="b"),
            DriftItem(component="graph", level=DriftLevel.MEDIUM, description="c"),
        ])
        assert report.overall_level == DriftLevel.HIGH

    def test_critical_level(self) -> None:
        report = DriftReport(items=[
            DriftItem(component="system", level=DriftLevel.CRITICAL, description="rot"),
        ])
        assert report.overall_level == DriftLevel.CRITICAL

    def test_summary_includes_counts(self) -> None:
        report = DriftReport(items=[
            DriftItem(component="a", level=DriftLevel.LOW, description="x"),
            DriftItem(component="b", level=DriftLevel.LOW, description="y"),
            DriftItem(component="c", level=DriftLevel.HIGH, description="z"),
        ])
        assert "2 low" in report.summary
        assert "1 high" in report.summary

    def test_to_context_string_groups_by_component(self) -> None:
        report = DriftReport(items=[
            DriftItem(component="graph", level=DriftLevel.LOW, description="graph issue"),
            DriftItem(component="decision", level=DriftLevel.MEDIUM, description="decision issue"),
        ])
        ctx = report.to_context_string()
        assert "Graph" in ctx
        assert "Decision" in ctx
        assert "graph issue" in ctx
        assert "decision issue" in ctx

    def test_to_context_string_shows_rot_alert(self) -> None:
        report = DriftReport(
            items=[DriftItem(component="system", level=DriftLevel.CRITICAL, description="rot")],
            rot_detected=True,
            rot_percentage=0.45,
        )
        ctx = report.to_context_string()
        assert "CONTEXT ROT ALERT" in ctx
        assert "45%" in ctx

    def test_context_string_caps_items_per_component(self) -> None:
        items = [
            DriftItem(component="graph", level=DriftLevel.LOW, description=f"issue {i}")
            for i in range(15)
        ]
        report = DriftReport(items=items)
        ctx = report.to_context_string()
        assert "... and 5 more" in ctx


# ---------------------------------------------------------------------------
# DriftDetector tests
# ---------------------------------------------------------------------------


@pytest.fixture
def drift_config() -> DriftConfig:
    return DriftConfig(rot_threshold_pct=0.3)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path


class TestDriftDetectorDecisions:
    @pytest.fixture
    def mock_decision_store(self) -> AsyncMock:
        store = AsyncMock()
        return store

    @pytest.mark.asyncio
    async def test_stale_decision_detected(
        self, drift_config: DriftConfig, project_root: Path, mock_decision_store: AsyncMock
    ) -> None:
        past_date = datetime.now(UTC) - timedelta(days=10)
        record = DecisionRecord(
            id="DEC-001",
            title="Old decision",
            status=DecisionStatus.ACTIVE,
            context="test",
            decision="test",
            review_by=past_date,
        )
        mock_decision_store.load_all.return_value = [record]

        detector = DriftDetector(
            config=drift_config,
            project_root=project_root,
            decision_store=mock_decision_store,
        )

        report = await detector.full_check()
        stale_items = [i for i in report.items if "past" in i.description.lower() and "review" in i.description.lower()]
        assert len(stale_items) >= 1

    @pytest.mark.asyncio
    async def test_future_review_date_not_stale(
        self, drift_config: DriftConfig, project_root: Path, mock_decision_store: AsyncMock
    ) -> None:
        future_date = datetime.now(UTC) + timedelta(days=30)
        record = DecisionRecord(
            id="DEC-001",
            title="Fresh decision",
            status=DecisionStatus.ACTIVE,
            context="test",
            decision="test",
            review_by=future_date,
        )
        mock_decision_store.load_all.return_value = [record]

        detector = DriftDetector(
            config=drift_config,
            project_root=project_root,
            decision_store=mock_decision_store,
        )

        report = await detector.full_check()
        stale_items = [i for i in report.items if "past" in i.description.lower() and "review" in i.description.lower()]
        assert len(stale_items) == 0

    @pytest.mark.asyncio
    async def test_orphaned_anchor_detected(
        self, drift_config: DriftConfig, project_root: Path, mock_decision_store: AsyncMock
    ) -> None:
        record = DecisionRecord(
            id="DEC-002",
            title="Has orphaned anchor",
            status=DecisionStatus.ACTIVE,
            context="test",
            decision="test",
            code_anchors=[
                CodeAnchor(file_path=project_root / "nonexistent.py"),
            ],
        )
        mock_decision_store.load_all.return_value = [record]

        detector = DriftDetector(
            config=drift_config,
            project_root=project_root,
            decision_store=mock_decision_store,
        )

        report = await detector.full_check()
        orphan_items = [i for i in report.items if "deleted" in i.description.lower() or "orphan" in i.description.lower()]
        assert len(orphan_items) >= 1

    @pytest.mark.asyncio
    async def test_expired_constraint_detected(
        self, drift_config: DriftConfig, project_root: Path, mock_decision_store: AsyncMock
    ) -> None:
        past_date = datetime.now(UTC) - timedelta(days=5)
        record = DecisionRecord(
            id="DEC-003",
            title="Has expired constraint",
            status=DecisionStatus.ACTIVE,
            context="test",
            decision="test",
            constraints=[
                Constraint(
                    description="Temporary workaround",
                    source="technical",
                    is_hard=True,
                    expiry_date=past_date,
                ),
            ],
        )
        mock_decision_store.load_all.return_value = [record]

        detector = DriftDetector(
            config=drift_config,
            project_root=project_root,
            decision_store=mock_decision_store,
        )

        report = await detector.full_check()
        expired_items = [i for i in report.items if "expired" in i.description.lower()]
        assert len(expired_items) >= 1


class TestContextRotDetection:
    @pytest.mark.asyncio
    async def test_rot_not_triggered_with_few_decisions(
        self, drift_config: DriftConfig, project_root: Path
    ) -> None:
        """Fewer than 5 decisions should not trigger rot, even if all are stale."""
        store = AsyncMock()
        past = datetime.now(UTC) - timedelta(days=100)
        records = [
            DecisionRecord(
                id=f"DEC-{i:03d}",
                title=f"Stale {i}",
                status=DecisionStatus.ACTIVE,
                context="test",
                decision="test",
                review_by=past,
            )
            for i in range(3)
        ]
        store.load_all.return_value = records

        detector = DriftDetector(
            config=drift_config,
            project_root=project_root,
            decision_store=store,
        )

        report = await detector.full_check()
        assert report.rot_detected is False

    @pytest.mark.asyncio
    async def test_rot_triggered_above_threshold(
        self, drift_config: DriftConfig, project_root: Path
    ) -> None:
        """More than 30% stale with 5+ decisions should trigger rot."""
        store = AsyncMock()
        past = datetime.now(UTC) - timedelta(days=100)
        future = datetime.now(UTC) + timedelta(days=100)

        records = []
        # 4 stale out of 6 total = 66% > 30% threshold
        for i in range(4):
            records.append(DecisionRecord(
                id=f"DEC-{i:03d}",
                title=f"Stale {i}",
                status=DecisionStatus.ACTIVE,
                context="test",
                decision="test",
                review_by=past,
            ))
        for i in range(4, 6):
            records.append(DecisionRecord(
                id=f"DEC-{i:03d}",
                title=f"Fresh {i}",
                status=DecisionStatus.ACTIVE,
                context="test",
                decision="test",
                review_by=future,
            ))

        store.load_all.return_value = records

        detector = DriftDetector(
            config=drift_config,
            project_root=project_root,
            decision_store=store,
        )

        report = await detector.full_check()
        assert report.rot_detected is True
        assert report.rot_percentage > 0.3

    @pytest.mark.asyncio
    async def test_rot_not_triggered_below_threshold(
        self, drift_config: DriftConfig, project_root: Path
    ) -> None:
        """Less than 30% stale should not trigger rot."""
        store = AsyncMock()
        past = datetime.now(UTC) - timedelta(days=100)
        future = datetime.now(UTC) + timedelta(days=100)

        records = []
        # 1 stale out of 5 = 20% < 30%
        records.append(DecisionRecord(
            id="DEC-000",
            title="Stale",
            status=DecisionStatus.ACTIVE,
            context="test",
            decision="test",
            review_by=past,
        ))
        for i in range(1, 5):
            records.append(DecisionRecord(
                id=f"DEC-{i:03d}",
                title=f"Fresh {i}",
                status=DecisionStatus.ACTIVE,
                context="test",
                decision="test",
                review_by=future,
            ))

        store.load_all.return_value = records

        detector = DriftDetector(
            config=drift_config,
            project_root=project_root,
            decision_store=store,
        )

        report = await detector.full_check()
        assert report.rot_detected is False


class TestCheckFiles:
    @pytest.mark.asyncio
    async def test_changed_file_flags_related_decisions(
        self, drift_config: DriftConfig, project_root: Path
    ) -> None:
        """When a file changes that's anchored by a decision, flag it."""
        changed_file = project_root / "changed.py"
        changed_file.write_text("# changed")

        store = AsyncMock()
        store.load_all.return_value = [
            DecisionRecord(
                id="DEC-001",
                title="Related to changed file",
                status=DecisionStatus.ACTIVE,
                context="test",
                decision="test",
                code_anchors=[CodeAnchor(file_path=changed_file)],
            ),
        ]

        detector = DriftDetector(
            config=drift_config,
            project_root=project_root,
            decision_store=store,
        )

        report = await detector.check_files([changed_file])
        related = [i for i in report.items if "DEC-001" in i.description]
        assert len(related) >= 1

    @pytest.mark.asyncio
    async def test_unrelated_file_no_flags(
        self, drift_config: DriftConfig, project_root: Path
    ) -> None:
        """Changing a file that no decision references should produce no decision drift."""
        changed_file = project_root / "unrelated.py"
        changed_file.write_text("# unrelated")

        other_file = project_root / "other.py"
        other_file.write_text("# other")

        store = AsyncMock()
        store.load_all.return_value = [
            DecisionRecord(
                id="DEC-001",
                title="Anchored elsewhere",
                status=DecisionStatus.ACTIVE,
                context="test",
                decision="test",
                code_anchors=[CodeAnchor(file_path=other_file)],
            ),
        ]

        detector = DriftDetector(
            config=drift_config,
            project_root=project_root,
            decision_store=store,
        )

        report = await detector.check_files([changed_file])
        decision_items = [i for i in report.items if i.component == "decision"]
        assert len(decision_items) == 0
