"""Intent store — YAML-based persistence for intents.

Intents are stored as individual YAML files in .codebase-intel/intents/.
Like decisions, they're human-readable and git-friendly.

Example intent file:

```yaml
id: INT-001
title: "Live efficiency tracking must work end-to-end"
status: active
priority: 1
criteria:
  - description: "Analytics tracker exists"
    criterion_type: file_exists
    check_value: "src/codebase_intel/analytics/tracker.py"
  - description: "MCP server passes tracker to assembler"
    criterion_type: file_contains
    check_value: "src/codebase_intel/mcp/server.py::analytics_tracker"
  - description: "get_context records events in analytics.db"
    criterion_type: wired
    check_value: "assembler -> analytics"
  - description: "Dashboard shows real data"
    criterion_type: cli_works
    check_value: "codebase-intel dashboard"
```
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from codebase_intel.intent.models import (
    AcceptanceCriterion,
    CriterionType,
    Intent,
    IntentStatus,
)

logger = logging.getLogger(__name__)


class IntentStore:
    """Manages intent records stored as YAML files."""

    def __init__(self, intents_dir: Path) -> None:
        self._intents_dir = intents_dir
        self._cache: dict[str, Intent] = {}

    @property
    def intents_dir(self) -> Path:
        return self._intents_dir

    def load_all(self) -> list[Intent]:
        """Load all intents from the intents directory."""
        if not self._intents_dir.exists():
            return []

        intents: list[Intent] = []
        for yaml_file in sorted(self._intents_dir.glob("*.yaml")):
            try:
                intent = self._load_file(yaml_file)
                if intent:
                    intents.append(intent)
                    self._cache[intent.id] = intent
            except Exception as exc:
                logger.warning("Failed to load intent %s: %s", yaml_file, exc)

        return intents

    def get(self, intent_id: str) -> Intent | None:
        if intent_id in self._cache:
            return self._cache[intent_id]
        self.load_all()
        return self._cache.get(intent_id)

    def get_active(self) -> list[Intent]:
        """Get all active (unverified) intents."""
        all_intents = self.load_all()
        return [i for i in all_intents if i.status == IntentStatus.ACTIVE]

    def save(self, intent: Intent) -> Path:
        """Save an intent to a YAML file."""
        self._intents_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{intent.id}.yaml"
        file_path = self._intents_dir / filename

        data = self._intent_to_dict(intent)
        content = yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

        file_path.write_text(content, encoding="utf-8")
        self._cache[intent.id] = intent

        logger.info("Saved intent %s to %s", intent.id, file_path)
        return file_path

    def update_status(self, intent_id: str, status: IntentStatus) -> None:
        """Update an intent's status after verification."""
        intent = self.get(intent_id)
        if not intent:
            return

        # Pydantic frozen model — reconstruct
        updated = intent.model_copy(update={
            "status": status,
            "updated_at": datetime.now(UTC),
            "verified_at": datetime.now(UTC) if status == IntentStatus.VERIFIED else intent.verified_at,
        })
        self.save(updated)

    def update_criteria(self, intent_id: str, criteria: list[AcceptanceCriterion]) -> None:
        """Update criteria after verification run."""
        intent = self.get(intent_id)
        if not intent:
            return

        updated = intent.model_copy(update={
            "criteria": criteria,
            "updated_at": datetime.now(UTC),
        })
        self.save(updated)

    def next_id(self) -> str:
        """Generate the next intent ID."""
        all_intents = self.load_all()
        max_num = 0
        for i in all_intents:
            if i.id.startswith("INT-"):
                try:
                    num = int(i.id.split("-", 1)[1])
                    max_num = max(max_num, num)
                except (ValueError, IndexError):
                    pass
        return f"INT-{max_num + 1:03d}"

    def _load_file(self, yaml_file: Path) -> Intent | None:
        content = yaml_file.read_text(encoding="utf-8")
        if not content.strip():
            return None

        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            return None

        # Parse criteria
        criteria = []
        for c_data in data.get("criteria", []):
            if isinstance(c_data, dict):
                if "criterion_type" in c_data:
                    c_data["criterion_type"] = CriterionType(c_data["criterion_type"])
                criteria.append(AcceptanceCriterion(**c_data))
        data["criteria"] = criteria

        if "status" in data:
            data["status"] = IntentStatus(data["status"])

        return Intent(**data)

    def _intent_to_dict(self, intent: Intent) -> dict[str, Any]:
        data = intent.model_dump(mode="json")
        # Simplify criteria for YAML readability
        if data.get("criteria"):
            simplified = []
            for c in data["criteria"]:
                entry: dict[str, Any] = {
                    "description": c["description"],
                    "criterion_type": c["criterion_type"],
                    "check_value": c["check_value"],
                }
                if c.get("verified"):
                    entry["verified"] = True
                if c.get("failure_reason"):
                    entry["failure_reason"] = c["failure_reason"]
                simplified.append(entry)
            data["criteria"] = simplified
        return data
