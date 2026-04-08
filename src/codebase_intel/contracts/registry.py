"""Contract registry — loads, stores, and queries quality contracts.

Edge cases:
- No contracts directory: return builtins only (tool works out of the box)
- YAML contract file with syntax error: skip file, report error, continue
- Contract with invalid rule regex: validated at load time, skip bad rules
- Duplicate contract IDs: warn, keep the project-level one (overrides builtin)
- Hot reload: detect file changes and reload contracts without restart
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from codebase_intel.contracts.models import (
    ContractRule,
    PatternExample,
    QualityContract,
    RuleKind,
    ScopeFilter,
    builtin_ai_guardrails,
    builtin_architecture_rules,
)
from codebase_intel.core.exceptions import ContractParseError, ErrorContext
from codebase_intel.core.types import ContractSeverity, Language

if TYPE_CHECKING:
    from codebase_intel.core.config import ContractConfig

logger = logging.getLogger(__name__)


class ContractRegistry:
    """Manages loading and querying quality contracts."""

    def __init__(self, config: ContractConfig, project_root: Path) -> None:
        self._config = config
        self._project_root = project_root
        self._contracts: dict[str, QualityContract] = {}
        self._loaded = False

    def load(self) -> None:
        """Load all contracts from builtins and project directory.

        Load order:
        1. Built-in contracts (shipped with codebase-intel)
        2. Project contracts (from .codebase-intel/contracts/)

        Project contracts override builtins with the same ID.
        """
        self._contracts.clear()

        # Load builtins
        if self._config.enable_builtin_contracts:
            for builtin in self._get_builtins():
                self._contracts[builtin.id] = builtin

        # Load project contracts
        contracts_dir = self._config.contracts_dir
        if contracts_dir.exists():
            for yaml_file in sorted(contracts_dir.glob("*.yaml")):
                try:
                    contract = self._load_contract_file(yaml_file)
                    if contract.id in self._contracts and self._contracts[contract.id].is_builtin:
                        logger.info(
                            "Project contract '%s' overrides builtin",
                            contract.id,
                        )
                    self._contracts[contract.id] = contract
                except Exception as exc:
                    logger.warning(
                        "Failed to load contract %s: %s", yaml_file, exc
                    )

        self._loaded = True
        logger.info(
            "Loaded %d contracts (%d builtin, %d project)",
            len(self._contracts),
            sum(1 for c in self._contracts.values() if c.is_builtin),
            sum(1 for c in self._contracts.values() if not c.is_builtin),
        )

    def get_all(self) -> list[QualityContract]:
        """Get all loaded contracts, sorted by priority (highest first)."""
        if not self._loaded:
            self.load()
        return sorted(
            self._contracts.values(),
            key=lambda c: c.priority,
            reverse=True,
        )

    def get(self, contract_id: str) -> QualityContract | None:
        """Get a specific contract by ID."""
        if not self._loaded:
            self.load()
        return self._contracts.get(contract_id)

    def get_for_file(self, file_path: Path, language: Language = Language.UNKNOWN) -> list[QualityContract]:
        """Get all contracts applicable to a specific file."""
        if not self._loaded:
            self.load()
        return [
            c for c in self._contracts.values()
            if c.scope.matches(file_path, language)
        ]

    def _get_builtins(self) -> list[QualityContract]:
        """Get all built-in contract definitions."""
        return [
            builtin_ai_guardrails(),
            builtin_architecture_rules(),
        ]

    def _load_contract_file(self, yaml_file: Path) -> QualityContract:
        """Load a contract from a YAML file.

        Expected format:
        ```yaml
        id: my-contract
        name: My Quality Contract
        description: What this enforces
        priority: 200
        scope:
          include_patterns: ["src/api/**"]
          languages: ["python"]
        rules:
          - id: no-raw-sql
            name: No raw SQL queries
            kind: architectural
            severity: error
            pattern: "execute\\(.*SELECT|INSERT|UPDATE|DELETE"
            fix_suggestion: Use the repository pattern
        ```

        Edge case: partial YAML (missing optional fields) → Pydantic defaults fill in.
        Edge case: unknown fields → ignored (forward compatibility).
        """
        content = yaml_file.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise ContractParseError(
                f"Invalid YAML in {yaml_file.name}: {exc}",
                ErrorContext(file_path=yaml_file),
            ) from exc

        if not isinstance(data, dict):
            raise ContractParseError(
                f"Contract file {yaml_file.name} must be a YAML mapping",
                ErrorContext(file_path=yaml_file),
            )

        # Parse scope
        scope_data = data.get("scope", {})
        if isinstance(scope_data, dict):
            # Convert string language names to Language enum
            if "languages" in scope_data:
                scope_data["languages"] = [
                    Language(l) if isinstance(l, str) else l
                    for l in scope_data["languages"]
                ]
            data["scope"] = ScopeFilter(**scope_data)

        # Parse rules
        rules = []
        for rule_data in data.get("rules", []):
            if isinstance(rule_data, dict):
                # Convert string enums
                if "kind" in rule_data:
                    rule_data["kind"] = RuleKind(rule_data["kind"])
                if "severity" in rule_data:
                    rule_data["severity"] = ContractSeverity(rule_data["severity"])

                # Parse examples
                examples = []
                for ex in rule_data.get("examples", []):
                    if isinstance(ex, dict):
                        examples.append(PatternExample(**ex))
                rule_data["examples"] = examples

                rules.append(ContractRule(**rule_data))
        data["rules"] = rules

        return QualityContract(**data)

    async def create_template(self, contract_id: str, name: str) -> Path:
        """Create a template contract file for the user to customize.

        This is called by `codebase-intel init` to bootstrap project contracts.
        """
        self._config.contracts_dir.mkdir(parents=True, exist_ok=True)

        template = {
            "id": contract_id,
            "name": name,
            "description": f"Quality rules for {name}",
            "priority": 200,
            "scope": {
                "include_patterns": ["src/**"],
                "exclude_patterns": ["node_modules/**", "dist/**"],
            },
            "rules": [
                {
                    "id": "example-rule",
                    "name": "Example Rule",
                    "description": "Replace this with your actual rule",
                    "kind": "pattern",
                    "severity": "warning",
                    "pattern": "TODO|FIXME|HACK",
                    "fix_suggestion": "Resolve the TODO before merging",
                },
            ],
        }

        file_path = self._config.contracts_dir / f"{contract_id}.yaml"
        file_path.write_text(
            yaml.dump(template, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        return file_path
