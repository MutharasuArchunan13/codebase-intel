"""Auto-contract generator — discovers patterns in your codebase and generates rules.

THIS IS WHAT NOBODY ELSE DOES.

Instead of manually writing contracts, this module analyzes your existing code
and detects the patterns your team already follows:

- "Every API endpoint uses async def" → generates an async-enforcement rule
- "No file imports directly from the database layer except repositories" → layer rule
- "All service classes follow the naming pattern XxxService" → naming rule
- "Error handling always uses custom exception classes, never bare except" → pattern rule
- "Every test file has a corresponding source file" → coverage rule

The generated contracts are DRAFTS — human reviews and activates them.
This is intentional: auto-generated rules shouldn't silently enforce.

Why this matters:
- New team members learn conventions instantly
- AI agents follow patterns without being told
- Patterns that exist in practice become documented and enforced
- Zero manual contract writing needed to get started
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codebase_intel.contracts.models import (
    ContractRule,
    PatternExample,
    QualityContract,
    RuleKind,
    ScopeFilter,
)
from codebase_intel.core.types import ContractSeverity, Language

if TYPE_CHECKING:
    from codebase_intel.graph.storage import GraphStorage

logger = logging.getLogger(__name__)


@dataclass
class DetectedPattern:
    """A pattern detected in the codebase."""

    name: str
    description: str
    kind: RuleKind
    confidence: float  # 0.0-1.0: how consistently the pattern is followed
    occurrences: int  # How many files follow this pattern
    violations: int  # How many files break this pattern
    examples: list[str] = field(default_factory=list)  # File paths as examples
    counter_examples: list[str] = field(default_factory=list)
    suggested_rule: ContractRule | None = None


class AutoContractGenerator:
    """Analyzes a codebase and generates quality contracts from detected patterns."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    async def analyze(
        self,
        storage: GraphStorage | None = None,
    ) -> list[DetectedPattern]:
        """Run all pattern detectors and return discovered patterns.

        Each detector focuses on a specific type of pattern:
        - Async patterns (all handlers async vs mixed)
        - Import patterns (layer violations, circular imports)
        - Naming conventions
        - Error handling patterns
        - File organization patterns
        - Test coverage patterns
        """
        patterns: list[DetectedPattern] = []

        source_files = self._collect_source_files()
        if not source_files:
            return patterns

        patterns.extend(self._detect_async_patterns(source_files))
        patterns.extend(self._detect_import_layer_patterns(source_files))
        patterns.extend(self._detect_naming_conventions(source_files))
        patterns.extend(self._detect_error_handling_patterns(source_files))
        patterns.extend(self._detect_file_organization(source_files))
        patterns.extend(self._detect_docstring_patterns(source_files))

        # Filter to patterns with high confidence
        return [p for p in patterns if p.confidence >= 0.7]

    def generate_contract(
        self,
        patterns: list[DetectedPattern],
        contract_id: str = "auto-detected",
    ) -> QualityContract:
        """Convert detected patterns into a quality contract."""
        rules = []
        for pattern in patterns:
            if pattern.suggested_rule:
                rules.append(pattern.suggested_rule)

        return QualityContract(
            id=contract_id,
            name="Auto-Detected Project Conventions",
            description=(
                f"Quality rules auto-generated from analyzing {len(patterns)} "
                f"patterns in this codebase. Review and activate rules as needed."
            ),
            priority=150,  # Lower than manual contracts
            scope=ScopeFilter(exclude_tests=True, exclude_generated=True),
            rules=rules,
            tags=["auto-generated"],
        )

    # -------------------------------------------------------------------
    # Pattern detectors
    # -------------------------------------------------------------------

    def _detect_async_patterns(
        self, files: dict[Path, str]
    ) -> list[DetectedPattern]:
        """Detect if the project consistently uses async patterns."""
        patterns: list[DetectedPattern] = []

        async_defs = 0
        sync_defs = 0
        async_files: list[str] = []
        sync_files: list[str] = []

        for fp, content in files.items():
            if fp.suffix != ".py":
                continue

            has_async = bool(re.search(r"async\s+def\s+\w+", content))
            has_sync = bool(re.search(r"(?<!async\s)def\s+\w+", content))

            # Only count if file has route/endpoint patterns
            is_handler = any(kw in content for kw in ("@router.", "@app.", "async def get", "async def post", "async def create"))

            if is_handler:
                if has_async:
                    async_defs += 1
                    async_files.append(str(fp.relative_to(self._project_root)))
                if has_sync and not has_async:
                    sync_defs += 1
                    sync_files.append(str(fp.relative_to(self._project_root)))

        total = async_defs + sync_defs
        if total >= 3 and async_defs > sync_defs:
            confidence = async_defs / total
            patterns.append(DetectedPattern(
                name="Async handlers",
                description=f"This project uses async handlers ({async_defs}/{total} handler files are async)",
                kind=RuleKind.PATTERN,
                confidence=confidence,
                occurrences=async_defs,
                violations=sync_defs,
                examples=async_files[:3],
                counter_examples=sync_files[:3],
                suggested_rule=ContractRule(
                    id="auto-async-handlers",
                    name="Use async for all handler functions",
                    description=f"Detected pattern: {confidence:.0%} of handler files use async. Keep it consistent.",
                    kind=RuleKind.PATTERN,
                    severity=ContractSeverity.WARNING,
                    pattern=r"(?<!async\s)def\s+(get_|post_|put_|delete_|patch_|create_|update_|list_)",
                    fix_suggestion="Use `async def` for handler functions to match project convention.",
                ),
            ))

        return patterns

    def _detect_import_layer_patterns(
        self, files: dict[Path, str]
    ) -> list[DetectedPattern]:
        """Detect layer violation patterns in imports.

        Common pattern: routes/api files should not import database/ORM directly.
        """
        patterns: list[DetectedPattern] = []

        # Detect if routes import from db/models directly
        route_files_importing_db = 0
        route_files_clean = 0
        violating: list[str] = []
        clean: list[str] = []

        db_patterns = re.compile(r"from\s+\w*(models?|db|database|orm|tortoise|sqlalchemy)\w*\s+import")

        for fp, content in files.items():
            rel = str(fp.relative_to(self._project_root))

            is_route = any(kw in rel.lower() for kw in ("route", "router", "api", "endpoint", "view"))
            if not is_route:
                continue

            if db_patterns.search(content):
                route_files_importing_db += 1
                violating.append(rel)
            else:
                route_files_clean += 1
                clean.append(rel)

        total = route_files_importing_db + route_files_clean
        if total >= 3 and route_files_clean > route_files_importing_db:
            confidence = route_files_clean / total
            patterns.append(DetectedPattern(
                name="Layer separation (routes ↛ DB)",
                description=f"{confidence:.0%} of route files don't import DB directly — enforce this pattern",
                kind=RuleKind.ARCHITECTURAL,
                confidence=confidence,
                occurrences=route_files_clean,
                violations=route_files_importing_db,
                examples=clean[:3],
                counter_examples=violating[:3],
                suggested_rule=ContractRule(
                    id="auto-no-db-in-routes",
                    name="No direct DB imports in route handlers",
                    description="Route files should not import from database/model modules directly. Use a service layer.",
                    kind=RuleKind.ARCHITECTURAL,
                    severity=ContractSeverity.WARNING,
                    pattern=r"from\s+\w*(models?|db|database)\w*\s+import",
                    fix_suggestion="Import from a service module instead. Routes → Services → Repositories → DB.",
                ),
            ))

        return patterns

    def _detect_naming_conventions(
        self, files: dict[Path, str]
    ) -> list[DetectedPattern]:
        """Detect consistent naming conventions for classes and functions."""
        patterns: list[DetectedPattern] = []

        # Detect service class naming: FooService, BarService
        service_names = Counter()
        for fp, content in files.items():
            for match in re.finditer(r"class\s+(\w+Service)\b", content):
                service_names[match.group(1)] += 1

        if len(service_names) >= 3:
            patterns.append(DetectedPattern(
                name="Service class naming",
                description=f"Found {len(service_names)} service classes following XxxService pattern",
                kind=RuleKind.PATTERN,
                confidence=0.9,
                occurrences=len(service_names),
                violations=0,
                examples=list(service_names.keys())[:5],
                suggested_rule=ContractRule(
                    id="auto-service-naming",
                    name="Service classes must be named XxxService",
                    description="Detected convention: all service classes follow the XxxService naming pattern.",
                    kind=RuleKind.PATTERN,
                    severity=ContractSeverity.INFO,
                    fix_suggestion="Name service classes with the Service suffix: UserService, OrderService, etc.",
                ),
            ))

        return patterns

    def _detect_error_handling_patterns(
        self, files: dict[Path, str]
    ) -> list[DetectedPattern]:
        """Detect error handling conventions."""
        patterns: list[DetectedPattern] = []

        custom_exception_files = 0
        bare_except_files = 0
        custom_examples: list[str] = []
        bare_examples: list[str] = []

        for fp, content in files.items():
            if fp.suffix != ".py":
                continue

            rel = str(fp.relative_to(self._project_root))
            has_custom = bool(re.search(r"raise\s+\w+Error\(|raise\s+\w+Exception\(", content))
            has_bare = bool(re.search(r"except\s*:", content))

            if has_custom:
                custom_exception_files += 1
                custom_examples.append(rel)
            if has_bare:
                bare_except_files += 1
                bare_examples.append(rel)

        if custom_exception_files >= 3:
            total = custom_exception_files + bare_except_files
            confidence = custom_exception_files / max(total, 1)

            patterns.append(DetectedPattern(
                name="Custom exception handling",
                description=f"{custom_exception_files} files use custom exceptions. {bare_except_files} use bare except.",
                kind=RuleKind.PATTERN,
                confidence=confidence,
                occurrences=custom_exception_files,
                violations=bare_except_files,
                examples=custom_examples[:3],
                counter_examples=bare_examples[:3],
                suggested_rule=ContractRule(
                    id="auto-no-bare-except",
                    name="No bare except clauses",
                    description="This project uses custom exception classes. Avoid bare `except:` clauses.",
                    kind=RuleKind.PATTERN,
                    severity=ContractSeverity.WARNING,
                    pattern=r"except\s*:",
                    fix_suggestion="Catch specific exceptions: `except ValueError:` or `except CustomError:`",
                ),
            ))

        return patterns

    def _detect_file_organization(
        self, files: dict[Path, str]
    ) -> list[DetectedPattern]:
        """Detect file organization patterns."""
        patterns: list[DetectedPattern] = []

        # Detect if tests mirror source structure
        source_modules = set()
        test_modules = set()

        for fp in files:
            rel = fp.relative_to(self._project_root)
            parts = rel.parts

            if any(p in ("tests", "test", "__tests__") for p in parts):
                test_modules.add(fp.stem.replace("test_", "").replace("_test", ""))
            elif fp.suffix == ".py" and not fp.stem.startswith("_"):
                source_modules.add(fp.stem)

        if source_modules and test_modules:
            covered = source_modules & test_modules
            coverage_pct = len(covered) / max(len(source_modules), 1)

            if coverage_pct >= 0.3:
                patterns.append(DetectedPattern(
                    name="Test coverage structure",
                    description=f"{len(covered)}/{len(source_modules)} source modules have matching test files ({coverage_pct:.0%})",
                    kind=RuleKind.ARCHITECTURAL,
                    confidence=coverage_pct,
                    occurrences=len(covered),
                    violations=len(source_modules) - len(covered),
                    examples=list(covered)[:5],
                    suggested_rule=ContractRule(
                        id="auto-test-coverage",
                        name="Every source module should have a test file",
                        description=f"Detected: {coverage_pct:.0%} of modules have tests. Maintain this coverage.",
                        kind=RuleKind.ARCHITECTURAL,
                        severity=ContractSeverity.INFO,
                        fix_suggestion="Create test_<module>.py for new modules.",
                    ),
                ))

        return patterns

    def _detect_docstring_patterns(
        self, files: dict[Path, str]
    ) -> list[DetectedPattern]:
        """Detect docstring conventions."""
        patterns: list[DetectedPattern] = []

        with_docstrings = 0
        without_docstrings = 0

        for fp, content in files.items():
            if fp.suffix != ".py":
                continue

            # Count public functions with/without docstrings
            funcs = re.findall(r'def\s+([a-z]\w+)\s*\(.*?\).*?:\s*\n(\s+""")?', content, re.DOTALL)
            for name, docstring in funcs:
                if name.startswith("_"):
                    continue
                if docstring:
                    with_docstrings += 1
                else:
                    without_docstrings += 1

        total = with_docstrings + without_docstrings
        if total >= 10:
            confidence = with_docstrings / total
            if confidence >= 0.6:
                patterns.append(DetectedPattern(
                    name="Docstring convention",
                    description=f"{confidence:.0%} of public functions have docstrings",
                    kind=RuleKind.PATTERN,
                    confidence=confidence,
                    occurrences=with_docstrings,
                    violations=without_docstrings,
                    suggested_rule=ContractRule(
                        id="auto-docstrings",
                        name="Public functions should have docstrings",
                        description=f"Detected: {confidence:.0%} of public functions have docstrings. Maintain this.",
                        kind=RuleKind.PATTERN,
                        severity=ContractSeverity.INFO,
                        fix_suggestion="Add a one-line docstring explaining what the function does.",
                    ),
                ))

        return patterns

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _collect_source_files(self) -> dict[Path, str]:
        """Collect all source files and their content."""
        files: dict[Path, str] = {}
        skip_dirs = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build", ".tox"}

        def _walk(directory: Path) -> None:
            try:
                for entry in sorted(directory.iterdir()):
                    if entry.is_dir():
                        if entry.name not in skip_dirs and not entry.name.startswith("."):
                            _walk(entry)
                    elif entry.is_file() and entry.suffix in (".py", ".ts", ".tsx", ".js", ".jsx"):
                        try:
                            content = entry.read_text(encoding="utf-8", errors="ignore")
                            if len(content) < 500_000:  # Skip huge files
                                files[entry] = content
                        except OSError:
                            pass
            except PermissionError:
                pass

        _walk(self._project_root)
        return files
