"""Intent verifier — runs acceptance criteria checks against the actual codebase.

This is the "trust but verify" engine. When an agent says "I'm done,"
the verifier checks every criterion and reports what's actually true.

Verification types:
- file_exists: does the file exist at the given path?
- file_contains: does the file contain the pattern?
- function_exists: does the function/class exist in the code graph?
- wired: does module A actually import/use module B?
- cli_works: does the CLI command run without error?
- mcp_tool_exists: is the MCP tool registered in the server?
- grep_match: does the pattern match anywhere in the codebase?
- grep_no_match: is the pattern absent (something was removed)?
- test_passes: does the test pass?
- custom: does the shell command return exit code 0?
- manual: requires human confirmation (always returns unverified)
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codebase_intel.intent.models import (
    AcceptanceCriterion,
    CriterionType,
    Intent,
    IntentStatus,
)

logger = logging.getLogger(__name__)


class VerificationResult:
    """Result of verifying a single criterion."""

    def __init__(
        self,
        criterion: AcceptanceCriterion,
        passed: bool,
        failure_reason: str | None = None,
    ) -> None:
        self.criterion = criterion
        self.passed = passed
        self.failure_reason = failure_reason


class IntentVerificationReport:
    """Result of verifying all criteria for an intent."""

    def __init__(self, intent: Intent) -> None:
        self.intent_id = intent.id
        self.intent_title = intent.title
        self.results: list[VerificationResult] = []
        self.verified_at = datetime.now(UTC)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def status(self) -> IntentStatus:
        if not self.results:
            return IntentStatus.ACTIVE
        if self.all_passed:
            return IntentStatus.VERIFIED
        if self.passed_count > 0:
            return IntentStatus.PARTIAL
        return IntentStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "intent_title": self.intent_title,
            "status": self.status.value,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "total": self.total,
            "verified_at": self.verified_at.isoformat(),
            "results": [
                {
                    "description": r.criterion.description,
                    "type": r.criterion.criterion_type.value,
                    "passed": r.passed,
                    "failure_reason": r.failure_reason,
                }
                for r in self.results
            ],
            "gaps": [
                {
                    "description": r.criterion.description,
                    "reason": r.failure_reason or "Not verified",
                }
                for r in self.results
                if not r.passed
            ],
        }

    def to_context_string(self) -> str:
        """For agent context — shows exactly what passed and what didn't."""
        lines = [
            f"## Verification: {self.intent_title} [{self.intent_id}]",
            f"Result: **{self.status.value.upper()}** — {self.passed_count}/{self.total} criteria met",
            "",
        ]

        for r in self.results:
            icon = "PASS" if r.passed else "FAIL"
            lines.append(f"- [{icon}] {r.criterion.description}")
            if r.failure_reason:
                lines.append(f"  Reason: {r.failure_reason}")

        if not self.all_passed:
            lines.append(f"\n**{self.failed_count} criteria FAILED. Do NOT mark this as done.**")
            lines.append("Fix the failing criteria and run verification again.")

        return "\n".join(lines)


class IntentVerifier:
    """Runs acceptance criteria checks against the actual codebase."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    async def verify(self, intent: Intent) -> IntentVerificationReport:
        """Verify all criteria for an intent."""
        report = IntentVerificationReport(intent)

        for criterion in intent.criteria:
            result = await self._verify_criterion(criterion)
            report.results.append(result)

        return report

    async def _verify_criterion(self, criterion: AcceptanceCriterion) -> VerificationResult:
        """Verify a single criterion. Dispatches by type."""
        try:
            check_map = {
                CriterionType.FILE_EXISTS: self._check_file_exists,
                CriterionType.FILE_CONTAINS: self._check_file_contains,
                CriterionType.FUNCTION_EXISTS: self._check_function_exists,
                CriterionType.WIRED: self._check_wired,
                CriterionType.CLI_WORKS: self._check_cli_works,
                CriterionType.MCP_TOOL_EXISTS: self._check_mcp_tool,
                CriterionType.GREP_MATCH: self._check_grep_match,
                CriterionType.GREP_NO_MATCH: self._check_grep_no_match,
                CriterionType.TEST_PASSES: self._check_test_passes,
                CriterionType.CUSTOM: self._check_custom,
                CriterionType.MANUAL: self._check_manual,
            }

            checker = check_map.get(criterion.criterion_type)
            if checker is None:
                return VerificationResult(
                    criterion, False, f"Unknown criterion type: {criterion.criterion_type}"
                )

            return await checker(criterion)

        except Exception as exc:
            return VerificationResult(criterion, False, f"Check error: {exc}")

    # -------------------------------------------------------------------
    # Individual checks
    # -------------------------------------------------------------------

    async def _check_file_exists(self, c: AcceptanceCriterion) -> VerificationResult:
        path = self._resolve_path(c.check_value)
        if path.exists():
            return VerificationResult(c, True)
        return VerificationResult(c, False, f"File not found: {c.check_value}")

    async def _check_file_contains(self, c: AcceptanceCriterion) -> VerificationResult:
        """check_value format: 'path/to/file.py::pattern_to_find'"""
        parts = c.check_value.split("::", 1)
        if len(parts) != 2:
            return VerificationResult(c, False, "Invalid format. Use 'file_path::pattern'")

        file_path = self._resolve_path(parts[0].strip())
        pattern = parts[1].strip()

        if not file_path.exists():
            return VerificationResult(c, False, f"File not found: {parts[0]}")

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return VerificationResult(c, False, f"Cannot read file: {exc}")

        if re.search(pattern, content):
            return VerificationResult(c, True)
        return VerificationResult(c, False, f"Pattern '{pattern}' not found in {parts[0]}")

    async def _check_function_exists(self, c: AcceptanceCriterion) -> VerificationResult:
        """Check if a function/class exists by grepping the source."""
        pattern = rf"(def|class|function|async def)\s+{re.escape(c.check_value)}\b"
        for py_file in self._project_root.rglob("*.py"):
            if any(skip in str(py_file) for skip in ("node_modules", ".venv", "__pycache__", ".git")):
                continue
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                if re.search(pattern, content):
                    return VerificationResult(c, True)
            except OSError:
                continue
        return VerificationResult(c, False, f"Function/class '{c.check_value}' not found in codebase")

    async def _check_wired(self, c: AcceptanceCriterion) -> VerificationResult:
        """check_value format: 'module_a -> module_b'
        Checks that module_a imports or references module_b.
        """
        parts = c.check_value.split("->", 1)
        if len(parts) != 2:
            return VerificationResult(c, False, "Invalid format. Use 'module_a -> module_b'")

        source_name = parts[0].strip()
        target_name = parts[1].strip()

        # Find source files matching module_a
        for py_file in self._project_root.rglob("*.py"):
            if any(skip in str(py_file) for skip in ("node_modules", ".venv", "__pycache__", ".git")):
                continue
            if source_name in str(py_file):
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                    # Check for import or usage of target
                    if re.search(rf"(import|from)\s+.*{re.escape(target_name)}", content):
                        return VerificationResult(c, True)
                    if target_name in content:
                        return VerificationResult(c, True)
                except OSError:
                    continue

        return VerificationResult(
            c, False,
            f"'{source_name}' does not import or reference '{target_name}'"
        )

    async def _check_cli_works(self, c: AcceptanceCriterion) -> VerificationResult:
        """Run a CLI command and check it exits with code 0."""
        try:
            result = subprocess.run(
                c.check_value,
                shell=True,  # noqa: S602
                capture_output=True,
                timeout=30,
                cwd=str(self._project_root),
            )
            if result.returncode == 0:
                return VerificationResult(c, True)
            stderr = result.stderr.decode("utf-8", errors="ignore")[:200]
            return VerificationResult(c, False, f"Exit code {result.returncode}: {stderr}")
        except subprocess.TimeoutExpired:
            return VerificationResult(c, False, "Command timed out (30s)")
        except Exception as exc:
            return VerificationResult(c, False, f"Command failed: {exc}")

    async def _check_mcp_tool(self, c: AcceptanceCriterion) -> VerificationResult:
        """Check if an MCP tool name exists in the server code."""
        server_file = self._project_root / "src" / "codebase_intel" / "mcp" / "server.py"
        if not server_file.exists():
            # Try relative to project
            for candidate in self._project_root.rglob("mcp/server.py"):
                server_file = candidate
                break

        if not server_file.exists():
            return VerificationResult(c, False, "MCP server file not found")

        content = server_file.read_text(encoding="utf-8", errors="ignore")
        tool_name = c.check_value.strip()
        if f'name="{tool_name}"' in content or f"name='{tool_name}'" in content:
            return VerificationResult(c, True)
        return VerificationResult(c, False, f"MCP tool '{tool_name}' not found in server.py")

    async def _check_grep_match(self, c: AcceptanceCriterion) -> VerificationResult:
        """check_value format: 'pattern::path_glob' or just 'pattern'"""
        parts = c.check_value.split("::", 1)
        pattern = parts[0].strip()
        path_glob = parts[1].strip() if len(parts) > 1 else "**/*.py"

        for f in self._project_root.glob(path_glob):
            if f.is_file() and not any(skip in str(f) for skip in (".git", "__pycache__", "node_modules", ".venv")):
                try:
                    content = f.read_text(encoding="utf-8", errors="ignore")
                    if re.search(pattern, content):
                        return VerificationResult(c, True)
                except OSError:
                    continue

        return VerificationResult(c, False, f"Pattern '{pattern}' not found in {path_glob}")

    async def _check_grep_no_match(self, c: AcceptanceCriterion) -> VerificationResult:
        """Inverse of grep_match — pattern should NOT be found."""
        parts = c.check_value.split("::", 1)
        pattern = parts[0].strip()
        path_glob = parts[1].strip() if len(parts) > 1 else "**/*.py"

        for f in self._project_root.glob(path_glob):
            if f.is_file() and not any(skip in str(f) for skip in (".git", "__pycache__", "node_modules", ".venv")):
                try:
                    content = f.read_text(encoding="utf-8", errors="ignore")
                    if re.search(pattern, content):
                        return VerificationResult(
                            c, False, f"Pattern '{pattern}' still found in {f.name}"
                        )
                except OSError:
                    continue

        return VerificationResult(c, True)

    async def _check_test_passes(self, c: AcceptanceCriterion) -> VerificationResult:
        """Run a specific test and check it passes."""
        try:
            result = subprocess.run(
                f"python -m pytest {c.check_value} -x -q --no-header",
                shell=True,  # noqa: S602
                capture_output=True,
                timeout=60,
                cwd=str(self._project_root),
            )
            if result.returncode == 0:
                return VerificationResult(c, True)
            stdout = result.stdout.decode("utf-8", errors="ignore")[-200:]
            return VerificationResult(c, False, f"Test failed: {stdout}")
        except subprocess.TimeoutExpired:
            return VerificationResult(c, False, "Test timed out (60s)")
        except Exception as exc:
            return VerificationResult(c, False, f"Test error: {exc}")

    async def _check_custom(self, c: AcceptanceCriterion) -> VerificationResult:
        """Run a custom shell command."""
        return await self._check_cli_works(c)

    async def _check_manual(self, c: AcceptanceCriterion) -> VerificationResult:
        """Manual checks always return unverified — human must confirm."""
        return VerificationResult(c, False, "Requires manual verification by a human")

    def _resolve_path(self, path_str: str) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self._project_root / p
