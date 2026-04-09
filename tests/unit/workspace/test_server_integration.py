"""Tests for MCP server workspace routing integration."""

from __future__ import annotations

from codebase_intel.mcp.server import _extract_file_hint


class TestExtractFileHint:
    def test_file_path_arg(self) -> None:
        assert _extract_file_hint({"file_path": "/a/b/c.py"}) == "/a/b/c.py"

    def test_file_arg(self) -> None:
        assert _extract_file_hint({"file": "/a/b/c.py"}) == "/a/b/c.py"

    def test_files_array(self) -> None:
        assert _extract_file_hint({"files": ["/a/b.py", "/c/d.py"]}) == "/a/b.py"

    def test_changed_files_array(self) -> None:
        assert _extract_file_hint({"changed_files": ["/x/y.py"]}) == "/x/y.py"

    def test_empty_args(self) -> None:
        assert _extract_file_hint({}) is None

    def test_empty_files_array(self) -> None:
        assert _extract_file_hint({"files": []}) is None

    def test_empty_string_file_path(self) -> None:
        assert _extract_file_hint({"file_path": ""}) is None

    def test_priority_file_path_over_files(self) -> None:
        """file_path takes priority over files array."""
        result = _extract_file_hint({
            "file_path": "/direct.py",
            "files": ["/array.py"],
        })
        assert result == "/direct.py"

    def test_task_only_no_hint(self) -> None:
        """get_context with only task has no file hint."""
        assert _extract_file_hint({"task": "add rate limiting"}) is None

    def test_non_string_values_ignored(self) -> None:
        assert _extract_file_hint({"file_path": 123}) is None
        assert _extract_file_hint({"files": [123]}) is None
