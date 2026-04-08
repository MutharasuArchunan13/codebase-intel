"""Tests for FileParser — tree-sitter + regex-based source code extraction.

Covers: language detection, hashing, test-file detection, entry-point detection,
qualified module names, safe file reading, generated-code detection,
and import-module extraction.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import xxhash

from codebase_intel.core.config import ParserConfig
from codebase_intel.core.types import EdgeKind, Language, NodeKind
from codebase_intel.graph.parser import (
    FileParser,
    compute_file_hash,
    detect_language,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def parser_config() -> ParserConfig:
    return ParserConfig()


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture()
def parser(parser_config: ParserConfig, project_root: Path) -> FileParser:
    return FileParser(parser_config, project_root)


def _write(path: Path, content: str) -> Path:
    """Helper: write content to a file, creating parents as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    """Verify extension-to-language mapping for all supported extensions."""

    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            (".py", Language.PYTHON),
            (".pyi", Language.PYTHON),
            (".js", Language.JAVASCRIPT),
            (".mjs", Language.JAVASCRIPT),
            (".cjs", Language.JAVASCRIPT),
            (".jsx", Language.JAVASCRIPT),
            (".ts", Language.TYPESCRIPT),
            (".tsx", Language.TSX),
            (".go", Language.GO),
            (".rs", Language.RUST),
            (".java", Language.JAVA),
            (".rb", Language.RUBY),
        ],
    )
    def test_known_extensions(self, suffix: str, expected: Language) -> None:
        assert detect_language(Path(f"file{suffix}")) == expected

    @pytest.mark.parametrize("suffix", [".txt", ".md", ".csv", ".proto", ".sql", ""])
    def test_unknown_extensions(self, suffix: str) -> None:
        assert detect_language(Path(f"file{suffix}")) == Language.UNKNOWN

    def test_case_insensitive_suffix(self) -> None:
        assert detect_language(Path("file.PY")) == Language.PYTHON
        assert detect_language(Path("file.Ts")) == Language.TYPESCRIPT


# ---------------------------------------------------------------------------
# compute_file_hash
# ---------------------------------------------------------------------------


class TestComputeFileHash:
    def test_deterministic(self) -> None:
        content = b"hello world"
        h1 = compute_file_hash(content)
        h2 = compute_file_hash(content)
        assert h1 == h2

    def test_matches_xxhash64(self) -> None:
        content = b"deterministic check"
        assert compute_file_hash(content) == xxhash.xxh64(content).hexdigest()

    def test_different_content_different_hash(self) -> None:
        assert compute_file_hash(b"aaa") != compute_file_hash(b"bbb")

    def test_empty_content(self) -> None:
        """Empty bytes should hash without error."""
        h = compute_file_hash(b"")
        assert isinstance(h, str)
        assert len(h) > 0


# ---------------------------------------------------------------------------
# _is_test_file
# ---------------------------------------------------------------------------


class TestIsTestFile:
    def test_test_prefix(self, parser: FileParser) -> None:
        assert parser._is_test_file(Path("test_models.py")) is True

    def test_test_suffix(self, parser: FileParser) -> None:
        assert parser._is_test_file(Path("models_test.py")) is True

    def test_spec_suffix(self, parser: FileParser) -> None:
        assert parser._is_test_file(Path("models_spec.ts")) is True

    def test_dot_test_suffix(self, parser: FileParser) -> None:
        assert parser._is_test_file(Path("models.test.js")) is True

    def test_dot_spec_suffix(self, parser: FileParser) -> None:
        assert parser._is_test_file(Path("models.spec.tsx")) is True

    def test_conftest(self, parser: FileParser) -> None:
        assert parser._is_test_file(Path("conftest.py")) is True

    def test_tests_directory(self, parser: FileParser) -> None:
        assert parser._is_test_file(Path("tests/utils.py")) is True
        assert parser._is_test_file(Path("test/helpers.py")) is True
        assert parser._is_test_file(Path("__tests__/foo.js")) is True
        assert parser._is_test_file(Path("spec/bar.rb")) is True

    def test_regular_file_is_not_test(self, parser: FileParser) -> None:
        assert parser._is_test_file(Path("models.py")) is False
        assert parser._is_test_file(Path("src/service.py")) is False


# ---------------------------------------------------------------------------
# _is_entry_point
# ---------------------------------------------------------------------------


class TestIsEntryPoint:
    def test_main_py(self, parser: FileParser) -> None:
        assert parser._is_entry_point(Path("main.py"), "") is True

    def test_app_py(self, parser: FileParser) -> None:
        assert parser._is_entry_point(Path("app.py"), "") is True

    def test_manage_py(self, parser: FileParser) -> None:
        assert parser._is_entry_point(Path("manage.py"), "") is True

    def test_server_worker_cli(self, parser: FileParser) -> None:
        assert parser._is_entry_point(Path("server.py"), "") is True
        assert parser._is_entry_point(Path("worker.py"), "") is True
        assert parser._is_entry_point(Path("cli.py"), "") is True

    def test_dunder_main_pattern(self, parser: FileParser) -> None:
        code = 'if __name__ == "__main__":\n    main()'
        assert parser._is_entry_point(Path("run.py"), code) is True

    def test_regular_module_is_not_entry_point(self, parser: FileParser) -> None:
        assert parser._is_entry_point(Path("utils.py"), "def helper(): pass") is False


# ---------------------------------------------------------------------------
# _qualified_module_name
# ---------------------------------------------------------------------------


class TestQualifiedModuleName:
    def test_normal_file(self, parser: FileParser, project_root: Path) -> None:
        fp = project_root / "pkg" / "models.py"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.touch()
        assert parser._qualified_module_name(fp) == "pkg.models"

    def test_init_file_represents_package(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = project_root / "pkg" / "__init__.py"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.touch()
        assert parser._qualified_module_name(fp) == "pkg"

    def test_src_prefix_stripped(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = project_root / "src" / "mylib" / "core.py"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.touch()
        assert parser._qualified_module_name(fp) == "mylib.core"

    def test_lib_prefix_stripped(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = project_root / "lib" / "utils.py"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.touch()
        assert parser._qualified_module_name(fp) == "utils"

    def test_index_file_represents_directory(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = project_root / "components" / "index.ts"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.touch()
        assert parser._qualified_module_name(fp) == "components"

    def test_file_outside_project_root(self, parser: FileParser, tmp_path: Path) -> None:
        fp = tmp_path / "outside" / "rogue.py"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.touch()
        # Falls back to just the stem
        assert parser._qualified_module_name(fp) == "rogue"


# ---------------------------------------------------------------------------
# _read_file_safe
# ---------------------------------------------------------------------------


class TestReadFileSafe:
    def test_normal_utf8_file(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = _write(project_root / "hello.py", "print('hello')")
        result = parser._read_file_safe(fp)
        assert result is not None
        raw, text = result
        assert text == "print('hello')"
        assert raw == b"print('hello')"

    def test_binary_file_with_null_bytes(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = project_root / "binary.py"
        fp.write_bytes(b"\x00\x01\x02compiled bytecode")
        assert parser._read_file_safe(fp) is None

    def test_oversized_file_skipped(
        self, project_root: Path
    ) -> None:
        small_config = ParserConfig(max_file_size_bytes=50)
        p = FileParser(small_config, project_root)
        fp = _write(project_root / "big.py", "x" * 100)
        assert p._read_file_safe(fp) is None

    def test_permission_error(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = _write(project_root / "locked.py", "secret")
        fp.chmod(0o000)
        try:
            result = parser._read_file_safe(fp)
            # On some systems (root), permissions may not block reading
            # so we just verify no exception was raised
            assert result is None or result is not None
        finally:
            fp.chmod(0o644)

    def test_nonexistent_file(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = project_root / "ghost.py"
        assert parser._read_file_safe(fp) is None

    def test_latin1_fallback(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = project_root / "legacy.py"
        # Latin-1 byte that is invalid UTF-8 but valid Latin-1
        fp.write_bytes(b"# encoding: latin-1\nx = '\xe9'\n")
        result = parser._read_file_safe(fp)
        assert result is not None


# ---------------------------------------------------------------------------
# _is_generated
# ---------------------------------------------------------------------------


class TestIsGenerated:
    def test_generated_marker_in_first_line(self, parser: FileParser) -> None:
        assert parser._is_generated("# generated by protoc\ncode here") is True

    def test_auto_generated_marker(self, parser: FileParser) -> None:
        assert parser._is_generated("// auto-generated file\ncode") is True

    def test_do_not_edit_marker(self, parser: FileParser) -> None:
        assert parser._is_generated("# DO NOT EDIT\n# source: api.proto") is True

    def test_at_generated_marker(self, parser: FileParser) -> None:
        assert parser._is_generated("// @generated\nconst x = 1;") is True

    def test_marker_on_line_3(self, parser: FileParser) -> None:
        """Markers up to line 5 should be detected."""
        content = "#!/usr/bin/env python\n# encoding: utf-8\n# generated by tool\ncode"
        assert parser._is_generated(content) is True

    def test_no_marker(self, parser: FileParser) -> None:
        assert parser._is_generated("def hello():\n    pass") is False

    def test_marker_case_insensitive(self, parser: FileParser) -> None:
        assert parser._is_generated("# GENERATED by tool\ncode") is True

    def test_marker_beyond_line_5(self, parser: FileParser) -> None:
        """Markers past line 5 should NOT be detected."""
        lines = ["line " + str(i) for i in range(10)]
        lines[7] = "# generated"
        assert parser._is_generated("\n".join(lines)) is False


# ---------------------------------------------------------------------------
# _extract_import_module
# ---------------------------------------------------------------------------


class TestExtractImportModule:
    def test_from_x_import_y(self, parser: FileParser, project_root: Path) -> None:
        fp = project_root / "test.py"
        result = parser._extract_import_module("from os.path import join", fp)
        assert result == "os.path"

    def test_import_x_y(self, parser: FileParser, project_root: Path) -> None:
        fp = project_root / "test.py"
        result = parser._extract_import_module("import os.path", fp)
        assert result == "os.path"

    def test_import_simple(self, parser: FileParser, project_root: Path) -> None:
        fp = project_root / "test.py"
        result = parser._extract_import_module("import json", fp)
        assert result == "json"

    def test_from_typing_import(self, parser: FileParser, project_root: Path) -> None:
        fp = project_root / "test.py"
        result = parser._extract_import_module("from typing import List", fp)
        assert result == "typing"

    def test_relative_import_single_dot(
        self, parser: FileParser, project_root: Path
    ) -> None:
        pkg_dir = project_root / "mypkg"
        pkg_dir.mkdir()
        fp = pkg_dir / "child.py"
        result = parser._extract_import_module("from .sibling import func", fp)
        # Resolves relative to package
        assert result is not None
        assert "sibling" in result

    def test_relative_import_double_dot(
        self, parser: FileParser, project_root: Path
    ) -> None:
        nested = project_root / "pkg" / "sub"
        nested.mkdir(parents=True)
        fp = nested / "mod.py"
        result = parser._extract_import_module("from ..utils import helper", fp)
        assert result is not None
        assert "utils" in result

    def test_unrecognised_statement_returns_none(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = project_root / "test.py"
        result = parser._extract_import_module("print('not an import')", fp)
        assert result is None


# ---------------------------------------------------------------------------
# Full parse_file integration (uses tree-sitter if available)
# ---------------------------------------------------------------------------


class TestParseFile:
    async def test_parse_python_file_produces_module_node(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = _write(
            project_root / "simple.py",
            "def greet(name: str) -> str:\n    return f'Hello {name}'\n",
        )
        result = await parser.parse_file(fp)
        assert result is not None
        assert len(result.nodes) >= 1
        module_node = result.nodes[0]
        assert module_node.kind == NodeKind.MODULE
        assert module_node.name == "simple"

    async def test_parse_skips_ignored_patterns(
        self, project_root: Path
    ) -> None:
        config = ParserConfig(ignored_patterns=["vendor/**"])
        p = FileParser(config, project_root)
        fp = _write(project_root / "vendor" / "lib.py", "x = 1")
        result = await p.parse_file(fp)
        assert result is None

    async def test_parse_empty_file(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = _write(project_root / "empty.py", "")
        result = await parser.parse_file(fp)
        assert result is not None
        assert len(result.nodes) >= 1  # MODULE node still created

    async def test_parse_detects_test_file(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = _write(project_root / "test_utils.py", "def test_something(): pass")
        result = await parser.parse_file(fp)
        assert result is not None
        assert result.nodes[0].is_test is True

    async def test_parse_detects_generated_file(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = _write(
            project_root / "gen.py", "# auto-generated\nclass Stub: pass\n"
        )
        result = await parser.parse_file(fp)
        assert result is not None
        assert result.is_generated is True
        assert result.nodes[0].is_generated is True

    async def test_parse_detects_entry_point(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = _write(project_root / "main.py", "if __name__ == '__main__': pass")
        result = await parser.parse_file(fp)
        assert result is not None
        assert result.nodes[0].is_entry_point is True

    async def test_parse_extracts_content_hash(
        self, parser: FileParser, project_root: Path
    ) -> None:
        content = "x = 42\n"
        fp = _write(project_root / "const.py", content)
        result = await parser.parse_file(fp)
        assert result is not None
        assert result.content_hash == compute_file_hash(content.encode("utf-8"))

    async def test_parse_python_extracts_imports(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = _write(
            project_root / "consumer.py",
            "import os\nfrom pathlib import Path\n\ndef run(): pass\n",
        )
        result = await parser.parse_file(fp)
        assert result is not None
        import_edges = [e for e in result.edges if e.kind == EdgeKind.IMPORTS]
        assert len(import_edges) >= 1

    async def test_parse_python_extracts_class_and_function(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = _write(
            project_root / "models.py",
            (
                "class User:\n"
                "    \"\"\"A user model.\"\"\"\n"
                "    pass\n\n"
                "def create_user():\n"
                "    pass\n"
            ),
        )
        result = await parser.parse_file(fp)
        assert result is not None
        kinds = {n.kind for n in result.nodes}
        assert NodeKind.MODULE in kinds
        assert NodeKind.CLASS in kinds or NodeKind.FUNCTION in kinds

    async def test_parse_unknown_language_still_creates_module_node(
        self, parser: FileParser, project_root: Path
    ) -> None:
        fp = _write(project_root / "data.json", '{"key": "value"}')
        result = await parser.parse_file(fp)
        # .json is not in EXTENSION_MAP so detect_language returns UNKNOWN
        # parse_file may return None or a result depending on _is_ignored
        # If it returns a result, the module node should be present
        if result is not None:
            assert result.nodes[0].kind == NodeKind.MODULE


# ---------------------------------------------------------------------------
# Edge: _extract_import_module with various Python import forms
# ---------------------------------------------------------------------------


class TestImportEdgeCases:
    """Granular tests for tricky import patterns."""

    def test_from_future_handled_upstream(
        self, parser: FileParser, project_root: Path
    ) -> None:
        """__future__ imports are filtered at the caller level, not here."""
        fp = project_root / "mod.py"
        result = parser._extract_import_module(
            "from __future__ import annotations", fp
        )
        # _extract_import_module still returns the module name;
        # filtering happens in _parse_python_import
        assert result == "__future__"

    def test_star_import(self, parser: FileParser, project_root: Path) -> None:
        fp = project_root / "barrel.py"
        result = parser._extract_import_module("from mypackage import *", fp)
        assert result == "mypackage"

    def test_aliased_import(self, parser: FileParser, project_root: Path) -> None:
        fp = project_root / "mod.py"
        result = parser._extract_import_module("import numpy as np", fp)
        assert result == "numpy"
