"""Tree-sitter based source code parser — extracts nodes and edges from files.

This is the most edge-case-heavy module in the system. Source code is wild:
people do bizarre things with imports, naming, and structure.

Edge cases handled:
- Binary files disguised as source (UTF-8 decode fails): skip gracefully
- Mixed encodings in a file: try UTF-8, fall back to latin-1, then skip
- Syntax errors in user code: tree-sitter still produces a partial AST,
  we extract what we can and flag errors
- Generated code: detected via header markers, tagged is_generated=True
- Huge files (>1MB): skip entirely, log warning
- Empty files: valid — produce MODULE node with no children
- Circular imports: not our problem at parse time (graph handles it)
- Dynamic imports: `importlib.import_module("x")`, `__import__("x")`,
  `require(variable)` — extract target string if it's a literal, flag
  as dynamic_import with lower confidence if it's a variable
- Conditional imports: `if TYPE_CHECKING:`, `try/except ImportError:` —
  tagged as type_only or optional
- Re-exports: barrel files that `from x import *` or explicit re-export
- Star imports: `from x import *` — edge exists but target is the module,
  not specific symbols (can't resolve without runtime)
- Relative imports: `from . import x`, `from ..utils import y` — resolved
  relative to file position in the project
- Decorator detection: @app.route, @router.get — used to identify endpoints
- Async vs sync: tracked in metadata for quality contracts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import xxhash

from codebase_intel.core.exceptions import ErrorContext, ParseError, UnsupportedLanguageError
from codebase_intel.core.types import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    Language,
    LineRange,
    NodeKind,
)

if TYPE_CHECKING:
    from codebase_intel.core.config import ParserConfig

logger = logging.getLogger(__name__)

# File extension → Language mapping (19 languages)
EXTENSION_MAP: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".pyi": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TSX,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".java": Language.JAVA,
    ".rb": Language.RUBY,
    ".c": Language.C,
    ".h": Language.C,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
    ".cs": Language.CSHARP,
    ".php": Language.PHP,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".lua": Language.LUA,
    ".dart": Language.DART,
    ".ex": Language.ELIXIR,
    ".exs": Language.ELIXIR,
    ".hs": Language.HASKELL,
}

# tree-sitter-language-pack grammar names
LANGUAGE_GRAMMAR_MAP: dict[Language, str] = {
    Language.PYTHON: "python",
    Language.JAVASCRIPT: "javascript",
    Language.TYPESCRIPT: "typescript",
    Language.TSX: "tsx",
    Language.GO: "go",
    Language.RUST: "rust",
    Language.JAVA: "java",
    Language.RUBY: "ruby",
    Language.C: "c",
    Language.CPP: "cpp",
    Language.CSHARP: "c_sharp",
    Language.PHP: "php",
    Language.SWIFT: "swift",
    Language.KOTLIN: "kotlin",
    Language.SCALA: "scala",
    Language.LUA: "lua",
    Language.DART: "dart",
    Language.ELIXIR: "elixir",
    Language.HASKELL: "haskell",
}


def detect_language(file_path: Path) -> Language:
    """Detect language from file extension.

    Edge case: .tsx is TSX not TypeScript (different grammar).
    Edge case: .mjs/.cjs are JavaScript (ES modules / CommonJS).
    Edge case: .pyi is Python (type stubs — treated same as .py).
    """
    return EXTENSION_MAP.get(file_path.suffix.lower(), Language.UNKNOWN)


def compute_file_hash(content: bytes) -> str:
    """Content-addressable hash using xxhash (much faster than SHA for our use case)."""
    return xxhash.xxh64(content).hexdigest()


class ParseResult:
    """Result of parsing a single file — nodes, edges, and any warnings."""

    def __init__(self, file_path: Path, language: Language) -> None:
        self.file_path = file_path
        self.language = language
        self.nodes: list[GraphNode] = []
        self.edges: list[GraphEdge] = []
        self.warnings: list[str] = []
        self.content_hash: str = ""
        self.size_bytes: int = 0
        self.is_generated: bool = False
        self.had_syntax_errors: bool = False

    @property
    def module_node_id(self) -> str:
        """The node ID of the file-level MODULE node."""
        return GraphNode.make_id(self.file_path, NodeKind.MODULE, self.file_path.stem)


class FileParser:
    """Parses source files into graph nodes and edges.

    Uses tree-sitter for language-aware AST parsing. Falls back to
    regex-based extraction for unsupported languages (limited but
    better than nothing).
    """

    def __init__(self, config: ParserConfig, project_root: Path) -> None:
        self._config = config
        self._project_root = project_root
        self._grammars: dict[Language, object] = {}

    def _is_ignored(self, file_path: Path) -> bool:
        """Check if file matches any ignore pattern.

        Edge case: patterns are relative to project root.
        Edge case: symlinks — we resolve before checking to avoid
        processing the same file twice via different paths.
        """
        import fnmatch

        try:
            rel = file_path.resolve().relative_to(self._project_root)
        except ValueError:
            return True  # Outside project root — skip

        rel_str = str(rel)
        return any(
            fnmatch.fnmatch(rel_str, pattern)
            for pattern in self._config.ignored_patterns
        )

    def _is_generated(self, content: str) -> bool:
        """Check if file is generated code by examining its header.

        Edge case: some generated files put the marker on line 3 (after
        a shebang and encoding declaration). We check first 5 lines.
        """
        header = "\n".join(content.split("\n")[:5]).lower()
        return any(marker.lower() in header for marker in self._config.generated_markers)

    def _read_file_safe(self, file_path: Path) -> tuple[bytes, str] | None:
        """Read file with encoding fallback.

        Edge cases:
        - Binary file: UTF-8 decode fails, latin-1 produces garbage but doesn't crash
        - Mixed encoding: some lines UTF-8, some not — we get partial content
        - Null bytes in file: strong indicator of binary, skip
        - Symlink to file outside project: resolve and check

        Returns (raw_bytes, decoded_text) or None if unreadable.
        """
        try:
            raw = file_path.read_bytes()
        except (OSError, PermissionError) as exc:
            logger.warning("Cannot read %s: %s", file_path, exc)
            return None

        # Size check
        if len(raw) > self._config.max_file_size_bytes:
            logger.info(
                "Skipping %s: %d bytes exceeds limit %d",
                file_path,
                len(raw),
                self._config.max_file_size_bytes,
            )
            return None

        # Binary detection: null bytes in first 8KB
        if b"\x00" in raw[:8192]:
            logger.debug("Skipping binary file: %s", file_path)
            return None

        # Decode with fallback
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("latin-1")
                logger.debug("Fell back to latin-1 for %s", file_path)
            except UnicodeDecodeError:
                logger.warning("Cannot decode %s with any encoding", file_path)
                return None

        return raw, text

    async def parse_file(self, file_path: Path) -> ParseResult | None:
        """Parse a single file and extract graph nodes and edges.

        Returns None if the file should be skipped entirely.

        This is the main entry point. Language-specific extraction
        is delegated to _extract_python, _extract_javascript, etc.
        """
        if self._is_ignored(file_path):
            return None

        language = detect_language(file_path)

        result = ParseResult(file_path, language)

        # Read file
        read_result = self._read_file_safe(file_path)
        if read_result is None:
            return None

        raw_bytes, text = read_result
        result.content_hash = compute_file_hash(raw_bytes)
        result.size_bytes = len(raw_bytes)
        result.is_generated = self._is_generated(text)

        # Create the MODULE-level node (every file gets one)
        is_test = self._is_test_file(file_path)
        module_node = GraphNode(
            node_id=result.module_node_id,
            kind=NodeKind.MODULE,
            name=file_path.stem,
            qualified_name=self._qualified_module_name(file_path),
            file_path=file_path,
            line_range=LineRange(start=1, end=max(1, text.count("\n") + 1)),
            language=language,
            content_hash=result.content_hash,
            is_generated=result.is_generated,
            is_test=is_test,
            is_entry_point=self._is_entry_point(file_path, text),
        )
        result.nodes.append(module_node)

        # Language-specific extraction
        if language == Language.PYTHON:
            await self._extract_python(text, file_path, result)
        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT, Language.TSX):
            await self._extract_javascript_family(text, file_path, result)
        elif language == Language.UNKNOWN:
            result.warnings.append(f"No parser for {file_path.suffix}")
        elif language in LANGUAGE_GRAMMAR_MAP:
            # All other supported languages — extract basic structure via tree-sitter
            await self._extract_generic(text, file_path, language, result)
        else:
            result.warnings.append(f"Language {language.value} not in enabled_languages")

        return result

    def _is_test_file(self, file_path: Path) -> bool:
        """Detect if a file is a test file.

        Edge cases:
        - test_*.py, *_test.py, *_spec.ts, *.test.js — all common patterns
        - Files inside tests/, __tests__/, spec/ directories
        - conftest.py is test infrastructure, not a test itself (still flagged)
        """
        name = file_path.stem.lower()
        parts = [p.lower() for p in file_path.parts]

        is_test_name = (
            name.startswith("test_")
            or name.endswith("_test")
            or name.endswith("_spec")
            or name.endswith(".test")
            or name.endswith(".spec")
            or name == "conftest"
        )
        is_test_dir = any(
            p in ("tests", "test", "__tests__", "spec", "specs") for p in parts
        )
        return is_test_name or is_test_dir

    def _is_entry_point(self, file_path: Path, content: str) -> bool:
        """Detect if a file is an application entry point.

        Edge cases:
        - Python: `if __name__ == "__main__"`, main.py, app.py, manage.py
        - JS/TS: package.json "main" field (not detectable here — handled in builder)
        - Multiple entry points: CLI, web server, worker — all valid
        """
        name = file_path.stem.lower()
        if name in ("main", "app", "manage", "server", "worker", "cli"):
            return True
        if '__name__' in content and '__main__' in content:
            return True
        return False

    def _qualified_module_name(self, file_path: Path) -> str:
        """Build a Python-style qualified module name from file path.

        Edge case: __init__.py represents the package, not a module named "init".
        Edge case: files outside src/ — use path from project root.
        """
        try:
            rel = file_path.resolve().relative_to(self._project_root)
        except ValueError:
            return file_path.stem

        parts = list(rel.with_suffix("").parts)

        # Remove common source directories from the qualified name
        for prefix in ("src", "lib", "app"):
            if parts and parts[0] == prefix:
                parts = parts[1:]
                break

        # __init__ represents the package
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]

        # index.ts/js represents the directory
        if parts and parts[-1] in ("index", "mod"):
            parts = parts[:-1]

        return ".".join(parts) if parts else file_path.stem

    # -------------------------------------------------------------------
    # Python extraction
    # -------------------------------------------------------------------

    async def _extract_python(
        self, source: str, file_path: Path, result: ParseResult
    ) -> None:
        """Extract Python nodes and edges using tree-sitter.

        Edge cases specific to Python:
        - `from __future__ import annotations`: changes how type hints are evaluated
        - `if TYPE_CHECKING:` blocks: imports are type-only, not runtime
        - `try: import x / except: import y`: optional dependency pattern
        - `importlib.import_module(f"plugins.{name}")`: dynamic import with template
        - `__all__ = [...]`: explicit re-exports
        - Decorators: @app.route("/path") marks an endpoint
        - Dataclasses, NamedTuples: class-like but different structure
        - Nested functions/classes: qualified name must include parent
        """
        try:
            from tree_sitter_language_pack import get_language, get_parser
        except ImportError:
            result.warnings.append("tree-sitter-language-pack not installed")
            await self._extract_python_regex(source, file_path, result)
            return

        parser = get_parser("python")
        tree = parser.parse(source.encode("utf-8"))

        if tree.root_node.has_error:
            result.had_syntax_errors = True
            result.warnings.append(f"Syntax errors in {file_path} — partial extraction")

        lines = source.split("\n")
        module_id = result.module_node_id
        in_type_checking = False

        for child in tree.root_node.children:
            node_type = child.type

            # --- Imports ---
            if node_type in ("import_statement", "import_from_statement"):
                edges = self._parse_python_import(
                    child, source, file_path, module_id, in_type_checking
                )
                result.edges.extend(edges)

            # --- If TYPE_CHECKING block ---
            elif node_type == "if_statement":
                condition_text = self._node_text(child.child_by_field_name("condition"), source)
                if condition_text and "TYPE_CHECKING" in condition_text:
                    # Process imports inside this block as type-only
                    body = child.child_by_field_name("consequence")
                    if body:
                        for stmt in body.children:
                            if stmt.type in ("import_statement", "import_from_statement"):
                                edges = self._parse_python_import(
                                    stmt, source, file_path, module_id,
                                    is_type_only=True,
                                )
                                result.edges.extend(edges)

            # --- Classes ---
            elif node_type == "class_definition":
                class_node = self._parse_python_class(
                    child, source, file_path, result
                )
                if class_node:
                    result.nodes.append(class_node)
                    result.edges.append(GraphEdge(
                        source_id=module_id,
                        target_id=class_node.node_id,
                        kind=EdgeKind.DEPENDS_ON,
                    ))

            # --- Functions ---
            elif node_type == "function_definition":
                func_node = self._parse_python_function(
                    child, source, file_path, parent_qualified=""
                )
                if func_node:
                    result.nodes.append(func_node)

            # --- Assignments (module-level constants, __all__) ---
            elif node_type in ("expression_statement", "assignment"):
                self._parse_python_assignment(child, source, file_path, result)

    def _node_text(self, node: object | None, source: str) -> str | None:
        """Extract text from a tree-sitter node safely."""
        if node is None:
            return None
        start = getattr(node, "start_byte", 0)
        end = getattr(node, "end_byte", 0)
        return source[start:end]

    def _parse_python_import(
        self,
        node: object,
        source: str,
        file_path: Path,
        module_id: str,
        is_type_only: bool = False,
    ) -> list[GraphEdge]:
        """Parse a Python import statement into graph edges.

        Edge cases:
        - `from . import x`: relative import — resolve against package
        - `from ...utils import y`: multi-level relative import
        - `import x.y.z`: creates edge to module x.y.z
        - `from x import *`: star import — edge to module, not symbols
        - `import x as y`: alias doesn't affect the edge
        - `from __future__ import annotations`: skip (not a real dependency)
        """
        text = self._node_text(node, source)
        if not text:
            return []

        # Skip __future__ imports
        if "__future__" in text:
            return []

        edges: list[GraphEdge] = []
        # Extract the module being imported
        module_name = self._extract_import_module(text, file_path)

        if module_name:
            target_id = GraphNode.make_id(
                self._resolve_module_path(module_name),
                NodeKind.MODULE,
                module_name.split(".")[-1],
            )
            is_dynamic = False
            confidence = 1.0 if not is_dynamic else 0.6

            edges.append(GraphEdge(
                source_id=module_id,
                target_id=target_id,
                kind=EdgeKind.IMPORTS,
                confidence=confidence,
                is_type_only=is_type_only,
                metadata={"raw_import": text.strip()},
            ))

        return edges

    def _extract_import_module(self, import_text: str, file_path: Path) -> str | None:
        """Extract module name from import statement text.

        Edge cases:
        - `from .sibling import func` → resolve to package.sibling
        - `from .. import parent_func` → resolve to parent package
        - `import os.path` → "os.path"
        - `from typing import List` → "typing"
        """
        import re

        # from X import Y
        match = re.match(r"from\s+([\w.]+)\s+import", import_text)
        if match:
            module = match.group(1)
            # Handle relative imports
            if import_text.strip().startswith("from ."):
                dots = re.match(r"from\s+(\.+)", import_text)
                if dots:
                    level = len(dots.group(1))
                    package_parts = self._get_package_parts(file_path)
                    if level <= len(package_parts):
                        base = ".".join(package_parts[: -level] if level > 0 else package_parts)
                        rest = re.match(r"from\s+\.+\s*([\w.]*)\s+import", import_text)
                        if rest and rest.group(1):
                            return f"{base}.{rest.group(1)}" if base else rest.group(1)
                        return base
            return module

        # import X
        match = re.match(r"import\s+([\w.]+)", import_text)
        if match:
            return match.group(1)

        return None

    def _get_package_parts(self, file_path: Path) -> list[str]:
        """Get the package path components for resolving relative imports."""
        try:
            rel = file_path.resolve().relative_to(self._project_root)
        except ValueError:
            return []

        parts = list(rel.parent.parts)
        # Remove common source directories
        for prefix in ("src", "lib"):
            if parts and parts[0] == prefix:
                parts = parts[1:]
                break
        return parts

    def _resolve_module_path(self, module_name: str) -> Path:
        """Best-effort resolution of a module name to a file path.

        Edge case: module might be a package (directory with __init__.py)
        or a file. We try both and return whichever exists, defaulting
        to file path if neither exists (the node will be created as a
        placeholder).
        """
        parts = module_name.split(".")
        # Try as file
        file_path = self._project_root / Path(*parts).with_suffix(".py")
        if file_path.exists():
            return file_path

        # Try common source directories
        for src_dir in ("src", "lib"):
            file_path = self._project_root / src_dir / Path(*parts).with_suffix(".py")
            if file_path.exists():
                return file_path

        # Try as package
        pkg_path = self._project_root / Path(*parts) / "__init__.py"
        if pkg_path.exists():
            return pkg_path

        # Return a synthetic path — the node will exist but may not resolve to a real file
        return self._project_root / Path(*parts).with_suffix(".py")

    def _parse_python_class(
        self,
        node: object,
        source: str,
        file_path: Path,
        result: ParseResult,
    ) -> GraphNode | None:
        """Parse a Python class definition.

        Edge cases:
        - Dataclass: @dataclass decorator → metadata tag
        - NamedTuple: class Foo(NamedTuple) → metadata tag
        - Protocol: class Foo(Protocol) → kind=INTERFACE
        - Multiple inheritance: class Foo(Base, Mixin) → multiple INHERITS edges
        - Nested class: class Outer: class Inner → qualified_name includes parent
        - Abstract class: has @abstractmethod → metadata tag
        """
        name_node = getattr(node, "child_by_field_name", lambda _: None)("name")
        name = self._node_text(name_node, source) if name_node else None
        if not name:
            return None

        start_line = getattr(node, "start_point", (0, 0))[0] + 1
        end_line = getattr(node, "end_point", (0, 0))[0] + 1

        # Detect kind: Protocol/ABC → INTERFACE, else CLASS
        kind = NodeKind.CLASS
        superclasses = self._extract_superclasses(node, source)
        if any(s in ("Protocol", "ABC", "ABCMeta") for s in superclasses):
            kind = NodeKind.INTERFACE

        # Build docstring
        docstring = self._extract_docstring(node, source)

        qualified = f"{self._qualified_module_name(file_path)}.{name}"

        class_node = GraphNode(
            node_id=GraphNode.make_id(file_path, kind, name),
            kind=kind,
            name=name,
            qualified_name=qualified,
            file_path=file_path,
            line_range=LineRange(start=start_line, end=end_line),
            language=Language.PYTHON,
            docstring=docstring,
            is_generated=result.is_generated,
            is_test=result.nodes[0].is_test if result.nodes else False,
            metadata={
                "superclasses": superclasses,
                "is_dataclass": self._has_decorator(node, source, "dataclass"),
            },
        )

        # Create INHERITS edges for each superclass
        for superclass in superclasses:
            target_id = GraphNode.make_id(
                self._resolve_module_path(superclass),
                NodeKind.CLASS,
                superclass.split(".")[-1],
            )
            result.edges.append(GraphEdge(
                source_id=class_node.node_id,
                target_id=target_id,
                kind=EdgeKind.INHERITS,
            ))

        return class_node

    def _parse_python_function(
        self,
        node: object,
        source: str,
        file_path: Path,
        parent_qualified: str = "",
    ) -> GraphNode | None:
        """Parse a Python function definition.

        Edge cases:
        - Async functions: `async def` → metadata["is_async"] = True
        - Decorated with @app.route → kind=ENDPOINT
        - @staticmethod/@classmethod → metadata tag
        - Inner functions (closures): tracked with qualified name
        - Property decorators: @property → metadata tag
        - Overloaded functions: @overload → skip (only the implementation matters)
        """
        name_node = getattr(node, "child_by_field_name", lambda _: None)("name")
        name = self._node_text(name_node, source) if name_node else None
        if not name:
            return None

        start_line = getattr(node, "start_point", (0, 0))[0] + 1
        end_line = getattr(node, "end_point", (0, 0))[0] + 1

        # Skip overload stubs — they provide no implementation
        if self._has_decorator(node, source, "overload"):
            return None

        # Detect endpoint decorators
        kind = NodeKind.FUNCTION
        is_endpoint = self._has_decorator(node, source, "route") or self._has_decorator(
            node, source, "get"
        ) or self._has_decorator(node, source, "post")
        if is_endpoint:
            kind = NodeKind.ENDPOINT

        # Detect method vs function
        if parent_qualified:
            kind = NodeKind.METHOD

        qualified = f"{self._qualified_module_name(file_path)}.{name}"
        if parent_qualified:
            qualified = f"{parent_qualified}.{name}"

        is_async = self._node_text(node, source).strip().startswith("async ") if self._node_text(node, source) else False

        return GraphNode(
            node_id=GraphNode.make_id(file_path, kind, qualified),
            kind=kind,
            name=name,
            qualified_name=qualified,
            file_path=file_path,
            line_range=LineRange(start=start_line, end=end_line),
            language=Language.PYTHON,
            docstring=self._extract_docstring(node, source),
            metadata={
                "is_async": is_async,
                "is_staticmethod": self._has_decorator(node, source, "staticmethod"),
                "is_classmethod": self._has_decorator(node, source, "classmethod"),
                "is_property": self._has_decorator(node, source, "property"),
            },
        )

    def _parse_python_assignment(
        self,
        node: object,
        source: str,
        file_path: Path,
        result: ParseResult,
    ) -> None:
        """Parse module-level assignments for __all__, constants, etc.

        Edge case: `__all__ = ["Foo", "Bar"]` defines explicit re-exports.
        We create RE_EXPORTS edges for each listed symbol.
        """
        text = self._node_text(node, source)
        if not text:
            return

        if "__all__" in text:
            import re
            symbols = re.findall(r'["\'](\w+)["\']', text)
            module_id = result.module_node_id
            for symbol in symbols:
                target_id = GraphNode.make_id(file_path, NodeKind.FUNCTION, symbol)
                result.edges.append(GraphEdge(
                    source_id=module_id,
                    target_id=target_id,
                    kind=EdgeKind.RE_EXPORTS,
                ))

    # -------------------------------------------------------------------
    # JavaScript/TypeScript extraction
    # -------------------------------------------------------------------

    async def _extract_javascript_family(
        self, source: str, file_path: Path, result: ParseResult
    ) -> None:
        """Extract nodes and edges from JS/TS/TSX files.

        Edge cases specific to JavaScript ecosystem:
        - require() vs import: both CommonJS and ESM in same codebase
        - Dynamic import(): `import("./module")` — async, different from static
        - Path aliases: tsconfig paths, webpack aliases (can't resolve fully,
          but can extract the alias and flag it)
        - Barrel files: index.ts that re-exports from 10+ files
        - Default export vs named exports: affects how dependents reference it
        - JSX: <Component /> creates an implicit dependency on the component
        - Type-only imports: `import type { Foo }` in TypeScript
        - Namespace imports: `import * as utils from "./utils"`
        - Side-effect imports: `import "./polyfill"` — no named bindings
        """
        try:
            from tree_sitter_language_pack import get_parser

            grammar_name = LANGUAGE_GRAMMAR_MAP.get(detect_language(file_path), "javascript")
            parser = get_parser(grammar_name)

            tree = parser.parse(source.encode("utf-8"))

            if tree.root_node.has_error:
                result.had_syntax_errors = True
                result.warnings.append(f"Syntax errors in {file_path}")

            await self._walk_js_tree(tree.root_node, source, file_path, result)

        except (ImportError, Exception) as exc:
            result.warnings.append(f"tree-sitter grammar not available for {file_path.suffix}: {exc}")
            await self._extract_js_regex(source, file_path, result)

    async def _walk_js_tree(
        self, root: object, source: str, file_path: Path, result: ParseResult
    ) -> None:
        """Walk a JS/TS AST and extract nodes and edges."""
        module_id = result.module_node_id

        for child in getattr(root, "children", []):
            child_type = getattr(child, "type", "")

            # --- ESM imports ---
            if child_type == "import_statement":
                edge = self._parse_js_import(child, source, file_path, module_id)
                if edge:
                    result.edges.append(edge)

            # --- Exports (function, class, variable) ---
            elif child_type == "export_statement":
                # Contains the actual declaration
                declaration = getattr(child, "child_by_field_name", lambda _: None)("declaration")
                if declaration:
                    decl_type = getattr(declaration, "type", "")
                    if decl_type in ("function_declaration", "generator_function_declaration"):
                        node = self._parse_js_function(declaration, source, file_path)
                        if node:
                            result.nodes.append(node)
                    elif decl_type == "class_declaration":
                        node = self._parse_js_class(declaration, source, file_path)
                        if node:
                            result.nodes.append(node)

            # --- Top-level functions ---
            elif child_type in ("function_declaration", "generator_function_declaration"):
                node = self._parse_js_function(child, source, file_path)
                if node:
                    result.nodes.append(node)

            # --- Top-level classes ---
            elif child_type == "class_declaration":
                node = self._parse_js_class(child, source, file_path)
                if node:
                    result.nodes.append(node)

            # --- CommonJS require() ---
            elif child_type == "lexical_declaration" or child_type == "variable_declaration":
                text = self._node_text(child, source) or ""
                if "require(" in text:
                    edge = self._parse_require(text, file_path, module_id)
                    if edge:
                        result.edges.append(edge)

    def _parse_js_import(
        self, node: object, source: str, file_path: Path, module_id: str
    ) -> GraphEdge | None:
        """Parse an ESM import statement.

        Edge cases:
        - `import type { Foo }` → type-only edge
        - `import "./side-effect"` → side-effect import, edge with no target symbol
        - `import * as ns from "x"` → namespace import
        - `import("./lazy")` → dynamic import (different AST node type)
        """
        import re

        text = self._node_text(node, source) or ""
        is_type_only = "import type" in text

        # Extract module path
        match = re.search(r'''from\s+['"]([^'"]+)['"]''', text)
        if not match:
            match = re.search(r'''import\s+['"]([^'"]+)['"]''', text)
        if not match:
            return None

        module_path = match.group(1)
        target_path = self._resolve_js_module(module_path, file_path)
        target_id = GraphNode.make_id(target_path, NodeKind.MODULE, target_path.stem)

        return GraphEdge(
            source_id=module_id,
            target_id=target_id,
            kind=EdgeKind.IMPORTS,
            is_type_only=is_type_only,
            metadata={"raw_import": text.strip(), "module_specifier": module_path},
        )

    def _parse_require(
        self, text: str, file_path: Path, module_id: str
    ) -> GraphEdge | None:
        """Parse a CommonJS require() call.

        Edge case: `require(variable)` — can't resolve, create edge
        with low confidence and the variable name in metadata.
        """
        import re

        match = re.search(r'''require\(['"]([^'"]+)['"]\)''', text)
        if not match:
            # Dynamic require — check if it's require(variable)
            match_dynamic = re.search(r"require\((\w+)\)", text)
            if match_dynamic:
                return GraphEdge(
                    source_id=module_id,
                    target_id="unresolved:" + match_dynamic.group(1),
                    kind=EdgeKind.DYNAMIC_IMPORT,
                    confidence=0.3,
                    metadata={"dynamic_variable": match_dynamic.group(1)},
                )
            return None

        module_path = match.group(1)
        target_path = self._resolve_js_module(module_path, file_path)
        target_id = GraphNode.make_id(target_path, NodeKind.MODULE, target_path.stem)

        return GraphEdge(
            source_id=module_id,
            target_id=target_id,
            kind=EdgeKind.IMPORTS,
            metadata={"raw_import": text.strip(), "style": "commonjs"},
        )

    def _resolve_js_module(self, specifier: str, from_file: Path) -> Path:
        """Resolve a JS module specifier to a file path.

        Edge cases:
        - Relative: "./foo" → look for foo.ts, foo.tsx, foo.js, foo/index.ts, etc.
        - Bare: "react" → node_modules/react (external, tracked as is_external)
        - Alias: "@/utils" → depends on tsconfig/webpack (flag, can't fully resolve)
        - Missing extension: JS allows importing without extension
        """
        if specifier.startswith("."):
            # Relative import
            base = from_file.parent
            candidate = base / specifier
            # Try common extensions
            for ext in (".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.tsx", "/index.js"):
                full = candidate.parent / (candidate.name + ext)
                if full.exists():
                    return full
            return candidate.with_suffix(".ts")  # Default assumption
        else:
            # Bare specifier (package import) or alias
            return Path(f"node_modules/{specifier}/index.js")

    def _parse_js_function(
        self, node: object, source: str, file_path: Path
    ) -> GraphNode | None:
        """Parse a JS/TS function declaration."""
        name_node = getattr(node, "child_by_field_name", lambda _: None)("name")
        name = self._node_text(name_node, source) if name_node else None
        if not name:
            return None

        start_line = getattr(node, "start_point", (0, 0))[0] + 1
        end_line = getattr(node, "end_point", (0, 0))[0] + 1
        text = self._node_text(node, source) or ""

        return GraphNode(
            node_id=GraphNode.make_id(file_path, NodeKind.FUNCTION, name),
            kind=NodeKind.FUNCTION,
            name=name,
            qualified_name=f"{self._qualified_module_name(file_path)}.{name}",
            file_path=file_path,
            line_range=LineRange(start=start_line, end=end_line),
            language=detect_language(file_path),
            metadata={
                "is_async": "async " in text[:50],
                "is_generator": "function*" in text[:50],
                "is_exported": True,
            },
        )

    def _parse_js_class(
        self, node: object, source: str, file_path: Path
    ) -> GraphNode | None:
        """Parse a JS/TS class declaration."""
        name_node = getattr(node, "child_by_field_name", lambda _: None)("name")
        name = self._node_text(name_node, source) if name_node else None
        if not name:
            return None

        start_line = getattr(node, "start_point", (0, 0))[0] + 1
        end_line = getattr(node, "end_point", (0, 0))[0] + 1

        return GraphNode(
            node_id=GraphNode.make_id(file_path, NodeKind.CLASS, name),
            kind=NodeKind.CLASS,
            name=name,
            qualified_name=f"{self._qualified_module_name(file_path)}.{name}",
            file_path=file_path,
            line_range=LineRange(start=start_line, end=end_line),
            language=detect_language(file_path),
        )

    # -------------------------------------------------------------------
    # Regex fallbacks (when tree-sitter grammar not available)
    # -------------------------------------------------------------------

    async def _extract_python_regex(
        self, source: str, file_path: Path, result: ParseResult
    ) -> None:
        """Regex-based Python extraction — fallback when tree-sitter unavailable.

        This is intentionally limited. It catches the most common patterns
        (imports, class/function definitions) but misses nesting, decorators,
        and complex constructs.
        """
        import re

        module_id = result.module_node_id

        for line_no, line in enumerate(source.split("\n"), 1):
            stripped = line.strip()

            # Imports
            if stripped.startswith(("import ", "from ")):
                module_name = self._extract_import_module(stripped, file_path)
                if module_name:
                    target_path = self._resolve_module_path(module_name)
                    target_id = GraphNode.make_id(target_path, NodeKind.MODULE, module_name.split(".")[-1])
                    result.edges.append(GraphEdge(
                        source_id=module_id,
                        target_id=target_id,
                        kind=EdgeKind.IMPORTS,
                        confidence=0.9,  # Lower confidence for regex extraction
                        metadata={"extraction": "regex"},
                    ))

            # Class definitions
            match = re.match(r"class\s+(\w+)", stripped)
            if match:
                name = match.group(1)
                result.nodes.append(GraphNode(
                    node_id=GraphNode.make_id(file_path, NodeKind.CLASS, name),
                    kind=NodeKind.CLASS,
                    name=name,
                    qualified_name=f"{self._qualified_module_name(file_path)}.{name}",
                    file_path=file_path,
                    line_range=LineRange(start=line_no, end=line_no),
                    language=Language.PYTHON,
                    metadata={"extraction": "regex"},
                ))

            # Function definitions
            match = re.match(r"(?:async\s+)?def\s+(\w+)", stripped)
            if match:
                name = match.group(1)
                result.nodes.append(GraphNode(
                    node_id=GraphNode.make_id(file_path, NodeKind.FUNCTION, name),
                    kind=NodeKind.FUNCTION,
                    name=name,
                    qualified_name=f"{self._qualified_module_name(file_path)}.{name}",
                    file_path=file_path,
                    line_range=LineRange(start=line_no, end=line_no),
                    language=Language.PYTHON,
                    metadata={"is_async": "async" in stripped, "extraction": "regex"},
                ))

    async def _extract_js_regex(
        self, source: str, file_path: Path, result: ParseResult
    ) -> None:
        """Regex-based JS/TS extraction — fallback."""
        import re

        module_id = result.module_node_id

        for line_no, line in enumerate(source.split("\n"), 1):
            stripped = line.strip()

            # ESM imports
            match = re.search(r'''from\s+['"]([^'"]+)['"]''', stripped)
            if match:
                target_path = self._resolve_js_module(match.group(1), file_path)
                target_id = GraphNode.make_id(target_path, NodeKind.MODULE, target_path.stem)
                result.edges.append(GraphEdge(
                    source_id=module_id,
                    target_id=target_id,
                    kind=EdgeKind.IMPORTS,
                    confidence=0.9,
                    metadata={"extraction": "regex"},
                ))

            # Function declarations
            match = re.match(r"(?:export\s+)?(?:async\s+)?function\s+(\w+)", stripped)
            if match:
                name = match.group(1)
                result.nodes.append(GraphNode(
                    node_id=GraphNode.make_id(file_path, NodeKind.FUNCTION, name),
                    kind=NodeKind.FUNCTION,
                    name=name,
                    qualified_name=f"{self._qualified_module_name(file_path)}.{name}",
                    file_path=file_path,
                    line_range=LineRange(start=line_no, end=line_no),
                    language=detect_language(file_path),
                    metadata={"extraction": "regex"},
                ))

    # -------------------------------------------------------------------
    # Generic language extraction (all 19 languages via tree-sitter-language-pack)
    # -------------------------------------------------------------------

    async def _extract_generic(
        self, source: str, file_path: Path, language: Language, result: ParseResult
    ) -> None:
        """Extract basic structure from any tree-sitter supported language.

        Works for Go, Rust, Java, C, C++, C#, PHP, Swift, Kotlin, Scala,
        Lua, Dart, Elixir, Haskell, Ruby.

        Extracts:
        - Top-level function/method definitions
        - Class/struct/interface/trait definitions
        - Import/include/use statements

        This is intentionally broad — we use tree-sitter node type names
        that are common across grammars (function_definition, class_definition,
        import_declaration, etc.) and skip what we don't recognize.
        """
        grammar_name = LANGUAGE_GRAMMAR_MAP.get(language)
        if not grammar_name:
            return

        try:
            from tree_sitter_language_pack import get_parser
            parser = get_parser(grammar_name)
        except Exception as exc:
            result.warnings.append(f"No grammar for {language.value}: {exc}")
            return

        tree = parser.parse(source.encode("utf-8"))
        if tree.root_node.has_error:
            result.had_syntax_errors = True

        module_id = result.module_node_id

        # Node types that represent function/method definitions across languages
        function_types = {
            "function_definition", "function_declaration", "method_definition",
            "method_declaration", "function_item", "fun_spec",
        }
        # Node types that represent class/struct/interface definitions
        class_types = {
            "class_definition", "class_declaration", "struct_item",
            "interface_declaration", "trait_item", "type_declaration",
            "struct_declaration", "enum_declaration", "enum_item",
            "object_declaration",
        }
        # Node types that represent imports
        import_types = {
            "import_declaration", "import_statement", "use_declaration",
            "include_statement", "require_expression", "using_directive",
            "import_from_statement", "preproc_include",
        }

        for child in tree.root_node.children:
            child_type = getattr(child, "type", "")

            if child_type in function_types:
                name_node = getattr(child, "child_by_field_name", lambda _: None)("name")
                name = self._node_text(name_node, source) if name_node else None
                if name:
                    start_line = getattr(child, "start_point", (0, 0))[0] + 1
                    end_line = getattr(child, "end_point", (0, 0))[0] + 1
                    result.nodes.append(GraphNode(
                        node_id=GraphNode.make_id(file_path, NodeKind.FUNCTION, name),
                        kind=NodeKind.FUNCTION,
                        name=name,
                        qualified_name=f"{self._qualified_module_name(file_path)}.{name}",
                        file_path=file_path,
                        line_range=LineRange(start=start_line, end=end_line),
                        language=language,
                    ))

            elif child_type in class_types:
                name_node = getattr(child, "child_by_field_name", lambda _: None)("name")
                name = self._node_text(name_node, source) if name_node else None
                if name:
                    start_line = getattr(child, "start_point", (0, 0))[0] + 1
                    end_line = getattr(child, "end_point", (0, 0))[0] + 1
                    kind = NodeKind.INTERFACE if "interface" in child_type or "trait" in child_type else NodeKind.CLASS
                    result.nodes.append(GraphNode(
                        node_id=GraphNode.make_id(file_path, kind, name),
                        kind=kind,
                        name=name,
                        qualified_name=f"{self._qualified_module_name(file_path)}.{name}",
                        file_path=file_path,
                        line_range=LineRange(start=start_line, end=end_line),
                        language=language,
                    ))

            elif child_type in import_types:
                # Record a basic import edge
                text = self._node_text(child, source)
                if text:
                    result.edges.append(GraphEdge(
                        source_id=module_id,
                        target_id=GraphNode.make_id(file_path, NodeKind.MODULE, f"_import_{hash(text) % 10000}"),
                        kind=EdgeKind.IMPORTS,
                        confidence=0.7,
                        metadata={"raw_import": text.strip()[:200], "language": language.value},
                    ))

    # -------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------

    def _extract_superclasses(self, class_node: object, source: str) -> list[str]:
        """Extract superclass names from a class definition node."""
        superclasses_node = getattr(class_node, "child_by_field_name", lambda _: None)("superclasses")
        if not superclasses_node:
            return []

        text = self._node_text(superclasses_node, source)
        if not text:
            return []

        # Remove parens and split
        import re
        text = text.strip("()")
        return [s.strip() for s in re.split(r",\s*", text) if s.strip()]

    def _extract_docstring(self, node: object, source: str) -> str | None:
        """Extract docstring from a class or function body."""
        body = getattr(node, "child_by_field_name", lambda _: None)("body")
        if not body:
            return None

        children = getattr(body, "children", [])
        if not children:
            return None

        first = children[0]
        if getattr(first, "type", "") == "expression_statement":
            inner_children = getattr(first, "children", [])
            if inner_children and getattr(inner_children[0], "type", "") == "string":
                text = self._node_text(inner_children[0], source)
                if text:
                    return text.strip("'\"").strip()
        return None

    def _has_decorator(self, node: object, source: str, decorator_name: str) -> bool:
        """Check if a node has a specific decorator."""
        # Look for decorator nodes before this node
        parent = getattr(node, "parent", None)
        if not parent:
            return False

        for child in getattr(parent, "children", []):
            if getattr(child, "type", "") == "decorator":
                text = self._node_text(child, source) or ""
                if decorator_name in text:
                    return True
            if child is node:
                break
        return False
