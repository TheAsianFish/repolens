"""
chunker.py

Parses Python, JavaScript, and TypeScript source files into semantically
complete chunks using Tree-sitter AST parsing. Each chunk represents
exactly one function or class definition — never an arbitrary line slice.

Architecture notes:
- One Parser per language, cached at module level (_PARSER_CACHE).
- Language detected from file extension via EXTENSION_TO_LANGUAGE.
- JS/TS chunks use the same Chunk dataclass as Python chunks.
- Downstream systems (store, retriever, LLM) are language-agnostic.
"""

import tiktoken
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts
from tree_sitter import Language, Parser

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Maps file extension → language name used throughout this module.
EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py":  "python",
    ".ts":  "typescript",
    ".tsx": "tsx",
    ".js":  "javascript",
    ".jsx": "javascript",
}

# Python AST node types that produce chunks.
CHUNK_NODE_TYPES = {"function_definition", "class_definition"}

# JS/TS AST node types that produce chunks.
JS_CHUNK_NODE_TYPES = {
    "function_declaration",
    "arrow_function",
    "function_expression",
    "class_declaration",
    "method_definition",
}

MAX_CHUNK_TOKENS = 300

# Languages where JS/TS walking logic applies.
_JS_LANGUAGES = {"javascript", "typescript", "tsx"}

# One Parser instance per language, created on first use.
_PARSER_CACHE: dict[str, Any] = {}


def _get_cached_parser(language: str) -> Parser:
    """Return a cached Parser for the given language, creating it if needed."""
    if language not in _PARSER_CACHE:
        if language == "python":
            lang = Language(tspython.language())
        elif language == "javascript":
            lang = Language(tsjs.language())
        elif language == "typescript":
            lang = Language(tsts.language_typescript())
        elif language == "tsx":
            lang = Language(tsts.language_tsx())
        else:
            raise ValueError(f"Unsupported language: {language}")
        _PARSER_CACHE[language] = Parser(lang)
    return _PARSER_CACHE[language]


@dataclass
class Chunk:
    """
    A single semantically complete unit of source code.

    Every chunk is exactly one function or class definition.
    Metadata fields support downstream re-ranking and call graph
    expansion.
    """
    file_path: str
    node_type: str          # "function" or "class" (JS/TS); "function_definition" or "class_definition" (Python)
    name: str               # Function or class name
    source: str             # Raw source text of this chunk
    start_line: int         # 1-indexed, inclusive
    end_line: int           # 1-indexed, inclusive
    token_count: int        # Exact token count via tiktoken
    calls: list[str]        # Names of functions called within this chunk
    docstring: str | None   # First string literal if used as docstring
    parent_class: str | None   # Enclosing class name for methods
    is_truncated: bool      # True if source was cut at MAX_CHUNK_TOKENS


def count_tokens(text: str) -> int:
    """Return the exact token count for text using cl100k_base."""
    return len(_TOKENIZER.encode(text))


def extract_name(node, source_bytes: bytes) -> str:
    """
    Extract the name identifier from a Python function or class AST node.
    """
    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
    return "<unknown>"


def extract_calls(node, source_bytes: bytes) -> list[str]:
    """
    Walk a Python function or class node and collect names of all
    functions called within it.
    """
    found: set[str] = set()
    _collect_calls(node, source_bytes, found)
    return sorted(found)


def _collect_calls(node, source_bytes: bytes, found: set[str]) -> None:
    """Recursive helper for extract_calls (Python)."""
    for child in node.children:
        if child.type == "call":
            func_node = child.children[0] if child.children else None
            if func_node is not None:
                if func_node.type == "identifier":
                    found.add(
                        source_bytes[
                            func_node.start_byte:func_node.end_byte
                        ].decode("utf-8")
                    )
                elif func_node.type == "attribute":
                    identifiers = [
                        c for c in func_node.children
                        if c.type == "identifier"
                    ]
                    if identifiers:
                        last = identifiers[-1]
                        found.add(
                            source_bytes[
                                last.start_byte:last.end_byte
                            ].decode("utf-8")
                        )
        _collect_calls(child, source_bytes, found)


def extract_docstring(node, source_bytes: bytes) -> str | None:
    """
    Extract the docstring from a Python function or class node if one exists.
    """
    body = None
    for child in node.children:
        if child.type == "block":
            body = child
            break

    if body is None or not body.children:
        return None

    first_stmt = None
    for child in body.children:
        if child.type not in {"newline", "comment", "indent"}:
            first_stmt = child
            break

    if first_stmt is None or first_stmt.type != "expression_statement":
        return None

    for child in first_stmt.children:
        if child.type == "string":
            raw = source_bytes[child.start_byte:child.end_byte].decode("utf-8")
            return raw.strip('"""').strip("'''").strip('"').strip("'").strip()

    return None


def _extract_js_calls(node) -> list[str]:
    """
    Recursively scan a JS/TS AST node for call_expression nodes.
    For each call_expression, extract the callee name.
    Returns a deduplicated list of callee name strings preserving
    first-seen order.
    """
    calls: list[str] = []

    if node.type == "call_expression":
        callee = node.children[0] if node.children else None
        if callee is not None:
            if callee.type == "identifier":
                calls.append(callee.text.decode("utf-8"))
            elif callee.type == "member_expression":
                prop = callee.child_by_field_name("property")
                if prop:
                    calls.append(prop.text.decode("utf-8"))

    for child in node.children:
        calls.extend(_extract_js_calls(child))

    seen: set[str] = set()
    result: list[str] = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _extract_js_name_from_parent(node) -> str:
    """
    Infer the name of an arrow_function or function_expression from its
    parent AST node.

    Returns the identifier name if the parent is a variable_declarator
    (const/let/var foo = ...) or an assignment_expression (foo = ...).
    Returns "" if no name can be determined.
    """
    parent = node.parent
    if parent is None:
        return ""
    if parent.type == "variable_declarator":
        for child in parent.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
    elif parent.type == "assignment_expression":
        left = parent.children[0] if parent.children else None
        if left is not None and left.type == "identifier":
            return left.text.decode("utf-8")
    return ""


def _make_chunk(
    node,
    file_path: str,
    node_type: str,
    name: str,
    calls: list[str],
    docstring: str | None,
    parent_class: str | None,
) -> Chunk:
    """Build a Chunk from an AST node, applying the token cap."""
    source_text = node.text.decode("utf-8")
    token_count = count_tokens(source_text)
    is_truncated = False

    if token_count > MAX_CHUNK_TOKENS:
        encoded = _TOKENIZER.encode(source_text)
        source_text = _TOKENIZER.decode(encoded[:MAX_CHUNK_TOKENS])
        token_count = MAX_CHUNK_TOKENS
        is_truncated = True

    return Chunk(
        file_path=file_path,
        node_type=node_type,
        name=name,
        source=source_text,
        start_line=node.start_point[0] + 1,
        end_line=node.end_point[0] + 1,
        token_count=token_count,
        calls=calls,
        docstring=docstring,
        parent_class=parent_class,
        is_truncated=is_truncated,
    )


def chunk_file(file_path: str | Path) -> list[Chunk]:
    """
    Parse a source file and return a list of Chunk objects, one per
    top-level or class-level function or class definition.

    Supported extensions: .py, .ts, .tsx, .js, .jsx
    Unknown extensions return an empty list without raising.

    Args:
        file_path: Path to a source file.

    Returns:
        List of Chunk objects sorted by start_line.

    Raises:
        ValueError: If the file does not exist.
        OSError: If the file cannot be read.
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path}")

    ext = file_path.suffix.lower()
    language = EXTENSION_TO_LANGUAGE.get(ext)
    if language is None:
        return []

    parser = _get_cached_parser(language)
    source_bytes = file_path.read_bytes()
    tree = parser.parse(source_bytes)

    chunks: list[Chunk] = []
    _walk_tree(tree.root_node, source_bytes, str(file_path), chunks, language)

    return sorted(chunks, key=lambda c: c.start_line)


def _walk_tree(
    node,
    source_bytes: bytes,
    file_path: str,
    chunks: list[Chunk],
    language: str,
    enclosing_class: str | None = None,
) -> None:
    """
    Recursively walk the AST and extract function and class nodes.

    Dispatches to Python or JS/TS handling based on the language parameter.
    Tracks enclosing class context so methods carry parent_class metadata.
    """
    for child in node.children:
        if language == "python" and child.type in CHUNK_NODE_TYPES:
            _handle_python_node(
                child, node, source_bytes, file_path, chunks, language, enclosing_class
            )
        elif language in _JS_LANGUAGES and child.type in JS_CHUNK_NODE_TYPES:
            _handle_js_node(
                child, source_bytes, file_path, chunks, language, enclosing_class
            )
        else:
            _walk_tree(child, source_bytes, file_path, chunks, language, enclosing_class)


def _handle_python_node(
    child,
    node,
    source_bytes: bytes,
    file_path: str,
    chunks: list[Chunk],
    language: str,
    enclosing_class: str | None,
) -> None:
    """
    Extract a Python function or class chunk and descend into class bodies.

    When a function or class is wrapped in a decorated_definition node,
    the source text is taken from that parent so decorator lines are included.
    """
    source_node = node if node.type == "decorated_definition" else child

    source_text = source_bytes[
        source_node.start_byte:source_node.end_byte
    ].decode("utf-8")

    token_count = count_tokens(source_text)
    is_truncated = False

    if token_count > MAX_CHUNK_TOKENS:
        encoded = _TOKENIZER.encode(source_text)
        source_text = _TOKENIZER.decode(encoded[:MAX_CHUNK_TOKENS])
        token_count = MAX_CHUNK_TOKENS
        is_truncated = True

    name = extract_name(child, source_bytes)

    chunks.append(Chunk(
        file_path=file_path,
        node_type=child.type,
        name=name,
        source=source_text,
        start_line=source_node.start_point[0] + 1,
        end_line=source_node.end_point[0] + 1,
        token_count=token_count,
        calls=extract_calls(child, source_bytes),
        docstring=extract_docstring(child, source_bytes),
        parent_class=enclosing_class,
        is_truncated=is_truncated,
    ))

    if child.type == "class_definition":
        _walk_tree(child, source_bytes, file_path, chunks, language, enclosing_class=name)


def _handle_js_node(
    child,
    source_bytes: bytes,
    file_path: str,
    chunks: list[Chunk],
    language: str,
    enclosing_class: str | None,
) -> None:
    """
    Extract a JS/TS function, class, or method chunk.

    For class_declaration, descend into the class body so methods are
    chunked separately with parent_class set. For all function types,
    stop descending to avoid double-chunking nested functions.
    """
    ntype = child.type

    if ntype == "function_declaration":
        name = ""
        for c in child.children:
            if c.type == "identifier":
                name = c.text.decode("utf-8")
                break
        if not name:
            return
        chunks.append(_make_chunk(
            child, file_path, "function", name,
            calls=_extract_js_calls(child),
            docstring="",
            parent_class=enclosing_class,
        ))

    elif ntype in ("arrow_function", "function_expression"):
        name = _extract_js_name_from_parent(child)
        if not name:
            return
        chunks.append(_make_chunk(
            child, file_path, "function", name,
            calls=_extract_js_calls(child),
            docstring="",
            parent_class=enclosing_class,
        ))

    elif ntype == "class_declaration":
        name = ""
        for c in child.children:
            if c.type == "identifier":
                name = c.text.decode("utf-8")
                break
        if not name:
            return
        chunks.append(_make_chunk(
            child, file_path, "class", name,
            calls=[],
            docstring="",
            parent_class=enclosing_class,
        ))
        _walk_tree(child, source_bytes, file_path, chunks, language, enclosing_class=name)

    elif ntype == "method_definition":
        name = ""
        for c in child.children:
            if c.type == "property_identifier":
                name = c.text.decode("utf-8")
                break
        if not name:
            return
        chunks.append(_make_chunk(
            child, file_path, "function", name,
            calls=_extract_js_calls(child),
            docstring="",
            parent_class=enclosing_class,
        ))
