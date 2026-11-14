"""Parse PostgreSQL plain dumps into a SQLite database.

The Coffman and Weaver files use two dump styles. Mondial loads rows with
INSERT statements. IMDb and Wikipedia load rows with COPY ... FROM stdin.
Schema files add primary keys and foreign keys with ALTER TABLE.
"""

from __future__ import annotations

import bz2
import logging
import re
import sqlite3
from pathlib import Path

from semstruct.data.schema import (
    Attribute,
    ForeignKey,
    Relation,
    Schema,
    quote_ident,
    unqualified,
)

logger = logging.getLogger(__name__)

_TYPE_RE = re.compile(
    r"^(?:"
    r"character\s+varying|double\s+precision|timestamp\s+with(?:out)?\s+time\s+zone|"
    r"time\s+with(?:out)?\s+time\s+zone|bit\s+varying|character|varchar|char|"
    r"numeric|decimal|smallint|bigint|integer|int|real|text|boolean|bool|date|"
    r"bytea|tsvector|serial|bigserial|float|double|name|json|jsonb|oid|"
    r"smallserial|interval|money|uuid|xml"
    r")(?:\s*\([^)]*\))?(?:\[\])?",
    re.IGNORECASE | re.DOTALL,
)
_COPY_RE = re.compile(
    r"^COPY\s+(.+?)\s*\(([^)]*)\)\s+FROM\s+stdin\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_COPY_NO_COLS_RE = re.compile(
    r"^COPY\s+(.+?)\s+FROM\s+stdin\s*;?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_INSERT_RE = re.compile(
    r"^INSERT\s+INTO\s+(\S+)\s*(?:\(([^)]*)\))?\s*VALUES\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_ALTER_RE = re.compile(
    r"^ALTER\s+TABLE\s+(?:ONLY\s+)?(.+?)\s+ADD\s+CONSTRAINT\s+\S+\s+(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_PK_RE = re.compile(r"^PRIMARY\s+KEY\s*\(([^)]*)\)", re.IGNORECASE | re.DOTALL)
_FK_RE = re.compile(
    r"^FOREIGN\s+KEY\s*\(([^)]*)\)\s*REFERENCES\s+(\S+)\s*\(([^)]*)\)",
    re.IGNORECASE | re.DOTALL,
)
_COMMENT_RE = re.compile(
    r"^COMMENT\s+ON\s+TABLE\s+(.+?)\s+IS\s+'(.*)'\s*$",
    re.IGNORECASE | re.DOTALL,
)
_TABLE_CONSTRAINT_RE = re.compile(
    r"^(?:CONSTRAINT\s+\S+\s+)?(?:PRIMARY\s+KEY|FOREIGN\s+KEY|UNIQUE|CHECK)\b",
    re.IGNORECASE | re.DOTALL,
)


def open_sql(path: Path):
    if path.name.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return path.open("r", encoding="utf-8", errors="replace", newline="")


def iter_sql_events(stream):
    """Yield (`sql`, statement), (`copy_start`, statement), (`copy_row`, line), (`copy_end`, None).

    Scanning uses substring search so multi-megabyte `INSERT` statements stay
    practical. PostgreSQL escapes are left intact for the literal parser.
    """
    chunks: list[str] = []
    state = "code"
    dollar_tag: str | None = None
    copy_mode = False
    for line in stream:
        if copy_mode:
            raw = line[:-1] if line.endswith("\n") else line
            if raw.endswith("\r"):
                raw = raw[:-1]
            if raw == "\\.":
                copy_mode = False
                yield ("copy_end", None)
            else:
                yield ("copy_row", raw)
            continue
        index = 0
        length = len(line)
        while index < length:
            if state == "block_comment":
                end = line.find("*/", index)
                if end == -1:
                    index = length
                    break
                state = "code"
                index = end + 2
                continue
            if state == "dollar":
                assert dollar_tag is not None
                end = line.find(dollar_tag, index)
                if end == -1:
                    chunks.append(line[index:])
                    index = length
                    break
                chunks.append(line[index : end + len(dollar_tag)])
                index = end + len(dollar_tag)
                state = "code"
                dollar_tag = None
                continue
            if state == "single":
                quote = line.find("'", index)
                slash = line.find("\\", index)
                if quote == -1 and slash == -1:
                    chunks.append(line[index:])
                    index = length
                    break
                if slash != -1 and (quote == -1 or slash < quote):
                    if slash + 1 < length:
                        chunks.append(line[index : slash + 2])
                        index = slash + 2
                        continue
                    chunks.append(line[index:])
                    index = length
                    break
                if quote + 1 < length and line[quote + 1] == "'":
                    chunks.append(line[index : quote + 2])
                    index = quote + 2
                    continue
                chunks.append(line[index : quote + 1])
                index = quote + 1
                state = "code"
                continue
            semicolon = line.find(";", index)
            comment = line.find("--", index)
            block = line.find("/*", index)
            quote = line.find("'", index)
            dollar = line.find("$", index)
            candidates = [
                (position, marker)
                for position, marker in (
                    (semicolon, ";"),
                    (comment, "--"),
                    (block, "/*"),
                    (quote, "'"),
                    (dollar, "$"),
                )
                if position != -1
            ]
            if not candidates:
                chunks.append(line[index:])
                index = length
                break
            position, marker = min(candidates)
            chunks.append(line[index:position])
            if marker == ";":
                text = "".join(chunks).strip()
                chunks = []
                if text:
                    yield ("sql", text)
                    if _is_copy_stdin(text):
                        yield ("copy_start", text)
                        copy_mode = True
                        index = length
                        break
                index = position + 1
                continue
            if marker == "--":
                index = length
                break
            if marker == "/*":
                state = "block_comment"
                index = position + 2
                continue
            if marker == "'":
                chunks.append("'")
                state = "single"
                index = position + 1
                continue
            tag = _dollar_tag(line, position)
            if tag is None:
                chunks.append("$")
                index = position + 1
                continue
            chunks.append(tag)
            state = "dollar"
            dollar_tag = tag
            index = position + len(tag)
    trailing = "".join(chunks).strip()
    if trailing and state == "code":
        yield ("sql", trailing)


def _dollar_tag(line: str, index: int) -> str | None:
    if line[index] != "$":
        return None
    end = index + 1
    while end < len(line) and (line[end].isalnum() or line[end] == "_"):
        end += 1
    if end < len(line) and line[end] == "$":
        return line[index : end + 1]
    return None


def _is_copy_stdin(statement: str) -> bool:
    return bool(re.match(r"COPY\s", statement, re.IGNORECASE) and re.search(r"FROM\s+stdin\s*$", statement, re.IGNORECASE))


def read_schema(path: Path) -> Schema:
    schema = Schema()
    with open_sql(path) as stream:
        for kind, payload in iter_sql_events(stream):
            if kind == "sql":
                apply_sql(schema, payload)
    return schema


def apply_sql(schema: Schema, statement: str) -> None:
    text = statement.strip().rstrip(";").strip()
    if not text:
        return
    head = text[:24].upper()
    if head.startswith("CREATE TABLE"):
        relation = parse_create_table(text)
        if relation is not None and relation.attributes:
            schema.add(relation)
        return
    if head.startswith("ALTER TABLE"):
        apply_alter(schema, text)
        return
    if head.startswith("COMMENT ON TABLE"):
        apply_comment(schema, text)


def parse_create_table(statement: str) -> Relation | None:
    match = re.match(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?", statement, re.IGNORECASE)
    if match is None:
        return None
    index = match.end()
    while index < len(statement) and statement[index].isspace():
        index += 1
    name_start = index
    in_quote = False
    while index < len(statement):
        char = statement[index]
        if char == '"':
            in_quote = not in_quote
        elif char == "(" and not in_quote:
            break
        index += 1
    if index >= len(statement) or statement[index] != "(":
        return None
    table_name = unqualified(statement[name_start:index])
    body = _balanced(statement, index)
    if body is None:
        return None
    relation = Relation(name=table_name)
    for item in _split_commas(body):
        piece = item.strip()
        if not piece:
            continue
        if _TABLE_CONSTRAINT_RE.match(piece):
            _apply_table_constraint(relation, piece)
            continue
        parsed = _parse_column(piece)
        if parsed is None:
            continue
        attribute, foreign_key = parsed
        relation.attributes.append(attribute)
        if attribute.primary_key and attribute.name not in relation.primary_key:
            relation.primary_key.append(attribute.name)
        if foreign_key is not None:
            _add_foreign_key(relation, foreign_key)
    return relation


def apply_alter(schema: Schema, statement: str) -> None:
    match = _ALTER_RE.match(statement.strip().rstrip(";").strip())
    if match is None:
        return
    table_name = unqualified(match.group(1))
    relation = schema.relations.get(table_name)
    if relation is None:
        return
    body = match.group(2).strip()
    primary = _PK_RE.match(body)
    if primary is not None:
        relation.primary_key = _split_idents(primary.group(1))
        names = set(relation.primary_key)
        for attribute in relation.attributes:
            attribute.primary_key = attribute.name in names
        return
    foreign = _FK_RE.match(body)
    if foreign is None:
        return
    columns = _split_idents(foreign.group(1))
    ref_table = unqualified(foreign.group(2))
    ref_columns = _split_idents(foreign.group(3))
    _add_foreign_key(relation, ForeignKey(columns, ref_table, ref_columns))


def apply_comment(schema: Schema, statement: str) -> None:
    match = _COMMENT_RE.match(statement.strip().rstrip(";").strip())
    if match is None:
        return
    relation = schema.relations.get(unqualified(match.group(1)))
    if relation is not None:
        relation.comment = match.group(2).replace("''", "'")


def parse_insert(statement: str) -> tuple[str, list[str] | None, list[list[str | None]]] | None:
    text = statement.strip().rstrip(";").strip()
    match = _INSERT_RE.match(text)
    if match is None:
        return None
    table_name = unqualified(match.group(1))
    columns = _split_idents(match.group(2)) if match.group(2) else None
    tuples = parse_sql_tuples(match.group(3))
    return table_name, columns, tuples


def parse_copy_header(statement: str) -> tuple[str, list[str] | None] | None:
    text = statement.strip()
    match = _COPY_RE.match(text)
    if match is not None:
        return unqualified(match.group(1)), _split_idents(match.group(2))
    match = _COPY_NO_COLS_RE.match(text)
    if match is not None:
        return unqualified(match.group(1)), None
    return None


def parse_sql_tuples(text: str) -> list[list[str | None]]:
    tuples: list[list[str | None]] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index] in " \t\r\n,":
            index += 1
        if index >= length:
            break
        if text[index] != "(":
            raise ValueError(f"Expected '(' in VALUES near {text[index:index + 40]!r}")
        index += 1
        fields: list[str | None] = []
        while True:
            while index < length and text[index] in " \t\r\n":
                index += 1
            if index < length and text[index] == ")":
                index += 1
                break
            value, index = _parse_sql_literal(text, index)
            fields.append(value)
            while index < length and text[index] in " \t\r\n":
                index += 1
            if index < length and text[index] == ",":
                index += 1
                continue
            if index < length and text[index] == ")":
                index += 1
                break
            raise ValueError(f"Malformed VALUES near {text[max(0, index - 10):index + 40]!r}")
        tuples.append(fields)
    return tuples


def unescape_copy_field(field: str) -> str | None:
    if field == r"\N":
        return None
    output: list[str] = []
    index = 0
    while index < len(field):
        char = field[index]
        if char == "\\" and index + 1 < len(field):
            nxt = field[index + 1]
            mapped = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "\\": "\\"}
            if nxt in mapped:
                output.append(mapped[nxt])
                index += 2
                continue
            if nxt.isdigit():
                octal = nxt
                cursor = index + 2
                while cursor < len(field) and len(octal) < 3 and field[cursor] in "01234567":
                    octal += field[cursor]
                    cursor += 1
                output.append(chr(int(octal, 8)))
                index = cursor
                continue
            output.append(nxt)
            index += 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def split_copy_row(line: str) -> list[str | None]:
    return [unescape_copy_field(field) for field in line.split("\t")]


def load_sqlite(schema_path: Path, data_path: Path, sqlite_path: Path) -> Schema:
    """Build a SQLite copy of one benchmark database. An existing file is replaced."""
    schema = read_schema(schema_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-journal", "-wal", "-shm"):
        leftover = Path(str(sqlite_path) + suffix) if suffix else sqlite_path
        if leftover.exists():
            leftover.unlink()
    connection = sqlite3.connect(sqlite_path)
    try:
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("PRAGMA cache_size = -200000")
        _create_tables(connection, schema)
        connection.execute(
            "CREATE TABLE semstruct_search_id (search_id INTEGER PRIMARY KEY, relation TEXT NOT NULL)"
        )
        _load_rows(connection, schema, data_path)
        _create_indexes(connection, schema)
        connection.commit()
    finally:
        connection.close()
    logger.info("loaded %s relations into %s", len(schema.relations), sqlite_path)
    return schema


def _load_rows(connection: sqlite3.Connection, schema: Schema, data_path: Path) -> None:
    batches: dict[tuple[str, tuple[str, ...]], list[tuple[str | None, ...]]] = {}
    search_rows: list[tuple[int, str]] = []
    pending = 0
    loaded = 0
    copy_table: str | None = None
    copy_columns: list[str] | None = None

    def flush() -> None:
        nonlocal pending
        for (table, columns), rows in batches.items():
            column_sql = ", ".join(quote_ident(column) for column in columns)
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO {quote_ident(table)} ({column_sql}) VALUES ({placeholders})",
                rows,
            )
        batches.clear()
        if search_rows:
            connection.executemany(
                "INSERT OR IGNORE INTO semstruct_search_id (search_id, relation) VALUES (?, ?)",
                search_rows,
            )
            search_rows.clear()
        connection.commit()
        pending = 0

    def add_row(table: str, columns: list[str], values: list[str | None]) -> None:
        nonlocal pending, loaded
        if table not in schema.relations:
            return
        if len(values) != len(columns):
            logger.warning("skipping %s row with %s values for %s columns", table, len(values), len(columns))
            return
        key = (table, tuple(columns))
        batches.setdefault(key, []).append(tuple(values))
        if "__search_id" in columns:
            raw = values[columns.index("__search_id")]
            if raw not in (None, ""):
                search_rows.append((int(float(str(raw).strip())), table))
        pending += 1
        loaded += 1
        if pending >= 5000:
            flush()
        if loaded % 50000 == 0:
            logger.info("loaded %s rows (latest table %s)", loaded, table)

    with open_sql(data_path) as stream:
        for kind, payload in iter_sql_events(stream):
            if kind == "copy_start":
                header = parse_copy_header(payload)
                if header is None:
                    copy_table = None
                    copy_columns = None
                    continue
                copy_table, declared = header
                if copy_table not in schema.relations:
                    logger.warning("skipping COPY into unknown table %s", copy_table)
                    copy_table = None
                    copy_columns = None
                    continue
                copy_columns = declared or [attribute.name for attribute in schema.relations[copy_table].attributes]
            elif kind == "copy_row":
                if copy_table is None or copy_columns is None:
                    continue
                add_row(copy_table, copy_columns, split_copy_row(payload))
            elif kind == "copy_end":
                copy_table = None
                copy_columns = None
            elif kind == "sql":
                if not payload.lstrip().upper().startswith("INSERT"):
                    continue
                parsed = parse_insert(payload)
                if parsed is None:
                    continue
                table, columns, tuples = parsed
                if table not in schema.relations:
                    continue
                resolved = columns or [attribute.name for attribute in schema.relations[table].attributes]
                for values in tuples:
                    add_row(table, resolved, values)
    flush()
    logger.info("finished loading %s rows from %s", loaded, data_path.name)


def _create_tables(connection: sqlite3.Connection, schema: Schema) -> None:
    for relation in schema.relations.values():
        columns = ", ".join(
            f"{quote_ident(attribute.name)} TEXT" for attribute in relation.attributes
        )
        connection.execute(f"CREATE TABLE {quote_ident(relation.name)} ({columns})")


def _create_indexes(connection: sqlite3.Connection, schema: Schema) -> None:
    seen: set[tuple[str, str]] = set()
    for relation in schema.relations.values():
        columns = list(relation.primary_key)
        if any(attribute.name == "__search_id" for attribute in relation.attributes):
            columns.append("__search_id")
        for foreign_key in relation.foreign_keys:
            columns.extend(foreign_key.columns)
        for column in columns:
            key = (relation.name, column)
            if key in seen or not any(attribute.name == column for attribute in relation.attributes):
                continue
            seen.add(key)
            index = re.sub(r"[^A-Za-z0-9_]", "_", f"idx_{relation.name}_{column}")[:60]
            connection.execute(
                f"CREATE INDEX {quote_ident(index)} ON {quote_ident(relation.name)} ({quote_ident(column)})"
            )


def _apply_table_constraint(relation: Relation, piece: str) -> None:
    body = re.sub(r"^CONSTRAINT\s+\S+\s+", "", piece, count=1, flags=re.IGNORECASE).strip()
    primary = _PK_RE.match(body)
    if primary is not None:
        relation.primary_key = _split_idents(primary.group(1))
        names = set(relation.primary_key)
        for attribute in relation.attributes:
            if attribute.name in names:
                attribute.primary_key = True
        return
    foreign = _FK_RE.match(body)
    if foreign is not None:
        _add_foreign_key(
            relation,
            ForeignKey(
                _split_idents(foreign.group(1)),
                unqualified(foreign.group(2)),
                _split_idents(foreign.group(3)),
            ),
        )


def _parse_column(piece: str) -> tuple[Attribute, ForeignKey | None] | None:
    name, rest = _read_ident(piece.strip())
    if not name:
        return None
    rest = rest.strip()
    type_match = _TYPE_RE.match(rest)
    if type_match is None:
        type_token = rest.split()[0] if rest.split() else "text"
        data_type = type_token
        tail = rest[len(type_token) :].strip()
    else:
        data_type = re.sub(r"\s+", " ", type_match.group(0)).strip()
        tail = rest[type_match.end() :].strip()
    nullable = re.search(r"\bNOT\s+NULL\b", tail, re.IGNORECASE) is None
    primary_key = re.search(r"\bPRIMARY\s+KEY\b", tail, re.IGNORECASE) is not None
    attribute = Attribute(name=name, data_type=data_type, nullable=nullable, primary_key=primary_key)
    reference = re.search(r"\bREFERENCES\s+(\S+)\s*\(([^)]*)\)", tail, re.IGNORECASE)
    foreign_key = None
    if reference is not None:
        foreign_key = ForeignKey(
            [name],
            unqualified(reference.group(1)),
            _split_idents(reference.group(2)),
        )
    return attribute, foreign_key


def _read_ident(text: str) -> tuple[str, str]:
    if text.startswith('"'):
        index = 1
        chars: list[str] = []
        while index < len(text):
            char = text[index]
            if char == '"' and index + 1 < len(text) and text[index + 1] == '"':
                chars.append('"')
                index += 2
                continue
            if char == '"':
                return "".join(chars), text[index + 1 :]
            chars.append(char)
            index += 1
        return "".join(chars), ""
    match = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text)
    if match is None:
        return "", text
    return match.group(0), text[match.end() :]


def _split_idents(text: str) -> list[str]:
    return [unqualified(part) for part in text.split(",") if part.strip()]


def _add_foreign_key(relation: Relation, foreign_key: ForeignKey) -> None:
    for existing in relation.foreign_keys:
        if (
            existing.columns == foreign_key.columns
            and existing.ref_table == foreign_key.ref_table
            and existing.ref_columns == foreign_key.ref_columns
        ):
            return
    relation.foreign_keys.append(foreign_key)


def _balanced(text: str, open_index: int) -> str | None:
    depth = 0
    state = "code"
    index = open_index
    while index < len(text):
        char = text[index]
        if state == "single":
            if char == "'" and index + 1 < len(text) and text[index + 1] == "'":
                index += 2
                continue
            if char == "'":
                state = "code"
            index += 1
            continue
        if char == "'":
            state = "single"
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[open_index + 1 : index]
        index += 1
    return None


def _split_commas(body: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    state = "code"
    index = 0
    while index < len(body):
        char = body[index]
        if state == "single":
            current.append(char)
            if char == "'" and index + 1 < len(body) and body[index + 1] == "'":
                current.append(body[index + 1])
                index += 2
                continue
            if char == "'":
                state = "code"
            index += 1
            continue
        if char == "'":
            state = "single"
            current.append(char)
        elif char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1
    if current and "".join(current).strip():
        parts.append("".join(current))
    return parts


def _parse_sql_literal(text: str, index: int) -> tuple[str | None, int]:
    if text.startswith("E'", index) or text.startswith("e'", index):
        value, index = _parse_sql_string(text, index + 1)
        return value, index
    if text.startswith("'", index):
        return _parse_sql_string(text, index)
    match = re.match(
        r"[A-Za-z_][A-Za-z0-9_]*|[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
        text[index:],
    )
    if match is None:
        raise ValueError(f"Bad SQL literal near {text[index:index + 40]!r}")
    token = match.group(0)
    if token.upper() == "NULL":
        return None, index + len(token)
    return token, index + len(token)


def _parse_sql_string(text: str, index: int) -> tuple[str, int]:
    if text[index] != "'":
        raise ValueError("string literal does not start with a quote")
    index += 1
    output: list[str] = []
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\" and index + 1 < length:
            nxt = text[index + 1]
            mapped = {"'": "'", "\\": "\\", "n": "\n", "t": "\t", "r": "\r"}
            output.append(mapped.get(nxt, nxt))
            index += 2
            continue
        if char == "'":
            if index + 1 < length and text[index + 1] == "'":
                output.append("'")
                index += 2
                continue
            return "".join(output), index + 1
        output.append(char)
        index += 1
    raise ValueError("unterminated SQL string")
