"""Coffman and Weaver topic files.

Each numbered file starts with a `# keyword query` line. Later lines that are
not comments list relevant joined tuples. Tuple ids are `__search_id` values.
"""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

_RESULT_RE = re.compile(r"\(\s*\[([^\]]*)\]\s*,\s*\[[^\]]*\]\s*\)")
_METADATA = {
    "[reference standard]",
    "imdb2009",
    "imdb",
    "mondial",
    "wikipedia",
    "wikipedia2008",
}


@dataclass
class Topic:
    topic_id: str
    query: str
    search_ids: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"topic_id": self.topic_id, "query": self.query, "search_ids": list(self.search_ids)}


def extract_topics(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(destination, filter="data")
        else:
            tar.extractall(destination)
    return destination


def load_topics(directory: Path) -> list[Topic]:
    files = sorted(directory.rglob("*.txt"))
    topics = []
    for path in files:
        if not path.stem.isdigit():
            continue
        topic = parse_topic_file(path)
        if topic.query:
            topics.append(topic)
    topics.sort(key=lambda item: int(item.topic_id))
    return topics


def parse_topic_file(path: Path) -> Topic:
    topic_id = str(int(path.stem))
    query = ""
    search_ids: list[int] = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            comment = line[1:].strip()
            if not query and comment and not _is_metadata(comment) and not comment.startswith("("):
                query = comment
            continue
        code = line.split("#", 1)[0].strip()
        for match in _RESULT_RE.finditer(code):
            for piece in match.group(1).split(","):
                piece = piece.strip()
                if piece:
                    search_ids.append(int(piece))
    return Topic(topic_id=topic_id, query=query, search_ids=search_ids)


def topic_query_norms(topics: list[Topic]) -> set[str]:
    from semstruct.data.elements import normalize_text

    return {normalize_text(topic.query) for topic in topics if topic.query}


def _is_metadata(comment: str) -> bool:
    return comment.strip().lower() in _METADATA
