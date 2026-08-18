from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  Warning: skipping {os.path.basename(fp)} (scanned PDF; OCR required).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    import numpy as np

    base_metadata = dict(metadata or {})
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n+", text.strip())
        if sentence.strip()
    ]
    if not sentences:
        return []

    embeddings = _get_semantic_model().encode(sentences)
    groups: list[list[str]] = [[sentences[0]]]

    for index in range(1, len(sentences)):
        previous = embeddings[index - 1]
        current = embeddings[index]
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(current)) + 1e-9
        similarity = float(np.dot(previous, current) / denominator)
        if similarity < threshold:
            groups.append([sentences[index]])
        else:
            groups[-1].append(sentences[index])

    return [
        Chunk(
            text=" ".join(group),
            metadata={**base_metadata, "strategy": "semantic", "chunk_index": index},
        )
        for index, group in enumerate(groups)
    ]


@lru_cache(maxsize=1)
def _get_semantic_model():
    """Load the semantic embedding model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("parent_size and child_size must be positive")

    base_metadata = dict(metadata or {})
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n+", text)
        if paragraph.strip()
    ]
    if not paragraphs:
        return [], []

    parent_units: list[str] = []
    for paragraph in paragraphs:
        parent_units.extend(_split_by_size(paragraph, parent_size))
    parent_texts = _pack_units(parent_units, parent_size, separator="\n\n")

    parents: list[Chunk] = []
    children: list[Chunk] = []
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{parent_index}"
        parents.append(Chunk(
            text=parent_text,
            metadata={
                **base_metadata,
                "strategy": "hierarchical",
                "chunk_type": "parent",
                "parent_id": parent_id,
                "chunk_index": parent_index,
            },
        ))

        for child_index, child_text in enumerate(_split_by_size(parent_text, child_size)):
            children.append(Chunk(
                text=child_text,
                metadata={
                    **base_metadata,
                    "strategy": "hierarchical",
                    "chunk_type": "child",
                    "chunk_index": child_index,
                },
                parent_id=parent_id,
            ))

    return parents, children


def _pack_units(units: list[str], size: int, separator: str = " ") -> list[str]:
    """Greedily combine ordered units without exceeding ``size`` characters."""
    packed: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}{separator}{unit}" if current else unit
        if current and len(candidate) > size:
            packed.append(current)
            current = unit
        else:
            current = candidate
    if current:
        packed.append(current)
    return packed


def _split_by_size(text: str, size: int) -> list[str]:
    """Split at sentence/word boundaries while enforcing a character limit."""
    if len(text) <= size:
        return [text]

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    units: list[str] = []
    for sentence in sentences:
        if len(sentence) <= size:
            units.append(sentence)
            continue

        current = ""
        for word in sentence.split():
            if len(word) > size:
                if current:
                    units.append(current)
                    current = ""
                units.extend(word[start:start + size] for start in range(0, len(word), size))
                continue
            candidate = f"{current} {word}" if current else word
            if current and len(candidate) > size:
                units.append(current)
                current = word
            else:
                current = candidate
        if current:
            units.append(current)

    return _pack_units(units, size)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    base_metadata = dict(metadata or {})
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    current_header: str | None = None
    current_level: int | None = None
    current_lines: list[str] = []
    in_fenced_block = False

    def flush_section() -> None:
        section_text = "\n".join(current_lines).strip()
        if not section_text:
            return
        section_name = (
            re.sub(r"^#{1,3}\s+", "", current_header).strip()
            if current_header
            else "preamble"
        )
        chunks.append(Chunk(
            text=section_text,
            metadata={
                **base_metadata,
                "strategy": "structure",
                "section": section_name,
                "header_level": current_level,
                "chunk_index": len(chunks),
            },
        ))

    for line in text.splitlines():
        stripped = line.strip()
        was_in_fenced_block = in_fenced_block
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fenced_block = not in_fenced_block

        header_match = (
            None
            if was_in_fenced_block or in_fenced_block
            else re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
        )
        if header_match:
            flush_section()
            current_header = line.strip()
            current_level = len(header_match.group(1))
            current_lines = [current_header]
        else:
            current_lines.append(line)

    flush_section()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
