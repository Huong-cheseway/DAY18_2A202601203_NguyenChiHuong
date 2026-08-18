from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""
    if OPENAI_API_KEY:
        try:
            return _chat_completion(
                "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
                text,
                max_tokens=150,
            )
        except Exception as exc:
            print(f"  ⚠️  OpenAI summarize failed: {exc}")
    return _fallback_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0 or not text.strip():
        return []
    if OPENAI_API_KEY:
        try:
            content = _chat_completion(
                f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                "Mỗi câu hỏi nằm trên một dòng, không thêm giải thích.",
                text,
                max_tokens=200,
            )
            questions = [_clean_question(line) for line in content.splitlines() if line.strip()]
            return [question for question in questions if question][:n_questions]
        except Exception as exc:
            print(f"  ⚠️  OpenAI HyQA failed: {exc}")
    return _fallback_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text.strip():
        return text
    if OPENAI_API_KEY:
        try:
            context = _chat_completion(
                "Viết đúng 1 câu ngắn mô tả đoạn văn nằm ở đâu trong tài liệu và nói về chủ đề gì.",
                f"Tài liệu: {document_title or 'Không rõ'}\n\nĐoạn văn:\n{text}",
                max_tokens=80,
            )
            return f"{context}\n\n{text}" if context else text
        except Exception as exc:
            print(f"  ⚠️  OpenAI contextual failed: {exc}")
    return f"{_fallback_context(document_title)}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if OPENAI_API_KEY and text.strip():
        try:
            result = _chat_json(
                'Trích xuất metadata và chỉ trả về JSON hợp lệ: '
                '{"topic":"...","entities":["..."],'
                '"category":"policy|hr|it|finance","language":"vi|en"}.',
                text,
                max_tokens=150,
            )
            return _normalize_metadata(result)
        except Exception as exc:
            print(f"  ⚠️  OpenAI metadata failed: {exc}")
    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    fallback = {
        "summary": _fallback_summary(text),
        "questions": _fallback_questions(text, 3),
        "context": _fallback_context(source),
        "metadata": _fallback_metadata(text),
    }
    if not OPENAI_API_KEY or not text.strip():
        return fallback

    try:
        result = _chat_json(
            """Phân tích đoạn văn và chỉ trả về JSON hợp lệ theo cấu trúc:
{
  "summary": "tóm tắt 2-3 câu",
  "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
  "context": "1 câu mô tả vị trí và chủ đề của đoạn văn trong tài liệu",
  "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
}""",
            f"Tài liệu: {source or 'Không rõ'}\n\nĐoạn văn:\n{text}",
            max_tokens=400,
        )
        questions = result.get("questions")
        metadata = result.get("metadata")
        return {
            "summary": str(result.get("summary") or fallback["summary"]),
            "questions": (
                [str(question).strip() for question in questions if str(question).strip()][:3]
                if isinstance(questions, list)
                else fallback["questions"]
            ),
            "context": str(result.get("context") or fallback["context"]),
            "metadata": (
                _normalize_metadata(metadata)
                if isinstance(metadata, dict)
                else fallback["metadata"]
            ),
        }
    except Exception as exc:
        print(f"  ⚠️  Enrichment API failed: {exc}")
        return fallback


def _chat_completion(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """Make one OpenAI chat-completion call and return stripped text."""
    from openai import OpenAI

    response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def _chat_json(system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
    """Make one completion call and parse a JSON object, including fenced JSON."""
    content = _chat_completion(system_prompt, user_prompt, max_tokens)
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.IGNORECASE)
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM response does not contain a JSON object")
    result = json.loads(content[start:end + 1])
    if not isinstance(result, dict):
        raise ValueError("LLM response must be a JSON object")
    return result


def _fallback_summary(text: str) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    return " ".join(sentences[:2]) if sentences else text.strip()


def _fallback_questions(text: str, n_questions: int) -> list[str]:
    sentences = [part.strip() for part in re.split(r"[.!?\n]+", text) if len(part.strip()) > 10]
    return [f"{sentence.rstrip('?')}?" for sentence in sentences[:n_questions]]


def _fallback_context(source: str) -> str:
    return (
        f"Đoạn trích này thuộc tài liệu {source} và cung cấp thông tin chính sách liên quan."
        if source
        else "Đoạn trích này cung cấp thông tin chính sách liên quan."
    )


def _fallback_metadata(text: str) -> dict:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ("mật khẩu", "vpn", "dữ liệu", "bảo mật")):
        category = "it"
    elif any(keyword in lowered for keyword in ("lương", "chi phí", "thanh toán", "tài chính")):
        category = "finance"
    elif any(keyword in lowered for keyword in ("nhân viên", "nghỉ", "thử việc", "đào tạo")):
        category = "hr"
    else:
        category = "policy"
    language = "vi" if re.search(r"[À-ỹĐđ]", text) else "en"
    return {"topic": "general", "entities": [], "category": category, "language": language}


def _normalize_metadata(metadata: dict) -> dict:
    fallback = _fallback_metadata("")
    entities = metadata.get("entities", [])
    return {
        "topic": str(metadata.get("topic") or fallback["topic"]),
        "entities": [str(entity) for entity in entities] if isinstance(entities, list) else [],
        "category": str(metadata.get("category") or fallback["category"]),
        "language": str(metadata.get("language") or fallback["language"]),
    }


def _clean_question(question: str) -> str:
    question = re.sub(r"^\s*\d+\s*[.)-]?\s*", "", question).strip(" -")
    return question if not question or question.endswith("?") else f"{question}?"


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
