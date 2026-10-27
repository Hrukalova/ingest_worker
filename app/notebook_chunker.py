"""
notebook_chunker.py — нарезка блоков на чанки по пайплайну из ноутбука команды.

Порт ячейки 11 ноутбука MIEM_Last_Stable_Pipeline без привязки к manifest: на вход идут
только блоки, метаданные документа (название, категория) подставляются при построении текста
для эмбеддинга — embedding_text().
"""

from __future__ import annotations

import re
from typing import Dict, List

from notebook_parsing import clean_text

# Минимальная длина чанка: короче — отбрасывается; абзац длиннее порога не переносится в
# следующий чанк как перекрытие.
MIN_CHUNK_CHARS = 80
OVERLAP_MAX_CHARS = 350


def split_long_text(text: str, max_chars: int = 1400) -> List[str]:
    """Режет длинный текст по предложениям; слишком длинное предложение — жёстко."""
    text = clean_text(text)

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                parts.append(current)

            if len(sentence) <= max_chars:
                current = sentence
            else:
                for i in range(0, len(sentence), max_chars):
                    parts.append(sentence[i:i + max_chars])
                current = ""

    if current:
        parts.append(current)

    return parts


def build_document_chunks(blocks: List[Dict], max_chars: int = 1600) -> List[Dict]:
    """Собирает блоки в чанки: [{"chunk_index", "section", "text"}].

    Разные разделы не склеиваются; последний абзац чанка (если не длиннее 350 символов)
    переходит в следующий как перекрытие; чанки короче 80 символов отбрасываются.
    """
    chunks = []
    current_parts: List[str] = []
    current_section = ""
    chunk_index = 0

    def flush():
        nonlocal current_parts, chunk_index

        if not current_parts:
            return

        text = clean_text("\n".join(current_parts))

        if len(text) >= MIN_CHUNK_CHARS:
            chunks.append({
                "chunk_index": chunk_index,
                "section": current_section,
                "text": text,
            })
            chunk_index += 1

        overlap = current_parts[-1]
        current_parts = [overlap] if len(overlap) <= OVERLAP_MAX_CHARS else []

    for block in blocks:
        section = block.get("section") or ""
        text = clean_text(block.get("text", ""))

        if not text:
            continue

        if current_section and section and section != current_section:
            flush()
            current_parts = []

        if section:
            current_section = section

        for piece in split_long_text(text):
            candidate = "\n".join(current_parts + [piece])

            if current_parts and len(candidate) > max_chars:
                flush()

            current_parts.append(piece)

    flush()
    return chunks


def embedding_text(title: str, category: str, section: str, text: str) -> str:
    """Текст чанка для эмбеддинга и BM25. Префикс `passage: ` добавляет embedding-svc.

    Формат — контракт с graph-rag-svc (BM25 строится по такому же тексту), менять нельзя.
    """
    return (
        f"Название документа: {title}\n"
        f"Категория: {category}\n"
        f"Раздел: {section}\n"
        f"Текст: {text}"
    )
