"""
notebook_parsing.py — разбор документов в блоки по пайплайну из ноутбука команды.

Порт ячейки 9 ноутбука MIEM_Last_Stable_Pipeline: вместо путей к файлам на вход идут байты
(их отдаёт get_content), результат тот же — плоский список блоков
{"section": str, "kind": str, "text": str}.

Какие форматы: DOCX (заголовки по стилю, таблицы строками «Заголовок: значение | …»),
PDF (блок на страницу, раздел «Страница N»), TXT (заголовки `#`).
"""

from __future__ import annotations

import io
import re
from typing import Dict, List, Optional

from docx import Document
from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from errors import DocumentParseError

# Шум, который попадает в текст при выгрузке страниц сайта; список взят из ноутбука как есть.
_NOISE_PATTERNS = [
    r"Мы используем файлы cookie.*",
    r"Настройки файлов cookie.*",
    r"Нашли опечатку\?.*",
    r"Ctrl\s*\+\s*Enter.*",
    r"Javascript.*",
]


def clean_text(text: Optional[str]) -> str:
    """Убирает markdown-ссылки, cookie-мусор и лишние пробелы."""
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.I)

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_docx_blocks(parent):
    """Абзацы и таблицы DOCX в порядке следования в документе."""
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    else:
        parent_elm = parent._tc

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def table_to_blocks(table, section: str) -> List[Dict]:
    """Каждая строка таблицы (кроме шапки) — блок «Заголовок: значение | …»."""
    rows = []

    for row in table.rows:
        values = [clean_text(cell.text) for cell in row.cells]
        if any(values):
            rows.append(values)

    if not rows:
        return []

    header = rows[0]
    result = []

    for row in rows[1:] if len(rows) > 1 else rows:
        pairs = []

        for i, value in enumerate(row):
            if not value:
                continue

            if i < len(header) and header[i] and header[i] != value:
                pairs.append(f"{header[i]}: {value}")
            else:
                pairs.append(value)

        text = " | ".join(pairs)

        if text:
            result.append({"section": section, "kind": "table", "text": text})

    return result


def extract_docx_blocks(content: bytes) -> List[Dict]:
    doc = Document(io.BytesIO(content))
    blocks = []
    section = ""

    for item in iter_docx_blocks(doc):
        if isinstance(item, Paragraph):
            text = clean_text(item.text)

            if not text:
                continue

            style_name = ""
            try:
                style_name = (item.style.name or "").lower()
            except Exception:
                pass

            is_heading = "heading" in style_name or "заголов" in style_name

            if is_heading:
                section = text

            blocks.append({
                "section": section,
                "kind": "heading" if is_heading else "paragraph",
                "text": text,
            })

        elif isinstance(item, Table):
            blocks.extend(table_to_blocks(item, section))

    return blocks


def extract_txt_blocks(content: bytes) -> List[Dict]:
    raw = clean_text(content.decode("utf-8", errors="ignore"))

    blocks = []
    section = ""
    buffer: List[str] = []

    def flush():
        nonlocal buffer
        if buffer:
            text = clean_text(" ".join(buffer))
            if text:
                blocks.append({"section": section, "kind": "paragraph", "text": text})
            buffer = []

    for line in raw.splitlines():
        line = clean_text(line)

        if not line:
            flush()
            continue

        if re.match(r"^#{1,6}\s+", line):
            flush()
            section = re.sub(r"^#{1,6}\s+", "", line).strip()
            blocks.append({"section": section, "kind": "heading", "text": section})
        else:
            buffer.append(line)

    flush()
    return blocks


def extract_pdf_blocks(content: bytes) -> List[Dict]:
    reader = PdfReader(io.BytesIO(content))
    blocks = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue

        blocks.append({"section": f"Страница {page_num}", "kind": "page", "text": text})

    return blocks


def extract_blocks(content: bytes, doc_type: str) -> List[Dict]:
    """Разбирает файл в блоки. doc_type: "pdf" | "docx" | "txt".

    Нечитаемый (битый) DOCX или PDF — DocumentParseError: повтор его не исправит,
    документ сразу уходит в failed. TXT не падает: нечитаемые байты отбрасываются.
    """
    if doc_type == "txt":
        return extract_txt_blocks(content)
    if doc_type not in ("docx", "pdf"):
        return []

    extractor = extract_docx_blocks if doc_type == "docx" else extract_pdf_blocks
    try:
        return extractor(content)
    except Exception as exc:
        raise DocumentParseError(f"{doc_type.upper()} не разбирается: {exc}") from exc
