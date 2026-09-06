"""
chunker.py — SmartChunker для ingest_worker

Итоги бенчмарка (cohesion score, чем выше — тем лучше смысловые границы):

  Формат      Метод          cohesion   скорость
  ─────────────────────────────────────────────
  TXT         M2_Sentence    +0.047     мгновенно  ✅ ЛИДЕР
  DOCX+hdg    M5_Structure   +0.035     мгновенно  ✅ ЛИДЕР
  PDF+TOC     M5_Structure   ожидается  мгновенно  ✅
  PDF без TOC M4_Recursive   +0.007     мгновенно  ✅ ЛИДЕР
  M1_Fixed    везде          отриц.     мгновенно  ✗ не используем
  M3_Semantic везде          -0.02      46–378 с   ✗ слишком медленно

Стратегия выбора:
  PDF  + есть TOC    → M5_Structure  (по главам оглавления из PyMuPDF)
  PDF  без TOC       → M4_Recursive  (рекурсивная нарезка: \\n\\n→\\n→". ")
  DOCX + Heading 1/2 → M5_Structure  (по heading-стилям)
  DOCX без заголовков→ M2_Sentence   (по предложениям)
  TXT  / прочее      → M2_Sentence   (по предложениям)
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Dict, Optional

logger = logging.getLogger("SmartChunker")

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

# Сколько символов приходится на один токен эмбеддера. Замеряно на корпусе проекта
# (русскоязычные регламенты НИУ ВШЭ) токенизатором BAAI/bge-m3: в среднем 4.53, медиана 4.75,
# на плотном тексте — таблицы, перечни, юридические формулировки — до 2.26. Берём 4.0, то есть
# чуть консервативнее среднего: чанки выходят немного меньше цели, а не больше неё.
#
# Прежняя оценка `слова × 1.3` рассчитана на английский и на русском занижала вдвое, а с
# прежней англоязычной моделью — в 5.8 раза: её словарь BERT почти не содержал русских
# подслов и дробил слова на отдельные буквы. Чанки выходили по 5854 символа при окне модели
# в 512 токенов, и sentence-transformers молча обрезал вход, не поднимая ошибки.
#
# Значение зависит от модели: при смене EMBEDDER_MODEL_NAME его нужно перемерить.
CHARS_PER_TOKEN = float(os.getenv("CHARS_PER_TOKEN", "4.0"))


def _count_tokens(text: str) -> int:
    """Оценка числа токенов по длине текста в символах.

    По символам, а не по словам: длинные слова, таблицы и текст без пробелов оценка по
    словам занижает особенно сильно, а разброс символов на токен заметно уже.
    """
    return max(1, round(len(text) / CHARS_PER_TOKEN))


def _split_sentences(text: str) -> List[str]:
    """Разбивка текста на предложения."""
    raw = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [s.strip() for s in raw if len(s.split()) >= 3]


def _split_oversized(piece: str, chunk_size: int) -> List[str]:
    """Режет кусок, который сам по себе длиннее целевого размера, по словам.

    Нужен потому, что группировка по предложениям умеет ставить границу только МЕЖДУ
    предложениями. Если `_split_sentences` не нашла границ вообще (текст без точек —
    таблицы, перечни, выгрузки страниц) или одно предложение оказалось длиннее цели,
    кусок уходил в чанк целиком: так появлялись чанки по 19 500 символов, на которых
    embedding-svc отвечал 413.
    """
    if _count_tokens(piece) <= chunk_size:
        return [piece]

    words = piece.split()
    if not words:
        return [piece]

    # Слов на чанк — из той же калибровки: chunk_size токенов при CHARS_PER_TOKEN символов
    # на токен, делённые на среднюю длину слова в этом куске.
    avg_word_len = max(1.0, len(piece) / len(words))
    words_per_chunk = max(1, int(chunk_size * CHARS_PER_TOKEN / avg_word_len))

    out: List[str] = []
    for i in range(0, len(words), words_per_chunk):
        part = " ".join(words[i : i + words_per_chunk]).strip()
        if part:
            out.append(part)
    return out


def _build_header(doc_title: str, breadcrumb: Optional[str]) -> str:
    lines = [f"Документ: {doc_title}"]
    if breadcrumb:
        lines.append(f"Категория: {breadcrumb}")
    lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# M2: Sentence-based — лучший для TXT, fallback для DOCX
# Cohesion: +0.047 (TXT), оптимально для равномерной нарезки
# ---------------------------------------------------------------------------

def _chunk_by_sentences(
    text: str,
    chunk_size: int = 512,
    min_tokens: int = 50,
) -> List[tuple]:
    """
    Нарезает текст по предложениям, группируя до chunk_size токенов.
    Возвращает [(chunk_text, {start_offset, end_offset}), ...].
    """
    sentences = _split_sentences(text)
    if not sentences:
        # Границ предложений нет вовсе — режем по словам, иначе весь документ станет
        # одним чанком независимо от его размера.
        parts = _split_oversized(text.strip(), chunk_size)
        return [(p, {"start_offset": 0, "end_offset": len(text)}) for p in parts]

    # Предложение длиннее цели граница между предложениями не разобьёт — дробим заранее.
    sentences = [p for sent in sentences for p in _split_oversized(sent, chunk_size)]

    chunks: List[tuple] = []
    current: List[str] = []
    current_tokens = 0
    search_pos = 0

    def _flush():
        nonlocal search_pos
        body = " ".join(current)
        start = text.find(current[0], search_pos)
        start = max(0, start)
        end = start + len(body)
        search_pos = start
        chunks.append((body, {"start_offset": start, "end_offset": end}))

    for sent in sentences:
        t = _count_tokens(sent)
        if current_tokens + t > chunk_size and current:
            _flush()
            current = []
            current_tokens = 0
        current.append(sent)
        current_tokens += t

    if current:
        _flush()

    # Склейка слишком мелких хвостовых чанков
    if len(chunks) >= 2 and _count_tokens(chunks[-1][0]) < min_tokens:
        prev_text, prev_meta = chunks[-2]
        last_text, last_meta = chunks[-1]
        merged = prev_text + " " + last_text
        chunks[-2] = (merged, {
            "start_offset": prev_meta["start_offset"],
            "end_offset": last_meta["end_offset"],
        })
        chunks.pop()

    return chunks


# ---------------------------------------------------------------------------
# M4: Recursive — лучший для PDF без структуры
# Cohesion: +0.007, нулевое время, нет переполнения
# ---------------------------------------------------------------------------

def _chunk_recursive(
    text: str,
    chunk_size: int = 512,
    overlap_tokens: int = 32,
    separators: Optional[List[str]] = None,
) -> List[tuple]:
    """
    Рекурсивно делит текст по иерархии разделителей:
    \\n\\n → \\n → ". " → " "
    Возвращает [(chunk_text, {start_offset, end_offset}), ...].
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    def _split(t: str, seps: List[str]) -> List[str]:
        if not seps:
            return [t] if t.strip() else []
        sep = seps[0]
        parts = t.split(sep)
        result: List[str] = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if not candidate:
                continue
            if _count_tokens(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    result.append(current)
                if _count_tokens(part.strip()) <= chunk_size:
                    current = part.strip()
                else:
                    sub = _split(part, seps[1:])
                    result.extend(sub[:-1])
                    current = sub[-1] if sub else ""
        if current:
            result.append(current)
        return result

    parts = [p for p in _split(text, separators) if p.strip()]
    if not parts:
        return [(text.strip(), {"start_offset": 0, "end_offset": len(text)})]

    chunks: List[tuple] = []
    search_pos = 0
    prev_tail = ""

    for body in parts:
        # Перекрытие: добавляем хвост предыдущего чанка
        if prev_tail:
            body = prev_tail + " " + body
            words = body.split()
            if len(words) > int(chunk_size * 1.1):
                body = " ".join(words[:chunk_size])

        anchor = body[:50].strip()
        start = text.find(anchor, search_pos)
        start = max(0, start)
        end = start + len(body)
        search_pos = max(search_pos, start)
        chunks.append((body, {"start_offset": start, "end_offset": end}))

        words = body.split()
        prev_tail = " ".join(words[-overlap_tokens:]) if overlap_tokens > 0 else ""

    return chunks


# ---------------------------------------------------------------------------
# M5: Structure-aware — лучший для DOCX с heading / PDF с TOC
# Cohesion: +0.035 (DOCX), intra_sim=0.727 (лучший в тесте)
# ---------------------------------------------------------------------------

_MIN_SECTION_TOKENS = 80   # меньше — склеивается с соседней секцией


def _chunk_by_structure(
    text: str,
    sections: List[Dict],
    chunk_size: int = 512,
) -> List[tuple]:
    """
    Нарезает текст по готовым секциям (из TOC или Heading-стилей).
    Секции с token_count > chunk_size дробятся рекурсивно (M4).
    Слишком мелкие секции (<_MIN_SECTION_TOKENS) склеиваются.
    Возвращает [(chunk_text, {start_offset, end_offset, section_title}), ...].
    """
    if not sections:
        return _chunk_recursive(text, chunk_size)

    raw: List[tuple] = []
    search_pos = 0

    for sec in sections:
        sec_text = sec.get("text", "").strip()
        title = sec.get("title", "")
        if not sec_text:
            continue

        if _count_tokens(sec_text) <= chunk_size:
            anchor = sec_text[:50]
            start = text.find(anchor, search_pos)
            start = max(0, start)
            end = start + len(sec_text)
            search_pos = max(search_pos, start)
            raw.append((sec_text, {
                "start_offset": start,
                "end_offset": end,
                "section_title": title,
            }))
        else:
            # Большая секция — дробим M4
            sub = _chunk_recursive(sec_text, chunk_size)
            for sc_text, sc_meta in sub:
                anchor = sc_text[:40].strip()
                start = text.find(anchor, search_pos)
                start = max(0, start)
                end = start + len(sc_text)
                search_pos = max(search_pos, start)
                raw.append((sc_text, {
                    "start_offset": start,
                    "end_offset": end,
                    "section_title": title,
                }))

    # Склейка слишком мелких секций
    merged: List[tuple] = []
    buf_text: Optional[str] = None
    buf_meta: Optional[Dict] = None

    for chunk_text, meta in raw:
        if _count_tokens(chunk_text) < _MIN_SECTION_TOKENS:
            if buf_text is None:
                buf_text, buf_meta = chunk_text, dict(meta)
            else:
                buf_text += "\n\n" + chunk_text
                buf_meta["end_offset"] = meta["end_offset"]
        else:
            if buf_text is not None:
                chunk_text = buf_text + "\n\n" + chunk_text
                meta = dict(meta)
                meta["start_offset"] = buf_meta["start_offset"]
                buf_text = buf_meta = None
            merged.append((chunk_text, meta))

    if buf_text is not None:
        if merged:
            last_text, last_meta = merged[-1]
            merged[-1] = (last_text + "\n\n" + buf_text, {
                **last_meta,
                "end_offset": buf_meta["end_offset"],
            })
        else:
            merged.append((buf_text, buf_meta))

    return merged if merged else _chunk_recursive(text, chunk_size)


# ---------------------------------------------------------------------------
# SmartChunker — публичный класс
# ---------------------------------------------------------------------------

class SmartChunker:
    """
    Умный чанкер: автоматически выбирает метод нарезки
    по типу документа и его структуре.

    Параметры:
        chunk_size    — целевой размер чанка в токенах (default: 512)
        min_tokens    — минимум токенов в чанке (default: 50)
        overlap_tokens— перекрытие при M4 (default: 32)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        min_tokens: int = 50,
        overlap_tokens: int = 32,
    ):
        self.chunk_size = chunk_size
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens

    async def create_chunks(
        self,
        text: str,
        doc_title: str,
        doc_type: str = "txt",
        sections: Optional[List[Dict]] = None,
        topic_breadcrumb: Optional[str] = None,
    ) -> List[Dict]:
        """
        Нарезает текст на чанки с метаданными.

        Аргументы:
            text            — извлечённый текст документа
            doc_title       — название документа
            doc_type        — "pdf" | "docx" | "txt" (определяет стратегию)
            sections        — список секций [{title, text}] из парсера
            topic_breadcrumb— категория/тема для header injection

        Возвращает:
            List[{"text": str, "meta": {...}}]
        """
        if not text or not text.strip():
            logger.warning(f"Пустой текст для '{doc_title}', пропускаем.")
            return []

        # --- Выбор метода ---
        has_structure = bool(sections)

        if doc_type == "pdf":
            if has_structure:
                method = "M5_Structure"
                raw = _chunk_by_structure(text, sections, self.chunk_size)
            else:
                method = "M4_Recursive"
                raw = _chunk_recursive(text, self.chunk_size, self.overlap_tokens)

        elif doc_type == "docx":
            if has_structure:
                method = "M5_Structure"
                raw = _chunk_by_structure(text, sections, self.chunk_size)
            else:
                method = "M2_Sentence"
                raw = _chunk_by_sentences(text, self.chunk_size, self.min_tokens)

        else:  # txt, html, unknown
            method = "M2_Sentence"
            raw = _chunk_by_sentences(text, self.chunk_size, self.min_tokens)

        # --- Сборка результата ---
        result: List[Dict] = []
        header = _build_header(doc_title, topic_breadcrumb)

        for idx, (chunk_text, meta) in enumerate(raw):
            final_text = f"{header}\n{chunk_text}"
            first_line = chunk_text.split("\n")[0]
            section_path = (
                meta.get("section_title")
                or (first_line[:100] if len(first_line) < 120 else None)
            )
            meta.update({
                "chunk_index": idx,
                "token_count": _count_tokens(final_text),
                "doc_title": doc_title,
                "doc_type": doc_type,
                "topic_breadcrumb": topic_breadcrumb,
                "section_path": section_path,
                "chunk_method": method,
            })
            result.append({"text": final_text, "meta": meta})

        logger.info(
            f"✂️  '{doc_title}' [{doc_type}] → {len(result)} чанков ({method})"
        )
        return result
