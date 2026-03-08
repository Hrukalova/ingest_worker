# services/ingest_worker/chunker.py
"""
Новичок №2 — "The Indexer": SemanticChunker
============================================
Реализует:
1. Semantic Splitting   — нарезка текста на чанки с порогом t_sim по косинусному сходству.
2. Header Injection     — вклейка "хлебных крошек" из иерархии topics в начало каждого чанка.

Алгоритм Semantic Splitting (аналог Kaggle-ноутбука):
  ─ Разбиваем текст на «сырые» предложения.
  ─ Создаём скользящее окно: каждый «буфер» объединяет предложение с N соседями.
  ─ Считаем косинусное сходство между соседними буферами.
  ─ Там, где сходство падает ниже порога t_sim, — точка разрыва (boundaries).
  ─ Склеиваем предложения между разрывами → чанки.
  ─ Чанки слишком длинные разбиваем по max_tokens, слишком короткие — объединяем.
"""

from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger("SemanticChunker")


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Простая сегментация на предложения по знакам препинания и переносам строк."""
    # Разбиваем по . ! ? и переносам строк
    raw = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    # Убираем пустые и слишком короткие (< 3 слов) фрагменты
    return [s.strip() for s in raw if len(s.split()) >= 3]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство двух нормализованных векторов."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _token_count(text: str) -> int:
    """Быстрая оценка числа токенов через слова (≈ 1.3 токена на слово)."""
    return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class SemanticChunker:
    """
    Семантический чанкер с Header Injection.

    Параметры
    ----------
    embedder : объект с методом .encode(texts) -> np.ndarray
        Передаётся извне готовая модель (SentenceTransformer или любая другая).
    t_sim : float
        Порог косинусного сходства. Там, где sim < t_sim, ставим разрыв.
        Типичное значение 0.45–0.55.
    max_tokens : int
        Максимальное число токенов в одном чанке.
    min_tokens : int
        Минимальное число токенов; слишком короткие чанки склеивает со следующим.
    window_size : int
        Размер скользящего окна (сколько предложений берём для буфера).
    """

    def __init__(
        self,
        embedder=None,
        t_sim: float = 0.5,
        max_tokens: int = 512,
        min_tokens: int = 30,
        window_size: int = 2,
    ):
        self.embedder = embedder
        self.t_sim = t_sim
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Публичный метод
    # ------------------------------------------------------------------

    async def create_chunks(
        self,
        text: str,
        doc_title: str,
        topic_breadcrumb: Optional[str] = None,
    ) -> List[Dict]:
        """
        Вход:
            text             — чистый текст документа
            doc_title        — название документа (для Header Injection)
            topic_breadcrumb — строка вида "Категория: Про деньги > Стипендии > Правительства РФ"
                               (формируется снаружи из листа topics)

        Выход:
            Список словарей:
            {
                "text": "<заголовок>\n<контент>",   # финальный текст чанка с инжектью
                "meta": {
                    "chunk_index": int,
                    "start_sentence": int,
                    "end_sentence": int,
                    "token_count": int,
                    "doc_title": str,
                    "topic_breadcrumb": str | None,
                }
            }
        """
        if not text or not text.strip():
            logger.warning(f"Пустой текст для документа '{doc_title}', пропускаем.")
            return []

        sentences = _split_sentences(text)
        if not sentences:
            # Если текст не разбился — один чанк из всего текста
            sentences = [text.strip()]

        # Если предложений мало — не нужен semantic split
        if len(sentences) <= 3 or self.embedder is None:
            chunks_text = self._simple_fallback(sentences)
        else:
            chunks_text = self._semantic_split(sentences)

        result = []
        for idx, (chunk_text, meta) in enumerate(chunks_text):
            # Header Injection
            header = self._build_header(doc_title, topic_breadcrumb)
            final_text = f"{header}\n{chunk_text}"

            meta.update({
                "chunk_index": idx,
                "token_count": _token_count(final_text),
                "doc_title": doc_title,
                "topic_breadcrumb": topic_breadcrumb,
            })
            result.append({"text": final_text, "meta": meta})

        logger.info(
            f"✂️  Документ '{doc_title}': {len(sentences)} предложений → {len(result)} чанков"
            + (" (semantic)" if self.embedder else " (fallback)")
        )
        return result

    # ------------------------------------------------------------------
    # Семантическое разбиение
    # ------------------------------------------------------------------

    def _semantic_split(self, sentences: List[str]) -> List[tuple]:
        """
        Основной алгоритм:
        1. Строим «буферные» представления (скользящее окно).
        2. Векторизуем буферы.
        3. Ищем точки разрыва по порогу t_sim.
        4. Дополнительные разрывы по max_tokens.
        5. Склеиваем слишком маленькие чанки.
        """
        # --- 1. Буферы ---
        buffers = []
        for i, _ in enumerate(sentences):
            start = max(0, i - self.window_size)
            end = min(len(sentences), i + self.window_size + 1)
            buffers.append(" ".join(sentences[start:end]))

        # --- 2. Векторизация ---
        try:
            embeddings = self.embedder.encode(buffers, show_progress_bar=False)
        except Exception as e:
            logger.error(f"Ошибка векторизации в chunker: {e}. Используем fallback.")
            return self._simple_fallback(sentences)

        # --- 3. Точки разрыва по сходству ---
        boundaries = set()
        for i in range(len(embeddings) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.t_sim:
                boundaries.add(i + 1)  # разрыв перед предложением i+1

        # --- 4. Дополнительные разрывы по max_tokens ---
        boundaries = self._add_token_boundaries(sentences, boundaries)

        # --- 5. Сборка чанков ---
        raw_chunks = self._assemble_chunks(sentences, boundaries)

        # --- 6. Склейка мелких чанков ---
        raw_chunks = self._merge_small_chunks(raw_chunks)

        return raw_chunks

    def _add_token_boundaries(
        self, sentences: List[str], boundaries: set
    ) -> set:
        """Принудительно добавляем разрыв, если накопилось > max_tokens токенов."""
        result = set(boundaries)
        current_tokens = 0
        current_start = 0
        for i, sent in enumerate(sentences):
            current_tokens += _token_count(sent)
            if current_tokens >= self.max_tokens and i > current_start:
                result.add(i + 1)
                current_tokens = 0
                current_start = i + 1
        return result

    def _assemble_chunks(
        self, sentences: List[str], boundaries: set
    ) -> List[tuple]:
        """Соединяем предложения в чанки и отдаём вместе с метаданными."""
        chunks = []
        current: List[str] = []
        start_idx = 0
        for i, sent in enumerate(sentences):
            if i in boundaries and current:
                chunks.append((" ".join(current), {"start_sentence": start_idx, "end_sentence": i - 1}))
                current = []
                start_idx = i
            current.append(sent)
        if current:
            chunks.append((" ".join(current), {"start_sentence": start_idx, "end_sentence": len(sentences) - 1}))
        return chunks

    def _merge_small_chunks(self, chunks: List[tuple]) -> List[tuple]:
        """Объединяем чанки меньше min_tokens со следующим."""
        if not chunks:
            return chunks
        merged = []
        i = 0
        while i < len(chunks):
            text, meta = chunks[i]
            if _token_count(text) < self.min_tokens and i + 1 < len(chunks):
                # Склеиваем с следующим
                next_text, next_meta = chunks[i + 1]
                combined_text = text + " " + next_text
                combined_meta = {
                    "start_sentence": meta["start_sentence"],
                    "end_sentence": next_meta["end_sentence"],
                }
                chunks[i + 1] = (combined_text, combined_meta)
                i += 1
                continue
            merged.append((text, meta))
            i += 1
        return merged

    # ------------------------------------------------------------------
    # Fallback (без модели или мало предложений)
    # ------------------------------------------------------------------

    def _simple_fallback(self, sentences: List[str]) -> List[tuple]:
        """Простая нарезка по max_tokens — используется при отсутствии embedder."""
        chunks = []
        current: List[str] = []
        current_tokens = 0
        start_idx = 0
        for i, sent in enumerate(sentences):
            t = _token_count(sent)
            if current_tokens + t > self.max_tokens and current:
                chunks.append((" ".join(current), {"start_sentence": start_idx, "end_sentence": i - 1}))
                current = []
                current_tokens = 0
                start_idx = i
            current.append(sent)
            current_tokens += t
        if current:
            chunks.append((" ".join(current), {"start_sentence": start_idx, "end_sentence": len(sentences) - 1}))
        return chunks

    # ------------------------------------------------------------------
    # Header Injection
    # ------------------------------------------------------------------

    @staticmethod
    def _build_header(doc_title: str, breadcrumb: Optional[str]) -> str:
        """
        Формирует заголовочный блок чанка.

        Пример результата:
            Документ: Положение о стипендиях МИЭМ
            Категория: Про деньги > Стипендии > Правительства РФ
            ---
        """
        lines = [f"Документ: {doc_title}"]
        if breadcrumb:
            lines.append(f"Категория: {breadcrumb}")
        lines.append("---")
        return "\n".join(lines)
