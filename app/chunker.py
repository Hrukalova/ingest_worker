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
    """Сегментация на предложения."""
    raw = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [s.strip() for s in raw if len(s.split()) >= 3]

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство векторов."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def _token_count(text: str) -> int:
    """Оценка числа токенов."""
    return int(len(text.split()) * 1.3)

# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class SemanticChunker:
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

    async def create_chunks(
        self,
        text: str,
        doc_title: str,
        topic_breadcrumb: Optional[str] = None,
    ) -> List[Dict]:
        """
        Вход: текст, заголовок, хлебные крошки.
        Выход: список чанков с метаданными и офсетами.
        """
        if not text or not text.strip():
            logger.warning(f"Пустой текст для документа '{doc_title}', пропускаем.")
            return []

        sentences = _split_sentences(text)
        if not sentences:
            sentences = [text.strip()]

        # Основной выбор алгоритма
        if len(sentences) <= 3 or self.embedder is None:
            # Передаем исходный текст для расчета офсетов
            raw_chunks = self._simple_fallback(text, sentences)
        else:
            raw_chunks = self._semantic_split(text, sentences)

        result = []
        for idx, (chunk_text, meta) in enumerate(raw_chunks):
            # Header Injection (Requirement 4)
            header = self._build_header(doc_title, topic_breadcrumb)
            final_text = f"{header}\n{chunk_text}"

            # Попытка извлечь заголовок секции (для section_path)
            # Берем первое предложение чанка, если оно похоже на заголовок (короткое)
            first_sent = chunk_text.split('\n')[0]
            section = first_sent[:100] if len(first_sent) < 120 else None

            meta.update({
                "chunk_index": idx,
                "token_count": _token_count(final_text),
                "doc_title": doc_title,
                "topic_breadcrumb": topic_breadcrumb,
                "section_path": section
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

    def _semantic_split(self, full_text: str, sentences: List[str]) -> List[tuple]:
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
            logger.error(f"Ошибка векторизации: {e}. Используем fallback.")
            return self._simple_fallback(full_text, sentences)

        # --- 3. Точки разрыва по сходству ---
        boundaries = set()
        for i in range(len(embeddings) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self.t_sim:
                boundaries.add(i + 1)

        # --- 4. Дополнительные разрывы по max_tokens ---
        boundaries = self._add_token_boundaries(sentences, boundaries)

        # --- 5. Сборка чанков (ТЕПЕРЬ С ОФСЕТАМИ) ---
        raw_chunks = self._assemble_chunks(full_text, sentences, boundaries)

        # --- 6. Склейка мелких чанков ---
        raw_chunks = self._merge_small_chunks(raw_chunks)

        return raw_chunks

    def _add_token_boundaries(self, sentences: List[str], boundaries: set) -> set:
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

    def _assemble_chunks(self, full_text: str, sentences: List[str], boundaries: set) -> List[tuple]:
        """
        Собирает предложения в чанки и вычисляет офсеты (Requirement 10).
        """
        chunks = []
        current_sentences = []
        start_sentence_idx = 0

        # Мы отслеживаем текущую позицию поиска в тексте, чтобы не найти одинаковые предложения в разных местах
        search_pos = 0

        for i, sent in enumerate(sentences):
            if i in boundaries and current_sentences:
                chunk_body = " ".join(current_sentences)

                # Находим точные границы текста
                start_off = full_text.find(current_sentences[0], search_pos)
                last_sent = current_sentences[-1]
                end_off = full_text.find(last_sent, start_off) + len(last_sent)

                # Обновляем позицию поиска, чтобы следующий чанк искался после этого
                search_pos = start_off

                chunks.append((chunk_body, {
                    "start_offset": max(0, start_off),
                    "end_offset": end_off,
                    "start_sentence": start_sentence_idx,
                    "end_sentence": i - 1
                }))
                current_sentences = []
                start_sentence_idx = i

            current_sentences.append(sent)

        # Обработка последнего чанка
        if current_sentences:
            chunk_body = " ".join(current_sentences)
            start_off = full_text.find(current_sentences[0], search_pos)
            last_sent = current_sentences[-1]
            end_off = full_text.find(last_sent, start_off) + len(last_sent)
            chunks.append((chunk_body, {
                "start_offset": max(0, start_off),
                "end_offset": end_off,
                "start_sentence": start_sentence_idx,
                "end_sentence": len(sentences) - 1
            }))
        return chunks

    def _merge_small_chunks(self, chunks: List[tuple]) -> List[tuple]:
        if not chunks: return chunks
        merged = []
        i = 0
        while i < len(chunks):
            text, meta = chunks[i]
            if _token_count(text) < self.min_tokens and i + 1 < len(chunks):
                next_text, next_meta = chunks[i + 1]
                combined_text = text + " " + next_text
                combined_meta = {
                    "start_offset": meta["start_offset"],
                    "end_offset": next_meta["end_offset"],
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
    # Fallback
    # ------------------------------------------------------------------

    def _simple_fallback(self, full_text: str, sentences: List[str]) -> List[tuple]:
        """Простая нарезка с сохранением офсетов."""
        boundaries = set()
        current_tokens = 0
        for i, sent in enumerate(sentences):
            t = _token_count(sent)
            if current_tokens + t > self.max_tokens:
                boundaries.add(i)
                current_tokens = 0
            current_tokens += t

        return self._assemble_chunks(full_text, sentences, boundaries)

    # ------------------------------------------------------------------
    # Header Injection
    # ------------------------------------------------------------------

    @staticmethod
    def _build_header(doc_title: str, breadcrumb: Optional[str]) -> str:
        lines = [f"Документ: {doc_title}"]
        if breadcrumb:
            lines.append(f"Категория: {breadcrumb}")
        lines.append("---")
        return "\n".join(lines)
