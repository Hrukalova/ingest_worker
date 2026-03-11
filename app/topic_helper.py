# services/ingest_worker/topic_helper.py
"""
Новичок №2 — "The Indexer": TopicHelper
========================================
Загружает лист «topics» из Excel-метаданных и строит «хлебные крошки»
(breadcrumbs) для Header Injection.

Структура листа topics (ожидаемые колонки):
  topic_id  | parent_id | name
  ----------|-----------|------------------------------
  1         | None      | Про деньги
  2         | 1         | Стипендии
  3         | 2         | Правительства РФ

Связь с листом documents:
  documents.topic_id → topics.topic_id

Пример breadcrumb для topic_id=3:
  "Про деньги > Стипендии > Правительства РФ"
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger("TopicHelper")


class TopicHelper:
    """
    Строит иерархию топиков и отдаёт breadcrumb по topic_id.

    Использование
    -------------
    helper = TopicHelper()
    helper.load_from_excel("metadata.xlsx", sheet_name="topics")
    breadcrumb = helper.get_breadcrumb(topic_id=3)
    # → "Про деньги > Стипендии > Правительства РФ"

    Или загрузить из словаря уже готовых данных:
    helper.load_from_records([
        {"topic_id": 1, "parent_id": None, "name": "Про деньги"},
        {"topic_id": 2, "parent_id": 1, "name": "Стипендии"},
    ])
    """

    def __init__(self):
        # {topic_id -> {"name": str, "parent_id": int | None}}
        self._topics: Dict[int, Dict] = {}

    # ------------------------------------------------------------------
    # Загрузка данных
    # ------------------------------------------------------------------

    def load_from_excel(self, xlsx_path: str, sheet_name: str = "topics") -> None:
        """Загружает топики из Excel-файла."""
        try:
            import pandas as pd
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
            self.load_from_dataframe(df)
        except Exception as e:
            logger.error(f"Не удалось загрузить топики из '{xlsx_path}': {e}")

    def load_from_dataframe(self, df) -> None:
        """Загружает топики из pandas DataFrame."""
        import pandas as pd
        self._topics = {}
        for _, row in df.iterrows():
            topic_id = int(row.get("topic_id", 0))
            parent_raw = row.get("parent_id", None)
            parent_id = None if pd.isna(parent_raw) else int(parent_raw)
            name = str(row.get("name", "")).strip()
            if topic_id and name:
                self._topics[topic_id] = {"name": name, "parent_id": parent_id}
        logger.info(f"✅ Загружено {len(self._topics)} топиков.")

    def load_from_records(self, records: list) -> None:
        """
        Загружает топики из списка словарей.
        Каждый словарь: {"topic_id": int, "parent_id": int|None, "name": str}
        """
        self._topics = {}
        for rec in records:
            tid = rec.get("topic_id")
            if tid is None:
                continue
            self._topics[int(tid)] = {
                "name": str(rec.get("name", "")).strip(),
                "parent_id": rec.get("parent_id"),
            }
        logger.info(f"✅ Загружено {len(self._topics)} топиков.")

    # ------------------------------------------------------------------
    # Получение breadcrumb
    # ------------------------------------------------------------------

    def get_breadcrumb(self, topic_id: Optional[int]) -> Optional[str]:
        """
        Возвращает строку хлебных крошек для topic_id.

        Пример: "Про деньги > Стипендии > Правительства РФ"
        Если topic_id=None или не найден — возвращает None.
        """
        if topic_id is None:
            return None
        topic_id = int(topic_id)
        path = []
        visited = set()
        current_id: Optional[int] = topic_id
        while current_id is not None:
            if current_id in visited:
                logger.warning(f"Цикл в иерархии топиков для topic_id={topic_id}")
                break
            visited.add(current_id)
            node = self._topics.get(current_id)
            if node is None:
                break
            path.append(node["name"])
            parent = node["parent_id"]
            current_id = int(parent) if parent is not None else None

        if not path:
            return None
        # Путь собрали от листа к корню — переворачиваем
        return " > ".join(reversed(path))

    def is_loaded(self) -> bool:
        return bool(self._topics)
