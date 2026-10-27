"""Эмбеддинги быстрых ответов и полезных ссылок.

Строки library.quick_answer_variants и library.links без вектора ждут воркера: их добавляет
db-svc (импорт), а триггер шлёт NOTIFY в тот же канал, что и документы. Здесь они
досчитываются тем же embedding-svc и той же моделью, что и чанки (kind="passage").

Текст варианта берётся как есть: нормализацию (нижний регистр, удаление всего, кроме букв,
цифр и пробелов, strip) делает импортёр db-svc, а graph-rag-svc применяет ту же функцию к
вопросу, поэтому здесь повторять её нельзя — сравнивались бы разные строки.

Модуль не импортирует main.py: воркер запускается как `python app/main.py`, и повторный
импорт создал бы второй движок БД и задвоил метрики. Зависимости передаёт main.main() через
configure().
"""

import logging
import re

from sqlalchemy import text

from metrics import AUX_EMBEDDED

logger = logging.getLogger("ingest-worker")

# Заполняются в configure(); в тестах подменяются напрямую.
_embed = None
async_session = None


def configure(embed, session_factory) -> None:
    """Подключает _embed и фабрику сессий из main.py."""
    global _embed, async_session
    _embed = embed
    async_session = session_factory


# Единое определение «пустого» текста для SQL и Python: непустой — содержит хотя бы один символ
# вне ASCII-пробельных ` \t\n\r\f\v` (NBSP и юникодные пробелы считаются непустыми с обеих
# сторон). Строка попадает в SQL как есть: Postgres сам разворачивает \t, \n... в регулярном
# выражении, поэтому экранировать их через E'' не нужно. Расхождение определений привело бы к
# тому, что отброшенная Python строка выбиралась бы снова и блокировала очередь.
NON_BLANK_CLASS = r"[^ \t\n\r\f\v]"
_NON_BLANK_RE = re.compile(NON_BLANK_CLASS)


def _is_blank(value) -> bool:
    return value is None or _NON_BLANK_RE.search(value) is None


# FOR UPDATE ... SKIP LOCKED: два воркера не берут одни и те же строки. У варианта признак
# активности лежит в родительском быстром ответе.
SELECT_VARIANTS = text(f"""
    SELECT v.id, v.text
    FROM library.quick_answer_variants v
    JOIN library.quick_answers qa ON qa.id = v.quick_answer_id
    WHERE v.embedding IS NULL AND qa.is_active AND v.text ~ '{NON_BLANK_CLASS}'
    ORDER BY v.id
    LIMIT :limit
    FOR UPDATE OF v SKIP LOCKED
""")
SELECT_LINKS = text(f"""
    SELECT id, description
    FROM library.links
    WHERE embedding IS NULL AND is_active AND description ~ '{NON_BLANK_CLASS}'
    ORDER BY id
    LIMIT :limit
    FOR UPDATE SKIP LOCKED
""")

UPDATE_VARIANTS = text("""
    UPDATE library.quick_answer_variants
    SET embedding = CAST(:vec AS vector), embedding_model = :model
    WHERE id = :id
""")
UPDATE_LINKS = text("""
    UPDATE library.links
    SET embedding = CAST(:vec AS vector), embedding_model = :model
    WHERE id = :id
""")


async def _embed_pending(select, update, kind: str, limit: int) -> int:
    if _embed is None or async_session is None:
        raise RuntimeError("aux_index.configure() не вызван")

    async with async_session() as session:
        rows = (await session.execute(select, {"limit": limit})).all()
        # Пустые тексты отсеивает уже SQL с тем же определением (NON_BLANK_CLASS), так что здесь
        # фильтр срабатывает лишь на случай расхождения: пустая строка не уйдёт в embedding-svc.
        rows = [row for row in rows if not _is_blank(row[1])]
        if not rows:
            return 0

        # Исключение embedding-svc уходит наверх до первой записи; незавершённая транзакция
        # откатывается при выходе из сессии, строки остаются без вектора и будут взяты снова.
        vectors, model = await _embed([row[1] for row in rows])
        if len(vectors) != len(rows):
            raise RuntimeError(
                f"embedding-svc вернул {len(vectors)} векторов на {len(rows)} текстов ({kind})"
            )

        await session.execute(update, [
            {"id": row[0], "vec": str([float(x) for x in vec]), "model": model}
            for row, vec in zip(rows, vectors)
        ])
        await session.commit()

    AUX_EMBEDDED.labels(kind=kind).inc(len(rows))
    logger.info("Эмбеддинги (%s): %d", kind, len(rows))
    return len(rows)


async def embed_pending_quick_answers(limit: int = 256) -> int:
    """Считает векторы активных вариантов быстрых ответов без эмбеддинга; возвращает их число."""
    return await _embed_pending(SELECT_VARIANTS, UPDATE_VARIANTS, "quick_answer_variant", limit)


async def embed_pending_links(limit: int = 256) -> int:
    """Считает векторы описаний активных ссылок без эмбеддинга; возвращает их число."""
    return await _embed_pending(SELECT_LINKS, UPDATE_LINKS, "link", limit)


async def embed_pending_aux() -> int:
    """Один проход по обеим таблицам; возвращает, сколько строк обработано суммарно."""
    return await embed_pending_quick_answers() + await embed_pending_links()
