import asyncio
import time
import logging
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp
import asyncpg

from sqlalchemy import select, delete, text, func, or_, and_, ForeignKey, Text, Integer, DateTime, BigInteger, Boolean, Double
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, mapped_column, Mapped
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

from errors import PermanentIngestError
from parsers import get_content, parse_pdf, parse_docx, parse_txt
from chunker import SmartChunker

load_dotenv()

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

DATABASE_URL  = os.environ["DATABASE_URL"]
CHUNK_SIZE    = int(os.environ.get("CHUNK_SIZE", 512))
CHUNK_MIN     = int(os.environ.get("CHUNK_MIN_TOKENS", 50))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP_TOKENS", 32))

# Эмбеддинги считает ТОЛЬКО embedding-svc — единый источник на всю систему.
# Имя модели здесь намеренно не задаётся: авторитетным является то, что вернул сервис,
# иначе появляется второе место, объявляющее модель, и они расходятся.
EMBEDDING_SVC_URL        = os.environ.get("EMBEDDING_SVC_URL", "http://embedding-svc:8000")
EMBEDDING_DIM            = int(os.environ.get("EMBEDDING_DIM", 1024))
EMBED_REQUEST_TIMEOUT    = int(os.environ.get("EMBED_REQUEST_TIMEOUT_SEC", 120))
# Размер батча при обращении к embedding-svc. Должен быть заведомо меньше лимита на
# стороне сервиса: документ может дать сотни чанков, и отправлять их одним запросом
# значит рисковать памятью общего для всей системы сервиса эмбеддингов.
EMBED_BATCH_SIZE         = int(os.environ.get("EMBED_BATCH_SIZE", 64))
EMBED_STARTUP_RETRIES    = int(os.environ.get("EMBED_STARTUP_RETRIES", 30))
EMBED_STARTUP_RETRY_WAIT = int(os.environ.get("EMBED_STARTUP_RETRY_WAIT_SEC", 5))

# Повторы обработки документа. Ошибка, которая может пройти сама (недоступен
# embedding-svc, MinIO, БД), возвращает документ в pending с отложенным временем следующей
# попытки: пауза INGEST_RETRY_BASE_SEC удваивается с каждой попыткой до INGEST_RETRY_MAX_SEC.
# После INGEST_MAX_ATTEMPTS попыток — failed. Ошибки, которые повтор не исправит
# (PermanentIngestError: слишком большой или битый файл), дают failed сразу.
INGEST_MAX_ATTEMPTS   = int(os.environ.get("INGEST_MAX_ATTEMPTS", 5))
INGEST_RETRY_BASE_SEC = int(os.environ.get("INGEST_RETRY_BASE_SEC", 30))
INGEST_RETRY_MAX_SEC  = int(os.environ.get("INGEST_RETRY_MAX_SEC", 3600))
# Документ в processing дольше этого срока считается брошенным (воркер упал или был
# перезапущен посреди обработки) и забирается снова — это тоже расходует попытку.
# Должен быть заведомо больше времени обработки самого большого документа.
INGEST_PROCESSING_TIMEOUT_SEC = int(os.environ.get("INGEST_PROCESSING_TIMEOUT_SEC", 1800))

# Доставка задач. Новые задачи приходят через LISTEN/NOTIFY: триггер на library.documents
# шлёт уведомление в канал INGEST_NOTIFY_CHANNEL, когда документ становится pending
# (миграция 006_ingest_queue.sql). Опрос остаётся страховкой: раз в INGEST_POLL_FALLBACK_SEC
# при работающем LISTEN и раз в INGEST_POLL_DEGRADED_SEC, если соединение для LISTEN потеряно.
INGEST_NOTIFY_CHANNEL    = os.environ.get("INGEST_NOTIFY_CHANNEL", "ingest_jobs")
INGEST_POLL_FALLBACK_SEC = int(os.environ.get("INGEST_POLL_FALLBACK_SEC", 60))
INGEST_POLL_DEGRADED_SEC = int(os.environ.get("INGEST_POLL_DEGRADED_SEC", 5))

# Healthcheck. HTTP-сервера у воркера нет, поэтому живость сообщается файлом: фоновая задача
# обновляет его время изменения каждые HEARTBEAT_INTERVAL_SEC, а healthcheck в compose
# проверяет, что файл свежий. Задача запускается только после стартовых проверок, так что
# свежий файл значит «проверки пройдены и event loop не завис».
HEARTBEAT_FILE         = os.environ.get("HEARTBEAT_FILE", "/tmp/ingest-worker.heartbeat")
HEARTBEAT_INTERVAL_SEC = int(os.environ.get("HEARTBEAT_INTERVAL_SEC", 10))

SERVICE_NAME                 = os.environ.get("SERVICE_NAME", "ingest-worker")
SERVICE_TOKEN                = os.environ.get("SERVICE_TOKEN", "")
INTERNAL_AUTH_HEADER_NAME    = os.environ.get("INTERNAL_AUTH_HEADER_NAME", "X-Service-Token")
INTERNAL_SERVICE_NAME_HEADER = os.environ.get("INTERNAL_SERVICE_NAME_HEADER", "X-Service-Name")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("IngestWorker")

# ---------------------------------------------------------------------------
# Модели данных (схема library / core)
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class University(Base):
    __tablename__ = "universities"
    __table_args__ = {"schema": "core"}
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    short_name: Mapped[Optional[str]] = mapped_column(Text)


class Campus(Base):
    __tablename__ = "campuses"
    __table_args__ = {"schema": "core"}
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    university_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("core.universities.id"))
    address_text: Mapped[Optional[str]] = mapped_column(Text)


class Faculty(Base):
    __tablename__ = "faculties"
    __table_args__ = {"schema": "core"}
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    university_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("core.universities.id"))
    name: Mapped[str] = mapped_column(Text, nullable=False)


class Building(Base):
    __tablename__ = "buildings"
    __table_args__ = {"schema": "core"}
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campus_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("core.campuses.id"))
    name: Mapped[str] = mapped_column(Text, nullable=False)
    address: Mapped[Optional[str]] = mapped_column(Text)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = {"schema": "library"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True))
    title: Mapped[str] = mapped_column(Text, nullable=False)
    ingest_status: Mapped[str] = mapped_column(Text, default="pending")
    ingest_error: Mapped[Optional[str]] = mapped_column(Text)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    checksum: Mapped[Optional[str]] = mapped_column(Text)
    content_version: Mapped[int] = mapped_column(Integer, default=1)
    # Колонки очереди (миграция 006_ingest_queue.sql).
    ingest_attempts: Mapped[int] = mapped_column(Integer, default=0)
    ingest_next_attempt_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    ingest_started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class DocumentFile(Base):
    __tablename__ = "document_files"
    __table_args__ = {"schema": "library"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("library.documents.id", ondelete="CASCADE"))
    storage_backend: Mapped[str] = mapped_column(Text, default="local")
    storage_path: Mapped[str] = mapped_column(Text)
    bucket: Mapped[Optional[str]] = mapped_column(Text)
    storage_key: Mapped[Optional[str]] = mapped_column(Text)
    file_name: Mapped[str] = mapped_column(Text)
    mime_type: Mapped[str] = mapped_column(Text)


class Contact(Base):
    """Контактные лица (библиотека, деканаты, etc.)."""
    __tablename__ = "contacts"
    __table_args__ = {"schema": "library"}
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[Optional[str]] = mapped_column(Text)
    email: Mapped[Optional[str]] = mapped_column(Text)
    phone: Mapped[Optional[str]] = mapped_column(Text)
    website_url: Mapped[Optional[str]] = mapped_column(Text)
    office_text: Mapped[Optional[str]] = mapped_column(Text)
    building_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("core.buildings.id"))
    floor: Mapped[Optional[str]] = mapped_column(Text)
    room: Mapped[Optional[str]] = mapped_column(Text)
    hours_json: Mapped[Optional[dict]] = mapped_column(JSONB)
    description: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    scope_json: Mapped[Optional[dict]] = mapped_column(JSONB)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class Place(Base):
    """Физические места (аудитории, библиотеки, кабинеты)."""
    __tablename__ = "places"
    __table_args__ = {"schema": "library"}
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    place_type: Mapped[Optional[str]] = mapped_column(Text)
    building_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("core.buildings.id"))
    address_text: Mapped[Optional[str]] = mapped_column(Text)
    floor: Mapped[Optional[str]] = mapped_column(Text)
    room: Mapped[Optional[str]] = mapped_column(Text)
    latitude: Mapped[Optional[float]] = mapped_column(Double)
    longitude: Mapped[Optional[float]] = mapped_column(Double)
    description: Mapped[Optional[str]] = mapped_column(Text)
    how_to_find: Mapped[Optional[str]] = mapped_column(Text)
    hours_json: Mapped[Optional[dict]] = mapped_column(JSONB)
    contacts_json: Mapped[Optional[dict]] = mapped_column(JSONB)
    map_url: Mapped[Optional[str]] = mapped_column(Text)
    scope_json: Mapped[Optional[dict]] = mapped_column(JSONB)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = {"schema": "library"}

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("library.documents.id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    section_path: Mapped[Optional[str]] = mapped_column(Text)
    start_offset: Mapped[Optional[int]] = mapped_column(Integer)
    end_offset: Mapped[Optional[int]] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer)
    metadata_json: Mapped[dict] = mapped_column(JSONB, default={})


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"
    __table_args__ = {"schema": "library"}

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("library.chunks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding: Mapped[list] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str] = mapped_column(Text)


# ---------------------------------------------------------------------------
# Сервисы (движок БД, chunker) — эмбеддинги считает embedding-svc, см. _embed()
# ---------------------------------------------------------------------------

engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# SmartChunker — выбирает метод автоматически по типу файла
chunker = SmartChunker(
    chunk_size=CHUNK_SIZE,
    min_tokens=CHUNK_MIN,
    overlap_tokens=CHUNK_OVERLAP,
)


def _internal_auth_headers() -> dict:
    return {
        INTERNAL_SERVICE_NAME_HEADER: SERVICE_NAME,
        INTERNAL_AUTH_HEADER_NAME: SERVICE_TOKEN,
    }


async def _embed_batch(session: aiohttp.ClientSession, texts: list) -> tuple:
    """Один запрос к embedding-svc. Возвращает (векторы, имя модели)."""
    async with session.post(
        f"{EMBEDDING_SVC_URL.rstrip('/')}/embed",
        json={"texts": texts, "normalize": True},
        headers=_internal_auth_headers(),
    ) as response:
        response.raise_for_status()
        data = await response.json()

    vectors = data["embeddings"]
    dimension = data.get("dimension") or (len(vectors[0]) if vectors else 0)
    if dimension != EMBEDDING_DIM:
        raise RuntimeError(
            f"embedding-svc вернул векторы размерности {dimension}, "
            f"а ожидается {EMBEDDING_DIM} (EMBEDDING_DIM)"
        )
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"embedding-svc вернул {len(vectors)} векторов на {len(texts)} текстов"
        )
    return vectors, data["model"]


async def _embed(texts: list) -> tuple:
    """Считает эмбеддинги через embedding-svc. Возвращает (векторы, имя модели).

    Отправка идёт батчами по EMBED_BATCH_SIZE: у документа могут быть сотни чанков, а
    embedding-svc — общий для всей системы, и один огромный запрос способен исчерпать его
    память, остановив заодно и поиск.

    Ни дополнения нулями, ни обрезки: несовпадение размерности — это ошибка конфигурации,
    а не повод подгонять данные. Прежняя версия грузила собственный SentenceTransformer и
    подгоняла всё под 1536 измерений, из-за чего вставка в колонку VECTOR(1024) падала и
    откатывала документ целиком вместе с чанками.
    """
    if not texts:
        return [], ""

    vectors: list = []
    model_name = ""
    timeout = aiohttp.ClientTimeout(total=EMBED_REQUEST_TIMEOUT)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            batch_vectors, batch_model = await _embed_batch(session, batch)

            # Модель не должна смениться посреди документа: иначе часть чанков окажется
            # в одном векторном пространстве, часть — в другом, и поиск станет молча врать.
            if model_name and batch_model != model_name:
                raise RuntimeError(
                    f"embedding-svc сменил модель посреди документа: "
                    f"было '{model_name}', стало '{batch_model}'"
                )
            model_name = batch_model
            vectors.extend(batch_vectors)

    return vectors, model_name


async def _column_dimension(schema_table: str, column: str) -> Optional[int]:
    """Фактическая размерность pgvector-колонки в БД (None — если не задана)."""
    async with async_session() as session:
        declared = (
            await session.execute(
                text(
                    # to_regclass(:rel), а не :rel::regclass — SQLAlchemy не распознаёт
                    # параметр, за которым сразу идёт `::`, и принимает его за оператор
                    # приведения типа: `:rel` уходил в Postgres буквально и валил запрос.
                    # Побочный плюс: для несуществующей таблицы функция вернёт NULL,
                    # а приведение бросало бы исключение.
                    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                    "WHERE attrelid = to_regclass(:rel) AND attname = :col AND NOT attisdropped"
                ),
                {"rel": schema_table, "col": column},
            )
        ).scalar()

    if not declared:
        return None
    match = re.search(r"\((\d+)\)", declared)
    return int(match.group(1)) if match else None


async def verify_embedding_setup() -> None:
    """Сверяет «модель ↔ колонка БД» ДО начала обработки документов.

    Раньше расхождение размерности обнаруживалось только в момент вставки — и, поскольку
    чанки и эмбеддинги пишутся одной транзакцией, откатывало документ целиком, помечая его
    как failed. Теперь воркер просто не стартует и пишет, что именно с чем разошлось.

    embedding-svc грузит модель не мгновенно, поэтому проверка повторяется с паузой, а не
    падает на первой же неудаче.
    """
    info = None
    last_error = None
    for attempt in range(1, EMBED_STARTUP_RETRIES + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=EMBED_REQUEST_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{EMBEDDING_SVC_URL.rstrip('/')}/ping") as response:
                    response.raise_for_status()
                    info = await response.json()
            break
        except Exception as exc:
            last_error = exc
            logger.info(
                f"Ожидаем embedding-svc ({attempt}/{EMBED_STARTUP_RETRIES}): {exc}"
            )
            await asyncio.sleep(EMBED_STARTUP_RETRY_WAIT)

    if info is None:
        raise RuntimeError(
            f"embedding-svc недоступен по адресу {EMBEDDING_SVC_URL}: {last_error}"
        )

    service_dim = info.get("dimension")
    if service_dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"Модель '{info.get('model')}' в embedding-svc даёт {service_dim} измерений, "
            f"а EMBEDDING_DIM={EMBEDDING_DIM}. Смена модели эмбеддингов требует пересчёта "
            f"всех чанков и графа — см. README graph-rag-svc, §13.1."
        )

    column_dim = await _column_dimension("library.chunk_embeddings", "embedding")
    if column_dim is None:
        raise RuntimeError(
            "Не удалось определить размерность library.chunk_embeddings.embedding: таблицы "
            "или колонки нет, либо тип объявлен без размера. Примените схему БД перед запуском."
        )
    if column_dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"Колонка library.chunk_embeddings.embedding объявлена как VECTOR({column_dim}), "
            f"а EMBEDDING_DIM={EMBEDDING_DIM}. Приведите схему БД и конфигурацию к одному "
            f"значению перед запуском."
        )

    logger.info(
        f"Эмбеддинги: модель '{info.get('model')}', {EMBEDDING_DIM} измерений — "
        f"согласовано с library.chunk_embeddings"
    )


# ---------------------------------------------------------------------------
# Определение типа файла
# ---------------------------------------------------------------------------

def _detect_doc_type(file_name: str, mime_type: str) -> str:
    """Возвращает 'pdf' | 'docx' | 'txt' по имени файла или MIME-типу."""
    name = file_name.lower()
    if name.endswith(".pdf") or "pdf" in mime_type:
        return "pdf"
    if name.endswith(".docx") or "wordprocessingml" in mime_type:
        return "docx"
    return "txt"


# ---------------------------------------------------------------------------
# Основной pipeline обработки документа
# ---------------------------------------------------------------------------

_PARSERS = {"pdf": parse_pdf, "docx": parse_docx, "txt": parse_txt}


def _claimable():
    """Условие «документ можно брать в работу».

    - pending, у которого наступило время попытки (NULL — новый документ, без задержки);
    - processing, брошенный дольше INGEST_PROCESSING_TIMEOUT_SEC назад. Раньше документ,
      на котором воркер упал или был перезапущен, навсегда оставался в processing: статус
      коммитится до обработки, и повторно его никто не брал. NULL в ingest_started_at —
      документы, застрявшие так до миграции 006.
    """
    now = func.now()
    stale_before = now - func.make_interval(0, 0, 0, 0, 0, 0, INGEST_PROCESSING_TIMEOUT_SEC)
    return or_(
        and_(
            Document.ingest_status == "pending",
            or_(Document.ingest_next_attempt_at.is_(None), Document.ingest_next_attempt_at <= now),
        ),
        and_(
            Document.ingest_status == "processing",
            or_(Document.ingest_started_at.is_(None), Document.ingest_started_at < stale_before),
        ),
    )


def _retry_delay_sec(attempt: int) -> int:
    return min(INGEST_RETRY_BASE_SEC * 2 ** (attempt - 1), INGEST_RETRY_MAX_SEC)


async def _record_failure(doc_id: uuid.UUID, attempt: int, exc: Exception) -> None:
    """Пишет результат неудачной попытки: отложенный повтор либо окончательный failed.

    Отдельной сессией: основная откатилась вместе с частично записанными чанками.
    """
    permanent = isinstance(exc, PermanentIngestError)
    if permanent or attempt >= INGEST_MAX_ATTEMPTS:
        status, next_at = "failed", None
        err_text = str(exc) if permanent else f"Попытка {attempt}/{INGEST_MAX_ATTEMPTS}: {exc}"
        logger.error(f"Ошибка на {doc_id}, документ помечен failed: {err_text}")
    else:
        delay = _retry_delay_sec(attempt)
        status, next_at = "pending", datetime.now(timezone.utc) + timedelta(seconds=delay)
        err_text = f"Попытка {attempt}/{INGEST_MAX_ATTEMPTS}: {exc}"
        logger.warning(f"Ошибка на {doc_id}, повтор через {delay} с: {err_text}")

    async with async_session() as err_session:
        try:
            await err_session.execute(
                text("""
                    UPDATE library.documents
                    SET ingest_status          = :status,
                        ingest_error           = :err,
                        ingest_next_attempt_at = :next_at
                    WHERE id = :id
                """),
                {"status": status, "err": err_text, "next_at": next_at, "id": doc_id},
            )
            await err_session.commit()
        except Exception as db_err:
            # Документ остаётся в processing и будет забран снова по INGEST_PROCESSING_TIMEOUT_SEC.
            logger.error(f"Не удалось записать ошибку в БД: {db_err}")


async def process_one_document() -> bool:
    async with async_session() as session:
        # 1. Выбираем задачу (SKIP LOCKED — параллельность без коллизий)
        query = (
            select(Document, DocumentFile)
            .join(DocumentFile, Document.id == DocumentFile.document_id)
            .where(_claimable())
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        res = await session.execute(query)
        result_row = res.unique().fetchone()
        if not result_row:
            return False

        doc, df = result_row
        doc_id = doc.id

        # Попытки кончились, а документ снова в очереди. Обычный путь сюда не ведёт: после
        # последней неудачной попытки документ сразу получает failed. Попадает сюда брошенный
        # processing — документ, на котором воркер падал целиком (например, по памяти) и не
        # успевал записать ошибку. Без этой проверки такой документ валил бы воркер бесконечно.
        if doc.ingest_attempts >= INGEST_MAX_ATTEMPTS:
            if doc.ingest_status == "processing":
                reason = (
                    "обработка прерывалась без записи результата — вероятно, воркер падал "
                    "на этом документе"
                )
            else:
                reason = f"последняя ошибка: {doc.ingest_error}"
            doc.ingest_status = "failed"
            doc.ingest_next_attempt_at = None
            doc.ingest_error = f"Исчерпано {doc.ingest_attempts} попыток обработки; {reason}"
            await session.commit()
            logger.error(f"{doc.title}: {doc.ingest_error}")
            return True

        attempt = doc.ingest_attempts + 1
        logger.info(f"Обработка: {doc.title} (попытка {attempt}/{INGEST_MAX_ATTEMPTS})")

        try:
            # 2. Ставим статус "processing"
            doc.ingest_status = "processing"
            doc.ingest_attempts = attempt
            doc.ingest_started_at = func.now()
            await session.commit()
            await session.refresh(doc)

            # 3. Загрузка файла (с проверкой MAX_FILE_SIZE_MB)
            content = await get_content(
                df.storage_path,
                df.storage_backend,
                df.bucket,
                df.storage_key,
            )

            # 4. Парсинг — все парсеры возвращают (text, sections). В отдельном потоке:
            # разбор большого PDF занимает секунды, и всё это время event loop не обслуживал
            # бы ни heartbeat, ни соединение LISTEN.
            doc_type = _detect_doc_type(df.file_name, df.mime_type)
            text_data, sections = await asyncio.to_thread(_PARSERS[doc_type], content)

            # 5. Умная нарезка (метод выбирается автоматически)
            chunks_list = await chunker.create_chunks(
                text=text_data,
                doc_title=doc.title,
                doc_type=doc_type,
                sections=sections,
            )

            # 6. Удаляем старые чанки, пишем новые
            await session.execute(delete(Chunk).where(Chunk.document_id == doc_id))

            if chunks_list:
                embeddings, embedding_model = await _embed([c["text"] for c in chunks_list])
                logger.info(f"Создано чанков: {len(chunks_list)}")

                for i, (c_data, v) in enumerate(zip(chunks_list, embeddings)):
                    new_chunk = Chunk(
                        document_id=doc_id,
                        chunk_index=i,
                        text=c_data["text"],
                        section_path=c_data["meta"].get("section_path"),
                        start_offset=c_data["meta"].get("start_offset"),
                        end_offset=c_data["meta"].get("end_offset"),
                        token_count=c_data["meta"].get("token_count", 0),
                        metadata_json=c_data["meta"],
                    )
                    session.add(new_chunk)
                    await session.flush()

                    session.add(ChunkEmbedding(
                        chunk_id=new_chunk.id,
                        embedding=v,
                        embedding_model=embedding_model,
                    ))

            # 7. Финализация. Счётчик попыток обнуляется: если документ позже вернут в
            # pending (новая версия), у него снова будет полный запас попыток.
            doc.ingest_status = "indexed"
            doc.indexed_at = datetime.now()
            doc.ingest_error = None
            doc.ingest_attempts = 0
            doc.ingest_next_attempt_at = None
            await session.commit()
            logger.info(f"Готово: {doc.title}")

        except Exception as e:
            await session.rollback()
            await _record_failure(doc_id, attempt, e)

    return True


# ---------------------------------------------------------------------------
# Доставка задач: LISTEN/NOTIFY + страховочный опрос
# ---------------------------------------------------------------------------

class JobSignal:
    """Будит главный цикл по уведомлению из Postgres.

    Раньше воркер опрашивал БД каждые 5 секунд: задержка до 5 с на каждый новый документ и
    постоянные запросы в простое. Теперь триггер на library.documents шлёт NOTIFY, когда
    документ становится pending, и воркер просыпается сразу. Отдельный брокер не нужен, и
    сервисы, создающие документы (library-svc, db-svc, crawler-svc), менять не пришлось:
    уведомление шлёт сама БД при любом способе записи.

    LISTEN держится на отдельном соединении asyncpg — пул SQLAlchemy раздаёт соединения на
    время транзакции и для подписки не подходит. Если соединение потеряно, воркер не падает,
    а переходит на опрос раз в INGEST_POLL_DEGRADED_SEC и переподключается на каждом круге.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._conn: Optional[asyncpg.Connection] = None
        self._dsn = make_url(DATABASE_URL).set(drivername="postgresql").render_as_string(
            hide_password=False
        )
        self._degraded_logged = False

    @property
    def connected(self) -> bool:
        return self._conn is not None and not self._conn.is_closed()

    async def connect(self) -> None:
        if self.connected:
            return
        try:
            self._conn = await asyncpg.connect(self._dsn)
            await self._conn.add_listener(INGEST_NOTIFY_CHANNEL, self._on_notify)
            # Обрыв соединения будит цикл сразу: он проверит очередь (уведомления за время
            # обрыва потеряны) и переподключится, а не проспит до страховочного опроса.
            self._conn.add_termination_listener(self._on_notify)
            self._degraded_logged = False
            logger.info(f"Подписка на уведомления о задачах: LISTEN {INGEST_NOTIFY_CHANNEL}")
            # Пока подписки не было, уведомления терялись — проверяем очередь сразу.
            self._event.set()
        except Exception as exc:
            self._conn = None
            if not self._degraded_logged:
                logger.warning(
                    f"LISTEN {INGEST_NOTIFY_CHANNEL} недоступен, работаем опросом раз в "
                    f"{INGEST_POLL_DEGRADED_SEC} с: {exc}"
                )
                self._degraded_logged = True

    def _on_notify(self, *_args) -> None:
        self._event.set()

    async def wait(self, timeout: float) -> None:
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
        except asyncio.TimeoutError:
            pass
        self._event.clear()


async def _seconds_until_next_retry() -> Optional[float]:
    """Через сколько секунд наступит ближайший отложенный повтор (None — таких нет).

    NOTIFY приходит, когда документ становится pending, а не когда наступает время его
    повтора, поэтому время пробуждения под повтор считает сам воркер.
    """
    async with async_session() as session:
        return (
            await session.execute(
                text(
                    "SELECT EXTRACT(EPOCH FROM min(ingest_next_attempt_at) - now()) "
                    "FROM library.documents "
                    "WHERE ingest_status = 'pending' AND ingest_next_attempt_at > now()"
                )
            )
        ).scalar()


async def _heartbeat() -> None:
    while True:
        try:
            Path(HEARTBEAT_FILE).touch()
        except OSError as exc:
            logger.error(f"Не удалось обновить heartbeat-файл {HEARTBEAT_FILE}: {exc}")
        await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------

async def main():
    # Файл от прошлого запуска контейнера (restart сохраняет файловую систему) не должен
    # выдавать ещё не прошедший стартовые проверки воркер за здоровый.
    Path(HEARTBEAT_FILE).unlink(missing_ok=True)

    logger.info("IngestWorker запущен, проверяем конфигурацию эмбеддингов...")
    await verify_embedding_setup()

    signal = JobSignal()
    await signal.connect()
    heartbeat = asyncio.create_task(_heartbeat())  # noqa: F841 — держим ссылку, чтобы задачу не собрал GC
    logger.info("Ожидаем задачи...")

    # Отчёт о завершении прохода. Раньше строка «Ожидаем задачи...» печаталась только один
    # раз, до цикла, а опустевшую очередь воркер встречал молчанием: лог просто обрывался на
    # последнем документе. По нему нельзя было отличить «всё обработано» от «завис», из-за
    # чего приходилось лезть в базу и считать статусы руками.
    #
    # Пишем ровно на переходе «работал → очередь пуста», а не на каждом пробуждении: иначе
    # простой засыпал бы лог одинаковыми строками.
    batch_processed = 0
    batch_started: float | None = None

    while True:
        try:
            # Засечка берётся ДО обработки: иначе отсчёт начинался бы уже после первого
            # документа и проход из одного документа всегда показывал бы 0.0 с.
            tick = time.monotonic()
            processed = await process_one_document()
            if processed:
                if batch_started is None:
                    batch_started = tick
                batch_processed += 1
                continue

            if batch_processed:
                elapsed = time.monotonic() - (batch_started or time.monotonic())
                logger.info(
                    "Очередь пуста: обработано документов за проход — %d, заняло %.1f с. "
                    "Ожидаем задачи...",
                    batch_processed,
                    elapsed,
                )
                batch_processed = 0
                batch_started = None

            await signal.connect()
            timeout = INGEST_POLL_FALLBACK_SEC if signal.connected else INGEST_POLL_DEGRADED_SEC
            retry_in = await _seconds_until_next_retry()
            if retry_in is not None:
                timeout = min(timeout, max(float(retry_in), 0.5))
            await signal.wait(timeout)
        except Exception as e:
            logger.error(f"Критический сбой цикла: {e}")
            await asyncio.sleep(INGEST_POLL_DEGRADED_SEC)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка воркера")
