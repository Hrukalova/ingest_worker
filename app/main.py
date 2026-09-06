import asyncio
import time
import logging
import os
import re
import uuid
from datetime import datetime
from typing import Optional

import aiohttp

from sqlalchemy import select, delete, text, ForeignKey, Text, Integer, DateTime, BigInteger, Boolean, Double
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, mapped_column, Mapped
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

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

async def process_one_document() -> bool:
    async with async_session() as session:
        # 1. Выбираем задачу (SKIP LOCKED — параллельность без коллизий)
        query = (
            select(Document, DocumentFile)
            .join(DocumentFile, Document.id == DocumentFile.document_id)
            .where(Document.ingest_status == "pending")
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        res = await session.execute(query)
        result_row = res.unique().fetchone()
        if not result_row:
            return False

        doc, df = result_row
        doc_id = doc.id
        logger.info(f"Обработка: {doc.title}")

        try:
            # 2. Ставим статус "processing"
            doc.ingest_status = "processing"
            await session.commit()
            await session.refresh(doc)

            # 3. Загрузка файла
            content = await get_content(
                df.storage_path,
                df.storage_backend,
                df.bucket,
                df.storage_key,
            )

            # 4. Парсинг — все парсеры возвращают (text, sections)
            doc_type = _detect_doc_type(df.file_name, df.mime_type)
            if doc_type == "pdf":
                text_data, sections = parse_pdf(content)
            elif doc_type == "docx":
                text_data, sections = parse_docx(content)
            else:
                text_data, sections = parse_txt(content)

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

            # 7. Финализация
            doc.ingest_status = "indexed"
            doc.indexed_at = datetime.now()
            doc.ingest_error = None
            await session.commit()
            logger.info(f"Готово: {doc.title}")

        except Exception as e:
            await session.rollback()
            err_text = str(e)
            logger.error(f"Ошибка на {doc_id}: {err_text}")

            async with async_session() as err_session:
                try:
                    await err_session.execute(
                        text("""
                            UPDATE library.documents
                            SET ingest_status = 'failed',
                                ingest_error  = :err
                            WHERE id = :id
                        """),
                        {"err": err_text, "id": doc_id},
                    )
                    await err_session.commit()
                except Exception as db_err:
                    logger.error(f"Не удалось записать ошибку в БД: {db_err}")

    return True


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------

async def main():
    logger.info("IngestWorker запущен, проверяем конфигурацию эмбеддингов...")
    await verify_embedding_setup()
    logger.info("Ожидаем задачи...")

    # Отчёт о завершении прохода. Раньше строка «Ожидаем задачи...» печаталась только один
    # раз, до цикла, а опустевшую очередь воркер встречал молчанием: лог просто обрывался на
    # последнем документе. По нему нельзя было отличить «всё обработано» от «завис», из-за
    # чего приходилось лезть в базу и считать статусы руками.
    #
    # Пишем ровно на переходе «работал → очередь пуста», а не каждые 5 секунд: иначе простой
    # засыпал бы лог одинаковыми строками.
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

            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Критический сбой цикла: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка воркера")
