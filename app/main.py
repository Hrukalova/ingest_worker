import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

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
EMBEDDER_NAME = os.environ.get("EMBEDDER_MODEL_NAME", "mixedbread-ai/mxbai-embed-large-v1")
CHUNK_SIZE    = int(os.environ.get("CHUNK_SIZE", 512))
CHUNK_MIN     = int(os.environ.get("CHUNK_MIN_TOKENS", 50))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP_TOKENS", 32))

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
    embedding: Mapped[list] = mapped_column(Vector(1536))
    embedding_model: Mapped[str] = mapped_column(Text)


# ---------------------------------------------------------------------------
# Сервисы (движок БД, embedder, chunker)
# ---------------------------------------------------------------------------

engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Embedder
try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBEDDER_NAME)
    logger.info(f"Модель {EMBEDDER_NAME} загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки embedder: {e}")
    embedder = None

# SmartChunker — выбирает метод автоматически по типу файла
chunker = SmartChunker(
    chunk_size=CHUNK_SIZE,
    min_tokens=CHUNK_MIN,
    overlap_tokens=CHUNK_OVERLAP,
)


def _embed(texts: list) -> list:
    """Генерация эмбеддингов с дополнением до 1536 измерений."""
    if not embedder:
        return [[0.0] * 1536 for _ in texts]
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    result = []
    for v in vectors:
        v_list = v.tolist()
        if len(v_list) < 1536:
            v_list += [0.0] * (1536 - len(v_list))
        result.append(v_list[:1536])
    return result


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
                embeddings = _embed([c["text"] for c in chunks_list])
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
                        embedding_model=EMBEDDER_NAME,
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
    logger.info("IngestWorker запущен, ожидаем задачи...")
    while True:
        try:
            processed = await process_one_document()
            if not processed:
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Критический сбой цикла: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановка воркера")
