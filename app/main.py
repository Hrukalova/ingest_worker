import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select, delete, text, ForeignKey, Text, Integer, DateTime, BigInteger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, mapped_column, Mapped
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv

# Импорт наших улучшенных модулей
from parsers import get_content, parse_pdf, parse_docx, parse_txt
from chunker import SemanticChunker

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Конфигурация
# ---------------------------------------------------------------------------
DATABASE_URL  = os.environ["DATABASE_URL"]
EMBEDDER_NAME = os.environ.get("EMBEDDER_MODEL_NAME", "mixedbread-ai/mxbai-embed-large-v1")
CHUNK_SIZE    = int(os.environ.get("CHUNK_SIZE", 1000))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("IngestWorker")

# ---------------------------------------------------------------------------
# 2. Модели данных (Этап 1: Схема library)
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass

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

    chunk_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("library.chunks.id", ondelete="CASCADE"), primary_key=True)
    embedding: Mapped[list] = mapped_column(Vector(1536)) # Требование: 1536
    embedding_model: Mapped[str] = mapped_column(Text)

# ---------------------------------------------------------------------------
# 3. Сервисы
# ---------------------------------------------------------------------------
engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(EMBEDDER_NAME)
    logger.info(f"✅ Модель {EMBEDDER_NAME} загружена")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {e}")
    embedder = None

chunker = SemanticChunker(embedder=embedder, max_tokens=CHUNK_SIZE)

def _embed(texts: list) -> list:
    """Генерация эмбеддингов с доведением до 1536 размерности (Этап 1)."""
    if not embedder:
        return [[0.0]*1536 for _ in texts]

    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    result = []
    for v in vectors:
        v_list = v.tolist()
        # Если модель дает 1024, дополняем нулями до 1536 по требованию босса
        if len(v_list) < 1536:
            v_list += [0.0] * (1536 - len(v_list))
        result.append(v_list[:1536])
    return result

async def process_one_document() -> bool:
    # Открываем сессию без автоматического начала транзакции
    async with async_session() as session:
        # 1. Выбор задачи
        query = (
            select(Document, DocumentFile)
            .join(DocumentFile, Document.id == DocumentFile.document_id)
            .where(Document.ingest_status == "pending")
            .with_for_update(skip_locked=True)
            .limit(1)
        )

        res = await session.execute(query)
        result_row = res.unique().fetchone() # Используем unique() для надежности

        if not result_row:
            return False

        doc, df = result_row
        doc_id = doc.id  # Сохраняем ID, чтобы не обращаться к объекту в случае ошибки
        logger.info(f"🔄 Обработка: {doc.title}")

        try:
            # 2. Ставим статус "в процессе" и сразу сохраняем
            doc.ingest_status = "processing"
            await session.commit()

            # Переоткрываем сессию для основной работы, так как commit закрывает транзакцию
            await session.refresh(doc)

            # 3. Скачивание (aiohttp или local)
            content = await get_content(df.storage_path, df.storage_backend, df.bucket, df.storage_key)

            # 4. Парсинг
            ext = df.file_name.lower()
            if ext.endswith(".pdf"): text_data = parse_pdf(content)
            elif ext.endswith(".docx"): text_data = parse_docx(content)
            elif ext.endswith(".txt"): text_data = parse_txt(content)
            else: text_data = content.decode("utf-8", errors="replace")

            # 5. Чанкинг (наш асинхронный чанкер)
            chunks_list = await chunker.create_chunks(text_data, doc.title)

            # 6. Удаление старых чанков и запись новых
            await session.execute(delete(Chunk).where(Chunk.document_id == doc_id))

            if chunks_list:
                embeddings = _embed([c["text"] for c in chunks_list])
                logger.info(f"🔢 Создано чанков: {len(chunks_list)}")

                for i, (c_data, v) in enumerate(zip(chunks_list, embeddings)):
                    new_chunk = Chunk(
                        document_id=doc_id,
                        chunk_index=i,
                        text=c_data["text"],
                        section_path=c_data["meta"].get("section_path"),
                        start_offset=c_data["meta"].get("start_offset"),
                        end_offset=c_data["meta"].get("end_offset"),
                        token_count=c_data["meta"].get("token_count", 0),
                        metadata_json=c_data["meta"]
                    )
                    session.add(new_chunk)
                    await session.flush() # Получаем ID чанка асинхронно

                    emb_obj = ChunkEmbedding(
                        chunk_id=new_chunk.id,
                        embedding=v,
                        embedding_model=EMBEDDER_NAME
                    )
                    session.add(emb_obj)

            # 7. Финализация
            doc.ingest_status = "indexed"
            doc.indexed_at = datetime.now()
            doc.ingest_error = None
            await session.commit()
            logger.info(f"✅ Готово: {doc.title}")

        except Exception as e:
            await session.rollback()
            err_msg = str(e)
            logger.error(f"❌ Ошибка на документе {doc_id}: {err_msg}")

            # Пишем ошибку в базу в отдельном блоке
            async with async_session() as error_session:
                await error_session.execute(
                    text("UPDATE library.documents SET ingest_status = 'failed', ingest_error = :err WHERE id = :id"),
                    {"err": err_msg, "id": doc_id}
                )
                await error_session.commit()

        return True


# 5. Цикл запуска
# ---------------------------------------------------------------------------
async def main():
    logger.info("🚀 IngestWorker запущен и ожидает задач...")
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
        logger.info("🛑 Остановка воркера")
