# ingest_worker/main.py
"""
Ingest Worker — отдельный микросервис
======================================
Подключается к PostgreSQL + pgvector, берёт документы из очереди (status='pending'),
парсит HTML/PDF/DOCX, нарезает на семантические чанки, генерирует эмбеддинги
и сохраняет в таблицу chunks.

Запуск:
    python main.py

Переменные окружения (.env):
    DATABASE_URL=postgresql+asyncpg://admin:password@localhost:5432/hse_rag
    EMBEDDER_MODEL_NAME=mixedbread-ai/mxbai-embed-large-v1
    CHUNK_SIZE=1000
    CHUNK_OVERLAP=100
    # Необязательно:
    GOOGLE_DRIVE_FOLDER_ID=...
    GOOGLE_APPLICATION_CREDENTIALS=google_creds.json
"""

import asyncio
import logging
import os

from sqlalchemy import select, delete, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, DeclarativeBase, mapped_column, Mapped
from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
import uuid

from dotenv import load_dotenv
load_dotenv()

# Локальные модули воркера
from chunker import SemanticChunker
from parsers import parse_url, get_content, parse_pdf, parse_docx
from topic_helper import TopicHelper

# ---------------------------------------------------------------------------
# Конфиг из .env
# ---------------------------------------------------------------------------
DATABASE_URL      = os.environ["DATABASE_URL"]
EMBEDDER_NAME     = os.environ.get("EMBEDDER_MODEL_NAME", "mixedbread-ai/mxbai-embed-large-v1")
CHUNK_SIZE        = int(os.environ.get("CHUNK_SIZE", 1000))

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("IngestWorker")

# ---------------------------------------------------------------------------
# БД
# ---------------------------------------------------------------------------
engine = create_async_engine(DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title          = Column(String(512), nullable=False)
    source_type    = Column(String(50))
    raw_content_link = Column(String(1024))
    content_text   = Column(Text)
    status         = Column(String(20), default="pending")
    doc_metadata   = Column(JSONB, default={})


class Chunk(Base):
    __tablename__ = "chunks"
    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id   = Column(UUID(as_uuid=True))
    text          = Column(Text, nullable=False)
    embedding     = Column(Vector(1024))
    chunk_metadata = Column(JSONB, default={})


# ---------------------------------------------------------------------------
# Эмбеддинг-модель
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    logger.info(f"⏳ Загружаю модель: {EMBEDDER_NAME}")
    embedder = SentenceTransformer(EMBEDDER_NAME)
    logger.info("✅ Модель загружена")
except Exception as e:
    logger.error(f"❌ Ошибка загрузки модели: {e}")
    embedder = None

chunker      = SemanticChunker(embedder=embedder, max_tokens=CHUNK_SIZE)
topic_helper = TopicHelper()

METADATA_XLSX = os.environ.get("METADATA_XLSX", "")


def _load_topics():
    if not topic_helper.is_loaded() and METADATA_XLSX and os.path.exists(METADATA_XLSX):
        topic_helper.load_from_excel(METADATA_XLSX, sheet_name="topics")


def _embed(texts: list) -> list:
    if embedder is None:
        return [[0.0] * 1024 for _ in texts]
    return [v.tolist() for v in embedder.encode(texts, batch_size=32, normalize_embeddings=True)]


async def _delete_old_chunks(session, doc_id):
    await session.execute(delete(Chunk).where(Chunk.document_id == doc_id))


async def process_one_document() -> bool:
    _load_topics()
    async with async_session() as session:
        res = await session.execute(
            select(Document).where(Document.status == "pending")
            .with_for_update(skip_locked=True).limit(1)
        )
        doc = res.scalar_one_or_none()
        if not doc:
            return False

        logger.info(f"🔄 [{doc.id}] {doc.title}")
        doc.status = "processing"
        await session.commit()

        try:
            link = doc.raw_content_link or ""
            content_type = (doc.doc_metadata or {}).get("content_type", "html")
            if link.startswith("http"):
                if link.endswith(".pdf") or content_type == "pdf":
                    text_content = parse_pdf(await get_content(link))
                elif link.endswith(".docx") or content_type == "docx":
                    text_content = parse_docx(await get_content(link))
                else:
                    text_content = await parse_url(link)
            elif link:
                raw = await get_content(link)
                text_content = parse_pdf(raw) if link.endswith(".pdf") else \
                               parse_docx(raw) if link.endswith(".docx") else raw.decode("utf-8", errors="replace")
            else:
                text_content = doc.content_text or ""

            doc.content_text = text_content
            topic_id   = (doc.doc_metadata or {}).get("topic_id")
            breadcrumb = topic_helper.get_breadcrumb(topic_id)
            chunks     = await chunker.create_chunks(text_content, doc.title, breadcrumb)

            if chunks:
                embeddings = _embed([c["text"] for c in chunks])
                await _delete_old_chunks(session, doc.id)
                for c, emb in zip(chunks, embeddings):
                    session.add(Chunk(document_id=doc.id, text=c["text"],
                                     embedding=emb, chunk_metadata=c["meta"]))

            doc.status = "indexed"
            logger.info(f"✅ {doc.title} — {len(chunks)} чанков")

        except Exception as exc:
            logger.error(f"❌ {doc.title}: {exc}", exc_info=True)
            doc.status = "failed"

        await session.commit()
        return True


async def main():
    logger.info("🚀 IngestWorker запущен")
    while True:
        try:
            processed = await process_one_document()
        except Exception as exc:
            logger.error(f"Критическая ошибка: {exc}", exc_info=True)
            processed = False
        if not processed:
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 Остановлен")
