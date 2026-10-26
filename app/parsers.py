"""
parsers.py — парсеры документов для ingest_worker

Поддерживаемые форматы: PDF, DOCX, TXT.
HTML-страницы конвертируются во внешнем сервисе и поступают как TXT.

Каждый парсер возвращает кортеж:
    (text: str, sections: List[Dict])

sections — список структурных секций вида:
    {"title": str, "level": int, "text": str}

Для PDF секции извлекаются из TOC (оглавления) через PyMuPDF.
Для DOCX секции извлекаются из Heading 1/2/3 стилей.
Для TXT секции пустые ([]) — текст нарезается по предложениям (M2).
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import List, Dict, Tuple, Optional

import aiohttp
import aioboto3
from aiobotocore.config import AioConfig
import fitz          # PyMuPDF
from docx import Document as DocxReader

import s3_tls
from errors import DocumentParseError, FileTooLargeError

# ---------------------------------------------------------------------------
# Параметры S3 / MinIO
# ---------------------------------------------------------------------------

# Имена переменных здесь S3_*, тогда как остальные сервисы читают MINIO_*. Значения в
# compose отображаются одни в другие, но при запуске воркера вручную об этом легко забыть.
# Умолчание localhost:9000 внутри контейнера указывает на него самого, и ошибка выглядит
# как недоступный MinIO, а не как незаданная настройка, поэтому адрес пишем в лог при старте.
S3_ENDPOINT  = os.getenv("S3_ENDPOINT",   "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")

logging.getLogger("IngestWorker").info("Хранилище документов: %s", S3_ENDPOINT)

# Предел размера исходного файла. Файл целиком читается в память и затем разбирается
# (PyMuPDF держит в памяти ещё и собственное представление), поэтому без предела один
# огромный файл способен уронить воркер по OOM — и каждый перезапуск снова брал бы его же.
# Проверка идёт до загрузки, где размер известен заранее (S3, локальный файл, Content-Length),
# и во время чтения — для HTTP-ответов без Content-Length или с неверным.
MAX_FILE_SIZE_MB = float(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = int(MAX_FILE_SIZE_MB * 1024 * 1024)


def _check_size(size: Optional[int], source: str) -> None:
    if size is not None and size > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"Файл {source} весит {size / 1024 / 1024:.1f} МБ, "
            f"предел — {MAX_FILE_SIZE_MB:g} МБ (MAX_FILE_SIZE_MB)"
        )


# ---------------------------------------------------------------------------
# Загрузка контента
# ---------------------------------------------------------------------------

async def get_content_from_s3(bucket: str, key: str) -> bytes:
    """Загрузка файла из MinIO / S3."""
    session = aioboto3.Session()
    async with session.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        verify=s3_tls.verify(),
        config=AioConfig(client_cert=s3_tls.client_cert()),
    ) as s3:
        try:
            response = await s3.get_object(Bucket=bucket, Key=key)
            _check_size(response.get("ContentLength"), f"s3://{bucket}/{key}")
            async with response["Body"] as stream:
                return await stream.read()
        except FileTooLargeError:
            raise
        except Exception as e:
            raise Exception(f"S3 download failed (bucket={bucket}, key={key}): {e}")


async def get_content(
    path_or_url: str,
    backend: str = "local",
    bucket: Optional[str] = None,
    key: Optional[str] = None,
) -> bytes:
    """Универсальный загрузчик: local, http или s3/minio."""
    if backend in ("minio", "s3"):
        if not bucket or not key:
            raise ValueError(f"backend='{backend}' требует bucket и key")
        return await get_content_from_s3(bucket, key)

    if path_or_url.startswith("http"):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                path_or_url,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}: {path_or_url}")
                _check_size(resp.content_length, path_or_url)
                data = bytearray()
                async for piece in resp.content.iter_chunked(1024 * 1024):
                    data.extend(piece)
                    _check_size(len(data), path_or_url)
                return bytes(data)

    clean = path_or_url.strip()
    if os.path.exists(clean):
        _check_size(os.path.getsize(clean), clean)
        with open(clean, "rb") as f:
            return f.read()
    raise FileNotFoundError(f"Файл не найден: {clean}")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def parse_pdf(file_bytes: bytes) -> Tuple[str, List[Dict]]:
    """
    Извлекает текст и структуру из PDF.

    Структура (sections) берётся из TOC (get_toc()).
    Если TOC пустой — sections = [], чанкер применит M4_Recursive.

    Возвращает: (full_text, sections)
    """
    try:
        sections: List[Dict] = []
        full_parts: List[str] = []

        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            toc = doc.get_toc()   # [[level, title, page], ...]

            if toc:
                # Строим границы секций по TOC
                pages = [e[2] for e in toc] + [doc.page_count + 1]
                for i, (level, title, start_page) in enumerate(toc):
                    end_page = pages[i + 1] if i + 1 < len(pages) else doc.page_count + 1
                    text_parts = []
                    for p in range(start_page - 1, min(end_page - 1, doc.page_count)):
                        t = doc[p].get_text("text").strip()
                        if t:
                            text_parts.append(t)
                    sec_text = "\n".join(text_parts).strip()
                    if sec_text:
                        sections.append({"title": title, "level": level, "text": sec_text})
                        full_parts.append(sec_text)
            else:
                # Нет TOC — читаем страницы подряд
                for page in doc:
                    t = page.get_text("text").strip()
                    if t:
                        full_parts.append(t)

        full_text = "\n\n".join(full_parts)
        return full_text, sections

    except Exception as e:
        raise DocumentParseError(f"PDF parse failed: {e}") from e


# ---------------------------------------------------------------------------
# DOCX
# ---------------------------------------------------------------------------

_HEADING_STYLES = {
    "heading 1": 1, "heading 2": 2, "heading 3": 3,
    "заголовок 1": 1, "заголовок 2": 2, "заголовок 3": 3,
}


def parse_docx(file_bytes: bytes) -> Tuple[str, List[Dict]]:
    """
    Извлекает текст и структуру из DOCX.

    Структура (sections) строится по Heading 1/2/3 стилям.
    Если заголовков нет — sections = [], чанкер применит M2_Sentence.

    Возвращает: (full_text, sections)
    """
    try:
        doc = DocxReader(io.BytesIO(file_bytes))
        sections: List[Dict] = []
        full_lines: List[str] = []
        current_title: Optional[str] = None
        current_level: int = 0
        current_lines: List[str] = []

        for para in doc.paragraphs:
            style_name = para.style.name.lower() if para.style else ""
            text = para.text.strip()
            if not text:
                continue

            level = _HEADING_STYLES.get(style_name, 0)
            if level > 0:
                # Сохраняем накопленную секцию
                if current_title and current_lines:
                    sections.append({
                        "title": current_title,
                        "level": current_level,
                        "text": "\n".join(current_lines),
                    })
                current_title = text
                current_level = level
                current_lines = []
            else:
                current_lines.append(text)

            full_lines.append(text)

        # Последняя секция
        if current_title and current_lines:
            sections.append({
                "title": current_title,
                "level": current_level,
                "text": "\n".join(current_lines),
            })

        # Если заголовков не было — одна секция из всего текста
        if not sections and full_lines:
            sections = []  # пустой → M2_Sentence fallback

        full_text = "\n\n".join(full_lines)
        return full_text, sections

    except Exception as e:
        raise DocumentParseError(f"DOCX parse failed: {e}") from e


# ---------------------------------------------------------------------------
# TXT
# ---------------------------------------------------------------------------

def parse_txt(file_bytes: bytes) -> Tuple[str, List[Dict]]:
    """
    Декодирует TXT-файл.
    Возвращает: (text, [])  — секций нет, чанкер применит M2_Sentence.
    """
    try:
        text = file_bytes.decode("utf-8", errors="replace")
        return text, []
    except Exception as e:
        raise DocumentParseError(f"TXT parse failed: {e}") from e
