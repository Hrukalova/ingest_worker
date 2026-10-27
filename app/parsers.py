"""
parsers.py — парсеры документов для ingest_worker

Здесь только загрузка файла (local, http, S3/MinIO) с проверкой размера.
Разбор содержимого на блоки — в notebook_parsing.py.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import aiohttp
import aioboto3
from aiobotocore.config import AioConfig

import s3_tls
from errors import FileTooLargeError

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
# (парсер держит в памяти ещё и собственное представление), поэтому без предела один
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


