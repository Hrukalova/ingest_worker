import io
import os
import aiohttp
import fitz  # PyMuPDF
import aioboto3
from docx import Document as DocxReader
from readability import Document as ReadabilityDoc
from bs4 import BeautifulSoup

# Параметры подключения к MinIO (подгружаются из окружения)
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:9000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "minioadmin")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "minioadmin")

async def get_content_from_s3(bucket: str, key: str) -> bytes:
    """
    Загрузка файла из MinIO/S3 (Требование 8).
    """
    session = aioboto3.Session()
    async with session.client(
        's3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY
    ) as s3:
        try:
            response = await s3.get_object(Bucket=bucket, Key=key)
            async with response['Body'] as stream:
                return await stream.read()
        except Exception as e:
            raise Exception(f"S3 download failed (bucket: {bucket}, key: {key}): {str(e)}")

async def get_content(path_or_url: str, backend: str = "local", bucket: str = None, key: str = None) -> bytes:
    """
    Универсальный загрузчик: local, http или s3 (Требование 8).
    """
    # 1. Если бэкенд - MinIO или S3
    if backend in ["minio", "s3"]:
        if not bucket or not key:
            raise ValueError(f"For backend '{backend}' bucket and key are required")
        return await get_content_from_s3(bucket, key)

    # 2. Если путь - это URL
    if path_or_url.startswith("http"):
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(path_or_url, timeout=15) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    raise Exception(f"HTTP Error: status {resp.status}")
            except Exception as e:
                raise Exception(f"URL download failed: {str(e)}")

    # 3. Если путь локальный
    else:
        clean_path = path_or_url.strip()
        if os.path.exists(clean_path):
            try:
                with open(clean_path, "rb") as f:
                    return f.read()
            except Exception as e:
                raise Exception(f"Local file read failed: {str(e)}")
        raise FileNotFoundError(f"File not found on disk: {clean_path}")

# --- Функции парсинга (Требование 9) ---

def parse_pdf(file_bytes: bytes) -> str:
    """Извлечение текста из PDF."""
    try:
        text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        return text
    except Exception as e:
        raise Exception(f"Failed to parse PDF: {str(e)}")

def parse_docx(file_bytes: bytes) -> str:
    """Извлечение текста из DOCX."""
    try:
        doc = DocxReader(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        raise Exception(f"Failed to parse DOCX: {str(e)}")

def parse_txt(file_bytes: bytes) -> str:
    """Чтение обычного текста (Требование 9)."""
    try:
        return file_bytes.decode("utf-8", errors="replace")
    except Exception as e:
        raise Exception(f"Failed to decode TXT: {str(e)}")

async def parse_url(url: str) -> str:
    """Парсинг HTML через Readability для очистки контента."""
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=10) as resp:
                html = await resp.text()
                doc = ReadabilityDoc(html)
                soup = BeautifulSoup(doc.summary(), "lxml")
                return f"{doc.title()}\n{soup.get_text(separator=' ', strip=True)}"
        except Exception as e:
            raise Exception(f"Failed to parse URL/HTML: {str(e)}")
