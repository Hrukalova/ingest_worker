# services/ingest_worker/parsers.py
import aiohttp
import fitz  # PyMuPDF
from docx import Document as DocxReader
from readability import Document as ReadabilityDoc
from bs4 import BeautifulSoup
import io
import os

async def get_content(path_or_url: str) -> bytes:
    """Универсальный загрузчик: качает по HTTP или читает с диска."""
    if path_or_url.startswith("http"):
        async with aiohttp.ClientSession() as session:
            async with session.get(path_or_url, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.read()
                raise Exception(f"Ошибка загрузки: статус {resp.status}")
    else:
        # Убираем возможные лишние пробелы из пути
        clean_path = path_or_url.strip()
        if os.path.exists(clean_path):
            with open(clean_path, "rb") as f:
                return f.read()
        raise FileNotFoundError(f"Файл не найден по пути: {clean_path}")

async def parse_url(url: str) -> str:
    """Парсинг HTML через Readability для очистки от мусора."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=10) as resp:
            html = await resp.text()
            doc = ReadabilityDoc(html)
            clean_html = doc.summary()
            soup = BeautifulSoup(clean_html, "lxml")
            return f"{doc.title()}\n{soup.get_text(separator=' ', strip=True)}"

def parse_pdf(file_bytes: bytes) -> str:
    """Парсинг PDF через PyMuPDF."""
    text = ""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def parse_docx(file_bytes: bytes) -> str:
    """Парсинг DOCX."""
    doc = DocxReader(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs])
