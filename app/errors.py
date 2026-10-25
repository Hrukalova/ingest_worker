"""Классы ошибок обработки документа.

Воркер повторяет неудачную обработку с растущей паузой (см. INGEST_MAX_ATTEMPTS в main.py),
но только для ошибок, которые могут пройти сами: недоступен embedding-svc, MinIO, БД.
Ошибки, которые повтор не исправит, наследуются от PermanentIngestError — документ с такой
ошибкой сразу получает статус failed, не тратя попытки и время.
"""


class PermanentIngestError(Exception):
    """Ошибка, которую повтор не исправит: документ сразу уходит в failed."""


class FileTooLargeError(PermanentIngestError):
    """Файл больше MAX_FILE_SIZE_MB."""


class DocumentParseError(PermanentIngestError):
    """Файл не разбирается парсером своего формата (битый PDF/DOCX, неверная кодировка)."""
