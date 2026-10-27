"""Метрики Prometheus ingest-worker.

У воркера нет HTTP-сервера приложения, поэтому метрики отдаёт отдельный маленький сервер на
METRICS_PORT (start_metrics_server). Порт открыт только внутри сети compose и отдаёт одни
метрики.
"""

import os

from prometheus_client import Counter, Gauge, Histogram, start_http_server

METRICS_PORT = int(os.environ.get("METRICS_PORT", "9464"))

QUEUE_DEPTH = Gauge(
    "pochemuchnik_ingest_queue_depth",
    "Документы в очереди индексации по статусу",
    ["status"],  # pending | processing
)
JOBS = Counter(
    "pochemuchnik_ingest_jobs_total",
    "Завершённые попытки обработки документов",
    ["result"],  # indexed | retry | failed
)
JOB_DURATION = Histogram(
    "pochemuchnik_ingest_job_duration_seconds",
    "Длительность одной попытки обработки документа",
    ["result"],  # indexed | error
    buckets=(1, 2, 5, 10, 20, 40, 80, 160, 320, 640),
)
CHUNKS_CREATED = Counter(
    "pochemuchnik_ingest_chunks_created_total",
    "Чанки, записанные в library.chunks",
)
JOB_ERRORS = Counter(
    "pochemuchnik_ingest_job_errors_total",
    "Ошибки обработки по типу",
    ["type"],  # FileTooLargeError | DocumentParseError | PermanentIngestError | transient
)


def start_metrics_server() -> None:
    start_http_server(METRICS_PORT)
