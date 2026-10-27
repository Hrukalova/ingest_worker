import asyncio
import os
import sys
import uuid
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import pytest  # noqa: E402
from prometheus_client import REGISTRY  # noqa: E402

import main  # noqa: E402
from errors import DocumentParseError  # noqa: E402


def sample(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


class FakeSession:
    """Сессия без БД: первый execute отдаёт выбранный документ, остальные ничего не возвращают."""

    def __init__(self, row=None, rows=None):
        self.row = row
        self.rows = rows or []
        self.executed = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def execute(self, _query, _params=None):
        self.executed += 1
        session = self
        if self.executed == 1 and self.row is not None:
            return SimpleNamespace(unique=lambda: SimpleNamespace(fetchone=lambda: session.row))
        return SimpleNamespace(all=lambda: session.rows)

    def add(self, _obj):
        pass

    async def flush(self):
        pass

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def refresh(self, _obj):
        pass


def make_doc(attempts=0):
    return SimpleNamespace(
        id=uuid.uuid4(), title="ПУД", ingest_attempts=attempts, ingest_status="pending",
        ingest_error=None, ingest_next_attempt_at=None, ingest_started_at=None, indexed_at=None,
    )


def make_file():
    return SimpleNamespace(storage_path=None, storage_backend="minio", bucket="b", storage_key="k",
                           file_name="a.pdf", mime_type="application/pdf")


def patch_pipeline(monkeypatch, chunks, fail_with=None):
    async def get_content(*_args):
        if fail_with is not None:
            raise fail_with
        return b"%PDF"

    async def create_chunks(**_kwargs):
        return chunks

    async def embed(texts):
        return [[0.0] * 3 for _ in texts], "bge-m3"

    async def record_failure(*_args):
        pass

    monkeypatch.setattr(main, "get_content", get_content)
    monkeypatch.setitem(main._PARSERS, "pdf", lambda _content: ("text", []))
    monkeypatch.setattr(main.chunker, "create_chunks", create_chunks)
    monkeypatch.setattr(main, "_embed", embed)
    monkeypatch.setattr(main, "_record_failure", record_failure)


def run_one(monkeypatch, doc, session=None):
    session = session or FakeSession(row=(doc, make_file()))
    monkeypatch.setattr(main, "async_session", lambda: session)
    return asyncio.run(main.process_one_document())


def test_indexed_document_counts_job_chunks_and_duration(monkeypatch):
    chunks = [{"text": "a", "meta": {}}, {"text": "b", "meta": {}}, {"text": "c", "meta": {}}]
    patch_pipeline(monkeypatch, chunks)
    jobs = sample("pochemuchnik_ingest_jobs_total", result="indexed")
    created = sample("pochemuchnik_ingest_chunks_created_total")
    durations = sample("pochemuchnik_ingest_job_duration_seconds_count", result="indexed")

    assert run_one(monkeypatch, make_doc()) is True

    assert sample("pochemuchnik_ingest_jobs_total", result="indexed") == jobs + 1
    assert sample("pochemuchnik_ingest_chunks_created_total") == created + 3
    assert sample("pochemuchnik_ingest_job_duration_seconds_count", result="indexed") == durations + 1


def test_failed_attempt_observes_error_duration_and_does_not_count_chunks(monkeypatch):
    patch_pipeline(monkeypatch, [], fail_with=RuntimeError("minio down"))
    created = sample("pochemuchnik_ingest_chunks_created_total")
    errors = sample("pochemuchnik_ingest_job_duration_seconds_count", result="error")

    run_one(monkeypatch, make_doc())

    assert sample("pochemuchnik_ingest_job_duration_seconds_count", result="error") == errors + 1
    assert sample("pochemuchnik_ingest_chunks_created_total") == created


def test_exhausted_attempts_count_as_failed_job(monkeypatch):
    patch_pipeline(monkeypatch, [])
    before = sample("pochemuchnik_ingest_jobs_total", result="failed")
    run_one(monkeypatch, make_doc(attempts=main.INGEST_MAX_ATTEMPTS))
    assert sample("pochemuchnik_ingest_jobs_total", result="failed") == before + 1


def test_transient_error_is_retried_and_counted_as_transient(monkeypatch):
    jobs = sample("pochemuchnik_ingest_jobs_total", result="retry")
    errors = sample("pochemuchnik_ingest_job_errors_total", type="transient")
    monkeypatch.setattr(main, "async_session", lambda: FakeSession())

    asyncio.run(main._record_failure(uuid.uuid4(), 1, ConnectionError("embedding-svc down")))

    assert sample("pochemuchnik_ingest_jobs_total", result="retry") == jobs + 1
    assert sample("pochemuchnik_ingest_job_errors_total", type="transient") == errors + 1


def test_permanent_error_fails_immediately_and_is_counted_by_class(monkeypatch):
    jobs = sample("pochemuchnik_ingest_jobs_total", result="failed")
    errors = sample("pochemuchnik_ingest_job_errors_total", type="DocumentParseError")
    monkeypatch.setattr(main, "async_session", lambda: FakeSession())

    asyncio.run(main._record_failure(uuid.uuid4(), 1, DocumentParseError("битый PDF")))

    assert sample("pochemuchnik_ingest_jobs_total", result="failed") == jobs + 1
    assert sample("pochemuchnik_ingest_job_errors_total", type="DocumentParseError") == errors + 1


def test_queue_depth_reports_zero_for_missing_statuses(monkeypatch):
    monkeypatch.setattr(main, "async_session", lambda: FakeSession(rows=[("pending", 4)]))
    asyncio.run(main.refresh_queue_depth())
    assert sample("pochemuchnik_ingest_queue_depth", status="pending") == 4
    assert sample("pochemuchnik_ingest_queue_depth", status="processing") == 0
