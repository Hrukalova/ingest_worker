import os
import sys

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from sqlalchemy.dialects import postgresql  # noqa: E402

import main  # noqa: E402


def test_claim_query_locks_only_documents():
    """Роль ingest_user имеет на document_files только SELECT: без OF блокировка строк
    охватила бы обе таблицы и дала бы permission denied."""
    sql = str(main._claim_query().compile(dialect=postgresql.dialect()))
    assert "FOR UPDATE OF documents SKIP LOCKED" in sql
    assert "FOR UPDATE SKIP LOCKED" not in sql
