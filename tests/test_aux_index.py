import asyncio
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import pytest  # noqa: E402
from prometheus_client import REGISTRY  # noqa: E402

import aux_index  # noqa: E402


def sample(**labels) -> float:
    return REGISTRY.get_sample_value("pochemuchnik_ingest_aux_embedded_total", labels) or 0.0


class FakeSession:
    """Сессия без БД: первый execute отдаёт выборку, остальные запоминаются как записи."""

    def __init__(self, rows):
        self.rows = rows
        self.selects = []
        self.updates = []  # (запрос, параметры)
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def execute(self, query, params=None):
        if not self.selects and not self.updates:
            self.selects.append((query, params))
            return SimpleNamespace(all=lambda: list(self.rows))
        self.updates.append((str(query), params))
        return SimpleNamespace()

    async def commit(self):
        self.commits += 1


@pytest.fixture
def wired(monkeypatch):
    """Подменяет _embed и фабрику сессий; возвращает управляющий объект."""
    state = SimpleNamespace(session=None, embed_calls=[], error=None)

    async def fake_embed(texts):
        state.embed_calls.append(list(texts))
        if state.error:
            raise state.error
        return [[float(i)] * 3 for i, _ in enumerate(texts)], "e5-test"

    monkeypatch.setattr(aux_index, "_embed", fake_embed)
    monkeypatch.setattr(aux_index, "async_session", lambda: state.session)
    return state


@pytest.mark.parametrize("func, kind", [
    (aux_index.embed_pending_quick_answers, "quick_answer_variant"),
    (aux_index.embed_pending_links, "link"),
])
def test_rows_without_embedding_get_vector_and_model(wired, func, kind):
    wired.session = FakeSession([("id-1", "первый"), ("id-2", "второй")])
    before = sample(kind=kind)

    assert asyncio.run(func()) == 2

    assert wired.embed_calls == [["первый", "второй"]]
    (_, params), = wired.session.updates
    assert [p["id"] for p in params] == ["id-1", "id-2"]
    assert {p["model"] for p in params} == {"e5-test"}
    assert params[1]["vec"] == "[1.0, 1.0, 1.0]"
    assert wired.session.commits == 1
    assert sample(kind=kind) == before + 2


def test_selection_only_takes_uncomputed_active_rows():
    # посчитанные строки не трогаются: это гарантирует условие выборки
    for sql in (aux_index.SELECT_VARIANTS, aux_index.SELECT_LINKS):
        text = str(sql)
        assert "embedding IS NULL" in text
        assert "is_active" in text
        assert "SKIP LOCKED" in text


@pytest.mark.parametrize("func", [aux_index.embed_pending_quick_answers, aux_index.embed_pending_links])
def test_embedding_failure_writes_nothing_and_reraises(wired, func):
    wired.session = FakeSession([("id-1", "текст")])
    wired.error = RuntimeError("embedding-svc недоступен")

    with pytest.raises(RuntimeError):
        asyncio.run(func())

    assert wired.session.updates == []
    assert wired.session.commits == 0


@pytest.mark.parametrize("func", [aux_index.embed_pending_quick_answers, aux_index.embed_pending_links])
def test_empty_selection_returns_zero_without_embed(wired, func):
    wired.session = FakeSession([])

    assert asyncio.run(func()) == 0

    assert wired.embed_calls == []
    assert wired.session.updates == []


def test_pending_aux_sums_both(monkeypatch):
    async def qa(limit=256):
        return 2

    async def links(limit=256):
        return 3

    monkeypatch.setattr(aux_index, "embed_pending_quick_answers", qa)
    monkeypatch.setattr(aux_index, "embed_pending_links", links)
    assert asyncio.run(aux_index.embed_pending_aux()) == 5


def test_unconfigured_raises(monkeypatch):
    monkeypatch.setattr(aux_index, "_embed", None)
    monkeypatch.setattr(aux_index, "async_session", None)
    with pytest.raises(RuntimeError):
        asyncio.run(aux_index.embed_pending_links())


def test_selection_excludes_blank_text():
    # пустые тексты не выбираются вовсе: не эмбеддятся и не перевыбираются на каждом проходе
    assert "btrim(coalesce(v.text, '')) <> ''" in str(aux_index.SELECT_VARIANTS)
    assert "btrim(coalesce(description, '')) <> ''" in str(aux_index.SELECT_LINKS)


@pytest.mark.parametrize("func", [aux_index.embed_pending_quick_answers, aux_index.embed_pending_links])
def test_blank_rows_are_never_embedded(wired, func):
    wired.session = FakeSession([("id-1", ""), ("id-2", "  \n"), ("id-3", None)])

    assert asyncio.run(func()) == 0

    assert wired.embed_calls == []
    assert wired.session.updates == []


def test_blank_rows_skipped_in_mixed_batch(wired):
    wired.session = FakeSession([("id-1", " "), ("id-2", "текст")])

    assert asyncio.run(aux_index.embed_pending_links()) == 1

    assert wired.embed_calls == [["текст"]]
    (_, params), = wired.session.updates
    assert [p["id"] for p in params] == ["id-2"]
