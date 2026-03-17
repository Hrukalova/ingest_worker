# Ingest Worker (HSE Auto-Mentor)

Микросервис-индексатор. Забирает документы из PostgreSQL, парсит их и сохраняет векторные представления в pgvector.

## Основные функции
- **Схема БД:** Работает в схеме `library` (PostgreSQL + pgvector).
- **Парсинг:** Поддержка PDF, DOCX, TXT и HTML (через Readability).
- **Чанкинг:** Семантическое разбиение на основе косинусного сходства предложений.
- **Эмбеддинги:** Генерация векторов размерностью 1536 (модель mxbai-embed-large-v1 с дополнением).
- **Надежность:** Атомарные транзакции, отслеживание офсетов (start/end) и логирование ошибок в БД.

## Запуск
1. Настройте `.env` (DATABASE_URL, S3_ENDPOINT).
2. Установите зависимости: `pip install -r requirements.txt`.
3. Запустите воркер: `python app/main.py`.


## Как тестировать (Quick Start)
Для проверки работы сервиса нужно вручную добавить задачу в БД через DBeaver:

1. **Добавить документ и файл:**
```sql
INSERT INTO library.topics (name, slug) VALUES ('Тест', 'test');
INSERT INTO library.documents (id, topic_id, title, category, source_type, content_type, ingest_status)
VALUES ('00000000-0000-0000-0000-000000000001', (SELECT id FROM library.topics LIMIT 1), 'Тестовый файл', 'FAQ', 'file', 'pdf', 'pending');
INSERT INTO library.document_files (document_id, storage_backend, storage_path, file_name)
VALUES ('00000000-0000-0000-0000-000000000001', 'local', 'C:/путь/к/файлу.pdf', 'test.pdf');
