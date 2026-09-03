# ingest_worker

Микросервис обработки и индексации документов для RAG-системы. Он забирает документы со статусом `pending` из PostgreSQL, загружает исходный файл, извлекает текст, автоматически выбирает стратегию чанкинга, генерирует embeddings и сохраняет чанки и векторы в PostgreSQL/pgvector.

**Важно:** сервис является background worker, а не HTTP API. Отдельных REST endpoints в текущей реализации нет; интеграция с другими сервисами происходит через PostgreSQL.

## Быстрый запуск

### Локально

1. Установить зависимости:

```bash
pip install -r requirements.txt
```

2. Создать `.env` на основе `.env.example` и указать минимум `DATABASE_URL`.

3. Запустить worker:

```bash
python app/main.py
```

Worker будет постоянно проверять очередь документов. Если задач нет, он ждёт 5 секунд и проверяет снова.

### Docker

```bash
docker compose up --build
```

Текущий `docker-compose.yml` поднимает PostgreSQL с поддержкой pgvector. Сам worker и MinIO/S3 в compose-файле не описаны и должны предоставляться отдельно.

---


Микросервис обработки документов для RAG-системы.
Читает задачи из PostgreSQL, парсит файлы, нарезает на чанки, сохраняет векторные эмбеддинги.

## Архитектура

```
PostgreSQL (library.documents)
        │  pending
        ▼
  IngestWorker (main.py)
        │
        ├── parse_pdf   → SmartChunker (M5 или M4)
        ├── parse_docx  → SmartChunker (M5 или M2)
        └── parse_txt   → SmartChunker (M2)
        │
        ▼
  library.chunks + library.chunk_embeddings
```

## Стратегия чанкинга (SmartChunker)

Выбор метода производится автоматически на основе типа и структуры документа.
Итоги внутреннего бенчмарка (метрика: **cohesion = intra_sim − inter_sim**):

| Формат | Условие | Метод | Cohesion |
|--------|---------|-------|----------|
| PDF | есть оглавление (TOC) | **M5 Structure-aware** | лучший |
| PDF | нет оглавления | **M4 Recursive** | +0.007 |
| DOCX | есть Heading 1/2/3 | **M5 Structure-aware** | +0.035 |
| DOCX | нет заголовков | **M2 Sentence** | — |
| TXT | всегда | **M2 Sentence** | +0.047 |

> HTML-страницы конвертируются в TXT во внешнем сервисе и поступают как `.txt`.

### Методы чанкинга

- **M2 Sentence** — разбивка по предложениям, группировка до `CHUNK_SIZE` токенов. Лучший cohesion на TXT.
- **M4 Recursive** — рекурсивная нарезка по иерархии `\n\n → \n → ". " → " "` с перекрытием. Лучший для PDF без структуры.
- **M5 Structure-aware** — нарезка по секциям из TOC (PDF) или Heading-стилей (DOCX). Секции > `CHUNK_SIZE` дробятся через M4.

## Структура проекта

```
app/
  main.py      — pipeline, модели БД, главный цикл
  chunker.py   — SmartChunker (M2 / M4 / M5)
  parsers.py   — parse_pdf, parse_docx, parse_txt
requirements.txt
Dockerfile
docker-compose.yml
001_init_modified.sql
```

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `DATABASE_URL` | — | PostgreSQL connection string (asyncpg) |
| `EMBEDDER_MODEL_NAME` | `mixedbread-ai/mxbai-embed-large-v1` | Sentence-transformers модель |
| `CHUNK_SIZE` | `512` | Целевой размер чанка (токены) |
| `CHUNK_MIN_TOKENS` | `50` | Минимум токенов (мелкие склеиваются) |
| `CHUNK_OVERLAP_TOKENS` | `32` | Перекрытие при M4 Recursive |
| `S3_ENDPOINT` | `http://localhost:9000` | MinIO / S3 endpoint |
| `S3_ACCESS_KEY` | `minioadmin` | — |
| `S3_SECRET_KEY` | `minioadmin` | — |

## Запуск

```bash
# Локально
pip install -r requirements.txt
python app/main.py

# Docker
docker compose up --build
```

## Формат данных в БД

`library.chunks`:
- `text` — текст чанка с header injection (`Документ: ... / Категория: ... / ---`)
- `section_path` — заголовок раздела (из TOC или Heading)
- `start_offset` / `end_offset` — байтовые позиции в исходном тексте
- `token_count` — оценочное число токенов
- `metadata_json` — полный набор метаданных: `doc_type`, `chunk_method`, `chunk_index`, и др.

`library.chunk_embeddings`:
- `embedding` — вектор 1536 измерений (дополняется нулями если модель < 1536)
- `embedding_model` — имя использованной модели


## Технические проблемы и ограничения

Ниже перечислены проблемы, которые следует учитывать перед production-эксплуатацией.

### Высокий приоритет

1. **Нет healthcheck/readiness endpoint.**
   Worker нельзя стандартным HTTP-механизмом проверить как живой/готовый сервис.

2. **Нет retry/backoff для отдельных документов.**
   При ошибке документ переводится в `failed`. Автоматического повторного запуска с backoff в текущем коде нет.

3. **Опасное поведение при ошибке загрузки embedding-модели.**
   Если `SentenceTransformer` не загрузился, код создаёт нулевые векторы размерности 1536 вместо остановки обработки. В результате документ потенциально может получить статус `indexed`, хотя качественного embedding нет.

4. **Нет ограничения максимального размера файла.**
   Файл полностью загружается в память перед parsing.

### Средний приоритет

5. **HTTP-загрузка имеет фиксированный timeout 15 секунд.**
   Для больших документов или медленного хранилища этого может быть недостаточно.

6. **Polling вместо очереди сообщений.**
   Worker постоянно опрашивает PostgreSQL с интервалом 5 секунд. Это просто, но создаёт задержку до следующего polling и дополнительную нагрузку на БД.

7. **HTML зависит от внешнего сервиса.**
   Worker сам HTML не парсит: HTML должен быть заранее преобразован во внешний TXT.

8. **Ограниченная обработка содержимого документов.**
   Pipeline ориентирован на текст. Отдельного извлечения изображений, таблиц и другого мультимедийного содержимого нет.

9. **Нет отдельного механизма ручного запуска/reindex.**
   Для повторной обработки нужно управлять статусом документа через БД или реализовать дополнительный orchestration-механизм.

### Рекомендации

Перед production рекомендуется как минимум:
- добавить health/readiness и метрики;
- сделать retry с ограничением числа попыток и backoff;
- не допускать `indexed` при невалидном embedding;
- добавить лимиты размера/времени обработки файла;
- предусмотреть безопасный механизм reindex;
- определить единый production-механизм доставки задач (polling либо message broker).
