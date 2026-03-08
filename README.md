# Ingest Worker

Микросервис-индексатор для RAG пайплайна (HSE Auto-Mentor).
Этот сервис работает в фоновом режиме:
1. Выбирает документы со статусом "pending" из базы данных PostgreSQL.
2. Парсит их содержимое (HTML, PDF, DOCX).
3. Применяет семантическое разбиение на чанки (`SemanticChunker`).
4. Вклеивает метаданные и "хлебные крошки" (Header Injection).
5. Создаёт векторные эмбеддинги (`mixedbread-ai/mxbai-embed-large-v1` по умолчанию).
6. Сохраняет готовые вектора в `pgvector` (таблица `chunks`).

## Использование (Stand-alone)
Этот проект не имеет зависимостей от других сервисов. Все модели базы данных включены в `main.py`.

### 1. Настройка окружения
Скопируйте `.env.example` в `.env` и задайте параметры подключения.

```env
DATABASE_URL=postgresql+asyncpg://admin:password@localhost:5432/hse_rag
EMBEDDER_MODEL_NAME=mixedbread-ai/mxbai-embed-large-v1
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

### 2. Установка зависимостей
```powershell
pip install -r requirements.txt
```

### 3. Запуск
```powershell
python main.py
```
