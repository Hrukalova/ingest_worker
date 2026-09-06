FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Это background worker, а не HTTP-сервис: порт не слушается, healthcheck'а нет.
# Запуск именно как `python app/main.py`, а не `-m app.main`: модули внутри app/
# импортируют друг друга плоско (`from parsers import ...`), и такой запуск кладёт
# каталог app/ в sys.path.
CMD ["python", "app/main.py"]
