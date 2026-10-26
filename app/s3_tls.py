"""Клиентский сертификат для обращений к MinIO (взаимная TLS-аутентификация).

MinIO стоит за прокси с проверкой клиентского сертификата (main/infra/mtls/minio.conf): права
сервиса определяет CN его сертификата, а адрес вида https://minio:8443. Сертификат берётся из
тех же переменных, что и для вызовов других сервисов: SERVICE_MTLS_CERT, SERVICE_MTLS_KEY,
SERVICE_MTLS_CA. Без них клиенты работают как раньше, по открытому HTTP.

Сертификат живёт сутки и обновляется рядом лежащим контейнером. Клиенты boto3 и aioboto3 здесь
создаются на каждую операцию и читают файлы заново; клиент minio-py дольше живёт и пересоздаётся
по времени изменения файла сертификата (library-svc).

Модуль одинаков в library-svc, db-svc и ingest-worker (разные репозитории, общего кода нет):
менять нужно во всех трёх, ищите s3_tls.py.
"""

from __future__ import annotations

import os


def enabled() -> bool:
    return bool(os.getenv("SERVICE_MTLS_CERT"))


def verify() -> str | None:
    """Корень CA для проверки сервера (параметр verify клиента boto3/aioboto3) или None."""
    return (os.getenv("SERVICE_MTLS_CA") or None) if enabled() else None


def client_cert() -> tuple[str, str] | None:
    """Пара (сертификат, ключ) для botocore.config.Config(client_cert=...) или None."""
    if not enabled():
        return None
    return os.environ["SERVICE_MTLS_CERT"], os.environ["SERVICE_MTLS_KEY"]


def cert_stamp() -> int:
    """Меняется, когда сертификат обновили: по нему пересоздаются долгоживущие клиенты."""
    cert = os.getenv("SERVICE_MTLS_CERT")
    try:
        return os.stat(cert).st_mtime_ns if cert else 0
    except OSError:
        return 0


def urllib3_pool():
    """PoolManager с клиентским сертификатом для minio-py или None без mTLS."""
    if not enabled():
        return None
    import urllib3

    return urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=5.0, read=300.0),
        maxsize=10,
        cert_reqs="CERT_REQUIRED",
        ca_certs=os.getenv("SERVICE_MTLS_CA") or None,
        cert_file=os.environ["SERVICE_MTLS_CERT"],
        key_file=os.environ["SERVICE_MTLS_KEY"],
        retries=urllib3.Retry(total=5, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504]),
    )
