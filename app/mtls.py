"""Клиентский сертификат для вызовов других сервисов (взаимная TLS-аутентификация).

Сервисы graph-rag-svc и embedding-svc принимают соединения только через прокси рядом с собой
(main/infra/mtls), который требует клиентский сертификат нашего внутреннего CA и по его CN
решает, что вызывающему можно. Здесь вызывающая сторона собирает контекст TLS: свой сертификат
(идентичность сервиса) и корень CA для проверки сервера.

Если SERVICE_MTLS_CERT не задан, возвращается None, и клиенты работают как раньше, по
открытому HTTP без проверки сертификата; получатель в этом случае должен работать в режиме
INTERNAL_AUTH_MODE=insecure, иначе он такие вызовы не примет.

Контекст собирается заново при КАЖДОМ создании клиента: сертификат живёт сутки и обновляется
рядом лежащим контейнером, а прочитанный один раз остался бы в памяти истёкшим. Клиенты в этих
сервисах создаются на каждый вызов, так что обновлённый сертификат подхватывается сам.

Модуль одинаков в chat-svc, db-svc, graph-rag-svc и ingest-worker (разные репозитории, общего
кода нет): менять нужно во всех сразу, ищите mtls.py.

Переменные:
  SERVICE_MTLS_CERT  сертификат этого сервиса (цепочка: лист и промежуточный)
  SERVICE_MTLS_KEY   закрытый ключ
  SERVICE_MTLS_CA    корневой сертификат CA, которым подписан сертификат сервера
"""

from __future__ import annotations

import os
import ssl


def ssl_context() -> ssl.SSLContext | None:
    """Свежий контекст с клиентским сертификатом или None, если mTLS не включён."""
    cert = os.getenv("SERVICE_MTLS_CERT")
    if not cert:
        return None
    ctx = ssl.create_default_context(cafile=os.getenv("SERVICE_MTLS_CA") or None)
    ctx.load_cert_chain(certfile=cert, keyfile=os.getenv("SERVICE_MTLS_KEY") or None)
    return ctx


def httpx_kwargs() -> dict:
    """Аргументы для httpx.Client / httpx.AsyncClient: {} без mTLS, иначе verify=<контекст>."""
    ctx = ssl_context()
    return {"verify": ctx} if ctx is not None else {}
