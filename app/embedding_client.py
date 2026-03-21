import os
import requests

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-svc:3500/embed")
SERVICE_NAME = os.getenv("SERVICE_NAME", "ingest-worker")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN", "")
INTERNAL_AUTH_HEADER_NAME = os.getenv("INTERNAL_AUTH_HEADER_NAME", "X-Service-Token")
INTERNAL_SERVICE_NAME_HEADER = os.getenv("INTERNAL_SERVICE_NAME_HEADER", "X-Service-Name")


def build_internal_auth_headers() -> dict[str, str]:
    return {
        INTERNAL_SERVICE_NAME_HEADER: SERVICE_NAME,
        INTERNAL_AUTH_HEADER_NAME: SERVICE_TOKEN,
    }


def embed_texts(texts: list[str], normalize: bool = True) -> list[list[float]]:
    response = requests.post(
        EMBEDDING_SERVICE_URL,
        json={"texts": texts, "normalize": normalize},
        headers=build_internal_auth_headers(),
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data["embeddings"]