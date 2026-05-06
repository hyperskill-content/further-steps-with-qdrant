import os

from qdrant_client import QdrantClient

from .config import QDRANT_HOST, QDRANT_PORT

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

# Exact k-NN over hundreds of thousands of vectors can exceed the default httpx
# 5s read timeout on a cold segment cache. 120s is a safe upper bound for this
# project's workload.
DEFAULT_TIMEOUT_S = 120


def build_client(timeout: int = DEFAULT_TIMEOUT_S) -> QdrantClient:
    # Localhost connections must not be tunneled through SOCKS/HTTP proxies; httpx
    # builds the proxy transport before consulting NO_PROXY, so we strip the env
    # vars during client construction. They're restored immediately after.
    saved = {k: os.environ.pop(k) for k in _PROXY_ENV_VARS if k in os.environ}
    try:
        return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=timeout)
    finally:
        os.environ.update(saved)
