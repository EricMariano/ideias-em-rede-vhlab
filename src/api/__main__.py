"""Entrada `python -m src.api`."""

import logging

import uvicorn

from src.api.config import get_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        timeout_keep_alive=180,
        reload=False,
    )


if __name__ == "__main__":
    main()
