"""Ortam değişkenleri (.env) — HF token vb."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_project_env() -> str | None:
    load_dotenv(PROJECT_ROOT / ".env")

    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    if token:
        os.environ["HF_TOKEN"] = token.strip()
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token.strip()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

    return token.strip() if token else None


def get_hf_token() -> str | None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    return token.strip() if token else None
