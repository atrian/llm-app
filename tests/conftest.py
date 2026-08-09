"""
Pytest fixtures для Ragas-тестов.

Конфигурация через переменные окружения (значения по умолчанию):
  GOLDENS_PATH               tests/goldens.json
  RESULT_JSON_PATH           tests/results/ragas_results.json

  THRESH_FAITHFULNESS        0.70
  THRESH_ANSWER_RELEVANCY    0.30
  THRESH_CONTEXT_RECALL      0.70

Все остальные переменные (TARGET_*, EVAL_*, EMBEDDING_*) пробрасываются
в ragas_pipeline — см. DEFAULTS в tests/ragas_pipeline.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(key: str, default: str) -> str:
    return (os.getenv(key) or default).strip()


@pytest.fixture(scope="session")
def goldens_path() -> Path:
    raw = _env("GOLDENS_PATH", "tests/goldens.json")
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


@pytest.fixture(scope="session")
def goldens(goldens_path: Path) -> list[dict]:
    with open(goldens_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def evaluation_results() -> dict:
    """
    Запускает полный Ragas-пайплайн один раз за сессию.
    Возвращает dict с ключами summary и cases.
    """
    from tests.ragas_pipeline import run_pipeline

    return run_pipeline()


@pytest.fixture(scope="session")
def summary(evaluation_results: dict) -> dict:
    return evaluation_results["summary"]


@pytest.fixture(scope="session")
def threshold_faithfulness() -> float:
    return float(_env("THRESH_FAITHFULNESS", "0.70"))


@pytest.fixture(scope="session")
def threshold_answer_relevancy() -> float:
    return float(_env("THRESH_ANSWER_RELEVANCY", "0.30"))


@pytest.fixture(scope="session")
def threshold_context_recall() -> float:
    return float(_env("THRESH_CONTEXT_RECALL", "0.70"))
