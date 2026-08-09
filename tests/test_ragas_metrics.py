"""
Pytest-тесты качества LLM с метриками Ragas и quality gates.

Тесты разделены на три группы:
1. Threshold-тесты на средние значения по factual-кейсам
2. Тест на отсутствие грубых галлюцинаций по каждому кейсу
3. Структурные тесты золотого датасета

Запуск:
    pytest tests/ -v
"""

from __future__ import annotations

import pytest


def _factual_cases(evaluation_results: dict) -> list[dict]:
    return [c for c in evaluation_results["cases"] if c.get("category") != "negative"]


def _negative_cases(evaluation_results: dict) -> list[dict]:
    return [c for c in evaluation_results["cases"] if c.get("category") == "negative"]


def _mean(cases: list[dict], metric: str) -> float | None:
    vals = [c["metrics"][metric] for c in cases if c.get("metrics", {}).get(metric) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Quality-gate тесты на средние значения (только factual-кейсы)
# ---------------------------------------------------------------------------

class TestFactualMetricThresholds:
    """Пороги проверяются по factual-кейсам (где есть ответ в контексте)."""

    def test_faithfulness_above_threshold(self, evaluation_results, threshold_faithfulness):
        """Faithfulness (анти-галлюцинации) по factual-кейсам >= порога."""
        cases = _factual_cases(evaluation_results)
        value = _mean(cases, "faithfulness")
        assert value is not None, "Метрика faithfulness не была рассчитана"
        assert value >= threshold_faithfulness, (
            f"Faithfulness {value:.4f} ниже порога {threshold_faithfulness:.2f}"
        )

    def test_answer_relevancy_above_threshold(self, evaluation_results, threshold_answer_relevancy):
        """Answer Relevancy по factual-кейсам >= порога."""
        cases = _factual_cases(evaluation_results)
        value = _mean(cases, "answer_relevancy")
        assert value is not None, "Метрика answer_relevancy не была рассчитана"
        assert value >= threshold_answer_relevancy, (
            f"Answer Relevancy {value:.4f} ниже порога {threshold_answer_relevancy:.2f}"
        )

    def test_context_recall_above_threshold(self, evaluation_results, threshold_context_recall):
        """Context Recall по factual-кейсам >= порога."""
        cases = _factual_cases(evaluation_results)
        value = _mean(cases, "context_recall")
        assert value is not None, "Метрика context_recall не была рассчитана"
        assert value >= threshold_context_recall, (
            f"Context Recall {value:.4f} ниже порога {threshold_context_recall:.2f}"
        )


# ---------------------------------------------------------------------------
# Проверка на галлюцинации
# ---------------------------------------------------------------------------

class TestHallucinationGates:

    def test_negative_cases_no_hallucination(self, evaluation_results):
        """
        Негативные кейсы: модель не должна галлюцинировать.
        Ответ должен содержать признак отсутствия информации
        ("нет информации", "не нашёл" и т.п.).
        """
        cases = _negative_cases(evaluation_results)
        assert len(cases) > 0, "В датасете нет негативных кейсов"

        hallucinated = []
        refusal_markers = (
            "нет информации",
            "не нашёл",
            "не нашел",
            "в предоставленном контексте нет",
            "не содержится",
            "отсутствует",
        )
        for case in cases:
            answer_lower = (case.get("answer") or "").lower()
            if not any(m in answer_lower for m in refusal_markers):
                hallucinated.append(
                    f"  [{case['id']}] модель не отказалась отвечать: {case['answer'][:80]}"
                )
        assert not hallucinated, (
            "Модель галлюцинирует на негативных кейсах:\n" + "\n".join(hallucinated)
        )


# ---------------------------------------------------------------------------
# Дымовой тест: данные корректны
# ---------------------------------------------------------------------------

class TestGoldenDataset:

    def test_goldens_count(self, goldens):
        """В наборе должно быть от 10 до 20 золотых примеров."""
        assert 10 <= len(goldens) <= 20, (
            f"Ожидалось 10-20 golden примеров, получено {len(goldens)}"
        )

    def test_goldens_structure(self, goldens):
        """Каждый golden должен содержать обязательные поля."""
        required = {"id", "question", "ground_truth", "contexts", "category"}
        for g in goldens:
            missing = required - set(g.keys())
            assert not missing, (
                f"Golden {g.get('id', '?')} не содержит поля: {missing}"
            )

    def test_goldens_have_negative_cases(self, goldens):
        """В наборе должны быть негативные кейсы (для проверки на галлюцинации)."""
        cats = {g["category"] for g in goldens}
        assert "negative" in cats, "Нет негативных кейсов для проверки на галлюцинации"
