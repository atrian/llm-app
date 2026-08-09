"""
Пайплайн оценки LLM с помощью Ragas.

Генерирует ответы целевой модели на золотых примерах (tests/goldens.json),
считает метрики Ragas (Faithfulness, AnswerRelevancy, ContextRecall) и
сохраняет результаты в JSON и HTML.

Все провайдеры (target / evaluator / embeddings) используют OpenAI-compatible API.

Переменные окружения (значения по умолчанию):
  TARGET_PROVIDER          ollama
  TARGET_BASE_URL          http://localhost:11434/v1
  TARGET_MODEL             qwen3.6:27b
  TARGET_API_KEY           ollama

  EVAL_PROVIDER            ollama
  EVAL_BASE_URL            http://localhost:11434/v1
  EVAL_MODEL               qwen3.6:27b
  EVAL_API_KEY             ollama

  EMBEDDING_PROVIDER       ollama
  EMBEDDING_BASE_URL       http://localhost:11434/v1
  EMBEDDING_MODEL          qwen3-embedding:8b
  EMBEDDING_API_KEY        ollama

  GOLDENS_PATH             tests/goldens.json
  RESULT_JSON_PATH         tests/results/ragas_results.json
  RESULT_HTML_PATH         tests/results/ragas_results.html
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LcOpenAIEmbeddings

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*deprecated.*", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=r"Importing .* from 'ragas\.metrics' is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r"LangchainLLMWrapper is deprecated.*",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("ragas_pipeline")

DEFAULTS = {
    "TARGET_PROVIDER": "ollama",
    "TARGET_BASE_URL": "http://localhost:11434/v1",
    "TARGET_MODEL": "qwen3.6:27b",
    "TARGET_API_KEY": "ollama",
    "EVAL_PROVIDER": "ollama",
    "EVAL_BASE_URL": "http://localhost:11434/v1",
    "EVAL_MODEL": "qwen3.6:27b",
    "EVAL_API_KEY": "ollama",
    "EMBEDDING_PROVIDER": "ollama",
    "EMBEDDING_BASE_URL": "http://localhost:11434/v1",
    "EMBEDDING_MODEL": "qwen3-embedding:8b",
    "EMBEDDING_API_KEY": "ollama",
    "GOLDENS_PATH": "tests/goldens.json",
    "RESULT_JSON_PATH": "tests/results/ragas_results.json",
    "RESULT_HTML_PATH": "tests/results/ragas_results.html",
}


def env(key: str) -> str:
    return (os.getenv(key) or DEFAULTS.get(key, "")).strip()


def load_goldens(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    log.info("Загружено золотых примеров: %d из %s", len(data), p)
    return data


def make_openai_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


SYSTEM_PROMPT = (
    "Вы — ассистент службы поддержки кредитной компании North Star Credit. "
    "Отвечайте ТОЛЬКО фактами из предоставленного контекста. "
    "Если ответа нет в контексте — ответьте: «В предоставленном контексте нет информации». "
    "Не придумывайте факты, цифры или правила."
)


def generate_answer(
    question: str,
    contexts: list[str],
    *,
    client: OpenAI,
    model: str,
) -> str:
    joined = "\n---\n".join(contexts) if contexts else "(контекст отсутствует)"
    prompt = (
        f"Контекст:\n{joined}\n\n"
        f"Вопрос: {question}\n"
        f"Краткий ответ:"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=300,
    )
    return (resp.choices[0].message.content or "").strip()


def build_eval_dataset(
    goldens: list[dict[str, Any]],
    *,
    client: OpenAI,
    model: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    log.info("Генерация ответов моделью %s ...", model)
    for g in tqdm(goldens, desc="Генерация"):
        answer = generate_answer(
            g["question"], g["contexts"], client=client, model=model
        )
        rows.append(
            {
                "id": g["id"],
                "category": g.get("category", "factual"),
                "question": g["question"],
                "answer": answer,
                "contexts": list(g["contexts"]),
                "ground_truth": g["ground_truth"],
            }
        )
    return pd.DataFrame(rows)


def run_ragas_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Запуск Ragas-оценки ...")

    eval_model = env("EVAL_MODEL")
    eval_base_url = env("EVAL_BASE_URL")
    eval_api_key = env("EVAL_API_KEY")

    emb_model = env("EMBEDDING_MODEL")
    emb_base_url = env("EMBEDDING_BASE_URL")
    emb_api_key = env("EMBEDDING_API_KEY")

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=eval_model,
            temperature=0,
            api_key=eval_api_key,
            base_url=eval_base_url,
        )
    )

    evaluator_embeddings = LcOpenAIEmbeddings(
        model=emb_model,
        api_key=emb_api_key,
        base_url=emb_base_url,
        check_embedding_ctx_length=False,
    )

    hf_ds = Dataset.from_pandas(
        df[["question", "answer", "contexts", "ground_truth"]]
    )

    metrics = [Faithfulness(), AnswerRelevancy(), ContextRecall()]

    log.info(
        "Ragas config: eval_model=%s, emb_model=%s, rows=%d",
        eval_model,
        emb_model,
        len(hf_ds),
    )

    result = evaluate(
        dataset=hf_ds,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=True,
        raise_exceptions=False,
        batch_size=4,
    )

    try:
        details = result.to_pandas()
    except AttributeError:
        details = pd.DataFrame(result)

    metric_cols = [c for c in details.columns if c not in df.columns]
    for col in metric_cols:
        df[col] = details[col].values

    return df


def _safe_mean(series: pd.Series) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.mean())


def compute_summary(df: pd.DataFrame) -> dict[str, float | None]:
    metric_names = ["faithfulness", "answer_relevancy", "context_recall"]
    summary: dict[str, float | None] = {}
    for m in metric_names:
        col = m if m in df.columns else m.replace("_", " ")
        if col in df.columns:
            summary[m] = _safe_mean(df[col])
        else:
            summary[m] = None
    return summary


def _fmt_html_row(metric: str, value: float | None) -> str:
    vstr = "N/A" if value is None else f"{value:.4f}"
    return f"<tr><td>{metric}</td><td>{vstr}</td></tr>"


def generate_html_report(
    df: pd.DataFrame,
    summary: dict[str, float | None],
    *,
    target_model: str,
    eval_model: str,
    emb_model: str,
) -> str:
    detail_cols = [
        "id",
        "category",
        "question",
        "answer",
        "ground_truth",
    ]
    for m in ["faithfulness", "answer_relevancy", "context_recall"]:
        col = m if m in df.columns else m.replace("_", " ")
        if col in df.columns:
            detail_cols.append(col)

    detail_html = df[detail_cols].to_html(
        index=False, escape=False, table_id="details", border=0
    )

    summary_rows = "\n".join(_fmt_html_row(m, v) for m, v in summary.items())

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Ragas Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 2em; background: #f8f9fa; }}
  h1, h2 {{ color: #333; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; background: #fff; }}
  th, td {{ border: 1px solid #dee2e6; padding: 8px 12px; text-align: left; }}
  th {{ background: #e9ecef; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  .summary-table {{ width: auto; }}
  .meta {{ color: #6c757d; font-size: 0.9em; margin-bottom: 1em; }}
</style>
</head>
<body>
<h1>Отчёт Ragas-оценки</h1>
<div class="meta">
  <p>Target model: <b>{target_model}</b></p>
  <p>Evaluator LLM: <b>{eval_model}</b></p>
  <p>Embedding model: <b>{emb_model}</b></p>
</div>
<h2>Средние метрики</h2>
<table class="summary-table">
<tr><th>Метрика</th><th>Значение</th></tr>
{summary_rows}
</table>
<h2>Детализация по кейсам</h2>
{detail_html}
</body>
</html>"""


def save_results(
    df: pd.DataFrame,
    summary: dict[str, float | None],
    *,
    target_model: str,
    eval_model: str,
    emb_model: str,
    json_path: str,
    html_path: str,
) -> None:
    json_p = Path(json_path)
    html_p = Path(html_path)
    json_p.parent.mkdir(parents=True, exist_ok=True)
    html_p.parent.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        case: dict[str, Any] = {
            "id": row.get("id"),
            "category": row.get("category"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "ground_truth": row.get("ground_truth"),
            "metrics": {},
        }
        for m in ["faithfulness", "answer_relevancy", "context_recall"]:
            col = m if m in df.columns else m.replace("_", " ")
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val):
                    case["metrics"][m] = float(val)
                else:
                    case["metrics"][m] = None
            else:
                case["metrics"][m] = None
        cases.append(case)

    payload: dict[str, Any] = {
        "target_model": target_model,
        "eval_model": eval_model,
        "embedding_model": emb_model,
        "summary": summary,
        "cases": cases,
    }

    with open(json_p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("JSON-отчёт сохранён: %s", json_p)

    html = generate_html_report(
        df,
        summary,
        target_model=target_model,
        eval_model=eval_model,
        emb_model=emb_model,
    )
    with open(html_p, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("HTML-отчёт сохранён: %s", html_p)


def run_pipeline() -> dict[str, Any]:
    """Запускает полный пайплайн и возвращает dict со сводкой и кейсами."""
    goldens = load_goldens(env("GOLDENS_PATH"))

    target_base_url = env("TARGET_BASE_URL")
    target_api_key = env("TARGET_API_KEY")
    target_model = env("TARGET_MODEL")

    target_client = make_openai_client(target_base_url, target_api_key)

    df = build_eval_dataset(
        goldens, client=target_client, model=target_model
    )

    df = run_ragas_evaluation(df)

    summary = compute_summary(df)

    save_results(
        df,
        summary,
        target_model=target_model,
        eval_model=env("EVAL_MODEL"),
        emb_model=env("EMBEDDING_MODEL"),
        json_path=env("RESULT_JSON_PATH"),
        html_path=env("RESULT_HTML_PATH"),
    )

    print("\n=== Сводка метрик ===")
    for k, v in summary.items():
        vstr = "N/A" if v is None else f"{v:.4f}"
        print(f"  {k:25s}: {vstr}")

    print("\n=== Детализация ===")
    for _, row in df.iterrows():
        cid = row.get("id", "?")
        cat = row.get("category", "")
        print(f"\n[{cid}] ({cat}) Q: {row.get('question', '')[:80]}")
        print(f"     A: {str(row.get('answer', ''))[:120]}")
        scores = []
        for m in ["faithfulness", "answer_relevancy", "context_recall"]:
            col = m if m in df.columns else m.replace("_", " ")
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val):
                    scores.append(f"{m}={float(val):.3f}")
                else:
                    scores.append(f"{m}=N/A")
        print(f"     Scores: {', '.join(scores)}")

    # Формируем cases в том же формате, что в save_results
    structured_cases: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        case_out: dict[str, Any] = {
            "id": row.get("id"),
            "category": row.get("category"),
            "question": row.get("question"),
            "answer": row.get("answer"),
            "ground_truth": row.get("ground_truth"),
            "metrics": {},
        }
        for m in ["faithfulness", "answer_relevancy", "context_recall"]:
            col = m if m in df.columns else m.replace("_", " ")
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val):
                    case_out["metrics"][m] = float(val)
                else:
                    case_out["metrics"][m] = None
            else:
                case_out["metrics"][m] = None
        structured_cases.append(case_out)

    return {
        "summary": summary,
        "cases": structured_cases,
    }


if __name__ == "__main__":
    run_pipeline()
