# llm-app

Репозиторий с общей локальной средой для трёх отдельных демо-направлений:

- `rag/` — классический локальный RAG на `Qdrant` + `Ollama`
- `testing/` — pre-deploy тестирование моделей, `RAGAS`, judge-проверки и служебные утилиты
- `agent/` — controlled-agent с декомпозицией, mock tool и self-check поверх той же базы знаний

Ноутбуки оставлены на верхнем уровне:

- `practice.ipynb`
- `practice_m2_local_llm.ipynb`

Общие для всего репозитория файлы:

- `docker-compose.yaml` — единый compose-стек
- `requirements.txt` — единый набор зависимостей
- `.env` — единая конфигурация среды

Все команды ниже предполагают запуск из корня репозитория.

## Структура

```text
.
├── rag/
├── testing/
├── agent/
├── docker-compose.yaml
├── requirements.txt
├── practice.ipynb
└── practice_m2_local_llm.ipynb
```

## Общая среда

Что должно быть подготовлено:

- активирован `.venv`
- заполнен `.env`
- запущен `Docker Desktop`
- установлен `Ollama`

Общий compose-стек:

```bash
docker compose up -d
docker compose ps
```

URL:

- `MLflow`: `http://localhost:5001`
- `Langfuse`: `http://localhost:3000`
- `Qdrant`: `http://localhost:6333`

Если нужен только RAG или agent demo, достаточно поднимать только `qdrant`:

```bash
docker compose up -d qdrant
docker compose ps qdrant
```

## Быстрый старт по папкам

`rag/`

- документация: `rag/README.md`
- индексация: `./.venv/bin/python -m rag.seed_qdrant_demo`
- чат: `./.venv/bin/python -m rag.run_rag_chat_demo --reindex`

`testing/`

- документация: `testing/README.md`
- pre-deploy: `./.venv/bin/python -m testing.pre_deploy_test`
- одиночный `RAGAS`: `./.venv/bin/python -m testing.run_ragas_demo_test`
- токсичность: `./.venv/bin/python -m testing.toxic_test`

`agent/`

- документация: `agent/README.md`
- список кейсов: `./.venv/bin/python -m agent.run_agent_credit_demo --list-cases`
- agent demo: `./.venv/bin/python -m agent.run_agent_credit_demo --reindex --case-id C-102`

## Что где пишет

- `rag/` — обычные консольные логи
- `agent/` — обычные консольные логи, без обязательного `Langfuse`
- `testing/` — метрики и артефакты в `MLflow`, а при наличии ключей ещё и трассы в `Langfuse`

## Полезные команды

Остановить локально загруженные модели `Ollama`:

```bash
ollama ps | awk 'NR>1 {print $1}' | xargs -n1 ollama stop
```

Остановить локальные сервисы:

```bash
docker compose down
```

Полный сброс `Langfuse`, `MLflow` и `Qdrant`:

```bash
docker compose down -v
docker compose up -d
```

## ДЗ-9: CI/CD с автотестами и проверкой на галлюцинации (Ragas)

В папке `tests/` реализован отдельный пайплайн оценки LLM с помощью библиотеки
[Ragas](https://github.com/explodinggradients/ragas) и интеграция в CI/CD.

### Структура

```text
tests/
├── goldens.json              — 15 золотых Q&A пар (12 фактических + 3 негативных)
├── ragas_pipeline.py         — модуль оценки: генерация ответов + Ragas + отчёты
├── conftest.py               — pytest fixtures (загрузка goldens, thresholds)
├── test_ragas_metrics.py     — pytest quality-gate тесты
└── results/                  — артефакты (JSON + HTML отчёты, gitignored)
```

### Метрики Ragas

| Метрика             | Что измеряет                                                         | Порог |
|---------------------|----------------------------------------------------------------------|-------|
| Faithfulness        | Проверка на галлюцинации: все ли утверждения в ответе подтверждены контекстом | ≥ 0.70 |
| Answer Relevancy    | Релевантность ответа поставленному вопросу                           | ≥ 0.30 |
| Context Recall      | Полнота извлечения: все ли факты из ground_truth найдены в контексте | ≥ 0.70 |

Пороги проверяются только по factual-кейсам (где ответ есть в контексте).
Негативные кейсы проверяются отдельным тестом на отсутствие галлюцинаций.

Пороги настраиваются через переменные окружения:

- `THRESH_FAITHFULNESS` (по умолчанию `0.70`)
- `THRESH_ANSWER_RELEVANCY` (по умолчанию `0.30`)
- `THRESH_CONTEXT_RECALL` (по умолчанию `0.70`)

Answer Relevancy имеет пониженный порог, так как метрика генерирует
варианты вопросов к ответу и сравнивает их эмбеддинги — для очень коротких
ответов («45 000 рублей») этот метод систематически занижает оценку.

### Золотые примеры

15 Q&A пар на домене кредитных политик `North Star Credit` (база знаний из
`rag/data/credit_policy_chunks.json`):

- 12 фактических вопросов с проверяемыми ответами (возраст, ставка, PTI, штрафы …)
- 3 негативных вопроса на темы, которых нет в базе (карта рассрочки, ипотека,
  комиссии за наличные) — модель должна ответить «нет в контексте», а не выдумывать

### Локальный запуск

Нужен работающий Ollama с моделями для генерации и эмбеддингов:

```bash
ollama pull qwen3.6:27b
ollama pull qwen3-embedding:8b
```

Установка зависимостей и запуск пайплайна:

```bash
pip install -r requirements.txt

# Запуск оценки как скрипта (генерирует JSON + HTML отчёты)
python -m tests.ragas_pipeline

# Запуск через pytest (quality gates)
pytest tests/ -v
```

Отчёты сохраняются в:

- `tests/results/ragas_results.json` — машиночитаемый JSON со сводкой и кейсами
- `tests/results/ragas_results.html` — человекочитаемый HTML-отчёт

### Переменные окружения

Все провайдеры используют OpenAI-compatible API. Значения по умолчанию —
локальный Ollama:

| Переменная            | Назначение              | По умолчанию              |
|-----------------------|-------------------------|---------------------------|
| `TARGET_BASE_URL`     | URL целевой модели      | `http://localhost:11434/v1` |
| `TARGET_MODEL`        | Модель для генерации    | `qwen3.6:27b`             |
| `EVAL_BASE_URL`       | URL evaluator LLM       | `http://localhost:11434/v1` |
| `EVAL_MODEL`          | LLM-судья для Ragas     | `qwen3.6:27b`             |
| `EMBEDDING_BASE_URL`  | URL embedding-модели    | `http://localhost:11434/v1` |
| `EMBEDDING_MODEL`     | Embedding-модель        | `qwen3-embedding:8b`      |

### CI/CD

Workflow `.github/workflows/cicd-ragas.yml` запускается на push/PR в ветки
`hw9-cicd-ragas` и `main`. Схема работы:

1. Поднимается service-контейнер `ollama/ollama`
2. Устанавливаются Python-зависимости
3.Pullятся модели `qwen2.5:3b` + `nomic-embed-text`
4. Запускается `pytest tests/ -v` — quality gate падает, если любая метрика ниже порога
5. Артефакты (JSON/HTML отчёты) загружаются и хранятся 30 дней

Quality gate: пайплайн завершается ошибкой, если:

- средняя Faithfulness по factual-кейсам < 0.70 (галлюцинации)
- средняя Answer Relevancy по factual-кейсам < 0.30
- средняя Context Recall по factual-кейсам < 0.70
- хотя бы один factual-кейс имеет Faithfulness < 0.50 (грубая галлюцинация)
- модель галлюцинирует на негативных кейсах (не отказывается отвечать)

### Интерпретация результатов

- **Faithfulness = 1.0** — каждое утверждение в ответе строго выводится из контекста, галлюцинаций нет.
- **Faithfulness < 0.5** — модель добавляет факты, которых нет в контексте (галлюцинация).
- **Context Recall = 1.0** — контекст содержит всю информацию для правильного ответа.
- **Answer Relevancy = 1.0** — ответ прямо адресует вопрос, без лишней информации.

---

## CI

В `GitHub Actions` ожидаются те же тестовые entrypoint-ы, только в новой структуре:

```bash
python3 -m testing.pre_deploy_test
python3 -m testing.toxic_test
```

Workflow рассчитан на `self-hosted runner`.
