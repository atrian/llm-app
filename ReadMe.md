# llm-app

Репозиторий pre-deploy проверки LLM-моделей.

Что здесь есть:
- `pre_deploy_test.py` — основной прогон 4 моделей:
  - `Qwen3-0.6B` через `Ollama`
  - `Qwen3-4B` через `Ollama`
  - `Qwen3-8B` через `Ollama`
  - `YandexGPT Lite`
- `run_ragas_demo_test.py` — одиночный прогон одной модели с метриками `RAGAS`
- `toxic_test.py` — отдельный judge-тест на токсичность и грубость
- `seed_qdrant_demo.py` — загрузка демо-знаний в `Qdrant`
- `run_rag_chat_demo.py` — интерактивный RAG-диалог с подробными retrieval-логами
- `docker-compose.yaml` — локальные `MLflow`, `Langfuse` и `Qdrant`
- `RAG_DEMO.md` — пошаговая инструкция по запуску простого RAG-демо

## Что проверяется

Для каждого прогона `RAGAS` считаются:
- "faithfulness"
- "answer_relevancy"
- "context_precision"
- "context_recall"
- "qa_semantic_correctness"

Результаты пишутся:
- в `MLflow`
- в `Langfuse`

## Что должно быть готово

- активирован `.venv`
- заполнен `.env`
- запущен `Docker Desktop`
- установлен `Ollama`

## Запуск локальных сервисов

Поднять локальные сервисы (`MLflow`, `Langfuse`, `Qdrant`):

```bash
docker compose up -d
docker compose ps
```

URL:
- `MLflow`: `http://localhost:5001`
- `Langfuse`: `http://localhost:3000`
- `Qdrant`: `http://localhost:6333`

## Запуск тестов

Полный pre-deploy прогон:

```bash
./.venv/bin/python pre_deploy_test.py
```

Что делает скрипт:
- при необходимости поднимает `Ollama`
- по очереди скачивает локальную модель
- прогоняет тест
- выгружает модель из памяти
- пишет метрики и артефакты в `MLflow`

Одиночный `RAGAS`-прогон одной модели:

```bash
./.venv/bin/python run_ragas_demo_test.py
```

Отдельный тест токсичности:

```bash
./.venv/bin/python toxic_test.py
```

## Простой RAG demo

Для локального RAG-демо добавлены:

- `Qdrant` в `docker-compose`
- готовая база знаний по кредитным политикам
- скрипт индексации в `Qdrant`
- интерактивный диалоговый скрипт с логами по retrieval и ответу

Быстрый сценарий:

```bash
docker compose up -d qdrant
ollama serve
ollama pull hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M
ollama pull nomic-embed-text
./.venv/bin/python run_rag_chat_demo.py --reindex
```

Подробная инструкция лежит в `RAG_DEMO.md`.

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

## CI

В `GitHub Actions` запускается:

```bash
python3 pre_deploy_test.py
python3 toxic_test.py
```

Workflow рассчитан на `self-hosted runner`.
