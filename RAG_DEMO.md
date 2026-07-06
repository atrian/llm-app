# RAG demo в этом репозитории

Этот репозиторий изначально был набором локальных проверок LLM-моделей с `MLflow` и `Langfuse`. Для демо сюда добавлен отдельный простой RAG-сценарий на `Qdrant` + локальном `Ollama`.

Что добавлено:

- `qdrant` в [docker-compose.yaml](/Users/newuser/Documents/repo/llm-app/docker-compose.yaml)
- [rag_demo_data/credit_policy_chunks.json](/Users/newuser/Documents/repo/llm-app/rag_demo_data/credit_policy_chunks.json) — готовая демо-база знаний по кредитным политикам
- [seed_qdrant_demo.py](/Users/newuser/Documents/repo/llm-app/seed_qdrant_demo.py) — пересоздаёт коллекцию, показывает чанки и грузит их в `Qdrant`
- [run_rag_chat_demo.py](/Users/newuser/Documents/repo/llm-app/run_rag_chat_demo.py) — запускает интерактивный RAG-диалог и подробно логирует retrieval
- [rag_demo_common.py](/Users/newuser/Documents/repo/llm-app/rag_demo_common.py) — общая логика загрузки знаний, embeddings, Qdrant search и генерации ответа

## 1. Что это за demo

Домен знаний: вымышленные политики потребительских кредитов компании `North Star Credit`.

Почему такой вариант удобен для демо:

- вопросы хорошо ложатся на retrieval: возраст, доход, документы, ставка, досрочное погашение, реструктуризация
- ответы короткие и проверяемые
- хорошо видно, какие именно чанки подтянул `Qdrant`

## 2. Что нужно поднять

Нужны два локальных рантайма:

1. `Qdrant` через Docker
2. `Ollama` с двумя локальными моделями:
   - chat-модель: `hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M`
   - embedding-модель: `nomic-embed-text`

Если нужен только RAG-демо слой, достаточно поднять только `qdrant`:

```bash
docker compose up -d qdrant
docker compose ps qdrant
```

Если хочешь параллельно оставить текущие `MLflow` и `Langfuse`, можно поднимать всё как раньше:

```bash
docker compose up -d
docker compose ps
```

`Qdrant` будет доступен на:

- `http://localhost:6333` — REST API
- `localhost:6334` — gRPC

Проверка доступности `Qdrant`:

```bash
curl http://localhost:6333/collections
```

## 3. Что нужно в Ollama

Поднять сервер:

```bash
ollama serve
```

Один раз скачать модели:

```bash
ollama pull hf.co/Qwen/Qwen3-4B-GGUF:Q4_K_M
ollama pull nomic-embed-text
```

## 4. Как наполнить Qdrant знаниями

Скрипт читает [rag_demo_data/credit_policy_chunks.json](/Users/newuser/Documents/repo/llm-app/rag_demo_data/credit_policy_chunks.json), печатает все чанки, делает embedding для каждого чанка через `Ollama`, пересоздаёт коллекцию `credit_policy_demo` и заливает точки в `Qdrant`.

Важно: после последних правок retrieval стал гибридным, а embedding теперь строится не только по `text`, но и по `section/title/keywords`. Поэтому после обновления кода обязательно один раз сделай `--reindex`, иначе в `Qdrant` останутся старые вектора.

Команда:

```bash
./.venv/bin/python seed_qdrant_demo.py
```

Если нужна другая коллекция:

```bash
./.venv/bin/python seed_qdrant_demo.py --collection my_demo_collection
```

Что видно в логах:

- полный список чанков
- `chunk_id`, `section`, `title`, `keywords`
- размерность embedding-вектора
- факт пересоздания коллекции
- число точек, загруженных в `Qdrant`

## 5. Как запустить диалог

Базовый запуск:

```bash
./.venv/bin/python run_rag_chat_demo.py
```

Самый удобный первый запуск с гарантированным переиндексированием:

```bash
./.venv/bin/python run_rag_chat_demo.py --reindex
```

Одноразовый запрос без интерактивного цикла:

```bash
./.venv/bin/python run_rag_chat_demo.py --question "Можно ли погасить кредит досрочно?"
```

Во время интерактивного диалога доступны команды:

- `:help`
- `:chunks`
- `exit`

## 6. Что именно логирует RAG-скрипт

Для каждого вопроса [run_rag_chat_demo.py](/Users/newuser/Documents/repo/llm-app/run_rag_chat_demo.py) пишет в лог:

- адрес `Qdrant`, имя коллекции, `top_k`, `score_threshold`
- сколько vector-кандидатов берётся до rerank
- исходный вопрос
- набор нормализованных терминов вопроса для lexical rerank
- размерность embedding-вектора вопроса
- итоговый `top-k` после гибридного rerank
- `final_score`, `vector_score`, `lexical_score`
- `chunk_id`, `section`, `title`, `keywords`
- `matched_terms`
- короткий preview текста каждого чанка
- сколько чанков ушло в prompt
- какой `chat_model` использовался
- итоговый ответ RAG

То есть по логам можно буквально показать цепочку:

`вопрос -> embedding -> qdrant top-k -> score -> контекст -> ответ`

## 7. Примеры вопросов для демо

- `Какие требования к возрасту и регистрации у заёмщика?`
- `Какая минимальная сумма частичного досрочного погашения?`
- `Когда заявка уходит на ручную проверку?`
- `Какая ставка у зарплатных клиентов?`
- `Через сколько аннулируется POS-кредит, если товара нет?`
- `Доступна ли реструктуризация после одного платежа?`
- `Какой штраф за просрочку?`

Негативный контрольный вопрос:

- `Есть ли страховка по кредиту?`


## 8. Какие чанки здесь используются

Чанки лежат прямо в JSON и уже готовы к индексации. Это сделано специально для простого и предсказуемого демо: можно открыть файл и сразу увидеть, какие знания реально попадут в `Qdrant`.

Пример двух чанков:

`credit-05-auto-decline`

> Система отклоняет заявку автоматически, если у клиента есть текущая просрочка по любому кредиту, если в последние 180 дней была просрочка больше 30 дней или если показатель долговой нагрузки PTI превышает 60 процентов.

`credit-10-early-repayment`

> Полное и частичное досрочное погашение доступно без комиссии в любой день действия договора. Минимальная сумма частичного досрочного погашения составляет 5 000 рублей.

Полный список смотри в [rag_demo_data/credit_policy_chunks.json](/Users/newuser/Documents/repo/llm-app/rag_demo_data/credit_policy_chunks.json).


Если машина слабая и важнее отзывчивость, можно временно заменить chat-модель:

```bash
./.venv/bin/python run_rag_chat_demo.py --chat-model hf.co/Qwen/Qwen3-0.6B-GGUF:Q8_0
```

Если важнее качество и хватает памяти, можно пробовать `8B`:

```bash
./.venv/bin/python run_rag_chat_demo.py --chat-model hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M
```

## 9. Полезные переменные окружения

- `RAG_QDRANT_HOST`
- `RAG_QDRANT_PORT`
- `RAG_QDRANT_COLLECTION`
- `RAG_CHAT_MODEL`
- `RAG_EMBEDDING_MODEL`
- `RAG_TOP_K`
- `OLLAMA_BASE_URL`
- `OLLAMA_API_KEY`

Пример:

```bash
RAG_TOP_K=5 ./.venv/bin/python run_rag_chat_demo.py
```

## 10. Как остановить demo

Остановить только `Qdrant`:

```bash
docker compose stop qdrant
```

Полностью остановить весь compose-стек:

```bash
docker compose down
```

Остановить загруженные модели `Ollama`:

```bash
ollama ps | awk 'NR>1 {print $1}' | xargs -n1 ollama stop
```

