# Retrieval Agent MVP — Техническая спецификация

**Статус:** готово к имплементации
**Цель:** SID-1-like retrieval-агент, возвращающий ранжированный список релевантных документов на диалоговый вход. Поддерживает несколько корпусов (MuSiQue, SBOL FAQ) через раздельные конфиги и индексы.
**Обучение:** не в этой итерации, но трассы и инфраструктура закладываются так, чтобы потом не переделывать.

---

## 1. Scope

### В рамках MVP

- Один режим работы: retrieval (возврат списка `doc_id`, ранжированных по релевантности).
- Один инструмент: `local_search` над faiss-индексом корпуса (MuSiQue или SBOL FAQ, выбирается конфигом).
- Диалоговый вход: агент отвечает на последний user-turn с учётом истории.
- Модель: Qwen3.5-35B-A3B через vLLM (fallback: Qwen3.5-27B dense).
- Оценка: NDCG@10 на MuSiQue dev vs single-shot baseline.

### Вне scope (отложено)

- Text-ответ (второй режим SID-1).
- Web-поиск, web-fetch, hierarchical retrieval.
- Prune chunks tool.
- Look_up / python executor.
- RL/SFT обучение.
- Multi-turn eval (сам harness поддерживает диалог, но MuSiQue — single-turn).

---

## 2. Архитектура

```
┌─────────────────┐     HTTP      ┌──────────────────┐
│  Agent Harness  │ ────────────► │   Tool Server    │
│  (Python)       │               │   (FastAPI)      │
│                 │               │                  │
│  ReAct loop     │               │  local_search    │
└────────┬────────┘               └────────┬─────────┘
         │                                 │
         │ OpenAI API                      │
         ▼                                 ▼
┌─────────────────┐               ┌──────────────────┐
│  vLLM server    │               │  Faiss index     │
│  Qwen3.5-35B    │               │  + embedder      │
└─────────────────┘               └──────────────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │  MuSiQue corpus  │
                                  │  (passages)      │
                                  └──────────────────┘
```

Три независимых процесса:
1. **vLLM server** — сервинг Qwen3.5-35B-A3B, OpenAI-compatible API.
2. **Tool server** — FastAPI, обёртка над faiss + embedder.
3. **Agent harness** — клиент, оркестрирует ReAct-цикл, ходит в vLLM и в Tool server.

Такое разделение позволит:
- Перезапускать компоненты независимо (vLLM греется долго).
- Подменить Tool server на mock-версию для будущего RL, не трогая harness.
- Логировать rollouts в одном месте (harness).

---

## 3. Faiss-индекс

Каждый датасет живёт в своём подкаталоге: `data/processed/<dataset>/` и `indexes/<dataset>/`. Конфиг (`configs/<dataset>.yaml`) связывает их. Переключение между датасетами — `CONFIG=configs/sbol.yaml bash scripts/serve_tool.sh`.

### Корпус — MuSiQue

**Источник:** MuSiQue (train + dev paragraphs, union), сырые файлы с [официального репо](https://github.com/StonyBrookNLP/musique).

**Структура сырых файлов MuSiQue:**
```
musique_ans_v1.0_train.jsonl
musique_ans_v1.0_dev.jsonl
musique_full_v1.0_train.jsonl
musique_full_v1.0_dev.jsonl
```

Нас интересуют `_ans_` файлы (answerable questions). Каждая строка jsonl — один пример вида:
```json
{
  "id": "2hop__123456_789012",
  "question": "...",
  "answer": "...",
  "paragraphs": [
    {
      "idx": 0,
      "title": "Article title",
      "paragraph_text": "...",
      "is_supporting": true
    },
    ...
  ],
  "question_decomposition": [...]
}
```

**Парсер (`indexing/parse_musique.py`):**

1. Пройти по всем строкам train+dev jsonl, извлечь все `paragraphs` из каждого примера.
2. Для каждого parag'а сформировать `(title, paragraph_text)` pair.
3. **Дедупликация** по content hash: `sha256((title + "\n\n" + paragraph_text).strip().lower())`. На практике одни и те же параграфы переиспользуются в сотнях вопросов — дедуп уменьшает корпус в ~100 раз.
4. Присвоить стабильный `doc_id` формата `musique_p_{6-digit-counter}`.
5. Сохранить маппинг `source_id → doc_id`, где `source_id = f"{split}:{question_id}:{paragraph_idx}"`. Он нужен для eval — чтобы gold paragraph indices из оригинального MuSiQue-примера превратить в наши `doc_id`.

**Выход парсера:**

```
data/processed/musique/
  corpus.jsonl              # {doc_id, title, text}, по строке на документ
  source_to_doc_id.json     # {source_id: doc_id}
  stats.json                # {num_raw, num_dedup, num_truncated, ...}
  musique_dev_eval.jsonl    # eval-датасет (question, gold_doc_ids)
```

**Единица индексации:** параграф. Gold-labels MuSiQue на уровне параграфа — единица индекса должна совпадать.

**Формат записи в корпусе (и в `metadata.jsonl` индекса):**
```json
{
  "doc_id": "musique_p_000001",
  "title": "Article title",
  "text": "Full paragraph text..."
}
```

### Корпус — SBOL FAQ

**Источник:** `data/raw/sbol/faq_index_28_apr.json` — массив FAQ-записей.

**Парсер (`indexing/parse_sbol.py`):** каждая FAQ-запись → один документ; `alternative_questions` → eval-строки.

**Формат документа:**
```json
{
  "doc_id": "sbol_<question_id>",
  "title": "<question>",
  "text": "Вопрос:<question>\nРаздел:<sections>\nОтвет:<answer>"
}
```

**Формат eval-строки** (одна на каждый `alternative_question`):
```json
{
  "question_id": "sbol_<question_id>_alt_<i>",
  "question": "<alternative_question>",
  "gold_doc_ids": ["sbol_<question_id>"]
}
```

**Выход парсера:**

```
data/processed/sbol/
  corpus.jsonl    # ~7 385 документов
  eval.jsonl      # ~5 623 eval-строки (только записи с alternative_questions)
```

### Embedder

| Датасет | Модель | Размерность |
|---------|--------|-------------|
| MuSiQue | `intfloat/e5-large-v2` | 1024 |
| SBOL FAQ | `intfloat/multilingual-e5-large` | 1024 |

Обе модели — одна архитектура, одинаковая размерность, одинаковые префиксы. Переключение — только `embedder.name` в конфиге; индекс пересобирается.

- Максимальная длина входа: 512 токенов (passages длиннее обрезаются)
- L2-нормализация эмбеддингов перед добавлением в индекс (обязательно для cosine через IndexFlatIP)

**Префиксы (критично — E5 без них работает заметно хуже):**
- При индексации passages: `"passage: " + text`
- При поиске query: `"query: " + text`

Префиксы добавляются внутри `embedder.py`, снаружи ни индексатор, ни tool server про них знать не должны. `local_search.query` приходит в tool server как обычный текст.

**Warning на длину.** При индексации логируем распределение длин и считаем % truncated passages — если больше 1%, стоит задуматься о sliding-window chunking или смене embedder.

### Тип индекса

`IndexFlatIP` (exact search, cosine через нормализацию). Для корпуса MuSiQue (~500k параграфов) — приемлемо по latency, не требует обучения.

Если в будущем корпус вырастет — переход на `IndexHNSWFlat` или `IndexIVFFlat` без изменений в API.

### Персистентность

```
indexes/
  musique/
    faiss.index        # бинарный faiss
    metadata.jsonl     # по строке на doc_id, в порядке векторов в индексе
    config.json        # embedder name, version, dim, normalization
  sbol/
    faiss.index
    metadata.jsonl
    config.json
```

`metadata.jsonl` и `faiss.index` должны оставаться согласованными: i-я строка метаданных соответствует i-му вектору. Конфиг датасета (`configs/<dataset>.yaml`) указывает `index.dir` на нужный подкаталог.

---

## 4. Tool Server

### Стек

- Python 3.11+
- FastAPI + uvicorn
- faiss-cpu (по умолчанию). `faiss-gpu` опционально через `index.use_gpu: true` — см. раздел 8.
- sentence-transformers или голый transformers для embedder

### API

#### `POST /local_search`

**Request:**
```json
{
  "query": "string",
  "top_k": 10
}
```

**Response:**
```json
{
  "results": [
    {
      "doc_id": "musique_p_000123",
      "title": "Article title",
      "text": "Paragraph text...",
      "score": 0.8421
    }
  ],
  "latency_ms": 45
}
```

**Поведение:**
- `top_k` по умолчанию 10, максимум 50.
- `score` — raw cosine similarity (после нормализации эмбеддингов).
- Порядок results — убывание score.
- `title` и `text` отдаются отдельными полями (не склеенными в `full_text`). В MuSiQue title часто несёт ключевую сущность для дисамбигуации однотипных параграфов — модель должна видеть его явно. Tool server отвечает за форматирование при сериализации в чат-сообщение (см. ниже).

**Сериализация tool result в chat message.** При добавлении результата в messages для модели harness рендерит результат как:
```
[doc_id: musique_p_000123 | score: 0.84]
Title: Article title

Paragraph text...
```
Формат фиксирован в `agent/prompts.py` рядом с системным промптом, версионируется вместе с ним.

#### `POST /lookup_by_id` (опционально, не на MVP)

**Request:**
```json
{ "doc_ids": ["musique_p_000123", "musique_p_000789"] }
```

**Response:**
```json
{
  "docs": [
    { "doc_id": "musique_p_000123", "title": "...", "text": "..." },
    { "doc_id": "musique_p_000789", "title": "...", "text": "..." }
  ]
}
```

Нужен только если потребуется восстанавливать тексты для `doc_id`, не встречавшихся в trajectory (на MVP harness собирает `seen_passages` по ходу rollout, и этого достаточно). Заложить как stub в `tool_server/main.py`, не реализовывать — комментарий с TODO.

#### `GET /healthz`

Проверка, что индекс загружен и embedder готов. Возвращает 200 / 503.

#### `GET /stats`

```json
{
  "num_docs": 512843,
  "embedder": "intfloat/e5-large-v2",
  "dim": 1024,
  "index_type": "IndexFlatIP"
}
```

### Кеш

Опциональный in-memory LRU-кеш `query_hash → results` (размер 10k). Нужен для:
- Повторяющихся вызовов внутри одного rollout (если агент перегенерирует один и тот же query).
- Будущих reproducible прогонов для RL.

На MVP можно выключить, но структура под кеш должна быть на месте.

---

## 5. Agent Harness

### Вход

```python
class AgentInput:
    messages: list[Message]  # [{"role": "user"|"assistant", "content": str}, ...]
    max_turns: int = 8
    max_tool_calls: int = 10
    top_k_default: int = 10
```

`messages` — весь диалог. Последнее сообщение — обязательно `user`. Агент отвечает именно на него, учитывая контекст предыдущих turns.

### Выход

```python
class RankedPassage:
    doc_id: str
    title: str
    text: str
    rank: int                       # 0-indexed, в порядке как в <answer>
    best_score: float | None        # лучший score из всех вызовов, где встречался doc_id
    first_seen_turn: int            # на каком turn впервые появился в tool_results
    num_times_retrieved: int        # сколько раз встречался по ходу rollout

class AgentOutput:
    ranked_doc_ids: list[str]            # итоговый ранжированный список doc_id (как раньше)
    ranked_passages: list[RankedPassage] # то же самое, но с восстановленным текстом
    trajectory: Trajectory               # полная трасса rollout
    stopped_reason: Literal["answer", "max_turns", "max_tool_calls", "parse_error"]
```

**Сборка `ranked_passages`.** Модель в `<answer>` возвращает только `doc_id` — это правильное разделение ответственности (компактный output, простая валидация, нет риска перефразирования пассажа моделью). Восстановление текстов делает harness:

1. По ходу rollout harness накапливает `seen_passages: dict[doc_id, RankedPassage]` из всех `local_search` results. Для повторно встреченного `doc_id` обновляются `best_score` (max) и `num_times_retrieved` (++).
2. После парсинга `<answer>` harness фильтрует список по `seen_passages` (галлюцинированные id отбрасываются с warning, как и в текущей валидации) и собирает `ranked_passages` в том порядке, что задала модель.
3. `ranked_doc_ids` остаётся для обратной совместимости и для вычисления NDCG (метрика не требует текстов).

**Опциональный fallback через tool server.** Если в будущем понадобится возвращать пассажи, которые модель указала, но которых нет в `seen_passages` (маловероятно при текущей валидации, но возможно в обученных моделях), tool server может предоставить endpoint `POST /lookup_by_id` (см. раздел 4). На MVP не нужен.

**Title в tool_results.** Tool server в response `local_search` возвращает `title` и `text` отдельными полями (а не склеенными в `full_text`). Это нужно потому, что в MuSiQue title часто несёт ключевую сущность ("Barack Obama" vs параграф про его раннюю карьеру) — без явного title модель хуже дисамбигуирует однотипные параграфы. Конкретный формат — см. раздел 4.

### Системный промпт (шаблон)

```
You are a retrieval agent. Given a conversation, find documents
relevant to the user's latest message, using the full conversation
as context.

You have one tool:
- local_search(query: str, top_k: int = 10) — returns a list of
  documents with doc_id, title, text, and relevance score.

You may call local_search multiple times with different queries
as needed. When you have enough information, return a ranked list
of doc_ids from most to least relevant.

Final answer format (must be exact):
<answer>doc_id_1, doc_id_2, doc_id_3, ...</answer>

Return only doc_ids you have actually seen in search results.
```

Финальный промпт требует итераций. Зафиксировать в коде как константу в `prompts.py` с версионированием (`PROMPT_V1`, `PROMPT_V2`...).

### ReAct-цикл

```
messages = [system_prompt] + user_messages
for turn in range(max_turns):
    response = vllm.chat.completions.create(
        messages=messages,
        tools=[local_search_schema],
        tool_choice="auto",
    )
    messages.append(response.message)

    if response.message.tool_calls:
        for call in response.message.tool_calls:
            result = tool_server.local_search(**call.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result)
            })
            total_tool_calls += 1
            if total_tool_calls >= max_tool_calls:
                break
        continue

    # No tool call → check for final answer
    answer = parse_answer(response.message.content)
    if answer is not None:
        return AgentOutput(ranked_doc_ids=answer, ...)

return AgentOutput(stopped_reason="max_turns", ...)
```

**Параллельные tool calls:** поддерживаются — если модель возвращает несколько `tool_calls` в одном turn, все исполняются, результаты добавляются в messages в том же порядке. Это важно для будущей производительности (SID-1 явно про это пишет как про свойство обученных моделей, но и не-обученный Qwen3.5 это умеет через tool_choice).

### Парсер ответа

```python
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def parse_answer(text: str) -> list[str] | None:
    m = ANSWER_RE.search(text)
    if not m:
        return None
    raw = m.group(1)
    return [x.strip() for x in raw.split(",") if x.strip()]
```

Валидация:
- Все `doc_id` из ответа должны встречаться в tool_results trajectory. Галлюцинированные id отбрасываются с warning в лог.
- Пустой ответ → `stopped_reason="parse_error"`.

### Tool schema для OpenAI API

```json
{
  "type": "function",
  "function": {
    "name": "local_search",
    "description": "Search local corpus for documents relevant to the query.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query"},
        "top_k": {"type": "integer", "default": 10, "maximum": 50}
      },
      "required": ["query"]
    }
  }
}
```

---

## 6. Трассы (trajectories)

Каждый rollout пишется в JSON-файл. Формат подобран так, чтобы быть совместимым с будущим veRL/Search-R1-style обучением.

```json
{
  "trajectory_id": "uuid",
  "timestamp": "2026-04-24T14:32:11Z",
  "model": "Qwen3.5-35B-A3B",
  "prompt_version": "v1",
  "input": {
    "messages": [...],
    "max_turns": 8,
    "max_tool_calls": 10
  },
  "messages_full": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
    ...
  ],
  "tool_calls": [
    {
      "turn": 1,
      "tool": "local_search",
      "arguments": {"query": "...", "top_k": 10},
      "result_summary": {"num_results": 10, "top_doc_ids": [...]},
      "latency_ms": 45
    }
  ],
  "output": {
    "ranked_doc_ids": [...],
    "stopped_reason": "answer",
    "num_turns": 3,
    "num_tool_calls": 2
  },
  "tokens": {
    "prompt_tokens": 1523,
    "completion_tokens": 412,
    "total_tokens": 1935
  }
}
```

### TI/TO consistency check

Периодическая (раз в N rollouts) проверка:
```
reconstructed = apply_chat_template(parse(trace.messages_full))
assert reconstructed == original_tokens
```

Нужно для будущего RL. На MVP — warning в лог при расхождении, не блокирующе.

---

## 7. Оценка

### Датасет

MuSiQue dev, подвыборка 200 примеров для итераций + полный dev для финального числа.

**Построение eval-датасета из сырого MuSiQue:**

Для каждого примера в `musique_ans_v1.0_dev.jsonl`:
1. Собрать `gold_source_ids = [f"dev:{example.id}:{p.idx}" for p in example.paragraphs if p.is_supporting]`
2. Через `source_to_doc_id.json` получить `gold_doc_ids`.
3. Сохранить как eval-пример.

Формат примера:
```json
{
  "question_id": "2hop__123456",
  "question": "text",
  "gold_doc_ids": ["musique_p_000012", "musique_p_000789"],
  "answer": "text"
}
```

На MVP в harness передаём `messages = [{"role": "user", "content": question}]` — single-turn формат, но совместимый с будущим multi-turn eval.

### Метрика

**Основная:** NDCG@10 по `gold_doc_ids`.

**Вспомогательные:**
- Recall@10 (для сравнимости с литературой)
- Precision@k для k ∈ {1, 3, 5, 10}
- Средняя длина траектории (num_turns, num_tool_calls)
- % примеров с `stopped_reason="max_turns"` или `"parse_error"` — показатель качества harness

### Baseline

**Single-shot RAG:** один вызов `local_search(user_question, top_k=10)`, возвращается as-is без агента. Если агент не бьёт baseline по NDCG@10 — проблема в harness/промпте, не в модели.

### Budget sweep

Прогоны на `max_tool_calls ∈ {1, 3, 5, 8}`. Строим tool-efficiency кривую: NDCG@10 как функция бюджета.

---

## 8. Конфигурация и запуск

### Инфраструктура

**Целевое железо:** 4× A100 80GB на одной ноде.

**Распределение по GPU:**

| Компонент | GPU | Память | Комментарий |
|-----------|-----|--------|-------------|
| vLLM (Qwen3.5-35B-A3B) | 0, 1 | ~70GB total в bf16 | `--tensor-parallel-size 2`; fp8 квантизация позволит уместить в одну GPU, но MoE routing в квантизации иногда просаживает quality — сначала бенчмарк в bf16 |
| Embedder (E5 / multilingual-E5) | 2 | ~2GB | Отдельная GPU — чтобы batched encoding при индексации не конкурировал с инференсом |
| Faiss IndexFlatIP | CPU (по умолчанию) или GPU 3 | RAM ~2GB / VRAM <5GB | См. ниже |
| Agent harness | CPU | — | Тонкий оркестратор, GPU не нужна |

**Faiss CPU vs GPU.** На текущем масштабе (MuSiQue, ~50k уникальных passages после дедупа, dim=1024) `faiss-cpu` IndexFlatIP даёт latency ~5–20ms на запрос — это меньше generation latency vLLM, не бутылочное горлышко. На 80GB A100 индекс полностью влезает (даже при росте до 5M векторов это <25GB), но **выигрыш по latency на текущем корпусе не окупает дополнительную точку отказа**.

Решение: **MVP — CPU**, GPU 3 держим в резерве. В коде заложен переключатель, миграция — изменение одной строки в конфиге.

```python
# tool_server/index.py
def build_index(dim: int, use_gpu: bool = False, gpu_id: int = 3):
    cpu_index = faiss.IndexFlatIP(dim)
    if not use_gpu:
        return cpu_index
    res = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
```

**Когда переключаться на GPU:** корпус >5M векторов; batched eval с большим параллелизмом запросов; миграция на бóльшую русскую базу. На GPU 3 также может поехать второй embedder для batched re-indexing своей базы (чтобы не блокировать tool-server-овый embedder при онлайн-запросах).

**Почему такая раскладка:**
- Tool server и vLLM изолированы по GPU — можно перезапускать vLLM (долгий прогрев) не трогая индекс.
- Embedder на отдельной GPU важен, когда будешь строить индекс для своей локальной базы (batched encoding ~100k passages — час работы, не хочется блокировать vLLM).
- GPU 3 свободна как буфер — для faiss-gpu, дополнительного embedder, или экспериментов с reranker'ом в будущем.

**Запуск:**
```bash
# Terminal 1: vLLM
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3.5-35B-A3B \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --port 8000

# Terminal 2: Tool server (embedder на GPU 2; faiss на CPU)
CUDA_VISIBLE_DEVICES=2 python -m tool_server.main --port 8100

# Terminal 3: Eval / harness
python -m eval.run_eval --config configs/default.yaml
```

Если в будущем включится `index.use_gpu: true`, в Terminal 2 нужно будет дать видимость GPU 3 тоже: `CUDA_VISIBLE_DEVICES=2,3`, embedder остаётся на `cuda:0` (логический id внутри процесса = GPU 2), faiss на `cuda:1` (= GPU 3).

**max-model-len = 32768:** нативные 262k у Qwen3.5-35B-A3B избыточны для MVP, полный контекст не используется и кушает KV-cache. Для 8 turns × ~3k tokens/turn = ~24k хватает с запасом. Если упрёшься в обрезание контекста — поднимем.

**Предварительные оценки для Qwen3.5-35B-A3B:**
- bf16: ~70GB VRAM на веса + KV cache. На 2× A100 80GB с tp=2 — комфортно.
- fp8 квантизация: ~35GB, уместится в одну GPU, но MoE routing в квантизации иногда просаживает quality — сначала бенчмарк в bf16.

### Структура репозитория

```
retrieval-agent/
  README.md
  pyproject.toml

  tool_server/
    main.py              # FastAPI entrypoint
    index.py             # faiss wrapper
    embedder.py          # E5 wrapper с префиксами
    schemas.py           # pydantic models

  agent/
    harness.py           # ReAct loop
    prompts.py           # versioned prompts
    schemas.py           # AgentInput, AgentOutput, Message
    parser.py            # answer parser

  indexing/
    parse_musique.py     # сырые jsonl → corpus.jsonl + source_to_doc_id.json
    parse_sbol.py        # FAQ json → corpus.jsonl + eval.jsonl
    build_index.py       # corpus.jsonl → faiss.index + metadata.jsonl
    embedder_batch.py    # batched encoding для build_index

  eval_/
    build_eval.py        # сырой dev → eval jsonl с gold_doc_ids
    run_eval.py          # NDCG@10 на dev
    baseline.py          # single-shot RAG
    metrics.py           # NDCG, Recall, Precision@k

  trajectories/
    writer.py            # JSON trace writer
    checker.py           # TI/TO consistency

  configs/
    default.yaml         # MuSiQue (e5-large-v2)
    sbol.yaml            # SBOL FAQ (multilingual-e5-large)

  scripts/
    serve_vllm.sh
    serve_tool.sh        # CONFIG= переключает датасет
    build_all.sh         # parse_musique → build_index → build_eval (MuSiQue)
    build_sbol.sh        # parse_sbol → build_index (SBOL)
    run_eval.sh          # CONFIG= + OUT_DIR= переключают датасет

  data/
    raw/
      musique/           # сырые файлы MuSiQue (не в git)
      sbol/              # faq_index_*.json (не в git)
    processed/
      musique/           # corpus.jsonl, source_to_doc_id.json, musique_dev_eval.jsonl, ...
      sbol/              # corpus.jsonl, eval.jsonl

  indexes/
    musique/             # faiss.index, metadata.jsonl, config.json
    sbol/                # faiss.index, metadata.jsonl, config.json

  trajectories_data/     # JSON-логи прогонов (не в git)
```

### Конфиг (configs/default.yaml — MuSiQue)

```yaml
model:
  name: Qwen/Qwen3.5-35B-A3B
  vllm_url: http://localhost:8000/v1
  max_tokens: 4096
  temperature: 0.6

tool_server:
  url: http://localhost:8100
  timeout_s: 30

embedder:
  name: intfloat/e5-large-v2
  dim: 1024
  device: cuda:0              # внутри процесса tool_server (CUDA_VISIBLE_DEVICES=2 снаружи)
  batch_size: 64
  max_length: 512
  query_prefix: "query: "
  passage_prefix: "passage: "

index:
  dir: ./indexes/musique/
  top_k_default: 10
  top_k_max: 50
  use_gpu: false              # MVP — CPU; переключить на true при росте корпуса
  gpu_id: 3                   # используется только при use_gpu: true

agent:
  max_turns: 8
  max_tool_calls: 10
  prompt_version: v1

eval:
  raw_musique_dir: ./data/raw/musique/
  processed_dir: ./data/processed/musique/
  eval_dataset_path: ./data/processed/musique/musique_dev_eval.jsonl
  subset_size: 200

trajectories:
  output_dir: ./trajectories_data/
  ti_to_check_every_n: 20
```

Для SBOL: `configs/sbol.yaml` — те же поля, `embedder.name: intfloat/multilingual-e5-large`, `index.dir: ./indexes/sbol/`, `eval.eval_dataset_path: ./data/processed/sbol/eval.jsonl`.

---

## 9. Этапы имплементации

| # | Этап | Выход | Критерий готовности |
|---|------|-------|---------------------|
| 1 | Парсинг MuSiQue | corpus.jsonl + source_to_doc_id.json + eval.jsonl | Количество уникальных passages совпадает с литературой (~50k для MuSiQue ans); % supporting paragraphs корректно маппится |
| 2 | Индексация | faiss.index + metadata.jsonl | `/stats` возвращает корректное num_docs; sanity search на очевидных запросах даёт релевантный топ-1 |
| 3 | Tool server | запущенный FastAPI | `/local_search` на 10 ручных запросах возвращает разумные результаты |
| 4 | vLLM сервинг | запущенный Qwen3.5-35B-A3B с tp=2 | `/v1/chat/completions` отвечает на ping; tool calling работает на синтетическом примере |
| 5 | Harness skeleton | ReAct-цикл + парсер | 10 синтетических single-turn вопросов → корректный формат ответа |
| 6 | Baseline | single-shot RAG eval | NDCG@10 на MuSiQue dev подвыборке (200 примеров) |
| 7 | End-to-end агент | harness на MuSiQue | NDCG@10 на тех же 200 примерах, сравнение с baseline |
| 8 | Budget sweep | tool-efficiency curve | 4 точки: budget ∈ {1,3,5,8} |
| 9 | Анализ траекторий | sanity check промптов | ручная проверка 20 траекторий, итерация промпта при необходимости |

---

## 10. Известные открытые вопросы

1. **Fallback на Qwen3.5-27B dense:** если Qwen3.5-35B-A3B не сервится через vLLM или нестабилен на tool-calling — переходим на 27B без изменений в harness. Решение принимается после этапа 4 (vLLM сервинг + tool calling на синтетическом примере).
2. **Диалоговое eval:** на MVP не делаем, но стоит заложить хотя бы 10 ручных multi-turn примеров для sanity check того, что harness корректно работает с историей длиннее одного сообщения.
3. **2WikiMultiHopQA:** изначально в памяти проекта фигурирует union MuSiQue + 2Wiki. На MVP решено начать только с MuSiQue для простоты. Добавление 2Wiki — после того, как будет чистый NDCG на MuSiQue, чтобы не смешивать переменные.
4. **Точная GPU-конфигурация:** спецификация написана под 4× A100 80GB. При смене железа нужно уточнить `--tensor-parallel-size` и `CUDA_VISIBLE_DEVICES`-раскладку.
5. **Русская база знаний (SBOL FAQ) — реализовано.** `configs/sbol.yaml` + `indexing/parse_sbol.py` + `scripts/build_sbol.sh`. Документ индексируется как `"Вопрос:<q>\nРаздел:<s>\nОтвет:<a>"`, embedder — `intfloat/multilingual-e5-large` (drop-in замена: та же архитектура, та же размерность 1024, те же префиксы).
    - **Системный промпт остаётся английским.** Qwen3.5 обучен преимущественно на английском tool-calling и instruction-following — структурные форматы (function schemas, теги `<answer>`) надёжнее соблюдаются при английских инструкциях. Модель при этом нормально оперирует русскими данными в tool_results.
    - **Language hint в промпте:** добавить строку вида `"The corpus and user queries are in Russian. Generate local_search queries in Russian to match the corpus language."` Без неё модель иногда генерит запросы по-английски, что с multilingual-e5 работает (он cross-lingual), но даёт чуть худший recall, чем same-language retrieval. Это бамп `prompt_version` → v2 в `prompts.py`.
6. **Thinking mode (Qwen3 extended reasoning).** На MVP выключен (`enable_thinking: false`) — baseline собирается в non-thinking режиме. Причины:
    - **Латентность множится в multi-turn loop.** При `max_turns=8` каждый turn с thinking даёт +500–2000 reasoning-токенов до видимого вывода → eval из 200 примеров кратно дольше, цикл итерации промпта замедляется.
    - **Thinking + tool calling в OpenAI-compatible API vLLM** — не самая отлаженная комбинация. Reasoning-вывод и `tool_calls` field используют разные структурные маркеры; парсер vLLM при `tool_choice="auto"` иногда некорректно их разделяет, особенно при незакрытых `<think>` блоках. На MVP это лишний источник `parse_error` траекторий, маскирующий реальные проблемы в промпте (https://github.com/vllm-project/vllm/issues/39056).
    - **Эмпирически выгода для retrieval скромная.** RL-trained search agents (DeepResearcher, Search-R1, WebDancer, ToolOrchestra) обучаются без heavy thinking — планирующее поведение возникает из ReAct + tool-feedback loop. Это сигнал, что для retrieval-агентов без обучения thinking даёт меньший прирост, чем для math/code задач.
    - **TI/TO consistency.** При `enable_thinking=false` поведение chat template детерминированнее, что важно для будущего veRL/Search-R1-style RL.

    **План действий:** этапы 5–7 — без thinking, получаем чистое NDCG-число. После этапа 7 — один ablation-прогон с `enable_thinking=true` на тех же 200 примерах, сравнение NDCG / средней длины trajectory / latency. Бюджет на thinking при включении ограничивать через системный промпт ("Keep your reasoning concise, under N sentences"), пока vLLM OpenAI API не получит первоклассный `thinking_budget` параметр. Решение по умолчанию для будущих этапов принимается на основе ablation-чисел, а не интуиции.
