---
title: sql-query-env
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# sql-query-env

An **OpenEnv** environment where AI agents learn to write correct SQL queries of
increasing complexity against a realistic e-commerce database.

Built for the **Meta PyTorch OpenEnv Hackathon** (Scaler School of Technology, April 2026).

---

## Overview

The agent receives a natural-language question and the database schema, submits a
SQL `SELECT` query, and receives a **deterministic reward (0.0–1.0)** based on
how closely the query result matches the expected output.

### Why SQL?

- SQL is one of the most in-demand skills for AI coding assistants
- Grading is 100 % deterministic — execute and compare
- Natural easy→hard progression avoids reward sparsity
- Immediate real-world utility for the RL/agent community

---

## Tasks

| ID | Difficulty | Description |
|----|-----------|-------------|
| E1 | Easy      | List all product names and their prices |
| E2 | Easy      | Find all customers from New York |
| E3 | Easy      | Count total orders |
| M1 | Medium    | Total revenue per product category (JOIN + GROUP BY) |
| M2 | Medium    | Top 5 customers by total spend (multi-table JOIN) |
| M3 | Medium    | Orders containing Electronics products (DISTINCT + JOIN) |
| H1 | Hard      | Rank customers by spend within each city (window functions) |
| H2 | Hard      | Above-average customers with percentile rank (CTE + PERCENT_RANK) |

---

## Grading

| Score | Condition |
|-------|-----------|
| **1.0** | Exact match — same rows, columns, and values |
| **0.7** | Correct columns + >= 90 % of expected rows present |
| **0.4** | Correct columns + 50-89 % of rows present |
| **0.2** | Partial column overlap |
| **0.0** | SQL error or completely wrong result |

Partial credit avoids sparse reward signals — critical for RL training.

---

## Evaluation harness

The environment rewards an *episode*. To measure whether a model can actually
write SQL, this repo also ships a small, reproducible evaluation harness in
`eval/` — separate golden set, hard metrics, and a CLI that prints a table you
can paste into a PR.

### Golden set

`eval/dataset.py` defines 12 items across four difficulty bands (easy / medium /
hard / expert), each with a reference query that is executed against the seeded
database. A test asserts every reference query runs and returns rows, so a
broken item fails CI instead of quietly scoring 0.

### Metrics

Metrics are deliberately multiple, because "score" hides disagreements:

| Metric | Meaning |
|---|---|
| **exact match** | identical column names, identical rows, identical order |
| **result match** | identical column names, row order ignored |
| **value match** | identical row values, **column names ignored** |
| mean graded reward | the environment's own partial-credit rubric |
| error rate | predictions that raised a SQL error |

`value match` exists because the reward grader and exact match are column-name
sensitive: `SELECT COUNT(*)` and `SELECT COUNT(*) AS order_count` return the same
answer, but only one of them is "exact". Reporting all three makes the
difference visible instead of burying it in one number. Results carry a 95%
Wilson interval, because 12 items cannot support a precise percentage.

### Run it

```bash
uv run python -m eval.run --predictor oracle          # harness self-check, must be 100%
uv run python -m eval.run --predictor naive           # deterministic floor, no network
EVAL_LLM_API_KEY=... uv run python -m eval.run \
    --predictor llm --model anthropic/claude-sonnet-4.5 --out eval/results/claude.json
uv run python -m eval.run --replay eval/results/claude.json   # re-score, no model calls
```

The `llm` predictor speaks OpenAI-compatible `/chat/completions`, so OpenRouter,
vLLM, Ollama and OpenAI all work; `EVAL_LLM_BASE_URL` and `EVAL_LLM_MODEL`
override the defaults. Only that predictor needs a key — `oracle` and `naive`
run offline, and CI runs those.

### Measured results

12 items, temperature 0, single attempt per item. Reproduce with `--replay`
against the committed JSON in `eval/results/`.

| Predictor | Exact match | Value match | Mean reward | SQL errors |
|---|---:|---:|---:|---:|
| oracle (upper bound, sanity check) | 100.0% | 100.0% | 1.00 | 0.0% |
| naive fixed template (floor) | 8.3% | 8.3% | 0.13 | 0.0% |
| claude-sonnet-4.5 | 50.0% | 83.3% | 0.55 | 0.0% |

By difficulty, claude-sonnet-4.5: easy 75%, medium 50%, hard 0%, expert 50%
exact match. It never produced invalid SQL — every miss was a semantic one
(missing rounding in an aggregate, a `DISTINCT CASE` that double-counted, a
window ranking that ordered by the pre-aggregated column).

### What these numbers do and do not say

- They measure **execution correctness on 12 hand-written questions against one
  50-row database**. That is a smoke test with a wide interval, not a benchmark.
- `exact match` punishes alias differences, so treat 50% as a *lower bound* on
  semantic correctness; `value match` (83.3%) is the closer proxy, and the gap
  between them is exactly the alias effect.
- There is no LLM judge and no claim about explanation quality. Faithfulness of
  natural-language answers is not measured here at all.

---

## Database Schema

```sql
customers(id, name, email, city)
products(id, name, category, price)
orders(id, customer_id, order_date, status)
order_items(id, order_id, product_id, quantity, unit_price)
```

Each episode loads a **fresh in-memory SQLite database** populated with ~50 rows
per table from `data/schema.sql`.

---

## Quick Start

### 1 - Install dependencies

```bash
uv sync
```

### 2 - Start the server

```bash
uv run uvicorn server.main:app --host 0.0.0.0 --port 8000
```

### 3 - Interact via curl

```bash
# Reset - start a new episode (cycles tasks automatically)
curl -X POST http://localhost:8000/reset

# Reset a specific task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "E1"}'

# Step - submit a SQL query
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"sql_query": "SELECT name, price FROM products ORDER BY id"}'

# State - inspect current episode
curl http://localhost:8000/state

# List all available tasks
curl http://localhost:8000/tasks
```

### 4 - Run the baseline inference agent

```bash
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o \
HF_TOKEN=<your-api-key> \
uv run python inference.py
```

### 5 - Docker

```bash
docker build -t sql-query-env .
docker run -p 8000:8000 sql-query-env
```

---

## API Reference

### `POST /reset`

Start a new episode.

**Request body** (optional):
```json
{ "task_id": "M2" }
```

**Response** (`SQLObservation`):
```json
{
  "done": false,
  "reward": 0.0,
  "task_description": "List the top 5 customers by total amount spent...",
  "schema_info": "Tables available: ...",
  "feedback": "Episode started. Submit your SQL query.",
  "metadata": {
    "task_id": "M2",
    "difficulty": "medium",
    "max_attempts": 5
  }
}
```

### `POST /step`

Submit a SQL query.

**Request body**:
```json
{ "sql_query": "SELECT c.name, SUM(...) FROM customers c JOIN ..." }
```

**Response** (`SQLObservation`):
```json
{
  "done": false,
  "reward": 0.7,
  "task_description": "...",
  "schema_info": "...",
  "feedback": "Correct columns. Found 90% of expected rows (9/10).",
  "metadata": { "attempts": 1, "max_attempts": 5 }
}
```

### `GET /state`

Returns current episode state (`SQLState`).

### `GET /tasks`

List all 8 tasks with metadata.

### `GET /health`

Returns `{"status": "ok"}`.

---

## Project Structure

```
openenv-hackathon/
├── pyproject.toml          # uv project + dependencies
├── .python-version         # 3.13
├── Dockerfile              # Container build
├── openenv.yaml            # Environment manifest
├── inference.py            # Baseline agent (OpenAI client)
├── README.md
├── data/
│   └── schema.sql          # SQLite schema + seed data
└── server/
    ├── __init__.py
    ├── main.py             # FastAPI app
    ├── models.py           # Pydantic models
    ├── environment.py      # Core env logic + state machine
    └── tasks/
        ├── __init__.py
        ├── definitions.py  # 8 task definitions
        └── grader.py       # SQL execution + scoring
```

---

## License

MIT
