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
