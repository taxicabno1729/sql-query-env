"""
Baseline inference agent for the SQL Query Training Environment.

Usage:
    API_BASE_URL=https://api.openai.com/v1 \
    MODEL_NAME=gpt-4o \
    HF_TOKEN=<your-key> \
    uv run python inference.py

Environment variables:
    API_BASE_URL   OpenAI-compatible API base URL
    MODEL_NAME     Model name to use (default: gpt-4o)
    HF_TOKEN       API key (checked first); falls back to OPENAI_API_KEY
    ENV_URL        URL of the running SQL environment (default: http://localhost:8000)
"""

import os
import re
import sys

import httpx
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

if not API_KEY:
    print("ERROR: Set HF_TOKEN or OPENAI_API_KEY", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
http = httpx.Client(base_url=ENV_URL, timeout=30.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_sql(text: str) -> str:
    """Extract the first SQL block from model output."""
    # Try fenced code block first
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: look for a SELECT statement
    match = re.search(r"(SELECT\b.*?)(?:;|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def call_model(task_description: str, schema_info: str, feedback: str, attempt: int) -> str:
    system_prompt = (
        "You are an expert SQL assistant. "
        "Write correct SQLite SELECT queries to answer questions about an e-commerce database. "
        "Always wrap your final SQL in a ```sql ... ``` code block. "
        "Do not include any DML (INSERT/UPDATE/DELETE) or DDL statements."
    )
    user_prompt = (
        f"## Task\n{task_description}\n\n"
        f"## Database Schema\n{schema_info}\n\n"
        + (f"## Previous feedback (attempt {attempt - 1})\n{feedback}\n\n" if attempt > 1 else "")
        + "Write the SQL query:"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content or ""


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(task_id: str) -> dict:
    """Run a single task episode and return a summary dict."""
    # Reset
    reset_resp = http.post("/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    task_description = obs["task_description"]
    schema_info = obs["schema_info"]
    feedback = obs["feedback"]
    best_score = 0.0
    attempt = 0

    print(f"\n{'='*60}")
    print(f"Task {task_id}: {task_description}")
    print("="*60)

    while not obs["done"]:
        attempt += 1
        raw_output = call_model(task_description, schema_info, feedback, attempt)
        sql_query = extract_sql(raw_output)

        print(f"\n  Attempt {attempt}: {sql_query[:80]}{'...' if len(sql_query) > 80 else ''}")

        step_resp = http.post("/step", json={"sql_query": sql_query})
        step_resp.raise_for_status()
        obs = step_resp.json()

        reward = obs["reward"]
        feedback = obs["feedback"]
        best_score = max(best_score, reward)

        print(f"  Reward: {reward:.2f} | {feedback}")

    print(f"\n  Final best score: {best_score:.2f}")
    return {"task_id": task_id, "best_score": best_score, "attempts": attempt}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Check environment is reachable
    try:
        http.get("/health").raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {ENV_URL}: {e}", file=sys.stderr)
        sys.exit(1)

    # Get list of tasks
    tasks_resp = http.get("/tasks")
    tasks_resp.raise_for_status()
    task_ids = [t["task_id"] for t in tasks_resp.json()]

    print(f"Running {len(task_ids)} tasks with model={MODEL_NAME}")

    results = []
    for task_id in task_ids:
        result = run_episode(task_id)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    total = sum(r["best_score"] for r in results)
    for r in results:
        print(f"  {r['task_id']:4s}  score={r['best_score']:.2f}  attempts={r['attempts']}")
    print(f"\n  Average score: {total / len(results):.3f}")


if __name__ == "__main__":
    main()
