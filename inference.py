"""
Inference Script — SQL Query Training Environment
==================================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Optional:
    LOCAL_IMAGE_NAME   Docker image name (when using from_docker_image())
    ENV_URL            Running environment URL (default: http://localhost:8000)

Usage:
    API_BASE_URL=https://router.huggingface.co/v1 \\
    MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \\
    HF_TOKEN=hf_xxx \\
    uv run python inference.py
"""

import json
import os
import re
import sys

import httpx
from openai import OpenAI

# ── Environment variables (API_BASE_URL and MODEL_NAME have defaults; HF_TOKEN does not) ──

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional: used with from_docker_image()
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

if not HF_TOKEN:
    print(json.dumps({"type": "ERROR", "message": "HF_TOKEN env var is required"}))
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
http = httpx.Client(base_url=ENV_URL, timeout=30.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_sql(text: str) -> str:
    """Extract the first SQL block from model output."""
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
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
    """Run a single task episode, emitting START / STEP / END structured logs."""
    reset_resp = http.post("/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    task_description = obs["task_description"]
    schema_info = obs["schema_info"]
    feedback = obs["feedback"]
    best_score = 0.0
    attempt = 0

    # ── START ──
    print(json.dumps({
        "type": "START",
        "task_id": task_id,
        "model": MODEL_NAME,
        "task_description": task_description,
    }))
    sys.stdout.flush()

    while not obs["done"]:
        attempt += 1
        raw_output = call_model(task_description, schema_info, feedback, attempt)
        sql_query = extract_sql(raw_output)

        step_resp = http.post("/step", json={"sql_query": sql_query})
        step_resp.raise_for_status()
        obs = step_resp.json()

        reward = obs["reward"]
        feedback = obs["feedback"]
        best_score = max(best_score, reward)

        # ── STEP ──
        print(json.dumps({
            "type": "STEP",
            "task_id": task_id,
            "attempt": attempt,
            "reward": reward,
            "best_score": best_score,
            "sql": sql_query,
            "feedback": feedback,
            "done": obs["done"],
        }))
        sys.stdout.flush()

    # ── END ──
    print(json.dumps({
        "type": "END",
        "task_id": task_id,
        "best_score": best_score,
        "attempts": attempt,
    }))
    sys.stdout.flush()

    return {"task_id": task_id, "best_score": best_score, "attempts": attempt}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        http.get("/health").raise_for_status()
    except Exception as e:
        print(json.dumps({"type": "ERROR", "message": f"Cannot reach environment at {ENV_URL}: {e}"}))
        sys.exit(1)

    tasks_resp = http.get("/tasks")
    tasks_resp.raise_for_status()
    task_ids = [t["task_id"] for t in tasks_resp.json()]

    results = []
    for task_id in task_ids:
        result = run_episode(task_id)
        results.append(result)

    avg = sum(r["best_score"] for r in results) / len(results)
    print(json.dumps({
        "type": "SUMMARY",
        "model": MODEL_NAME,
        "results": results,
        "average_score": round(avg, 4),
    }))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
