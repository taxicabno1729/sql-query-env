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
from typing import Any
from urllib import error, request


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload))
    sys.stdout.flush()


def fail(message: str) -> "Never":
    emit({"type": "ERROR", "message": message})
    raise SystemExit(1)


def get_settings() -> dict[str, str | None]:
    return {
        "api_base_url": os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        "model_name": os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct"),
        "hf_token": os.getenv("HF_TOKEN"),
        "local_image_name": os.getenv("LOCAL_IMAGE_NAME"),
        "env_url": os.getenv("ENV_URL", "http://localhost:8000"),
    }


def join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Any:
    url = join_url(base_url, path)
    body = None
    req_headers = {"Accept": "application/json"}

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    if headers:
        req_headers.update(headers)

    req = request.Request(url, data=body, headers=req_headers, method=method)

    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"{method} {url} failed with status {exc.code}: {detail or exc.reason}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc
    except Exception as exc:  # pragma: no cover - defensive boundary for validator runs
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:200] if raw else "<empty response>"
        raise RuntimeError(f"{method} {url} returned invalid JSON: {preview}") from exc


def extract_sql(text: str) -> str:
    """Extract the first SQL block from model output."""
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r"(SELECT\b.*?)(?:;|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def call_model(
    settings: dict[str, str | None],
    task_description: str,
    schema_info: str,
    feedback: str,
    attempt: int,
) -> str:
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

    response = request_json(
        settings["api_base_url"] or "",
        "/chat/completions",
        method="POST",
        payload={
            "model": settings["model_name"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        },
        headers={"Authorization": f"Bearer {settings['hf_token']}"},
        timeout=60.0,
    )

    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Model response missing choices[0].message.content") from exc


def env_request(
    settings: dict[str, str | None],
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> Any:
    return request_json(
        settings["env_url"] or "",
        path,
        method=method,
        payload=payload,
        timeout=30.0,
    )


def run_episode(settings: dict[str, str | None], task_id: str) -> dict[str, Any]:
    """Run a single task episode, emitting START / STEP / END structured logs."""
    obs = env_request(settings, "/reset", method="POST", payload={"task_id": task_id})

    task_description = obs["task_description"]
    schema_info = obs["schema_info"]
    feedback = obs["feedback"]
    best_score = 0.0
    attempt = 0

    emit(
        {
            "type": "START",
            "task_id": task_id,
            "model": settings["model_name"],
            "task_description": task_description,
        }
    )

    while not obs["done"]:
        attempt += 1
        raw_output = call_model(settings, task_description, schema_info, feedback, attempt)
        sql_query = extract_sql(raw_output)

        obs = env_request(settings, "/step", method="POST", payload={"sql_query": sql_query})

        reward = obs["reward"]
        feedback = obs["feedback"]
        best_score = max(best_score, reward)

        emit(
            {
                "type": "STEP",
                "task_id": task_id,
                "attempt": attempt,
                "reward": reward,
                "best_score": best_score,
                "sql": sql_query,
                "feedback": feedback,
                "done": obs["done"],
            }
        )

    emit(
        {
            "type": "END",
            "task_id": task_id,
            "best_score": best_score,
            "attempts": attempt,
        }
    )

    return {"task_id": task_id, "best_score": best_score, "attempts": attempt}


def main() -> None:
    settings = get_settings()

    if not settings["hf_token"]:
        fail("HF_TOKEN env var is required")

    try:
        env_request(settings, "/health")
    except Exception as exc:
        fail(f"Cannot reach environment at {settings['env_url']}: {exc}")

    try:
        tasks = env_request(settings, "/tasks")
        task_ids = [task["task_id"] for task in tasks]
    except Exception as exc:
        fail(f"Unable to load tasks from environment: {exc}")

    if not task_ids:
        fail("Environment returned no tasks")

    results = []
    for task_id in task_ids:
        try:
            results.append(run_episode(settings, task_id))
        except Exception as exc:
            fail(f"Task {task_id} failed: {exc}")

    avg = sum(result["best_score"] for result in results) / len(results)
    emit(
        {
            "type": "SUMMARY",
            "model": settings["model_name"],
            "results": results,
            "average_score": round(avg, 4),
        }
    )


if __name__ == "__main__":
    main()
