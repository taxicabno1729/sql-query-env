"""
FastAPI application exposing the OpenEnv-compatible HTTP interface.

Endpoints (OpenEnv spec):
  GET  /health         Health check  → {"status": "healthy"}
  GET  /metadata       Environment metadata
  GET  /schema         Action / observation / state JSON schemas
  POST /mcp            MCP JSON-RPC endpoint (tools/list)
  POST /reset          Start a new episode
  POST /step           Submit a SQL action
  GET  /state          Current episode state
  GET  /tasks          List all available tasks
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .environment import SQLEnvironment
from .models import SQLAction, SQLObservation, SQLState
from .tasks.definitions import TASKS

app = FastAPI(
    title="SQL Query Training Environment",
    description=(
        "An OpenEnv environment where AI agents learn to write SQL queries "
        "of increasing complexity against a realistic e-commerce database."
    ),
    version="0.1.0",
)

_env = SQLEnvironment()


# ── Request / response helpers ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str | None = None


# ── OpenEnv required endpoints ────────────────────────────────────────────────

@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "name": "sql-query-env",
        "description": (
            "An OpenEnv environment where AI agents learn to write SQL queries "
            "of increasing complexity against a realistic e-commerce database."
        ),
        "version": "0.1.0",
        "author": "Aniket Mazumdar",
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "action": SQLAction.model_json_schema(),
        "observation": SQLObservation.model_json_schema(),
        "state": SQLState.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(request: Request) -> dict:
    """Minimal MCP JSON-RPC endpoint (tools/list)."""
    body = await request.json()
    method = body.get("method", "")
    req_id = body.get("id", 1)

    if method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "reset",
                    "description": "Reset the environment and start a new episode.",
                    "inputSchema": {"type": "object", "properties": {"task_id": {"type": "string"}}},
                },
                {
                    "name": "step",
                    "description": "Submit a SQL query and receive a reward.",
                    "inputSchema": SQLAction.model_json_schema(),
                },
            ]
        }
    else:
        result = {"error": f"Unsupported method: {method}"}

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


@app.post("/reset", response_model=SQLObservation)
def reset(req: ResetRequest = ResetRequest()) -> SQLObservation:
    """
    Start a new episode.  Pass `task_id` (e.g. "E1", "M2", "H1") to pick a
    specific task, or omit to cycle through all tasks in order.
    """
    try:
        return _env.reset(task_id=req.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=SQLObservation)
def step(action: SQLAction) -> SQLObservation:
    """Submit a SQL query and receive a reward + feedback observation."""
    try:
        return _env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=SQLState)
def state() -> SQLState:
    """Return the current episode state (task, attempts, score history)."""
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/tasks")
def list_tasks() -> list[dict]:
    """List all available tasks with metadata."""
    return [
        {
            "task_id": t.task_id,
            "difficulty": t.difficulty,
            "question": t.question,
            "max_attempts": t.max_attempts,
            "tags": t.tags,
        }
        for t in TASKS
    ]


def start() -> None:
    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=False)
