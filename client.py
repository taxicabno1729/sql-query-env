"""
HTTP client for the SQL Query Training Environment.

Provides a simple synchronous client for interacting with the running server.

Usage:
    from client import SQLEnvClient

    client = SQLEnvClient("http://localhost:8000")
    obs = client.reset(task_id="E1")
    print(obs["task_description"])

    obs = client.step("SELECT name, price FROM products ORDER BY id")
    print(obs["reward"], obs["feedback"])
"""

import httpx

from models import SQLAction, SQLObservation, SQLState


class SQLEnvClient:
    """Synchronous HTTP client for the SQL Query Training Environment."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def health(self) -> dict:
        return self._http.get("/health").raise_for_status().json()

    def reset(self, task_id: str | None = None) -> SQLObservation:
        body = {"task_id": task_id} if task_id else {}
        resp = self._http.post("/reset", json=body).raise_for_status()
        return SQLObservation(**resp.json())

    def step(self, sql_query: str) -> SQLObservation:
        action = SQLAction(sql_query=sql_query)
        resp = self._http.post("/step", json=action.model_dump()).raise_for_status()
        return SQLObservation(**resp.json())

    def state(self) -> SQLState:
        resp = self._http.get("/state").raise_for_status()
        return SQLState(**resp.json())

    def tasks(self) -> list[dict]:
        return self._http.get("/tasks").raise_for_status().json()

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "SQLEnvClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
